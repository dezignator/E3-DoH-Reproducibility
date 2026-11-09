#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
src/protocols/doq_adapter.py
============================

E3-DoH Reproducibility — DNS-over-QUIC (DoQ) protocol adapter.

Chapter 5 of the thesis: E3-DoH (Enhanced Evolutionary Encryption for
Multi-Protocol DNS Security — DoH, DoT, DoQ).

This module is the concrete adapter for DoQ over UDP/853 with QUIC v1
(default_e3_config.yaml §5: `doq.default_port: 853`,
`doq.quic_version: 1`). It inherits from ProtocolAdapter and implements
the four abstract methods:

    parse_pcap(pcap_path)                    -> Iterator[FlowState]
    extract_protocol_specific_features(flow) -> Dict[str, float]
    apply_sandbox_action(flow, action)       -> SandboxResult
    _metric_tensor_kwargs(x)                 -> Dict[str, Any]

HOW DOQ DIFFERS FROM DOH AND DOT
--------------------------------
DoQ is the *most structurally different* of the three protocols, and
the differences show up in every part of the adapter:

1. UDP, not TCP. DoQ runs over UDP/853 (default_e3_config.yaml §5).
   The flow classification is therefore on the UDP 5-tuple, and the
   flow-keying logic uses `ip.src/ip.dst/udp.sport/udp.dport` — not
   the TCP variant. The adapter's parse_pcap uses scapy's UDP layer,
   not TCP.

2. QUIC, not TLS records. QUIC has its own packet format, distinct
   from the TLS record format. A QUIC packet has either:
     - a *long header* (first byte has bit 7 set), used during the
       handshake, with a plaintext 4-byte version, a connection ID,
       and a packet type in bits 4-5 (Initial / 0-RTT / Handshake /
       Retry);
     - a *short header* (first byte has bit 7 clear), used for 1-RTT
       data, with a connection ID of negotiated length and no version.
   The TLS handshake in QUIC is *inside* CRYPTO frames inside QUIC
   Initial and Handshake packets — it is not directly in the UDP
   payload the way it is in DoT. The adapter therefore cannot simply
   scan the first payload byte for a TLS record type; it has to
   classify the QUIC packet type instead.

3. Connection ID and migration. QUIC has a connection ID (CID) that
   is preserved across network-path changes. This is what enables the
   QUIC connection-migration feature that the thesis uses for the
   DoQ sandbox action (thesis §5.6.4, p.133). It also means a QUIC
   flow can change its 5-tuple (e.g. client moves from Wi-Fi to
   cellular) mid-connection while remaining the *same* connection
   from the application's perspective. The adapter uses the initial
   5-tuple as the flow key, which means a migrated connection will
   appear as two flows in the extracted feature store. This is a
   documented limitation; see "Connection migration" below.

4. The sandbox action. DoQ's primitive is `migrate_doq_connection`
   (thesis §5.6.4, p.133), not `isolate_doh_stream` (DoH) and not
   `reset_dot_tls` (DoT). The migration *preserves* the crypto
   session — the CID is unchanged and the session keys still work —
   but redirects the packets to a sandbox path. This is fundamentally
   different from the DoH/DoT actions, which terminate or reset
   something. See the "Sandbox action" section below.

5. The metric tensor. DoQ uses `math_utils.metric_tensor_doq`
   (Eq. 84, p.130), which takes `(d2x_dt2, Q)` and the coefficient η
   from riemannian_metrics.yaml §5. The second-derivative argument
   `d2x_dt2` is the most demanding of the three metric-tensor inputs;
   see the "Metric tensor kwargs" section below.

WHAT IS OBSERVABLE WITHOUT DECRYPTION
-------------------------------------
QUIC's plaintext surface is *narrower* than TLS-in-TCP's, because
QUIC encrypts more of its header than TLS does:

  Observable (plaintext):
    - UDP packet headers (src/dst IP and port, sizes, timestamps)
    - QUIC first byte (header form bit, packet type bits, PN-length bits)
    - QUIC long-header version (4 bytes) and CID lengths
    - QUIC connection ID (in Initial packets; the length is negotiated
      during the handshake and stays fixed thereafter for short headers)
    - the "fixed bit" (bit 6) which is always 1 for QUIC v1
    - inter-arrival times, session duration

  Encrypted (inside QUIC's AEAD protection):
    - the CRYPTO frames carrying the TLS handshake
    - the STREAM frames carrying DNS queries and responses
    - the ACK frames
    - the CONNECTION_CLOSE and other control frames
    - the SNI extension (unless ECH is used, in which case the SNI is
      not observable for DoQ either, just as for DoT/DoH)

The three protocol-specific features are therefore QUIC-level proxies
for what the thesis calls for, just as the DoH/DoT features are
TLS-level proxies. Each is documented below.

THE THREE PROTOCOL-SPECIFIC FEATURES
------------------------------------
1. `subdomain_entropy` (riemannian_metrics.yaml §2):

   What the thesis wants: a measure of the queried subdomain's
   character-distribution complexity.

   What is observable: the SNI, if it can be extracted. For DoH/DoT,
   the SNI is in the first TLS record of the first payload; for DoQ,
   the SNI is inside a CRYPTO frame inside an Initial packet, and
   extracting it requires parsing the QUIC long header *and* the
   CRYPTO frame structure *and* the TLS ClientHello — which is more
   parsing than the metadata-only constraint permits.

   The adapter therefore uses the **documented fallback**: the Shannon
   entropy of the first 512 bytes of the first payload, treated as a
   byte string. This is the same fallback that `data/preprocess_pcap.py`
   uses when it cannot extract an SNI; the two paths agree by
   construction on the DoQ branch.

   This is a *weaker* feature for DoQ than for DoH/DoT, because the
   fallback captures the entropy of the QUIC header + encrypted
   payload bytes rather than the entropy of the subdomain. It is
   nonetheless the documented choice, and the DoQ detection accuracy
   reported in Table 26 (99.7%) was measured with it.

2. `query_type_distribution_l1` (riemannian_metrics.yaml §2):

   What the thesis wants: a measure of the DNS query-type
   distribution's distance from the benign baseline.

   What is observable: the QUIC packet-type distribution. The four
   categories the adapter uses are (matching `data/preprocess_pcap.py`'s
   `_quic_packet_type`):

       0 : short header (1-RTT)
       1 : Initial (long header, type 0)
       2 : 0-RTT    (long header, type 1)
       3 : Handshake (long header, type 2)

   The Retry type (long header, type 3) is folded into category 3 by
   the `(first >> 4) & 0x03` mask. This is the encoding used by the
   offline preprocess_pcap.py, and the DoQ accuracy in Table 26 was
   computed with it.

   *Documented limitation*: this encoding places Initial packets
   (whose first byte has bit 7 set and bits 4-5 = 00) and short-header
   1-RTT packets (whose first byte has bit 7 clear) in the same
   category (index 0). They are not the same kind of packet, but the
   mask `(first >> 4) & 0x03` collapses them. This is an artefact of
   the offline implementation, and the online adapter matches it for
   consistency. A reviewer who wants to distinguish Initial from 1-RTT
   would need to change both the offline and the online encoding in
   the same commit, and would need to re-generate
   expected_outputs/table_26_expected.json.

3. `protocol_specific_feature` (riemannian_metrics.yaml §2):

   What the thesis wants: "for DoQ it is a measure of QUIC stream
   parallelism" (riemannian_metrics.yaml §2).

   What is observable: the fraction of short-header (1-RTT) packets.
   A benign DoQ flow has a mix of long-header (handshake) and
   short-header (1-RTT) packets; a DoQ tunnel that keeps a long-lived
   connection has almost exclusively short-header packets, because
   the handshake happened once at the beginning. The fraction of
   short-header packets is therefore a coarse proxy for the "how much
   of the connection is in steady-state data flow" aspect of stream
   parallelism.

   Why this is a weaker proxy than DoH/DoT's: stream parallelism is
   about the *number of active streams* at any moment, which is inside
   the STREAM frames and therefore encrypted. The fraction of
   short-header packets only captures the *ratio of 1-RTT to handshake
   packets*, not the number of streams within 1-RTT. The thesis
   acknowledges that this feature is a proxy for the actual quantity
   (riemannian_metrics.yaml §2); the DoQ accuracy of 99.7% on the
   reference testbed is the empirical evidence that the proxy is
   discriminative on the evaluated data.

CONNECTION MIGRATION (documented limitation)
-------------------------------------------
QUIC connection migration allows a client to change its IP address or
port mid-connection while keeping the same connection ID. The
application sees one continuous connection; a passive observer sees
two different 5-tuples, because the flow key (5-tuple) has changed.

The adapter uses the initial 5-tuple as the flow key. If a connection
migrates mid-capture, the adapter will emit *two* FlowState objects
for what is logically *one* connection. This is a documented
limitation of the metadata-only extraction approach: full
connection-ID tracking would require parsing the QUIC CID field
(which is plaintext in Initial packets but not in every subsequent
packet), maintaining a CID-to-flow map across the capture, and
re-keying the accumulator on migration. That is significantly more
parsing than the metadata-only constraint permits, and the thesis's
reported results were produced with the 5-tuple keying (the
preprocess_pcap.py implementation uses the same 5-tuple keying).

The adapter emits a WARNING log record for each flow whose server
port is 853 and whose client IP/port changes within a short window
of a previous flow's last packet, to help a reviewer spot a potential
migration in a real capture. The warning does not change the flow
data; it is diagnostic only.

METRIC TENSOR KWARGS
--------------------
The DoQ metric tensor (Eq. 84, p.130) is:

    g_DoQ(x) = δ_ij + η (∂²x/∂t²)ᵀ Q (∂²x/∂t²) dt⁴

The `Q` is the congestion-control operator; the base class builds it
from riemannian_metrics.yaml §5 and passes it to math_utils.
The `d2x_dt2` (the second derivative of the feature vector) is what
this adapter must supply.

The second derivative cannot be observed from a single feature vector,
so the adapter uses a documented proxy, consistent with the approach
of the DoH and DoT adapters:

    d2x_dt2 ≈ (x - _DOQ_BASELINE) / _DOQ_TAU²

where `_DOQ_TAU` is the characteristic acceleration timescale. The
choice of `_DOQ_TAU` is documented in the module constant below; the
key rationale is that QUIC's congestion control loop operates on
~10-50 ms timescales (CUBIC/KUBIC in production resolvers have
RTT-minima of 20-40 ms in typical deployments), so the characteristic
acceleration timescale is shorter than DoH/DoT's 300 ms first-
derivative timescale. The second derivative is inherently more
sensitive to short-term variation than the first derivative.

The `dt` in Eq. 84 is set to 1.0 (the default in
math_utils.metric_tensor_doq), matching the DoH/DoT adapters.

SANDBOX ACTION
--------------
`migrate_doq_connection` (thesis §5.6.4, p.133) migrates the QUIC
connection to a sandbox path while preserving the crypto session
(the connection ID is unchanged, so the TLS session continues). This
is different in kind from the DoH/DoT actions:

    DoH: isolate_doh_stream      -- terminate the stream
    DoT: reset_dot_tls           -- reset the TLS session
    DoQ: migrate_doq_connection  -- migrate the connection, session lives

The reference latency for `migrate_doq_connection` is 1.7 ms
(sandbox_mdp_params.yaml §6), the highest of the three actions,
because the migration involves a CONNECTION_CLOSE/connection-ID
handshake with the sandbox peer.

In toy mode (no real network namespace), the action is a bookkeeping
operation and the measured latency is microseconds. In full mode, the
action includes the routing change and takes ~1.7 ms.

Idempotency: a flow that has already been migrated is a no-op on a
second call (the connection is already in the sandbox). Unlike the
DoH/DoT actions, the migrated connection is *still alive* — subsequent
packets continue to flow through the sandbox path, and the adapter's
sandbox_reset check does not prevent a subsequent legitimate migration
to a different sandbox if the first one is later deemed insufficient.
For simplicity, the adapter treats any second migration as a no-op
and records the fact in the returned SandboxResult's `detail` field.

SECURITY BOUNDARY
-----------------
Same as the DoH and DoT adapters: no security claim from this module.
The framework's security boundary is in aee_params.yaml §10 and
qhe_params.yaml §9. The DoQ-specific accuracy of 99.7% (Table 26) is
an empirical measurement on the reference testbed, not a proof.

Thread safety: not thread-safe (see base_adapter.py module docstring).
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

import numpy as np

from data.data_utils import FEATURE_COLUMNS
from src.protocols.base_adapter import (
    FlowState,
    ProtocolAdapter,
    SandboxResult,
    SandboxActionError,
    FeatureExtractionError,
    PROTOCOL_FEATURES,
    SANDBOX_ACTIONS,
)


# ---------------------------------------------------------------------------
# Constants — DoQ baselines for the metric-tensor kwargs
# ---------------------------------------------------------------------------
# _DOQ_BASELINE is the feature vector of a "typical benign DoQ flow".
# Its values are derived from:
#   - the thesis's description of DoQ as the lowest-latency of the
#     three protocols (aee_params.yaml §8: "DoQ is slowest [to
#     update AEE parameters] because the QUIC stream-multiplexing
#     layer gives the AEE engine a less predictable per-query
#     boundary", but Table 26 shows DoQ has the *lowest* P99
#     latency at 0.8 ms — the AEE update time and the query
#     latency are different quantities),
#   - the toy corpus's benign distribution (download_datasets.py,
#     TOY_CORPUS.feature_distributions),
#   - the DoH and DoT baselines, adjusted for DoQ's lower latency
#     and smaller per-query footprint.
#
# The critical differences from the DoH/DoT baselines:
#   - shorter inter_arrival_time_mean (DoQ has lower handshake
#     latency and pipelined queries),
#   - shorter session_duration (DoQ is often used for short
#     query bursts),
#   - higher protocol_specific_feature (a larger fraction of
#     1-RTT packets, because the handshake is a smaller fraction
#     of a DoQ session than of a DoH/DoT session).

_DOQ_BASELINE = np.array([
    0.05,    # inter_arrival_time_mean    (50 ms; lower than DoT's 80 ms)
    0.015,   # inter_arrival_time_std
    68.0,    # query_size_mean
    8.0,     # query_size_std
    110.0,   # response_size_mean
    15.0,    # response_size_std
    0.62,    # query_response_ratio
    3.1,     # subdomain_entropy
    0.10,    # query_type_distribution_l1 (tighter than DoH/DoT; short header dominates)
    0.85,    # burstiness
    6.0,     # session_duration           (shorter than DoH's 12 s, DoT's 8 s)
    0.95,    # protocol_specific_feature  (higher 1-RTT fraction; less handshake overhead)
], dtype=np.float64)

if _DOQ_BASELINE.shape != (len(FEATURE_COLUMNS),):
    raise RuntimeError(
        f"_DOQ_BASELINE has shape {_DOQ_BASELINE.shape}, "
        f"expected ({len(FEATURE_COLUMNS)},). Update _DOQ_BASELINE "
        f"if FEATURE_COLUMNS changes."
    )

# The characteristic acceleration timescale for the second-derivative
# proxy. DoQ's congestion control loop reacts on ~10-50 ms timescales
# in typical deployments, so the acceleration timescale is set to
# 50 ms. This is deliberately shorter than the DoH/DoT first-derivative
# timescale (300 ms / 80 ms) because the second derivative is more
# sensitive to short-term variation: a second derivative over 300 ms
# would be almost identical to a first derivative, losing the
# acceleration signal that QUIC's control loop is designed to expose.
_DOQ_TAU = 0.05

# The `dt` parameter of math_utils.metric_tensor_doq. The default in
# that function is 1.0; the adapter exposes it as a constant so that
# a subclass or a test can override it without editing the module.
_DOQ_DT = 1.0

# DoQ server port (default_e3_config.yaml §5, doq.default_port = 853).
# DoQ uses UDP, not TCP, so the port is on the UDP socket.
_DOQ_PORT = 853

# QUIC packet-type encoding, matching data/preprocess_pcap.py's
# `_quic_packet_type`. See the module docstring for the documented
# collision between category 0 (short header) and category 0 (Initial).
_QUIC_SHORT_HEADER = 0
_QUIC_INITIAL = 1
_QUIC_0RTT = 2
_QUIC_HANDSHAKE = 3
_QUIC_N_CATEGORIES = 4

# First-payload retention, matching preprocess_pcap.py.
_MAX_PAYLOAD_KEEP_BYTES = 512

# The characteristic window for spotting a potential connection
# migration (see the "Connection migration" section of the module
# docstring). Two flows from the same client IP on different ports
# whose last/first packets are within this window are flagged as a
# possible migration.
_MIGRATION_DETECTION_WINDOW_S = 1.0


# ---------------------------------------------------------------------------
# DoQAdapter
# ---------------------------------------------------------------------------

class DoQAdapter(ProtocolAdapter):
    """
    Concrete adapter for DNS-over-QUIC (DoQ) over UDP/853 with QUIC v1.

    See the module docstring for the design rationale, the QUIC-specific
    observables, the connection-migration limitation, and the metric-
    tensor kwargs.
    """

    protocol = "doq"
    protocol_full_name = "DNS-over-QUIC"

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config)
        # The QUIC version (default_e3_config.yaml §5:
        # doq.quic_version: 1). The adapter's QUIC packet-type parsing
        # is version-specific: the long-header type bits are defined
        # differently in QUIC v2 (RFC 9369) than in QUIC v1 (RFC 9000).
        # The adapter only supports v1; a v2 configuration would need
        # a different `_quic_packet_type` (the v2 type mapping is
        # 0=Retry, 1=Initial, 2=0-RTT, 3=Handshake, which is different
        # from v1's 0=Initial, 1=0-RTT, 2=Handshake, 3=Retry).
        try:
            quic_version = config["protocols"]["doq"]["quic_version"]
        except (KeyError, TypeError):
            quic_version = 1
        if int(quic_version) != 1:
            raise FeatureExtractionError(
                f"DoQ adapter only supports quic_version=1; config specifies "
                f"quic_version={quic_version!r}. QUIC v2 (RFC 9369) uses "
                f"a different long-header type mapping, which the adapter's "
                f"_quic_packet_type function does not implement."
            )
        self._quic_version = int(quic_version)
        # Expose the baselines as instance attributes so a subclass or
        # a test can override them.
        self._baseline = _DOQ_BASELINE.copy()
        self._tau = _DOQ_TAU
        self._dt = _DOQ_DT

    # -- 1. parse_pcap -----------------------------------------------------

    def parse_pcap(self, pcap_path: Path) -> Iterator[FlowState]:
        """
        Parse a PCAP file and yield one FlowState per DoQ flow.

        Uses scapy's PcapReader to stream the PCAP. For each packet,
        determines if it belongs to a DoQ flow (UDP port 853) and
        accumulates it. The accumulation:
            - adds query/response packet sizes to the flow's
              query_sizes / response_sizes
            - adds the timestamp to the flow's query_times
            - for query packets, extracts the QUIC packet type from
              the first byte of the UDP payload (the first byte's
              header-form bit and type bits are plaintext)
            - for the first query packet, retains up to 512 bytes for
              the entropy-based subdomain feature

        Yields
        ------
        FlowState objects, one per DoQ flow, at end of PCAP. A flow
        that has migrated mid-capture may yield *two* FlowStates; see
        the "Connection migration" section of the module docstring for
        the documented limitation.
        """
        try:
            from scapy.all import PcapReader  # type: ignore
            from scapy.layers.inet import IP, UDP  # type: ignore
            from scapy.layers.l2 import Ether  # type: ignore
        except ImportError as e:
            raise FeatureExtractionError(
                "DoQAdapter.parse_pcap requires scapy. "
                "Install with: pip install scapy==2.5.0"
            ) from e

        pcap_path = Path(pcap_path)
        if not pcap_path.exists():
            raise FileNotFoundError(f"PCAP not found: {pcap_path}")

        self._log.info(
            "doq_parse_start",
            pcap=str(pcap_path),
            quic_version=self._quic_version,
            size_bytes=pcap_path.stat().st_size,
        )

        flows: Dict[tuple, FlowState] = {}
        n_packets_total = 0
        n_packets_kept = 0
        migration_warnings = 0

        # For the migration heuristic, track the (client_ip, last_ts)
        # of every recently-closed DoQ flow, so a new flow from the
        # same client IP within _MIGRATION_DETECTION_WINDOW_S can be
        # flagged as a possible migration.
        recent_client_flows: Dict[str, tuple] = {}

        try:
            with PcapReader(str(pcap_path)) as reader:
                for pkt in reader:
                    n_packets_total += 1
                    if Ether not in pkt or IP not in pkt or UDP not in pkt:
                        continue
                    ip = pkt[IP]
                    udp = pkt[UDP]
                    # DoQ: UDP on port 853 (either direction).
                    if int(udp.dport) != _DOQ_PORT and int(udp.sport) != _DOQ_PORT:
                        continue

                    ts = float(pkt.time)
                    a = (ip.src, ip.dst, int(udp.sport), int(udp.dport))
                    b = (ip.dst, ip.src, int(udp.dport), int(udp.sport))
                    key = min(a, b)

                    flow = flows.get(key)
                    if flow is None:
                        flow = FlowState(
                            flow_id=self.make_flow_id(*key, "udp"),
                            protocol="DoQ",
                            client_ip=ip.src, client_port=int(udp.sport),
                            server_ip=ip.dst, server_port=int(udp.dport),
                        )
                        flow.protocol_data = {
                            "quic_packet_types": [],
                            "first_payload": None,
                            "n_short_header_packets": 0,
                        }
                        flows[key] = flow
                        # Migration heuristic: if a flow from the same
                        # client IP closed within the migration
                        # window, warn.
                        prev = recent_client_flows.get(ip.src)
                        if prev is not None:
                            prev_ts, prev_flow_id = prev
                            if ts - prev_ts < _MIGRATION_DETECTION_WINDOW_S:
                                migration_warnings += 1
                                self._log.warning(
                                    "doq_possible_connection_migration",
                                    client_ip=ip.src,
                                    previous_flow_id=prev_flow_id,
                                    new_flow_id=flow.flow_id,
                                    gap_s=round(ts - prev_ts, 4),
                                    note="Possible QUIC connection migration; "
                                         "the adapter uses 5-tuple keying, so "
                                         "a migrated connection appears as two "
                                         "flows. See the module docstring.",
                                )
                        recent_client_flows[ip.src] = (ts, flow.flow_id)

                    payload = bytes(udp.payload) if udp.payload else b""
                    is_query = (
                        ip.src == flow.client_ip
                        and int(udp.sport) == flow.client_port
                    )
                    pkt_size = len(pkt)

                    if is_query:
                        flow.add_query(ts, pkt_size)
                        self._accumulate_doq_query(flow, payload)
                    else:
                        flow.add_response(ts, pkt_size)
                    n_packets_kept += 1
        except Exception as e:
            self._log.error(
                "doq_parse_error",
                pcap=str(pcap_path), error=repr(e),
            )
            raise FeatureExtractionError(
                f"failed to parse {pcap_path}: {e}"
            ) from e

        self._log.info(
            "doq_parse_complete",
            pcap=str(pcap_path),
            packets_total=n_packets_total,
            packets_kept=n_packets_kept,
            flows=len(flows),
            migration_warnings=migration_warnings,
        )

        for flow in flows.values():
            if flow.n_packets() > 0:
                yield flow

    # -- 2. extract_protocol_specific_features -----------------------------

    def extract_protocol_specific_features(self, flow: FlowState) -> Dict[str, float]:
        """
        Extract the three DoQ protocol-specific features:

            subdomain_entropy
            query_type_distribution_l1
            protocol_specific_feature

        See the module docstring for what each feature measures and why
        it is a weaker proxy for DoQ than for DoH/DoT.
        """
        data = flow.protocol_data or {}
        packet_types: List[int] = list(data.get("quic_packet_types", []))
        first_payload: Optional[bytes] = data.get("first_payload")

        # --- subdomain_entropy ---
        # For DoQ, the SNI is inside a CRYPTO frame inside an Initial
        # packet. The adapter does not implement a full QUIC CRYPTO-
        # frame parser (that would violate the metadata-only
        # constraint). It uses the documented fallback: the Shannon
        # entropy of the first 512 bytes of the first payload, treated
        # as a byte string. This matches preprocess_pcap.py's fallback
        # behaviour for DoQ.
        if first_payload:
            subdomain_entropy = _shannon_entropy_bytes(first_payload)
        else:
            subdomain_entropy = 0.0

        # --- query_type_distribution_l1 ---
        # L1 distance over the QUIC packet-type categories (0-3). See
        # the module docstring for the documented category collision
        # between short-header and Initial packets.
        if packet_types:
            l1 = _l1_distance_uniform(packet_types, _QUIC_N_CATEGORIES)
        else:
            l1 = 0.0

        # --- protocol_specific_feature ---
        # Fraction of short-header (1-RTT) packets. This is the
        # documented proxy for stream parallelism.
        if packet_types:
            n_short = sum(1 for t in packet_types if t == _QUIC_SHORT_HEADER)
            fraction_short = n_short / len(packet_types)
        else:
            fraction_short = 0.0

        return {
            "subdomain_entropy": float(subdomain_entropy),
            "query_type_distribution_l1": float(l1),
            "protocol_specific_feature": float(fraction_short),
        }

    # -- 3. apply_sandbox_action -------------------------------------------

    def apply_sandbox_action(
        self, flow: FlowState, action: str,
    ) -> SandboxResult:
        """
        Apply a sandbox action to a DoQ flow.

        Only `migrate_doq_connection` is valid for DoQ
        (SANDBOX_ACTIONS["doq"]). The migration *preserves* the crypto
        session — the connection ID is unchanged, so the TLS session
        continues — but redirects the packets to a sandbox path (thesis
        §5.6.4, p.133).

        This is fundamentally different from the DoH/DoT actions, which
        terminate or reset the underlying transport. A migrated DoQ
        connection is *still alive* from the application's perspective;
        only the network path has changed. The adapter records the
        migration in flow.protocol_data["sandbox_migrated"] and, if
        the flow is migrated a second time, treats the second migration
        as a no-op (the connection is already in the sandbox).

        In toy mode (no real network namespace), the action is a
        bookkeeping operation and the measured latency is microseconds.
        In full mode, the action includes the routing change and takes
        ~1.7 ms (sandbox_mdp_params.yaml §6, the highest of the three
        protocol actions).
        """
        expected = SANDBOX_ACTIONS["doq"]
        if action != expected:
            raise SandboxActionError(
                f"DoQ adapter cannot apply action {action!r}; "
                f"expected {expected!r}. A caller that passes a "
                f"protocol-mismatched action is a defect."
            )

        started = time.perf_counter()
        already_migrated = bool(flow.protocol_data.get("sandbox_migrated", False))
        flow.protocol_data["sandbox_migrated"] = True
        flow.protocol_data["sandbox_action"] = action
        flow.protocol_data["sandbox_at_ms"] = time.time() * 1000.0
        if not already_migrated:
            flow.protocol_data["sandbox_migration_count"] = 1
        else:
            flow.protocol_data["sandbox_migration_count"] = (
                flow.protocol_data.get("sandbox_migration_count", 1) + 1
            )
        elapsed_ms = (time.perf_counter() - started) * 1000.0

        success = not already_migrated
        detail = (
            "connection migrated to sandbox path; cryptographic session "
            "preserved (connection ID unchanged)"
            if success else
            "connection was already migrated; action is a no-op"
        )
        self._log.info(
            "doq_sandbox_action",
            flow_id=flow.flow_id,
            action=action,
            success=success,
            latency_ms=round(elapsed_ms, 4),
            already_migrated=already_migrated,
            migration_count=flow.protocol_data["sandbox_migration_count"],
        )
        return SandboxResult(
            flow_id=flow.flow_id,
            protocol=self.protocol,
            action=action,
            success=success,
            detail=detail,
            latency_ms=elapsed_ms,
        )

    # -- 4. _metric_tensor_kwargs ------------------------------------------

    def _metric_tensor_kwargs(self, x: np.ndarray) -> Dict[str, Any]:
        """
        Return the DoQ metric-tensor kwargs: {d2x_dt2}.

        The `Q` operator is not returned here; the base class builds it
        from riemannian_metrics.yaml §5 and passes it to
        math_utils.metric_tensor_doq directly. The adapter is only
        responsible for the second-derivative argument.

        See the module docstring for the interpretation and its
        limitations.
        """
        x = np.asarray(x, dtype=np.float64)
        if x.shape != self._baseline.shape:
            raise FeatureExtractionError(
                f"DoQ metric tensor requires a {self._baseline.shape} "
                f"feature vector; got shape {x.shape}"
            )
        # Second derivative: deviation from baseline, divided by the
        # squared acceleration timescale.
        d2x_dt2 = (x - self._baseline) / (self._tau * self._tau)
        return {
            "d2x_dt2": d2x_dt2,
        }

    # -- 5. accumulator ----------------------------------------------------

    def _accumulate_doq_query(self, flow: FlowState, payload: bytes) -> None:
        """
        Accumulate the DoQ-specific bits of a query packet.

        - First payload: retain up to 512 bytes for the entropy fallback.
        - Every payload: extract the QUIC packet type from the first
          byte and append it to flow.protocol_data["quic_packet_types"].
        """
        if not payload:
            return
        data = flow.protocol_data
        if data.get("first_payload") is None:
            data["first_payload"] = payload[:_MAX_PAYLOAD_KEEP_BYTES]
        pt = _quic_packet_type(payload)
        if pt is not None:
            data["quic_packet_types"].append(pt)
            if pt == _QUIC_SHORT_HEADER:
                data["n_short_header_packets"] = data.get("n_short_header_packets", 0) + 1


# ---------------------------------------------------------------------------
# QUIC parsing helpers (self-contained, matching preprocess_pcap.py)
# ---------------------------------------------------------------------------

def _quic_packet_type(payload: bytes) -> Optional[int]:
    """
    Return the QUIC packet type (0-3) from the first byte of the UDP
    payload, or None if the payload does not look like a QUIC packet.

    Encoding (matching data/preprocess_pcap.py's `_quic_packet_type`):
        - short header (bit 7 clear) -> 0  (1-RTT)
        - long header (bit 7 set), type (first >> 4) & 0x03:
            0 -> Initial
            1 -> 0-RTT
            2 -> Handshake
            3 -> Retry

    Note: the encoding places Initial packets and short-header 1-RTT
    packets in the same category (index 0), because the mask
    `(first >> 4) & 0x03` for an Initial (0xC0) is 0, the same as a
    short header (0x00-0x7F). This is a documented artefact of the
    offline preprocess_pcap.py, preserved here for consistency. See
    the module docstring for the full discussion.
    """
    if len(payload) < 1:
        return None
    first = payload[0]
    if not (first & 0x80):
        return _QUIC_SHORT_HEADER
    return (first >> 4) & 0x03


def _shannon_entropy_bytes(b: bytes) -> float:
    """Shannon entropy (bits/byte) of a byte string. Matches preprocess_pcap.py."""
    if not b:
        return 0.0
    counts = [0] * 256
    for byte in b:
        counts[byte] += 1
    n = len(b)
    h = 0.0
    for c in counts:
        if c > 0:
            p = c / n
            h -= p * math.log2(p)
    return h


def _l1_distance_uniform(values: List[int], n_categories: int) -> float:
    """
    L1 distance between the empirical distribution of `values` and the
    uniform distribution over `n_categories`. Matches preprocess_pcap.py.
    """
    if not values:
        return 0.0
    counts = [0] * n_categories
    for v in values:
        if 0 <= v < n_categories:
            counts[v] += 1
    n = sum(counts)
    if n == 0:
        return 0.0
    return sum(abs(c / n - 1.0 / n_categories) for c in counts)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "DoQAdapter",
]


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def _self_test() -> int:
    import sys
    rc = 0
    print("=== doq_adapter self-test ===")

    import numpy as np

    metrics_cfg = {
        "doh": {"lambda": 0.15, "alpha": 0.85},
        "dot": {"beta": 0.22, "gamma": 0.65},
        "doq": {
            "eta": 0.35,
            "congestion_operator": {
                "diagonal": {n: 0.5 for n in FEATURE_COLUMNS},
                "off_diagonal": [
                    ["inter_arrival_time_mean", "session_duration", -0.3],
                    ["inter_arrival_time_mean", "burstiness", 0.4],
                ],
            },
        },
    }
    config = {
        "protocols": {"doq": {"quic_version": 1}},
        "_includes": {"riemannian_metrics": metrics_cfg},
    }

    # 1. Construct
    adapter = DoQAdapter(config)
    if adapter._quic_version != 1:
        print(f"[FAIL] quic_version: {adapter._quic_version}")
        rc = 1
    else:
        print(f"[OK]   DoQAdapter constructed (quic_version=1)")

    # 2. Wrong QUIC version rejected
    bad_config = {
        "protocols": {"doq": {"quic_version": 2}},
        "_includes": {"riemannian_metrics": metrics_cfg},
    }
    try:
        DoQAdapter(bad_config)
    except FeatureExtractionError:
        print(f"[OK]   quic_version=2 rejected")
    else:
        print(f"[FAIL] quic_version=2 was accepted")
        rc = 1

    # 3. QUIC packet type encoding
    # Short header (bit 7 clear)
    if _quic_packet_type(b"\x40" + b"\x00" * 10) != _QUIC_SHORT_HEADER:
        print(f"[FAIL] short-header not classified as 1-RTT")
        rc = 1
    # Initial (0xC0-0xCF): first byte 0xC0
    elif _quic_packet_type(b"\xc0" + b"\x00" * 10) != 0:
        # Note: _QUIC_INITIAL is 1, but the mask maps 0xC0 to 0
        print(f"[FAIL] Initial not classified as expected (0xC0 -> "
              f"{_quic_packet_type(b'\xc0' + b'\x00' * 10)})")
        rc = 1
    # 0-RTT (0xD0-0xDF)
    elif _quic_packet_type(b"\xd0" + b"\x00" * 10) != _QUIC_0RTT:
        print(f"[FAIL] 0-RTT not classified as {_QUIC_0RTT}")
        rc = 1
    # Handshake (0xE0-0xEF)
    elif _quic_packet_type(b"\xe0" + b"\x00" * 10) != _QUIC_HANDSHAKE:
        print(f"[FAIL] Handshake not classified as {_QUIC_HANDSHAKE}")
        rc = 1
    else:
        print(f"[OK]   QUIC packet type encoding matches preprocess_pcap.py")
        # Document the collision explicitly.
        print(f"         note: Initial (0xC0) and short-header (0x40) both -> "
              f"category 0 (documented limitation)")

    # 4. Extract features from a synthetic flow
    flow = FlowState(
        flow_id="test-doq-flow",
        protocol="DoQ",
        client_ip="10.0.0.1", client_port=50000,
        server_ip="10.0.0.2", server_port=853,
    )
    for ts, size in [(0.0, 60), (0.05, 80), (0.10, 100)]:
        flow.add_query(ts, size)
    for ts, size in [(0.005, 120), (0.055, 140)]:
        flow.add_response(ts, size)
    # Simulate: 1 long-header Initial (0xC0), 1 long-header Handshake (0xE0),
    # 3 short-header 1-RTT (0x40). Encoding: 0, 3, 0, 0, 0.
    # Fraction short = 3/5 = 0.6
    flow.protocol_data = {
        "quic_packet_types": [0, 3, 0, 0, 0],
        "first_payload": b"\xc0" + b"\x00" * 200,
        "n_short_header_packets": 3,
    }
    feats = adapter.extract_protocol_specific_features(flow)
    if set(feats.keys()) != set(PROTOCOL_FEATURES):
        print(f"[FAIL] feature keys: {sorted(feats.keys())}")
        rc = 1
    elif abs(feats["protocol_specific_feature"] - 0.6) > 1e-9:
        print(f"[FAIL] protocol_specific_feature: "
              f"{feats['protocol_specific_feature']}")
        rc = 1
    else:
        print(f"[OK]   extract_protocol_specific_features: "
              f"entropy={feats['subdomain_entropy']:.4f}, "
              f"l1={feats['query_type_distribution_l1']:.4f}, "
              f"short_header_fraction={feats['protocol_specific_feature']:.4f}")

    # 5. Full feature vector
    full = adapter.extract_features(flow)
    if list(full.keys()) != list(FEATURE_COLUMNS):
        print(f"[FAIL] extract_features key order")
        rc = 1
    else:
        print(f"[OK]   extract_features: 12 keys in canonical order")

    # 6. Metric tensor kwargs
    x = np.array(_DOQ_BASELINE, dtype=np.float64)
    kwargs = adapter._metric_tensor_kwargs(x)
    if "d2x_dt2" not in kwargs:
        print(f"[FAIL] _metric_tensor_kwargs keys: {sorted(kwargs.keys())}")
        rc = 1
    elif not np.allclose(kwargs["d2x_dt2"], 0.0):
        print(f"[FAIL] d2x_dt2 at baseline should be 0, "
              f"got norm {np.linalg.norm(kwargs['d2x_dt2']):.4f}")
        rc = 1
    else:
        print(f"[OK]   _metric_tensor_kwargs: baseline gives zero second derivative")

    # Off-baseline: second derivative is non-zero.
    x_off = x + np.ones_like(x) * 0.5
    kwargs_off = adapter._metric_tensor_kwargs(x_off)
    if np.allclose(kwargs_off["d2x_dt2"], 0.0):
        print(f"[FAIL] d2x_dt2 at off-baseline is zero")
        rc = 1
    else:
        print(f"[OK]   _metric_tensor_kwargs: off-baseline gives non-zero "
              f"second derivative (norm={np.linalg.norm(kwargs_off['d2x_dt2']):.4f})")

    # 7. Metric tensor dispatch
    g = adapter.get_metric_tensor(x)
    if g.shape != (len(FEATURE_COLUMNS), len(FEATURE_COLUMNS)):
        print(f"[FAIL] metric_tensor shape: {g.shape}")
        rc = 1
    elif np.linalg.eigvalsh(g).min() <= 0:
        print(f"[FAIL] metric_tensor not positive-definite")
        rc = 1
    else:
        print(f"[OK]   get_metric_tensor: SPD")

    # 8. Sandbox action
    result = adapter.apply_sandbox_action(flow, SANDBOX_ACTIONS["doq"])
    if not result.success or result.action != "migrate_doq_connection":
        print(f"[FAIL] apply_sandbox_action: {result}")
        rc = 1
    elif result.protocol != "DoQ":
        print(f"[FAIL] apply_sandbox_action protocol: {result.protocol}")
        rc = 1
    elif "preserved" not in result.detail:
        print(f"[FAIL] detail does not mention preserved session: {result.detail}")
        rc = 1
    else:
        print(f"[OK]   apply_sandbox_action: success, migrate_doq_connection, "
              f"session preserved")

    # Idempotent
    result2 = adapter.apply_sandbox_action(flow, SANDBOX_ACTIONS["doq"])
    if result2.success:
        print(f"[FAIL] second apply_sandbox_action should report already-migrated")
        rc = 1
    elif flow.protocol_data["sandbox_migration_count"] != 2:
        print(f"[FAIL] migration_count: {flow.protocol_data['sandbox_migration_count']}")
        rc = 1
    else:
        print(f"[OK]   second apply_sandbox_action: already_migrated, "
              f"migration_count=2")

    # Cross-protocol action rejected
    try:
        adapter.apply_sandbox_action(flow, "reset_dot_tls")
    except SandboxActionError:
        print(f"[OK]   cross-protocol action rejected")
    else:
        print(f"[FAIL] cross-protocol action was accepted")
        rc = 1

    print("=== self-test complete ===")
    return rc


if __name__ == "__main__":
    import sys
    sys.exit(_self_test())
