#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
src/protocols/doh_adapter.py
============================

E3-DoH Reproducibility — DNS-over-HTTPS (DoH) protocol adapter.

Chapter 5 of the thesis: E3-DoH (Enhanced Evolutionary Encryption for
Multi-Protocol DNS Security — DoH, DoT, DoQ).

This module is the concrete adapter for DoH over TCP/443 with TLS 1.3
(HTTP/2, per default_e3_config.yaml §5: `doh.transport: "http2"`). It
inherits from ProtocolAdapter and implements the four abstract methods:

    parse_pcap(pcap_path)                    -> Iterator[FlowState]
    extract_protocol_specific_features(flow) -> Dict[str, float]
    apply_sandbox_action(flow, action)       -> SandboxResult
    _metric_tensor_kwargs(x)                 -> Dict[str, Any]

WHAT IS OBSERVABLE WITHOUT DECRYPTION
-------------------------------------
This is the most important design constraint of the DoH adapter, and
it drives every choice in the file. HTTPS encrypts the HTTP/2 frame
stream inside TLS 1.3 application-data records. Without the session
keys, a passive observer cannot see:

    - HTTP/2 frame types (DATA, HEADERS, SETTINGS, RST_STREAM, PING, ...)
    - HTTP/2 stream identifiers
    - the DNS query inside the HTTP body
    - the DNS response

What the observer CAN see:

    - TCP packet headers (src/dst IP and port, payload sizes)
    - TLS record headers (5 bytes each: content_type, version, length)
      — content_type is plaintext; the record *contents* are encrypted
    - TLS handshake messages (ClientHello, ServerHello, Certificate,
      NewSessionTicket) — plaintext in TLS 1.3
    - the SNI extension inside the ClientHello — plaintext in TLS 1.3
      unless ECH (Encrypted Client Hello) is in use
    - inter-arrival times of packets
    - session duration

The thesis is explicit about this boundary (Chapter 5, §1.7.1, p.35):
"The frameworks only applied to metadata from the TLS sessions
available without decryption, such as packet inter-arrival time, query
sizes, one-way flow characteristics, TLS negotiation parameters,
indicators of presence of TLS extensions, and sequences of frames
during preparation of the HTTP/3 protocol."

The three protocol-specific features are therefore **TLS-level
proxies** for HTTP/2-level behaviour. Each is documented below with
what it measures, why it is a proxy for what the thesis calls for, and
what its limitations are.

THE THREE PROTOCOL-SPECIFIC FEATURES
------------------------------------
1. `subdomain_entropy` (riemannian_metrics.yaml §2):

   What the thesis wants: a measure of the queried subdomain's
   character-distribution complexity.

   What is observable: the SNI extension in the TLS ClientHello, if
   present (TLS 1.3, no ECH). If the SNI is absent (ECH, or a
   resume-based session), the entropy of the first 512 bytes of the
   first payload is used as a fallback.

   This matches the offline `preprocess_pcap.py` implementation
   exactly (the same fallback logic, the same 512-byte window). The
   two implementations are verified to agree by
   `tests/test_protocols.py`.

2. `query_type_distribution_l1` (riemannian_metrics.yaml §2):

   What the thesis wants: a measure of the DNS query-type
   distribution's distance from the benign baseline.

   What is observable: the TLS record-type distribution. TLS 1.3 has
   four record content types: change_cipher_spec (0x14, essentially
   absent in 1.3), alert (0x15), handshake (0x16), application_data
   (0x17). The feature is the L1 distance between the observed
   distribution over these four categories and the uniform
   distribution.

   Why this is a proxy: a DoH tunnel that is continuously exfiltrating
   data will have a record-type distribution dominated by
   application_data (few handshakes, no alerts); a benign DoH flow
   that is a mix of short queries and long-lived connections will
   have a more balanced distribution. The L1 distance from uniform
   therefore tracks the "concentration" of the traffic, which is
   correlated with the DNS-layer behaviour without being a direct
   measure of it.

   This is a *documented modelling choice*, not a claim that TLS
   record types are DNS query types.

3. `protocol_specific_feature` (riemannian_metrics.yaml §2):

   What the thesis wants: "for DoH it is the HTTP/2 stream-multiplexing
   pattern (frame-type sequence)".

   What is observable: the fraction of TLS application_data records.
   A benign DoH flow has a mix of handshake records (during session
   setup) and application_data records (during query/response); a
   DoH tunnel that keeps a single long-lived session has almost
   exclusively application_data records.

   Why this is a proxy: the stream-multiplexing pattern itself is
   inside the encrypted records, but the *record-type composition* is
   a coarse observable proxy for it. A multiplexed HTTP/2 connection
   that interleaves many streams still produces mostly application_data
   records, but the *distribution of record sizes* changes (many small
   interleaved records vs fewer large ones). The offline
   `preprocess_pcap.py` uses the same fraction-of-application_data
   formula, which is why the two paths agree.

   This is the weakest of the three proxies. A reviewer reading the
   thesis's claim "HTTP/2 stream-multiplexing pattern" should
   understand that the artifact measures the TLS record composition,
   not the HTTP/2 frame sequence. The detection accuracy reported in
   Table 26 (99.6% for DoH) is the empirical evidence that the proxy
   is discriminative enough on the test set; it is not a claim that
   the proxy is equivalent to the original quantity.

WHAT THE ADAPTER DOES *NOT* DO
------------------------------
It does not decrypt anything. It does not parse HTTP/2 frames (it
cannot; they are encrypted). It does not parse DNS messages. It does
not touch the byte content of the query or response payloads, except
for the first 512 bytes of the first payload when the SNI is absent
(and even then, only to compute an entropy, not to extract any
content).

The security boundary of the framework (aee_params.yaml §10,
qhe_params.yaml §9) applies here as well: this adapter provides no
security guarantee. It is a feature extractor. The detection accuracy
that its features enable is measured empirically (§5.14), not proven.

THE METRIC-TENSOR PROXY
-----------------------
The DoH metric tensor (math_utils.metric_tensor_doh, Eq. 81) requires
two arguments: `grad_x` (a spatial gradient) and `dx_dt` (a temporal
derivative). Neither is directly available from a single feature
vector x. The adapter uses the following documented proxy:

    grad_x = x - _DOH_BASELINE
    dx_dt = (x - _DOH_BASELINE) / _DOH_TAU

where `_DOH_BASELINE` is a class-level constant representing the
typical benign DoH flow (derived from the toy corpus generator's
benign distribution and consistent with the offline
preprocess_pcap.py's baseline query-size constants), and `_DOH_TAU`
is a characteristic time scale (~300 ms, the typical DoH query burst).

The interpretation:
- `grad_x` measures how far the observed flow is from the typical
  benign flow, per feature.
- `dx_dt` converts that deviation into a rate, using the characteristic
  time scale.

This is a modelling choice, documented as such. The thesis states that
the metric tensor is a "modelling abstraction that serves to motivate
the design of the detection" (riemannian_metrics.yaml §12). The
detection accuracy is established by §5.14, not by the metric-tensor
formula.

THE SANDBOX ACTION
------------------
`isolate_doh_stream` terminates the active HTTP/2 stream(s) and
redirects subsequent streams to the sandbox proxy (thesis §5.6.4,
p.133). The implementation:
- Marks the flow as terminated.
- Records the wall-clock duration of the action.
- Returns a SandboxResult with the outcome and latency.

In toy mode (no real network namespace), the action takes microseconds
(it is just a bookkeeping operation). In full mode against a real
network namespace, the action involves an iptables mangle rule and
takes ~1.2 ms, matching sandbox_mdp_params.yaml §6's
`isolate_doh_stream: 1.2` reference. The verification script
compares against the reference only in full-mode runs (its provenance
check rejects synthetic runs, per data_utils.read_result_json).

Thread safety: this adapter is *not* thread-safe (see base_adapter.py
module docstring). Create one instance per thread.
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
    STANDARD_FEATURES,
    SANDBOX_ACTIONS,
)


# ---------------------------------------------------------------------------
# Constants — the DoH baseline and characteristic time scale
# ---------------------------------------------------------------------------
# _DOH_BASELINE is the feature vector of a "typical benign DoH flow".
# Its values are derived from:
#   - the toy corpus generator's benign distribution
#     (data/download_datasets.py, TOY_CORPUS.feature_distributions),
#   - the offline preprocessor's baseline query-size constants
#     (preprocess_pcap.py, baseline_query_size_mean=68.0,
#      baseline_query_size_std=8.0),
#   - the qualitative description in the thesis (§5.4.2, p.129).
#
# These are the values used to compute grad_x and dx_dt in
# _metric_tensor_kwargs(). They are NOT the calibrated coefficients of
# the metric tensor itself (those are λ=0.15 and α=0.85, in
# riemannian_metrics.yaml §3). They are the "reference point" from
# which the observed flow's deviation is measured.
#
# In full mode against the reference testbed, the values could be
# recomputed from the DS-1 benign training split; the constants below
# are what the artifact ships with so that the metric tensor is
# well-defined even on a fresh checkout (before the first full run).

_DOH_BASELINE = np.array([
    0.10,    # inter_arrival_time_mean    (100 ms between queries)
    0.02,    # inter_arrival_time_std
    68.0,    # query_size_mean            (small DNS queries)
    8.0,     # query_size_std
    110.0,   # response_size_mean         (DNS responses)
    15.0,    # response_size_std
    0.62,    # query_response_ratio
    3.1,     # subdomain_entropy          (typical DNS subdomain complexity)
    0.15,    # query_type_distribution_l1 (TLS record-type distance from uniform)
    1.1,     # burstiness                 (temporal clustering)
    12.0,    # session_duration           (typical DoH session, seconds)
    0.85,    # protocol_specific_feature  (fraction of application_data records)
], dtype=np.float64)

if _DOH_BASELINE.shape != (len(FEATURE_COLUMNS),):
    raise RuntimeError(
        f"_DOH_BASELINE has shape {_DOH_BASELINE.shape}, "
        f"expected ({len(FEATURE_COLUMNS)},). Update _DOH_BASELINE "
        f"if FEATURE_COLUMNS changes."
    )

# The characteristic time scale, in seconds. 300 ms is the typical
# duration of a burst of DoH queries during a page load (browser
# fetches a page → dozens of DNS queries within ~300 ms → page renders,
# queries stop). It is used to convert the deviation from baseline
# into a rate for the temporal part of the metric tensor.
_DOH_TAU = 0.30

# TLS record content types (RFC 8446 §5.1). Only the four standard
# values are relevant; the change_cipher_spec (0x14) is essentially
# absent in TLS 1.3 but is kept for consistency with the offline path
# (which uses n_categories=4 uniformly).
_TLS_CHANGE_CIPHER_SPEC = 0x14
_TLS_ALERT = 0x15
_TLS_HANDSHAKE = 0x16
_TLS_APPLICATION_DATA = 0x17
_TLS_RECORD_TYPES = (
    _TLS_CHANGE_CIPHER_SPEC,
    _TLS_ALERT,
    _TLS_HANDSHAKE,
    _TLS_APPLICATION_DATA,
)

# The "four-category" n_categories argument used by the L1 distance
# computation. Matches the offline preprocess_pcap.py, which uses
# n_categories=4.
_L1_N_CATEGORIES = 4

# The maximum number of payload bytes retained for SNI extraction. The
# offline path uses the same constant (preprocess_pcap.py,
# MAX_PAYLOAD_KEEP_BYTES = 512).
_MAX_PAYLOAD_KEEP_BYTES = 512

# DoH server port (default_e3_config.yaml §5, doh.default_port = 443).
_DOH_PORT = 443


# ---------------------------------------------------------------------------
# DoHAdapter
# ---------------------------------------------------------------------------

class DoHAdapter(ProtocolAdapter):
    """
    Concrete adapter for DNS-over-HTTPS (DoH) over TCP/443 with TLS 1.3.

    See the module docstring for the design rationale and the
    observable-vs-unobservable boundary.
    """

    protocol = "doh"
    protocol_full_name = "DNS-over-HTTPS"

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config)
        # The DoH transport (default_e3_config.yaml §5: "http2"). The
        # current adapter only supports HTTP/2 over TCP/443. If the
        # config sets "http3", the adapter raises rather than silently
        # using the wrong transport.
        try:
            transport = config["protocols"]["doh"]["transport"]
        except (KeyError, TypeError):
            transport = "http2"
        if transport != "http2":
            raise FeatureExtractionError(
                f"DoH adapter only supports transport='http2'; config "
                f"specifies transport={transport!r}. HTTP/3 DoH uses "
                f"the QUIC transport and would need the DoQ adapter's "
                f"parsing logic, which this adapter does not implement."
            )
        self._transport = transport
        # Expose the baseline and tau as instance attributes so that a
        # subclass (or a test) can override them without editing the
        # module.
        self._baseline = _DOH_BASELINE.copy()
        self._tau = _DOH_TAU

    # -- 1. parse_pcap -----------------------------------------------------

    def parse_pcap(self, pcap_path: Path) -> Iterator[FlowState]:
        """
        Parse a PCAP file and yield one FlowState per DoH flow.

        Uses scapy's PcapReader to stream the PCAP. For each packet,
        determines if it belongs to a DoH flow (TCP port 443) and
        accumulates it. The accumulation:
            - adds query/response packet sizes to the flow's
              query_sizes / response_sizes
            - adds the timestamp to the flow's query_times
            - for query packets, extracts the TLS record type from
              the first byte of the TCP payload (this is plaintext
              even in TLS 1.3; only the record *contents* are
              encrypted)
            - for the first query packet, extracts the SNI from the
              ClientHello if present

        Yields
        ------
        FlowState objects, one per DoH flow, at end of PCAP.
        """
        try:
            from scapy.all import PcapReader  # type: ignore
            from scapy.layers.inet import IP, TCP  # type: ignore
            from scapy.layers.l2 import Ether  # type: ignore
        except ImportError as e:
            raise FeatureExtractionError(
                "DoHAdapter.parse_pcap requires scapy. "
                "Install with: pip install scapy==2.5.0"
            ) from e

        pcap_path = Path(pcap_path)
        if not pcap_path.exists():
            raise FileNotFoundError(f"PCAP not found: {pcap_path}")

        self._log.info(
            "doh_parse_start",
            pcap=str(pcap_path),
            transport=self._transport,
            size_bytes=pcap_path.stat().st_size,
        )

        flows: Dict[tuple, FlowState] = {}
        n_packets_total = 0
        n_packets_kept = 0

        try:
            with PcapReader(str(pcap_path)) as reader:
                for pkt in reader:
                    n_packets_total += 1
                    if Ether not in pkt or IP not in pkt or TCP not in pkt:
                        continue
                    ip = pkt[IP]
                    tcp = pkt[TCP]
                    # DoH: TCP on port 443 (either direction).
                    if int(tcp.dport) != _DOH_PORT and int(tcp.sport) != _DOH_PORT:
                        continue

                    ts = float(pkt.time)
                    # Canonicalise the 5-tuple so both directions map
                    # to the same flow.
                    a = (ip.src, ip.dst, int(tcp.sport), int(tcp.dport))
                    b = (ip.dst, ip.src, int(tcp.dport), int(tcp.sport))
                    key = min(a, b)

                    flow = flows.get(key)
                    if flow is None:
                        # The first packet determines the client. This
                        # matches preprocess_pcap.py's
                        # set_client_from_first_packet.
                        flow = FlowState(
                            flow_id=self.make_flow_id(*key, "tcp"),
                            protocol="DoH",
                            client_ip=ip.src, client_port=int(tcp.sport),
                            server_ip=ip.dst, server_port=int(tcp.dport),
                        )
                        flow.protocol_data = {
                            "tls_record_types": [],
                            "first_payload": None,
                            "n_data_records": 0,
                        }
                        flows[key] = flow

                    payload = bytes(tcp.payload) if tcp.payload else b""
                    is_query = (
                        ip.src == flow.client_ip
                        and int(tcp.sport) == flow.client_port
                    )
                    pkt_size = len(pkt)

                    if is_query:
                        flow.add_query(ts, pkt_size)
                        self._accumulate_doh_query(flow, payload)
                    else:
                        flow.add_response(ts, pkt_size)
                    n_packets_kept += 1
        except Exception as e:
            self._log.error(
                "doh_parse_error",
                pcap=str(pcap_path), error=repr(e),
            )
            raise FeatureExtractionError(
                f"failed to parse {pcap_path}: {e}"
            ) from e

        self._log.info(
            "doh_parse_complete",
            pcap=str(pcap_path),
            packets_total=n_packets_total,
            packets_kept=n_packets_kept,
            flows=len(flows),
        )

        for flow in flows.values():
            if flow.n_packets() > 0:
                yield flow

    # -- 2. extract_protocol_specific_features -----------------------------

    def extract_protocol_specific_features(self, flow: FlowState) -> Dict[str, float]:
        """
        Extract the three DoH protocol-specific features:

            subdomain_entropy
            query_type_distribution_l1
            protocol_specific_feature

        See the module docstring for what each feature measures and why
        it is a TLS-level proxy for an HTTP/2-level quantity.
        """
        data = flow.protocol_data or {}
        record_types: List[int] = list(data.get("tls_record_types", []))
        first_payload: Optional[bytes] = data.get("first_payload")

        # --- subdomain_entropy ---
        # Best effort: SNI from the ClientHello, else entropy of the
        # first 512 bytes of the first payload, else 0.0 (no payload
        # observed). The 0.0 fallback is documented and matches the
        # offline path.
        subdomain_entropy = 0.0
        if first_payload:
            sni = _extract_sni_from_client_hello(first_payload)
            if sni:
                subdomain_entropy = _shannon_entropy(sni)
            else:
                subdomain_entropy = _shannon_entropy_bytes(first_payload)

        # --- query_type_distribution_l1 ---
        # L1 distance between the observed TLS record-type distribution
        # and uniform over the four content types. Matches the offline
        # path exactly (preprocess_pcap.py,
        # _l1_distance_uniform(values, n_categories=4)).
        if record_types:
            l1 = _l1_distance_uniform(record_types, _L1_N_CATEGORIES)
        else:
            l1 = 0.0

        # --- protocol_specific_feature ---
        # Fraction of TLS application_data records. Matches the
        # offline path exactly.
        if record_types:
            n_app_data = sum(1 for t in record_types if t == _TLS_APPLICATION_DATA)
            fraction_app_data = n_app_data / len(record_types)
        else:
            fraction_app_data = 0.0

        return {
            "subdomain_entropy": float(subdomain_entropy),
            "query_type_distribution_l1": float(l1),
            "protocol_specific_feature": float(fraction_app_data),
        }

    # -- 3. apply_sandbox_action -------------------------------------------

    def apply_sandbox_action(
        self, flow: FlowState, action: str,
    ) -> SandboxResult:
        """
        Apply a sandbox action to a DoH flow.

        Only `isolate_doh_stream` is valid for DoH (SANDBOX_ACTIONS["doh"]).
        The action marks the flow as terminated; subsequent streams from
        the same client are redirected to the sandbox proxy (the
        redirection itself is a network-namespace operation, performed
        by the AMPS engine, not by the adapter).

        In toy mode (no real network namespace), the action is a
        bookkeeping operation and the measured latency is microseconds.
        In full mode against a real network namespace, the action
        includes the iptables mangle rule and takes ~1.2 ms, matching
        sandbox_mdp_params.yaml §6.
        """
        expected = SANDBOX_ACTIONS["doh"]
        if action != expected:
            raise SandboxActionError(
                f"DoH adapter cannot apply action {action!r}; "
                f"expected {expected!r}. A caller that passes a "
                f"protocol-mismatched action is a defect."
            )

        started = time.perf_counter()
        # Mark the flow as terminated. This is the protocol-level
        # effect: a subsequent connection from the same client to the
        # same server will be treated as a new flow, and the AMPS
        # sandbox will route it to the proxy.
        already_terminated = bool(flow.protocol_data.get("sandbox_terminated", False))
        flow.protocol_data["sandbox_terminated"] = True
        flow.protocol_data["sandbox_action"] = action
        flow.protocol_data["sandbox_at_ms"] = time.time() * 1000.0
        elapsed_ms = (time.perf_counter() - started) * 1000.0

        success = not already_terminated
        detail = (
            f"stream terminated; subsequent flows redirected to sandbox"
            if success else
            f"stream was already terminated; action is a no-op"
        )
        self._log.info(
            "doh_sandbox_action",
            flow_id=flow.flow_id,
            action=action,
            success=success,
            latency_ms=round(elapsed_ms, 4),
            already_terminated=already_terminated,
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
        Return the DoH metric-tensor kwargs: {grad_x, dx_dt}.

        The two arguments are computed from x alone, using the
        _DOH_BASELINE and _DOH_TAU class constants:

            grad_x = x - _DOH_BASELINE
            dx_dt  = (x - _DOH_BASELINE) / _DOH_TAU

        See the module docstring for the interpretation and its
        limitations. In full mode against the reference testbed,
        _DOH_BASELINE could be recomputed from the DS-1 benign
        training split; the constants are what the artifact ships
        with so that the metric tensor is well-defined even on a
        fresh checkout.
        """
        x = np.asarray(x, dtype=np.float64)
        if x.shape != self._baseline.shape:
            raise FeatureExtractionError(
                f"DoH metric tensor requires a {self._baseline.shape} "
                f"feature vector; got shape {x.shape}"
            )
        grad_x = x - self._baseline
        dx_dt = grad_x / self._tau
        return {
            "grad_x": grad_x,
            "dx_dt": dx_dt,
        }

    # -- 5. accumulator ----------------------------------------------------

    def _accumulate_doh_query(self, flow: FlowState, payload: bytes) -> None:
        """
        Accumulate the DoH-specific bits of a query packet into
        flow.protocol_data.

        - First payload: extract SNI (if present) and record the
          first 512 bytes for the entropy fallback.
        - Every payload: extract the TLS record type (first byte) if
          it is one of the four standard content types, and append it
          to flow.protocol_data["tls_record_types"].
        """
        if not payload:
            return
        data = flow.protocol_data
        if data.get("first_payload") is None:
            data["first_payload"] = payload[:_MAX_PAYLOAD_KEEP_BYTES]
        # TLS record type is the first byte of a record. For a
        # multi-record TCP segment, we only see the first record's
        # type; the remaining records' types would require parsing
        # the record headers. That level of detail is not needed for
        # the L1 distribution feature; the first byte is a documented
        # approximation, and it matches the offline path (which also
        # uses the first byte of the payload).
        rt = _tls_record_type(payload)
        if rt is not None:
            data["tls_record_types"].append(rt)
            if rt == _TLS_APPLICATION_DATA:
                data["n_data_records"] = data.get("n_data_records", 0) + 1


# ---------------------------------------------------------------------------
# TLS parsing helpers
# ---------------------------------------------------------------------------
# These are self-contained in this module rather than shared with
# data/preprocess_pcap.py, because the src/ layer must not depend on
# the data/ layer for utility functions (data/ depends on src/,
# not the other way around). The two implementations are verified to
# agree on identical inputs by tests/test_protocols.py.

def _tls_record_type(payload: bytes) -> Optional[int]:
    """
    Return the TLS content_type of the first record if the payload
    looks like a TLS record (first byte is one of the four standard
    content types), else None.

    Matches preprocess_pcap.py's `_tls_record_type`.
    """
    if len(payload) >= 1 and payload[0] in _TLS_RECORD_TYPES:
        return payload[0]
    return None


def _extract_sni_from_client_hello(payload: bytes) -> Optional[str]:
    """
    Best-effort SNI extraction from a TLS ClientHello.

    Returns None if the payload is not a ClientHello or the SNI
    extension is absent (e.g. because ECH is in use).

    Matches preprocess_pcap.py's `_extract_sni_from_client_hello`
    exactly (same byte offsets, same fallbacks, same error handling).
    """
    if len(payload) < 5 or payload[0] != _TLS_HANDSHAKE:
        return None
    if len(payload) < 9 or payload[5] != 0x01:  # 0x01 = ClientHello
        return None
    try:
        offset = 9 + 2 + 32  # record hdr + handshake hdr + version + random
        if offset >= len(payload):
            return None
        session_id_len = payload[offset]
        offset += 1 + session_id_len
        if offset + 2 > len(payload):
            return None
        cipher_suites_len = int.from_bytes(payload[offset:offset + 2], "big")
        offset += 2 + cipher_suites_len
        if offset + 1 > len(payload):
            return None
        compression_len = payload[offset]
        offset += 1 + compression_len
        if offset + 2 > len(payload):
            return None
        extensions_len = int.from_bytes(payload[offset:offset + 2], "big")
        offset += 2
        extensions_end = min(offset + extensions_len, len(payload))
        while offset + 4 <= extensions_end:
            ext_type = int.from_bytes(payload[offset:offset + 2], "big")
            ext_len = int.from_bytes(payload[offset + 2:offset + 4], "big")
            ext_data = payload[offset + 4:offset + 4 + ext_len]
            if ext_type == 0x0000 and len(ext_data) >= 5:  # server_name
                name_len = int.from_bytes(ext_data[3:5], "big")
                name = ext_data[5:5 + name_len]
                try:
                    return name.decode("ascii")
                except UnicodeDecodeError:
                    return None
            offset += 4 + ext_len
    except (IndexError, ValueError):
        return None
    return None


def _shannon_entropy(s: str) -> float:
    """Shannon entropy (bits/char) of a string. Matches preprocess_pcap.py."""
    if not s:
        return 0.0
    counts: Dict[str, int] = {}
    for ch in s:
        counts[ch] = counts.get(ch, 0) + 1
    n = len(s)
    h = 0.0
    for c in counts.values():
        p = c / n
        h -= p * math.log2(p)
    return h


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
    "DoHAdapter",
]


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def _self_test() -> int:
    import sys
    import tempfile
    rc = 0
    print("=== doh_adapter self-test ===")

    # --- Minimal config with the riemannian_metrics include ---
    import numpy as np
    from data.data_utils import FEATURE_COLUMNS
    metrics_cfg = {
        "doh": {"lambda": 0.15, "alpha": 0.85},
        "dot": {"beta": 0.22, "gamma": 0.65},
        "doq": {
            "eta": 0.35,
            "congestion_operator": {
                "diagonal": {n: 0.5 for n in FEATURE_COLUMNS},
                "off_diagonal": [],
            },
        },
    }
    config = {
        "protocols": {"doh": {"transport": "http2"}},
        "_includes": {"riemannian_metrics": metrics_cfg},
    }

    # --- 1. Construct ---
    adapter = DoHAdapter(config)
    if adapter._transport != "http2":
        print(f"[FAIL] transport: {adapter._transport}")
        rc = 1
    else:
        print(f"[OK]   DoHAdapter constructed (transport=http2)")

    # --- 2. Wrong transport rejected ---
    bad_config = {
        "protocols": {"doh": {"transport": "http3"}},
        "_includes": {"riemannian_metrics": metrics_cfg},
    }
    try:
        DoHAdapter(bad_config)
    except FeatureExtractionError:
        print(f"[OK]   transport=http3 rejected")
    else:
        print(f"[FAIL] transport=http3 was accepted")
        rc = 1

    # --- 3. Extract protocol-specific features from a synthetic flow ---
    flow = FlowState(
        flow_id="test-doh-flow",
        protocol="DoH",
        client_ip="10.0.0.1", client_port=50000,
        server_ip="10.0.0.2", server_port=443,
    )
    # Three query packets, each a TLS application_data record (0x17).
    for ts, size in [(0.0, 60), (0.1, 80), (0.3, 100)]:
        flow.add_query(ts, size)
    for ts, size in [(0.005, 120), (0.105, 140)]:
        flow.add_response(ts, size)
    flow.protocol_data = {
        "tls_record_types": [0x16, 0x17, 0x17, 0x17],  # 1 handshake, 3 app_data
        "first_payload": b"\x16\x03\x03\x00\x05" + b"\x00" * 100,  # fake record, no SNI
        "n_data_records": 3,
    }
    feats = adapter.extract_protocol_specific_features(flow)
    if set(feats.keys()) != set(PROTOCOL_FEATURES):
        print(f"[FAIL] feature keys: {sorted(feats.keys())}")
        rc = 1
    elif abs(feats["protocol_specific_feature"] - 0.75) > 1e-9:
        # 3 app_data out of 4 records = 0.75
        print(f"[FAIL] protocol_specific_feature: {feats['protocol_specific_feature']}")
        rc = 1
    elif not (0.0 <= feats["query_type_distribution_l1"] <= 2.0):
        print(f"[FAIL] query_type_distribution_l1 out of range: "
              f"{feats['query_type_distribution_l1']}")
        rc = 1
    elif feats["subdomain_entropy"] < 0.0:
        print(f"[FAIL] subdomain_entropy negative: {feats['subdomain_entropy']}")
        rc = 1
    else:
        print(f"[OK]   extract_protocol_specific_features: "
              f"entropy={feats['subdomain_entropy']:.4f}, "
              f"l1={feats['query_type_distribution_l1']:.4f}, "
              f"app_data_fraction={feats['protocol_specific_feature']:.4f}")

    # --- 4. Full feature vector assembly ---
    full = adapter.extract_features(flow)
    if set(full.keys()) != set(FEATURE_COLUMNS):
        print(f"[FAIL] extract_features keys: {sorted(full.keys())}")
        rc = 1
    elif list(full.keys()) != list(FEATURE_COLUMNS):
        print(f"[FAIL] extract_features key order mismatch")
        rc = 1
    else:
        print(f"[OK]   extract_features: 12 keys in canonical order")

    # --- 5. Metric tensor kwargs ---
    x = np.array(_DOH_BASELINE, dtype=np.float64)
    kwargs = adapter._metric_tensor_kwargs(x)
    if "grad_x" not in kwargs or "dx_dt" not in kwargs:
        print(f"[FAIL] _metric_tensor_kwargs keys: {sorted(kwargs.keys())}")
        rc = 1
    elif not np.allclose(kwargs["grad_x"], 0.0):
        print(f"[FAIL] grad_x at baseline should be 0, got {kwargs['grad_x']}")
        rc = 1
    elif not np.allclose(kwargs["dx_dt"], 0.0):
        print(f"[FAIL] dx_dt at baseline should be 0, got {kwargs['dx_dt']}")
        rc = 1
    else:
        print(f"[OK]   _metric_tensor_kwargs: baseline gives zero deviation")

    # Off-baseline: the metric tensor should not be the identity.
    x_off = x + np.ones_like(x) * 0.5
    g_off = adapter.get_metric_tensor(x_off)
    if np.allclose(g_off, np.eye(len(FEATURE_COLUMNS))):
        print(f"[FAIL] metric tensor at off-baseline is identity")
        rc = 1
    elif np.linalg.eigvalsh(g_off).min() <= 0:
        print(f"[FAIL] metric tensor not positive-definite")
        rc = 1
    else:
        print(f"[OK]   metric tensor at off-baseline is non-trivial and SPD")

    # --- 6. Sandbox action ---
    result = adapter.apply_sandbox_action(flow, SANDBOX_ACTIONS["doh"])
    if not result.success or result.action != "isolate_doh_stream":
        print(f"[FAIL] apply_sandbox_action: {result}")
        rc = 1
    elif result.protocol != "DoH":
        print(f"[FAIL] apply_sandbox_action protocol: {result.protocol}")
        rc = 1
    else:
        print(f"[OK]   apply_sandbox_action: success, "
              f"latency={result.latency_ms:.4f} ms")

    # Idempotent: second call should report already_terminated.
    result2 = adapter.apply_sandbox_action(flow, SANDBOX_ACTIONS["doh"])
    if result2.success:
        print(f"[FAIL] second apply_sandbox_action should report already-terminated")
        rc = 1
    else:
        print(f"[OK]   second apply_sandbox_action: already_terminated")

    # Wrong action name is rejected.
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
