#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
src/protocols/dot_adapter.py
============================

E3-DoH Reproducibility — DNS-over-TLS (DoT) protocol adapter.

Chapter 5 of the thesis: E3-DoH (Enhanced Evolutionary Encryption for
Multi-Protocol DNS Security — DoH, DoT, DoQ).

This module is the concrete adapter for DoT over TCP/853 with TLS 1.3
(default_e3_config.yaml §5: `dot.default_port: 853`,
`dot.tls_min_version: "1.3"`). It inherits from ProtocolAdapter and
implements the four abstract methods:

    parse_pcap(pcap_path)                    -> Iterator[FlowState]
    extract_protocol_specific_features(flow) -> Dict[str, float]
    apply_sandbox_action(flow, action)       -> SandboxResult
    _metric_tensor_kwargs(x)                 -> Dict[str, Any]

HOW DOT DIFFERS FROM DOH
------------------------
DoT is the closest sibling to DoH — both run over TCP with TLS 1.3,
both use the same four TLS record content types, both encrypt the same
DNS wire format. The differences that matter for the adapter are:

1. Port: DoT is on TCP/853, DoH on TCP/443. The adapter classifies a
   flow as DoT by checking for port 853 (default_e3_config.yaml §5),
   not 443.

2. Record-boundary structure. Thesis §5.4.4 (p.129) states that the
   DoT metric tensor "was developed to describe the behavior of TLS
   1.3-based DNS resolution, such as the handshake dynamics, the
   boundaries of records and the ratio of query size to response
   size." Unlike DoH, whose HTTP/2 stream multiplexing interleaves
   streams within a single connection, DoT carries each DNS query/
   response pair as a self-contained sequence of TLS records. The
   record boundary therefore carries information: a DoT tunnel that
   exfiltrates data will have an anomalous record-size distribution
   (many small records or one very large record) compared to a benign
   flow. The DoT metric tensor's β (bandwidth) and γ (RTT) terms
   capture this, without needing an extra feature beyond the twelve
   canonical ones (see the Metric tensor kwargs section below).

3. The three protocol-specific features. The base class requires the
   adapter to produce exactly `subdomain_entropy`,
   `query_type_distribution_l1`, and `protocol_specific_feature`.
   The thesis's `protocol_specific_feature` description for DoT is
   "TLS record-boundary structure" (riemannian_metrics.yaml §2).
   The adapter implements this as the fraction of application_data
   records, which is the same formula used by `data/preprocess_pcap.py`
   for DoT (see that module's protocol-specific-feature branch). The
   two are verified to agree by `tests/test_protocols.py`. The
   distinction between DoT and DoH is carried by the *metric tensor*
   (β, γ vs λ, α), not by a different feature formula.

4. Sandbox action. DoT's sandbox primitive is `reset_dot_tls`
   (thesis §5.6.4, p.133): "Terminate the TLS session, reset the
   transport context, and force the next connection through the
   sandbox." A DoT session reset is *not* the same as a DoH stream
   termination: DoH has an HTTP/2 layer with multiplexed streams that
   can be terminated individually, whereas DoT has one TLS session
   per connection, so the reset is a full connection reset.

5. Metric tensor. DoT uses `math_utils.metric_tensor_dot`
   (Eq. 83, p.129), which takes `(x_bandwidth, rtt)` and the
   coefficients β, γ from riemannian_metrics.yaml §4. This is
   dispatched by the base class, so the adapter only has to provide
   the kwargs.

WHAT IS OBSERVABLE WITHOUT DECRYPTION
-------------------------------------
The DoT observable surface is almost identical to DoH's (see
`doh_adapter.py` module docstring for the full discussion):

    - TCP packet headers (src/dst IP and port, sizes, timestamps)
    - TLS record headers (content_type plaintext, contents encrypted)
    - TLS handshake messages (ClientHello, ServerHello, ...)
    - SNI extension in the ClientHello (plaintext unless ECH)
    - inter-arrival times, session duration

The adapter does not decrypt, does not parse DNS wire-format, and does
not touch payload content. It computes the three protocol-specific
features from the same TLS-level observables as the DoH adapter; the
difference is in the *formula* the metric tensor applies to them (β, γ
vs λ, α), not in what is extracted.

METRIC TENSOR KWARGS
--------------------
The DoT metric tensor (Eq. 83, p.129) is:

    g_DoT(x) = (1 + β ||x||_Bandwidth²) δ_ij + γ R_rtt dτ²

Two protocol-specific quantities:

    ||x||_Bandwidth : the L2 norm of the traffic state projected onto
                      the bandwidth dimension
    R_rtt           : the round-trip time

Neither is directly a feature in x. The adapter uses the following
documented proxies:

    ||x||_Bandwidth ≈ sqrt( query_size_mean² + response_size_mean² )
                      / _DOT_BASELINE_BANDWIDTH

        The L2 norm of the two size features, normalised by the
        baseline bandwidth. The normalisation makes the result
        dimensionless, which is what the metric-tensor formula
        requires (the β coefficient is dimensionless).

    R_rtt           ≈ inter_arrival_time_mean / _DOT_BASELINE_RTT

        The inter-arrival time mean, normalised by the baseline RTT.
        This is the closest available proxy in the twelve-feature
        vector for the RTT: the inter-arrival time of consecutive
        queries includes the server's response time, which is
        dominated by the RTT when queries are pipelined back-to-back.
        For a non-pipelined flow, the inter-arrival time is dominated
        by client think-time and is a weaker RTT proxy — documented as
        a limitation.

The baselines are documented below as class-level constants. In full
mode against the reference testbed they can be recomputed from the
DS-1 benign DoT training split; the constants are what the artifact
ships with so that the metric tensor is well-defined on a fresh
checkout.

SECURITY BOUNDARY
-----------------
Same as the DoH adapter (and the framework as a whole): no security
claim from this module. See aee_params.yaml §10 and qhe_params.yaml
§9 for the full boundary.

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
# Constants — DoT baselines for the metric-tensor kwargs
# ---------------------------------------------------------------------------
# _DOT_BASELINE is the feature vector of a "typical benign DoT flow".
# Values are derived from:
#   - the thesis's description of DoT as a lower-latency, single-message
#     transport (riemannian_metrics.yaml §4 and the "DoT is fastest"
#     note in aee_params.yaml §8),
#   - the toy corpus's benign distribution (download_datasets.py,
#     TOY_CORPUS.feature_distributions),
#   - the DoH baseline (doh_adapter.py, _DOH_BASELINE), adjusted for
#     DoT's lower inter-arrival time and slightly smaller session.
#
# The critical differences from the DoH baseline are:
#   - smaller inter_arrival_time_mean (DoT has lower handshake latency,
#     so queries are closer together),
#   - lower session_duration (DoT is more often used for one-shot
#     queries than DoH),
#   - slightly different protocol_specific_feature (TLS record
#     boundaries are cleaner in DoT because there is no multiplexing).

_DOT_BASELINE = np.array([
    0.08,    # inter_arrival_time_mean    (80 ms between queries; lower than DoH)
    0.02,    # inter_arrival_time_std
    68.0,    # query_size_mean            (same DNS queries)
    8.0,     # query_size_std
    110.0,   # response_size_mean
    15.0,    # response_size_std
    0.62,    # query_response_ratio
    3.1,     # subdomain_entropy
    0.12,    # query_type_distribution_l1 (slightly tighter TLS record dist. than DoH)
    0.9,     # burstiness
    8.0,     # session_duration           (shorter than DoH's 12 s)
    0.90,    # protocol_specific_feature  (cleaner record boundaries; higher app_data fraction)
], dtype=np.float64)

if _DOT_BASELINE.shape != (len(FEATURE_COLUMNS),):
    raise RuntimeError(
        f"_DOT_BASELINE has shape {_DOT_BASELINE.shape}, "
        f"expected ({len(FEATURE_COLUMNS)},). Update _DOT_BASELINE "
        f"if FEATURE_COLUMNS changes."
    )

# Baseline bandwidth: the L2 norm of (query_size_mean, response_size_mean)
# at the baseline flow.  sqrt(68² + 110²) ≈ 129.4. The adapter divides
# the observed L2 norm by this value to get a dimensionless ratio.
_DOT_BASELINE_BANDWIDTH = float(math.sqrt(68.0 * 68.0 + 110.0 * 110.0))

# Baseline RTT: the inter-arrival time mean at the baseline flow, in
# seconds. The adapter divides the observed inter-arrival mean by this
# value to get a dimensionless ratio. Matches the DoT baseline above
# (inter_arrival_time_mean = 0.08 s).
_DOT_BASELINE_RTT = 0.08

# The characteristic transport-coordinate step for the DoT metric
# tensor. The `dτ` in Eq. 83 is a normalisation; using a fixed 1.0
# means the rtt term in the tensor has the same units as the bandwidth
# term. This matches the default in math_utils.metric_tensor_dot
# (d_tau=1.0).
_DOT_D_TAU = 1.0

# DoT server port (default_e3_config.yaml §5, dot.default_port = 853).
_DOT_PORT = 853

# TLS record content types (RFC 8446 §5.1). Same four as the DoH adapter.
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

# Four-category L1 distance, matching preprocess_pcap.py.
_L1_N_CATEGORIES = 4

# First-payload retention, matching preprocess_pcap.py.
_MAX_PAYLOAD_KEEP_BYTES = 512


# ---------------------------------------------------------------------------
# DoTAdapter
# ---------------------------------------------------------------------------

class DoTAdapter(ProtocolAdapter):
    """
    Concrete adapter for DNS-over-TLS (DoT) over TCP/853 with TLS 1.3.

    See the module docstring for the design rationale, the record-
    boundary distinction from DoH, and the metric-tensor kwargs.
    """

    protocol = "dot"
    protocol_full_name = "DNS-over-TLS"

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config)
        # The DoT TLS minimum version (default_e3_config.yaml §5:
        # "1.3"). The adapter enforces that the configured value is
        # 1.3; a lower version would change the record-header format
        # (TLS 1.2 records have the same 5-byte header but the
        # content-type semantics differ during renegotiation, which
        # the adapter does not handle).
        try:
            tls_min = config["protocols"]["dot"]["tls_min_version"]
        except (KeyError, TypeError):
            tls_min = "1.3"
        if str(tls_min) != "1.3":
            raise FeatureExtractionError(
                f"DoT adapter only supports tls_min_version='1.3'; config "
                f"specifies {tls_min!r}. TLS 1.2 renegotiation would produce "
                f"record-type sequences the adapter does not classify."
            )
        self._tls_min_version = str(tls_min)
        # Expose the baselines as instance attributes so that a subclass
        # or a test can override them without editing the module.
        self._baseline = _DOT_BASELINE.copy()
        self._baseline_bandwidth = _DOT_BASELINE_BANDWIDTH
        self._baseline_rtt = _DOT_BASELINE_RTT
        self._d_tau = _DOT_D_TAU

    # -- 1. parse_pcap -----------------------------------------------------

    def parse_pcap(self, pcap_path: Path) -> Iterator[FlowState]:
        """
        Parse a PCAP file and yield one FlowState per DoT flow.

        Uses scapy's PcapReader to stream the PCAP. For each packet,
        determines if it belongs to a DoT flow (TCP port 853) and
        accumulates it. The accumulation:
            - adds query/response packet sizes to the flow's
              query_sizes / response_sizes
            - adds the timestamp to the flow's query_times
            - for query packets, extracts the TLS record type from
              the first byte of the TCP payload
            - for the first query packet, retains up to 512 bytes for
              the SNI-based subdomain_entropy feature

        The DoT adapter is deliberately identical in structure to the
        DoH adapter, with two differences: the port (853 vs 443) and
        the flow_id's transport tag (both "tcp", but the protocol
        string that goes into the FlowState is "DoT").

        Yields
        ------
        FlowState objects, one per DoT flow, at end of PCAP.
        """
        try:
            from scapy.all import PcapReader  # type: ignore
            from scapy.layers.inet import IP, TCP  # type: ignore
            from scapy.layers.l2 import Ether  # type: ignore
        except ImportError as e:
            raise FeatureExtractionError(
                "DoTAdapter.parse_pcap requires scapy. "
                "Install with: pip install scapy==2.5.0"
            ) from e

        pcap_path = Path(pcap_path)
        if not pcap_path.exists():
            raise FileNotFoundError(f"PCAP not found: {pcap_path}")

        self._log.info(
            "dot_parse_start",
            pcap=str(pcap_path),
            tls_min_version=self._tls_min_version,
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
                    # DoT: TCP on port 853 (either direction).
                    if int(tcp.dport) != _DOT_PORT and int(tcp.sport) != _DOT_PORT:
                        continue

                    ts = float(pkt.time)
                    a = (ip.src, ip.dst, int(tcp.sport), int(tcp.dport))
                    b = (ip.dst, ip.src, int(tcp.dport), int(tcp.sport))
                    key = min(a, b)

                    flow = flows.get(key)
                    if flow is None:
                        flow = FlowState(
                            flow_id=self.make_flow_id(*key, "tcp"),
                            protocol="DoT",
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
                        self._accumulate_dot_query(flow, payload)
                    else:
                        flow.add_response(ts, pkt_size)
                    n_packets_kept += 1
        except Exception as e:
            self._log.error(
                "dot_parse_error",
                pcap=str(pcap_path), error=repr(e),
            )
            raise FeatureExtractionError(
                f"failed to parse {pcap_path}: {e}"
            ) from e

        self._log.info(
            "dot_parse_complete",
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
        Extract the three DoT protocol-specific features:

            subdomain_entropy
            query_type_distribution_l1
            protocol_specific_feature

        The formulas are the same as the DoH adapter's (and the same
        as `data/preprocess_pcap.py`'s DoT branch). The DoT-vs-DoH
        distinction is carried by the metric tensor (β, γ vs λ, α) and
        by the DoT baseline, not by a different feature formula.
        See the module docstring for why.
        """
        data = flow.protocol_data or {}
        record_types: List[int] = list(data.get("tls_record_types", []))
        first_payload: Optional[bytes] = data.get("first_payload")

        # --- subdomain_entropy ---
        subdomain_entropy = 0.0
        if first_payload:
            sni = _extract_sni_from_client_hello(first_payload)
            if sni:
                subdomain_entropy = _shannon_entropy(sni)
            else:
                subdomain_entropy = _shannon_entropy_bytes(first_payload)

        # --- query_type_distribution_l1 ---
        if record_types:
            l1 = _l1_distance_uniform(record_types, _L1_N_CATEGORIES)
        else:
            l1 = 0.0

        # --- protocol_specific_feature ---
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
        Apply a sandbox action to a DoT flow.

        Only `reset_dot_tls` is valid for DoT (SANDBOX_ACTIONS["dot"]).
        The action terminates the TLS session, resets the transport
        context, and forces the next connection through the sandbox
        (thesis §5.6.4, p.133).

        This differs from the DoH adapter's `isolate_doh_stream` in
        two ways:

          1. A DoH stream can be terminated individually within a
             multiplexed HTTP/2 connection; a DoT session cannot,
             because there is one TLS session per TCP connection.
             The reset is therefore a *connection-level* action, not
             a *stream-level* action.

          2. The reference latency for `reset_dot_tls` is 1.6 ms
             (sandbox_mdp_params.yaml §6), slightly higher than the
             1.2 ms for `isolate_doh_stream`, because it must tear
             down the full TLS session.

        In toy mode (no real network namespace), the action is a
        bookkeeping operation and the measured latency is microseconds.
        In full mode against a real network namespace, the action
        includes the TCP RST and TLS close_notify and takes ~1.6 ms.
        """
        expected = SANDBOX_ACTIONS["dot"]
        if action != expected:
            raise SandboxActionError(
                f"DoT adapter cannot apply action {action!r}; "
                f"expected {expected!r}. A caller that passes a "
                f"protocol-mismatched action is a defect."
            )

        started = time.perf_counter()
        already_reset = bool(flow.protocol_data.get("sandbox_reset", False))
        flow.protocol_data["sandbox_reset"] = True
        flow.protocol_data["sandbox_action"] = action
        flow.protocol_data["sandbox_at_ms"] = time.time() * 1000.0
        elapsed_ms = (time.perf_counter() - started) * 1000.0

        success = not already_reset
        detail = (
            "TLS session reset; transport context cleared; subsequent "
            "connections routed through sandbox"
            if success else
            "TLS session was already reset; action is a no-op"
        )
        self._log.info(
            "dot_sandbox_action",
            flow_id=flow.flow_id,
            action=action,
            success=success,
            latency_ms=round(elapsed_ms, 4),
            already_reset=already_reset,
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
        Return the DoT metric-tensor kwargs: {x_bandwidth, rtt}.

        See the module docstring for the interpretation and its
        limitations.
        """
        x = np.asarray(x, dtype=np.float64)
        if x.shape != self._baseline.shape:
            raise FeatureExtractionError(
                f"DoT metric tensor requires a {self._baseline.shape} "
                f"feature vector; got shape {x.shape}"
            )
        # ||x||_Bandwidth: L2 norm of the two size features, normalised
        # by the baseline bandwidth.
        qs = float(x[2])   # query_size_mean
        rs = float(x[4])   # response_size_mean
        x_bandwidth = math.sqrt(qs * qs + rs * rs) / self._baseline_bandwidth

        # R_rtt: inter-arrival time mean, normalised by the baseline RTT.
        # This is the closest available proxy in the twelve-feature
        # vector; see the module docstring for the limitation.
        iat = float(x[0])
        rtt = iat / self._baseline_rtt

        return {
            "x_bandwidth": x_bandwidth,
            "rtt": rtt,
        }

    # -- 5. accumulator ----------------------------------------------------

    def _accumulate_dot_query(self, flow: FlowState, payload: bytes) -> None:
        """
        Accumulate the DoT-specific bits of a query packet.

        Same as DoH's _accumulate_doh_query: extract SNI (best-effort),
        retain first 512 bytes for entropy fallback, and append the
        TLS record type. The two are duplicated rather than shared
        because the src/ layer must not depend on the data/ layer for
        utility functions (data/ depends on src/, not vice versa), and
        the two adapters are conceptually independent.
        """
        if not payload:
            return
        data = flow.protocol_data
        if data.get("first_payload") is None:
            data["first_payload"] = payload[:_MAX_PAYLOAD_KEEP_BYTES]
        rt = _tls_record_type(payload)
        if rt is not None:
            data["tls_record_types"].append(rt)
            if rt == _TLS_APPLICATION_DATA:
                data["n_data_records"] = data.get("n_data_records", 0) + 1


# ---------------------------------------------------------------------------
# TLS parsing helpers (self-contained, same as doh_adapter.py)
# ---------------------------------------------------------------------------

def _tls_record_type(payload: bytes) -> Optional[int]:
    if len(payload) >= 1 and payload[0] in _TLS_RECORD_TYPES:
        return payload[0]
    return None


def _extract_sni_from_client_hello(payload: bytes) -> Optional[str]:
    """Best-effort SNI extraction (same as doh_adapter.py)."""
    if len(payload) < 5 or payload[0] != _TLS_HANDSHAKE:
        return None
    if len(payload) < 9 or payload[5] != 0x01:
        return None
    try:
        offset = 9 + 2 + 32
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
            if ext_type == 0x0000 and len(ext_data) >= 5:
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
    "DoTAdapter",
]


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def _self_test() -> int:
    import sys
    rc = 0
    print("=== dot_adapter self-test ===")

    import numpy as np

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
        "protocols": {"dot": {"tls_min_version": "1.3"}},
        "_includes": {"riemannian_metrics": metrics_cfg},
    }

    # 1. Construct
    adapter = DoTAdapter(config)
    if adapter._tls_min_version != "1.3":
        print(f"[FAIL] tls_min_version: {adapter._tls_min_version}")
        rc = 1
    else:
        print(f"[OK]   DoTAdapter constructed (tls_min_version=1.3)")

    # 2. Wrong TLS version rejected
    bad_config = {
        "protocols": {"dot": {"tls_min_version": "1.2"}},
        "_includes": {"riemannian_metrics": metrics_cfg},
    }
    try:
        DoTAdapter(bad_config)
    except FeatureExtractionError:
        print(f"[OK]   tls_min_version=1.2 rejected")
    else:
        print(f"[FAIL] tls_min_version=1.2 was accepted")
        rc = 1

    # 3. Extract features from a synthetic flow
    flow = FlowState(
        flow_id="test-dot-flow",
        protocol="DoT",
        client_ip="10.0.0.1", client_port=50000,
        server_ip="10.0.0.2", server_port=853,
    )
    for ts, size in [(0.0, 60), (0.08, 80), (0.16, 100)]:
        flow.add_query(ts, size)
    for ts, size in [(0.005, 120), (0.085, 140)]:
        flow.add_response(ts, size)
    flow.protocol_data = {
        "tls_record_types": [0x16, 0x17, 0x17, 0x17, 0x17],  # 1 handshake, 4 app_data
        "first_payload": b"\x16\x03\x03\x00\x05" + b"\x00" * 100,
        "n_data_records": 4,
    }
    feats = adapter.extract_protocol_specific_features(flow)
    if set(feats.keys()) != set(PROTOCOL_FEATURES):
        print(f"[FAIL] feature keys: {sorted(feats.keys())}")
        rc = 1
    elif abs(feats["protocol_specific_feature"] - 0.8) > 1e-9:
        # 4 app_data out of 5 records = 0.8
        print(f"[FAIL] protocol_specific_feature: {feats['protocol_specific_feature']}")
        rc = 1
    else:
        print(f"[OK]   extract_protocol_specific_features: "
              f"entropy={feats['subdomain_entropy']:.4f}, "
              f"l1={feats['query_type_distribution_l1']:.4f}, "
              f"app_data_fraction={feats['protocol_specific_feature']:.4f}")

    # 4. Full feature vector
    full = adapter.extract_features(flow)
    if list(full.keys()) != list(FEATURE_COLUMNS):
        print(f"[FAIL] extract_features key order")
        rc = 1
    else:
        print(f"[OK]   extract_features: 12 keys in canonical order")

    # 5. Metric tensor kwargs
    x = np.array(_DOT_BASELINE, dtype=np.float64)
    kwargs = adapter._metric_tensor_kwargs(x)
    if "x_bandwidth" not in kwargs or "rtt" not in kwargs:
        print(f"[FAIL] _metric_tensor_kwargs keys: {sorted(kwargs.keys())}")
        rc = 1
    elif abs(kwargs["x_bandwidth"] - 1.0) > 1e-9:
        print(f"[FAIL] x_bandwidth at baseline should be 1.0, "
              f"got {kwargs['x_bandwidth']}")
        rc = 1
    elif abs(kwargs["rtt"] - 1.0) > 1e-9:
        print(f"[FAIL] rtt at baseline should be 1.0, got {kwargs['rtt']}")
        rc = 1
    else:
        print(f"[OK]   _metric_tensor_kwargs: baseline gives (1.0, 1.0)")

    # 6. Metric tensor dispatch returns SPD
    g = adapter.get_metric_tensor(x)
    if g.shape != (len(FEATURE_COLUMNS), len(FEATURE_COLUMNS)):
        print(f"[FAIL] metric_tensor shape: {g.shape}")
        rc = 1
    elif np.linalg.eigvalsh(g).min() <= 0:
        print(f"[FAIL] metric_tensor not positive-definite")
        rc = 1
    else:
        print(f"[OK]   get_metric_tensor: SPD")

    # 7. Sandbox action
    result = adapter.apply_sandbox_action(flow, SANDBOX_ACTIONS["dot"])
    if not result.success or result.action != "reset_dot_tls":
        print(f"[FAIL] apply_sandbox_action: {result}")
        rc = 1
    elif result.protocol != "DoT":
        print(f"[FAIL] apply_sandbox_action protocol: {result.protocol}")
        rc = 1
    else:
        print(f"[OK]   apply_sandbox_action: success, reset_dot_tls")

    # Idempotent
    result2 = adapter.apply_sandbox_action(flow, SANDBOX_ACTIONS["dot"])
    if result2.success:
        print(f"[FAIL] second apply_sandbox_action should report already-reset")
        rc = 1
    else:
        print(f"[OK]   second apply_sandbox_action: already_reset")

    # Cross-protocol action rejected
    try:
        adapter.apply_sandbox_action(flow, "isolate_doh_stream")
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
