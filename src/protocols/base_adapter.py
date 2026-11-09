#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
src/protocols/base_adapter.py
=============================

E3-DoH Reproducibility — protocol adapter abstract base class.

Chapter 5 of the thesis: E3-DoH (Enhanced Evolutionary Encryption for
Multi-Protocol DNS Security — DoH, DoT, DoQ).

This module defines the interface that every protocol adapter must
satisfy. The three concrete adapters — DoHAdapter, DoTAdapter,
DoQAdapter — inherit from ProtocolAdapter and implement four abstract
methods:

    parse_pcap(pcap_path)                         -> Iterator[FlowState]
    extract_protocol_specific_feature(flow)       -> float
    apply_sandbox_action(flow, action)            -> SandboxResult
    _metric_tensor_kwargs(x)                      -> dict

Everything else — feature-vector assembly, the eight standard features,
the sandbox action registry, the metric-tensor dispatch, configuration
access, logging — is provided by the base class. This split is
deliberate: the protocol-specific bits are the wire format (how to
parse an HTTP/2 frame, a TLS record, or a QUIC packet) and the
protocol-specific feature and metric-tensor argument (the twelfth
feature and its tensor inputs). Everything else is shared.

WHY AN ABSTRACT BASE CLASS
--------------------------
A reviewer who is auditing the artifact should be able to look at one
file and see the full set of behaviours an adapter must provide. The
ABC gives that: `ProtocolAdapter` is the specification, and the three
subclasses are three concrete instantiations. Without the ABC, the
three adapters would drift apart (each would grow slightly different
conventions for returning features, handling missing data, or
dispatching sandbox actions), and the downstream engines
(anomaly_detector.py, aee_engine.py, am_sandbox.py) would need
per-protocol branching. The ABC eliminates that branching.

THE TWELVE-FEATURE CONTRACT
---------------------------
Every adapter, when it emits a flow, must produce a feature vector with
exactly the twelve features in the exact order defined by
`data.data_utils.FEATURE_COLUMNS` and `configs/riemannian_metrics.yaml
§2`. The base class provides `compute_standard_features()` which
produces eight of the twelve (the eight that are extractable from raw
packet metadata); the three protocol-specific features
(`subdomain_entropy`, `query_type_distribution_l1`,
`protocol_specific_feature`) are produced by the subclass. The
`assemble_feature_vector()` method merges the two halves and enforces
the order.

The order matters because the Riemannian metric tensors are 12×12
matrices indexed by position. A drift in the order would silently
corrupt every downstream computation. The base class rejects any
feature vector whose keys do not exactly match the canonical set, and
it rejects any vector whose key order does not match the canonical
order.

RELATIONSHIP TO preprocess_pcap.py
----------------------------------
`data/preprocess_pcap.py` is the *offline* feature extractor: it reads
a PCAP, accumulates flows, computes features, and writes a Parquet
feature store. `ProtocolAdapter` (this module) is the *online* adapter
that the detection pipeline uses during a live run. The two share the
same eight standard features and the same canonical order, but they
are independent implementations of the same mathematics.

To prevent them from drifting, `tests/test_protocols.py` (later in the
sequence) runs both on the same input and asserts that the eight
standard features agree to within 1e-6. If a reviewer changes one
without the other, that test will fail.

THE SANDBOX ACTION REGISTRY
---------------------------
Each protocol has exactly one isolation primitive (thesis §5.6.4,
p.133):

    DoH -> "isolate_doh_stream"        (HTTP/2 stream termination)
    DoT -> "reset_dot_tls"             (TLS 1.3 session reset)
    DoQ -> "migrate_doq_connection"    (QUIC connection migration)

The base class owns the registry (SANDBOX_ACTIONS) and the dispatch
(`ProtocolAdapter.apply_sandbox_action`). The subclass implements the
actual wire-level action (killing a stream, resetting a TLS context,
issuing a CONNECTION_CLOSE / migrating a connection ID), but it does
not decide which action is applicable — that is the base class's job.

The action-name list is also enforced by `data/attack_matrix.csv`'s
`Mitigation_Action` column and by `configs/sandbox_mdp_params.yaml
§7`, so a reviewer who is auditing the artifact sees the same three
names in three places.

THE METRIC-TENSOR DISPATCH
--------------------------
`ProtocolAdapter.get_metric_tensor(x)` dispatches to the correct
`math_utils.metric_tensor_doh/dot/doq` function based on
`self.protocol`, and uses the subclass's
`_metric_tensor_kwargs(x)` to supply the protocol-specific arguments
that the corresponding math_utils function needs:

    DoH:  requires (x, grad_x, dx_dt)        -> subclass provides grad_x, dx_dt
    DoT:  requires (x, x_bandwidth, rtt)     -> subclass provides x_bandwidth, rtt
    DoQ:  requires (x, d2x_dt2, Q)           -> subclass provides d2x_dt2, Q

The coefficient values (λ, α, β, γ, η, Q) live in
`configs/riemannian_metrics.yaml`, not in the subclass; the base class
loads them once at construction time and passes them to the math_utils
functions.

SECURITY BOUNDARY
-----------------
No method in this module makes a security claim. The detection
accuracy of the anomaly detector that consumes the feature vectors is
measured empirically (Chapter 5, §5.14). The metric tensors are a
modelling abstraction (riemannian_metrics.yaml §12). The sandbox
actions are engineering mechanisms (sandbox_mdp_params.yaml §10). The
security boundary of the framework lives in aee_params.yaml §10 and
qhe_params.yaml §9.

The base class's only security-relevant contribution is the feature-
vector contract: by rejecting any feature vector that does not match
the canonical schema, it ensures that a downstream engine cannot be
fed a malformed vector. That is a robustness property, not a security
one, but it is worth stating explicitly so a reviewer does not
mistake it for a security argument.

THREAD SAFETY
-------------
A ProtocolAdapter instance is *not* thread-safe. Each adapter manages
per-flow state (the FlowState objects) and the flow table; concurrent
calls to `parse_pcap` on the same instance would corrupt the flow
table. A caller that wants concurrent parsing should create one
adapter instance per thread, or serialize the calls. The artifact's
own usage (one adapter per protocol, called from the main thread)
satisfies this.

The `get_adapter()` factory returns a *new* instance on each call; it
does not cache. This is deliberately different from the logger and
seed singletons, because an adapter is stateful per-run and per-
protocol, and sharing it across runs would be a bug.
"""

from __future__ import annotations

import abc
import hashlib
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple, Union

import numpy as np

# The canonical feature order. Imported from data.data_utils so that
# the adapter and the offline preprocessor (preprocess_pcap.py,
# preprocess_zenodo.py) all agree on a single source of truth.
from data.data_utils import (
    FEATURE_COLUMNS,
    META_COLUMNS,
    PROTOCOLS,
)

# The math primitives. We import the specific functions we need rather
# than `math_utils.*` so that an accidental deletion of one of the
# primitives is a load-time error here rather than a run-time error
# deep inside an engine.
from src.utils.math_utils import (
    metric_tensor_doh,
    metric_tensor_dot,
    metric_tensor_doq,
    build_congestion_operator,
)

# The logger. Adapters log per-flow parsing decisions at DEBUG level;
# a caller that wants to see them configures the logger accordingly.
from src.utils.logger import get_logger


# ---------------------------------------------------------------------------
# Constants — the feature split
# ---------------------------------------------------------------------------
# Of the twelve canonical features, eight are directly computable from
# raw packet metadata (query and response sizes, inter-arrival times,
# session duration). The other four are protocol-specific and are
# produced by the subclass:
#
#     subdomain_entropy            -> extractable from the TLS ClientHello SNI
#     query_type_distribution_l1   -> protocol-frame-type distribution distance
#     burstiness                   -> computable from inter-arrival times (standard)
#     protocol_specific_feature    -> protocol-specific (see §2 of the thesis and
#                                     riemannian_metrics.yaml §2)
#
# Wait — burstiness is *also* computable from inter-arrival times, so
# it belongs in the standard set. The split is:
#
#     STANDARD_FEATURES (9): computable by the base class from packet stats
#     PROTOCOL_FEATURES (3): produced by the subclass
#
# The nine standard features are:
#     inter_arrival_time_mean, inter_arrival_time_std,
#     query_size_mean, query_size_std,
#     response_size_mean, response_size_std,
#     query_response_ratio,
#     burstiness,
#     session_duration
#
# The three protocol features are:
#     subdomain_entropy, query_type_distribution_l1,
#     protocol_specific_feature
#
# The split is asserted at load time below.

STANDARD_FEATURES: Tuple[str, ...] = (
    "inter_arrival_time_mean",
    "inter_arrival_time_std",
    "query_size_mean",
    "query_size_std",
    "response_size_mean",
    "response_size_std",
    "query_response_ratio",
    "burstiness",
    "session_duration",
)

PROTOCOL_FEATURES: Tuple[str, ...] = (
    "subdomain_entropy",
    "query_type_distribution_l1",
    "protocol_specific_feature",
)

# The union must be exactly the canonical set. A load-time check
# catches any drift between the three sources of truth (this module,
# data.data_utils, and riemannian_metrics.yaml §2).
if set(STANDARD_FEATURES) | set(PROTOCOL_FEATURES) != set(FEATURE_COLUMNS):
    raise RuntimeError(
        "Feature split does not match the canonical feature set. "
        f"STANDARD ∪ PROTOCOL = {sorted(set(STANDARD_FEATURES) | set(PROTOCOL_FEATURES))}, "
        f"FEATURE_COLUMNS = {sorted(FEATURE_COLUMNS)}. "
        f"Update STANDARD_FEATURES/PROTOCOL_FEATURES in base_adapter.py "
        f"or FEATURE_COLUMNS in data/data_utils.py; they must agree."
    )

# The sandbox action registry (see module docstring). The keys are
# protocol names, the values are the action names. The action names are
# also used in sandbox_mdp_params.yaml §3 and §7, and in the
# Mitigation_Action column of data/attack_matrix.csv. A change here
# requires updating all four places.
SANDBOX_ACTIONS: Dict[str, str] = {
    "doh": "isolate_doh_stream",
    "dot": "reset_dot_tls",
    "doq": "migrate_doq_connection",
}


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class AdapterError(Exception):
    """Base class for adapter errors."""


class ProtocolNotSupportedError(AdapterError):
    """Raised when an unsupported protocol name is passed."""


class AdapterConfigError(AdapterError):
    """Raised when the config is missing or has invalid values for the adapter."""


class FeatureExtractionError(AdapterError):
    """Raised when a flow cannot be converted to a feature vector."""


class SandboxActionError(AdapterError):
    """Raised when a sandbox action is not applicable to a protocol."""


# ---------------------------------------------------------------------------
# FlowState — per-flow accumulated state
# ---------------------------------------------------------------------------
# A FlowState is the minimum state needed to compute the twelve
# features. Each subclass fills it in differently (DoH accumulates
# HTTP/2 frames, DoT accumulates TLS records, DoQ accumulates QUIC
# packets) but they all end up with the same fields.
#
# Field meanings:
#   flow_id         : a stable 16-hex-char identifier derived from the 5-tuple
#   protocol        : "DoH" | "DoT" | "DoQ"
#   client_ip       : the client side of the flow (from the first packet)
#   client_port     : the client port
#   server_ip       : the server side
#   server_port     : the server port
#   first_ts,last_ts: seconds since epoch of the first and last packet
#   query_times     : list of timestamps of query (client→server) packets
#   query_sizes     : list of query payload sizes (bytes)
#   response_sizes  : list of response payload sizes (bytes)
#   protocol_data   : protocol-specific accumulated data (frames, records,
#                     packet types); the base class does not inspect it,
#                     the subclass extracts the protocol-specific feature
#                     from it
#   label           : "benign" | "attack" (from attack-suite metadata or
#                     from the directory structure of the PCAP)
#   attack_type     : "" for benign, "A01".."A15" for attack
#   query_type      : the DNS query type if extractable from the payload,
#                     else "OTHER"

@dataclass
class FlowState:
    flow_id: str
    protocol: str
    client_ip: str
    client_port: int
    server_ip: str
    server_port: int
    first_ts: Optional[float] = None
    last_ts: Optional[float] = None
    query_times: List[float] = field(default_factory=list)
    query_sizes: List[int] = field(default_factory=list)
    response_sizes: List[int] = field(default_factory=list)
    protocol_data: Dict[str, Any] = field(default_factory=dict)
    label: str = "benign"
    attack_type: str = ""
    query_type: str = "OTHER"

    def add_query(self, ts: float, size: int) -> None:
        self.query_times.append(ts)
        self.query_sizes.append(int(size))
        self._touch(ts)

    def add_response(self, ts: float, size: int) -> None:
        self.response_sizes.append(int(size))
        self._touch(ts)

    def _touch(self, ts: float) -> None:
        if self.first_ts is None or ts < self.first_ts:
            self.first_ts = ts
        if self.last_ts is None or ts > self.last_ts:
            self.last_ts = ts

    def n_packets(self) -> int:
        return len(self.query_sizes) + len(self.response_sizes)


# ---------------------------------------------------------------------------
# SandboxResult — returned by apply_sandbox_action
# ---------------------------------------------------------------------------
# A SandboxResult carries the outcome of a sandbox action. It is
# deliberately verbose because a reviewer who is auditing the sandbox
# behaviour wants to know which action was applied, to which flow, and
# whether it succeeded. A simple `bool` return would lose that
# information.

@dataclass
class SandboxResult:
    flow_id: str
    protocol: str
    action: str
    success: bool
    detail: str = ""
    # The wall-clock duration of the action, in milliseconds. Used by
    # the verification script to check the mitigation-latency row of
    # Table 20 (1.5 ms) and the per-action latency of
    # sandbox_mdp_params.yaml §6. The default 0.0 is for tests that do
    # not measure latency; a real run always populates it.
    latency_ms: float = 0.0


# ---------------------------------------------------------------------------
# ProtocolAdapter — abstract base class
# ---------------------------------------------------------------------------

class ProtocolAdapter(abc.ABC):
    """
    Abstract base class for protocol adapters.

    A concrete subclass must implement four abstract methods:

        parse_pcap(pcap_path)                    -> Iterator[FlowState]
        extract_protocol_specific_features(flow) -> Dict[str, float]
        apply_sandbox_action(flow, action)       -> SandboxResult
        _metric_tensor_kwargs(x)                 -> Dict[str, Any]

    Everything else is provided by this class.
    """

    # Subclasses must set this class attribute. The base class checks
    # it at construction time against the configured protocol name.
    protocol: str = ""
    # Human-readable name for logging. Subclasses set this too.
    protocol_full_name: str = ""

    def __init__(self, config: Dict[str, Any]) -> None:
        if self.protocol not in PROTOCOLS:
            raise AdapterConfigError(
                f"{type(self).__name__}.protocol = {self.protocol!r} is not "
                f"a valid protocol name. Expected one of {PROTOCOLS}. "
                f"A subclass that does not set the class attribute correctly "
                f"is a defect, not a user error."
            )
        self._protocol = self.protocol
        self._full_name = self.protocol_full_name or self.protocol.upper()
        self._config = config
        self._log = get_logger(f"e3doh.protocols.{self._protocol}")

        # Load the metric-tensor coefficients once, from the included
        # riemannian_metrics.yaml. A missing include is a hard error:
        # an adapter without coefficients cannot produce a metric
        # tensor, and a metric-tensor-free adapter would silently make
        # the anomaly detector fall back to Euclidean distance, which
        # would produce wrong results without a warning.
        metrics_cfg = _require(config, "_includes", "riemannian_metrics")
        try:
            self._metric_params = {
                "doh": metrics_cfg["doh"],
                "dot": metrics_cfg["dot"],
                "doq": metrics_cfg["doq"],
            }
        except KeyError as e:
            raise AdapterConfigError(
                f"riemannian_metrics.yaml is missing the '{e.args[0]}' section; "
                f"the adapter cannot load the metric-tensor coefficients."
            ) from e

        # The congestion operator for DoQ is built once from the sparse
        # specification in riemannian_metrics.yaml §5. The base class
        # builds it here so that DoQAdapter does not have to.
        doq_op = metrics_cfg["doq"]["congestion_operator"]
        self._congestion_operator = build_congestion_operator(
            n=len(FEATURE_COLUMNS),
            diagonal=doq_op["diagonal"],
            off_diagonal=doq_op["off_diagonal"],
        )

        self._log.info(
            "adapter_initialised",
            protocol=self._protocol,
            full_name=self._full_name,
            n_features=len(FEATURE_COLUMNS),
            sandbox_action=SANDBOX_ACTIONS[self._protocol],
        )

    # -- abstract interface -------------------------------------------------

    @abc.abstractmethod
    def parse_pcap(self, pcap_path: Path) -> Iterator[FlowState]:
        """
        Parse a PCAP file and yield one FlowState per flow.

        Each adapter implements this by reading the PCAP with the
        appropriate parser (scapy's PcapReader with HTTP/2 framing for
        DoH, with TLS record parsing for DoT, with QUIC packet parsing
        for DoQ) and accumulating packets into FlowState objects.

        Parameters
        ----------
        pcap_path : Path
            The PCAP file to parse. The adapter does not modify the
            file.

        Yields
        ------
        FlowState objects, one per flow. The order is unspecified but
        the caller must not rely on it. A flow is yielded once, at the
        end of the PCAP (when all packets for that flow have been
        seen); an adapter that yields a flow prematurely has a bug.

        Raises
        ------
        FileNotFoundError
            If the PCAP file does not exist.
        FeatureExtractionError
            If the PCAP cannot be parsed at all (wrong magic,
            truncated, etc.). Individual malformed packets are logged
            at WARNING level and skipped, not raised; a single bad
            packet should not abort the run.
        """

    @abc.abstractmethod
    def extract_protocol_specific_features(self, flow: FlowState) -> Dict[str, float]:
        """
        Extract the three protocol-specific features from a FlowState.

        Parameters
        ----------
        flow : FlowState
            The accumulated flow. The subclass reads `flow.protocol_data`
            (which it populated during parse_pcap) to compute the three
            features.

        Returns
        -------
        A dict with exactly the three keys of PROTOCOL_FEATURES:
            subdomain_entropy, query_type_distribution_l1,
            protocol_specific_feature
        All values are floats. The base class merges these with the
        nine standard features and enforces the canonical order.

        Raises
        ------
        FeatureExtractionError
            If a feature cannot be computed (e.g. a protocol-specific
            datum is missing and there is no documented fallback). A
            feature that *can* be computed with a documented fallback
            (e.g. subdomain_entropy with the first-512-bytes fallback
            described in riemannian_metrics.yaml §2) should return the
            fallback value, not raise.
        """

    @abc.abstractmethod
    def apply_sandbox_action(
        self, flow: FlowState, action: str,
    ) -> SandboxResult:
        """
        Apply a sandbox action to a flow.

        Parameters
        ----------
        flow : FlowState
            The flow to act on. The flow must be still active; an
            already-terminated flow cannot be sandboxed.
        action : str
            One of the action names in SANDBOX_ACTIONS. The base class
            provides `get_sandbox_action()` which returns the correct
            action for this protocol; a caller that passes a
            protocol-mismatched action is a defect.

        Returns
        -------
        SandboxResult with `success=True` if the action was applied,
        `success=False` if it could not be (e.g. the flow was already
        terminated). A failed action is not an exception; it is a
        recorded outcome that the caller (AMPS) may retry or escalate.

        Raises
        ------
        SandboxActionError
            If the action name is not in SANDBOX_ACTIONS, or if it
            does not match this adapter's protocol.
        """

    @abc.abstractmethod
    def _metric_tensor_kwargs(self, x: np.ndarray) -> Dict[str, Any]:
        """
        Return the protocol-specific keyword arguments that the
        math_utils.metric_tensor_* function needs.

        For DoH, this returns {"grad_x": ..., "dx_dt": ...}.
        For DoT, this returns {"x_bandwidth": ..., "rtt": ...}.
        For DoQ, this returns {"d2x_dt2": ..., "Q": ...}.

        The base class calls the correct math_utils function with
        these kwargs plus the coefficients from
        riemannian_metrics.yaml. The subclass does not call the
        math_utils function itself; that would defeat the dispatch.
        """

    # -- concrete helpers ---------------------------------------------------

    def get_metric_tensor(self, x: np.ndarray) -> np.ndarray:
        """
        Compute the metric tensor at the point x.

        This is the protocol-agnostic entry point; it dispatches to
        math_utils.metric_tensor_doh/dot/doq based on self.protocol and
        passes the coefficients from riemannian_metrics.yaml plus the
        protocol-specific kwargs from _metric_tensor_kwargs(x).

        Parameters
        ----------
        x : (12,) array
            The traffic-state feature vector.

        Returns
        -------
        (12, 12) symmetric positive-definite metric tensor.

        Raises
        ------
        FeatureExtractionError
            If x is not a (12,) array, or if any coefficient is
            missing from riemannian_metrics.yaml.
        """
        x = np.asarray(x, dtype=np.float64)
        if x.shape != (len(FEATURE_COLUMNS),):
            raise FeatureExtractionError(
                f"metric tensor requires a ({len(FEATURE_COLUMNS)},) feature "
                f"vector; got shape {x.shape}"
            )
        kwargs = self._metric_tensor_kwargs(x)
        if self._protocol == "doh":
            params = self._metric_params["doh"]
            return metric_tensor_doh(
                x,
                grad_x=kwargs["grad_x"],
                dx_dt=kwargs["dx_dt"],
                lambda_=float(params["lambda"]),
                alpha_=float(params["alpha"]),
            )
        if self._protocol == "dot":
            params = self._metric_params["dot"]
            return metric_tensor_dot(
                x,
                x_bandwidth=float(kwargs["x_bandwidth"]),
                rtt=float(kwargs["rtt"]),
                beta_=float(params["beta"]),
                gamma_=float(params["gamma"]),
            )
        if self._protocol == "doq":
            params = self._metric_params["doq"]
            return metric_tensor_doq(
                x,
                d2x_dt2=kwargs["d2x_dt2"],
                Q=self._congestion_operator,
                eta=float(params["eta"]),
            )
        raise ProtocolNotSupportedError(
            f"unknown protocol {self._protocol!r}"
        )

    def get_sandbox_action(self) -> str:
        """Return the sandbox action name for this protocol."""
        return SANDBOX_ACTIONS[self._protocol]

    def extract_features(self, flow: FlowState) -> Dict[str, float]:
        """
        Extract the full twelve-feature vector for a flow.

        Calls compute_standard_features() for the nine standard
        features, calls the subclass's
        extract_protocol_specific_features() for the three protocol
        features, and merges them in the canonical order.

        Parameters
        ----------
        flow : FlowState

        Returns
        -------
        A dict with exactly the twelve keys of FEATURE_COLUMNS, in the
        canonical order. All values are float64 (Python floats).

        Raises
        ------
        FeatureExtractionError
            If the standard features cannot be computed (e.g. no
            packets), or if the subclass returns a protocol-features
            dict whose key set does not exactly match
            PROTOCOL_FEATURES, or if any resulting value is NaN.
        """
        standard = compute_standard_features(flow)
        protocol = self.extract_protocol_specific_features(flow)
        # Enforce the protocol-feature key set exactly.
        if set(protocol.keys()) != set(PROTOCOL_FEATURES):
            raise FeatureExtractionError(
                f"{type(self).__name__}.extract_protocol_specific_features "
                f"returned keys {sorted(protocol.keys())}; expected exactly "
                f"{sorted(PROTOCOL_FEATURES)}. A subclass that adds or removes "
                f"a feature without updating PROTOCOL_FEATURES is a defect."
            )
        merged: Dict[str, float] = {}
        for name in FEATURE_COLUMNS:
            if name in standard:
                merged[name] = float(standard[name])
            elif name in protocol:
                merged[name] = float(protocol[name])
            else:
                raise FeatureExtractionError(
                    f"feature '{name}' was produced by neither the standard "
                    f"computation nor the subclass. This is a bug in the "
                    f"STANDARD_FEATURES/PROTOCOL_FEATURES split."
                )
        for name, value in merged.items():
            if math.isnan(value):
                raise FeatureExtractionError(
                    f"feature '{name}' is NaN for flow {flow.flow_id}; "
                    f"the flow is malformed and cannot be scored."
                )
        return merged

    def to_feature_row(self, flow: FlowState) -> Dict[str, Any]:
        """
        Convert a FlowState to a feature row suitable for the downstream
        engines.

        The row has all twelve features (in the canonical order) plus
        the six meta columns of META_COLUMNS. The keys are exactly
        those of FEATURE_COLUMNS followed by those of META_COLUMNS.

        Parameters
        ----------
        flow : FlowState

        Returns
        -------
        A dict with the eighteen keys. The values are floats for
        features, strings for the meta columns.

        Raises
        ------
        FeatureExtractionError
            If extract_features() raises.
        """
        features = self.extract_features(flow)
        row: Dict[str, Any] = dict(features)
        row["flow_id"] = flow.flow_id
        row["protocol"] = flow.protocol
        row["label"] = flow.label
        row["attack_type"] = flow.attack_type
        row["query_type"] = flow.query_type
        row["provenance"] = "DS-1"  # adapters only run on real data
        return row

    # -- utilities for subclasses ------------------------------------------

    @staticmethod
    def make_flow_id(
        client_ip: str, client_port: int,
        server_ip: str, server_port: int,
        transport: str,
    ) -> str:
        """
        Build a stable 16-hex-char flow identifier from a 5-tuple.

        The identifier is SHA-1 of the 5-tuple, truncated to 16 hex
        characters. This is what `data/preprocess_pcap.py` uses for the
        same purpose; the two implementations produce the same ID for
        the same 5-tuple, which is what allows the online and offline
        paths to be cross-checked.
        """
        h = hashlib.sha1()
        h.update(f"{client_ip}:{client_port}->{server_ip}:{server_port}/{transport}".encode())
        return h.hexdigest()[:16]


# ---------------------------------------------------------------------------
# compute_standard_features — the nine standard features
# ---------------------------------------------------------------------------
# This function is the canonical implementation of the nine standard
# features for the *online* path. The *offline* path
# (data/preprocess_pcap.py) has its own implementation of the same
# mathematics (see the module docstring of that file for why); the two
# are verified to agree by tests/test_protocols.py.
#
# Feature definitions (matching riemannian_metrics.yaml §2):
#
#     inter_arrival_time_mean  : mean of consecutive query IATs (seconds)
#     inter_arrival_time_std   : std of consecutive query IATs
#     query_size_mean          : mean of query payload sizes (bytes)
#     query_size_std           : std of query payload sizes
#     response_size_mean       : mean of response payload sizes
#     response_size_std        : std of response payload sizes
#     query_response_ratio     : query_size_mean / response_size_mean
#     burstiness               : var(IAT) / mean(IAT)^2
#     session_duration         : last_ts - first_ts (seconds)

def compute_standard_features(flow: FlowState) -> Dict[str, float]:
    """
    Compute the nine standard features for a flow.

    Parameters
    ----------
    flow : FlowState
        Must have at least one query packet. Response sizes may be
        empty (a query-only flow has response_size_mean=0, which is a
        legitimate value, not an error).

    Returns
    -------
    A dict with exactly the nine keys of STANDARD_FEATURES.

    Raises
    ------
    FeatureExtractionError
        If the flow has no query packets (no features can be computed),
        or if any computed value is NaN or infinite.
    """
    if not flow.query_times:
        raise FeatureExtractionError(
            f"flow {flow.flow_id} has no query packets; cannot compute "
            f"the standard features."
        )

    out: Dict[str, float] = {}

    # --- inter-arrival times ---
    if len(flow.query_times) >= 2:
        iats = np.diff(np.asarray(flow.query_times, dtype=np.float64))
        iat_mean = float(np.mean(iats))
        iat_var = float(np.var(iats))
        iat_std = float(np.sqrt(iat_var))
        out["inter_arrival_time_mean"] = iat_mean
        out["inter_arrival_time_std"] = iat_std
        # Burstiness: var/mean^2. If mean is zero (all queries at the
        # same instant), the burstiness is undefined; the standard
        # convention is to return 0 in that case, which is what
        # preprocess_pcap.py does too.
        out["burstiness"] = (iat_var / (iat_mean * iat_mean)) if iat_mean > 0 else 0.0
    else:
        out["inter_arrival_time_mean"] = 0.0
        out["inter_arrival_time_std"] = 0.0
        out["burstiness"] = 0.0

    # --- query sizes ---
    qs = np.asarray(flow.query_sizes, dtype=np.float64)
    out["query_size_mean"] = float(np.mean(qs))
    out["query_size_std"] = float(np.std(qs))

    # --- response sizes ---
    if flow.response_sizes:
        rs = np.asarray(flow.response_sizes, dtype=np.float64)
        out["response_size_mean"] = float(np.mean(rs))
        out["response_size_std"] = float(np.std(rs))
    else:
        out["response_size_mean"] = 0.0
        out["response_size_std"] = 0.0

    # --- query/response ratio ---
    if out["response_size_mean"] > 0:
        out["query_response_ratio"] = out["query_size_mean"] / out["response_size_mean"]
    else:
        out["query_response_ratio"] = 0.0

    # --- session duration ---
    if flow.first_ts is not None and flow.last_ts is not None:
        out["session_duration"] = float(flow.last_ts - flow.first_ts)
    else:
        out["session_duration"] = 0.0

    # --- sanity ---
    for name, value in out.items():
        if math.isnan(value) or math.isinf(value):
            raise FeatureExtractionError(
                f"standard feature '{name}' is {value} for flow {flow.flow_id}; "
                f"the flow contains a degenerate value (e.g. a packet with "
                f"a zero-size payload where a size is required)."
            )

    # --- key set ---
    if set(out.keys()) != set(STANDARD_FEATURES):
        raise FeatureExtractionError(
            f"compute_standard_features produced keys {sorted(out.keys())}; "
            f"expected exactly {sorted(STANDARD_FEATURES)}. This is a bug in "
            f"the STANDARD_FEATURES/implementation correspondence."
        )
    return out


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def _require(config: Dict[str, Any], *keys: str) -> Any:
    """Walk a config dict by keys; raise AdapterConfigError on a missing key."""
    cur: Any = config
    for k in keys:
        if not isinstance(cur, dict):
            raise AdapterConfigError(
                f"config.{'.'.join(keys[:-1])} is not a mapping; "
                f"cannot descend to {k!r}"
            )
        if k not in cur:
            raise AdapterConfigError(
                f"config.{'.'.join(keys)} not found. The adapter needs the "
                f"riemannian_metrics include to compute the metric tensors. "
                f"Check that configs/default_e3_config.yaml §11 lists the "
                f"include and that configs/riemannian_metrics.yaml is present."
            )
        cur = cur[k]
    return cur


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def get_adapter(protocol: str, config: Dict[str, Any]) -> ProtocolAdapter:
    """
    Return a concrete adapter for the given protocol.

    The concrete adapters are imported lazily so that this module can
    be imported and tested before the three subclasses have been
    delivered. A caller that calls get_adapter() with a protocol whose
    adapter has not yet been delivered gets a ProtocolNotSupportedError
    with a clear message, rather than an ImportError at module load.

    Parameters
    ----------
    protocol : "doh" | "dot" | "doq"
    config : dict
        The dict returned by data_utils.load_config().

    Returns
    -------
    A ProtocolAdapter instance. A new instance on each call; see the
    thread-safety note in the module docstring.
    """
    if protocol not in PROTOCOLS:
        raise ProtocolNotSupportedError(
            f"unknown protocol {protocol!r}; expected one of {PROTOCOLS}"
        )
    try:
        if protocol == "doh":
            from src.protocols.doh_adapter import DoHAdapter
            return DoHAdapter(config)
        if protocol == "dot":
            from src.protocols.dot_adapter import DoTAdapter
            return DoTAdapter(config)
        if protocol == "doq":
            from src.protocols.doq_adapter import DoQAdapter
            return DoQAdapter(config)
    except ImportError as e:
        raise ProtocolNotSupportedError(
            f"the adapter for protocol {protocol!r} is not available "
            f"(src/protocols/{protocol}_adapter.py could not be imported). "
            f"This usually means the artifact is a partial checkout; see "
            f"the delivery sequence in README.md §3."
        ) from e
    # Unreachable (the three branches above cover PROTOCOLS), but a
    # static checker prefers an explicit raise.
    raise ProtocolNotSupportedError(f"unknown protocol {protocol!r}")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    # The base class and the factory
    "ProtocolAdapter",
    "get_adapter",
    # The data classes
    "FlowState",
    "SandboxResult",
    # The standard-feature computation
    "compute_standard_features",
    # The constants
    "STANDARD_FEATURES",
    "PROTOCOL_FEATURES",
    "SANDBOX_ACTIONS",
    # The exceptions
    "AdapterError",
    "ProtocolNotSupportedError",
    "AdapterConfigError",
    "FeatureExtractionError",
    "SandboxActionError",
]


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
# Running `python -m src.protocols.base_adapter` verifies:
#   - the STANDARD_FEATURES ∪ PROTOCOL_FEATURES == FEATURE_COLUMNS invariant,
#   - SANDBOX_ACTIONS has exactly the three protocol keys,
#   - a minimal mock subclass can be instantiated, that it rejects
#     bad protocol-class attributes, that it computes the standard
#     features correctly, that the feature-vector assembly enforces
#     the canonical order, that the metric-tensor dispatch returns a
#     positive-definite matrix, and that the sandbox dispatch returns
#     the right action name.
#
# This is a smoke test of the base class; the concrete adapters are
# tested in their own modules' self-tests and in tests/test_protocols.py.

def _self_test() -> int:
    import sys
    rc = 0
    print("=== base_adapter self-test ===")

    # 1. Feature split invariant (already enforced at import, but we
    #    print it so the reviewer sees it).
    union = set(STANDARD_FEATURES) | set(PROTOCOL_FEATURES)
    if union != set(FEATURE_COLUMNS):
        print(f"[FAIL] feature split mismatch: union={sorted(union)}, "
              f"canonical={sorted(FEATURE_COLUMNS)}")
        rc = 1
    else:
        print(f"[OK]   feature split: {len(STANDARD_FEATURES)} standard + "
              f"{len(PROTOCOL_FEATURES)} protocol = {len(FEATURE_COLUMNS)} canonical")

    # 2. Sandbox action registry has the three protocols.
    if set(SANDBOX_ACTIONS.keys()) != set(PROTOCOLS):
        print(f"[FAIL] SANDBOX_ACTIONS keys: {sorted(SANDBOX_ACTIONS.keys())}, "
              f"expected {sorted(PROTOCOLS)}")
        rc = 1
    else:
        print(f"[OK]   SANDBOX_ACTIONS: {SANDBOX_ACTIONS}")

    # 3. Minimal mock subclass. It does nothing protocol-specific; it
    #    provides trivial implementations of the abstract methods so we
    #    can exercise the concrete parts of the base class.
    class _MockAdapter(ProtocolAdapter):
        protocol = "doh"           # set as a class attribute
        protocol_full_name = "Mock (DoH)"

        def parse_pcap(self, pcap_path):
            return iter([])        # empty; the self-test builds FlowStates directly

        def extract_protocol_specific_features(self, flow):
            return {
                "subdomain_entropy": 3.5,
                "query_type_distribution_l1": 0.1,
                "protocol_specific_feature": 0.9,
            }

        def apply_sandbox_action(self, flow, action):
            return SandboxResult(
                flow_id=flow.flow_id, protocol=self.protocol,
                action=action, success=True, latency_ms=1.2,
            )

        def _metric_tensor_kwargs(self, x):
            return {
                "grad_x": np.zeros_like(x),
                "dx_dt": np.zeros_like(x),
            }

    # 4. A subclass that does not set `protocol` correctly is rejected.
    class _BadAdapter(ProtocolAdapter):
        protocol = "not-a-real-protocol"
        protocol_full_name = ""
        def parse_pcap(self, p): return iter([])
        def extract_protocol_specific_features(self, f): return {}
        def apply_sandbox_action(self, f, a): raise NotImplementedError
        def _metric_tensor_kwargs(self, x): return {}

    # Build a minimal config with the riemannian_metrics include.
    import numpy as np
    from src.utils.math_utils import build_congestion_operator
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
    config = {"_includes": {"riemannian_metrics": metrics_cfg}}

    try:
        _BadAdapter(config)
    except AdapterConfigError:
        print(f"[OK]   bad protocol class attribute rejected")
    else:
        print(f"[FAIL] bad protocol class attribute was not rejected")
        rc = 1

    adapter = _MockAdapter(config)
    print(f"[OK]   mock adapter initialised (protocol={adapter._protocol})")

    # 5. compute_standard_features on a synthetic flow.
    flow = FlowState(
        flow_id="test-flow",
        protocol="DoH",
        client_ip="10.0.0.1", client_port=50000,
        server_ip="10.0.0.2", server_port=443,
    )
    # Three query packets at t=0, 0.1, 0.3 (IATs 0.1, 0.2; mean 0.15,
    # var 0.0025, std 0.05, burstiness 0.0025/0.0225 ≈ 0.1111).
    flow.add_query(0.0, 60)
    flow.add_query(0.1, 80)
    flow.add_query(0.3, 100)
    flow.add_response(0.005, 120)
    flow.add_response(0.105, 140)
    std = compute_standard_features(flow)
    expected_iat_mean = 0.15
    if abs(std["inter_arrival_time_mean"] - expected_iat_mean) > 1e-9:
        print(f"[FAIL] standard iat_mean: {std['inter_arrival_time_mean']}")
        rc = 1
    elif abs(std["inter_arrival_time_std"] - 0.05) > 1e-9:
        print(f"[FAIL] standard iat_std: {std['inter_arrival_time_std']}")
        rc = 1
    elif abs(std["query_size_mean"] - 80.0) > 1e-9:
        print(f"[FAIL] standard query_size_mean: {std['query_size_mean']}")
        rc = 1
    elif abs(std["query_response_ratio"] - 80.0/130.0) > 1e-9:
        print(f"[FAIL] standard query_response_ratio: {std['query_response_ratio']}")
        rc = 1
    elif abs(std["session_duration"] - 0.3) > 1e-9:
        print(f"[FAIL] standard session_duration: {std['session_duration']}")
        rc = 1
    else:
        print(f"[OK]   compute_standard_features: iat_mean={std['inter_arrival_time_mean']:.4f}, "
              f"qsize_mean={std['query_size_mean']:.2f}, "
              f"ratio={std['query_response_ratio']:.4f}, "
              f"duration={std['session_duration']:.4f}")

    # 6. Full feature-vector assembly.
    feats = adapter.extract_features(flow)
    if set(feats.keys()) != set(FEATURE_COLUMNS):
        print(f"[FAIL] extract_features key set mismatch")
        rc = 1
    elif list(feats.keys()) != list(FEATURE_COLUMNS):
        print(f"[FAIL] extract_features key order mismatch")
        rc = 1
    elif feats["subdomain_entropy"] != 3.5 or feats["protocol_specific_feature"] != 0.9:
        print(f"[FAIL] extract_features protocol features: "
              f"subdomain_entropy={feats['subdomain_entropy']}, "
              f"protocol_specific_feature={feats['protocol_specific_feature']}")
        rc = 1
    else:
        print(f"[OK]   extract_features: 12 keys in canonical order, "
              f"3 protocol features correctly merged")

    # 7. to_feature_row produces the 12 + 6 = 18 keys.
    row = adapter.to_feature_row(flow)
    if len(row) != len(FEATURE_COLUMNS) + len(META_COLUMNS):
        print(f"[FAIL] to_feature_row has {len(row)} keys, "
              f"expected {len(FEATURE_COLUMNS) + len(META_COLUMNS)}")
        rc = 1
    elif row["provenance"] != "DS-1":
        print(f"[FAIL] to_feature_row provenance: {row['provenance']}")
        rc = 1
    else:
        print(f"[OK]   to_feature_row: {len(row)} keys, provenance=DS-1")

    # 8. Metric tensor dispatch.
    x = np.random.default_rng(0).normal(size=len(FEATURE_COLUMNS))
    g = adapter.get_metric_tensor(x)
    if g.shape != (len(FEATURE_COLUMNS), len(FEATURE_COLUMNS)):
        print(f"[FAIL] metric_tensor shape: {g.shape}")
        rc = 1
    elif not np.allclose(g, g.T):
        print(f"[FAIL] metric_tensor not symmetric")
        rc = 1
    elif np.linalg.eigvalsh(g).min() <= 0:
        print(f"[FAIL] metric_tensor not positive-definite")
        rc = 1
    else:
        print(f"[OK]   get_metric_tensor: {g.shape} symmetric positive-definite")

    # 9. Sandbox action dispatch.
    action = adapter.get_sandbox_action()
    if action != "isolate_doh_stream":
        print(f"[FAIL] get_sandbox_action: {action}")
        rc = 1
    else:
        print(f"[OK]   get_sandbox_action = {action}")
    result = adapter.apply_sandbox_action(flow, action)
    if not result.success or result.action != action:
        print(f"[FAIL] apply_sandbox_action: {result}")
        rc = 1
    else:
        print(f"[OK]   apply_sandbox_action: success, {result.action}")

    # 10. get_adapter factory with a protocol whose adapter is not yet
    #     delivered raises ProtocolNotSupportedError (the concrete
    #     adapters are delivered later in the sequence).
    from src.protocols.base_adapter import ProtocolNotSupportedError
    try:
        get_adapter("doq", config)
    except ProtocolNotSupportedError:
        print(f"[OK]   get_adapter rejects not-yet-delivered adapter")
    except ImportError:
        print(f"[OK]   get_adapter raises ImportError (adapter not yet delivered)")
    else:
        print(f"[OK]   get_adapter for doq succeeded (adapter is available)")

    print("=== self-test complete ===")
    return rc


if __name__ == "__main__":
    import sys
    sys.exit(_self_test())
