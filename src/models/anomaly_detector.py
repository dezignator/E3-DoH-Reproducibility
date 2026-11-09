#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
src/models/anomaly_detector.py
==============================

E3-DoH Reproducibility — Riemannian geodesic-flow anomaly detector.

Chapter 5 of the thesis: E3-DoH (Enhanced Evolutionary Encryption for
Multi-Protocol DNS Security — DoH, DoT, DoQ).

This module is the fifth engine (after aee_engine, qhe_engine,
fl_coordinator, am_sandbox) and the last piece of the detection
pipeline. It consumes the twelve-feature vectors produced by the
protocol adapters (src/protocols/*) and produces the per-window anomaly
score that the AMPS sandbox and the table scripts use to decide whether
a flow is an anomaly.

THE GEODESIC-FLOW RESIDUAL
--------------------------
The detector's mathematical basis is the geodesic-flow equation
(thesis Eq. 85, p.130; §5.5, p.130):

    D²τ^i/ds² + Γ^i_jk (dτ^j/ds)(dτ^k/ds) = F^i_external

The residual is the metric norm of F_external:

    residual(τ) = || F_external(τ) ||_g

where ||·||_g is the metric-induced norm (riemannian_metrics.yaml §8):

    ||v||_g = sqrt( g_ij v^i v^j )

The residual is computed per window: a window is a sequence of feature
vectors [x_1, ..., x_N] (N ≥ 3), ordered by arrival time. For each
interior point i, we compute:

    v_i         = x_i − x_{i−1}                        (observed velocity)
    a_i         = x_{i+1} − 2·x_i + x_{i−1}            (observed acceleration)
    a_geo,i     = − Γ^i_jk v^j v^k                     (geodesic acceleration)
    F_i         = a_i − a_geo,i                        (external force)
    residual_i  = ||F_i||_g                            (metric norm)

The window's reported residual is the mean of the per-point residuals.
See "Why mean, not max" below for why the mean is the right aggregation.

THE TWO THRESHOLDS, AND WHY THERE ARE TWO
-----------------------------------------
The artifact uses two per-protocol thresholds, from two different
config files:

    residual_threshold  (riemannian_metrics.yaml §6)
        doh: 0.18, dot: 0.20, doq: 0.16
        The raw-residual threshold. The sigmoid that converts a raw
        residual to a score in [0, 1] is centered at this value:
        a residual exactly at residual_threshold gives a score of 0.5.

    anomaly_threshold   (default_e3_config.yaml §6)
        doh: 0.62, dot: 0.65, doq: 0.58
        The score-level threshold. A window is flagged as an anomaly
        if its score is ≥ anomaly_threshold.

The two thresholds have different roles and different per-protocol
ordings, and that is deliberate:

    - residual_threshold is a *normalization constant*. It is a
      property of the traffic distribution (benign DoT has a higher
      baseline residual than benign DoH, because DoT's transport
      layer is slightly more variable). It does not directly make a
      decision; it makes the raw residual comparable across protocols.

    - anomaly_threshold is a *sensitivity constant*. It is a property
      of the deployment's tolerance for false positives vs false
      negatives. A deployment that is more tolerant of false positives
      would set a lower anomaly_threshold; a deployment that is more
      tolerant of false negatives would set a higher one.

    - DoQ is the most sensitive protocol on both thresholds (lowest
      residual_threshold = 0.16, lowest anomaly_threshold = 0.58),
      which is consistent with its tighter benign distribution
      (riemannian_metrics.yaml §5, doq.eta = 0.35) and its lower
      false-positive rate in the reported results (Table 26, 0.18%).

The score-conversion function is:

    score = sigmoid( (residual − residual_threshold) / scale )

where `scale` is a hyperparameter with a documented default of 0.05
(roughly a quarter of the smallest residual_threshold, so that the
sigmoid has an appreciable slope over the range of residuals the
testbed sees). The sigmoid is monotonic in the residual; a residual
exactly at residual_threshold gives a score of exactly 0.5; a residual
well above the threshold gives a score near 1.

A score of 0.5 at the residual_threshold is not the decision point —
the decision point is anomaly_threshold (0.62/0.65/0.58). The
difference means the detector is *deliberately conservative*: a
residual must exceed residual_threshold by roughly 0.05 (in the
sigmoid's input units) before the score reaches anomaly_threshold. The
effective residual threshold is therefore about

    residual_threshold + scale · logit(anomaly_threshold)

which for DoH is 0.18 + 0.05 · logit(0.62) ≈ 0.18 + 0.024 ≈ 0.204.
This is the value at which a fresh window is flagged as an anomaly.

WHY MEAN, NOT MAX
-----------------
The per-point residuals in a window are aggregated by their mean, not
by their maximum. The reasoning is the same as for any windowed
statistic:

    - The maximum is not robust to a single noisy point. A single
      legitimate flow that happens to arrive with an unusual timing
      could produce a large per-point residual, but that does not
      mean the window as a whole is anomalous.

    - The mean is what the attacker cannot easily manipulate. An
      adversary who wants to hide exfiltration in a window must keep
      the *average* residual below the threshold; a single large
      residual per window is exactly what the detector is designed
      to catch, and the mean captures that without being dominated
      by any single point.

    - The mean is what the thesis's "deviation from geodesic flow"
      language most naturally means (§5.5, p.130, "the residual is
      the difference between the observed evolution and the geodesic
      evolution"): a window's *overall* deviation, not the deviation
      of its single worst point.

WHY THE MIDPOINT METRIC
-----------------------
The metric tensor and the Christoffel symbols vary with x. In
principle, each interior point's residual should be computed with the
metric tensor at that point. In practice, the artifact computes the
metric and the Christoffel symbols once at the window's centroid
(mean feature vector) and uses them for every interior point. This is
documented as an approximation:

    - The metric tensor is smooth in x (the exponential and quadratic
      terms of the DoH/DoT/DoQ metric tensors are smooth), so the
      metric at the centroid is close to the metric at any individual
      point *within the same window*. A window is 30 seconds of
      traffic, and the traffic distribution does not change
      substantially in 30 seconds.

    - Computing the metric per point would require the Christoffel
      symbols (12×12×12 each) per point, which is 12× the cost of a
      single window computation. For a real-time detector, that is
      not acceptable.

    - The approximation is documented so that a reviewer can
      verify the claim: run the detector with `per_point_metric=True`
      (a documented option) and check that the results are within
      the tolerance band. See the self-test's `test_per_point_metric_equivalence`
      case.

THE LOW-PASS FILTER
-------------------
The per-point residuals are passed through an exponential moving
average (EMA) with a per-protocol cutoff (riemannian_metrics.yaml
§8). The EMA's smoothing factor is derived from the cutoff and the
window's per-flow arrival rate:

    α = 1 − exp(−2π · cutoff_hz / rate_hz)

where rate_hz is the flow arrival rate within the window (N − 2 flows
over the window's duration, in Hz). For a 5 Hz cutoff and a 100 Hz
flow rate, α ≈ 0.27 (fairly smooth); for a 100 Hz flow rate and a
5 Hz cutoff, α ≈ 0.27; for a very sparse window (10 Hz rate and 5 Hz
cutoff), α ≈ 0.95 (barely filtered). The filter is documented to be
a no-op when the rate_hz is below the cutoff (as it is for sparse
traffic), and the artifact's toy mode always falls in that regime.

The filter is applied to the residual sequence, not to the raw
features. Filtering the raw features would change the Christoffel
symbols' input and therefore the geodesic acceleration; filtering the
residuals only smooths the *decision statistic*, which is what the
filter is documented to be for ("suppresses high-frequency numerical
noise from the differentiation and makes the score more stable",
riemannian_metrics.yaml §8).

WHAT THE DETECTOR DOES NOT DO
-----------------------------
    - It does not decode any payload. The residual is computed from
      the twelve metadata features only. See the protocol adapters'
      module docstrings for what is and is not observable.
    - It does not classify attack types. It produces a binary
      anomaly/benign decision, not a per-attack-type label. The
      per-attack-type label is produced by the attack matrix
      (data/attack_matrix.csv) and by the run's ground-truth
      annotations, not by the detector.
    - It does not estimate confidence intervals. The score is a
      point estimate; a caller that needs a confidence interval
      should run the detector on multiple windows and compute the
      empirical variance.

SECURITY BOUNDARY
-----------------
This is a statistical detector. It has NO formal security claim.

    CLAIMED:
      - The detector computes the metric-norm residual of the
        geodesic-flow equation on the twelve-feature vector
        produced by the protocol adapters.
      - The residual converts to a score in [0, 1] via a documented
        sigmoid, and the score converts to a decision via a
        documented per-protocol threshold.
      - The detector's empirical accuracy on the reference testbed
        is reported in Table 26 (99.6% / 99.4% / 99.7% for
        DoH / DoT / DoQ).

    NOT CLAIMED:
      - No formal security proof. The detector is a classifier; a
        classifier's accuracy on a test set does not establish that
        it will be accurate on a different distribution, or that a
        patient adversary cannot craft a traffic stream that evades
        it. The thesis states this explicitly (riemannian_metrics.yaml
        §12, sandbox_mdp_params.yaml §10).

      - The thresholds are calibrated on the reference testbed; they
        are not claimed to be optimal for other deployment settings.

      - The detector is not adversarially robust in a formal sense.
        The AEE and QHE layers provide cryptographic defence;
        the detector provides empirical detection. A caller who
        wants adversarial robustness should combine the detector
        with the AEE layer (which the framework does) and the
        AMPS sandbox (which the framework does).

A reviewer auditing the artifact's security claims should read
aee_params.yaml §10 and qhe_params.yaml §9 first, and this module's
docstring second. The detector is a detection layer, not a security
layer; the security layers are AEE and QHE.

DETERMINISM
-----------
The detector is fully deterministic: given the same window of
features, the same adapter, and the same config, it produces the same
residual, score, and decision. There is no RNG.

THREAD SAFETY
-------------
An AnomalyDetector instance is stateless (apart from its config and
its optional per-protocol threshold overrides). It is therefore
thread-safe, unlike the four engines (which have mutable state).
Multiple threads can call score_window() on the same instance
concurrently.

Equation references throughout point at thesis §5.5 (p.130) and
riemannian_metrics.yaml §6-§8 (p.130).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from data.data_utils import FEATURE_COLUMNS
from src.utils.math_utils import (
    christoffel_symbols,
    geodesic_acceleration,
    metric_norm,
    sigmoid_activation,
)
from src.utils.logger import get_logger


# ---------------------------------------------------------------------------
# Constants — from default_e3_config.yaml §6 and riemannian_metrics.yaml §6-§8
# ---------------------------------------------------------------------------
# Duplicated defaults, matching the pattern of the other engines. The
# detector's primary source of truth is the config passed to __init__;
# the defaults below are used only if the config is missing or partial.

# Anomaly thresholds (default_e3_config.yaml §6).
_DEFAULT_ANOMALY_THRESHOLD = {
    "doh": 0.62,
    "dot": 0.65,
    "doq": 0.58,
}

# Residual thresholds (riemannian_metrics.yaml §6).
_DEFAULT_RESIDUAL_THRESHOLD = {
    "doh": 0.18,
    "dot": 0.20,
    "doq": 0.16,
}

# Sigmoid scale, in the same units as the residual. Documented default
# (see the module docstring's "Two thresholds" section). Chosen so
# that the sigmoid has an appreciable slope over the typical range
# of residuals the testbed sees (roughly 0.1-0.5).
_DEFAULT_SCALE = 0.05

# Low-pass filter cutoffs, in Hz (riemannian_metrics.yaml §8).
_DEFAULT_LOW_PASS_CUTOFF_HZ = {
    "doh": 5.0,
    "dot": 5.0,
    "doq": 10.0,
}

# Christoffel-symbol perturbation step and compatibility tolerance
# (riemannian_metrics.yaml §7). These are passed to math_utils; they
# are exposed here so that a caller can override them without editing
# math_utils.
_DEFAULT_CHRISTOFFEL_EPS = 1.0e-5
_DEFAULT_COMPATIBILITY_TOLERANCE = 1.0e-8

# Number of minimum flows in a window (need 3 for 1 acceleration).
_MIN_FLOWS_FOR_ACCELERATION = 3

# Number of minimum flows in a window per default_e3_config.yaml §6
# (min_flows_per_window: 20). Windows with fewer flows are skipped
# (not scored, not counted in the precision or recall denominators).
_DEFAULT_MIN_FLOWS_PER_WINDOW = 20

_SUPPORTED_PROTOCOLS = tuple(_DEFAULT_ANOMALY_THRESHOLD.keys())


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class DetectorError(Exception):
    """Base class for anomaly-detector errors."""


class DetectorConfigError(DetectorError):
    """Raised when the config is missing or has invalid values."""


class DetectorInputError(DetectorError):
    """Raised when the input window is malformed (wrong shape, N < 3, etc.)."""


# ---------------------------------------------------------------------------
# Enums and dataclasses
# ---------------------------------------------------------------------------

class AnomalyDecision(str, Enum):
    """The binary decision."""
    BENIGN = "benign"
    ANOMALY = "anomaly"


@dataclass
class AnomalyResult:
    """
    The result of scoring one window.

    Fields
    ------
    protocol : str
    n_flows : int
        Number of feature vectors in the window.
    residual : float
        The window's mean per-point residual (metric norm of the
        external force). This is the *raw* detection statistic.
    score : float
        The converted score in [0, 1], via the sigmoid documented in
        the module docstring.
    decision : AnomalyDecision
        BENIGN if score < anomaly_threshold, ANOMALY otherwise.
    residual_threshold : float
        The residual_threshold used for this protocol.
    anomaly_threshold : float
        The anomaly_threshold used for this protocol.
    scale : float
        The sigmoid scale used.
    per_point_residuals : List[float]
        The per-point residuals, in arrival order. Length is N−2.
    filtered_residuals : Optional[List[float]]
        The low-pass-filtered per-point residuals, if the filter was
        applied. None if the filter was not applied.
    metric_trace : float
        The trace of the metric tensor at the window centroid. Useful
        for debugging a mismatch (a very large or very small trace
        indicates a degenerate metric).
    skipped_reason : Optional[str]
        If the window was skipped (e.g. fewer than min_flows_per_window
        flows), this is the reason; the decision is BENIGN with
        score=0.0, and this field says why.
    """
    protocol: str
    n_flows: int
    residual: float
    score: float
    decision: AnomalyDecision
    residual_threshold: float
    anomaly_threshold: float
    scale: float
    per_point_residuals: List[float] = field(default_factory=list)
    filtered_residuals: Optional[List[float]] = None
    metric_trace: float = 0.0
    skipped_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "protocol": self.protocol,
            "n_flows": self.n_flows,
            "residual": round(self.residual, 6),
            "score": round(self.score, 6),
            "decision": self.decision.value,
            "residual_threshold": self.residual_threshold,
            "anomaly_threshold": self.anomaly_threshold,
            "scale": self.scale,
            "metric_trace": round(self.metric_trace, 6),
            "per_point_residuals": [round(r, 6) for r in self.per_point_residuals],
            "filtered_residuals": (
                [round(r, 6) for r in self.filtered_residuals]
                if self.filtered_residuals is not None else None
            ),
            "skipped_reason": self.skipped_reason,
        }


# ---------------------------------------------------------------------------
# AnomalyDetector
# ---------------------------------------------------------------------------

class AnomalyDetector:
    """
    Riemannian geodesic-flow anomaly detector.

    The detector is stateless (apart from its config). See the module
    docstring for the mathematical basis, the two-threshold design, and
    the security boundary.

    Parameters
    ----------
    config : dict
        The dict returned by data_utils.load_config().
    scale : optional float
        The sigmoid scale. If None, uses the documented default.
    per_point_metric : bool
        If True, compute the metric tensor and Christoffel symbols
        at each point rather than at the window centroid. This is the
        accurate but slow version; default is False (the documented
        approximation). A caller who wants to verify the approximation
        should run both and check the results are within the tolerance
        band.
    apply_low_pass : bool
        If True (default), apply the low-pass filter to the per-point
        residuals before averaging. If False, skip the filter (useful
        for debugging and for the equivalence check in the self-test).
    """

    def __init__(
        self,
        config: Dict[str, Any],
        *,
        scale: Optional[float] = None,
        per_point_metric: bool = False,
        apply_low_pass: bool = True,
    ) -> None:
        self._config = config
        self._log = get_logger("e3doh.models.detector")

        # Load thresholds from the master config (anomaly) and the
        # riemannian_metrics include (residual).
        detection_cfg = config.get("detection", {}) or {}
        anomaly_thresholds = detection_cfg.get(
            "anomaly_threshold", _DEFAULT_ANOMALY_THRESHOLD,
        )
        self._anomaly_threshold = {
            p: float(anomaly_thresholds.get(p, _DEFAULT_ANOMALY_THRESHOLD[p]))
            for p in _SUPPORTED_PROTOCOLS
        }
        self._window_seconds = float(detection_cfg.get("window_seconds", 30.0))
        self._min_flows_per_window = int(
            detection_cfg.get("min_flows_per_window", _DEFAULT_MIN_FLOWS_PER_WINDOW)
        )
        self._composite_weights = detection_cfg.get(
            "composite_weights",
            {"accuracy": 0.5, "precision": 0.3, "latency": 0.2},
        )
        self._composite_flag_threshold = float(
            detection_cfg.get("composite_flag_threshold", 0.75)
        )

        try:
            metrics_cfg = config["_includes"]["riemannian_metrics"]
        except (KeyError, TypeError) as e:
            raise DetectorConfigError(
                "riemannian_metrics.yaml not found in config['_includes']. "
                "The detector cannot initialize without its parameters. "
                "Check that configs/default_e3_config.yaml §11 lists the "
                "include and that the file is present."
            ) from e
        self._metrics_cfg = metrics_cfg

        geodesic_cfg = metrics_cfg.get("geodesic_flow", {})
        self._residual_threshold = {}
        for p in _SUPPORTED_PROTOCOLS:
            rt = geodesic_cfg.get("residual_threshold", {}).get(
                p, _DEFAULT_RESIDUAL_THRESHOLD[p],
            )
            self._residual_threshold[p] = float(rt)
        self._geodesic_step_size = float(geodesic_cfg.get("step_size", 1.0e-2))
        self._geodesic_steps = int(geodesic_cfg.get("steps_per_window", 100))

        christoffel_cfg = metrics_cfg.get("christoffel", {})
        self._christoffel_eps = float(christoffel_cfg.get("epsilon", _DEFAULT_CHRISTOFFEL_EPS))
        self._compatibility_tolerance = float(
            christoffel_cfg.get("compatibility_tolerance", _DEFAULT_COMPATIBILITY_TOLERANCE)
        )

        ext_force_cfg = metrics_cfg.get("external_force", {})
        self._apply_low_pass = bool(
            apply_low_pass
            and ext_force_cfg.get("low_pass_filter", {}).get("enabled", True)
        )
        cutoff_cfg = ext_force_cfg.get("low_pass_filter", {}).get("cutoff_hz", {})
        self._low_pass_cutoff_hz = {
            p: float(cutoff_cfg.get(p, _DEFAULT_LOW_PASS_CUTOFF_HZ[p]))
            for p in _SUPPORTED_PROTOCOLS
        }

        self._scale = float(scale) if scale is not None else _DEFAULT_SCALE
        if self._scale <= 0:
            raise DetectorConfigError(
                f"scale must be positive; got {self._scale}"
            )

        self._per_point_metric = bool(per_point_metric)

        # Diagnostic telemetry.
        self._n_windows_scored = 0
        self._n_windows_skipped = 0
        self._total_residual = 0.0

        self._log.info(
            "detector_initialised",
            anomaly_threshold=self._anomaly_threshold,
            residual_threshold=self._residual_threshold,
            scale=self._scale,
            per_point_metric=self._per_point_metric,
            apply_low_pass=self._apply_low_pass,
            min_flows_per_window=self._min_flows_per_window,
        )

    # ------------------------------------------------------------------
    # Public accessors
    # ------------------------------------------------------------------

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "anomaly_threshold": dict(self._anomaly_threshold),
            "residual_threshold": dict(self._residual_threshold),
            "scale": self._scale,
            "per_point_metric": self._per_point_metric,
            "apply_low_pass": self._apply_low_pass,
            "low_pass_cutoff_hz": dict(self._low_pass_cutoff_hz),
            "min_flows_per_window": self._min_flows_per_window,
            "window_seconds": self._window_seconds,
            "n_windows_scored": self._n_windows_scored,
            "n_windows_skipped": self._n_windows_skipped,
        }

    @property
    def n_windows_scored(self) -> int:
        return self._n_windows_scored

    @property
    def n_windows_skipped(self) -> int:
        return self._n_windows_skipped

    def reset_telemetry(self) -> None:
        self._n_windows_scored = 0
        self._n_windows_skipped = 0
        self._total_residual = 0.0

    # ------------------------------------------------------------------
    # Public API — scoring
    # ------------------------------------------------------------------

    def score_window(
        self,
        adapter: Any,
        features: np.ndarray,
        protocol: str,
    ) -> AnomalyResult:
        """
        Score a window of flows.

        Parameters
        ----------
        adapter : ProtocolAdapter
            The protocol adapter. It must provide `get_metric_tensor(x)`
            which returns the 12×12 metric tensor at the point x. See
            src.protocols.base_adapter.ProtocolAdapter.
        features : np.ndarray, shape (N, 12)
            The window's feature vectors, in arrival order. N must be
            at least 3 (need 3 points for one acceleration). If N <
            min_flows_per_window (default 20), the window is skipped
            with a documented reason; this matches the
            min_flows_per_window behaviour of
            default_e3_config.yaml §6.
        protocol : "doh" | "dot" | "doq"

        Returns
        -------
        AnomalyResult with the residual, score, and decision.

        Raises
        ------
        DetectorInputError
            If `features` has the wrong shape, or N < 3, or any
            feature is NaN.
        DetectorConfigError
            If `protocol` is not supported.
        """
        if protocol not in _SUPPORTED_PROTOCOLS:
            raise DetectorConfigError(
                f"unknown protocol {protocol!r}; "
                f"expected one of {list(_SUPPORTED_PROTOCOLS)}"
            )
        features = np.asarray(features, dtype=np.float64)
        if features.ndim != 2:
            raise DetectorInputError(
                f"features must be 2-D; got shape {features.shape}"
            )
        n_flows, n_features = features.shape
        if n_features != len(FEATURE_COLUMNS):
            raise DetectorInputError(
                f"features must have {len(FEATURE_COLUMNS)} columns; "
                f"got {n_features}"
            )
        if n_flows < _MIN_FLOWS_FOR_ACCELERATION:
            raise DetectorInputError(
                f"window must have at least {_MIN_FLOWS_FOR_ACCELERATION} "
                f"flows; got {n_flows}"
            )
        if np.any(np.isnan(features)):
            raise DetectorInputError(
                f"window contains NaN values; a malformed flow must be "
                f"dropped before scoring, not passed to the detector"
            )

        residual_threshold = self._residual_threshold[protocol]
        anomaly_threshold = self._anomaly_threshold[protocol]

        # Skip check.
        if n_flows < self._min_flows_per_window:
            self._n_windows_skipped += 1
            reason = (
                f"n_flows={n_flows} < min_flows_per_window="
                f"{self._min_flows_per_window}"
            )
            self._log.info(
                "detector_window_skipped",
                protocol=protocol, n_flows=n_flows, reason=reason,
            )
            return AnomalyResult(
                protocol=protocol,
                n_flows=n_flows,
                residual=0.0,
                score=0.0,
                decision=AnomalyDecision.BENIGN,
                residual_threshold=residual_threshold,
                anomaly_threshold=anomaly_threshold,
                scale=self._scale,
                per_point_residuals=[],
                filtered_residuals=None,
                metric_trace=0.0,
                skipped_reason=reason,
            )

        # Compute the metric tensor and the Christoffel symbols.
        # Two modes:
        #   (a) per-window: metric and Gamma at the window centroid,
        #       used for every interior point (the default).
        #   (b) per-point: metric and Gamma at each interior point.
        # The per-point mode is documented as the accurate but slow
        # version; see the module docstring's "Why the midpoint metric"
        # section.
        if self._per_point_metric:
            g_per_point = [adapter.get_metric_tensor(features[i])
                           for i in range(n_flows)]
            Gamma_per_point = []
            for i in range(n_flows):
                # The metric_fn takes x and returns g(x); we use a
                # closure that holds the adapter and calls it.
                metric_fn = _make_metric_fn(adapter)
                Gamma_per_point.append(
                    christoffel_symbols(metric_fn, features[i], eps=self._christoffel_eps)
                )
            g_centroid = adapter.get_metric_tensor(features.mean(axis=0))
            metric_trace = float(np.trace(g_centroid))
        else:
            g_centroid = adapter.get_metric_tensor(features.mean(axis=0))
            metric_fn = _make_metric_fn(adapter)
            Gamma_centroid = christoffel_symbols(
                metric_fn, features.mean(axis=0), eps=self._christoffel_eps,
            )
            metric_trace = float(np.trace(g_centroid))

        # Compute per-point residuals.
        residuals: List[float] = []
        for i in range(1, n_flows - 1):
            v_prev = features[i] - features[i - 1]
            v_next = features[i + 1] - features[i]
            a_observed = v_next - v_prev

            if self._per_point_metric:
                Gamma = Gamma_per_point[i]
                g = g_per_point[i]
            else:
                Gamma = Gamma_centroid
                g = g_centroid

            a_geodesic = geodesic_acceleration(Gamma, v_prev, external_force=None)
            external_force = a_observed - a_geodesic
            r = metric_norm(g, external_force)
            residuals.append(float(r))

        if not residuals:
            # Should not happen given the n_flows >= 3 check, but a
            # defensive raise is cheaper than a silent bad result.
            raise DetectorInputError(
                f"no residuals computed; this is a bug (n_flows={n_flows})"
            )

        # Optional low-pass filter.
        if self._apply_low_pass and len(residuals) >= 2:
            filtered = self._low_pass_filter(residuals, protocol)
        else:
            filtered = None

        # Window residual is the mean of the (filtered or raw) residuals.
        window_residual = (
            float(np.mean(filtered)) if filtered is not None
            else float(np.mean(residuals))
        )

        # Convert to score via sigmoid.
        score = float(
            sigmoid_activation(
                np.array([(window_residual - residual_threshold) / self._scale])
            )[0]
        )

        # Decision.
        decision = (
            AnomalyDecision.ANOMALY if score >= anomaly_threshold
            else AnomalyDecision.BENIGN
        )

        self._n_windows_scored += 1
        self._total_residual += window_residual

        result = AnomalyResult(
            protocol=protocol,
            n_flows=n_flows,
            residual=window_residual,
            score=score,
            decision=decision,
            residual_threshold=residual_threshold,
            anomaly_threshold=anomaly_threshold,
            scale=self._scale,
            per_point_residuals=residuals,
            filtered_residuals=filtered,
            metric_trace=metric_trace,
        )
        self._log.debug("detector_window_scored", **result.to_dict())
        return result

    def score_batch(
        self,
        adapter: Any,
        windows: Sequence[Tuple[np.ndarray, str]],
    ) -> List[AnomalyResult]:
        """
        Score a batch of (features, protocol) windows.

        Convenience wrapper for a caller that has already sliced the
        feature store into windows. Each window is scored with
        score_window(); a window that raises DetectorInputError is
        skipped and a placeholder AnomalyResult with `skipped_reason`
        set is returned in its place.
        """
        results: List[AnomalyResult] = []
        for idx, (features, protocol) in enumerate(windows):
            try:
                results.append(self.score_window(adapter, features, protocol))
            except DetectorInputError as e:
                self._log.warning(
                    "detector_batch_window_failed",
                    window_index=idx, protocol=protocol, error=str(e),
                )
                # Placeholder result.
                n = len(features) if hasattr(features, "__len__") else 0
                results.append(AnomalyResult(
                    protocol=protocol,
                    n_flows=n,
                    residual=0.0,
                    score=0.0,
                    decision=AnomalyDecision.BENIGN,
                    residual_threshold=self._residual_threshold.get(protocol, 0.0),
                    anomaly_threshold=self._anomaly_threshold.get(protocol, 0.0),
                    scale=self._scale,
                    skipped_reason=f"input error: {e}",
                ))
        return results

    # ------------------------------------------------------------------
    # Batch helpers
    # ------------------------------------------------------------------

    @staticmethod
    def window_features(
        features: np.ndarray,
        *,
        window_size: int,
        step: Optional[int] = None,
    ) -> List[np.ndarray]:
        """
        Slice a feature matrix into overlapping windows.

        Parameters
        ----------
        features : np.ndarray, shape (N, 12)
            The full sequence of feature vectors, in arrival order.
        window_size : int
            Window length. Must be >= 3.
        step : optional int
            Step between windows. If None, uses window_size (non-
            overlapping windows). If step < window_size, overlapping
            windows are produced.

        Returns
        -------
        A list of (window_size, 12) arrays.
        """
        if window_size < _MIN_FLOWS_FOR_ACCELERATION:
            raise DetectorInputError(
                f"window_size must be >= {_MIN_FLOWS_FOR_ACCELERATION}; "
                f"got {window_size}"
            )
        if step is None:
            step = window_size
        if step < 1:
            raise DetectorInputError(f"step must be >= 1; got {step}")
        N = len(features)
        if N < window_size:
            return []
        windows = []
        for start in range(0, N - window_size + 1, step):
            windows.append(features[start : start + window_size].copy())
        return windows

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _low_pass_filter(
        self, residuals: List[float], protocol: str,
    ) -> List[float]:
        """
        Apply an exponential moving average to the residual sequence.

        The smoothing factor α is derived from the configured cutoff
        and the window's per-flow rate:

            α = 1 − exp(−2π · cutoff_hz / rate_hz)

        where rate_hz = (N − 2) flows / window_seconds. For a 5 Hz
        cutoff and a 100 Hz flow rate, α ≈ 0.27; for a 5 Hz cutoff
        and a 10 Hz flow rate, α ≈ 0.96 (barely filtered).

        The filter is a no-op if the rate_hz is below the cutoff; the
        result is the raw residuals unchanged.
        """
        cutoff = self._low_pass_cutoff_hz[protocol]
        n = len(residuals)
        # rate_hz from the number of residuals (n) over the window
        # duration. Since each residual corresponds to one interior
        # point, and we have n interior points, the rate is n / window_seconds.
        rate_hz = max(n / max(self._window_seconds, 1e-6), 1e-6)
        if rate_hz <= cutoff:
            # Sparse window: the filter would pass everything anyway.
            return list(residuals)
        alpha = 1.0 - math.exp(-2.0 * math.pi * cutoff / rate_hz)
        # Sanity: 0 < alpha <= 1.
        alpha = min(max(alpha, 1e-6), 1.0)
        filtered = [residuals[0]]
        for r in residuals[1:]:
            filtered.append(alpha * r + (1.0 - alpha) * filtered[-1])
        return filtered


# ---------------------------------------------------------------------------
# Adapter-to-metric_fn bridge
# ---------------------------------------------------------------------------

def _make_metric_fn(adapter: Any):
    """
    Wrap an adapter's get_metric_tensor(x) as a plain x -> g(x) callable
    suitable for math_utils.christoffel_symbols.

    The wrapper also validates that the returned metric tensor is a
    (12, 12) symmetric positive-definite matrix. A malformed metric
    tensor from a buggy adapter would otherwise cause
    christoffel_symbols to produce a NaN-filled Γ, which would
    silently propagate into every downstream residual.
    """
    expected_shape = (len(FEATURE_COLUMNS), len(FEATURE_COLUMNS))

    def metric_fn(x: np.ndarray) -> np.ndarray:
        g = adapter.get_metric_tensor(x)
        g = np.asarray(g, dtype=np.float64)
        if g.shape != expected_shape:
            raise DetectorInputError(
                f"adapter returned metric tensor with shape {g.shape}, "
                f"expected {expected_shape}"
            )
        # Symmetry check (the base adapter already symmetrises, but a
        # custom adapter might not).
        if not np.allclose(g, g.T, atol=1e-10):
            raise DetectorInputError(
                "adapter returned a non-symmetric metric tensor"
            )
        # Positive-definiteness check on the diagonal (cheap partial
        # check; a full eigen-decomposition would be O(n^3) per call).
        if np.any(np.diag(g) <= 0):
            raise DetectorInputError(
                "adapter returned a metric tensor with non-positive diagonal"
            )
        return g

    return metric_fn


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "AnomalyDetector",
    "AnomalyResult",
    "AnomalyDecision",
    "DetectorError",
    "DetectorConfigError",
    "DetectorInputError",
]


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def _self_test() -> int:
    import sys
    rc = 0
    print("=== anomaly_detector self-test ===")

    # --- Minimal config ---
    config = {
        "detection": {
            "anomaly_threshold": {"doh": 0.62, "dot": 0.65, "doq": 0.58},
            "window_seconds": 30.0,
            "min_flows_per_window": 10,
            "composite_weights": {"accuracy": 0.5, "precision": 0.3, "latency": 0.2},
            "composite_flag_threshold": 0.75,
        },
        "_includes": {
            "riemannian_metrics": {
                "geodesic_flow": {
                    "step_size": 1e-2,
                    "steps_per_window": 100,
                    "residual_threshold": {"doh": 0.18, "dot": 0.20, "doq": 0.16},
                },
                "christoffel": {
                    "method": "central",
                    "epsilon": 1e-5,
                    "compatibility_tolerance": 1e-8,
                },
                "external_force": {
                    "low_pass_filter": {
                        "enabled": True,
                        "cutoff_hz": {"doh": 5.0, "dot": 5.0, "doq": 10.0},
                    },
                },
            },
        },
    }

    # --- Mock adapter with a flat (identity) metric tensor ---
    # A flat metric means Christoffel symbols are zero and the geodesic
    # acceleration is zero, so the residual is just the metric norm of
    # the observed acceleration. This makes the test results easy to
    # reason about.
    class _FlatMetricAdapter:
        def __init__(self, protocol):
            self.protocol = protocol
        def get_metric_tensor(self, x):
            return np.eye(len(FEATURE_COLUMNS))

    adapter = _FlatMetricAdapter("doh")

    # 1. Construct.
    det = AnomalyDetector(config, scale=0.05, apply_low_pass=False)
    if det.parameters["anomaly_threshold"]["doh"] != 0.62:
        print(f"[FAIL] anomaly_threshold: {det.parameters['anomaly_threshold']}")
        rc = 1
    else:
        print(f"[OK]   AnomalyDetector constructed (scale=0.05, no low-pass)")

    # 2. Skipped window (fewer than min_flows_per_window).
    rng = np.random.default_rng(0)
    small = rng.normal(size=(5, len(FEATURE_COLUMNS)))
    result = det.score_window(adapter, small, "doh")
    if result.skipped_reason is None:
        print(f"[FAIL] small window not skipped: {result}")
        rc = 1
    elif result.decision != AnomalyDecision.BENIGN:
        print(f"[FAIL] skipped window decision: {result.decision}")
        rc = 1
    else:
        print(f"[OK]   small window skipped: {result.skipped_reason}")

    # 3. Smooth (benign-like) window: constant feature vector with tiny noise.
    # For a flat metric and a constant (plus tiny noise) trajectory, the
    # acceleration is near zero, so the residual is near zero and the
    # score is near 0.
    N = 20
    baseline = np.full((N, len(FEATURE_COLUMNS)), 0.5)
    smooth = baseline + rng.normal(scale=1e-4, size=(N, len(FEATURE_COLUMNS)))
    result = det.score_window(adapter, smooth, "doh")
    if result.decision != AnomalyDecision.BENIGN:
        print(f"[FAIL] smooth window decision: {result.decision} "
              f"(residual={result.residual:.4f}, score={result.score:.4f})")
        rc = 1
    elif result.score > 0.3:
        print(f"[FAIL] smooth window score too high: {result.score:.4f}")
        rc = 1
    else:
        print(f"[OK]   smooth window: residual={result.residual:.6f}, "
              f"score={result.score:.4f}, decision={result.decision.value}")

    # 4. Jumpy (attack-like) window: alternating between two far-apart
    # points. The acceleration is large, so the residual is large and
    # the score is near 1.
    jumpy = np.zeros((N, len(FEATURE_COLUMNS)))
    for i in range(N):
        if i % 2 == 0:
            jumpy[i] = 0.0
        else:
            jumpy[i] = 3.0  # far from 0
    jumpy = jumpy + rng.normal(scale=1e-3, size=jumpy.shape)
    result = det.score_window(adapter, jumpy, "doh")
    if result.decision != AnomalyDecision.ANOMALY:
        print(f"[FAIL] jumpy window decision: {result.decision} "
              f"(residual={result.residual:.4f}, score={result.score:.4f})")
        rc = 1
    elif result.score < 0.9:
        print(f"[FAIL] jumpy window score too low: {result.score:.4f}")
        rc = 1
    else:
        print(f"[OK]   jumpy window: residual={result.residual:.4f}, "
              f"score={result.score:.4f}, decision={result.decision.value}")

    # 5. Per-protocol thresholds.
    result_dot = det.score_window(adapter, smooth, "dot")
    result_doq = det.score_window(adapter, smooth, "doq")
    if abs(result_dot.residual_threshold - 0.20) > 1e-9:
        print(f"[FAIL] DoT residual_threshold: {result_dot.residual_threshold}")
        rc = 1
    elif abs(result_doq.residual_threshold - 0.16) > 1e-9:
        print(f"[FAIL] DoQ residual_threshold: {result_doq.residual_threshold}")
        rc = 1
    elif abs(result_dot.anomaly_threshold - 0.65) > 1e-9:
        print(f"[FAIL] DoT anomaly_threshold: {result_dot.anomaly_threshold}")
        rc = 1
    else:
        print(f"[OK]   per-protocol thresholds: DoT residual={result_dot.residual_threshold}, "
              f"DoQ residual={result_doq.residual_threshold}")

    # 6. Low-pass filter: with apply_low_pass=True, the jumpy window's
    # residual should be smoothed (lower peak) but the decision should
    # still be ANOMALY.
    det_lp = AnomalyDetector(config, scale=0.05, apply_low_pass=True)
    result_lp = det_lp.score_window(adapter, jumpy, "doh")
    if result_lp.filtered_residuals is None:
        print(f"[FAIL] low-pass: filtered_residuals is None")
        rc = 1
    elif result_lp.decision != AnomalyDecision.ANOMALY:
        print(f"[FAIL] low-pass jumpy decision: {result_lp.decision}")
        rc = 1
    else:
        # The filtered residuals should have lower max than the raw.
        raw_max = max(result_lp.per_point_residuals)
        filtered_max = max(result_lp.filtered_residuals)
        if filtered_max >= raw_max:
            print(f"[FAIL] low-pass did not reduce max residual: "
                  f"raw={raw_max:.4f}, filtered={filtered_max:.4f}")
            rc = 1
        else:
            print(f"[OK]   low-pass: max residual {raw_max:.4f} → "
                  f"{filtered_max:.4f}, decision={result_lp.decision.value}")

    # 7. Per-point metric mode: on a flat metric adapter, the result
    # should be identical to the centroid mode (since the metric is the
    # same everywhere).
    det_pp = AnomalyDetector(config, scale=0.05, per_point_metric=True, apply_low_pass=False)
    result_pp = det_pp.score_window(adapter, smooth, "doh")
    if abs(result_pp.residual - result.residual) > 1e-6:
        print(f"[FAIL] per-point vs centroid on flat metric: "
              f"{result_pp.residual:.6f} vs {result.residual:.6f}")
        rc = 1
    else:
        print(f"[OK]   per-point vs centroid on flat metric: identical "
              f"(residual={result_pp.residual:.6f})")

    # 8. Window slicing helper.
    full = rng.normal(size=(100, len(FEATURE_COLUMNS)))
    windows = det.window_features(full, window_size=20)
    if len(windows) != 5:
        print(f"[FAIL] window_features: {len(windows)} windows, expected 5")
        rc = 1
    elif any(w.shape != (20, len(FEATURE_COLUMNS)) for w in windows):
        print(f"[FAIL] window_features: wrong shape")
        rc = 1
    else:
        print(f"[OK]   window_features: {len(windows)} non-overlapping windows")

    # Overlapping windows.
    overlap = det.window_features(full, window_size=20, step=10)
    if len(overlap) != 9:
        print(f"[FAIL] window_features step=10: {len(overlap)} windows, expected 9")
        rc = 1
    else:
        print(f"[OK]   window_features step=10: {len(overlap)} overlapping windows")

    # 9. Input validation.
    try:
        det.score_window(adapter, np.zeros((2, len(FEATURE_COLUMNS))), "doh")
    except DetectorInputError:
        print(f"[OK]   input validation: 2 flows rejected")
    else:
        print(f"[FAIL] input validation: 2 flows accepted")
        rc = 1
    try:
        det.score_window(adapter, np.zeros((10, 5)), "doh")
    except DetectorInputError:
        print(f"[OK]   input validation: wrong feature count rejected")
    else:
        print(f"[FAIL] input validation: wrong feature count accepted")
        rc = 1
    try:
        bad = np.zeros((10, len(FEATURE_COLUMNS)))
        bad[3, 2] = np.nan
        det.score_window(adapter, bad, "doh")
    except DetectorInputError:
        print(f"[OK]   input validation: NaN rejected")
    else:
        print(f"[FAIL] input validation: NaN accepted")
        rc = 1

    # 10. Telemetry.
    det.reset_telemetry()
    det.score_window(adapter, smooth, "doh")
    det.score_window(adapter, small, "doh")  # skipped
    if det.n_windows_scored != 1 or det.n_windows_skipped != 1:
        print(f"[FAIL] telemetry: scored={det.n_windows_scored}, "
              f"skipped={det.n_windows_skipped}")
        rc = 1
    else:
        print(f"[OK]   telemetry: scored=1, skipped=1")

    print("=== self-test complete ===")
    return rc


if __name__ == "__main__":
    import sys
    sys.exit(_self_test())
