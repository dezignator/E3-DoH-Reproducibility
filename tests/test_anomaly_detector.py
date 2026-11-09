#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tests/test_anomaly_detector.py
==============================

E3-DoH Reproducibility — unit tests for src/models/anomaly_detector.py.

These tests verify the detector's public contract:

    - Construction and parameter loading
    - Scoring a "smooth" window (low score, BENIGN)
    - Scoring a "jumpy" window (high score, ANOMALY)
    - Skipping windows with too few flows
    - Per-protocol thresholds
    - The low-pass filter
    - The per-point metric mode (equivalence on a flat metric)
    - Window slicing helper
    - Input validation

The tests use a `_FlatMetricAdapter` mock so that the geometric
computation is simple (the metric tensor is the identity, the
Christoffel symbols are zero, so the residual reduces to the norm of
the observed acceleration). This isolates the detector's contract
from the protocol-specific metric tensors.

Run with:
    pytest tests/test_anomaly_detector.py -v
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from data.data_utils import FEATURE_COLUMNS
from src.models.anomaly_detector import (
    AnomalyDecision,
    AnomalyDetector,
    AnomalyResult,
    DetectorConfigError,
    DetectorInputError,
)


# =============================================================================
# Mock adapter
# =============================================================================

class _FlatMetricAdapter:
    """
    An adapter whose metric tensor is the identity everywhere.

    On a flat metric, the Christoffel symbols are zero, so the
    geodesic acceleration is zero, and the residual is just the
    metric norm of the observed acceleration.
    """
    def __init__(self, protocol: str = "doh"):
        self.protocol = protocol

    def get_metric_tensor(self, x: np.ndarray) -> np.ndarray:
        return np.eye(len(FEATURE_COLUMNS))


# =============================================================================
# Construction
# =============================================================================

class TestConstruction:
    def test_construct(self, minimal_am_config):
        det = AnomalyDetector(minimal_am_config, scale=0.05, apply_low_pass=False)
        assert det.parameters["scale"] == 0.05
        assert det.parameters["anomaly_threshold"]["doh"] == 0.62

    def test_missing_config_raises(self):
        with pytest.raises(DetectorConfigError):
            AnomalyDetector({})

    def test_negative_scale_rejected(self, minimal_am_config):
        with pytest.raises(DetectorConfigError):
            AnomalyDetector(minimal_am_config, scale=-0.05)

    def test_parameters_expose_per_protocol(self, minimal_am_config):
        det = AnomalyDetector(minimal_am_config)
        p = det.parameters
        assert p["anomaly_threshold"]["doh"] == 0.62
        assert p["anomaly_threshold"]["dot"] == 0.65
        assert p["anomaly_threshold"]["doq"] == 0.58


# =============================================================================
# Scoring: smooth vs jumpy
# =============================================================================

class TestScoring:
    def test_smooth_window_is_benign(self, minimal_am_config, smooth_window):
        adapter = _FlatMetricAdapter("doh")
        det = AnomalyDetector(minimal_am_config, apply_low_pass=False)
        result = det.score_window(adapter, smooth_window, "doh")
        assert result.decision == AnomalyDecision.BENIGN
        assert result.score < 0.5

    def test_jumpy_window_is_anomaly(self, minimal_am_config, jumpy_window):
        adapter = _FlatMetricAdapter("doh")
        det = AnomalyDetector(minimal_am_config, apply_low_pass=False)
        result = det.score_window(adapter, jumpy_window, "doh")
        assert result.decision == AnomalyDecision.ANOMALY
        assert result.score > 0.5

    def test_smooth_scores_lower_than_jumpy(
        self, minimal_am_config, smooth_window, jumpy_window,
    ):
        adapter = _FlatMetricAdapter("doh")
        det = AnomalyDetector(minimal_am_config, apply_low_pass=False)
        s_smooth = det.score_window(adapter, smooth_window, "doh").score
        s_jumpy = det.score_window(adapter, jumpy_window, "doh").score
        assert s_smooth < s_jumpy

    def test_result_has_per_point_residuals(
        self, minimal_am_config, smooth_window,
    ):
        adapter = _FlatMetricAdapter("doh")
        det = AnomalyDetector(minimal_am_config, apply_low_pass=False)
        result = det.score_window(adapter, smooth_window, "doh")
        # N-2 interior points.
        assert len(result.per_point_residuals) == len(smooth_window) - 2


# =============================================================================
# Skipping short windows
# =============================================================================

class TestSkipping:
    def test_too_few_flows_skipped(self, minimal_am_config):
        # min_flows_per_window is 10 in minimal_am_config.
        adapter = _FlatMetricAdapter("doh")
        det = AnomalyDetector(minimal_am_config, apply_low_pass=False)
        small = np.zeros((5, len(FEATURE_COLUMNS)))
        result = det.score_window(adapter, small, "doh")
        assert result.skipped_reason is not None
        assert result.decision == AnomalyDecision.BENIGN
        assert result.score == 0.0

    def test_too_few_flows_raises_before_skip(self, minimal_am_config):
        # A window with < 3 flows is a hard error (cannot compute
        # acceleration at all).
        adapter = _FlatMetricAdapter("doh")
        det = AnomalyDetector(minimal_am_config, apply_low_pass=False)
        with pytest.raises(DetectorInputError):
            det.score_window(adapter, np.zeros((2, len(FEATURE_COLUMNS))), "doh")


# =============================================================================
# Per-protocol thresholds
# =============================================================================

class TestPerProtocol:
    def test_dot_threshold(self, minimal_am_config, smooth_window):
        adapter = _FlatMetricAdapter("dot")
        det = AnomalyDetector(minimal_am_config, apply_low_pass=False)
        result = det.score_window(adapter, smooth_window, "dot")
        assert result.anomaly_threshold == 0.65

    def test_doq_threshold(self, minimal_am_config, smooth_window):
        adapter = _FlatMetricAdapter("doq")
        det = AnomalyDetector(minimal_am_config, apply_low_pass=False)
        result = det.score_window(adapter, smooth_window, "doq")
        assert result.anomaly_threshold == 0.58

    def test_unknown_protocol_rejected(self, minimal_am_config, smooth_window):
        adapter = _FlatMetricAdapter("doh")
        det = AnomalyDetector(minimal_am_config)
        with pytest.raises(DetectorConfigError):
            det.score_window(adapter, smooth_window, "foo")


# =============================================================================
# Low-pass filter
# =============================================================================

class TestLowPass:
    def test_filtered_residuals_present(self, minimal_am_config, jumpy_window):
        adapter = _FlatMetricAdapter("doh")
        det = AnomalyDetector(minimal_am_config, apply_low_pass=True)
        result = det.score_window(adapter, jumpy_window, "doh")
        assert result.filtered_residuals is not None

    def test_filter_reduces_peak(self, minimal_am_config, jumpy_window):
        adapter = _FlatMetricAdapter("doh")
        det = AnomalyDetector(minimal_am_config, apply_low_pass=True)
        result = det.score_window(adapter, jumpy_window, "doh")
        raw_max = max(result.per_point_residuals)
        filtered_max = max(result.filtered_residuals)
        assert filtered_max <= raw_max

    def test_disabled_has_no_filter(self, minimal_am_config, jumpy_window):
        adapter = _FlatMetricAdapter("doh")
        det = AnomalyDetector(minimal_am_config, apply_low_pass=False)
        result = det.score_window(adapter, jumpy_window, "doh")
        assert result.filtered_residuals is None


# =============================================================================
# Per-point metric mode
# =============================================================================

class TestPerPointMetric:
    def test_flat_metric_equivalence(self, minimal_am_config, smooth_window):
        # On a flat metric, per-point and centroid modes agree.
        adapter = _FlatMetricAdapter("doh")
        det_centroid = AnomalyDetector(
            minimal_am_config, per_point_metric=False, apply_low_pass=False,
        )
        det_per_point = AnomalyDetector(
            minimal_am_config, per_point_metric=True, apply_low_pass=False,
        )
        r_cent = det_centroid.score_window(adapter, smooth_window, "doh")
        r_pp = det_per_point.score_window(adapter, smooth_window, "doh")
        assert abs(r_cent.residual - r_pp.residual) < 1e-6


# =============================================================================
# Window slicing helper
# =============================================================================

class TestWindowSlicing:
    def test_non_overlapping(self, minimal_am_config):
        features = np.random.default_rng(0).normal(
            size=(100, len(FEATURE_COLUMNS)),
        )
        windows = AnomalyDetector.window_features(
            features, window_size=20,
        )
        assert len(windows) == 5
        assert all(w.shape == (20, len(FEATURE_COLUMNS)) for w in windows)

    def test_overlapping(self, minimal_am_config):
        features = np.random.default_rng(0).normal(
            size=(100, len(FEATURE_COLUMNS)),
        )
        windows = AnomalyDetector.window_features(
            features, window_size=20, step=10,
        )
        assert len(windows) == 9

    def test_too_small_features(self, minimal_am_config):
        features = np.random.default_rng(0).normal(
            size=(10, len(FEATURE_COLUMNS)),
        )
        windows = AnomalyDetector.window_features(
            features, window_size=20,
        )
        assert windows == []

    def test_invalid_window_size(self, minimal_am_config):
        with pytest.raises(DetectorInputError):
            AnomalyDetector.window_features(np.zeros((10, 12)), window_size=2)


# =============================================================================
# Input validation
# =============================================================================

class TestInputValidation:
    def test_wrong_feature_count_rejected(self, minimal_am_config):
        adapter = _FlatMetricAdapter("doh")
        det = AnomalyDetector(minimal_am_config)
        with pytest.raises(DetectorInputError):
            det.score_window(adapter, np.zeros((20, 5)), "doh")

    def test_nan_rejected(self, minimal_am_config):
        adapter = _FlatMetricAdapter("doh")
        det = AnomalyDetector(minimal_am_config)
        w = np.zeros((20, len(FEATURE_COLUMNS)))
        w[3, 2] = np.nan
        with pytest.raises(DetectorInputError):
            det.score_window(adapter, w, "doh")

    def test_non_2d_rejected(self, minimal_am_config):
        adapter = _FlatMetricAdapter("doh")
        det = AnomalyDetector(minimal_am_config)
        with pytest.raises(DetectorInputError):
            det.score_window(adapter, np.zeros(len(FEATURE_COLUMNS)), "doh")


# =============================================================================
# Telemetry
# =============================================================================

class TestTelemetry:
    def test_scored_and_skipped_counts(self, minimal_am_config, smooth_window):
        adapter = _FlatMetricAdapter("doh")
        det = AnomalyDetector(minimal_am_config, apply_low_pass=False)
        det.score_window(adapter, smooth_window, "doh")  # scored
        det.score_window(adapter, np.zeros((5, len(FEATURE_COLUMNS))), "doh")  # skipped
        assert det.n_windows_scored == 1
        assert det.n_windows_skipped == 1

    def test_reset(self, minimal_am_config, smooth_window):
        adapter = _FlatMetricAdapter("doh")
        det = AnomalyDetector(minimal_am_config)
        det.score_window(adapter, smooth_window, "doh")
        det.reset_telemetry()
        assert det.n_windows_scored == 0
        assert det.n_windows_skipped == 0


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
