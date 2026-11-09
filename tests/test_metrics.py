#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tests/test_metrics.py
=====================

E3-DoH Reproducibility — unit tests for src/utils/metrics.py.

These tests verify the metric definitions against analytically-known
values. They complement tests/test_math_utils.py (which tests the
low-level primitives that metrics.py uses) by covering the
confusion-matrix, latency, throughput, and per-protocol aggregation
helpers.

Run with:
    pytest tests/test_metrics.py -v
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

from src.utils.metrics import (
    DEFAULT_PERCENTILES,
    MIN_SAMPLES_FOR_P99,
    THROUGHPUT_LATENCY_MAX_MS,
    aggregate_per_protocol,
    brier_score,
    composite_detection_score,
    confusion_counts,
    detection_accuracy,
    f1_score,
    f1_weighted,
    false_positive_rate,
    latency_percentiles,
    latency_summary,
    precision,
    round_for_reporting,
    throughput_at_p99,
    true_positive_rate,
)


# =============================================================================
# confusion_counts
# =============================================================================

class TestConfusionCounts:
    def test_simple_binary(self):
        # y_true = [1, 1, 0, 0, 1, 0, 0, 1]
        # y_pred = [1, 0, 0, 1, 1, 0, 0, 1]
        tp, tn, fp, fn = confusion_counts(
            [1, 1, 0, 0, 1, 0, 0, 1],
            [1, 0, 0, 1, 1, 0, 0, 1],
        )
        assert (tp, tn, fp, fn) == (3, 3, 1, 1)

    def test_perfect_detector(self):
        tp, tn, fp, fn = confusion_counts([1, 0, 1, 0], [1, 0, 1, 0])
        assert (tp, tn, fp, fn) == (2, 2, 0, 0)

    def test_all_wrong(self):
        tp, tn, fp, fn = confusion_counts([1, 1, 0, 0], [0, 0, 1, 1])
        assert (tp, tn, fp, fn) == (0, 0, 2, 2)

    def test_string_labels(self):
        tp, tn, fp, fn = confusion_counts(
            ["attack", "benign", "attack", "benign"],
            ["attack", "benign", "benign", "attack"],
            positive_label="attack",
        )
        assert (tp, tn, fp, fn) == (1, 1, 1, 1)

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            confusion_counts([1, 0], [1])

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            confusion_counts([], [])

    def test_three_labels_raises(self):
        with pytest.raises(ValueError):
            confusion_counts([0, 1, 2], [0, 1, 2])

    def test_positive_label_must_appear(self):
        with pytest.raises(ValueError):
            confusion_counts([0, 1], [0, 1], positive_label=2)


# =============================================================================
# Scalar metrics from counts
# =============================================================================

class TestScalarMetrics:
    def test_detection_accuracy(self):
        assert detection_accuracy(3, 3, 1, 1) == pytest.approx(6 / 8)

    def test_detection_accuracy_zero_denom(self):
        # No predictions: accuracy is 0 (degenerate case).
        assert detection_accuracy(0, 0, 0, 0) == 0.0

    def test_fpr(self):
        assert false_positive_rate(3, 3, 1, 1) == pytest.approx(1 / 4)

    def test_fpr_zero_denom(self):
        assert false_positive_rate(0, 0, 0, 1) == 0.0

    def test_tpr(self):
        assert true_positive_rate(3, 3, 1, 1) == pytest.approx(3 / 4)

    def test_tpr_zero_denom(self):
        assert true_positive_rate(0, 3, 1, 0) == 0.0

    def test_precision(self):
        assert precision(3, 3, 1, 1) == pytest.approx(3 / 4)

    def test_precision_zero_denom(self):
        assert precision(0, 3, 0, 0) == 0.0

    def test_f1(self):
        # P = R = 0.75 → F1 = 0.75.
        assert f1_score(3, 3, 1, 1) == pytest.approx(0.75)

    def test_f1_zero_precision(self):
        assert f1_score(0, 10, 0, 5) == 0.0

    def test_f1_weighted(self):
        # Two classes: (tp, tn, fp, fn) = (9, 0, 0, 1) and (5, 5, 0, 0).
        # F1_class1 = 2*1*0.9/(1+0.9) ≈ 0.947. F1_class2 = 1.0.
        # Weighted: (10 * 0.947 + 5 * 1.0) / 15 ≈ 0.965.
        f = f1_weighted((9, 0, 0, 1), (5, 5, 0, 0))
        assert 0.9 < f < 1.0

    def test_brier_score(self):
        # (0.9, 0.1, 0.8, 0.2) vs (1, 0, 1, 0).
        # = (0.01 + 0.01 + 0.04 + 0.04) / 4 = 0.025
        bs = brier_score([1, 0, 1, 0], [0.9, 0.1, 0.8, 0.2])
        assert bs == pytest.approx(0.025)

    def test_brier_perfect(self):
        assert brier_score([1, 0], [1.0, 0.0]) == pytest.approx(0.0)


# =============================================================================
# Latency percentiles
# =============================================================================

class TestLatencyPercentiles:
    def test_known_distribution(self):
        # 100 evenly-spaced values from 1.0 to 10.0 ms.
        vals = list(np.linspace(1.0, 10.0, 100))
        p = latency_percentiles(vals, percentiles=[50.0, 95.0, 99.0])
        # p50 ≈ 5.455, p95 ≈ 9.55, p99 ≈ 9.91.
        assert 5.4 < p["p50"] < 5.5
        assert 9.5 < p["p95"] < 9.6
        assert 9.9 < p["p99"] < 9.95

    def test_single_value(self):
        p = latency_percentiles([5.0], percentiles=[50.0])
        assert p["p50"] == 5.0

    def test_empty_returns_nan(self):
        p = latency_percentiles([], percentiles=[50.0])
        assert math.isnan(p["p50"])

    def test_unit_conversion_us(self):
        # 1000 μs = 1 ms.
        p = latency_percentiles([1000.0, 1000.0], unit="us", percentiles=[50.0])
        assert p["p50"] == pytest.approx(1.0)

    def test_unit_conversion_s(self):
        p = latency_percentiles([1.0, 1.0], unit="s", percentiles=[50.0])
        assert p["p50"] == pytest.approx(1000.0)

    def test_small_sample_warns(self):
        with pytest.warns(RuntimeWarning):
            latency_percentiles(list(range(10)), percentiles=[99.0])

    def test_latency_summary(self):
        s = latency_summary([1.0, 2.0, 3.0, 4.0, 5.0])
        assert s["count"] == 5
        assert s["mean"] == pytest.approx(3.0)
        assert s["min"] == 1.0
        assert s["max"] == 5.0


# =============================================================================
# Throughput
# =============================================================================

class TestThroughput:
    def test_within_budget(self):
        qps = [1.0, 2.0, 4.0, 8.0, 12.0]
        p99 = [0.5, 0.9, 1.4, 1.9, 3.0]
        t = throughput_at_p99(qps, p99, latency_max_ms=2.0)
        assert t == 8.0

    def test_all_within_budget(self):
        qps = [1.0, 2.0]
        p99 = [0.5, 0.9]
        assert throughput_at_p99(qps, p99) == 2.0

    def test_none_within_budget(self):
        qps = [10.0, 20.0]
        p99 = [3.0, 4.0]
        assert throughput_at_p99(qps, p99) == 0.0

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            throughput_at_p99([1.0], [0.5, 0.9])

    def test_empty(self):
        assert throughput_at_p99([], []) == 0.0


# =============================================================================
# Composite detection score
# =============================================================================

class TestCompositeDetectionScore:
    def test_formula(self):
        # (TP+TN)/(TP+TN+FP+FN) = 6/8 = 0.75
        # (1 - FP/(TP+FP)) = 1 - 1/4 = 0.75
        # (1 - 1.5/2) = 0.25
        # → 0.75 * 0.75 * 0.25 = 0.140625
        s = composite_detection_score(3, 3, 1, 1, latency_p99=1.5, latency_max=2.0)
        assert s == pytest.approx(0.140625)

    def test_perfect_detector(self):
        s = composite_detection_score(100, 1000, 0, 0, latency_p99=0.0, latency_max=2.0)
        assert s == pytest.approx(1.0)

    def test_higher_latency_lower_score(self):
        s_low = composite_detection_score(90, 1000, 10, 10, latency_p99=0.5, latency_max=2.0)
        s_high = composite_detection_score(90, 1000, 10, 10, latency_p99=1.9, latency_max=2.0)
        assert s_low > s_high

    def test_higher_fpr_lower_score(self):
        s_low = composite_detection_score(90, 1000, 1, 10, latency_p99=1.0, latency_max=2.0)
        s_high = composite_detection_score(90, 1000, 50, 10, latency_p99=1.0, latency_max=2.0)
        assert s_low > s_high


# =============================================================================
# Per-protocol aggregation
# =============================================================================

class TestAggregatePerProtocol:
    def test_mean_and_best(self):
        agg = aggregate_per_protocol({
            "doh": {"accuracy": 0.996, "fpr": 0.0022, "p99": 1.7},
            "dot": {"accuracy": 0.994, "fpr": 0.0025, "p99": 1.9},
            "doq": {"accuracy": 0.997, "fpr": 0.0018, "p99": 0.8},
        })
        # Best accuracy: doq (highest).
        assert agg["best_protocol"]["accuracy"] == "doq"
        # Best fpr: doq (lowest).
        assert agg["best_protocol"]["fpr"] == "doq"
        # Best p99: doq (lowest).
        assert agg["best_protocol"]["p99"] == "doq"
        # Mean accuracy.
        assert agg["mean"]["accuracy"] == pytest.approx(
            (0.996 + 0.994 + 0.997) / 3, abs=1e-6,
        )

    def test_empty(self):
        agg = aggregate_per_protocol({})
        assert agg == {"mean": {}, "best_observed": {}, "best_protocol": {}}


# =============================================================================
# Reporting helpers
# =============================================================================

class TestRoundForReporting:
    def test_default_precision(self):
        assert round_for_reporting(0.123456) == 0.1235

    def test_custom_precision(self):
        assert round_for_reporting(0.123456, decimals=2) == 0.12

    def test_nan_passthrough(self):
        assert math.isnan(round_for_reporting(float("nan")))


# =============================================================================
# Module constants
# =============================================================================

class TestConstants:
    def test_throughput_latency_max(self):
        assert THROUGHPUT_LATENCY_MAX_MS == 2.0

    def test_default_percentiles(self):
        assert 50.0 in DEFAULT_PERCENTILES
        assert 99.0 in DEFAULT_PERCENTILES

    def test_min_samples_for_p99(self):
        assert MIN_SAMPLES_FOR_P99 == 100


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
