#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
src/utils/metrics.py
====================

E3-DoH Reproducibility — evaluation metrics.

Chapter 5 of the thesis: E3-DoH (Enhanced Evolutionary Encryption for
Multi-Protocol DNS Security — DoH, DoT, DoQ).

This module defines every metric that appears in Chapter 5's tables
and figures. It is the single source of truth for the *definition*
of each metric, so that a table script cannot accidentally use a
different definition from the verification script that checks its
output.

The thesis reports the following metrics:

    Detection accuracy               (TP+TN) / (TP+TN+FP+FN)
    False-positive rate              FP / (FP+TN)
    True-positive rate (recall)      TP / (TP+FN)
    Precision                        TP / (TP+FP)
    F1 (weighted)                    2PR / (P+R), class-weighted
    P50 / P95 / P99 latency          percentile of the decision-latency distribution
    Throughput (QPS)                 queries per second at the P99 ≤ 2 ms constraint
    Composite detection score        Eq. 107 (accuracy × precision × latency penalty)
    Brier score                      mean squared error of the probability estimates

The composite detection score is the one metric that requires
explanation, because it is the only metric that is not standard. It
is defined in thesis Eq. 107 (p.143) and the implementation here
matches that definition exactly. See `composite_detection_score()`.

CONVENTIONS
-----------
Confusion-matrix metrics
    The labels are always (attack, benign). "Positive" = attack;
    "negative" = benign. This matches the standard detection-theory
    convention and the thesis's own usage (a "false positive" is a
    benign flow flagged as attack).

    The four counts are always passed as `tp, tn, fp, fn` in that
    order, never as a confusion matrix. This is a deliberate choice:
    a 2×2 matrix has no fixed orientation, and a caller that passes
    it in the wrong orientation (rows=predicted vs rows=actual) is a
    silent defect that the type system cannot catch. The four
    scalars are unambiguous.

    If a caller has a (y_true, y_pred) pair, use `confusion_counts()`
    to get the four scalars before calling the metric.

Latency
    Latency is a *distribution*, not a scalar. The thesis reports P99
    (and sometimes P50/P95), which requires the full distribution.
    `latency_percentiles()` takes the distribution and returns the
    requested percentiles. A caller that has only a scalar should not
    use the percentile function; the artifact does not accept scalar
    latency as a substitute.

    The thesis unit is milliseconds. `latency_percentiles()` accepts
    an optional `unit` argument and converts to milliseconds if the
    caller passes a different unit; the default is milliseconds, which
    is what the reference values in Table 26 use.

Throughput
    Throughput is defined as the maximum QPS at which P99 ≤
    `latency_max_ms` (default 2.0 ms, matching thesis §5.12, L_max=2
    ms). It is computed from the (QPS, latency) pairs measured at each
    load level, not from a single aggregate. `throughput_at_p99()` is
    the function; a caller that has only a single (QPS, P99) pair
    should not use it, because the constraint is over the whole curve.

COMPOSITE DETECTION SCORE (Eq. 107)
-----------------------------------
The thesis defines:

    S_composite = (TP+TN)/(TP+TN+FP+FN)      # accuracy
                * (1 - FP/(TP+FP))           # 1 - false discovery rate
                * (1 - L_P99 / L_max)        # 1 - latency ratio

The three factors are: accuracy, precision, and latency. The
implementation here is exactly that; `math_utils.composite_detection_score`
is a pure function of the four counts and the two latencies, and
`composite_detection_score()` here is a thin wrapper that accepts
numpy arrays and scalars interchangeably and applies the same clamping
for degenerate denominators.

The composite score is *not* a probabilistic measure; it is a
deterministic scalar in [0, 1] used for ranking configurations. It
is reported in the thesis as a summary metric, and this artifact
reports it in the same way. Do NOT interpret it as a probability or
as a p-value.

SECURITY BOUNDARY
-----------------
None of the metrics in this module has a security meaning. A high
detection accuracy on a test set does not establish that the detector
is secure; it establishes that it classifies correctly on that set.
The security boundary of the framework is recorded in
aee_params.yaml §10 and qhe_params.yaml §9. This module is a
measurement tool, not a security argument.
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

# Import the thesis-equation composite score. Using the import rather
# than a local re-implementation guarantees that the definition is the
# same one that math_utils's self-test checks.
from src.utils.math_utils import composite_detection_score as _math_composite

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# The latency threshold used by the throughput definition. Matches
# thesis §5.12 (L_max = 2 ms) and the "P99 ≤ 2 ms" column of Table 26.
THROUGHPUT_LATENCY_MAX_MS: float = 2.0

# The default percentiles the artifact reports. The thesis reports
# P50, P95, P99 in different places; these are all of them.
DEFAULT_PERCENTILES: Tuple[float, ...] = (50.0, 95.0, 99.0)

# The minimum number of samples below which the p99 estimate is not
# meaningful. With N samples, the p99 estimate is the value at index
# ceil(0.99 * N) - 1; for N < 100 the p99 is the maximum, which is a
# very noisy estimate. The thesis's tests use N >> 100 (thousands of
# flows), so this is not an issue in practice; the constant is here so
# that a caller with fewer samples gets a warning rather than a
# misleading value.
MIN_SAMPLES_FOR_P99: int = 100

# The number of decimal places to round to in the output. The reference
# values in Table 26 are reported to 2 decimal places (99.6, 0.22,
# 1.7); the artifact rounds to 4 decimal places internally and lets
# the reporter round further. This is the internal precision, not the
# reporting precision.
_INTERNAL_PRECISION: int = 4


# ---------------------------------------------------------------------------
# 1. Confusion matrix primitives
# ---------------------------------------------------------------------------

def confusion_counts(
    y_true: Sequence[Union[int, bool, str]],
    y_pred: Sequence[Union[int, bool, str]],
    *,
    positive_label: Union[int, bool, str] = 1,
) -> Tuple[int, int, int, int]:
    """
    Compute the four confusion-matrix counts from a (y_true, y_pred)
    pair.

    Parameters
    ----------
    y_true, y_pred : sequences of the same length
        The ground-truth and predicted labels. The labels may be int
        (0/1), bool, or string ("attack"/"benign"); the positive label
        is given by `positive_label`.
    positive_label : the label that counts as "positive" (i.e. attack).
        Defaults to 1 (integer) which also matches True. For string
        labels, pass e.g. "attack".

    Returns
    -------
    (tp, tn, fp, fn)

    Raises
    ------
    ValueError
        If the two sequences have different lengths, are empty, or
        contain a label that is neither `positive_label` nor the
        complement. (The complement is inferred from the set of
        observed values, minus `positive_label`; if there are more than
        two distinct values, the function raises.)
    """
    y_true = list(y_true)
    y_pred = list(y_pred)
    if len(y_true) != len(y_pred):
        raise ValueError(
            f"y_true and y_pred have different lengths: "
            f"{len(y_true)} vs {len(y_pred)}"
        )
    if not y_true:
        raise ValueError("y_true is empty")
    distinct_true = set(y_true)
    distinct_pred = set(y_pred)
    distinct = distinct_true | distinct_pred
    if len(distinct) > 2:
        raise ValueError(
            f"expected at most two distinct labels, got {sorted(map(repr, distinct))}. "
            f"Map multi-class labels to a single positive class first."
        )
    if positive_label not in distinct:
        raise ValueError(
            f"positive_label {positive_label!r} does not appear in the data; "
            f"observed labels: {sorted(map(repr, distinct))}"
        )
    # The complement is the other label, if any.
    complement = (distinct - {positive_label})
    if len(complement) == 1:
        negative_label = next(iter(complement))
    else:
        # All labels are the same; both classes are the positive one.
        negative_label = None

    tp = sum(1 for t, p in zip(y_true, y_pred) if t == positive_label and p == positive_label)
    tn = sum(1 for t, p in zip(y_true, y_pred)
             if negative_label is not None and t == negative_label and p == negative_label)
    fp = sum(1 for t, p in zip(y_true, y_pred)
             if t != positive_label and p == positive_label)
    fn = sum(1 for t, p in zip(y_true, y_pred)
             if t == positive_label and p != positive_label)
    return int(tp), int(tn), int(fp), int(fn)


# ---------------------------------------------------------------------------
# 2. Scalar metrics from counts
# ---------------------------------------------------------------------------

def detection_accuracy(tp: int, tn: int, fp: int, fn: int) -> float:
    """
    Detection accuracy: (TP + TN) / (TP + TN + FP + FN).

    Returns 0.0 if the denominator is zero (a degenerate case: no
    predictions). This is deliberate; the accuracy of a detector that
    made no predictions is not defined, and 0 is a safe default that
    will be visible in the output rather than crashing the pipeline.
    """
    denom = tp + tn + fp + fn
    if denom == 0:
        return 0.0
    return float((tp + tn) / denom)


def false_positive_rate(tp: int, tn: int, fp: int, fn: int) -> float:
    """
    False-positive rate: FP / (FP + TN).

    Returns 0.0 if the denominator is zero (no benign flows). A
    detector that saw no benign flows cannot have a false-positive
    rate; 0 is the only defensible default.
    """
    denom = fp + tn
    if denom == 0:
        return 0.0
    return float(fp / denom)


def true_positive_rate(tp: int, tn: int, fp: int, fn: int) -> float:
    """
    True-positive rate (recall): TP / (TP + FN).

    Returns 0.0 if the denominator is zero (no attack flows).
    """
    denom = tp + fn
    if denom == 0:
        return 0.0
    return float(tp / denom)


def precision(tp: int, tn: int, fp: int, fn: int) -> float:
    """
    Precision: TP / (TP + FP).

    Returns 0.0 if the denominator is zero (no positive predictions).
    """
    denom = tp + fp
    if denom == 0:
        return 0.0
    return float(tp / denom)


def f1_score(tp: int, tn: int, fp: int, fn: int) -> float:
    """
    F1 score: harmonic mean of precision and recall.

    Returns 0.0 if precision or recall is 0.
    """
    p = precision(tp, tn, fp, fn)
    r = true_positive_rate(tp, tn, fp, fn)
    if p + r == 0:
        return 0.0
    return float(2 * p * r / (p + r))


def f1_weighted(
    *class_counts: Tuple[int, int, int, int],
) -> float:
    """
    Weighted F1 over multiple classes (or over multiple runs).

    Each positional argument is a (tp, tn, fp, fn) tuple for one
    class. The F1 is computed per class and weighted by the number of
    true instances in that class:

        F1_weighted = sum_i (n_i * F1_i) / sum_i n_i

    where n_i = tp_i + fn_i.

    The thesis uses weighted F1 for the CDS definition (Eq. 131); it is
    reported alongside AUPRC. For a binary detector, use `f1_score()`.
    """
    total_n = 0
    weighted_sum = 0.0
    for counts in class_counts:
        tp, tn, fp, fn = counts
        n_i = tp + fn
        if n_i == 0:
            continue
        total_n += n_i
        weighted_sum += n_i * f1_score(*counts)
    if total_n == 0:
        return 0.0
    return float(weighted_sum / total_n)


def brier_score(
    y_true: Sequence[Union[int, bool]],
    y_prob: Sequence[float],
) -> float:
    """
    Brier score: mean squared error of the probability estimates.

        BS = (1/N) * sum_i (p_i - y_i)^2

    `y_prob[i]` is the predicted probability that example i is
    positive; `y_true[i]` is 1 if it is positive and 0 otherwise.
    Lower is better; 0 is a perfect predictor.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    if y_true.shape != y_prob.shape:
        raise ValueError("y_true and y_prob must have the same shape")
    if len(y_true) == 0:
        return 0.0
    return float(np.mean((y_prob - y_true) ** 2))


# ---------------------------------------------------------------------------
# 3. Latency percentiles
# ---------------------------------------------------------------------------

def latency_percentiles(
    latencies: Sequence[float],
    *,
    percentiles: Sequence[float] = DEFAULT_PERCENTILES,
    unit: str = "ms",
) -> Dict[str, float]:
    """
    Compute the requested percentiles of a latency distribution.

    Parameters
    ----------
    latencies : sequence of float
        Decision latencies. The unit is given by `unit`.
    percentiles : sequence of float
        The percentiles to compute (default: 50, 95, 99).
    unit : "ms" | "us" | "s"
        The unit of `latencies`. The output is always in milliseconds,
        which is what the thesis uses.

    Returns
    -------
    A dict of {"p50": ..., "p95": ..., "p99": ...} (keys are
    "p<percentile>", with the percentile rounded to the nearest
    integer if it has a fractional part).

    Warns
    -----
    If len(latencies) < MIN_SAMPLES_FOR_P99 and 99 is in `percentiles`,
    a RuntimeWarning is emitted. The p99 estimate on a small sample is
    the maximum, which is a very noisy estimate; the warning is a
    reminder to use more samples if the p99 is being reported.
    """
    if len(latencies) == 0:
        return {f"p{int(p)}": float("nan") for p in percentiles}
    arr = np.asarray(latencies, dtype=np.float64)
    if unit == "ms":
        factor = 1.0
    elif unit == "us":
        factor = 1.0 / 1000.0
    elif unit == "s":
        factor = 1000.0
    else:
        raise ValueError(f"unknown unit {unit!r}; expected 'ms', 'us', or 's'")
    arr_ms = arr * factor

    if 99.0 in percentiles and len(arr_ms) < MIN_SAMPLES_FOR_P99:
        import warnings
        warnings.warn(
            f"p99 requested on a sample of {len(arr_ms)} < "
            f"{MIN_SAMPLES_FOR_P99}; the estimate is the maximum and is "
            f"very noisy. Use more samples if p99 is being reported.",
            RuntimeWarning, stacklevel=2,
        )

    out: Dict[str, float] = {}
    for p in percentiles:
        key = f"p{int(round(p))}"
        out[key] = float(np.percentile(arr_ms, p, method="linear"))
    return out


def latency_summary(latencies: Sequence[float], unit: str = "ms") -> Dict[str, float]:
    """
    Full summary of a latency distribution: count, mean, std, min, max,
    and the default percentiles (p50, p95, p99). All values are in
    milliseconds.
    """
    if len(latencies) == 0:
        return {
            "count": 0, "mean": float("nan"), "std": float("nan"),
            "min": float("nan"), "max": float("nan"),
            "p50": float("nan"), "p95": float("nan"), "p99": float("nan"),
        }
    arr = np.asarray(latencies, dtype=np.float64)
    if unit == "us":
        arr = arr / 1000.0
    elif unit == "s":
        arr = arr * 1000.0
    pcts = latency_percentiles(arr, unit="ms")
    return {
        "count": int(len(arr)),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        **pcts,
    }


# ---------------------------------------------------------------------------
# 4. Throughput at a P99 constraint
# ---------------------------------------------------------------------------

def throughput_at_p99(
    qps_values: Sequence[float],
    p99_values: Sequence[float],
    *,
    latency_max_ms: float = THROUGHPUT_LATENCY_MAX_MS,
) -> float:
    """
    The maximum QPS at which P99 ≤ latency_max_ms.

    Parameters
    ----------
    qps_values : sequence of float
        The QPS at each load level, monotonically non-decreasing.
    p99_values : sequence of float
        The measured P99 (ms) at each load level, same length.
    latency_max_ms : float
        The P99 latency constraint (default 2.0 ms, matching thesis
        §5.12).

    Returns
    -------
    The throughput in QPS. If no measured load level satisfies the
    constraint, returns 0.0 (rather than an extrapolated value, which
    would be unverifiable). If all load levels satisfy the constraint,
    returns the maximum QPS in `qps_values`; this is a *lower bound*
    on the true throughput, and the caller should increase the load
    range if a higher value is expected.

    Raises
    ------
    ValueError
        If the two sequences have different lengths.
    """
    if len(qps_values) != len(p99_values):
        raise ValueError(
            f"qps_values and p99_values must have the same length: "
            f"{len(qps_values)} vs {len(p99_values)}"
        )
    if len(qps_values) == 0:
        return 0.0
    # Find the last index where p99 <= latency_max_ms.
    best = 0.0
    for qps, p99 in zip(qps_values, p99_values):
        if p99 <= latency_max_ms and qps > best:
            best = float(qps)
    return best


# ---------------------------------------------------------------------------
# 5. Composite detection score (Eq. 107)
# ---------------------------------------------------------------------------

def composite_detection_score(
    tp: int, tn: int, fp: int, fn: int,
    *,
    latency_p99: float,
    latency_max: float = THROUGHPUT_LATENCY_MAX_MS,
) -> float:
    """
    The composite detection score of thesis Eq. 107 (p.143):

        S_composite = (TP+TN)/(TP+TN+FP+FN)
                    * (1 - FP/(TP+FP))
                    * (1 - L_P99 / L_max)

    This function is a thin wrapper around
    `math_utils.composite_detection_score`, which is the canonical
    implementation. It exists so that a caller that imports
    `src.utils.metrics` does not also need to import `math_utils` for
    this one function.

    Parameters
    ----------
    tp, tn, fp, fn : int
        Confusion-matrix counts.
    latency_p99 : float
        The P99 decision latency in milliseconds.
    latency_max : float
        The maximum admissible P99 (default 2.0 ms).

    Returns
    -------
    The composite score in [0, 1]. Higher is better.
    """
    return _math_composite(
        int(tp), int(tn), int(fp), int(fn),
        latency_p99=float(latency_p99),
        latency_max=float(latency_max),
    )


# ---------------------------------------------------------------------------
# 6. Per-protocol aggregation
# ---------------------------------------------------------------------------

def aggregate_per_protocol(
    per_protocol: Dict[str, Dict[str, float]],
) -> Dict[str, float]:
    """
    Aggregate the per-protocol metrics into a single summary.

    The thesis's Table 26 reports the three protocols side-by-side, and
    the per-figure scripts summarise them in a single number (the mean
    of the three). This function computes both:

        - the per-protocol mean of each metric, and
        - the "best observed" value (max for accuracy and throughput,
          min for latency and FPR).

    Parameters
    ----------
    per_protocol : {"doh": {...}, "dot": {...}, "doq": {...}}
        Each inner dict maps a metric name to a scalar.

    Returns
    -------
    A dict with three keys:
        "mean": {metric_name -> mean across protocols}
        "best_observed": {metric_name -> best value across protocols}
        "best_protocol": {metric_name -> protocol name that gave the best value}

    The notion of "best" depends on the metric: higher is better for
    accuracy, TPR, precision, F1, throughput; lower is better for FPR,
    latency, Brier score. The function encodes this per-metric
    direction in `_HIGHER_IS_BETTER` below. A metric not in the dict
    is assumed "higher is better"; a caller with a different
    convention should compute it themselves rather than rely on the
    default.

    Notes
    -----
    This function is what produces the "best observed" column of
    Table 26. The thesis's note under that table states: "the gap
    column reports the difference between the best and the next-best
    protocol on the same metric, in the metric's own units; gaps are
    not computed for rows that are not ranked." The "best_protocol"
    key is the name of the protocol that gave the best value; the
    "best_observed" key is the value itself.
    """
    protocols = list(per_protocol.keys())
    if not protocols:
        return {"mean": {}, "best_observed": {}, "best_protocol": {}}

    # Collect all metric names. A metric that appears in only some
    # protocols is included; the missing values are skipped in the
    # aggregation.
    all_metrics: set = set()
    for metrics in per_protocol.values():
        all_metrics.update(metrics.keys())

    mean_out: Dict[str, float] = {}
    best_out: Dict[str, float] = {}
    best_proto_out: Dict[str, str] = {}

    for metric in sorted(all_metrics):
        values = [(p, per_protocol[p][metric])
                  for p in protocols if metric in per_protocol[p]]
        if not values:
            continue
        arr = [v for _, v in values]
        mean_out[metric] = float(np.mean(arr))
        higher_is_better = _HIGHER_IS_BETTER.get(metric, True)
        if higher_is_better:
            best_idx = int(np.argmax(arr))
        else:
            best_idx = int(np.argmin(arr))
        best_proto_out[metric] = values[best_idx][0]
        best_out[metric] = float(values[best_idx][1])

    return {
        "mean": mean_out,
        "best_observed": best_out,
        "best_protocol": best_proto_out,
    }


# The per-metric direction map used by aggregate_per_protocol. A
# metric not in this dict is assumed "higher is better"; the
# following are the ones where lower is better.
_HIGHER_IS_BETTER: Dict[str, bool] = {
    "false_positive_rate": False,
    "fpr": False,
    "latency_p50_ms": False,
    "latency_p95_ms": False,
    "latency_p99_ms": False,
    "latency_ms": False,
    "brier_score": False,
    "detection_time_ms": False,
    "mitigation_time_ms": False,
}


# ---------------------------------------------------------------------------
# 7. Rounding and formatting for the reference sidecars
# ---------------------------------------------------------------------------

def round_for_reporting(value: float, decimals: int = _INTERNAL_PRECISION) -> float:
    """
    Round a value for inclusion in a result JSON sidecar.

    The artifact rounds internally to 4 decimal places. The
    per-table and per-figure scripts are responsible for any
    additional rounding they want to apply before writing to the
    sidecar; the verification script reads the rounded value and
    compares it within the tolerance band, so a value that is
    unnecessarily precise is not a correctness issue but is a
    readability one.

    NaN is passed through unchanged (the sidecar schema allows NaN in
    a `values` entry; the verification script treats NaN == NaN as a
    match, matching the IEEE 754 convention that the artifact's
    tolerance comparison uses).
    """
    if math.isnan(value):
        return value
    return round(float(value), decimals)


# ---------------------------------------------------------------------------
# 8. Public API
# ---------------------------------------------------------------------------

__all__ = [
    # Confusion matrix
    "confusion_counts",
    # Scalar metrics
    "detection_accuracy", "false_positive_rate", "true_positive_rate",
    "precision", "f1_score", "f1_weighted", "brier_score",
    # Latency
    "latency_percentiles", "latency_summary",
    # Throughput
    "throughput_at_p99",
    # Composite score
    "composite_detection_score",
    # Aggregation
    "aggregate_per_protocol",
    # Formatting
    "round_for_reporting",
    # Constants
    "THROUGHPUT_LATENCY_MAX_MS", "DEFAULT_PERCENTILES",
    "MIN_SAMPLES_FOR_P99",
]


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
# Running `python -m src.utils.metrics` verifies the metric definitions
# against small, analytically-known examples. It is a smoke test; the
# full test suite (tests/test_metrics.py, later in the sequence) checks
# the same functions against scikit-learn's implementations plus the
# tolerance-band aggregation.

def _self_test() -> int:
    import sys
    rc = 0
    print("=== metrics self-test ===")

    # --- Confusion counts ---
    # y_true  = [1, 1, 0, 0, 1, 0, 0, 1]
    # y_pred  = [1, 0, 0, 1, 1, 0, 0, 1]
    # tp = 3 (indices 0, 4, 7)
    # tn = 3 (indices 2, 5, 6)
    # fp = 1 (index 3)
    # fn = 1 (index 1)
    tp, tn, fp, fn = confusion_counts(
        [1, 1, 0, 0, 1, 0, 0, 1],
        [1, 0, 0, 1, 1, 0, 0, 1],
    )
    if (tp, tn, fp, fn) != (3, 3, 1, 1):
        print(f"[FAIL] confusion_counts: {(tp, tn, fp, fn)}")
        rc = 1
    else:
        print(f"[OK]   confusion_counts: tp={tp} tn={tn} fp={fp} fn={fn}")

    # --- Detection accuracy ---
    acc = detection_accuracy(3, 3, 1, 1)
    if abs(acc - 6/8) > 1e-12:
        print(f"[FAIL] detection_accuracy: {acc}")
        rc = 1
    else:
        print(f"[OK]   detection_accuracy = {acc:.6f}")

    # --- FPR ---
    fpr = false_positive_rate(3, 3, 1, 1)
    if abs(fpr - 1/4) > 1e-12:
        print(f"[FAIL] false_positive_rate: {fpr}")
        rc = 1
    else:
        print(f"[OK]   false_positive_rate = {fpr:.6f}")

    # --- TPR / recall ---
    tpr = true_positive_rate(3, 3, 1, 1)
    if abs(tpr - 3/4) > 1e-12:
        print(f"[FAIL] true_positive_rate: {tpr}")
        rc = 1
    else:
        print(f"[OK]   true_positive_rate = {tpr:.6f}")

    # --- Precision ---
    p = precision(3, 3, 1, 1)
    if abs(p - 3/4) > 1e-12:
        print(f"[FAIL] precision: {p}")
        rc = 1
    else:
        print(f"[OK]   precision = {p:.6f}")

    # --- F1 ---
    f1 = f1_score(3, 3, 1, 1)
    if abs(f1 - 3/4) > 1e-12:
        print(f"[FAIL] f1_score: {f1}")
        rc = 1
    else:
        print(f"[OK]   f1_score = {f1:.6f}")

    # --- Latency percentiles ---
    # 100 evenly-spaced values from 1.0 to 10.0 ms.
    lats = list(np.linspace(1.0, 10.0, 100))
    pcts = latency_percentiles(lats, percentiles=[50.0, 99.0])
    # p50 ≈ 5.455, p99 ≈ 9.91 (linear interpolation)
    if not (5.4 <= pcts["p50"] <= 5.5):
        print(f"[FAIL] latency_percentiles p50: {pcts['p50']}")
        rc = 1
    elif not (9.9 <= pcts["p99"] <= 9.95):
        print(f"[FAIL] latency_percentiles p99: {pcts['p99']}")
        rc = 1
    else:
        print(f"[OK]   latency_percentiles p50={pcts['p50']:.4f}, p99={pcts['p99']:.4f}")

    # --- Throughput at P99 ---
    qps = [1, 2, 4, 8, 12]
    p99 = [0.5, 0.9, 1.4, 1.9, 3.0]
    tput = throughput_at_p99(qps, p99, latency_max_ms=2.0)
    if tput != 8.0:
        print(f"[FAIL] throughput_at_p99: {tput}")
        rc = 1
    else:
        print(f"[OK]   throughput_at_p99 = {tput}")

    # --- Composite detection score ---
    # (TP+TN)/(TP+TN+FP+FN) = 6/8 = 0.75
    # (1 - FP/(TP+FP)) = 1 - 1/4 = 0.75
    # (1 - L/Lmax) = 1 - 1.5/2 = 0.25
    # → 0.75 * 0.75 * 0.25 = 0.140625
    cds = composite_detection_score(3, 3, 1, 1, latency_p99=1.5, latency_max=2.0)
    expected = 0.75 * 0.75 * 0.25
    if abs(cds - expected) > 1e-12:
        print(f"[FAIL] composite_detection_score: {cds} (expected {expected})")
        rc = 1
    else:
        print(f"[OK]   composite_detection_score = {cds:.6f}")

    # --- Per-protocol aggregation ---
    agg = aggregate_per_protocol({
        "doh": {"accuracy": 0.996, "fpr": 0.0022, "p99": 1.7},
        "dot": {"accuracy": 0.994, "fpr": 0.0025, "p99": 1.9},
        "doq": {"accuracy": 0.997, "fpr": 0.0018, "p99": 0.8},
    })
    if agg["best_protocol"]["accuracy"] != "doq":
        print(f"[FAIL] aggregate accuracy best: {agg['best_protocol'].get('accuracy')}")
        rc = 1
    elif agg["best_protocol"]["fpr"] != "doq":
        print(f"[FAIL] aggregate fpr best (lower is better): {agg['best_protocol'].get('fpr')}")
        rc = 1
    elif agg["best_protocol"]["p99"] != "doq":
        print(f"[FAIL] aggregate p99 best: {agg['best_protocol'].get('p99')}")
        rc = 1
    else:
        print(f"[OK]   aggregate_per_protocol (best on all three: doq)")

    # --- Brier score ---
    bs = brier_score([1, 0, 1, 0], [0.9, 0.1, 0.8, 0.2])
    # (0.1^2 + 0.1^2 + 0.2^2 + 0.2^2) / 4 = (0.01 + 0.01 + 0.04 + 0.04)/4 = 0.10 / 4 = 0.025
    if abs(bs - 0.025) > 1e-12:
        print(f"[FAIL] brier_score: {bs}")
        rc = 1
    else:
        print(f"[OK]   brier_score = {bs:.6f}")

    # --- round_for_reporting ---
    if round_for_reporting(0.123456) != 0.1235:
        print(f"[FAIL] round_for_reporting: {round_for_reporting(0.123456)}")
        rc = 1
    else:
        print(f"[OK]   round_for_reporting = {round_for_reporting(0.123456)}")

    print("=== self-test complete ===")
    return rc


if __name__ == "__main__":
    import sys
    sys.exit(_self_test())
