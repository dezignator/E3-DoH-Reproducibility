#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/evaluate_e3.py
======================

E3-DoH Reproducibility — standalone evaluation harness.

Chapter 5 of the thesis: E3-DoH (Enhanced Evolutionary Encryption for
Multi-Protocol DNS Security — DoH, DoT, DoQ).

`train_e3.py` runs the full pipeline (adapters + engines + detector)
once and writes a summary. `evaluate_e3.py` is the *evaluation-only*
counterpart: it loads a feature store, builds the detector + adapter,
and produces a per-window and per-flow evaluation report without
training anything. It is useful in three scenarios:

    1. A reviewer who has already run `train_e3.py` and wants to
       re-score the same feature store with a different detector
       configuration (e.g. a different sigmoid scale).

    2. A reviewer who wants a faster evaluation than the full
       `train_e3.py` run: evaluation is ~10x faster than training
       because it skips the AEE evolution and the FL rounds.

    3. A reviewer who wants to compare the detector's performance
       under two different protocols in the same session, using the
       same feature-store provenance and the same seed.

It does not write a training summary; it writes an *evaluation
report* (`outputs/tables/evaluation_report.json`) with the same
per-protocol structure as `train_e3.py`'s summary but without the
AEE/FL sections.

USAGE
-----
    # Evaluate all three protocols on the toy corpus
    python scripts/evaluate_e3.py --mode toy

    # Evaluate DoH only, with a custom sigmoid scale
    python scripts/evaluate_e3.py --mode toy --protocol doh --scale 0.10

    # Evaluate all protocols with per-point metric (slower, more
    # accurate; documented approximation in the anomaly_detector)
    python scripts/evaluate_e3.py --mode toy --per-point-metric

    # Write to a custom output path
    python scripts/evaluate_e3.py --mode toy --output outputs/tables/my_eval.json

Exit codes match `train_e3.py`.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from data.data_utils import (
    ConfigError,
    FeatureStoreSchemaError,
    ProvenanceError,
    PROTOCOLS,
    FEATURE_COLUMNS,
    load_config,
    load_feature_store,
)
from src.utils.logger import configure as configure_logger
from src.utils.logger import get_logger
from src.utils.seed import set_global_seed
from src.utils.metrics import (
    confusion_counts,
    detection_accuracy as compute_accuracy,
    false_positive_rate as compute_fpr,
    true_positive_rate as compute_tpr,
    precision as compute_precision,
    f1_score as compute_f1,
    brier_score as compute_brier,
    latency_percentiles,
)


_DEFAULT_OUTPUT = _REPO_ROOT / "outputs" / "tables" / "evaluation_report.json"
_WINDOW_SIZE = 50
_AEE_TRAIN_WINDOWS_MAX = 100


def _classify_window(labels: np.ndarray) -> str:
    return "attack" if np.any(labels == "attack") else "benign"


def _slice(features: np.ndarray, labels: np.ndarray, ws: int) -> List[Dict[str, Any]]:
    out = []
    n = len(features)
    for s in range(0, n - ws + 1, ws):
        w_feat = features[s:s+ws]
        w_labels = labels[s:s+ws]
        out.append({
            "features": w_feat,
            "labels": w_labels,
            "window_label": _classify_window(w_labels),
            "start_index": s,
        })
    return out


def _evaluate_protocol(
    protocol: str,
    config: Dict[str, Any],
    mode: str,
    *,
    scale: Optional[float],
    per_point_metric: bool,
    log: Any,
) -> Dict[str, Any]:
    """Evaluate the detector on one protocol's feature store."""
    result: Dict[str, Any] = {"protocol": protocol, "error": None}
    started = time.perf_counter()

    # Load features.
    try:
        require = "synthetic" if mode == "toy" else "DS-1"
        df = load_feature_store(protocol, require_provenance=require)
        log.info("features_loaded", protocol=protocol, rows=len(df))
    except (FeatureStoreSchemaError, ProvenanceError) as e:
        result["error"] = f"feature_store: {e}"
        return result

    features = df[list(FEATURE_COLUMNS)].to_numpy(dtype=np.float64)
    labels = df["label"].to_numpy()

    # Build adapter and detector.
    try:
        from src.protocols.base_adapter import get_adapter
        from src.models.anomaly_detector import AnomalyDetector
        adapter = get_adapter(protocol, config)
        detector = AnomalyDetector(
            config,
            scale=scale,
            per_point_metric=per_point_metric,
            apply_low_pass=True,
        )
    except Exception as e:
        result["error"] = f"build: {e}"
        log.error("build_failed", protocol=protocol, error=repr(e))
        return result

    # Slice and evaluate.
    windows = _slice(features, labels, _WINDOW_SIZE)
    if not windows:
        result["error"] = "no_windows"
        return result

    tp = tn = fp = fn = 0
    flow_tp = flow_tn = flow_fp = flow_fn = 0
    latencies_us: List[float] = []
    scores: List[float] = []
    predicted_probs: List[float] = []
    true_labels_bin: List[int] = []
    n_scored = 0
    n_skipped = 0

    for w in windows:
        try:
            t0 = time.perf_counter()
            r = detector.score_window(adapter, w["features"], protocol)
            latencies_us.append((time.perf_counter() - t0) * 1e6)
        except Exception as e:
            n_skipped += 1
            log.debug("score_failed", protocol=protocol, error=repr(e))
            continue

        if r.skipped_reason is not None:
            n_skipped += 1
            continue

        n_scored += 1
        scores.append(r.score)
        # For Brier, treat the score as the predicted probability of attack.
        predicted_probs.append(r.score)
        true_labels_bin.append(1 if w["window_label"] == "attack" else 0)

        is_attack = (w["window_label"] == "attack")
        is_flagged = (r.decision.value == "anomaly")
        if is_attack and is_flagged:
            tp += 1
        elif not is_attack and is_flagged:
            fp += 1
        elif is_attack and not is_flagged:
            fn += 1
        else:
            tn += 1

        n_attack = int(np.sum(w["labels"] == "attack"))
        n_benign = len(w["labels"]) - n_attack
        if is_flagged:
            flow_tp += n_attack
            flow_fp += n_benign
        else:
            flow_fn += n_attack
            flow_tn += n_benign

    # Aggregate metrics.
    metrics = {
        "n_windows": len(windows),
        "n_windows_scored": n_scored,
        "n_windows_skipped": n_skipped,
        "confusion": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
        "flow_confusion": {"tp": flow_tp, "tn": flow_tn, "fp": flow_fp, "fn": flow_fn},
        "detection_accuracy": compute_accuracy(tp, tn, fp, fn),
        "false_positive_rate": compute_fpr(tp, tn, fp, fn),
        "true_positive_rate": compute_tpr(tp, tn, fp, fn),
        "precision": compute_precision(tp, tn, fp, fn),
        "f1_score": compute_f1(tp, tn, fp, fn),
        "flow_accuracy": compute_accuracy(flow_tp, flow_tn, flow_fp, flow_fn),
        "flow_false_positive_rate": compute_fpr(flow_tp, flow_tn, flow_fp, flow_fn),
        "flow_true_positive_rate": compute_tpr(flow_tp, flow_tn, flow_fp, flow_fn),
        "brier_score": (
            compute_brier(true_labels_bin, predicted_probs)
            if predicted_probs else float("nan")
        ),
        "mean_anomaly_score": float(np.mean(scores)) if scores else float("nan"),
        "max_anomaly_score": float(np.max(scores)) if scores else float("nan"),
    }
    if latencies_us:
        lp = latency_percentiles(latencies_us, unit="us")
        metrics["latency_p50_ms"] = lp.get("p50", float("nan"))
        metrics["latency_p95_ms"] = lp.get("p95", float("nan"))
        metrics["latency_p99_ms"] = lp.get("p99", float("nan"))
    else:
        metrics["latency_p50_ms"] = float("nan")
        metrics["latency_p95_ms"] = float("nan")
        metrics["latency_p99_ms"] = float("nan")

    result["evaluation"] = metrics
    result["wall_time_s"] = round(time.perf_counter() - started, 3)
    log.info("protocol_evaluated", protocol=protocol, **{k: v for k, v in metrics.items() if isinstance(v, (int, float))})
    return result


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="evaluate_e3.py",
        description="E3-DoH standalone evaluation harness.",
    )
    p.add_argument("--mode", choices=("toy", "full"), default="toy")
    p.add_argument("--protocol", choices=list(PROTOCOLS) + ["all"], default="all")
    p.add_argument("--scale", type=float, default=None,
                   help="Override the sigmoid scale (default: 0.05).")
    p.add_argument("--per-point-metric", action="store_true",
                   help="Use the per-point metric tensor computation (slower, more accurate).")
    p.add_argument("--output", type=str, default=str(_DEFAULT_OUTPUT))
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--log-level", choices=("DEBUG", "INFO", "WARNING", "ERROR"), default="INFO")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    started = time.perf_counter()

    try:
        config = load_config(config_path=args.config)
    except ConfigError as e:
        print(f"Config error: {e}", file=sys.stderr)
        return 2
    config["runtime"]["mode"] = args.mode

    set_global_seed(int(config.get("runtime", {}).get("seed", 42)))

    log_cfg = config.get("logging", {}) or {}
    configure_logger(
        level=args.log_level,
        fmt=str(log_cfg.get("format", "json")),
        file_path=log_cfg.get("file", "outputs/logs/evaluate.log"),
        console=True,
    )
    log = get_logger("e3doh.evaluate")
    log.info("evaluate_start", mode=args.mode, protocol=args.protocol,
             scale=args.scale, per_point_metric=args.per_point_metric)

    protocols = list(PROTOCOLS) if args.protocol == "all" else [args.protocol]
    results: Dict[str, Dict[str, Any]] = {}
    for protocol in protocols:
        log.info("evaluate_protocol", protocol=protocol)
        try:
            r = _evaluate_protocol(
                protocol, config, args.mode,
                scale=args.scale, per_point_metric=args.per_point_metric, log=log,
            )
            results[protocol] = r
            if r.get("error"):
                log.warning("evaluate_failed", protocol=protocol, error=r["error"])
        except Exception as e:
            log.error("evaluate_exception", protocol=protocol, error=repr(e))
            results[protocol] = {"protocol": protocol, "error": repr(e)}

    # Aggregate.
    accs = [r["evaluation"]["detection_accuracy"] for r in results.values()
            if r.get("evaluation") and not math.isnan(r["evaluation"]["detection_accuracy"])]
    fprs = [r["evaluation"]["false_positive_rate"] for r in results.values()
            if r.get("evaluation") and not math.isnan(r["evaluation"]["false_positive_rate"])]
    p99s = [r["evaluation"]["latency_p99_ms"] for r in results.values()
            if r.get("evaluation") and not math.isnan(r["evaluation"]["latency_p99_ms"])]
    completed = sum(1 for r in results.values() if r.get("error") is None)

    aggregate = {
        "protocols_completed": completed,
        "detection_accuracy_mean": float(np.mean(accs)) if accs else None,
        "false_positive_rate_mean": float(np.mean(fprs)) if fprs else None,
        "latency_p99_max_ms": float(np.max(p99s)) if p99s else None,
    }

    elapsed = round(time.perf_counter() - started, 3)
    report = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "mode": args.mode,
        "seed": config.get("runtime", {}).get("seed", 42),
        "run_id": config.get("_run_id"),
        "wall_time_s": elapsed,
        "per_point_metric": args.per_point_metric,
        "scale": args.scale,
        "protocols": results,
        "aggregate": aggregate,
    }

    output = Path(args.output)
    if not output.is_absolute():
        output = _REPO_ROOT / output
    try:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
        log.info("evaluation_written", path=str(output))
    except OSError as e:
        log.error("evaluation_write_failed", error=str(e))
        return 1

    print()
    print(f"=== E3-DoH evaluation complete ({args.mode}) ===")
    print(f"Protocols completed: {completed}/{len(protocols)}")
    if aggregate["detection_accuracy_mean"] is not None:
        print(f"Mean accuracy: {aggregate['detection_accuracy_mean']:.4f}")
    if aggregate["false_positive_rate_mean"] is not None:
        print(f"Mean FPR: {aggregate['false_positive_rate_mean']:.6f}")
    if aggregate["latency_p99_max_ms"] is not None:
        print(f"Max P99 latency: {aggregate['latency_p99_max_ms']:.3f} ms")
    print(f"Wall time: {elapsed:.1f} s")
    print(f"Report: {output}")
    print()

    return 0 if completed == len(protocols) else 1


if __name__ == "__main__":
    sys.exit(main())
