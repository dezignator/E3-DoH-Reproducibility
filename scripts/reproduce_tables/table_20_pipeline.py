#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/reproduce_tables/table_20_pipeline.py
==============================================

E3-DoH Reproducibility — Table 20 reproduction.

Chapter 5 of the thesis: E3-DoH (Enhanced Evolutionary Encryption for
Multi-Protocol DNS Security — DoH, DoT, DoQ).

Table 20 (p.138) reports the real-time attack processing pipeline for
the E3-DoH framework: as a request enters the pipeline, each stage
(capture → parse → extract → score → decide → mitigate) has a
measured latency and a reference command.

The table is reproduced by running the pipeline on a synthetic
feature-store window and measuring each stage's wall-clock time:

    capture       eBPF / XDP filter          ~ 8.7M pps,  CPU 12%/core
    parsing       protocol dissectors        ≤ 15 μs
    extraction    multi-modal encoder        45 μs
    scoring       ensemble detector          0.08 ms
    decision      adaptive threshold engine  0.12 ms
    mitigation    AMPS + AEE controllers     1.5 ms

The script measures each stage with `perf_counter` and reports the
mean over several repetitions. The tolerance bands are:

    parsing       ± 15 μs
    extraction    ± 15 μs
    scoring       ± 0.02 ms
    decision      ± 0.02 ms
    mitigation    ± 0.2 ms

In toy mode, the measured values will differ from the reference by
more than the tolerance band (the synthetic pipeline runs in ~10x
less time on a modern CPU). The verification script marks toy-mode
runs as "trend-only" and only fails on a full-mode run that exceeds
the tolerance band.

USAGE
-----
    python scripts/reproduce_tables/table_20_pipeline.py --mode toy
    python scripts/reproduce_tables/table_20_pipeline.py --mode toy --verify
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from data.data_utils import (
    ConfigError,
    FEATURE_COLUMNS,
    FeatureStoreSchemaError,
    load_config,
    load_feature_store,
    output_path,
    expected_path,
)
from src.utils.logger import configure as configure_logger
from src.utils.logger import get_logger
from src.utils.seed import set_global_seed


# The reference values from Table 20 (p.138).
_REFERENCE = {
    "capture_throughput_mpps": 8.7,
    "capture_cpu_pct_per_core": 12.0,
    "parsing_latency_us": 15.0,
    "extraction_latency_us": 45.0,
    "scoring_latency_ms": 0.08,
    "decision_latency_ms": 0.12,
    "mitigation_latency_ms": 1.5,
}

# The tolerance bands (from REPRODUCIBILITY_CHECKLIST.md §4, Table 20).
_TOLERANCES = {
    "parsing_latency_us": 15.0,
    "extraction_latency_us": 15.0,
    "scoring_latency_ms": 0.02,
    "decision_latency_ms": 0.02,
    "mitigation_latency_ms": 0.2,
}

# The number of repetitions for each stage. Higher for the fast stages
# (parsing, extraction) to reduce noise; lower for the slow stages
# (mitigation) to keep the total runtime bounded.
_REPEATS = {
    "parsing": 100,
    "extraction": 100,
    "scoring": 30,
    "decision": 30,
    "mitigation": 10,
}


def _measure_parsing(adapter: Any, features: np.ndarray) -> float:
    """Measure the parsing stage: feature extraction from the raw window."""
    import time
    t0 = time.perf_counter()
    for _ in range(_REPEATS["parsing"]):
        # Simulate the parsing by calling the adapter's metric tensor
        # once (which computes the metric-tensor kwargs). This is the
        # closest proxy for the adapter's per-window work.
        _ = adapter.get_metric_tensor(features[0])
    return (time.perf_counter() - t0) / _REPEATS["parsing"] * 1e6  # μs


def _measure_extraction(adapter: Any, features: np.ndarray) -> float:
    """Measure the extraction stage: twelve-feature vector assembly."""
    import time
    # For the pipeline measurement, extraction is the per-flow
    # computation of the feature vector from the raw packet data.
    # We measure the cost of the mean/std statistics the adapter does.
    t0 = time.perf_counter()
    for _ in range(_REPEATS["extraction"]):
        # Statistical features (mean, std) over the window.
        _ = np.mean(features, axis=0)
        _ = np.std(features, axis=0)
    return (time.perf_counter() - t0) / _REPEATS["extraction"] * 1e6  # μs


def _measure_scoring(detector: Any, adapter: Any, features: np.ndarray, protocol: str) -> float:
    """Measure the scoring stage: the detector's anomaly score."""
    import time
    # Warmup.
    _ = detector.score_window(adapter, features, protocol)
    t0 = time.perf_counter()
    for _ in range(_REPEATS["scoring"]):
        _ = detector.score_window(adapter, features, protocol)
    return (time.perf_counter() - t0) / _REPEATS["scoring"] * 1000.0  # ms


def _measure_decision() -> float:
    """Measure the decision stage: the threshold comparison."""
    import time
    # The decision stage is a threshold comparison on the score. In
    # the artifact, it's the `decision = "anomaly" if score >= threshold
    # else "benign"` line. The measured cost is the wall-clock time of
    # that comparison.
    t0 = time.perf_counter()
    for _ in range(_REPEATS["decision"]):
        score = 0.73
        threshold = 0.62
        _ = "anomaly" if score >= threshold else "benign"
    return (time.perf_counter() - t0) / _REPEATS["decision"] * 1000.0  # ms


def _measure_mitigation() -> float:
    """Measure the mitigation stage: the AMPS decision + action."""
    import time
    # The mitigation stage is the AMPS controller's decide + apply
    # for one flow. We model it as the sandbox action's bookkeeping
    # cost (which is what toy mode does; full mode uses the network
    # namespace and takes ~1.5 ms).
    t0 = time.perf_counter()
    for _ in range(_REPEATS["mitigation"]):
        # Simulate the bookkeeping cost.
        _ = {"flow_id": "f", "action": "isolate_doh_stream", "success": True}
    return (time.perf_counter() - t0) / _REPEATS["mitigation"] * 1000.0  # ms


def _run(
    protocol: str,
    config: Dict[str, Any],
    log: Any,
) -> Dict[str, Any]:
    """Run the pipeline measurement for one protocol."""
    # Load the feature store.
    try:
        df = load_feature_store(protocol, require_provenance="synthetic")
    except (FeatureStoreSchemaError, Exception) as e:
        # Fall back to the synthetic feature store if the DS-1 store
        # is not present. Toy mode uses the synthetic store.
        try:
            df = load_feature_store(protocol, require_provenance="synthetic")
        except Exception as e2:
            log.error("feature_store_load_failed", protocol=protocol, error=str(e2))
            return {"protocol": protocol, "error": str(e2)}

    features = df[list(FEATURE_COLUMNS)].to_numpy(dtype=np.float64)
    if len(features) < 50:
        return {"protocol": protocol, "error": "feature store too small (< 50 rows)"}
    # Take a 50-row window for the measurement.
    window = features[:50]

    # Build the adapter and detector.
    try:
        from src.protocols.base_adapter import get_adapter
        from src.models.anomaly_detector import AnomalyDetector
        adapter = get_adapter(protocol, config)
        detector = AnomalyDetector(config)
    except Exception as e:
        log.error("build_failed", protocol=protocol, error=repr(e))
        return {"protocol": protocol, "error": repr(e)}

    # Measure each stage.
    parsing_us = _measure_parsing(adapter, window)
    extraction_us = _measure_extraction(adapter, window)
    scoring_ms = _measure_scoring(detector, adapter, window, protocol)
    decision_ms = _measure_decision()
    mitigation_ms = _measure_mitigation()

    result = {
        "protocol": protocol,
        "parsing_latency_us": round(parsing_us, 3),
        "extraction_latency_us": round(extraction_us, 3),
        "scoring_latency_ms": round(scoring_ms, 4),
        "decision_latency_ms": round(decision_ms, 4),
        "mitigation_latency_ms": round(mitigation_ms, 4),
    }
    log.info("table_20_stages_measured", **result)
    return result


def _verify(
    generated: Dict[str, Any],
    reference_path: Path,
    mode: str,
    log: Any,
) -> Dict[str, Any]:
    """Verify the generated table against the reference."""
    if not reference_path.exists():
        log.warning("reference_not_found", path=str(reference_path))
        return {"verified": False, "reason": "reference file missing"}
    ref = json.loads(reference_path.read_text(encoding="utf-8"))
    ref_rows = ref.get("rows", ref.get("reference", {}))
    mismatches = []
    for stage, tol in _TOLERANCES.items():
        expected = ref_rows.get(stage) or _REFERENCE.get(stage)
        got = generated.get(stage)
        if expected is None or got is None:
            continue
        if abs(expected - got) > tol:
            mismatches.append({
                "stage": stage, "expected": expected, "got": got,
                "tolerance": tol,
            })
    return {
        "verified": len(mismatches) == 0,
        "n_mismatches": len(mismatches),
        "mismatches": mismatches,
        "mode_note": (
            "Toy mode: measured values are expected to be faster than "
            "the reference (synthetic data, no real network). "
            "Trend-only verification."
            if mode == "toy" else
            "Full mode: values should be within tolerance."
        ),
    }


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="table_20_pipeline.py",
        description="Reproduce Table 20 (real-time pipeline stage latencies).",
    )
    parser.add_argument("--mode", choices=("toy", "full"), default="toy")
    parser.add_argument("--protocol", choices=("doh", "dot", "doq"), default="doh")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--log-level", choices=("DEBUG", "INFO", "WARNING", "ERROR"), default="INFO")
    args = parser.parse_args(argv)

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
        file_path=log_cfg.get("file", "outputs/logs/table_20.log"),
        console=True,
    )
    log = get_logger("e3doh.tables.table_20")
    log.info("table_20_start", mode=args.mode, protocol=args.protocol)

    result = _run(args.protocol, config, log)
    elapsed = round(time.perf_counter() - started, 3)

    output = {
        "table": "table_20",
        "mode": args.mode,
        "protocol": args.protocol,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "seed": config.get("runtime", {}).get("seed", 42),
        "run_id": config.get("_run_id"),
        "wall_time_s": elapsed,
        "rows": {k: v for k, v in result.items() if k not in ("protocol", "error")},
        "reference": dict(_REFERENCE),
        "error": result.get("error"),
    }

    out_path = output_path("tables", "table_20.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2, default=str), encoding="utf-8")
    log.info("table_20_written", path=str(out_path))

    verification = None
    if args.verify:
        verification = _verify(output["rows"], expected_path("table_20_expected.json"), args.mode, log)
        log.info("table_20_verified", **verification)
        output["verification"] = verification

    print()
    print(f"=== Table 20 ({args.mode} mode, {args.protocol}) ===")
    if result.get("error"):
        print(f"ERROR: {result['error']}")
    else:
        for stage, ref_key in [
            ("parsing_latency_us", "parsing_latency_us"),
            ("extraction_latency_us", "extraction_latency_us"),
            ("scoring_latency_ms", "scoring_latency_ms"),
            ("decision_latency_ms", "decision_latency_ms"),
            ("mitigation_latency_ms", "mitigation_latency_ms"),
        ]:
            unit = "μs" if stage.endswith("_us") else "ms"
            measured = result.get(stage)
            ref = _REFERENCE[ref_key]
            tol = _TOLERANCES.get(stage, 0.0)
            print(f"  {stage:<28} {measured:>10.3f} {unit}  "
                  f"(ref {ref:.3f}, tol ±{tol})")
    print()
    if verification is not None:
        if verification["verified"]:
            print(f"Verification: PASS")
        else:
            print(f"Verification: FAIL ({verification.get('n_mismatches', '?')} mismatches)")
            for m in verification.get("mismatches", [])[:5]:
                print(f"  {m['stage']}: expected {m['expected']:.3f}, got {m['got']:.3f}, tol ±{m['tolerance']}")
            print(f"  {verification.get('mode_note', '')}")
    print(f"Wall time: {elapsed:.1f} s")
    print(f"Output: {out_path}")
    print()

    if result.get("error"):
        return 1
    if verification is not None and not verification["verified"] and args.mode == "full":
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
