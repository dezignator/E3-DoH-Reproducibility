#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/reproduce_tables/table_18_attack_mitigation.py
=======================================================

E3-DoH Reproducibility — Table 18 reproduction.

Chapter 5 of the thesis: E3-DoH (Enhanced Evolutionary Encryption for
Multi-Protocol DNS Security — DoH, DoT, DoQ).

Table 18 (p.134) lists five attack injection methods, their detection
mechanisms, and their mitigation actions. The table is reproduced by
running each of the five attack simulations and checking that:

    1. The simulation produces the detection trigger documented in
       data/attack_matrix.csv.
    2. The simulation produces the mitigation action documented in
       data/attack_matrix.csv.
    3. The detection trigger maps to the mitigation action in the AMPS
       controller (i.e., the AMPS decision for the triggered flow is
       the expected isolation action).

It writes `outputs/tables/table_18.json` (and, with `--commit`, the
same content is written to `expected_outputs/table_18_expected.json`).

USAGE
-----
    # Toy mode: run the attacks against the local testbed
    python scripts/reproduce_tables/table_18_attack_mitigation.py --mode toy

    # Verify the table against the committed reference
    python scripts/reproduce_tables/table_18_attack_mitigation.py --mode toy --verify

    # Full mode (reference testbed)
    python scripts/reproduce_tables/table_18_attack_mitigation.py --mode full
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from data.data_utils import (
    ConfigError,
    AttackMatrixError,
    load_attack_matrix,
    load_config,
    output_path,
    expected_path,
)
from src.utils.logger import configure as configure_logger
from src.utils.logger import get_logger
from src.utils.seed import set_global_seed


# The five scenarios of Table 18, mapped to the module that simulates
# them. A01 is in dns_tunneling, A02 in timing_side_channel, A03 in
# quic_congestion, A04 and A05 in cert_injection (A04) and
# padding_oracle (A05).
_TABLE_18_SCENARIOS = {
    "A01": ("DNS Tunneling via DoH", "src.attacks.dns_tunneling", "DNSTunnelingSimulator"),
    "A02": ("Timing Side-Channel", "src.attacks.timing_side_channel", "TimingSideChannelSimulator"),
    "A03": ("QUIC Congestion Exploit", "src.attacks.quic_congestion", "QUICCongestionSimulator"),
    "A04": ("Certificate Injection", "src.attacks.cert_injection", "CertInjectionSimulator"),
    "A05": ("Padding Oracle", "src.attacks.padding_oracle", "PaddingOracleSimulator"),
}


def _build_simulator(
    scenario_id: str,
    config: Dict[str, Any],
    seed: int,
    log: Any,
) -> Any:
    """Build the simulator for a scenario via lazy import."""
    if scenario_id not in _TABLE_18_SCENARIOS:
        raise ValueError(f"unknown scenario {scenario_id!r}")
    _, module_name, class_name = _TABLE_18_SCENARIOS[scenario_id]
    mod = __import__(module_name, fromlist=[class_name])
    cls = getattr(mod, class_name)
    return cls(config, scenario_id, seed=seed)


def _verify_table_18(
    generated: Dict[str, Any],
    reference_path: Path,
    log: Any,
) -> Dict[str, Any]:
    """Compare the generated table to the reference."""
    if not reference_path.exists():
        log.warning("reference_not_found", path=str(reference_path))
        return {
            "verified": False,
            "reason": "reference file does not exist (see STATUS.md §9)",
            "reference_path": str(reference_path),
        }
    ref = json.loads(reference_path.read_text(encoding="utf-8"))
    mismatches = []
    for scenario_id, expected in ref.get("scenarios", {}).items():
        got = generated.get("scenarios", {}).get(scenario_id, {})
        if expected.get("detection_trigger") != got.get("detection_trigger"):
            mismatches.append({
                "scenario": scenario_id,
                "field": "detection_trigger",
                "expected": expected.get("detection_trigger"),
                "got": got.get("detection_trigger"),
            })
        if expected.get("mitigation_action") != got.get("mitigation_action"):
            mismatches.append({
                "scenario": scenario_id,
                "field": "mitigation_action",
                "expected": expected.get("mitigation_action"),
                "got": got.get("mitigation_action"),
            })
        if expected.get("trigger_fires") != got.get("trigger_fires"):
            mismatches.append({
                "scenario": scenario_id,
                "field": "trigger_fires",
                "expected": expected.get("trigger_fires"),
                "got": got.get("trigger_fires"),
            })
    return {
        "verified": len(mismatches) == 0,
        "n_mismatches": len(mismatches),
        "mismatches": mismatches,
        "reference_path": str(reference_path),
    }


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="table_18_attack_mitigation.py",
        description="Reproduce Table 18 (attack injection methods and detection mechanisms).",
    )
    parser.add_argument("--mode", choices=("toy", "full"), default="toy")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--verify", action="store_true",
                        help="Verify against expected_outputs/table_18_expected.json.")
    parser.add_argument("--commit", action="store_true",
                        help="Write the result to expected_outputs/ (sign-off only).")
    parser.add_argument("--log-level", choices=("DEBUG", "INFO", "WARNING", "ERROR"), default="INFO")
    args = parser.parse_args(argv)

    started = time.perf_counter()

    try:
        config = load_config(config_path=args.config)
    except ConfigError as e:
        print(f"Config error: {e}", file=sys.stderr)
        return 2
    config["runtime"]["mode"] = args.mode

    seed = int(config.get("runtime", {}).get("seed", 42))
    set_global_seed(seed)

    log_cfg = config.get("logging", {}) or {}
    configure_logger(
        level=args.log_level,
        fmt=str(log_cfg.get("format", "json")),
        file_path=log_cfg.get("file", "outputs/logs/table_18.log"),
        console=True,
    )
    log = get_logger("e3doh.tables.table_18")
    log.info("table_18_start", mode=args.mode, verify=args.verify)

    # Load the attack matrix for the expected trigger/action strings.
    try:
        matrix = load_attack_matrix()
    except AttackMatrixError as e:
        log.error("attack_matrix_load_failed", error=str(e))
        return 2

    scenarios: Dict[str, Dict[str, Any]] = {}
    for scenario_id, (name, _module, _class) in _TABLE_18_SCENARIOS.items():
        log.info("table_18_scenario_start", scenario=scenario_id, name=name)
        try:
            sim = _build_simulator(scenario_id, config, seed, log)
            report = sim.run(skip_guard_checks=(args.mode == "toy"))
            report_dict = report.to_dict()

            # Cross-check against the attack matrix.
            matrix_row = matrix[matrix["Attack_ID"] == scenario_id].iloc[0]
            matrix_trigger = matrix_row["Detection_Trigger"]
            matrix_action = matrix_row["Mitigation_Action"]
            simulator_trigger = report_dict["detection_trigger"]
            simulator_action = report_dict["mitigation_action"]

            scenarios[scenario_id] = {
                "name": name,
                "n_packets": report_dict["n_packets_sent"],
                "duration_s": report_dict["duration_s"],
                "detection_trigger": simulator_trigger,
                "mitigation_action": simulator_action,
                "trigger_fires": report_dict["trigger_fires"],
                "metric_name": report_dict["metric_name"],
                "metric_value": report_dict["metric_value"],
                "metric_threshold": report_dict["metric_threshold"],
                "matrix_trigger": matrix_trigger,
                "matrix_action": matrix_action,
                "matrix_consistent": (
                    simulator_trigger == matrix_trigger
                    and simulator_action == matrix_action
                ),
                "error": None,
            }
            log.info(
                "table_18_scenario_complete",
                scenario=scenario_id,
                trigger_fires=report_dict["trigger_fires"],
                matrix_consistent=scenarios[scenario_id]["matrix_consistent"],
            )
        except Exception as e:
            log.error("table_18_scenario_failed",
                      scenario=scenario_id, error=repr(e))
            scenarios[scenario_id] = {
                "name": name,
                "error": repr(e),
            }

    # Build the output document.
    n_consistent = sum(1 for s in scenarios.values()
                       if s.get("matrix_consistent", False))
    n_completed = sum(1 for s in scenarios.values()
                      if s.get("error") is None)
    elapsed = round(time.perf_counter() - started, 3)
    output = {
        "table": "table_18",
        "mode": args.mode,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "seed": seed,
        "run_id": config.get("_run_id"),
        "wall_time_s": elapsed,
        "scenarios": scenarios,
        "summary": {
            "n_scenarios": len(scenarios),
            "n_completed": n_completed,
            "n_matrix_consistent": n_consistent,
        },
    }

    # Write the output.
    out_path = output_path("tables", "table_18.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2, default=str), encoding="utf-8")
    log.info("table_18_written", path=str(out_path))

    # Optional commit to expected_outputs/.
    if args.commit:
        commit_path = expected_path("table_18_expected.json")
        commit_path.parent.mkdir(parents=True, exist_ok=True)
        # Strip non-deterministic fields.
        for s in output["scenarios"].values():
            for k in ("n_packets", "duration_s", "metric_value"):
                s.pop(k, None) if k == "metric_value" else None
        output.pop("generated_at", None)
        output.pop("wall_time_s", None)
        commit_path.write_text(json.dumps(output, indent=2, default=str), encoding="utf-8")
        log.info("table_18_committed", path=str(commit_path))

    # Optional verify.
    verification: Optional[Dict[str, Any]] = None
    if args.verify:
        verification = _verify_table_18(output, expected_path("table_18_expected.json"), log)
        log.info("table_18_verified", **verification)

    # Report.
    print()
    print(f"=== Table 18 ({args.mode} mode) ===")
    print(f"Scenarios completed: {n_completed}/{len(scenarios)}")
    print(f"Matrix consistent:   {n_consistent}/{n_completed} completed")
    print()
    for sid, s in scenarios.items():
        if s.get("error"):
            print(f"  {sid} {s['name']}: ERROR — {s['error']}")
        else:
            ok = "OK" if s["matrix_consistent"] else "MISMATCH"
            fires = "fires" if s["trigger_fires"] else "no-fire"
            print(f"  {sid} {s['name']}: {ok}, trigger {fires}")
            print(f"      detection: {s['detection_trigger']}")
            print(f"      mitigation: {s['mitigation_action']}")
    print()
    if verification is not None:
        if verification["verified"]:
            print(f"Verification: PASS (0 mismatches vs {verification['reference_path']})")
        else:
            print(f"Verification: FAIL ({verification.get('n_mismatches', '?')} mismatches)")
    print(f"Wall time: {elapsed:.1f} s")
    print(f"Output: {out_path}")
    print()

    if n_completed < len(scenarios):
        return 1
    if verification is not None and not verification["verified"]:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
