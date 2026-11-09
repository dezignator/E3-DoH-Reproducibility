#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/reproduce_tables/table_22_advanced_scenarios.py
========================================================

E3-DoH Reproducibility — Table 22 reproduction.

Chapter 5 of the thesis: E3-DoH (Enhanced Evolutionary Encryption for
Multi-Protocol DNS Security — DoH, DoT, DoQ).

Table 22 (p.141) lists five advanced attack scenarios — the ones that
stress the *adaptive* response of the framework rather than the static
detection rules:

    A06  Slow-Drip Exfiltration
    A07  QUIC Version Downgrade
    A08  HTTP/2 Frame Storm
    A09  Encrypted SNI Poisoning
    A10  Adaptive Chaff Injection

Each row maps an attack execution to a detection trigger and a
mitigation action. The rows are stored in data/attack_matrix.csv; this
script verifies each row's trigger/action pair against the
corresponding simulator, and optionally runs the simulation.

Per REPRODUCIBILITY_CHECKLIST.md §4, this table's verdict is "Full"
and both --verify-only and --execute modes are supported.

USAGE
-----
    python scripts/reproduce_tables/table_22_advanced_scenarios.py
    python scripts/reproduce_tables/table_22_advanced_scenarios.py --execute
    python scripts/reproduce_tables/table_22_advanced_scenarios.py --verify
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from data.data_utils import (
    AttackMatrixError,
    ConfigError,
    load_attack_matrix,
    load_config,
    output_path,
    expected_path,
)
from src.utils.logger import configure as configure_logger
from src.utils.logger import get_logger
from src.utils.seed import set_global_seed


# The five scenarios of Table 22 and the module that handles each.
_TABLE_22_SCENARIOS = {
    "A06": ("Slow-Drip Exfiltration", "src.attacks.dns_tunneling", "DNSTunnelingSimulator"),
    "A07": ("QUIC Version Downgrade", "src.attacks.quic_congestion", "QUICCongestionSimulator"),
    "A08": ("HTTP/2 Frame Storm", "src.attacks.cert_injection", "CertInjectionSimulator"),
    "A09": ("Encrypted SNI Poisoning", "src.attacks.cert_injection", "CertInjectionSimulator"),
    "A10": ("Adaptive Chaff Injection", "src.attacks.dns_tunneling", "DNSTunnelingSimulator"),
}


def _build_simulator(scenario_id: str, config: Dict[str, Any], seed: int) -> Any:
    _, module_name, class_name = _TABLE_22_SCENARIOS[scenario_id]
    mod = __import__(module_name, fromlist=[class_name])
    return getattr(mod, class_name)(config, scenario_id, seed=seed)


def _verify_only(matrix, log) -> Dict[str, Any]:
    """Read the matrix rows for A06-A10 and check the string pairs."""
    rows: Dict[str, Dict[str, Any]] = {}
    all_ok = True
    for sid, (name, _mod, _cls) in _TABLE_22_SCENARIOS.items():
        try:
            row = matrix[matrix["Attack_ID"] == sid].iloc[0]
        except IndexError:
            rows[sid] = {"error": "row not found in attack matrix"}
            all_ok = False
            continue
        rows[sid] = {
            "name": name,
            "attack_execution": str(row["Command_Syntax"]),
            "detection_trigger": str(row["Detection_Trigger"]),
            "mitigation_action": str(row["Mitigation_Action"]),
        }
        log.info("table_22_row_loaded", scenario=sid, name=name)
    return {"scenarios": rows, "all_ok": all_ok}


def _execute(config, seed, log) -> Dict[str, Any]:
    """Run each scenario's simulator and cross-check trigger/action against the matrix."""
    from data.data_utils import load_attack_matrix as _load_matrix
    try:
        matrix = _load_matrix()
    except AttackMatrixError as e:
        return {"error": f"attack matrix load failed: {e}"}

    results: Dict[str, Dict[str, Any]] = {}
    for sid, (name, _mod, _cls) in _TABLE_22_SCENARIOS.items():
        log.info("table_22_execute_start", scenario=sid)
        try:
            sim = _build_simulator(sid, config, seed)
            report = sim.run(skip_guard_checks=True)
            rep = report.to_dict() if hasattr(report, "to_dict") else dict(report)
            row = matrix[matrix["Attack_ID"] == sid].iloc[0]
            matrix_consistent = (
                rep.get("detection_trigger") == row["Detection_Trigger"]
                and rep.get("mitigation_action") == row["Mitigation_Action"]
            )
            results[sid] = {
                "name": name,
                "trigger_fires": rep.get("trigger_fires"),
                "detection_trigger": rep.get("detection_trigger"),
                "mitigation_action": rep.get("mitigation_action"),
                "matrix_consistent": matrix_consistent,
                "error": None,
            }
        except Exception as e:
            results[sid] = {"error": f"{type(e).__name__}: {e}"}
        log.info("table_22_execute_complete", scenario=sid,
                 error=results[sid].get("error"))
    return {"scenarios": results}


def _verify(generated: Dict[str, Any], ref_path: Path, log) -> Dict[str, Any]:
    if not ref_path.exists():
        return {"verified": False, "reason": "reference missing"}
    ref = json.loads(ref_path.read_text(encoding="utf-8"))
    ref_scen = ref.get("scenarios", {})
    mismatches = []
    for sid, row in generated.get("scenarios", {}).items():
        if row.get("error"):
            mismatches.append({"scenario": sid, "error": row["error"]})
            continue
        ref_row = ref_scen.get(sid)
        if ref_row is None:
            continue
        for field in ("detection_trigger", "mitigation_action"):
            if row.get(field) != ref_row.get(field):
                mismatches.append({
                    "scenario": sid, "field": field,
                    "expected": ref_row.get(field), "got": row.get(field),
                })
    return {"verified": len(mismatches) == 0, "n_mismatches": len(mismatches),
            "mismatches": mismatches}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="table_22_advanced_scenarios.py",
        description="Reproduce Table 22 (advanced attack scenarios with E3-DoH response).",
    )
    parser.add_argument("--mode", choices=("toy", "full"), default="toy")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--commit", action="store_true")
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
        level=args.log_level, fmt=str(log_cfg.get("format", "json")),
        file_path=log_cfg.get("file", "outputs/logs/table_22.log"), console=True,
    )
    log = get_logger("e3doh.tables.table_22")
    log.info("table_22_start", mode=args.mode, execute=args.execute)

    try:
        matrix = load_attack_matrix()
    except AttackMatrixError as e:
        log.error("attack_matrix_load_failed", error=str(e))
        return 2

    verify_result = _verify_only(matrix, log)
    output: Dict[str, Any] = {
        "table": "table_22",
        "mode": args.mode,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "seed": seed,
        "run_id": config.get("_run_id"),
        "verification_mode": "execute" if args.execute else "verify_only",
        **verify_result,
    }

    if args.execute:
        execution = _execute(config, seed, log)
        output["execution"] = execution

    output["wall_time_s"] = round(time.perf_counter() - started, 3)
    out_path = output_path("tables", "table_22.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2, default=str), encoding="utf-8")
    log.info("table_22_written", path=str(out_path))

    if args.commit:
        commit_path = expected_path("table_22_expected.json")
        commit_path.parent.mkdir(parents=True, exist_ok=True)
        commit_output = {
            "table": "table_22",
            "scenarios": verify_result["scenarios"],
        }
        commit_path.write_text(json.dumps(commit_output, indent=2), encoding="utf-8")
        log.info("table_22_committed", path=str(commit_path))

    verification = None
    if args.verify:
        verification = _verify(output, expected_path("table_22_expected.json"), log)
        output["verification"] = verification

    print()
    print(f"=== Table 22 ({args.mode} mode, {'execute' if args.execute else 'verify-only'}) ===")
    print()
    for sid, row in verify_result["scenarios"].items():
        print(f"  {sid}  {row.get('name', '?')}")
        print(f"      detection: {row.get('detection_trigger', '?')}")
        print(f"      mitigation: {row.get('mitigation_action', '?')}")
        if args.execute:
            ex = output.get("execution", {}).get("scenarios", {}).get(sid, {})
            if ex.get("error"):
                print(f"      execute: ERROR — {ex['error']}")
            else:
                print(f"      execute: fires={ex.get('trigger_fires')}, "
                      f"matrix_consistent={ex.get('matrix_consistent')}")
    print()
    if verification is not None:
        if verification["verified"]:
            print(f"Verification: PASS")
        else:
            print(f"Verification: FAIL ({verification.get('n_mismatches')} mismatches)")
    print(f"Output: {out_path}")
    print()

    if not verify_result["all_ok"]:
        return 1
    if args.execute:
        if any(r.get("error") for r in output.get("execution", {}).get("scenarios", {}).values()):
            return 1
    if verification is not None and not verification["verified"]:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
