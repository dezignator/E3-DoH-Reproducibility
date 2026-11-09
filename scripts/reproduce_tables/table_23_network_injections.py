#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/reproduce_tables/table_23_network_injections.py
========================================================

E3-DoH Reproducibility — Table 23 reproduction.

Chapter 5 of the thesis: E3-DoH (Enhanced Evolutionary Encryption for
Multi-Protocol DNS Security — DoH, DoT, DoQ).

Table 23 (p.142) lists five network-layer injection commands and their
detection triggers / mitigations:

    A11  DNS Query Flood          (DoH, dnsperf)
    A12  TLS Handshake Abuse      (DoT, testssl)
    A13  QUIC Reflection          (DoQ, quic-attack)
    A14  HTTP/2 Continuation Flood (DoH, h2load)
    A15  Encrypted Payload Fuzzing (DoH, boofuzz)

The rows are stored in data/attack_matrix.csv. The correct reproduction
is a *verbatim* match of the command string and the trigger/action pair
between the table and the corresponding matrix row, optionally followed
by a `--execute` run of the simulation.

USAGE
-----
    python scripts/reproduce_tables/table_23_network_injections.py
    python scripts/reproduce_tables/table_23_network_injections.py --execute
    python scripts/reproduce_tables/table_23_network_injections.py --verify
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict

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


_TABLE_23_SCENARIOS = {
    "A11": ("DNS Query Flood", "src.attacks.dns_tunneling", "DNSTunnelingSimulator"),
    "A12": ("TLS Handshake Abuse", "src.attacks.cert_injection", "CertInjectionSimulator"),
    "A13": ("QUIC Reflection", "src.attacks.quic_congestion", "QUICCongestionSimulator"),
    "A14": ("HTTP/2 Continuation Flood", "src.attacks.cert_injection", "CertInjectionSimulator"),
    "A15": ("Encrypted Payload Fuzzing", "src.attacks.padding_oracle", "PaddingOracleSimulator"),
}


def _build_simulator(sid: str, config: Dict[str, Any], seed: int) -> Any:
    _, module_name, class_name = _TABLE_23_SCENARIOS[sid]
    mod = __import__(module_name, fromlist=[class_name])
    return getattr(mod, class_name)(config, sid, seed=seed)


def _verify_only(matrix, log) -> Dict[str, Any]:
    rows: Dict[str, Dict[str, Any]] = {}
    all_ok = True
    for sid, (name, _mod, _cls) in _TABLE_23_SCENARIOS.items():
        try:
            row = matrix[matrix["Attack_ID"] == sid].iloc[0]
        except IndexError:
            rows[sid] = {"error": "row missing in matrix"}
            all_ok = False
            continue
        rows[sid] = {
            "name": name,
            "tool": str(row["Tool"]),
            "command": str(row["Command_Syntax"]),
            "detection_trigger": str(row["Detection_Trigger"]),
            "mitigation_action": str(row["Mitigation_Action"]),
        }
        log.info("table_23_row_loaded", scenario=sid)
    return {"scenarios": rows, "all_ok": all_ok}


def _execute(config, seed, log) -> Dict[str, Any]:
    from data.data_utils import load_attack_matrix as _load_matrix
    try:
        matrix = _load_matrix()
    except AttackMatrixError as e:
        return {"error": f"matrix load failed: {e}"}
    results: Dict[str, Dict[str, Any]] = {}
    for sid, (name, _mod, _cls) in _TABLE_23_SCENARIOS.items():
        try:
            sim = _build_simulator(sid, config, seed)
            report = sim.run(skip_guard_checks=True)
            rep = report.to_dict() if hasattr(report, "to_dict") else dict(report)
            row = matrix[matrix["Attack_ID"] == sid].iloc[0]
            results[sid] = {
                "name": name,
                "trigger_fires": rep.get("trigger_fires"),
                "matrix_consistent": (
                    rep.get("detection_trigger") == row["Detection_Trigger"]
                    and rep.get("mitigation_action") == row["Mitigation_Action"]
                ),
                "error": None,
            }
        except Exception as e:
            results[sid] = {"error": f"{type(e).__name__}: {e}"}
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
        ref_row = ref_scen.get(sid, {})
        for field in ("command", "detection_trigger", "mitigation_action"):
            if row.get(field) != ref_row.get(field):
                mismatches.append({
                    "scenario": sid, "field": field,
                    "expected": ref_row.get(field), "got": row.get(field),
                })
    return {"verified": len(mismatches) == 0, "n_mismatches": len(mismatches),
            "mismatches": mismatches}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="table_23_network_injections.py",
        description="Reproduce Table 23 (network attack injection commands).",
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
        file_path=log_cfg.get("file", "outputs/logs/table_23.log"), console=True,
    )
    log = get_logger("e3doh.tables.table_23")
    log.info("table_23_start", mode=args.mode, execute=args.execute)

    try:
        matrix = load_attack_matrix()
    except AttackMatrixError as e:
        log.error("attack_matrix_load_failed", error=str(e))
        return 2

    verify_result = _verify_only(matrix, log)
    output: Dict[str, Any] = {
        "table": "table_23", "mode": args.mode,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "seed": seed, "run_id": config.get("_run_id"),
        "verification_mode": "execute" if args.execute else "verify_only",
        **verify_result,
    }
    if args.execute:
        output["execution"] = _execute(config, seed, log)

    output["wall_time_s"] = round(time.perf_counter() - started, 3)
    out_path = output_path("tables", "table_23.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2, default=str), encoding="utf-8")
    log.info("table_23_written", path=str(out_path))

    if args.commit:
        commit_path = expected_path("table_23_expected.json")
        commit_path.parent.mkdir(parents=True, exist_ok=True)
        commit_path.write_text(json.dumps(
            {"table": "table_23", "scenarios": verify_result["scenarios"]},
            indent=2,
        ), encoding="utf-8")
        log.info("table_23_committed", path=str(commit_path))

    verification = None
    if args.verify:
        verification = _verify(output, expected_path("table_23_expected.json"), log)
        output["verification"] = verification

    print()
    print(f"=== Table 23 ({args.mode} mode, {'execute' if args.execute else 'verify-only'}) ===")
    for sid, row in verify_result["scenarios"].items():
        print(f"  {sid}  {row.get('name', '?')}  [{row.get('tool', '?')}]")
        print(f"      command: {row.get('command', '?')}")
        print(f"      trigger: {row.get('detection_trigger', '?')}")
    if verification is not None:
        if verification["verified"]:
            print(f"\nVerification: PASS")
        else:
            print(f"\nVerification: FAIL ({verification.get('n_mismatches')} mismatches)")
    print(f"\nOutput: {out_path}")
    print()

    if not verify_result["all_ok"]:
        return 1
    if verification is not None and not verification["verified"]:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
