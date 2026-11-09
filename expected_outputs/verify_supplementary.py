#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
expected_outputs/verify_supplementary.py
========================================

E3-DoH Reproducibility — supplementary verification for Tables 21–24.

The main `verification_script.py` (delivered in File 51) covers Tables
18, 19, 20, 25, 26 and Figures 11, 12, 13. Tables 21, 22, 23, and 24
are documented in `data/attack_matrix.csv` (their rows are the
A01–A15 scenarios) and have their own reproduction scripts, but they
are not in `_ALL_ARTIFACTS` because their comparators are simpler
(no numerical tolerances; the comparison is string equality).

This script closes that gap. It verifies each of the four tables by
comparing its reproduction script's output against the corresponding
`*_expected.json` file. Exit codes match `verification_script.py`.

WHY A SEPARATE SCRIPT
---------------------
    - Each table's comparator is a string comparison; the main
      script's numeric-tolerance machinery does not apply.
    - The main script's CLI (`--artifact`, `--all`) would need to be
      extended with four more comparator functions if these tables
      were folded in; a separate script keeps the main script's
      scope fixed.
    - A reviewer who wants to verify only the tables the main script
      covers can skip this one; a reviewer who wants full coverage
      runs both.

USAGE
-----
    # Verify one table
    python expected_outputs/verify_supplementary.py --table table_21

    # Verify all four
    python expected_outputs/verify_supplementary.py --all

    # Verbose
    python expected_outputs/verify_supplementary.py --all --verbose

EXIT CODES
----------
    0  every requested table reproduced
    1  at least one mismatch
    2  a required file is missing (fresh run not produced)
    3  a reference file is a placeholder
    4  bad CLI argument
    99 unexpected error
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_EXPECTED_DIR = _REPO_ROOT / "expected_outputs"
_OUTPUTS_DIR = _REPO_ROOT / "outputs" / "tables"


# ---------------------------------------------------------------------------
# Comparators
# ---------------------------------------------------------------------------

def _is_placeholder(value: Any) -> bool:
    if isinstance(value, str):
        return value == "<TO_BE_POPULATED>"
    if isinstance(value, dict):
        return any(_is_placeholder(v) for v in value.values())
    if isinstance(value, list):
        return any(_is_placeholder(v) for v in value)
    return False


def _compare_table_21(
    generated: Dict[str, Any], reference: Dict[str, Any], verbose: bool,
) -> Tuple[bool, List[Dict[str, Any]], int]:
    """
    Table 21: the four flag/value rows.

    Compares `value` for each flag in the reference against the
    generated rows. The `match` boolean from the reproduction is not
    part of the reference; the reference's `rows` are the ground truth.
    """
    mismatches: List[Dict[str, Any]] = []
    n = 0
    gen_rows = {r["flag"]: r for r in generated.get("rows", [])}
    for ref_row in reference.get("rows", []):
        flag = ref_row["flag"]
        n += 1
        got = gen_rows.get(flag)
        if got is None:
            mismatches.append({"flag": flag, "reason": "missing in generated"})
            continue
        if got.get("value") != ref_row.get("value"):
            mismatches.append({
                "flag": flag, "field": "value",
                "expected": ref_row.get("value"),
                "got": got.get("value"),
            })
    return len(mismatches) == 0, mismatches, n


def _compare_table_22(
    generated: Dict[str, Any], reference: Dict[str, Any], verbose: bool,
) -> Tuple[bool, List[Dict[str, Any]], int]:
    """
    Table 22: five scenarios, string equality on trigger/action.
    """
    mismatches: List[Dict[str, Any]] = []
    n = 0
    gen_scen = generated.get("scenarios", {})
    ref_scen = reference.get("scenarios", {})
    for sid, ref_row in ref_scen.items():
        got = gen_scen.get(sid, {})
        for field in ("detection_trigger", "mitigation_action"):
            n += 1
            if got.get(field) != ref_row.get(field):
                mismatches.append({
                    "scenario": sid, "field": field,
                    "expected": ref_row.get(field),
                    "got": got.get(field),
                })
    return len(mismatches) == 0, mismatches, n


def _compare_table_23(
    generated: Dict[str, Any], reference: Dict[str, Any], verbose: bool,
) -> Tuple[bool, List[Dict[str, Any]], int]:
    """
    Table 23: five scenarios, string equality on command/trigger/action.
    """
    mismatches: List[Dict[str, Any]] = []
    n = 0
    gen_scen = generated.get("scenarios", {})
    ref_scen = reference.get("scenarios", {})
    for sid, ref_row in ref_scen.items():
        got = gen_scen.get(sid, {})
        for field in ("command", "detection_trigger", "mitigation_action"):
            n += 1
            if got.get(field) != ref_row.get(field):
                mismatches.append({
                    "scenario": sid, "field": field,
                    "expected": ref_row.get(field),
                    "got": got.get(field),
                })
    return len(mismatches) == 0, mismatches, n


def _compare_table_24(
    generated: Dict[str, Any], reference: Dict[str, Any], verbose: bool,
) -> Tuple[bool, List[Dict[str, Any]], int]:
    """
    Table 24: five schema rows plus two overlap rows.
    """
    mismatches: List[Dict[str, Any]] = []
    n = 0

    # Schema rows: compare position-by-position on every field.
    gen_schema = generated.get("schema_rows", [])
    ref_schema = reference.get("schema_rows", [])
    if len(gen_schema) != len(ref_schema):
        mismatches.append({
            "field": "schema_rows_length",
            "expected": len(ref_schema),
            "got": len(gen_schema),
        })
    else:
        for ref_row, gen_row in zip(ref_schema, gen_schema):
            for field in ("injection_layer", "attack_method", "detection_sign", "countermeasure"):
                n += 1
                if gen_row.get(field) != ref_row.get(field):
                    mismatches.append({
                        "layer": ref_row.get("injection_layer"),
                        "field": field,
                        "expected": ref_row.get(field),
                        "got": gen_row.get(field),
                    })

    # Overlap rows: compare by scenario id.
    gen_overlap = {r["scenario"]: r for r in generated.get("overlap_rows", [])}
    for ref_row in reference.get("overlap_rows", []):
        sid = ref_row.get("scenario")
        got = gen_overlap.get(sid, {})
        for field in ("matrix_trigger", "matrix_mitigation"):
            n += 1
            if got.get(field) != ref_row.get(field):
                mismatches.append({
                    "scenario": sid, "field": field,
                    "expected": ref_row.get(field),
                    "got": got.get(field),
                })
    return len(mismatches) == 0, mismatches, n


_COMPARATORS: Dict[str, Callable] = {
    "table_21": _compare_table_21,
    "table_22": _compare_table_22,
    "table_23": _compare_table_23,
    "table_24": _compare_table_24,
}

_ALL_TABLES = sorted(_COMPARATORS.keys())


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def _verify_one(
    table: str, verbose: bool,
) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "table": table,
        "passed": False,
        "n_compared": 0,
        "n_mismatches": 0,
        "mismatches": [],
        "error": None,
    }

    gen_path = _OUTPUTS_DIR / f"{table}.json"
    ref_path = _EXPECTED_DIR / f"{table}_expected.json"

    if not gen_path.exists():
        result["error"] = (
            f"generated output missing: {gen_path} "
            f"(run scripts/reproduce_tables/{table}*.py first)"
        )
        return result

    if not ref_path.exists():
        result["error"] = f"reference missing: {ref_path}"
        return result

    try:
        generated = json.loads(gen_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        result["error"] = f"generated JSON unreadable: {e}"
        return result

    try:
        reference = json.loads(ref_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        result["error"] = f"reference JSON unreadable: {e}"
        return result

    if _is_placeholder(reference):
        result["error"] = "reference file is a placeholder"
        return result

    try:
        passed, mismatches, n = _COMPARATORS[table](generated, reference, verbose)
    except Exception as e:
        result["error"] = f"comparator raised: {type(e).__name__}: {e}"
        return result

    result["passed"] = passed
    result["n_compared"] = n
    result["n_mismatches"] = len(mismatches)
    result["mismatches"] = mismatches if verbose else mismatches[:10]
    return result


def _format_mismatch(m: Dict[str, Any]) -> str:
    parts = []
    for key in ("scenario", "flag", "layer", "field"):
        if key in m:
            parts.append(f"{key}={m[key]}")
    exp = m.get("expected", m.get("reason", "<none>"))
    got = m.get("got", "<none>")
    return f"{', '.join(parts)}: expected {exp!r}, got {got!r}"


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="verify_supplementary.py",
        description="Verify Tables 21–24 against their reference JSONs.",
    )
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--table", choices=_ALL_TABLES)
    g.add_argument("--all", action="store_true")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--json", action="store_true")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    tables = _ALL_TABLES if args.all else [args.table]

    results: List[Dict[str, Any]] = []
    if not args.json:
        print(f"=== Supplementary verification ({len(tables)} table(s)) ===")

    for table in tables:
        r = _verify_one(table, args.verbose)
        results.append(r)
        if not args.json:
            if r.get("error"):
                print(f"  {table:<12} ERROR: {r['error']}")
            else:
                status = "PASS" if r["passed"] else "FAIL"
                print(f"  {table:<12} {status}  "
                      f"({r['n_compared']} compared, "
                      f"{r['n_mismatches']} mismatches)")
                if not r["passed"] or args.verbose:
                    for m in r.get("mismatches", []):
                        print(f"      {_format_mismatch(m)}")

    n_pass = sum(1 for r in results if r["passed"])
    n_fail = sum(1 for r in results if not r["passed"] and not r.get("error"))
    n_err = sum(1 for r in results if r.get("error"))
    overall = (n_fail == 0 and n_err == 0)

    if args.json:
        print(json.dumps({
            "overall_pass": overall,
            "n_tables": len(tables),
            "n_pass": n_pass,
            "n_fail": n_fail,
            "n_error": n_err,
            "results": results,
        }, indent=2, default=str))
    else:
        print()
        print(f"  Summary: {n_pass} pass, {n_fail} fail, {n_err} error "
              f"out of {len(tables)}")
        print(f"  Overall: {'PASS' if overall else 'FAIL'}")
        print()

    if n_fail > 0:
        return 1
    if n_err > 0:
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
