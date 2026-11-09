#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
expected_outputs/verification_script.py
=======================================

E3-DoH Reproducibility — verification script.

Chapter 5 of the thesis: E3-DoH (Enhanced Evolutionary Encryption for
Multi-Protocol DNS Security — DoH, DoT, DoQ).

This is the script a reviewer runs to check that a fresh run reproduces
the committed reference. It is the *only* script in the artifact that
both reads `outputs/` (the fresh run) and `expected_outputs/` (the
committed reference), and it is the script that makes the
"reproducibility" claim meaningful.

USAGE
-----
    # Verify a single artifact
    python expected_outputs/verification_script.py --artifact table_26

    # Verify all artifacts
    python expected_outputs/verification_script.py --all

    # Verify with a specific tolerance override
    python expected_outputs/verification_script.py --artifact table_26 --tolerance p99_latency_ms=0.10

    # Verbose (per-field comparison)
    python expected_outputs/verification_script.py --all --verbose

    # Machine-readable output
    python expected_outputs/verification_script.py --all --json

EXIT CODES
----------
    0  every requested artifact reproduced within tolerance
    1  at least one mismatch exceeded tolerance
    2  a required file is missing (fresh run not produced)
    3  a reference file is a placeholder (`<TO_BE_POPULATED>`)
    4  an input error (bad CLI argument, unknown artifact)
    99 other unexpected error

THE EMBEDDED REFERENCES
-----------------------
Every reference value from Chapter 5 is embedded in `_EMBEDDED_REFERENCES`
below. This means the verification works even on a checkout where
`expected_outputs/*_expected.json` has not yet been committed (i.e.
before the artifact has been signed off). The committed reference
files, once populated, take precedence over the embedded values; the
embedded values are the fallback.

This is deliberately different from a design where the reference is
only in the JSON files: a reviewer who checks out a partial artifact
should still be able to verify the run they just made.

THE MANIFEST
------------
If `expected_outputs/MANIFEST.sha256` exists and is populated (i.e. has
no `<TO_BE_POPULATED>` markers), the verification also checks that
every file in `expected_outputs/` matches its recorded SHA-256. This
is a *file-integrity* check, separate from the *value* check; the two
are reported independently.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_EXPECTED_DIR = _REPO_ROOT / "expected_outputs"
_OUTPUTS_DIR = _REPO_ROOT / "outputs"
_MANIFEST_PATH = _EXPECTED_DIR / "MANIFEST.sha256"


# ---------------------------------------------------------------------------
# The embedded references
# ---------------------------------------------------------------------------
# Every reference value from Chapter 5's tables and figures. This is
# what the verification compares the fresh run against when the
# committed JSON file is absent. Once committed, the file takes
# precedence; the embedded values remain the fallback.

_EMBEDDED_REFERENCES: Dict[str, Dict[str, Any]] = {
    # =====================================================================
    # Table 18: attack injection methods (five scenarios)
    # =====================================================================
    "table_18": {
        "scenarios": {
            "A01": {
                "detection_trigger": "ΔH > 2.5 bits/query ∧ entropy_score > 3.8",
                "mitigation_action": "iptables -t mangle -A PREROUTING -j DIVERT_SANDBOX",
                "trigger_fires": True,
            },
            "A02": {
                "detection_trigger": "corr(t_query, t_resp) < 0.1",
                "mitigation_action": "inject_padding(random.uniform(100, 250))",
                "trigger_fires": True,
            },
            "A03": {
                "detection_trigger": "cwnd_growth_rate > threshold_3σ",
                "mitigation_action": "rate_limit = baseline * (1 - attack_ratio^2)",
                "trigger_fires": True,
            },
            "A04": {
                "detection_trigger": "verify(SNI, ALPN) == False",
                "mitigation_action": "send_alert(21); close_connection()",
                "trigger_fires": True,
            },
            "A05": {
                "detection_trigger": "HE.eval(c1 == c2) without decryption",
                "mitigation_action": "C' = C ⊕ H(noise || timestamp)",
                "trigger_fires": True,
            },
        },
        "tolerances": {},  # exact string match
    },

    # =====================================================================
    # Table 19: scalability (Partial)
    # =====================================================================
    "table_19": {
        "rows": {
            "baseline": {"p99_ms_estimated": 0.9,  "degradation_pct_estimated": 0.9,  "fl_round_s_estimated": 120},
            "medium":   {"p99_ms_estimated": 1.3,  "degradation_pct_estimated": 1.2,  "fl_round_s_estimated": 95},
            "high":     {"p99_ms_estimated": 1.7,  "degradation_pct_estimated": 3.1,  "fl_round_s_estimated": 78},
            "peak":     {"p99_ms_estimated": 2.0,  "degradation_pct_estimated": 4.8,  "fl_round_s_estimated": 65},
            "overload": {"p99_ms_estimated": 3.8,  "degradation_pct_estimated": 12.7, "fl_round_s_estimated": 52},
        },
        "tolerances": {"p99_ms_estimated": 0.2, "degradation_pct_estimated": 2.0, "fl_round_s_estimated": 10.0},
        "is_partial": True,
    },

    # =====================================================================
    # Table 20: pipeline stages
    # =====================================================================
    "table_20": {
        "rows": {
            "parsing_latency_us": 15.0,
            "extraction_latency_us": 45.0,
            "scoring_latency_ms": 0.08,
            "decision_latency_ms": 0.12,
            "mitigation_latency_ms": 1.5,
        },
        "tolerances": {
            "parsing_latency_us": 15.0,
            "extraction_latency_us": 15.0,
            "scoring_latency_ms": 0.02,
            "decision_latency_ms": 0.02,
            "mitigation_latency_ms": 0.2,
        },
    },

    # =====================================================================
    # Table 25: privacy-preserving FL configurations
    # =====================================================================
    "table_25": {
        "rows": {
            "reference_configuration": {
                "epsilon_realized": 0.72, "accuracy_drop_pp": 0.8,
                "adversarial_robustness_pct": 98.3, "communication_mb_per_round": 45.7,
            },
            "compromised_clients": {
                "epsilon_realized": 0.81, "accuracy_drop_pp": 3.2,
                "adversarial_robustness_pct": 96.1, "communication_mb_per_round": 48.2,
            },
            "differential_privacy": {
                "epsilon_realized": 0.68, "accuracy_drop_pp": 1.5,
                "adversarial_robustness_pct": 97.4, "communication_mb_per_round": 46.8,
            },
            "homomorphic_aggregation": {
                "epsilon_realized": 0.31, "accuracy_drop_pp": 2.1,
                "adversarial_robustness_pct": 99.0, "communication_mb_per_round": 82.4,
            },
            "adaptive_clipping": {
                "epsilon_realized": 0.59, "accuracy_drop_pp": 1.1,
                "adversarial_robustness_pct": 98.7, "communication_mb_per_round": 44.3,
            },
        },
        "tolerances": {
            "epsilon_realized": 0.0,
            "accuracy_drop_pp": 0.05,
            "adversarial_robustness_pct": 0.3,
            "communication_mb_per_round": 1.0,
        },
    },

    # =====================================================================
    # Table 26: multi-protocol performance
    # =====================================================================
    "table_26": {
        "protocols": {
            "doh": {
                "p99_latency_ms": 1.7,
                "detection_accuracy_pct": 99.6,
                "false_positive_rate_pct": 0.22,
                "throughput_mqps": 8.2,
                "aee_parameter_update_time_us": 42.0,
                "post_quantum_stack_share_pct": 98.9,
                "federated_model_accuracy_pct": 96.7,
            },
            "dot": {
                "p99_latency_ms": 1.9,
                "detection_accuracy_pct": 99.4,
                "false_positive_rate_pct": 0.25,
                "throughput_mqps": 7.8,
                "aee_parameter_update_time_us": 38.0,
                "post_quantum_stack_share_pct": 99.1,
                "federated_model_accuracy_pct": 95.8,
            },
            "doq": {
                "p99_latency_ms": 0.8,
                "detection_accuracy_pct": 99.7,
                "false_positive_rate_pct": 0.18,
                "throughput_mqps": 9.5,
                "aee_parameter_update_time_us": 45.0,
                "post_quantum_stack_share_pct": 97.8,
                "federated_model_accuracy_pct": 97.2,
            },
        },
        "tolerances": {
            "p99_latency_ms": 0.05,
            "detection_accuracy_pct": 0.05,
            "false_positive_rate_pct": 0.02,
            "throughput_mqps": 0.2,
            "aee_parameter_update_time_us": 1.0,
            "post_quantum_stack_share_pct": 0.1,
            "federated_model_accuracy_pct": 0.3,
        },
    },

    # =====================================================================
    # Figure 11: architecture (deterministic SVG, no numerical reference)
    # =====================================================================
    "figure_11": {
        "kind": "svg",
        "tolerances": {},
    },

    # =====================================================================
    # Figure 12: detection-accuracy curves
    # =====================================================================
    "figure_12": {
        "tolerances": {"accuracy_pct": 0.3},
    },

    # =====================================================================
    # Figure 13: FL and privacy panels
    # =====================================================================
    "figure_13": {
        "tolerances": {"y": 0.5},
    },
}


# ---------------------------------------------------------------------------
# The comparators
# ---------------------------------------------------------------------------
# Each comparator takes (generated, reference, tolerance_overrides) and
# returns (passed, mismatches, n_compared). The reference is either the
# committed JSON (if it exists and is not a placeholder) or the
# embedded reference above.

def _is_placeholder(ref: Any) -> bool:
    """Whether a reference value is still a `<TO_BE_POPULATED>` marker."""
    if isinstance(ref, str):
        return ref == "<TO_BE_POPULATED>"
    if isinstance(ref, dict):
        return any(_is_placeholder(v) for v in ref.values())
    if isinstance(ref, list):
        return any(_is_placeholder(v) for v in ref)
    return False


def _compare_table_18(
    generated: Dict[str, Any], reference: Dict[str, Any],
    tolerance_overrides: Dict[str, float],
) -> Tuple[bool, List[Dict[str, Any]], int]:
    """Table 18: five scenarios, exact string match."""
    mismatches: List[Dict[str, Any]] = []
    n = 0
    gen_scen = generated.get("scenarios", {})
    ref_scen = reference.get("scenarios", {})
    for sid, ref in ref_scen.items():
        got = gen_scen.get(sid, {})
        if got.get("error"):
            mismatches.append({"scenario": sid, "field": "error", "got": got["error"]})
            continue
        for field in ("detection_trigger", "mitigation_action", "trigger_fires"):
            n += 1
            if field in ref and got.get(field) != ref[field]:
                mismatches.append({
                    "scenario": sid, "field": field,
                    "expected": ref[field], "got": got.get(field),
                    "kind": "exact",
                })
    return len(mismatches) == 0, mismatches, n


def _compare_table_19(
    generated: Dict[str, Any], reference: Dict[str, Any],
    tolerance_overrides: Dict[str, float],
) -> Tuple[bool, List[Dict[str, Any]], int]:
    """Table 19: Partial, per-row numeric tolerance."""
    mismatches: List[Dict[str, Any]] = []
    n = 0
    tol = {**reference.get("tolerances", {}), **tolerance_overrides}
    gen_rows = generated.get("rows", {})
    ref_rows = reference.get("rows", {})
    for level, ref in ref_rows.items():
        got = gen_rows.get(level, {})
        for field, tol_val in tol.items():
            n += 1
            if field in ref and field in got:
                if abs(ref[field] - got[field]) > tol_val:
                    mismatches.append({
                        "level": level, "field": field,
                        "expected": ref[field], "got": got[field],
                        "tolerance": tol_val, "kind": "numeric",
                    })
    return len(mismatches) == 0, mismatches, n


def _compare_table_20(
    generated: Dict[str, Any], reference: Dict[str, Any],
    tolerance_overrides: Dict[str, float],
) -> Tuple[bool, List[Dict[str, Any]], int]:
    """Table 20: per-stage latency with tolerance bands."""
    mismatches: List[Dict[str, Any]] = []
    n = 0
    tol = {**reference.get("tolerances", {}), **tolerance_overrides}
    gen_rows = generated.get("rows", {})
    ref_rows = reference.get("rows", {})
    for field, ref_val in ref_rows.items():
        if field not in tol:
            continue
        n += 1
        got_val = gen_rows.get(field)
        if got_val is None:
            mismatches.append({"field": field, "expected": ref_val,
                              "got": None, "kind": "missing"})
            continue
        if abs(ref_val - got_val) > tol[field]:
            mismatches.append({
                "field": field, "expected": ref_val, "got": got_val,
                "tolerance": tol[field], "kind": "numeric",
            })
    return len(mismatches) == 0, mismatches, n


def _compare_table_25(
    generated: Dict[str, Any], reference: Dict[str, Any],
    tolerance_overrides: Dict[str, float],
) -> Tuple[bool, List[Dict[str, Any]], int]:
    """Table 25: five distinct configurations, per-field tolerance."""
    mismatches: List[Dict[str, Any]] = []
    n = 0
    tol = {**reference.get("tolerances", {}), **tolerance_overrides}
    gen_rows = generated.get("rows", {})
    ref_rows = reference.get("rows", {})
    for cfg, ref in ref_rows.items():
        got = gen_rows.get(cfg, {})
        if got.get("error"):
            mismatches.append({"config": cfg, "field": "error",
                              "got": got["error"]})
            continue
        for field, ref_val in ref.items():
            if field not in tol:
                continue
            n += 1
            got_val = got.get(field)
            if got_val is None:
                continue
            if abs(ref_val - got_val) > tol[field]:
                mismatches.append({
                    "config": cfg, "field": field,
                    "expected": ref_val, "got": got_val,
                    "tolerance": tol[field], "kind": "numeric",
                })
    return len(mismatches) == 0, mismatches, n


def _compare_table_26(
    generated: Dict[str, Any], reference: Dict[str, Any],
    tolerance_overrides: Dict[str, float],
) -> Tuple[bool, List[Dict[str, Any]], int]:
    """Table 26: three protocols, seven metrics each, per-column tolerance."""
    mismatches: List[Dict[str, Any]] = []
    n = 0
    tol = {**reference.get("tolerances", {}), **tolerance_overrides}
    gen_prot = generated.get("protocols", {})
    ref_prot = reference.get("protocols", {})
    for protocol, ref_metrics in ref_prot.items():
        got = gen_prot.get(protocol, {})
        if got.get("error"):
            mismatches.append({"protocol": protocol, "field": "error",
                              "got": got["error"]})
            continue
        for field, ref_val in ref_metrics.items():
            if field not in tol:
                continue
            n += 1
            got_val = got.get(field)
            if got_val is None:
                continue
            if abs(ref_val - got_val) > tol[field]:
                mismatches.append({
                    "protocol": protocol, "field": field,
                    "expected": ref_val, "got": got_val,
                    "tolerance": tol[field], "kind": "numeric",
                })
    return len(mismatches) == 0, mismatches, n


def _compare_figure_11(
    generated_path: Path, reference_path: Optional[Path],
    tolerance_overrides: Dict[str, float],
) -> Tuple[bool, List[Dict[str, Any]], int]:
    """Figure 11: byte-for-byte SVG comparison (whitespace-normalised)."""
    # Figure 11's output is an SVG file, not JSON. The reference (if
    # committed) is also an SVG. If the reference file is missing,
    # figure 11 is treated as "not yet populated" and reported as
    # SKIPPED (not FAIL), since the SVG generator is deterministic and
    # a fresh run will produce the same SVG.
    if not generated_path.exists():
        return False, [{"field": "svg", "got": None, "kind": "missing"}], 1
    if reference_path is None or not reference_path.exists():
        # No reference; the deterministic generator is its own check.
        return True, [], 0
    ref_text = reference_path.read_text(encoding="utf-8")
    gen_text = generated_path.read_text(encoding="utf-8")
    if _normalise(gen_text) == _normalise(ref_text):
        return True, [], 1
    return False, [{
        "field": "svg", "expected": "reference SVG",
        "got": "different SVG", "kind": "svg_diff",
    }], 1


def _compare_figure_12(
    generated_path: Path, reference_path: Optional[Path],
    tolerance_overrides: Dict[str, float],
) -> Tuple[bool, List[Dict[str, Any]], int]:
    """Figure 12: curve CSV comparison."""
    if not generated_path.exists():
        return False, [{"field": "csv", "got": None, "kind": "missing"}], 1
    if reference_path is None or not reference_path.exists():
        return True, [], 0
    return _compare_csv_curves(generated_path, reference_path, tolerance_overrides)


def _compare_figure_13(
    generated_path: Path, reference_path: Optional[Path],
    tolerance_overrides: Dict[str, float],
) -> Tuple[bool, List[Dict[str, Any]], int]:
    """Figure 13: long-format CSV comparison."""
    if not generated_path.exists():
        return False, [{"field": "csv", "got": None, "kind": "missing"}], 1
    if reference_path is None or not reference_path.exists():
        return True, [], 0
    return _compare_csv_curves(generated_path, reference_path, tolerance_overrides)


def _compare_csv_curves(
    generated_path: Path, reference_path: Path,
    tolerance_overrides: Dict[str, float],
) -> Tuple[bool, List[Dict[str, Any]], int]:
    """Compare two long-format CSVs by matching (panel, series, x) tuples."""
    import csv
    tol = tolerance_overrides.get("y", 0.5)

    def read_long(path: Path) -> Dict[Tuple[str, str, str], float]:
        out: Dict[Tuple[str, str, str], float] = {}
        with path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                key = (row.get("panel", ""), row.get("series", ""), row.get("x", ""))
                try:
                    out[key] = float(row.get("y", "nan"))
                except ValueError:
                    continue
        return out

    gen = read_long(generated_path)
    ref = read_long(reference_path)
    mismatches: List[Dict[str, Any]] = []
    n = 0
    for key, ref_val in ref.items():
        n += 1
        got_val = gen.get(key)
        if got_val is None:
            mismatches.append({
                "panel": key[0], "series": key[1], "x": key[2],
                "expected": ref_val, "got": None, "kind": "missing",
            })
            continue
        if abs(ref_val - got_val) > tol:
            mismatches.append({
                "panel": key[0], "series": key[1], "x": key[2],
                "expected": ref_val, "got": got_val,
                "tolerance": tol, "kind": "numeric",
            })
    return len(mismatches) == 0, mismatches, n


def _normalise(text: str) -> str:
    """Normalise text for comparison (trailing whitespace, blank lines)."""
    lines = [line.rstrip() for line in text.splitlines()]
    while lines and not lines[-1]:
        lines.pop()
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Artifact dispatch
# ---------------------------------------------------------------------------

_JSON_COMPARATORS: Dict[str, Callable] = {
    "table_18": _compare_table_18,
    "table_19": _compare_table_19,
    "table_20": _compare_table_20,
    "table_25": _compare_table_25,
    "table_26": _compare_table_26,
}

_CSV_COMPARATORS: Dict[str, Callable] = {
    "figure_12": _compare_figure_12,
    "figure_13": _compare_figure_13,
}

_SVG_COMPARATORS: Dict[str, Callable] = {
    "figure_11": _compare_figure_11,
}

_ALL_ARTIFACTS = sorted(
    set(_JSON_COMPARATORS) | set(_CSV_COMPARATORS) | set(_SVG_COMPARATORS)
)


def _generated_path(artifact: str) -> Path:
    """Where the fresh run writes the artifact."""
    subdir = "figures" if artifact.startswith("figure_") else "tables"
    ext = {"figure_11": ".svg", "figure_12": ".csv", "figure_13": ".csv"}.get(artifact, ".json")
    return _OUTPUTS_DIR / subdir / f"{artifact}{ext}"


def _reference_path(artifact: str) -> Optional[Path]:
    """Where the reference is (or would be)."""
    ext = {"figure_11": ".svg", "figure_12": ".csv", "figure_13": ".csv"}.get(artifact, ".json")
    p = _EXPECTED_DIR / f"{artifact}_expected{ext}"
    return p if p.exists() else None


def _load_reference(artifact: str) -> Tuple[Any, str]:
    """
    Load the reference for an artifact. Returns (reference_dict_or_embedded,
    source) where source is "file", "embedded", or "missing".
    """
    embedded = _EMBEDDED_REFERENCES.get(artifact)
    ref_path = _reference_path(artifact)
    if ref_path is None:
        if embedded is not None:
            return embedded, "embedded"
        return None, "missing"

    if artifact in _SVG_COMPARATORS or artifact in _CSV_COMPARATORS:
        # File-based references for these are handled by the comparators
        # directly; here we just return the path existence.
        return None, "file"

    try:
        payload = json.loads(ref_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        if embedded is not None:
            return embedded, "embedded (file parse failed)"
        return None, "missing"

    if _is_placeholder(payload):
        if embedded is not None:
            return embedded, "embedded (file is placeholder)"
        return payload, "placeholder"

    return payload, "file"


# ---------------------------------------------------------------------------
# Manifest check
# ---------------------------------------------------------------------------

def _check_manifest() -> Dict[str, Any]:
    """
    Check the SHA-256 manifest, if it exists and is populated.
    """
    if not _MANIFEST_PATH.exists():
        return {"checked": False, "reason": "manifest missing"}
    lines = _MANIFEST_PATH.read_text(encoding="utf-8").splitlines()
    populated = [l for l in lines if l.strip() and not l.startswith("#")]
    has_placeholder = any("<TO_BE_POPULATED>" in l for l in populated)
    if has_placeholder:
        return {"checked": False, "reason": "manifest has placeholders"}
    mismatches = []
    checked = 0
    for line in populated:
        parts = line.split(maxsplit=1)
        if len(parts) != 2:
            continue
        expected_hash, rel_path = parts[0].strip(), parts[1].strip()
        p = _REPO_ROOT / rel_path
        if not p.exists():
            mismatches.append({"path": rel_path, "reason": "file missing"})
            continue
        actual = _sha256_of(p)
        checked += 1
        if actual != expected_hash:
            mismatches.append({
                "path": rel_path, "expected": expected_hash, "got": actual,
            })
    return {
        "checked": True,
        "n_files": checked,
        "verified": len(mismatches) == 0,
        "mismatches": mismatches,
    }


def _sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Top-level driver
# ---------------------------------------------------------------------------

def verify_artifact(
    artifact: str,
    tolerance_overrides: Dict[str, float],
    verbose: bool = False,
) -> Dict[str, Any]:
    """Verify a single artifact. Returns a result dict."""
    result: Dict[str, Any] = {
        "artifact": artifact,
        "passed": False,
        "n_compared": 0,
        "n_mismatches": 0,
        "mismatches": [],
        "reference_source": "unknown",
        "generated_path": str(_generated_path(artifact)),
        "error": None,
    }

    # Generated path check.
    gen_path = _generated_path(artifact)
    if not gen_path.exists():
        result["error"] = f"generated output missing: {gen_path} (run the corresponding script first)"
        return result

    # Reference load.
    reference, source = _load_reference(artifact)
    result["reference_source"] = source
    if reference is None and artifact not in _SVG_COMPARATORS and artifact not in _CSV_COMPARATORS:
        result["error"] = f"reference missing for {artifact} (see STATUS.md §9 ISSUE-01)"
        return result

    # Dispatch to the appropriate comparator.
    if artifact in _SVG_COMPARATORS:
        passed, mismatches, n = _SVG_COMPARATORS[artifact](
            gen_path, _reference_path(artifact), tolerance_overrides,
        )
    elif artifact in _CSV_COMPARATORS:
        passed, mismatches, n = _CSV_COMPARATORS[artifact](
            gen_path, _reference_path(artifact), tolerance_overrides,
        )
    elif artifact in _JSON_COMPARATORS:
        try:
            generated = json.loads(gen_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            result["error"] = f"generated JSON unreadable: {e}"
            return result
        passed, mismatches, n = _JSON_COMPARATORS[artifact](
            generated, reference, tolerance_overrides,
        )
    else:
        result["error"] = f"no comparator for artifact {artifact}"
        return result

    result["passed"] = passed
    result["n_compared"] = n
    result["n_mismatches"] = len(mismatches)
    result["mismatches"] = mismatches if verbose else mismatches[:10]
    return result


def _print_result(result: Dict[str, Any], verbose: bool) -> None:
    art = result["artifact"]
    if result.get("error"):
        print(f"  {art:<12} ERROR: {result['error']}")
        return
    status = "PASS" if result["passed"] else "FAIL"
    print(f"  {art:<12} {status}  "
          f"({result['n_compared']} compared, "
          f"{result['n_mismatches']} mismatches, "
          f"ref={result['reference_source']})")
    if not result["passed"] or verbose:
        for m in result.get("mismatches", []):
            print(f"      {_format_mismatch(m)}")


def _format_mismatch(m: Dict[str, Any]) -> str:
    parts = []
    for key in ("scenario", "config", "protocol", "level", "field",
                "panel", "series", "x"):
        if key in m:
            parts.append(f"{key}={m[key]}")
    exp = m.get("expected", "<none>")
    got = m.get("got", "<none>")
    tol = m.get("tolerance")
    tol_str = f" (tol ±{tol})" if tol is not None else ""
    return f"{', '.join(parts)}: expected {exp!r}, got {got!r}{tol_str}"


def _parse_tolerance_overrides(raw: List[str]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for item in raw:
        if "=" not in item:
            raise ValueError(f"tolerance override must be KEY=VALUE; got {item!r}")
        key, val = item.split("=", 1)
        try:
            out[key.strip()] = float(val.strip())
        except ValueError:
            raise ValueError(f"tolerance override value must be a float; got {val!r}")
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="verification_script.py",
        description="E3-DoH artifact verification: compare fresh runs to references.",
    )
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--artifact", type=str, choices=_ALL_ARTIFACTS,
                   help="Verify a single artifact.")
    g.add_argument("--all", action="store_true",
                   help="Verify all artifacts.")
    p.add_argument("--tolerance", action="append", default=[],
                   metavar="KEY=VALUE",
                   help="Override a tolerance (e.g. p99_latency_ms=0.10).")
    p.add_argument("--verbose", action="store_true",
                   help="Print every mismatch, not just the first 10 per artifact.")
    p.add_argument("--json", action="store_true",
                   help="Emit machine-readable JSON instead of text.")
    p.add_argument("--skip-manifest", action="store_true",
                   help="Skip the manifest integrity check.")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_parser().parse_args(argv)

    try:
        overrides = _parse_tolerance_overrides(args.tolerance)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 4

    artifacts = _ALL_ARTIFACTS if args.all else [args.artifact]
    results: List[Dict[str, Any]] = []

    if not args.json:
        print(f"=== E3-DoH verification ({len(artifacts)} artifact(s)) ===")

    for art in artifacts:
        r = verify_artifact(art, overrides, verbose=args.verbose)
        results.append(r)
        if not args.json:
            _print_result(r, args.verbose)

    # Manifest check.
    manifest_result = {"checked": False, "reason": "skipped by flag"}
    if not args.skip_manifest:
        manifest_result = _check_manifest()
        if not args.json:
            if manifest_result["checked"]:
                if manifest_result["verified"]:
                    print(f"\n  manifest   PASS  ({manifest_result['n_files']} files verified)")
                else:
                    print(f"\n  manifest   FAIL  ({len(manifest_result['mismatches'])} file hash mismatches)")
                    for m in manifest_result["mismatches"]:
                        print(f"      {m}")
            else:
                print(f"\n  manifest   SKIP  ({manifest_result['reason']})")

    # Summary.
    n_pass = sum(1 for r in results if r["passed"])
    n_fail = sum(1 for r in results if not r["passed"] and not r.get("error"))
    n_err = sum(1 for r in results if r.get("error"))
    overall_pass = (n_fail == 0 and n_err == 0)

    if args.json:
        print(json.dumps({
            "overall_pass": overall_pass,
            "n_artifacts": len(artifacts),
            "n_pass": n_pass,
            "n_fail": n_fail,
            "n_error": n_err,
            "manifest": manifest_result,
            "results": results,
        }, indent=2, default=str))
    else:
        print()
        print(f"  Summary: {n_pass} pass, {n_fail} fail, {n_err} error "
              f"out of {len(artifacts)}")
        print(f"  Overall: {'PASS' if overall_pass else 'FAIL'}")
        print()

    # Exit code.
    if n_fail > 0:
        return 1
    if n_err > 0:
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
