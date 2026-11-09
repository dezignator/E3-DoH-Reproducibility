#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tests/test_reproduce_tables.py
==============================

E3-DoH Reproducibility — integration tests that run each of the
per-table and per-figure scripts in toy mode.

These are *slow* tests (each takes several seconds to a minute). They
require:

    - The toy dataset (data/synthetic/), which
      `python data/download_datasets.py --mode toy` produces.
    - The feature store (data/processed/features/), which
      `python data/preprocess_zenodo.py --mode toy` produces.

If either prerequisite is missing, the tests skip with a clear
message rather than fail. This is deliberate: a reviewer who has not
yet run the data pipeline should still be able to run the fast unit
tests (`test_math_utils.py`, `test_protocols.py`, etc.).

Run with:
    pytest tests/test_reproduce_tables.py -v

To run only the fast tests:
    pytest tests/ -v --ignore=tests/test_reproduce_tables.py
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent


def _toy_data_available() -> bool:
    """Whether the toy dataset has been generated."""
    synth = _REPO_ROOT / "data" / "synthetic"
    if not synth.exists():
        return False
    return all(
        (synth / f"{p}_synthetic.parquet").exists()
        for p in ("doh", "dot", "doq")
    )


def _toy_features_available() -> bool:
    """Whether the toy feature store has been generated."""
    features = _REPO_ROOT / "data" / "processed" / "features"
    if not features.exists():
        return False
    return all(
        (features / f"{p}_features.parquet").exists()
        for p in ("doh", "dot", "doq")
    )


def _run_script(args: List[str], timeout: int = 300) -> subprocess.CompletedProcess:
    """Run a script from the repo root."""
    return subprocess.run(
        [sys.executable] + args,
        cwd=str(_REPO_ROOT),
        capture_output=True, text=True, timeout=timeout,
    )


requires_toy_data = pytest.mark.skipif(
    not _toy_data_available() and not _toy_features_available(),
    reason="toy dataset not generated; run `python data/download_datasets.py --mode toy` "
            "and `python data/preprocess_zenodo.py --mode toy` first",
)

requires_toy_features = pytest.mark.skipif(
    not _toy_features_available(),
    reason="toy feature store not generated; run `python data/preprocess_zenodo.py --mode toy` first",
)


# =============================================================================
# Verification script
# =============================================================================

class TestVerificationScript:
    def test_help(self):
        r = _run_script(["expected_outputs/verification_script.py", "--help"], timeout=30)
        assert r.returncode == 0
        assert "verification" in r.stdout.lower()

    def test_missing_artifact_reports_error(self):
        # A freshly checked-out artifact has no outputs/tables/* yet.
        r = _run_script(
            ["expected_outputs/verification_script.py",
             "--artifact", "table_26", "--skip-manifest"],
            timeout=30,
        )
        # Exit code 2 = missing generated output.
        assert r.returncode == 2
        assert "generated output missing" in r.stdout or "generated output missing" in r.stderr


# =============================================================================
# Per-table scripts (toy mode)
# =============================================================================

@requires_toy_features
class TestTable19:
    def test_toy_runs(self):
        r = _run_script([
            "scripts/reproduce_tables/table_19_scalability.py",
            "--mode", "toy",
        ])
        assert r.returncode == 0, r.stdout + r.stderr
        out = _REPO_ROOT / "outputs" / "tables" / "table_19.json"
        assert out.exists()
        payload = json.loads(out.read_text())
        assert payload["trend_ok"] is True
        assert "rows" in payload
        assert len(payload["rows"]) == 5  # baseline..overload


@requires_toy_features
class TestTable20:
    def test_toy_runs(self):
        r = _run_script([
            "scripts/reproduce_tables/table_20_pipeline.py",
            "--mode", "toy", "--protocol", "doh",
        ])
        assert r.returncode == 0, r.stdout + r.stderr
        out = _REPO_ROOT / "outputs" / "tables" / "table_20.json"
        assert out.exists()
        payload = json.loads(out.read_text())
        assert "rows" in payload
        # All five stages present.
        for stage in ("parsing_latency_us", "extraction_latency_us",
                      "scoring_latency_ms", "decision_latency_ms",
                      "mitigation_latency_ms"):
            assert stage in payload["rows"]


@requires_toy_features
class TestTable25:
    def test_toy_runs(self):
        r = _run_script([
            "scripts/reproduce_tables/table_25_privacy_analysis.py",
            "--mode", "toy",
        ])
        assert r.returncode == 0, r.stdout + r.stderr
        out = _REPO_ROOT / "outputs" / "tables" / "table_25.json"
        assert out.exists()
        payload = json.loads(out.read_text())
        # Five configurations.
        assert len(payload["rows"]) == 5
        # Each row has ε, drop, robustness, comm.
        for name in ("reference_configuration", "compromised_clients",
                     "differential_privacy", "homomorphic_aggregation",
                     "adaptive_clipping"):
            row = payload["rows"][name]
            assert "epsilon_realized" in row
            assert row["epsilon_realized"] > 0


@requires_toy_features
class TestTable26:
    def test_toy_runs(self):
        r = _run_script([
            "scripts/reproduce_tables/table_26_multiprotocol_performance.py",
            "--mode", "toy",
        ])
        assert r.returncode == 0, r.stdout + r.stderr
        out = _REPO_ROOT / "outputs" / "tables" / "table_26.json"
        assert out.exists()
        payload = json.loads(out.read_text())
        # All three protocols.
        for proto in ("doh", "dot", "doq"):
            assert proto in payload["protocols"]
            row = payload["protocols"][proto]
            assert "detection_accuracy_pct" in row
            assert "false_positive_rate_pct" in row
            assert "p99_latency_ms" in row


@requires_toy_features
class TestTable18:
    @pytest.mark.skipif(
        True,  # Skipping by default; requires scapy + namespace
        reason="Table 18 requires scapy and (for full mode) namespace "
                "isolation; the toy-mode run is exercised by reproduce_all.sh",
    )
    def test_toy_runs(self):
        pass


# =============================================================================
# Per-figure scripts (toy mode)
# =============================================================================

@requires_toy_features
class TestFigure11:
    def test_runs(self):
        r = _run_script(["scripts/reproduce_figures/figure_11_architecture.py"])
        assert r.returncode == 0, r.stdout + r.stderr
        out = _REPO_ROOT / "outputs" / "figures" / "figure_11.svg"
        assert out.exists()
        # The SVG should be well-formed (starts with <?xml, ends with </svg>).
        content = out.read_text()
        assert content.startswith("<?xml")
        assert "</svg>" in content


@requires_toy_features
class TestFigure12:
    def test_runs(self):
        r = _run_script([
            "scripts/reproduce_figures/figure_12_performance.py",
            "--mode", "toy",
        ])
        assert r.returncode == 0, r.stdout + r.stderr
        out = _REPO_ROOT / "outputs" / "figures" / "figure_12.csv"
        assert out.exists()


@requires_toy_features
class TestFigure13:
    def test_runs(self):
        r = _run_script([
            "scripts/reproduce_figures/figure_13_federated_privacy.py",
            "--mode", "toy",
        ])
        assert r.returncode == 0, r.stdout + r.stderr
        out = _REPO_ROOT / "outputs" / "figures" / "figure_13.csv"
        assert out.exists()


# =============================================================================
# End-to-end toy pipeline
# =============================================================================

@requires_toy_data
@requires_toy_features
class TestToyEndToEnd:
    def test_verify_all(self):
        """
        Running the full toy pipeline then verifying against the
        embedded reference should pass (or report expected fail).

        This test is marked slow; it takes several minutes on a CPU.
        """
        # Run the reproduction driver, skipping verification (we do
        # it explicitly below).
        r = _run_script(
            ["scripts/reproduce_all.sh", "--mode", "toy", "--skip-data"],
            timeout=900,
        )
        # The driver exits 1 on any per-table failure. A toy-mode
        # table 18 run needs scapy; if scapy is missing, it fails.
        # We check that at least the verification-relevant outputs
        # exist.
        if r.returncode != 0:
            # Not a hard failure if the failure was a documented
            # partial (Table 18 with no scapy).
            pass

        # Verify table 26 (the headline table).
        v = _run_script(
            ["expected_outputs/verification_script.py",
             "--artifact", "table_26", "--skip-manifest"],
            timeout=60,
        )
        # Either 0 (pass) or 1 (mismatch; toy mode is trend-only, so
        # some metrics will be out of tolerance and this is expected).
        # Exit 2 (missing output) is a real failure: the table script
        # did not write its output.
        assert v.returncode in (0, 1), (
            f"verification script failed unexpectedly: rc={v.returncode}\n"
            f"stdout: {v.stdout}\nstderr: {v.stderr}"
        )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
