#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tests/test_data_utils.py
========================

E3-DoH Reproducibility — unit tests for data/data_utils.py.

These tests verify the shared data-access library's public contract:

    - load_config: reads master + includes, applies overrides, rejects
      unknown override keys, computes a run ID
    - load_feature_store: enforces the twelve-feature schema, rejects
      NaN, enforces provenance, returns the canonical column order
    - load_attack_matrix: validates the CSV's column set, row count,
      IDs A01..A15, protocol values, and empty cells
    - attack_matrix_row: returns a single row as a dict
    - load_reference: parses a reference JSON, enforces the
      values/tolerance key match, tolerates a missing file
    - write_result_json / read_result_json: round-trip, tolerance
      enforcement
    - exit_code_for: maps each exception class to its exit code
    - describe_artifact_state: returns the expected top-level keys
    - the eight constants (PROTOCOLS, FEATURE_COLUMNS, META_COLUMNS,
      PROVENANCE_VALUES, ATTACK_MATRIX_COLUMNS, ATTACK_IDS)

The tests that need real config/data files use `pytest.skip` when the
files are not present, so a partial checkout can still run the tests
that do not depend on the filesystem.

Run with:
    pytest tests/test_data_utils.py -v
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from data.data_utils import (
    ATTACK_IDS,
    ATTACK_MATRIX_COLUMNS,
    AttackMatrixError,
    ConfigError,
    DataUtilsError,
    FEATURE_COLUMNS,
    FeatureStoreSchemaError,
    META_COLUMNS,
    PROTOCOLS,
    PROVENANCE_VALUES,
    ProvenanceError,
    ReferenceNotFoundError,
    attack_matrix_row,
    config_path,
    describe_artifact_state,
    exit_code_for,
    feature_store_path,
    load_attack_matrix,
    load_config,
    load_feature_store,
    load_reference,
    output_path,
    read_result_json,
    reference_tolerance,
    reference_value,
    write_result_json,
)


# =============================================================================
# Constants
# =============================================================================

class TestConstants:
    def test_protocols(self):
        assert PROTOCOLS == ("doh", "dot", "doq")

    def test_feature_columns_count(self):
        assert len(FEATURE_COLUMNS) == 12

    def test_meta_columns_count(self):
        assert len(META_COLUMNS) == 6

    def test_provenance_values(self):
        assert set(PROVENANCE_VALUES) == {"synthetic", "DS-1"}

    def test_attack_matrix_columns(self):
        assert len(ATTACK_MATRIX_COLUMNS) == 7
        assert "Attack_ID" in ATTACK_MATRIX_COLUMNS

    def test_attack_ids(self):
        assert len(ATTACK_IDS) == 15
        assert ATTACK_IDS[0] == "A01"
        assert ATTACK_IDS[-1] == "A15"


# =============================================================================
# exit_code_for
# =============================================================================

class TestExitCodeFor:
    def test_config_error(self):
        assert exit_code_for(ConfigError("x")) == 1

    def test_feature_store_schema_error(self):
        assert exit_code_for(FeatureStoreSchemaError("x")) == 2

    def test_attack_matrix_error(self):
        assert exit_code_for(AttackMatrixError("x")) == 3

    def test_reference_not_found(self):
        assert exit_code_for(ReferenceNotFoundError("x")) == 4

    def test_provenance_error(self):
        assert exit_code_for(ProvenanceError("x")) == 5

    def test_data_utils_base(self):
        assert exit_code_for(DataUtilsError("x")) == 6

    def test_other_exception(self):
        assert exit_code_for(ValueError("x")) == 99


# =============================================================================
# Path helpers
# =============================================================================

class TestPathHelpers:
    def test_feature_store_path(self):
        for proto in PROTOCOLS:
            p = feature_store_path(proto)
            assert p.name == f"{proto}_features.parquet"

    def test_feature_store_path_unknown(self):
        with pytest.raises(ConfigError):
            feature_store_path("foo")

    def test_config_path_master(self):
        p = config_path("default_e3_config")
        assert p.name == "default_e3_config.yaml"

    def test_config_path_include(self):
        p = config_path("aee")
        assert p.name == "aee_params.yaml"

    def test_output_path_known_kinds(self):
        for kind in ("tables", "figures", "logs", "checkpoints"):
            p = output_path(kind, "test.json")
            assert kind in str(p)

    def test_output_path_unknown_kind(self):
        with pytest.raises(ConfigError):
            output_path("garbage", "test.json")


# =============================================================================
# load_config
# =============================================================================

@pytest.mark.skipif(
    not config_path("default_e3_config").exists(),
    reason="master config file not on disk",
)
class TestLoadConfig:
    def test_loads_master_and_includes(self):
        cfg = load_config()
        assert "artifact" in cfg
        assert "runtime" in cfg
        assert "_includes" in cfg
        assert "aee" in cfg["_includes"]
        assert "qhe" in cfg["_includes"]

    def test_run_id_present(self):
        cfg = load_config()
        assert cfg["_run_id"]
        assert len(cfg["_run_id"]) == 16

    def test_hashes_recorded(self):
        cfg = load_config()
        assert "default_e3_config" in cfg["_hashes"]
        # Every include should have a hash.
        for key in cfg["_includes"]:
            assert key in cfg["_hashes"]

    def test_override_known_key(self):
        cfg = load_config(overrides={"runtime.seed": 7})
        assert cfg["runtime"]["seed"] == 7

    def test_override_unknown_key_raises(self):
        with pytest.raises(ConfigError):
            load_config(overrides={"runtime.nonexistent": 7})

    def test_missing_file_raises(self):
        with pytest.raises(ConfigError):
            load_config(config_path="/tmp/nonexistent_e3doh_config.yaml")


# =============================================================================
# load_attack_matrix
# =============================================================================

@pytest.mark.skipif(
    not (_REPO_ROOT / "data" / "attack_matrix.csv").exists(),
    reason="attack_matrix.csv not on disk",
)
class TestLoadAttackMatrix:
    def test_loads(self):
        df = load_attack_matrix()
        assert len(df) == 15
        assert list(df["Attack_ID"]) == list(ATTACK_IDS)

    def test_columns_match(self):
        df = load_attack_matrix()
        assert tuple(df.columns) == ATTACK_MATRIX_COLUMNS

    def test_protocols_valid(self):
        df = load_attack_matrix()
        assert set(df["Protocol"]) <= {"DoH", "DoT", "DoQ"}

    def test_attack_matrix_row(self):
        row = attack_matrix_row("A01")
        assert row["Attack_ID"] == "A01"
        assert isinstance(row["Detection_Trigger"], str)

    def test_attack_matrix_row_unknown(self):
        with pytest.raises(AttackMatrixError):
            attack_matrix_row("A99")

    def test_missing_file_raises(self):
        with pytest.raises(AttackMatrixError):
            load_attack_matrix("/tmp/nonexistent_matrix.csv")


# =============================================================================
# load_feature_store
# =============================================================================

def _make_feature_store_parquet(
    tmp_dir: Path, protocol: str, n_rows: int = 20,
    provenance: str = "DS-1", with_nan: bool = False,
) -> Path:
    """Build a schema-valid Parquet file for tests."""
    rng = np.random.default_rng(0)
    data = {col: rng.normal(size=n_rows) for col in FEATURE_COLUMNS}
    data["flow_id"] = [f"f-{i}" for i in range(n_rows)]
    data["protocol"] = [protocol.upper()] * n_rows
    data["label"] = ["benign"] * n_rows
    data["attack_type"] = [""] * n_rows
    data["query_type"] = ["OTHER"] * n_rows
    data["provenance"] = [provenance] * n_rows
    df = pd.DataFrame(data)
    if with_nan:
        df.loc[0, FEATURE_COLUMNS[0]] = np.nan
    path = tmp_dir / f"{protocol}_features.parquet"
    df.to_parquet(path, index=False)
    return path


class TestLoadFeatureStore:
    def test_loads_valid(self, tmp_path):
        _make_feature_store_parquet(tmp_path, "doh", provenance="DS-1")
        df = load_feature_store(
            "doh", require_provenance="DS-1", feature_store_dir=tmp_path,
        )
        assert len(df) == 20
        # Column order: features first, then meta.
        assert list(df.columns[:len(FEATURE_COLUMNS)]) == list(FEATURE_COLUMNS)

    def test_unknown_protocol_raises(self, tmp_path):
        with pytest.raises(FeatureStoreSchemaError):
            load_feature_store("foo", feature_store_dir=tmp_path)

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FeatureStoreSchemaError):
            load_feature_store("doh", feature_store_dir=tmp_path)

    def test_nan_rejected(self, tmp_path):
        _make_feature_store_parquet(tmp_path, "doh", with_nan=True)
        with pytest.raises(FeatureStoreSchemaError):
            load_feature_store("doh", feature_store_dir=tmp_path)

    def test_provenance_mismatch_raises(self, tmp_path):
        _make_feature_store_parquet(tmp_path, "doh", provenance="synthetic")
        with pytest.raises(ProvenanceError):
            load_feature_store(
                "doh", require_provenance="DS-1", feature_store_dir=tmp_path,
            )

    def test_provenance_none_skips_check(self, tmp_path):
        _make_feature_store_parquet(tmp_path, "doh", provenance="synthetic")
        df = load_feature_store(
            "doh", require_provenance=None, feature_store_dir=tmp_path,
        )
        assert len(df) == 20


# =============================================================================
# load_reference
# =============================================================================

class TestLoadReference:
    def test_loads_valid(self, tmp_path):
        ref = {
            "table": "table_test",
            "values": {"a": 1.0, "b": 2.0},
            "tolerance": {"a": 0.1, "b": 0.2},
        }
        (tmp_path / "table_test_expected.json").write_text(
            json.dumps(ref), encoding="utf-8",
        )
        loaded = load_reference("table_test", expected_dir=tmp_path)
        assert loaded["values"]["a"] == 1.0

    def test_missing_raises(self, tmp_path):
        with pytest.raises(ReferenceNotFoundError):
            load_reference("nope", expected_dir=tmp_path)

    def test_missing_tolerance_raises(self, tmp_path):
        ref = {
            "values": {"a": 1.0, "b": 2.0},
            "tolerance": {"a": 0.1},  # missing b
        }
        (tmp_path / "table_test_expected.json").write_text(
            json.dumps(ref), encoding="utf-8",
        )
        with pytest.raises(ReferenceNotFoundError):
            load_reference("table_test", expected_dir=tmp_path)

    def test_reference_value(self):
        ref = {"values": {"a": 1.0}, "tolerance": {"a": 0.1}}
        assert reference_value(ref, "a") == 1.0

    def test_reference_value_missing(self):
        ref = {"values": {}, "tolerance": {}}
        with pytest.raises(ReferenceNotFoundError):
            reference_value(ref, "a")

    def test_reference_tolerance(self):
        ref = {"values": {"a": 1.0}, "tolerance": {"a": 0.1}}
        assert reference_tolerance(ref, "a") == 0.1


# =============================================================================
# write_result_json / read_result_json
# =============================================================================

class TestResultSidecar:
    def _mock_config(self) -> Dict[str, Any]:
        return {
            "runtime": {"mode": "toy", "seed": 42},
            "_run_id": "test-run-id",
            "_hashes": {"default_e3_config": "abc123"},
            "hardware": {"a100": {"backend": "emulated"},
                         "loihi": {"backend": "emulated"}},
        }

    def test_write_and_read(self, tmp_path):
        out = tmp_path / "result.json"
        write_result_json(
            artifact="table_test",
            values={"x": 1.0, "y": 2.0},
            tolerance={"x": 0.1, "y": 0.2},
            config=self._mock_config(),
            output_path=out,
        )
        assert out.exists()
        loaded = read_result_json(out)
        assert loaded["values"]["x"] == 1.0
        assert loaded["tolerance"]["x"] == 0.1

    def test_write_rejects_mismatched_keys(self, tmp_path):
        with pytest.raises(ConfigError):
            write_result_json(
                artifact="table_test",
                values={"x": 1.0, "y": 2.0},
                tolerance={"x": 0.1},  # missing y
                config=self._mock_config(),
                output_path=tmp_path / "result.json",
            )

    def test_read_missing_raises(self, tmp_path):
        with pytest.raises(ReferenceNotFoundError):
            read_result_json(tmp_path / "nonexistent.json")

    def test_provenance_recorded(self, tmp_path):
        out = tmp_path / "result.json"
        write_result_json(
            artifact="table_test",
            values={"x": 1.0},
            tolerance={"x": 0.1},
            config=self._mock_config(),
            output_path=out,
        )
        loaded = read_result_json(out)
        assert "provenance" in loaded
        prov = loaded["provenance"]
        assert prov["mode"] == "toy"
        assert prov["seed"] == 42
        assert prov["run_id"] == "test-run-id"
        assert prov["config_hashes"]["default_e3_config"] == "abc123"


# =============================================================================
# describe_artifact_state
# =============================================================================

class TestDescribeArtifactState:
    def test_returns_expected_keys(self):
        state = describe_artifact_state()
        for key in ("repo_root", "configs", "feature_store", "attack_matrix",
                    "expected_outputs", "manifest_datasets"):
            assert key in state

    def test_configs_listed(self):
        state = describe_artifact_state()
        for name in ("default_e3_config", "aee", "qhe", "fl_dp",
                     "sandbox_mdp", "riemannian_metrics", "attack_suite"):
            assert name in state["configs"]
            assert "path" in state["configs"][name]
            assert "exists" in state["configs"][name]

    def test_feature_store_listed(self):
        state = describe_artifact_state()
        for proto in PROTOCOLS:
            assert proto in state["feature_store"]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
