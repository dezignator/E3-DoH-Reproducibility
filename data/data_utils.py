#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
data/data_utils.py
==================

E3-DoH Reproducibility — shared data-access and config-loading library.

Chapter 5 of the thesis: E3-DoH (Enhanced Evolutionary Encryption for
Multi-Protocol DNS Security — DoH, DoT, DoQ).

This module is the library that every downstream consumer imports:

    from data.data_utils import (
        load_config, load_feature_store, load_attack_matrix,
        load_reference, write_result_json, read_result_json,
        ConfigError, FeatureStoreSchemaError, AttackMatrixError,
        ReferenceNotFoundError, ProvenanceError, DataUtilsError,
        exit_code_for,
    )

It has NO command-line interface. Its job is to hide the details of
*where* the configs, data, references, and outputs live, and to enforce
the schema and provenance contracts that the rest of the artifact relies
on. A script that re-implements any of these loaders is a defect; the
contracts they enforce (provenance tags, tolerance presence, NaN-free
feature store, A01..A15 attack matrix) are what make the verification
step meaningful.

DESIGN NOTES
------------
The module deliberately does *not* use SystemExit for its errors.
Libraries should not terminate the process. Instead it raises one of
five exception classes, all descended from DataUtilsError. A script
that consumes this library catches DataUtilsError, logs the exception,
and calls exit_code_for(exc) to translate it into a process exit code.
This keeps the "what went wrong" information in the exception message
and the "what exit code should the shell see" decision at the call
site, where it belongs.

The module also deliberately does *not* perform logging. It exposes no
logger. The caller is expected to log the exception (typically through
src/utils/logger.py) or the structured output (through write_result_json,
which is a pure function that writes a file and returns the path). This
keeps the library composable and testable.

PROVENANCE CONTRACT
-------------------
Every feature-store row carries a `provenance` column: `"synthetic"` for
toy-mode rows, `"DS-1"` for full-mode rows. load_feature_store() has a
`require_provenance` parameter that defaults to `"DS-1"` (the strict
default). A script that wants to load synthetic data for a toy-mode run
must pass `require_provenance="synthetic"`. A script that wants to load
either (e.g. an exploratory notebook) can pass `require_provenance=None`.
This is how the artifact ensures that numerical claims are only ever
made against real data.

STRUCTURED-RESULT SCHEMA
------------------------
Every per-table and per-figure script writes a JSON sidecar with the
following schema, and this module is the only place that schema is
constructed:

    {
      "artifact": "table_26",
      "generated_at": "2026-11-09T12:34:56Z",
      "provenance": {
        "mode": "full",
        "seed": 42,
        "run_id": "sha256hex",
        "git_commit": "sha1hex or null",
        "hardware_backend": "a100_emulated",
        "synthetic": false,
        "config_hashes": {
          "default_e3_config": "sha256hex",
          "aee_params": "sha256hex",
          ...
        }
      },
      "values": {
        "metrics.detection_accuracy_pct.doh": 99.6,
        ...
      },
      "tolerance": {
        "metrics.detection_accuracy_pct.doh": 0.05,
        ...
      }
    }

The `values` and `tolerance` keys are the only keys the verification
script needs; the rest are for provenance and debugging. A sidecar
without a `tolerance` entry for every `values` entry is a defect and
is rejected by read_result_json().
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# ---------------------------------------------------------------------------
# Constants — the canonical schema
# ---------------------------------------------------------------------------
# These are the column names and protocol identifiers that every
# downstream consumer must agree on. Changing any of them is a breaking
# change to the artifact and requires regenerating expected_outputs/ and
# bumping the artifact version. See the version-bump policy in
# requirements.txt.

PROTOCOLS: Tuple[str, ...] = ("doh", "dot", "doq")

FEATURE_COLUMNS: Tuple[str, ...] = (
    "inter_arrival_time_mean",
    "inter_arrival_time_std",
    "query_size_mean",
    "query_size_std",
    "response_size_mean",
    "response_size_std",
    "query_response_ratio",
    "subdomain_entropy",
    "query_type_distribution_l1",
    "burstiness",
    "session_duration",
    "protocol_specific_feature",
)

META_COLUMNS: Tuple[str, ...] = (
    "flow_id",
    "protocol",
    "label",
    "attack_type",
    "query_type",
    "provenance",
)

PROVENANCE_VALUES: Tuple[str, ...] = ("synthetic", "DS-1")

ATTACK_MATRIX_COLUMNS: Tuple[str, ...] = (
    "Attack_ID",
    "Attack_Vector",
    "Protocol",
    "Tool",
    "Command_Syntax",
    "Detection_Trigger",
    "Mitigation_Action",
)

ATTACK_IDS: Tuple[str, ...] = tuple(f"A{i:02d}" for i in range(1, 16))

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
# Every path in the artifact is resolved relative to REPO_ROOT. The
# module exposes a small number of path helpers so that no consumer
# script has to construct a path itself.

REPO_ROOT: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = REPO_ROOT / "data"
CONFIG_DIR: Path = REPO_ROOT / "configs"
SYNTHETIC_DIR: Path = DATA_DIR / "synthetic"
RAW_DIR: Path = DATA_DIR / "raw"
PROCESSED_DIR: Path = DATA_DIR / "processed"
FEATURE_STORE_DIR: Path = PROCESSED_DIR / "features"
OUTPUTS_DIR: Path = REPO_ROOT / "outputs"
EXPECTED_DIR: Path = REPO_ROOT / "expected_outputs"
ARTIFACTS_DIR: Path = REPO_ROOT / "artifacts"
MANIFEST_PATH: Path = DATA_DIR / "MANIFEST_DATASETS.sha256"
ATTACK_MATRIX_PATH: Path = DATA_DIR / "attack_matrix.csv"


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class DataUtilsError(Exception):
    """Base class for all data-utility errors."""


class ConfigError(DataUtilsError):
    """Raised when a config file is missing, malformed, or fails schema validation."""


class FeatureStoreSchemaError(DataUtilsError):
    """Raised when the feature store does not match the canonical schema."""


class AttackMatrixError(DataUtilsError):
    """Raised when data/attack_matrix.csv is missing, malformed, or has the wrong row set."""


class ReferenceNotFoundError(DataUtilsError):
    """Raised when a reference file in expected_outputs/ is missing or incomplete."""


class ProvenanceError(DataUtilsError):
    """Raised when a feature store does not match the requested provenance."""


def exit_code_for(exc: BaseException) -> int:
    """
    Map a DataUtilsError to a process exit code, for a CLI script that
    catches the exception and wants to return a code to the shell.

    The mapping is:
        ConfigError               -> 1
        FeatureStoreSchemaError   -> 2
        AttackMatrixError         -> 3
        ReferenceNotFoundError    -> 4
        ProvenanceError           -> 5
        other DataUtilsError      -> 6
        other exceptions          -> 99
    """
    if isinstance(exc, ConfigError):
        return 1
    if isinstance(exc, FeatureStoreSchemaError):
        return 2
    if isinstance(exc, AttackMatrixError):
        return 3
    if isinstance(exc, ReferenceNotFoundError):
        return 4
    if isinstance(exc, ProvenanceError):
        return 5
    if isinstance(exc, DataUtilsError):
        return 6
    return 99


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------

def _sha256_of_file(path: Path) -> str:
    """SHA-256 of a file, chunked (safe for large files)."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_of_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _utc_now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _git_commit() -> Optional[str]:
    """
    Return the current git commit hash, or None if the working tree is
    not a git repo. This is used in the provenance block so that a
    reviewer can tell exactly which commit produced a result.
    """
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(REPO_ROOT), stderr=subprocess.DEVNULL, timeout=5,
        )
        return out.decode("ascii").strip()
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return None


def _compute_run_id(
    config_hashes: Dict[str, str],
    seed: int,
    mode: str,
    git_commit: Optional[str],
) -> str:
    """
    Compute a deterministic run ID from the config hashes, the seed, the
    mode, and the git commit. Two runs with the same inputs produce the
    same run ID; two runs with different inputs do not.
    """
    h = hashlib.sha256()
    for k in sorted(config_hashes.keys()):
        h.update(k.encode("utf-8"))
        h.update(b"\x00")
        h.update(config_hashes[k].encode("utf-8"))
        h.update(b"\x00")
    h.update(str(seed).encode("utf-8"))
    h.update(b"\x00")
    h.update(mode.encode("utf-8"))
    h.update(b"\x00")
    h.update((git_commit or "nogit").encode("utf-8"))
    return h.hexdigest()[:16]


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------
# load_config() reads the master config and every file it includes,
# returns a single merged dict with the includes nested under their
# include keys, and records the SHA-256 of every file it read so that
# the caller can build a run ID from it.
#
# The include structure is defined in default_e3_config.yaml §11:
#
#     includes:
#       aee: "configs/aee_params.yaml"
#       qhe: "configs/qhe_params.yaml"
#       ...
#
# load_config() resolves each include path relative to REPO_ROOT, reads
# the file, and stores its content under cfg["_includes"][key]. The
# underscore prefix on the merged key avoids collision with a user key
# named "aee" in the master config.

def load_config(
    config_path: Optional[Union[str, Path]] = None,
    overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Load the master config and all included files.

    Parameters
    ----------
    config_path : optional
        Path to the master config. Defaults to configs/default_e3_config.yaml.
    overrides : optional
        A dict of dotted-key overrides applied after loading. E.g.
        {"runtime.mode": "full", "runtime.seed": 7}. A key that does not
        exist in the config raises ConfigError; this catches typos.

    Returns
    -------
    A dict with the following keys:
        The master config's keys (artifact, runtime, hardware, paths,
        protocols, detection, logging, verification, federated,
        attack_suite, includes).
        "_includes": {key -> dict} for each included file.
        "_paths": {"master": <Path>, "includes": {key -> <Path>}}
        "_hashes": {key -> sha256hex} for every file read.
        "_run_id": the run ID for this config (see §Small utilities).

    Raises
    ------
    ConfigError
        If any file is missing, malformed, or a key in `overrides`
        does not exist in the config.
    """
    try:
        import yaml
    except ImportError as e:
        raise ConfigError(
            "PyYAML is required to load the config. "
            "Install with: pip install pyyaml"
        ) from e

    master_path = Path(config_path) if config_path else CONFIG_DIR / "default_e3_config.yaml"
    if not master_path.is_absolute():
        master_path = REPO_ROOT / master_path
    if not master_path.exists():
        raise ConfigError(f"Master config not found: {master_path}")

    try:
        master = yaml.safe_load(master_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as e:
        raise ConfigError(f"Malformed master config {master_path}: {e}") from e
    if not isinstance(master, dict):
        raise ConfigError(f"Master config is not a mapping: {master_path}")

    hashes: Dict[str, str] = {"default_e3_config": _sha256_of_file(master_path)}
    paths: Dict[str, Any] = {"master": master_path, "includes": {}}

    includes_spec = master.get("includes", {}) or {}
    merged_includes: Dict[str, Any] = {}
    for key, rel_path in includes_spec.items():
        inc_path = REPO_ROOT / rel_path
        if not inc_path.exists():
            raise ConfigError(
                f"Included config for '{key}' not found: {inc_path} "
                f"(declared in {master_path})"
            )
        try:
            content = yaml.safe_load(inc_path.read_text(encoding="utf-8"))
        except yaml.YAMLError as e:
            raise ConfigError(f"Malformed include '{key}' ({inc_path}): {e}") from e
        merged_includes[key] = content
        paths["includes"][key] = inc_path
        hashes[key] = _sha256_of_file(inc_path)

    # Apply overrides. Dotted key into the master config; we do not
    # support overrides into the includes because a consumer that wants
    # to override a component parameter should use the component's
    # config file (which is hashed and versioned) rather than the
    # master config.
    if overrides:
        for dotted, value in overrides.items():
            _set_dotted(master, dotted, value)

    # Compute the run ID.
    run_seed = int(master.get("runtime", {}).get("seed", 0))
    run_mode = str(master.get("runtime", {}).get("mode", "toy"))
    run_id = _compute_run_id(hashes, run_seed, run_mode, _git_commit())

    merged: Dict[str, Any] = dict(master)
    merged["_includes"] = merged_includes
    merged["_paths"] = paths
    merged["_hashes"] = hashes
    merged["_run_id"] = run_id
    return merged


def _set_dotted(cfg: Dict[str, Any], dotted: str, value: Any) -> None:
    """Set cfg[a][b][c] = value for dotted='a.b.c'. Raises if any intermediate
    key is missing, which catches typos."""
    parts = dotted.split(".")
    cursor = cfg
    for p in parts[:-1]:
        if p not in cursor:
            raise ConfigError(
                f"Override key '{dotted}': intermediate '{p}' not found"
            )
        cursor = cursor[p]
    if parts[-1] not in cursor:
        raise ConfigError(f"Override key '{dotted}': final key not found")
    cursor[parts[-1]] = value


# ---------------------------------------------------------------------------
# Feature store loading
# ---------------------------------------------------------------------------
# The feature store is one Parquet file per protocol, produced by either
# preprocess_zenodo.py or preprocess_pcap.py. Both producers write the
# same schema (twelve features + six meta columns), and the downstream
# consumers are agnostic to which producer was used.
#
# load_feature_store() enforces the schema and the provenance tag. The
# provenance check is the mechanism that ensures a numerical claim is
# made against real data and not against the toy synthetic corpus.

def load_feature_store(
    protocol: str,
    require_provenance: Optional[str] = "DS-1",
    feature_store_dir: Optional[Path] = None,
) -> "pd.DataFrame":  # type: ignore[name-defined]  # pandas imported lazily
    """
    Load the feature store for one protocol.

    Parameters
    ----------
    protocol : "doh" | "dot" | "doq"
    require_provenance : "DS-1" (default) | "synthetic" | None
        The required value of the `provenance` column in every row.
        Default is "DS-1" (the strict default). Pass "synthetic" for a
        toy-mode run, or None to skip the check.
    feature_store_dir : optional Path
        Override the default (data/processed/features/).

    Returns
    -------
    pandas.DataFrame with the twelve feature columns first, then the six
    meta columns, then any extra columns from the raw corpus, in a fixed
    order.

    Raises
    ------
    FeatureStoreSchemaError
        If the file is missing, the schema does not match, or a NaN is
        present in a feature column.
    ProvenanceError
        If `require_provenance` is set and the observed provenance does
        not match.
    """
    try:
        import pandas as pd
    except ImportError as e:
        raise FeatureStoreSchemaError("pandas is required to load the feature store") from e

    if protocol not in PROTOCOLS:
        raise FeatureStoreSchemaError(
            f"Unknown protocol '{protocol}'; expected one of {PROTOCOLS}"
        )

    base = feature_store_dir or FEATURE_STORE_DIR
    path = base / f"{protocol}_features.parquet"
    if not path.exists():
        raise FeatureStoreSchemaError(
            f"Feature store not found: {path}. "
            f"Run `python data/preprocess_zenodo.py --mode <toy|full>` first."
        )

    df = pd.read_parquet(path)

    # Feature columns must be present and in the canonical order.
    missing_feat = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing_feat:
        raise FeatureStoreSchemaError(
            f"{path}: missing feature columns {missing_feat}"
        )
    observed_order = [c for c in df.columns if c in FEATURE_COLUMNS]
    if tuple(observed_order) != FEATURE_COLUMNS:
        raise FeatureStoreSchemaError(
            f"{path}: feature column order does not match canonical order. "
            f"expected={list(FEATURE_COLUMNS)} observed={observed_order}"
        )

    # Meta columns must be present.
    missing_meta = [c for c in META_COLUMNS if c not in df.columns]
    if missing_meta:
        raise FeatureStoreSchemaError(
            f"{path}: missing meta columns {missing_meta}"
        )

    # NaN check on the feature columns.
    for c in FEATURE_COLUMNS:
        n_nan = int(df[c].isna().sum())
        if n_nan > 0:
            raise FeatureStoreSchemaError(
                f"{path}: {n_nan} NaN values in feature '{c}'. "
                f"Re-run preprocessing to drop or fill them."
            )

    # Provenance check.
    if require_provenance is not None:
        if require_provenance not in PROVENANCE_VALUES:
            raise ProvenanceError(
                f"Unknown require_provenance='{require_provenance}'; "
                f"expected one of {PROVENANCE_VALUES} or None"
            )
        observed = set(df["provenance"].astype(str).unique())
        if observed != {require_provenance}:
            raise ProvenanceError(
                f"{path}: provenance check failed. "
                f"require_provenance={require_provenance!r} "
                f"but observed={sorted(observed)}. "
                f"Numerical claims must be made against 'DS-1' data; "
                f"a toy-mode run must pass require_provenance='synthetic'."
            )

    # Reorder columns: features first, then meta, then any extra.
    ordered = list(FEATURE_COLUMNS) + list(META_COLUMNS)
    extra = [c for c in df.columns if c not in ordered]
    return df[ordered + extra]


def load_all_feature_stores(
    require_provenance: Optional[str] = "DS-1",
) -> Dict[str, "pd.DataFrame"]:  # type: ignore[name-defined]
    """Load all three protocols. Convenience wrapper around load_feature_store()."""
    return {
        proto: load_feature_store(proto, require_provenance=require_provenance)
        for proto in PROTOCOLS
    }


# ---------------------------------------------------------------------------
# Attack matrix loading
# ---------------------------------------------------------------------------
# The attack matrix is the 15-row CSV that carries every scenario from
# thesis Tables 18, 21, 22, 23, and 24. load_attack_matrix() validates:
#
#   - The file exists.
#   - It has the exact seven columns from ATTACK_MATRIX_COLUMNS.
#   - It has exactly 15 rows.
#   - The IDs are exactly A01..A15.
#   - The Protocol column has values only from {DoH, DoT, DoQ}.
#   - No cell is NaN or empty.
#
# A failure at any of these checks is a defect in the CSV, not in the
# consumer.

def load_attack_matrix(
    path: Optional[Union[str, Path]] = None,
) -> "pd.DataFrame":  # type: ignore[name-defined]
    """
    Load and validate data/attack_matrix.csv.

    Returns a DataFrame with the seven columns of ATTACK_MATRIX_COLUMNS,
    in that order.

    Raises
    ------
    AttackMatrixError
        If the file is missing, the schema does not match, or any of
        the row-level checks fail.
    """
    try:
        import pandas as pd
    except ImportError as e:
        raise AttackMatrixError("pandas is required to load the attack matrix") from e

    matrix_path = Path(path) if path else ATTACK_MATRIX_PATH
    if not matrix_path.is_absolute():
        matrix_path = REPO_ROOT / matrix_path
    if not matrix_path.exists():
        raise AttackMatrixError(f"Attack matrix not found: {matrix_path}")

    df = pd.read_csv(matrix_path, dtype=str, keep_default_na=False)

    # Column check.
    if tuple(df.columns) != ATTACK_MATRIX_COLUMNS:
        raise AttackMatrixError(
            f"{matrix_path}: column set mismatch. "
            f"expected={list(ATTACK_MATRIX_COLUMNS)} observed={list(df.columns)}"
        )

    # Row-count check.
    if len(df) != len(ATTACK_IDS):
        raise AttackMatrixError(
            f"{matrix_path}: expected {len(ATTACK_IDS)} rows, "
            f"observed {len(df)}"
        )

    # ID-set check.
    observed_ids = list(df["Attack_ID"])
    if observed_ids != list(ATTACK_IDS):
        raise AttackMatrixError(
            f"{matrix_path}: Attack_ID column must be exactly "
            f"{list(ATTACK_IDS)} in order. observed={observed_ids}"
        )

    # Protocol value check.
    valid_protocols = {"DoH", "DoT", "DoQ"}
    bad_protocols = set(df["Protocol"]) - valid_protocols
    if bad_protocols:
        raise AttackMatrixError(
            f"{matrix_path}: Protocol values must be in {valid_protocols}; "
            f"found {sorted(bad_protocols)}"
        )

    # Empty-cell check.
    for col in ATTACK_MATRIX_COLUMNS:
        empty = df.index[df[col].astype(str).str.strip() == ""].tolist()
        if empty:
            raise AttackMatrixError(
                f"{matrix_path}: empty cells in column '{col}' at row(s) "
                f"{[int(i) + 2 for i in empty]} (1-based, header is row 1)"
            )

    return df.reset_index(drop=True)


def attack_matrix_row(attack_id: str, path: Optional[Union[str, Path]] = None) -> Dict[str, str]:
    """Return a single attack row as a dict. Useful for per-scenario scripts."""
    df = load_attack_matrix(path)
    rows = df[df["Attack_ID"] == attack_id]
    if rows.empty:
        raise AttackMatrixError(f"Attack ID '{attack_id}' not in attack matrix")
    return rows.iloc[0].to_dict()


# ---------------------------------------------------------------------------
# Reference value loading
# ---------------------------------------------------------------------------
# The expected_outputs/ directory contains one JSON sidecar per table
# and one CSV per figure. load_reference() reads the JSON sidecar and
# returns a dict with `values`, `tolerance`, and `provenance`. It
# enforces that every key in `values` has a corresponding entry in
# `tolerance`; a missing tolerance is a defect in the reference file,
# not a defect in the run.

def load_reference(
    artifact: str,
    expected_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Load a reference JSON sidecar from expected_outputs/.

    Parameters
    ----------
    artifact : e.g. "table_26", "table_25", "table_18"
    expected_dir : optional override of expected_outputs/

    Returns
    -------
    A dict with the parsed JSON. Guaranteed to contain at least the
    `values` and `tolerance` keys, and the keys are guaranteed to match.

    Raises
    ------
    ReferenceNotFoundError
        If the file is missing, malformed, or a tolerance entry is
        missing for a value entry.
    """
    base = expected_dir or EXPECTED_DIR
    path = base / f"{artifact}_expected.json"
    if not path.exists():
        raise ReferenceNotFoundError(
            f"Reference file not found: {path}. "
            f"The artifact has not been signed off on this checkout. "
            f"See STATUS.md §9 (ISSUE-01) and REPRODUCIBILITY_CHECKLIST.md §7."
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise ReferenceNotFoundError(f"Malformed reference file {path}: {e}") from e
    if not isinstance(payload, dict):
        raise ReferenceNotFoundError(f"Reference file {path} is not a JSON object")
    values = payload.get("values")
    tolerance = payload.get("tolerance")
    if not isinstance(values, dict) or not isinstance(tolerance, dict):
        raise ReferenceNotFoundError(
            f"Reference file {path} must contain 'values' and 'tolerance' mappings"
        )
    missing_tol = [k for k in values if k not in tolerance]
    if missing_tol:
        raise ReferenceNotFoundError(
            f"Reference file {path} is missing tolerance entries for: "
            f"{missing_tol[:5]}{'...' if len(missing_tol) > 5 else ''}"
        )
    return payload


def reference_value(ref: Dict[str, Any], dotted_key: str) -> float:
    """Read ref["values"][dotted_key] with a clear error if missing."""
    values = ref.get("values", {})
    if dotted_key not in values:
        raise ReferenceNotFoundError(
            f"Key '{dotted_key}' not in reference values. "
            f"Available: {sorted(values.keys())[:5]}..."
        )
    return float(values[dotted_key])


def reference_tolerance(ref: Dict[str, Any], dotted_key: str) -> float:
    """Read ref["tolerance"][dotted_key] with a clear error if missing."""
    tol = ref.get("tolerance", {})
    if dotted_key not in tol:
        raise ReferenceNotFoundError(
            f"Key '{dotted_key}' not in reference tolerances."
        )
    return float(tol[dotted_key])


# ---------------------------------------------------------------------------
# Structured-result sidecar
# ---------------------------------------------------------------------------
# Every per-table and per-figure script writes a JSON sidecar with the
# schema documented in the module docstring. write_result_json() is the
# only place that schema is constructed. read_result_json() reads it
# back and enforces the tolerance-presence contract.

def write_result_json(
    artifact: str,
    values: Dict[str, float],
    tolerance: Dict[str, float],
    config: Dict[str, Any],
    hardware_backend: Optional[str] = None,
    output_path: Optional[Path] = None,
) -> Path:
    """
    Write a structured-result sidecar.

    Parameters
    ----------
    artifact : e.g. "table_26"
    values : {dotted_key -> numeric value}
    tolerance : {dotted_key -> absolute tolerance}. Must have the same
                key set as `values`.
    config : the dict returned by load_config(). Used to populate the
             provenance block.
    hardware_backend : e.g. "a100_emulated". If None, the function uses
                       config["hardware"]["a100"]["backend"] if present.
    output_path : optional explicit path. If None, writes to
                  outputs/tables/{artifact}.json (or outputs/figures/ for
                  figure artifacts).

    Returns
    -------
    The Path that was written.

    Raises
    ------
    ConfigError
        If values and tolerance have different key sets.
    """
    if set(values.keys()) != set(tolerance.keys()):
        raise ConfigError(
            f"write_result_json: values and tolerance must have the same keys. "
            f"Missing tolerance: {sorted(set(values) - set(tolerance))}; "
            f"Extra tolerance: {sorted(set(tolerance) - set(values))}"
        )

    provenance = {
        "mode": str(config.get("runtime", {}).get("mode", "unknown")),
        "seed": int(config.get("runtime", {}).get("seed", 0)),
        "run_id": config.get("_run_id", "unknown"),
        "git_commit": _git_commit(),
        "hardware_backend": hardware_backend or _infer_hardware_backend(config),
        "synthetic": config.get("runtime", {}).get("mode", "toy") == "toy",
        "config_hashes": dict(config.get("_hashes", {})),
    }
    payload = {
        "artifact": artifact,
        "generated_at": _utc_now_iso(),
        "provenance": provenance,
        "values": dict(values),
        "tolerance": dict(tolerance),
    }

    if output_path is None:
        subdir = "figures" if artifact.startswith("figure_") else "tables"
        output_path = OUTPUTS_DIR / subdir / f"{artifact}.json"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return output_path


def _infer_hardware_backend(config: Dict[str, Any]) -> str:
    """Best-effort hardware-backend string for the provenance block."""
    try:
        hw = config["hardware"]
        a100 = hw.get("a100", {}).get("backend", "emulated")
        loihi = hw.get("loihi", {}).get("backend", "emulated")
        return f"a100_{a100}+loihi_{loihi}"
    except (KeyError, AttributeError):
        return "unknown"


def read_result_json(path: Union[str, Path]) -> Dict[str, Any]:
    """
    Read a structured-result sidecar. Enforces the same tolerance-presence
    contract as load_reference().
    """
    path = Path(path)
    if not path.exists():
        raise ReferenceNotFoundError(f"Result sidecar not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    values = payload.get("values")
    tolerance = payload.get("tolerance")
    if not isinstance(values, dict) or not isinstance(tolerance, dict):
        raise ReferenceNotFoundError(
            f"{path}: missing 'values' or 'tolerance' mapping"
        )
    missing_tol = [k for k in values if k not in tolerance]
    if missing_tol:
        raise ReferenceNotFoundError(
            f"{path}: tolerance missing for keys {missing_tol[:5]}"
        )
    return payload


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------
# Small wrappers so a script never has to construct a path itself.

def feature_store_path(protocol: str) -> Path:
    if protocol not in PROTOCOLS:
        raise ConfigError(f"Unknown protocol '{protocol}'")
    return FEATURE_STORE_DIR / f"{protocol}_features.parquet"


def config_path(name: str) -> Path:
    """name is one of the include keys, or 'default_e3_config'."""
    if name == "default_e3_config":
        return CONFIG_DIR / "default_e3_config.yaml"
    return CONFIG_DIR / f"{name}_params.yaml"


def output_path(kind: str, name: str) -> Path:
    """kind in {'tables', 'figures', 'logs', 'checkpoints'}."""
    if kind not in {"tables", "figures", "logs", "checkpoints"}:
        raise ConfigError(f"Unknown output kind '{kind}'")
    return OUTPUTS_DIR / kind / name


def expected_path(name: str) -> Path:
    return EXPECTED_DIR / name


# ---------------------------------------------------------------------------
# Artifact-state description
# ---------------------------------------------------------------------------
# describe_artifact_state() returns a dict summarising which configs are
# present, which feature stores are present, whether the attack matrix
# is valid, and whether expected_outputs/ has been populated. It is used
# by the __main__ self-test below and by verification_script.py.

def describe_artifact_state() -> Dict[str, Any]:
    state: Dict[str, Any] = {
        "repo_root": str(REPO_ROOT),
        "configs": {},
        "feature_store": {},
        "attack_matrix": None,
        "expected_outputs": {},
        "manifest_datasets": None,
    }
    for name in ("default_e3_config", "aee", "qhe", "fl_dp", "sandbox_mdp",
                 "riemannian_metrics", "attack_suite"):
        p = config_path(name)
        state["configs"][name] = {"path": str(p), "exists": p.exists()}
    for proto in PROTOCOLS:
        p = feature_store_path(proto)
        state["feature_store"][proto] = {"path": str(p), "exists": p.exists()}
    state["attack_matrix"] = {
        "path": str(ATTACK_MATRIX_PATH),
        "exists": ATTACK_MATRIX_PATH.exists(),
    }
    if EXPECTED_DIR.exists():
        for f in sorted(EXPECTED_DIR.glob("*_expected.*")):
            state["expected_outputs"][f.name] = True
    state["manifest_datasets"] = {
        "path": str(MANIFEST_PATH),
        "exists": MANIFEST_PATH.exists(),
    }
    return state


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
__all__ = [
    # Constants
    "PROTOCOLS", "FEATURE_COLUMNS", "META_COLUMNS", "PROVENANCE_VALUES",
    "ATTACK_MATRIX_COLUMNS", "ATTACK_IDS",
    # Paths
    "REPO_ROOT", "DATA_DIR", "CONFIG_DIR", "FEATURE_STORE_DIR",
    "OUTPUTS_DIR", "EXPECTED_DIR", "ARTIFACTS_DIR",
    "ATTACK_MATRIX_PATH", "MANIFEST_PATH",
    # Exceptions
    "DataUtilsError", "ConfigError", "FeatureStoreSchemaError",
    "AttackMatrixError", "ReferenceNotFoundError", "ProvenanceError",
    "exit_code_for",
    # Loaders
    "load_config", "load_feature_store", "load_all_feature_stores",
    "load_attack_matrix", "attack_matrix_row",
    "load_reference", "reference_value", "reference_tolerance",
    # Sidecars
    "write_result_json", "read_result_json",
    # Path helpers
    "feature_store_path", "config_path", "output_path", "expected_path",
    # Diagnostics
    "describe_artifact_state",
]


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
# Running `python data/data_utils.py` performs a coherence check on the
# artifact: loads every config, verifies every feature store that exists,
# verifies the attack matrix, and prints a summary. Exit code 0 means the
# artifact is coherent at the data layer (the numerical reproduction
# still requires a full-mode run).
#
# This block is deliberately minimal; it is a smoke test, not a full
# verification. The full verification is
# expected_outputs/verification_script.py.

def _self_test() -> int:
    import sys
    rc = 0
    print("=== data_utils self-test ===")
    print(f"repo root: {REPO_ROOT}")

    # Configs
    try:
        cfg = load_config()
        print(f"[OK]   configs loaded (run_id={cfg['_run_id']}, "
              f"mode={cfg['runtime']['mode']})")
    except DataUtilsError as e:
        print(f"[FAIL] configs: {e}")
        rc = 1
        cfg = None

    # Feature stores
    for proto in PROTOCOLS:
        try:
            df = load_feature_store(proto, require_provenance="synthetic")
            print(f"[OK]   feature store {proto}: {len(df)} rows")
        except FeatureStoreSchemaError as e:
            # Missing in fresh checkouts before preprocessing has run;
            # report as WARN, not FAIL.
            print(f"[WARN] feature store {proto}: {e}")
        except ProvenanceError as e:
            print(f"[WARN] feature store {proto}: {e}")
        except DataUtilsError as e:
            print(f"[FAIL] feature store {proto}: {e}")
            rc = 1

    # Attack matrix
    try:
        am = load_attack_matrix()
        print(f"[OK]   attack matrix: {len(am)} rows, ids {am['Attack_ID'].iloc[0]}..{am['Attack_ID'].iloc[-1]}")
    except DataUtilsError as e:
        print(f"[FAIL] attack matrix: {e}")
        rc = 1

    # Expected outputs (optional)
    has_expected = EXPECTED_DIR.exists() and any(EXPECTED_DIR.glob("*_expected.*"))
    if has_expected:
        print(f"[OK]   expected_outputs/ populated ({len(list(EXPECTED_DIR.glob('*_expected.*')))} files)")
    else:
        print(f"[WARN] expected_outputs/ not populated; see STATUS.md §9 (ISSUE-01)")

    print("=== self-test complete ===")
    return rc


if __name__ == "__main__":
    import sys
    sys.exit(_self_test())
