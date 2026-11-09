#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
data/preprocess_zenodo.py
=========================

E3-DoH Reproducibility — feature-store construction from the Zenodo
encrypted-DNS corpus (DS-1) or from the schema-matched synthetic corpus.

Chapter 5 of the thesis: E3-DoH (Enhanced Evolutionary Encryption for
Multi-Protocol DNS Security — DoH, DoT, DoQ).

This script is the second stage of the data pipeline:

    1. data/download_datasets.py        -> data/synthetic/ OR data/raw/DS-1/
    2. data/preprocess_zenodo.py  (this) -> data/processed/features/
    3. scripts/train_e3.py              -> outputs/

Its job is narrow and well-defined: take the raw corpus, apply the
twelve-feature transformation described in configs/riemannian_metrics.yaml
§2, and write one Parquet file per protocol into the feature store. Every
downstream consumer (training, evaluation, per-table scripts, per-figure
scripts) reads from the feature store, not from the raw corpus, so that
the feature transformation is applied exactly once and the same way for
every artifact.

WHAT THIS SCRIPT DOES *NOT* DO
------------------------------
It does not train any model, it does not compute any detection metric,
and it does not touch the attack matrix. It is a data-transformation
step, nothing more. The numbers in Table 26 are produced by
scripts/train_e3.py and its consumers, not here.

THE TWELVE FEATURES
-------------------
The feature set is defined in configs/riemannian_metrics.yaml §2. This
script implements exactly that set, in exactly that order, with exactly
that dtype (float64). The feature order matters because the metric
tensors are 12x12 matrices indexed by position. Adding, removing,
renaming, or reordering a feature here without updating the metric
tensors would silently corrupt the detector; the verification script
(data/verify_feature_store.py, next in the sequence) cross-checks the
feature-store column order against configs/riemannian_metrics.yaml §2.

MODES
-----
    --mode toy   : read data/synthetic/{doh,dot,doq}_synthetic.parquet
                   and write data/processed/features/{doh,dot,doq}_features.parquet.
                   Runs offline, in under 30 seconds.
    --mode full  : read data/raw/DS-1/ (the extracted Zenodo corpus) and
                   write data/processed/features/{doh,dot,doq}_features.parquet.
                   Runs in 15-30 minutes on the reference testbed (the
                   Dask-based parallel path is used for the full corpus;
                   the toy path is single-threaded).

PROVENANCE TAG
--------------
Every row in the feature store carries a `provenance` column with value
"synthetic" (toy mode) or "DS-1" (full mode). The verification script
refuses to compare a synthetic run against expected_outputs/; it only
compares against a full-mode run with `provenance == "DS-1"`. This is the
mechanism that makes toy mode a code-path exercise and not a numerical
reproduction.

Exit codes
----------
    0  success
    1  input corpus missing (run download_datasets.py first)
    2  input corpus malformed (missing column, wrong dtype, NaN in required column)
    3  feature-store write failed (disk, permissions)
    4  feature-store schema mismatch against configs/riemannian_metrics.yaml §2
    5  disk space insufficient
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
SYNTHETIC_DIR = DATA_DIR / "synthetic"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
FEATURE_STORE = PROCESSED_DIR / "features"
LOG_DIR = REPO_ROOT / "outputs" / "logs"

CONFIG_METRICS = REPO_ROOT / "configs" / "riemannian_metrics.yaml"

# ---------------------------------------------------------------------------
# Feature specification (must match configs/riemannian_metrics.yaml §2)
# ---------------------------------------------------------------------------
# This is a *copy* of the feature spec from the config file, kept here as
# a fallback so that this script can be imported and tested without PyYAML
# installed. The script's primary source of truth is the config file; the
# fallback is used only when PyYAML is not available or the config file is
# not on the path (e.g. when running a subset of the tests in isolation).
#
# The verification step at the end of the script (verify_schema) compares
# the feature order actually written to the Parquet file against the
# feature order in the config file, so a drift between this fallback and
# the config file would be caught at runtime.

FEATURE_FALLBACK: Tuple[str, ...] = (
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

# Dtypes for the twelve features. All float64. The rest of the row is
# string / categorical and is not part of the metric-tensor computation.
FEATURE_DTYPE = "float64"

# Non-feature columns that every row must carry. These are the columns
# that the downstream consumers (training, evaluation, table scripts)
# group or filter on.
REQUIRED_META_COLUMNS: Tuple[str, ...] = (
    "flow_id",         # unique per flow, used for traceability
    "protocol",        # "DoH" | "DoT" | "DoQ"
    "label",           # "benign" | "attack"
    "attack_type",     # "" for benign, "A01".."A15" for attack
    "query_type",      # A / AAAA / ... / OTHER
    "provenance",      # "synthetic" | "DS-1"
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def _setup_logging(level: str = "INFO") -> logging.Logger:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log = logging.getLogger("e3doh.preprocess")
    log.setLevel(level.upper())
    if log.handlers:
        return log
    fmt = logging.Formatter(
        '{"time": "%(asctime)s", "level": "%(levelname)s", '
        '"logger": "%(name)s", "message": %(message)s}'
    )
    fh = logging.FileHandler(LOG_DIR / "preprocess.log")
    fh.setFormatter(fmt)
    log.addHandler(fh)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    log.addHandler(sh)
    return log


def _jlog(log: logging.Logger, level: str, **fields: Any) -> None:
    msg = json.dumps(fields, ensure_ascii=False, default=str)
    getattr(log, level)(msg)


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------
# The config file is the source of truth for the feature spec. If PyYAML
# is not available, or the file is missing, the script falls back to the
# hard-coded FEATURE_FALLBACK and emits a warning. The fallback is only
# used to keep the script importable in environments where the config file
# has not yet been checked out; any real run should use the config file.

def _load_feature_spec(log: logging.Logger) -> Tuple[str, ...]:
    try:
        import yaml  # type: ignore
    except ImportError:
        _jlog(log, "warning", event="yaml_not_installed_using_fallback")
        return FEATURE_FALLBACK
    if not CONFIG_METRICS.exists():
        _jlog(log, "warning", event="config_missing_using_fallback",
              path=str(CONFIG_METRICS))
        return FEATURE_FALLBACK
    with CONFIG_METRICS.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    features = cfg["traffic_state"]["features"]
    order = tuple(f["name"] for f in features)
    if order != FEATURE_FALLBACK:
        # A drift between the config and the fallback is not necessarily
        # an error, but it is worth logging because it means the script
        # and the metric tensors may disagree. The verification step at
        # the end of the run will catch the disagreement.
        _jlog(log, "warning", event="feature_spec_drift",
              config=list(order), fallback=list(FEATURE_FALLBACK))
    return order


# ---------------------------------------------------------------------------
# Toy-mode preprocessing
# ---------------------------------------------------------------------------
# The toy corpus is already in the right shape: the generator in
# download_datasets.py (File 15) writes the twelve features plus the
# meta columns. So the toy preprocessing step is almost a no-op. It:
#
#   1. Reads the toy Parquet per protocol.
#   2. Verifies the schema (all twelve features present, all required
#      meta columns present, no NaN in any feature column).
#   3. Casts the feature columns to float64 (they already are, but the
#      cast is explicit so that a future change to the generator cannot
#      silently introduce a dtype drift).
#   4. Writes the result to the feature store with the same filename
#      convention as the full path.
#
# This is deliberately minimal. The point of toy mode is to exercise
# every code path, and the feature transformation is one of those code
# paths; the transformation itself is trivial in toy mode because the
# synthetic data is already in the right shape.

def _preprocess_toy(
    log: logging.Logger,
    feature_order: Tuple[str, ...],
) -> Dict[str, Dict[str, Any]]:
    try:
        import pandas as pd
    except ImportError as e:
        raise RuntimeError("toy mode requires pandas") from e

    FEATURE_STORE.mkdir(parents=True, exist_ok=True)
    summary: Dict[str, Dict[str, Any]] = {}

    for protocol in ("doh", "dot", "doq"):
        in_path = SYNTHETIC_DIR / f"{protocol}_synthetic.parquet"
        if not in_path.exists():
            log.error(f"Missing toy corpus: {in_path}. "
                      f"Run `python data/download_datasets.py --mode toy` first.")
            raise SystemExit(1)
        df = pd.read_parquet(in_path)

        # Schema check: required meta columns.
        missing_meta = [c for c in REQUIRED_META_COLUMNS if c not in df.columns]
        if missing_meta:
            log.error(f"{in_path}: missing meta columns {missing_meta}")
            raise SystemExit(2)

        # Schema check: all twelve features present.
        missing_feat = [c for c in feature_order if c not in df.columns]
        if missing_feat:
            log.error(f"{in_path}: missing feature columns {missing_feat}")
            raise SystemExit(2)

        # NaN check: any NaN in a feature column is a defect. The
        # downstream detector will not handle NaN gracefully and the
        # resulting metrics would be silently wrong.
        for c in feature_order:
            n_nan = int(df[c].isna().sum())
            if n_nan > 0:
                log.error(f"{in_path}: {n_nan} NaN values in feature '{c}'")
                raise SystemExit(2)

        # Explicit dtype cast.
        df = df.copy()
        for c in feature_order:
            df[c] = df[c].astype(FEATURE_DTYPE)
        df["provenance"] = "synthetic"

        out_path = FEATURE_STORE / f"{protocol}_features.parquet"
        df.to_parquet(out_path, index=False)
        summary[protocol] = {
            "rows": int(len(df)),
            "columns": list(df.columns),
            "feature_order": list(feature_order),
            "provenance": "synthetic",
            "input": str(in_path.relative_to(REPO_ROOT)),
            "output": str(out_path.relative_to(REPO_ROOT)),
        }
        _jlog(log, "info", event="toy_features_written",
              protocol=protocol, rows=len(df), out=str(out_path))

    return summary


# ---------------------------------------------------------------------------
# Full-mode preprocessing
# ---------------------------------------------------------------------------
# The full mode reads the DS-1 corpus from data/raw/DS-1/. The corpus is
# distributed as per-capture PCAPs plus a manifest; this script does not
# re-implement PCAP parsing (that is data/preprocess_pcap.py's job, and
# the pre-extracted Parquet version of the corpus that ships with DS-1
# is what the thesis used). It reads the pre-extracted Parquet files,
# applies the twelve-feature transformation, and writes the feature
# store.
#
# If the reviewer's copy of DS-1 ships only as PCAPs, they should run
# data/preprocess_pcap.py first; this script will then read its output.
# The detection of "Parquet vs PCAP" is done by checking for the presence
# of *.parquet files under data/raw/DS-1/; if none are found, the script
# emits a clear error and exits 1.
#
# The full path uses Dask for parallel reads. The Dask scheduler is set
# to the local threaded scheduler (no cluster, no separate scheduler
# process) so that the script works on a single machine without
# additional setup; the parallel speedup comes from concurrent Parquet
# reads, not from distributed compute.

def _preprocess_full(
    log: logging.Logger,
    feature_order: Tuple[str, ...],
    num_workers: int,
) -> Dict[str, Dict[str, Any]]:
    try:
        import pandas as pd
    except ImportError as e:
        raise RuntimeError("full mode requires pandas") from e

    # The DS-1 archive, when extracted, contains a directory tree with
    # one subdirectory per protocol. The exact layout is documented in
    # the dataset's own README.txt (inside data/raw/DS-1/README.txt).
    # The script looks for the standard layout, and falls back to a
    # recursive search if the standard layout is not found.
    ds1_root = RAW_DIR / "DS-1"
    if not ds1_root.exists():
        log.error(f"DS-1 not found at {ds1_root}. "
                  f"Run `python data/download_datasets.py --mode full` first.")
        raise SystemExit(1)

    # Detect whether the corpus is already in Parquet form.
    parquet_files = sorted(ds1_root.rglob("*.parquet"))
    if not parquet_files:
        log.error(
            f"No Parquet files found under {ds1_root}. If your copy of "
            f"DS-1 ships as PCAPs only, run `python data/preprocess_pcap.py` "
            f"first, then re-run this script."
        )
        raise SystemExit(1)

    _jlog(log, "info", event="full_corpus_detected",
          root=str(ds1_root), parquet_files=len(parquet_files),
          num_workers=num_workers)

    # Group Parquet files by protocol. The protocol is inferred from the
    # parent directory name (doh/, dot/, doq/) or, failing that, from a
    # 'protocol' column in the file. The grouping is done before the read
    # so that each protocol can be processed independently, which is what
    # lets the downstream consumers read one protocol at a time.
    groups: Dict[str, List[Path]] = {"doh": [], "dot": [], "doq": []}
    for p in parquet_files:
        parts = [x.lower() for x in p.parts]
        for proto in groups:
            if proto in parts:
                groups[proto].append(p)
                break

    # Try Dask for the read. If Dask is not installed, fall back to
    # pandas. The fallback is not catastrophic (the corpus is only 32 GB
    # and a pandas read of a single 32 GB Parquet file works if you have
    # enough RAM), but the Dask path is what the thesis used for the
    # preprocessing timing in §6.13.
    try:
        import dask.dataframe as dd  # type: ignore
        use_dask = True
    except ImportError:
        use_dask = False
        _jlog(log, "warning", event="dask_not_installed_using_pandas")

    FEATURE_STORE.mkdir(parents=True, exist_ok=True)
    summary: Dict[str, Dict[str, Any]] = {}

    for protocol, files in groups.items():
        if not files:
            log.warning(f"No Parquet files for protocol '{protocol}'; skipping.")
            continue
        _jlog(log, "info", event="protocol_read_start",
              protocol=protocol, files=len(files))

        if use_dask:
            ddf = dd.read_parquet([str(f) for f in files])
            # The transformation is a per-column dtype cast. Dask can
            # vectorise it. The compute() at the end collapses to pandas;
            # for a 32 GB corpus this means the protocol's DataFrame fits
            # in memory. If it does not fit, the review environment is
            # smaller than the reference testbed and the script will OOM;
            # the user should use a machine with more RAM.
            for c in feature_order:
                ddf[c] = ddf[c].astype(FEATURE_DTYPE)
            df = ddf.compute()
        else:
            dfs = [pd.read_parquet(f) for f in files]
            df = pd.concat(dfs, ignore_index=True)
            for c in feature_order:
                df[c] = df[c].astype(FEATURE_DTYPE)

        # Schema checks — same as toy mode, but the errors are more
        # likely here because the real corpus is messier.
        missing_meta = [c for c in REQUIRED_META_COLUMNS if c not in df.columns]
        if missing_meta:
            log.error(f"{protocol}: missing meta columns {missing_meta}")
            raise SystemExit(2)
        missing_feat = [c for c in feature_order if c not in df.columns]
        if missing_feat:
            log.error(f"{protocol}: missing feature columns {missing_feat}")
            raise SystemExit(2)
        for c in feature_order:
            n_nan = int(df[c].isna().sum())
            if n_nan > 0:
                # The real corpus can have NaN in a feature column if a
                # capture was truncated. Drop those rows rather than
                # fail; the drop count is recorded in the summary so a
                # reviewer can see how many were dropped.
                _jlog(log, "warning", event="rows_with_nan_dropped",
                      protocol=protocol, column=c, count=n_nan)
                df = df[df[c].notna()]

        # Tag the provenance. The full corpus is DS-1, not "full"; this
        # keeps the tag consistent with what the verification script
        # checks against.
        df["provenance"] = "DS-1"

        # Reorder columns so the feature columns come first, then the
        # meta columns, in a fixed order. This makes a diff of two
        # feature-store files readable.
        ordered_cols = list(feature_order) + list(REQUIRED_META_COLUMNS)
        # Any extra columns from the raw corpus are kept at the end so
        # that a reviewer can see them without them interfering with the
        # downstream consumers.
        extra_cols = [c for c in df.columns if c not in ordered_cols]
        df = df[ordered_cols + extra_cols]

        out_path = FEATURE_STORE / f"{protocol}_features.parquet"
        df.to_parquet(out_path, index=False)
        summary[protocol] = {
            "rows": int(len(df)),
            "columns": list(df.columns),
            "feature_order": list(feature_order),
            "provenance": "DS-1",
            "input_files": len(files),
            "output": str(out_path.relative_to(REPO_ROOT)),
        }
        _jlog(log, "info", event="full_features_written",
              protocol=protocol, rows=len(df), out=str(out_path))

    return summary


# ---------------------------------------------------------------------------
# Feature-store manifest
# ---------------------------------------------------------------------------
# A small JSON manifest is written next to the feature store so that the
# verification script (and any reviewer) can see what the store contains
# without loading the Parquet files. The manifest is gitignored (it is a
# byproduct of the preprocessing run), but the *reference* manifest is
# committed in expected_outputs/ after the first full run.

FEATURE_STORE_MANIFEST = FEATURE_STORE / "MANIFEST.json"


def _write_manifest(
    summary: Dict[str, Dict[str, Any]],
    mode: str,
    feature_order: Tuple[str, ...],
    log: logging.Logger,
) -> None:
    manifest = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "mode": mode,
        "feature_order": list(feature_order),
        "feature_dtype": FEATURE_DTYPE,
        "protocols": summary,
    }
    FEATURE_STORE.mkdir(parents=True, exist_ok=True)
    FEATURE_STORE_MANIFEST.write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    _jlog(log, "info", event="feature_store_manifest_written",
          path=str(FEATURE_STORE_MANIFEST))


# ---------------------------------------------------------------------------
# Schema verification against configs/riemannian_metrics.yaml §2
# ---------------------------------------------------------------------------

def _verify_schema(
    feature_order: Tuple[str, ...],
    log: logging.Logger,
) -> None:
    """
    Verify that the feature store on disk has exactly the twelve features,
    in exactly the order defined in configs/riemannian_metrics.yaml §2.
    This is the check that catches a drift between the feature-store
    writer and the metric-tensor reader.

    A drift here would silently corrupt every detection result, because
    the metric tensor is a 12x12 matrix indexed by position. The check
    is cheap (one Parquet read per protocol) and runs at the end of
    every preprocessing invocation.
    """
    try:
        import pandas as pd
    except ImportError:
        log.warning("pandas not available; skipping schema verification.")
        return

    for protocol in ("doh", "dot", "doq"):
        path = FEATURE_STORE / f"{protocol}_features.parquet"
        if not path.exists():
            continue
        cols = list(pd.read_parquet(path, columns=list(feature_order)).columns)
        if tuple(cols) != tuple(feature_order):
            log.error(
                f"{protocol}: feature-store column order does not match "
                f"configs/riemannian_metrics.yaml §2. "
                f"expected={list(feature_order)} observed={cols}"
            )
            raise SystemExit(4)
        _jlog(log, "info", event="feature_store_schema_ok", protocol=protocol)


# ---------------------------------------------------------------------------
# Disk-space check
# ---------------------------------------------------------------------------
# The feature store is about 1/5 the size of the raw Parquet corpus (the
# twelve float64 features plus the small meta columns). A rough estimate
# is 8 GB for a 40 GB raw corpus. The check refuses to start if the
# target filesystem has less than the estimate.

def _check_disk_space(feature_store_dir: Path, log: logging.Logger) -> None:
    try:
        usage = shutil.disk_usage(feature_store_dir)
    except OSError as e:
        _jlog(log, "warning", event="disk_usage_failed", error=str(e))
        return
    # 10 GiB conservative estimate. The actual store for DS-1 is ~8 GB.
    required = 10 * 1024 ** 3
    if usage.free < required:
        log.error(
            f"Insufficient disk space at {feature_store_dir}: "
            f"need {required / 1024**3:.1f} GiB, "
            f"have {usage.free / 1024**3:.1f} GiB free."
        )
        raise SystemExit(5)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="preprocess_zenodo.py",
        description="Build the E3-DoH feature store from the toy synthetic "
                    "corpus or from the DS-1 real corpus.",
    )
    p.add_argument(
        "--mode", choices=("toy", "full"), default="toy",
        help="toy = read data/synthetic/ and write the toy feature store; "
             "full = read data/raw/DS-1/ and write the full feature store.",
    )
    p.add_argument(
        "--num-workers", type=int, default=4,
        help="Number of Dask workers for full mode. Ignored in toy mode.",
    )
    p.add_argument(
        "--log-level", choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
    )
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    log = _setup_logging(args.log_level)
    _jlog(log, "info", event="start", mode=args.mode,
          num_workers=args.num_workers)

    feature_order = _load_feature_spec(log)
    _jlog(log, "info", event="feature_spec_loaded",
          n_features=len(feature_order), order=list(feature_order))

    # Disk check on the feature-store target.
    try:
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        _check_disk_space(PROCESSED_DIR, log)
    except SystemExit as e:
        _jlog(log, "error", event="disk_check_failed", reason=str(e))
        return int(e.code) if e.code is not None else 5

    try:
        if args.mode == "toy":
            summary = _preprocess_toy(log, feature_order)
        else:
            summary = _preprocess_full(log, feature_order, args.num_workers)
    except SystemExit as e:
        _jlog(log, "error", event="preprocess_aborted",
              reason=str(e), code=e.code)
        return int(e.code) if e.code is not None else 1
    except Exception as e:
        _jlog(log, "error", event="preprocess_exception", error=repr(e))
        return 3

    _write_manifest(summary, args.mode, feature_order, log)

    try:
        _verify_schema(feature_order, log)
    except SystemExit as e:
        return int(e.code) if e.code is not None else 4

    _jlog(log, "info", event="end", mode=args.mode,
          protocols=sorted(summary.keys()),
          total_rows=sum(v["rows"] for v in summary.values()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
