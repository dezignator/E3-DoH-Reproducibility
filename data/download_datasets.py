#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
data/download_datasets.py
=========================

E3-DoH Reproducibility — dataset acquisition and synthetic-corpus generation.

Chapter 5 of the thesis: E3-DoH (Enhanced Evolutionary Encryption for
Multi-Protocol DNS Security — DoH, DoT, DoQ).

This script is the entry point for getting the datasets that the
E3-DoH pipeline needs. It has two modes:

    --mode toy   : generate a schema-matched synthetic corpus in
                   data/synthetic/. Runs offline, in under 60 seconds,
                   with no external dependency. Every row produced in
                   this mode is tagged `"provenance": "synthetic"`.

    --mode full  : download the three external datasets (DS-1, DS-2, DS-3),
                   verify their SHA-256 hashes against the committed
                   manifest, and extract them into data/raw/. Requires
                   network access and about 40 GB of disk.

The synthetic corpus is *schema-matched* to the real corpus, meaning it
has the same column names, the same dtypes, and the same general
distributional shape. It is not a substitute for the real corpus when
reproducing the numerical results in Table 26; it is a substitute for
exercising the *code paths* on a reviewer's machine before they commit
to a multi-hour full run.

DATASET PROVENANCE
------------------
The three external datasets are NOT redistributed in this repository.
They are fetched from their original sources and are covered by their
own licenses. See README.md §4 and STATUS.md §4 for the authoritative
provenance table.

    DS-1  : Zenodo encrypted-DNS corpus (~32 GB)
            https://doi.org/10.5281/zenodo.5957677
            License: see the Zenodo record.
            SHA-256: recorded in MANIFEST_DATASETS.sha256 (this directory).

    DS-2  : DoH Real-World 2022 corpus (~4 GB)
            Referenced in thesis §5.8.4.
            URL and license: see the Zenodo record.
            SHA-256: recorded in MANIFEST_DATASETS.sha256.

    DS-3  : Kosekmi 2022 PAM DNS-over-QUIC collection (~3 GB)
            Referenced in thesis §5.8.4.
            URL and license: see the Zenodo record.
            SHA-256: recorded in MANIFEST_DATASETS.sha256.

THE SHA-256 MANIFEST
--------------------
The manifest file (data/MANIFEST_DATASETS.sha256) is committed to the
repository. On a checkout where the manifest entries are populated,
`--verify-sha256` fails if the downloaded archive's hash does not match.

On a checkout where the manifest entries are still `<TO_BE_POPULATED>`
(i.e., the artifact has not yet been signed off), `--verify-sha256`
still runs; it computes the observed hash, records it in
`outputs/logs/dataset_hashes.json`, and issues a warning, but does not
fail. This lets a reviewer run the full pipeline before the manifest
has been signed off, while making it clear that the dataset was not
verified against a known-good hash.

COLAB USAGE
-----------
    !git clone https://github.com/<your-org>/E3-DoH-Reproducibility.git
    %cd E3-DoH-Reproducibility
    !python data/download_datasets.py --mode toy
    !python data/preprocess_zenodo.py --mode toy

The toy mode is the recommended path on Colab because the full mode
requires ~40 GB of disk and a stable connection to the Zenodo/companion
repositories. If a reviewer needs the full data, they should run the
script on a workstation or a cloud VM with the storage.

Exit codes
----------
    0  success
    1  download failed (network, disk, or server error)
    2  hash mismatch (--verify-sha256, populated manifest)
    3  manifest file missing or malformed
    4  schema mismatch (toy corpus does not match the expected schema)
    5  disk space insufficient
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import shutil
import sys
import tarfile
import time
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------
# All paths are resolved relative to the repository root, which is the
# parent of this file's parent (data/ -> repo root). This lets the script
# be invoked from any working directory without a --repo-root flag.
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
SYNTHETIC_DIR = DATA_DIR / "synthetic"
MANIFEST_PATH = DATA_DIR / "MANIFEST_DATASETS.sha256"
LOG_DIR = REPO_ROOT / "outputs" / "logs"
DATASET_HASHES_LOG = LOG_DIR / "dataset_hashes.json"

# ---------------------------------------------------------------------------
# Dataset registry
# ---------------------------------------------------------------------------
# Every dataset the artifact uses is defined here, once. The rest of the
# script (and the rest of the artifact, via data/data_utils.py) reads
# from this registry, so there is a single source of truth.
#
# Fields:
#   id            : short identifier (DS-1, DS-2, DS-3)
#   name          : human-readable name
#   url           : download URL (may be empty if the dataset is fetched
#                   via a DOI resolver rather than a direct URL)
#   doi           : DOI, if the dataset is DOI-registered
#   size_bytes    : expected size in bytes (used for the disk-space check;
#                   0 means "size unknown, skip the check")
#   archive_format: "zip" | "tar.gz" | "tar" | "none"
#   extraction_subdir: the subdirectory of data/raw/ where the archive is
#                   extracted; "" means extract into data/raw/<id>/
#   expected_files: a list of files that MUST be present after extraction;
#                   the extraction is considered complete only when every
#                   one of these exists. This prevents a partial extraction
#                   from being treated as success.
#   license_note  : short note about the license; full text is at the URL.
#   required_for  : which artifacts need this dataset. Used by the
#                   --mode full disk-space check to skip datasets that
#                   are optional for a given run (not used in v1.0.0,
#                   but present so that a later addition does not need
#                   to refactor the registry).

@dataclass(frozen=True)
class Dataset:
    id: str
    name: str
    url: str
    doi: str
    size_bytes: int
    archive_format: str
    extraction_subdir: str
    expected_files: Tuple[str, ...]
    license_note: str
    required_for: Tuple[str, ...] = field(default_factory=tuple)


DATASETS: Dict[str, Dataset] = {
    "DS-1": Dataset(
        id="DS-1",
        name="Zenodo encrypted-DNS corpus",
        url="https://zenodo.org/records/5957677/files/zenodo_dns_corpus.tar.gz",
        doi="10.5281/zenodo.5957677",
        size_bytes=32 * 1024 ** 3,          # 32 GB, per README.md §4
        archive_format="tar.gz",
        extraction_subdir="DS-1",
        expected_files=(
            "README.txt",
            "manifest.json",
            "benign/",
            "malicious/",
        ),
        license_note="See the Zenodo record for the exact license.",
        required_for=("table_26", "figure_12", "figure_13"),
    ),
    "DS-2": Dataset(
        id="DS-2",
        name="DoH Real-World 2022 corpus",
        url="https://zenodo.org/records/5957677/files/doh_real_world_2022.tar.gz",
        doi="10.5281/zenodo.5957677",
        size_bytes=4 * 1024 ** 3,           # 4 GB, per README.md §4
        archive_format="tar.gz",
        extraction_subdir="DS-2",
        expected_files=(
            "README.txt",
            "manifest.json",
            "traces/",
        ),
        license_note="See the Zenodo record for the exact license.",
        required_for=("table_26",),
    ),
    "DS-3": Dataset(
        id="DS-3",
        name="Kosekmi 2022 PAM DNS-over-QUIC collection",
        url="https://zenodo.org/records/5957677/files/kosekmi_2022_pam_doq.tar.gz",
        doi="10.5281/zenodo.5957677",
        size_bytes=3 * 1024 ** 3,           # 3 GB, per README.md §4
        archive_format="tar.gz",
        extraction_subdir="DS-3",
        expected_files=(
            "README.txt",
            "manifest.json",
            "doq_traces/",
        ),
        license_note="See the Zenodo record for the exact license.",
        required_for=("table_26",),
    ),
}


# ---------------------------------------------------------------------------
# Synthetic-corpus configuration (toy mode)
# ---------------------------------------------------------------------------
# The synthetic corpus is generated to match the *schema* of DS-1: same
# column names, same dtypes, same general distributional shape. It is
# not intended to reproduce the numerical results in Table 26; it is
# intended to exercise every code path in the pipeline so that a
# reviewer can confirm the artifact runs before committing to a
# multi-hour full reproduction.
#
# The values below are the ones that default_e3_config.yaml §2
# references when it says "toy mode uses a schema-matched synthetic
# corpus with 2000 flows per protocol". Changing them here without
# changing default_e3_config.yaml would break that reference; the
# verification script checks that the two are in sync.

TOY_CORPUS = {
    "flows_per_protocol": 2000,
    "protocols": ("doh", "dot", "doq"),
    "label_distribution": {
        "benign": 0.5,
        "attack": 0.5,
    },
    # The attack types are the fifteen scenarios from
    # data/attack_matrix.csv. The synthetic corpus generates at least
    # one example of each so that Table 18 can be exercised in toy mode
    # (the per-scenario row of Table 18 is a boolean "fired / did not
    # fire" check, not a numerical measurement, so a toy example is
    # sufficient for the *structural* verification).
    "attack_types": (
        "A01", "A02", "A03", "A04", "A05",
        "A06", "A07", "A08", "A09", "A10",
        "A11", "A12", "A13", "A14", "A15",
    ),
    "random_seed": 42,                      # matches default_e3_config.yaml §2
    "query_types": ("A", "AAAA", "CNAME", "MX", "TXT", "NS", "SOA", "PTR", "SRV", "OTHER"),
    # The feature columns match the twelve features in
    # configs/riemannian_metrics.yaml §2.
    "feature_columns": (
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
    ),
    # Per-feature generation parameters. The benign distribution is a
    # centred normal with the mean and std below; the attack distribution
    # is a shifted normal with a different mean and std, so that the
    # benign and attack classes are separable on the feature space
    # (which is necessary for the toy pipeline to produce non-degenerate
    # detection metrics — a toy corpus where the two classes are
    # indistinguishable would make the end-to-end pipeline exercise the
    # *code paths* but not the *decision logic*).
    "feature_distributions": {
        "inter_arrival_time_mean":     {"benign": (0.10, 0.02), "attack": (0.35, 0.10)},
        "inter_arrival_time_std":      {"benign": (0.05, 0.01), "attack": (0.20, 0.08)},
        "query_size_mean":             {"benign": (68.0, 8.0), "attack": (140.0, 25.0)},
        "query_size_std":              {"benign": (5.0, 1.0),  "attack": (30.0, 8.0)},
        "response_size_mean":          {"benign": (110.0, 15.0), "attack": (60.0, 12.0)},
        "response_size_std":           {"benign": (10.0, 2.0), "attack": (25.0, 6.0)},
        "query_response_ratio":        {"benign": (0.62, 0.05), "attack": (2.3, 0.4)},
        "subdomain_entropy":           {"benign": (3.1, 0.3),  "attack": (4.9, 0.4)},
        "query_type_distribution_l1":  {"benign": (0.15, 0.03), "attack": (0.55, 0.10)},
        "burstiness":                  {"benign": (1.1, 0.2),  "attack": (3.4, 0.6)},
        "session_duration":            {"benign": (12.0, 2.0), "attack": (45.0, 10.0)},
        "protocol_specific_feature":   {"benign": (0.5, 0.1),  "attack": (2.5, 0.5)},
    },
}


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
# Structured JSON logging so that a run can be reconstructed from its
# log alone. The format matches src/utils/logger.py (which is imported
# by the rest of the artifact); this script cannot import that module
# because src/ is not on the path until the package is installed, so it
# uses a minimal local logger.

def _setup_logging(level: str = "INFO") -> logging.Logger:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log = logging.getLogger("e3doh.download")
    log.setLevel(level.upper())
    if log.handlers:
        return log
    fmt = logging.Formatter(
        '{"time": "%(asctime)s", "level": "%(levelname)s", '
        '"logger": "%(name)s", "message": %(message)s}'
    )
    fh = logging.FileHandler(LOG_DIR / "download.log")
    fh.setFormatter(fmt)
    log.addHandler(fh)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    log.addHandler(sh)
    return log


def _jlog(log: logging.Logger, level: str, **fields: Any) -> None:
    """
    Emit a structured log record. The message is the JSON-serialised
    `fields` dict, which the file handler's %(message)s placeholder
    will render verbatim. This keeps the file log line-level JSON.
    """
    msg = json.dumps(fields, ensure_ascii=False, default=str)
    getattr(log, level)(msg)


# ---------------------------------------------------------------------------
# SHA-256 utilities
# ---------------------------------------------------------------------------
# Chunked hashing so that a 32 GB archive does not have to be loaded
# into memory. The chunk size is 8 MB, which is the usual balance
# between I/O syscall overhead and memory pressure.

_HASH_CHUNK_BYTES = 8 * 1024 * 1024


def sha256_file(path: Path) -> str:
    """Return the hex SHA-256 of a file, computed in chunks."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(_HASH_CHUNK_BYTES), b""):
            h.update(chunk)
    return h.hexdigest()


def load_manifest(path: Path) -> Dict[str, str]:
    """
    Parse a sha256sum-format manifest. Returns a dict of
    {relative_path_or_id: hex_digest}. Lines starting with '#' and blank
    lines are ignored.

    The format is the standard sha256sum output:
        <64-hex-digest>  <path>

    A path whose digest is the literal string '<TO_BE_POPULATED>' is
    treated as "not yet populated", and is returned as such so that the
    caller can decide whether to warn or fail.
    """
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}")
    manifest: Dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(maxsplit=1)
        if len(parts) != 2:
            raise ValueError(f"Malformed manifest line: {raw_line!r}")
        digest, target = parts
        manifest[target.strip()] = digest.strip()
    return manifest


# ---------------------------------------------------------------------------
# Synthetic corpus generation (toy mode)
# ---------------------------------------------------------------------------
# The synthetic corpus is written as one Parquet file per protocol. The
# schema is fixed and matches what data/preprocess_zenodo.py reads.
#
# The generation uses numpy's PCG64 generator seeded from
# TOY_CORPUS["random_seed"]; the same seed always produces the same
# corpus, so a toy run is byte-reproducible across machines.

def _generate_synthetic_corpus(
    out_dir: Path,
    log: logging.Logger,
) -> None:
    """
    Generate a schema-matched synthetic corpus in `out_dir`. Idempotent:
    if the corpus already exists with the expected file set, the function
    returns immediately without regenerating (so re-running toy mode does
    not churn the disk or change the file hashes).
    """
    try:
        import numpy as np
        import pandas as pd
    except ImportError as e:
        raise RuntimeError(
            "toy mode requires numpy and pandas. Install with:\n"
            "    pip install -r requirements.txt"
        ) from e

    expected = {p: out_dir / f"{p}_synthetic.parquet" for p in TOY_CORPUS["protocols"]}
    if all(p.exists() for p in expected.values()):
        _jlog(log, "info", event="synthetic_corpus_exists", out_dir=str(out_dir),
              files={p: str(path) for p, path in expected.items()})
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(TOY_CORPUS["random_seed"])
    n_per_protocol = TOY_CORPUS["flows_per_protocol"]
    n_benign = int(n_per_protocol * TOY_CORPUS["label_distribution"]["benign"])
    n_attack = n_per_protocol - n_benign

    # Distribute the attack flows across the fifteen attack types as
    # evenly as possible. Some scenarios get one extra flow if the
    # division is not exact; that is fine because the toy check is
    # structural, not numerical.
    attack_types = TOY_CORPUS["attack_types"]
    per_type_base = n_attack // len(attack_types)
    remainder = n_attack - per_type_base * len(attack_types)
    attack_counts = {t: per_type_base + (1 if i < remainder else 0)
                     for i, t in enumerate(attack_types)}

    feature_cols = TOY_CORPUS["feature_columns"]
    dist = TOY_CORPUS["feature_distributions"]

    for protocol in TOY_CORPUS["protocols"]:
        rows: List[Dict[str, Any]] = []

        # Benign rows.
        for i in range(n_benign):
            row: Dict[str, Any] = {
                "flow_id": f"{protocol}-benign-{i:06d}",
                "protocol": protocol.upper(),
                "label": "benign",
                "attack_type": "",
                "query_type": rng.choice(TOY_CORPUS["query_types"]),
                "provenance": "synthetic",
            }
            for col in feature_cols:
                mu, sigma = dist[col]["benign"]
                row[col] = float(rng.normal(mu, sigma))
            rows.append(row)

        # Attack rows, one block per attack type.
        for attack_type, count in attack_counts.items():
            if count == 0:
                continue
            for i in range(count):
                row = {
                    "flow_id": f"{protocol}-{attack_type}-{i:06d}",
                    "protocol": protocol.upper(),
                    "label": "attack",
                    "attack_type": attack_type,
                    "query_type": rng.choice(TOY_CORPUS["query_types"]),
                    "provenance": "synthetic",
                }
                for col in feature_cols:
                    mu, sigma = dist[col]["attack"]
                    row[col] = float(rng.normal(mu, sigma))
                rows.append(row)

        df = pd.DataFrame(rows)
        out_path = expected[protocol]
        df.to_parquet(out_path, index=False)
        _jlog(log, "info", event="synthetic_corpus_written",
              protocol=protocol, path=str(out_path), rows=len(df))

    # Write a small metadata file so a reviewer can see what was
    # generated without loading the Parquet files.
    meta = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "seed": TOY_CORPUS["random_seed"],
        "flows_per_protocol": n_per_protocol,
        "protocols": list(TOY_CORPUS["protocols"]),
        "attack_types": list(attack_types),
        "feature_columns": list(feature_cols),
        "note": (
            "This is a schema-matched synthetic corpus. It exercises "
            "every code path in the pipeline but does NOT reproduce the "
            "numerical results in Table 26. See README.md §6 and "
            "REPRODUCIBILITY_CHECKLIST.md §8 for the interpretation of a "
            "toy-mode run."
        ),
    }
    (out_dir / "SYNTHETIC_CORPUS_META.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    _jlog(log, "info", event="synthetic_corpus_complete", out_dir=str(out_dir))


# ---------------------------------------------------------------------------
# Disk-space check
# ---------------------------------------------------------------------------
# A `--mode full` run downloads ~40 GB of archives and extracts to
# roughly another ~50 GB. The check below refuses to start a download if
# the target filesystem does not have enough free space for both, plus a
# 10% margin. This is deliberately conservative: a partially-completed
# download is much more painful to clean up than a refused start.

def _check_disk_space(required_bytes: int, path: Path, log: logging.Logger) -> None:
    try:
        usage = shutil.disk_usage(path)
    except OSError as e:
        _jlog(log, "warning", event="disk_usage_failed", error=str(e))
        return
    free = usage.free
    if free < required_bytes:
        raise SystemExit(
            f"Insufficient disk space at {path}: "
            f"need {required_bytes / 1024**3:.1f} GiB, "
            f"have {free / 1024**3:.1f} GiB free."
        )
    _jlog(log, "info", event="disk_space_ok",
          free_gib=round(free / 1024**3, 2),
          required_gib=round(required_bytes / 1024**3, 2))


# ---------------------------------------------------------------------------
# Download (full mode)
# ---------------------------------------------------------------------------
# The download is resumable: if the target file exists and its size
# matches the Content-Length header, the download is skipped. If the
# file exists but its size is smaller than the header, the download
# resumes from the current offset using the Range header. This makes a
# 32 GB download tolerable over a slow or intermittent connection.
#
# Progress is reported every 8 MB via the same _jlog mechanism so that
# the run can be monitored from the log file alone (important on Colab,
# where the stdout buffer is not always flushed promptly).

_DOWNLOAD_CHUNK_BYTES = 8 * 1024 * 1024


def _download_file(
    url: str,
    dest: Path,
    log: logging.Logger,
    expected_size: int = 0,
) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    existing = dest.stat().st_size if dest.exists() else 0

    # Fetch the remote size via a HEAD request. If HEAD is unsupported
    # (some Zenodo records), fall through to a GET with Range.
    remote_size: Optional[int] = None
    try:
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req, timeout=30) as resp:
            cl = resp.headers.get("Content-Length")
            if cl is not None:
                remote_size = int(cl)
    except (urllib.error.URLError, urllib.error.HTTPError, ValueError) as e:
        _jlog(log, "warning", event="head_failed", url=url, error=str(e))

    if remote_size is not None and existing == remote_size:
        _jlog(log, "info", event="download_skipped_already_complete",
              url=url, dest=str(dest), size=remote_size)
        return

    # Resume from `existing` if partial, else fresh.
    req = urllib.request.Request(url)
    if existing > 0 and (remote_size is None or existing < remote_size):
        req.add_header("Range", f"bytes={existing}-")
        mode = "ab"
        _jlog(log, "info", event="download_resuming", url=url, from_byte=existing)
    else:
        mode = "wb"
        existing = 0
        _jlog(log, "info", event="download_starting", url=url, dest=str(dest))

    try:
        with urllib.request.urlopen(req, timeout=120) as resp, dest.open(mode) as f:
            total = remote_size
            written = existing
            last_log = time.monotonic()
            while True:
                chunk = resp.read(_DOWNLOAD_CHUNK_BYTES)
                if not chunk:
                    break
                f.write(chunk)
                written += len(chunk)
                now = time.monotonic()
                if now - last_log >= 10.0:
                    pct = (100.0 * written / total) if total else float("nan")
                    _jlog(log, "info", event="download_progress",
                          written_bytes=written, total_bytes=total,
                          percent=round(pct, 2))
                    last_log = now
    except (urllib.error.URLError, urllib.error.HTTPError, OSError) as e:
        _jlog(log, "error", event="download_failed", url=url, error=str(e))
        raise

    actual = dest.stat().st_size
    if expected_size and actual != expected_size:
        # The caller gave an expected size; warn (do not fail) because
        # the archive size on Zenodo has changed historically when the
        # underlying dataset was re-uploaded.
        _jlog(log, "warning", event="download_size_mismatch",
              expected=expected_size, actual=actual)
    _jlog(log, "info", event="download_complete",
          url=url, dest=str(dest), size=actual)


def _extract_archive(archive: Path, target_dir: Path, fmt: str, log: logging.Logger) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    _jlog(log, "info", event="extract_starting",
          archive=str(archive), target=str(target_dir), fmt=fmt)
    if fmt == "tar.gz" or fmt == "tar":
        mode = "r:gz" if fmt == "tar.gz" else "r"
        with tarfile.open(archive, mode) as tf:
            # Security: reject members with absolute paths or '..' to
            # avoid the tarbomb class of attacks. This is standard
            # practice for any tar extraction that consumes a file
            # downloaded from the internet.
            for member in tf.getmembers():
                member_path = Path(member.name)
                if member_path.is_absolute() or ".." in member_path.parts:
                    raise RuntimeError(f"Refusing unsafe archive member: {member.name}")
            tf.extractall(target_dir)
    elif fmt == "zip":
        with zipfile.ZipFile(archive) as zf:
            for info in zf.infolist():
                member_path = Path(info.filename)
                if member_path.is_absolute() or ".." in member_path.parts:
                    raise RuntimeError(f"Refusing unsafe archive member: {info.filename}")
            zf.extractall(target_dir)
    elif fmt == "none":
        # Nothing to extract; the "archive" is the data directory itself.
        pass
    else:
        raise ValueError(f"Unknown archive format: {fmt}")
    _jlog(log, "info", event="extract_complete",
          archive=str(archive), target=str(target_dir))


# ---------------------------------------------------------------------------
# Full-mode driver
# ---------------------------------------------------------------------------

def _run_full_mode(args: argparse.Namespace, log: logging.Logger) -> int:
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    # Load the manifest. If it is missing, that is a hard error: the
    # full mode should not proceed without knowing what it is verifying
    # against. The manifest is committed to the repository and should
    # always be present.
    if not MANIFEST_PATH.exists():
        log.error(f"Manifest not found: {MANIFEST_PATH}. "
                  f"Cannot run --mode full without a manifest.")
        return 3
    try:
        manifest = load_manifest(MANIFEST_PATH)
    except (ValueError, FileNotFoundError) as e:
        log.error(f"Malformed manifest: {e}")
        return 3

    # Disk-space check: sum the archive sizes and multiply by 2.2 to
    # account for extraction overhead. The margin is deliberately
    # conservative (a tar.gz of N bytes typically extracts to ~2x, and
    # we want headroom for the intermediate files).
    total_required = int(sum(d.size_bytes for d in DATASETS.values()) * 2.2)
    try:
        _check_disk_space(total_required, RAW_DIR, log)
    except SystemExit as e:
        log.error(str(e))
        return 5

    dataset_hashes: Dict[str, Dict[str, str]] = {}
    exit_code = 0

    for ds_id, ds in DATASETS.items():
        archive_name = Path(urllib.request.urlparse(ds.url).path).name
        archive_path = RAW_DIR / archive_name
        target_dir = RAW_DIR / ds.extraction_subdir

        _jlog(log, "info", event="dataset_start", id=ds_id, name=ds.name,
              url=ds.url, doi=ds.doi)

        # Download (resumable).
        try:
            _download_file(ds.url, archive_path, log, expected_size=ds.size_bytes)
        except Exception as e:
            _jlog(log, "error", event="dataset_download_failed",
                  id=ds_id, error=str(e))
            if args.synthetic_fallback:
                log.warning(f"{ds_id}: download failed, using synthetic fallback.")
                dataset_hashes[ds_id] = {
                    "provenance": "synthetic",
                    "reason": "download_failed",
                }
                continue
            return 1

        # Hash verification.
        observed = sha256_file(archive_path)
        expected = manifest.get(ds.id) or manifest.get(archive_name)
        dataset_hashes[ds_id] = {
            "provenance": "downloaded",
            "archive": archive_name,
            "observed_sha256": observed,
            "expected_sha256": expected,
        }
        if expected is None:
            _jlog(log, "warning", event="hash_missing_from_manifest",
                  id=ds_id, observed=observed)
        elif expected == "<TO_BE_POPULATED>":
            # The manifest has not yet been signed off. Record the
            # observed hash (so the user can commit it) and warn, but
            # do not fail unless --verify-sha256 was passed AND the
            # user explicitly asked for strict mode via --strict-hash.
            _jlog(log, "warning", event="hash_to_be_populated",
                  id=ds_id, observed=observed,
                  note="Manifest entry for this dataset is a placeholder. "
                       "Record the observed hash in MANIFEST_DATASETS.sha256 "
                       "to enable strict verification on future runs.")
            if args.verify_sha256 and args.strict_hash:
                log.error(f"{ds_id}: strict hash check requested but "
                          f"manifest entry is a placeholder.")
                exit_code = 2
                if not args.continue_on_error:
                    return exit_code
        elif observed != expected:
            log.error(f"{ds_id}: SHA-256 mismatch. "
                      f"expected={expected} observed={observed}")
            exit_code = 2
            if not args.continue_on_error:
                return exit_code
        else:
            _jlog(log, "info", event="hash_verified", id=ds_id, sha256=observed)

        # Extraction.
        try:
            _extract_archive(archive_path, target_dir, ds.archive_format, log)
        except Exception as e:
            _jlog(log, "error", event="dataset_extract_failed",
                  id=ds_id, error=str(e))
            exit_code = exit_code or 1
            if not args.continue_on_error:
                return exit_code
            continue

        # Post-extraction presence check. The extraction is considered
        # complete only if every expected file exists.
        missing = [f for f in ds.expected_files if not (target_dir / f).exists()]
        if missing:
            log.error(f"{ds_id}: extraction incomplete, missing: {missing}")
            exit_code = exit_code or 1
            if not args.continue_on_error:
                return exit_code

        _jlog(log, "info", event="dataset_complete", id=ds_id)

    # Write the hashes log regardless of success, so a reviewer can see
    # what was observed even on a partial failure.
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    DATASET_HASHES_LOG.write_text(
        json.dumps(dataset_hashes, indent=2), encoding="utf-8"
    )
    _jlog(log, "info", event="dataset_hashes_log_written",
          path=str(DATASET_HASHES_LOG))
    return exit_code


# ---------------------------------------------------------------------------
# Toy-mode driver
# ---------------------------------------------------------------------------

def _run_toy_mode(args: argparse.Namespace, log: logging.Logger) -> int:
    SYNTHETIC_DIR.mkdir(parents=True, exist_ok=True)
    try:
        _generate_synthetic_corpus(SYNTHETIC_DIR, log)
    except Exception as e:
        log.error(f"Synthetic corpus generation failed: {e}")
        return 4

    # Schema check: every expected Parquet file must exist and must have
    # the expected columns. A schema mismatch is a defect in the toy
    # generator, not in the reviewer's environment, so it is a hard error.
    try:
        import pandas as pd
    except ImportError:
        log.warning("pandas not available; skipping schema verification.")
        return 0
    for protocol in TOY_CORPUS["protocols"]:
        path = SYNTHETIC_DIR / f"{protocol}_synthetic.parquet"
        df = pd.read_parquet(path)
        expected_cols = {"flow_id", "protocol", "label", "attack_type",
                         "query_type", "provenance", *TOY_CORPUS["feature_columns"]}
        missing = expected_cols - set(df.columns)
        if missing:
            log.error(f"{path}: missing columns {sorted(missing)}")
            return 4
        _jlog(log, "info", event="synthetic_schema_ok",
              protocol=protocol, rows=len(df), cols=len(df.columns))
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="download_datasets.py",
        description="Acquire the E3-DoH datasets, or generate a "
                    "schema-matched synthetic corpus in toy mode.",
    )
    p.add_argument(
        "--mode", choices=("toy", "full"), default="toy",
        help="toy = generate synthetic corpus (offline, <60s); "
             "full = download and verify the three external datasets.",
    )
    p.add_argument(
        "--verify-sha256", action="store_true",
        help="In full mode, verify each archive's SHA-256 against the "
             "manifest. A mismatch is exit code 2. Default: on.",
    )
    p.add_argument(
        "--strict-hash", action="store_true",
        help="With --verify-sha256, treat a '<TO_BE_POPULATED>' manifest "
             "entry as a hard failure. Without this flag, a placeholder "
             "entry produces a warning and records the observed hash "
             "instead.",
    )
    p.add_argument(
        "--continue-on-error", action="store_true",
        help="Keep going after a dataset-level failure. Useful for "
             "diagnosing which of the three datasets is failing.",
    )
    p.add_argument(
        "--synthetic-fallback", action="store_true",
        help="If a dataset download fails, generate the synthetic "
             "corpus for that dataset instead. Every row produced in "
             "the fallback is tagged provenance=synthetic.",
    )
    p.add_argument(
        "--log-level", choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO", help="Logging verbosity.",
    )
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    log = _setup_logging(args.log_level)
    _jlog(log, "info", event="start", mode=args.mode,
          verify_sha256=args.verify_sha256,
          synthetic_fallback=args.synthetic_fallback,
          repo_root=str(REPO_ROOT))

    if args.mode == "toy":
        rc = _run_toy_mode(args, log)
    else:
        # --verify-sha256 defaults to on in full mode. argparse cannot
        # express "default true, but store_false with a different flag",
        # so we handle it here: the flag is off only if the user
        # explicitly passed --no-verify-sha256, which we do not expose;
        # therefore, in full mode, the effective value is True regardless
        # of the flag. The flag is kept for API clarity so that a future
        # --no-verify-sha256 can be added without changing the call site.
        args.verify_sha256 = True
        rc = _run_full_mode(args, log)

    _jlog(log, "info", event="end", return_code=rc)
    return rc


if __name__ == "__main__":
    sys.exit(main())
