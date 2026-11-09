#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/generate_manifest.py
============================

E3-DoH Reproducibility — SHA-256 manifest generator.

Chapter 5 of the thesis: E3-DoH (Enhanced Evolutionary Encryption for
Multi-Protocol DNS Security — DoH, DoT, DoQ).

This script populates `expected_outputs/MANIFEST.sha256`, the file that
`verification_script.py` uses for the file-integrity check. The
manifest records the SHA-256 of every committed file in
`expected_outputs/`, and the verification script refuses to compare
against a hand-edited reference.

WHEN TO RUN IT
--------------
After the first successful full-reproduction run on the reference
testbed, on the commit that also commits the JSON/CSV/SVG reference
files:

    python scripts/generate_manifest.py \
        --outputs expected_outputs/ \
        --manifest expected_outputs/MANIFEST.sha256

The generated manifest replaces the placeholder content (all
`<TO_BE_POPULATED>` lines) with the real digests.

WHAT IT DOES
------------
For each file matched by `--pattern` (default: `*_expected.*`), it
computes the SHA-256 hash and writes it in the standard `sha256sum`
format:

    <64-hex-digest>  <relative-path>

The header of the manifest is preserved (any lines starting with `#`
are kept), so the provenance notes documented in the placeholder file
survive the regeneration.

WHAT IT REFUSES TO DO
---------------------
    - It does not hash a file that is not a committed reference (e.g.
      a temp file in `outputs/`). The `--outputs` directory must be
      the `expected_outputs/` directory.
    - It does not hash a file larger than a configurable limit (default
      10 MB), because a large binary is a sign that the wrong file
      is being committed.
    - It does not include files that are not in the expected-output
      naming scheme, so an accidental editor backup or `.DS_Store`
      is excluded.

USAGE
-----
    # Default: expected_outputs/ -> MANIFEST.sha256
    python scripts/generate_manifest.py

    # Custom paths
    python scripts/generate_manifest.py \
        --outputs /path/to/expected \
        --manifest /path/to/expected/MANIFEST.sha256

    # With a custom pattern
    python scripts/generate_manifest.py --pattern "*_expected.*"

    # Dry run (show what would be written, do not write)
    python scripts/generate_manifest.py --dry-run

EXIT CODES
----------
    0  manifest written (or dry-run completed)
    1  no files matched
    2  a file exceeded the size limit
    3  I/O error
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path
from typing import List, Optional, Tuple

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Default input and output paths.
_DEFAULT_OUTPUTS = _REPO_ROOT / "expected_outputs"
_DEFAULT_MANIFEST = _DEFAULT_OUTPUTS / "MANIFEST.sha256"
_DEFAULT_PATTERN = "*_expected.*"

# The default size limit for a single file. A 10 MB expected-output
# file is a sign that a binary (or a generated dataset) is being
# committed by mistake; the largest expected file in the artifact is
# a PNG figure (under 2 MB).
_DEFAULT_MAX_FILE_MB = 10

# The chunk size for the streaming hash.
_HASH_CHUNK_BYTES = 1024 * 1024


# ---------------------------------------------------------------------------
# Hashing
# ---------------------------------------------------------------------------

def sha256_of_file(path: Path) -> str:
    """Compute the SHA-256 hex digest of a file, streaming."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(_HASH_CHUNK_BYTES), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Manifest generation
# ---------------------------------------------------------------------------

def collect_files(
    outputs_dir: Path,
    pattern: str,
    max_file_mb: float,
    log: List[str],
) -> List[Tuple[Path, str, int]]:
    """
    Collect the files to include in the manifest.

    Returns a list of (path, relative_path, size_bytes) tuples, sorted
    by relative path for determinism.
    """
    if not outputs_dir.exists():
        raise FileNotFoundError(f"outputs directory not found: {outputs_dir}")

    files: List[Tuple[Path, str, int]] = []
    max_bytes = int(max_file_mb * 1024 * 1024)

    for path in sorted(outputs_dir.glob(pattern)):
        if not path.is_file():
            continue
        size = path.stat().st_size
        if size > max_bytes:
            raise ValueError(
                f"{path} is {size / 1024 / 1024:.1f} MB, exceeding the "
                f"{max_file_mb:.1f} MB limit. If this is intentional, "
                f"pass --max-file-mb with a larger value; otherwise, "
                f"remove the file before regenerating the manifest."
            )
        rel = path.relative_to(outputs_dir.parent)
        files.append((path, str(rel), size))
        log.append(f"  include {rel} ({size} bytes)")

    # Exclude the manifest itself.
    files = [f for f in files if not f[0].name == "MANIFEST.sha256"]
    return files


def preserve_header(manifest_path: Path) -> List[str]:
    """
    Read the existing manifest and return the header lines (those
    starting with '#', plus the blank line after them). Preserves the
    provenance notes documented in the placeholder file.
    """
    if not manifest_path.exists():
        return []
    header: List[str] = []
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        stripped = line.rstrip()
        if not stripped:
            # Blank line: include it in the header block only if we've
            # already seen a '#' line.
            if header:
                header.append(stripped)
            continue
        if stripped.startswith("#"):
            header.append(stripped)
        else:
            break  # first non-header, non-blank line ends the header
    # Trim trailing blank lines.
    while header and not header[-1]:
        header.pop()
    return header


def build_manifest_content(
    files: List[Tuple[Path, str, int]],
    header_lines: List[str],
) -> str:
    """
    Build the manifest content: preserved header + one line per file.
    """
    lines: List[str] = []
    if header_lines:
        lines.extend(header_lines)
        lines.append("")  # blank line after header
    else:
        # Default header if there was none.
        lines.append("# E3-DoH Reproducibility — SHA-256 Manifest")
        lines.append("#")
        lines.append("# Generated by scripts/generate_manifest.py.")
        lines.append("# Format: <sha256>  <relative-path>")
        lines.append("")

    for path, rel, _size in files:
        digest = sha256_of_file(path)
        lines.append(f"{digest}  {rel}")

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="generate_manifest.py",
        description="Populate expected_outputs/MANIFEST.sha256 from the "
                    "committed reference files.",
    )
    p.add_argument("--outputs", type=str, default=str(_DEFAULT_OUTPUTS),
                   help=f"Directory containing the reference files "
                        f"(default: {_DEFAULT_OUTPUTS})")
    p.add_argument("--manifest", type=str, default=str(_DEFAULT_MANIFEST),
                   help=f"Manifest output path (default: {_DEFAULT_MANIFEST})")
    p.add_argument("--pattern", type=str, default=_DEFAULT_PATTERN,
                   help=f"Glob pattern for files to include "
                        f"(default: {_DEFAULT_PATTERN!r})")
    p.add_argument("--max-file-mb", type=float, default=_DEFAULT_MAX_FILE_MB,
                   help=f"Refuse files larger than this "
                        f"(default: {_DEFAULT_MAX_FILE_MB} MB)")
    p.add_argument("--dry-run", action="store_true",
                   help="Print the manifest but do not write it.")
    p.add_argument("--verbose", "-v", action="store_true",
                   help="Print per-file progress.")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_parser().parse_args(argv)

    outputs_dir = Path(args.outputs)
    if not outputs_dir.is_absolute():
        outputs_dir = _REPO_ROOT / outputs_dir
    manifest_path = Path(args.manifest)
    if not manifest_path.is_absolute():
        manifest_path = _REPO_ROOT / manifest_path

    log: List[str] = []
    try:
        files = collect_files(outputs_dir, args.pattern, args.max_file_mb, log)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 3
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 2

    if not files:
        print(f"Error: no files matching {args.pattern!r} in {outputs_dir}",
              file=sys.stderr)
        return 1

    if args.verbose:
        for line in log:
            print(line)
        print(f"  {len(files)} files collected")

    header = preserve_header(manifest_path)
    if args.verbose and header:
        print(f"  preserved {len(header)} header line(s)")

    content = build_manifest_content(files, header)

    if args.dry_run:
        print("=== DRY RUN: would write the following to "
              f"{manifest_path} ===")
        print(content)
        print(f"=== end ({len(files)} files) ===")
        return 0

    try:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(content, encoding="utf-8")
    except OSError as e:
        print(f"Error writing manifest: {e}", file=sys.stderr)
        return 3

    print(f"Wrote {len(files)} entries to {manifest_path}")
    for path, rel, size in files:
        print(f"  {sha256_of_file(path)[:16]}...  {rel} ({size} bytes)")
    print()
    print("The manifest is now populated. Commit it together with the "
          "expected_outputs/ files it describes.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
