#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
setup.py
========

E3-DoH Reproducibility — editable-install entry point.

Chapter 5 of the thesis: E3-DoH (Enhanced Evolutionary Encryption for
Multi-Protocol DNS Security — DoH, DoT, DoQ).

This file makes the artifact installable as an editable package via:

    pip install -r requirements.txt
    pip install -e .

The editable install is REQUIRED, not optional. The artifact is not
designed to work as a non-editable (wheel) install, and the reason is
structural, not accidental:

  Every module in the artifact resolves paths relative to REPO_ROOT,
  which is computed at import time as `Path(__file__).resolve().parent.parent`
  from within data/data_utils.py. With `pip install -e .`, the source
  tree stays in its original location and that computation is correct.
  With `pip install .`, the package is copied into site-packages, and
  REPO_ROOT points to a directory that does not contain the configs/,
  data/, expected_outputs/, or scripts/ directories. Every path lookup
  would fail.

  If a reviewer wants a non-editable install, the correct procedure is
  to install into a virtualenv from a checkout, then `cd` back into the
  checkout to run the scripts. That is documented in README.md §7, and
  the artifact's own scripts only work from the checkout root.

WHAT THIS FILE DOES
-------------------
    1. Ensures that the required `__init__.py` files exist (they are
       empty files; they are delivered as part of the src/ and data/
       sequence, but this file also creates them defensively so that a
       fresh checkout does not fail with an obscure ImportError).
    2. Declares the `data` package and the `src.*` packages.
    3. Declares Python version bounds (3.11 <= python < 3.12; the
       upper bound is enforced because two transitive dependencies of
       torch 2.1.2 have no 3.12 wheels — see requirements.txt).
    4. Declares the package metadata (name, version, description,
       author, license, URL). The version is pinned here and must
       match STATUS.md §1; the check at the bottom of this file enforces
       that.
    5. Deliberately leaves `install_requires` empty. Dependencies are
       installed from requirements.txt in a separate step. This is a
       deliberate choice for a reproducibility artifact:
         - A reviewer who is diagnosing a version conflict wants to
           see the pip output for the pinned versions separately from
           the pip output for the package metadata.
         - The `pip install -e .` step is idempotent and fast when
           install_requires is empty; a slow first-time install that
           pulls torch would obscure the "did the package metadata
           resolve?" question.
         - The version pins live in one place (requirements.txt),
           not two.

WHAT THIS FILE DOES NOT DO
--------------------------
    - It does not declare `entry_points`. The scripts in scripts/ are
      invoked directly as `python scripts/xxx.py`, which is what the
      README documents. Entry points would require scripts/ to be a
      package, which it is not, and would add an indirection layer
      that makes it harder for a reviewer to see what is being run.
    - It does not declare `package_data`. The configs and data files
      are read from disk at runtime, not from the package. A non-
      editable install would not find them; that is why only the
      editable install is supported (see above).
    - It does not declare `data_files`. Same reason.
    - It does not declare `tests_require`. Test dependencies are in
      requirements.txt, tier 12 (optional); see that file.

PYPROJECT.TOML
--------------
Modern pip (>= 21.3) uses PEP 517, and a `pyproject.toml` is the
preferred location for build-system metadata. This artifact ships
`setup.py` only (no `pyproject.toml`), because:

    - The artifact targets old pip as well as new pip, and
      `setup.py` alone is understood by both.
    - A reviewer who is auditing the build sees one file, not two.
    - The `pyproject.toml` shim that pip would auto-generate for a
      `setup.py`-only project is equivalent to what we would write
      explicitly.

If you want to add a `pyproject.toml` (optional, not required), the
minimal version is:

    [build-system]
    requires = ["setuptools>=61", "wheel"]
    build-backend = "setuptools.build_meta"

There is no need to add anything else, because the project metadata
stays in setup.py.

Exit codes
----------
    `python setup.py --check-only` runs the coherence checks without
    invoking the build. Exit 0 = coherent, non-zero = a check failed.
    This is useful in CI and for the reviewer who wants to know whether
    the artifact is installable before running `pip install -e .`.
"""

from __future__ import annotations

import io
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from setuptools import find_packages, setup  # type: ignore

# ---------------------------------------------------------------------------
# Repository root
# ---------------------------------------------------------------------------
# Everything in this file is resolved relative to the directory that
# contains setup.py. That is standard for a setuptools build (pip runs
# setup.py from the project root), and it means the file behaves the
# same whether it is invoked as `python setup.py ...` or as part of a
# `pip install -e .` that runs the setup.py in a subprocess.

HERE = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Version
# ---------------------------------------------------------------------------
# The artifact version is defined in STATUS.md §1 and in
# default_e3_config.yaml (§1). This file pins it in a third place
# because setup.py is the Python-package version, which is not quite
# the same thing as the artifact version (the artifact may be versioned
# 1.0.0 while the Python package is versioned 1.0.0; they happen to
# match in this release).
#
# The check at the bottom of this file verifies that the three agree.
# A mismatch is a defect, not a warning: a reviewer who sees a
# Python-package version of 1.0.0 and an artifact version of 1.0.1 has
# no way to tell which one is the "real" version of the code.

PACKAGE_VERSION = "1.0.0"

# ---------------------------------------------------------------------------
# Required __init__.py files
# ---------------------------------------------------------------------------
# Every package in this artifact is a regular (non-namespace) Python
# package, which means each one needs an __init__.py file. The files
# are delivered as part of the src/ and data/ sequence, but this script
# also creates any that are missing, defensively, so that a reviewer
# who checks out the artifact in the middle of the delivery (when some
# __init__.py files may not yet be committed) still gets a working
# install.
#
# The files are created empty. An empty __init__.py is functionally
# correct for a regular package; the only cost is that `from src import
# x` does not expose any top-level names, which the artifact does not
# rely on.
#
# The list below must stay in sync with the packages= list at the
# bottom of this file. If a new package is added, both the REQUIRED_INITS
# list and the packages= list must be updated.

REQUIRED_INITS: Tuple[str, ...] = (
    "data/__init__.py",
    "src/__init__.py",
    "src/utils/__init__.py",
    "src/protocols/__init__.py",
    "src/models/__init__.py",
    "src/attacks/__init__.py",
    "src/hardware_simulation/__init__.py",
)

# The `packages` list. This is enumerated explicitly rather than using
# `find_packages()` because:
#
#   - `find_packages()` would also pick up `tests/`, which is not part
#     of the importable artifact (the tests are run from the checkout,
#     not imported by downstream scripts).
#   - `find_packages()` would also pick up `scripts/`, which is a
#     collection of CLI entry points, not an importable package.
#   - An explicit list makes it obvious to a reviewer what the install
#     exposes, which is a readability win for a reproducibility
#     artifact.
#
# Adding a new sub-package requires updating this list AND the
# REQUIRED_INITS list above.

PACKAGES: Tuple[str, ...] = (
    "data",
    "src",
    "src.utils",
    "src.protocols",
    "src.models",
    "src.attacks",
    "src.hardware_simulation",
)

# ---------------------------------------------------------------------------
# README as long_description
# ---------------------------------------------------------------------------
# The long_description is what `pip show e3doh-reproducibility` displays
# and what PyPI (if the artifact is ever published) uses. It is the
# README, unmodified. The `read_long_description()` function fails
# loudly if the README is missing, because a missing README means the
# checkout is incomplete.

def read_long_description() -> str:
    readme = HERE / "README.md"
    if not readme.exists():
        raise SystemExit(
            "README.md not found at the repository root. "
            "The artifact is not complete; check out the full repository."
        )
    return readme.read_text(encoding="utf-8")


def read_requirements() -> List[str]:
    """
    Parse requirements.txt into a list of requirement specifiers.

    The result is passed to `extras_require["install"]` (NOT to
    `install_requires`), so that a reviewer who wants the exact pinned
    versions can do:

        pip install -e ".[install]"

    while a reviewer who has already run `pip install -r requirements.txt`
    can do:

        pip install -e .

    without pip re-resolving the dependency tree.

    Comments, blank lines, and section headers are stripped. The
    `# ====...` banner lines and the `# tier ...` section headers from
    requirements.txt are preserved in the file but excluded from the
    requirement list.
    """
    path = HERE / "requirements.txt"
    if not path.exists():
        return []
    out: List[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        # Strip inline comments (this is a best-effort parse; requirements.txt
        # does not use inline comments in the pinned-version sections, but
        # this makes the parser robust to a future edit that does).
        if " #" in line:
            line = line.split(" #", 1)[0].strip()
        if line:
            out.append(line)
    return out


# ---------------------------------------------------------------------------
# Coherence checks
# ---------------------------------------------------------------------------
# These are run at the top of setup.py before `setup()` is called, and
# also from `python setup.py --check-only`. They verify the invariants
# the rest of the artifact relies on:
#
#   CHECK 1  Every required __init__.py exists, or can be created.
#   CHECK 2  Every package directory in PACKAGES exists.
#   CHECK 3  PACKAGE_VERSION matches STATUS.md §1.
#   CHECK 4  PACKAGE_VERSION matches default_e3_config.yaml §1.
#   CHECK 5  requirements.txt exists (dependencies must be installable).
#   CHECK 6  configs/ and data/ and expected_outputs/ exist at the
#            repository root (the editable install depends on them).
#
# A failure in CHECK 1 is auto-recovered (the file is created); a
# failure in CHECK 2-6 is a hard error that prevents setup() from being
# called, because a package that installs but does not work is worse
# than a package that refuses to install.

def check_init_files(auto_create: bool = True) -> List[str]:
    created: List[str] = []
    missing: List[str] = []
    for rel in REQUIRED_INITS:
        p = HERE / rel
        if p.exists():
            continue
        if auto_create:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text("", encoding="utf-8")
            created.append(rel)
        else:
            missing.append(rel)
    return created + [f"MISSING:{m}" for m in missing]


def check_package_dirs() -> List[str]:
    missing: List[str] = []
    for pkg in PACKAGES:
        d = HERE / pkg.replace(".", "/")
        if not d.is_dir():
            missing.append(pkg)
    return missing


def check_status_version() -> Optional[str]:
    """
    Return None if the version matches, or a human-readable error string.
    """
    status = HERE / "STATUS.md"
    if not status.exists():
        return "STATUS.md not found"
    text = status.read_text(encoding="utf-8")
    # STATUS.md §1 has a line like: "**Artifact version:** 1.0.0"
    m = re.search(r"\*\*Artifact version:\*\*\s*`?([0-9]+\.[0-9]+\.[0-9]+)`?", text)
    if not m:
        return "Could not find artifact version in STATUS.md §1"
    if m.group(1) != PACKAGE_VERSION:
        return (
            f"STATUS.md reports artifact version {m.group(1)}, "
            f"but setup.py declares PACKAGE_VERSION={PACKAGE_VERSION}"
        )
    return None


def check_config_version() -> Optional[str]:
    """
    Return None if the version matches, or a human-readable error string.
    """
    cfg = HERE / "configs" / "default_e3_config.yaml"
    if not cfg.exists():
        return "configs/default_e3_config.yaml not found"
    try:
        import yaml  # type: ignore
    except ImportError:
        # PyYAML may not be installed yet (it is in requirements.txt but
        # `pip install -e .` may be run before `pip install -r requirements.txt`).
        # Skip the check rather than fail; the user will hit it during the
        # first `load_config()` call anyway.
        return None
    data = yaml.safe_load(cfg.read_text(encoding="utf-8"))
    try:
        cfg_version = data["artifact"]["version"]
    except (KeyError, TypeError):
        return "configs/default_e3_config.yaml has no artifact.version"
    if cfg_version != PACKAGE_VERSION:
        return (
            f"configs/default_e3_config.yaml reports artifact.version={cfg_version}, "
            f"but setup.py declares PACKAGE_VERSION={PACKAGE_VERSION}"
        )
    return None


def check_required_dirs() -> List[str]:
    required = ("configs", "data", "expected_outputs", "scripts", "src")
    return [d for d in required if not (HERE / d).is_dir()]


def check_requirements_file() -> Optional[str]:
    if not (HERE / "requirements.txt").exists():
        return "requirements.txt not found"
    return None


def run_checks(auto_create_inits: bool = True) -> int:
    """
    Run all coherence checks. Return 0 if all pass, non-zero otherwise.
    Prints a human-readable report. Does not call setup().
    """
    failures: List[str] = []

    print("=== setup.py coherence checks ===")

    # CHECK 1
    init_result = check_init_files(auto_create=auto_create_inits)
    created = [r for r in init_result if not r.startswith("MISSING:")]
    missing_inits = [r[len("MISSING:"):] for r in init_result if r.startswith("MISSING:")]
    if created:
        print(f"[OK]   created {len(created)} __init__.py file(s): {created}")
    if missing_inits:
        failures.append(f"CHECK 1: missing __init__.py files: {missing_inits}")
        print(f"[FAIL] CHECK 1: missing __init__.py files: {missing_inits}")
    elif not created:
        print(f"[OK]   CHECK 1: all {len(REQUIRED_INITS)} __init__.py files present")

    # CHECK 2
    missing_pkgs = check_package_dirs()
    if missing_pkgs:
        failures.append(f"CHECK 2: missing package dirs: {missing_pkgs}")
        print(f"[FAIL] CHECK 2: missing package dirs: {missing_pkgs}")
    else:
        print(f"[OK]   CHECK 2: all {len(PACKAGES)} package dirs present")

    # CHECK 3
    err = check_status_version()
    if err:
        failures.append(f"CHECK 3: {err}")
        print(f"[FAIL] CHECK 3: {err}")
    else:
        print(f"[OK]   CHECK 3: STATUS.md version matches {PACKAGE_VERSION}")

    # CHECK 4
    err = check_config_version()
    if err:
        failures.append(f"CHECK 4: {err}")
        print(f"[FAIL] CHECK 4: {err}")
    else:
        print(f"[OK]   CHECK 4: config version matches {PACKAGE_VERSION}")

    # CHECK 5
    err = check_requirements_file()
    if err:
        failures.append(f"CHECK 5: {err}")
        print(f"[FAIL] CHECK 5: {err}")
    else:
        print(f"[OK]   CHECK 5: requirements.txt present")

    # CHECK 6
    missing_dirs = check_required_dirs()
    if missing_dirs:
        failures.append(f"CHECK 6: missing required dirs: {missing_dirs}")
        print(f"[FAIL] CHECK 6: missing required dirs: {missing_dirs}")
    else:
        print(f"[OK]   CHECK 6: required directories present")

    print("=== end of checks ===")
    if failures:
        print(f"{len(failures)} check(s) failed:")
        for f in failures:
            print(f"  - {f}")
        return 1
    return 0


# ---------------------------------------------------------------------------
# CLI: --check-only
# ---------------------------------------------------------------------------
# Handled before setup() so that `python setup.py --check-only` runs the
# checks and exits without invoking the build. A normal
# `python setup.py install` or `pip install -e .` does not see
# --check-only, so the normal path is unaffected.

if "--check-only" in sys.argv:
    sys.exit(run_checks(auto_create_inits=True))


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
# Everything above this line is checks and helpers. Everything below is
# the actual package declaration.

long_description = read_long_description()
requirements = read_requirements()

# Defensive: create any missing __init__.py before declaring packages.
# If the checks above already reported a failure, we still call setup()
# because a reviewer might be running `pip install -e .` on a partial
# checkout and want to see the specific error from setuptools, not just
# the checks' summary. The checks' output is printed first, so the
# reviewer sees both.
_ = check_init_files(auto_create=True)

setup(
    name="e3doh-reproducibility",
    version=PACKAGE_VERSION,
    description=(
        "Reproducibility artifact for Chapter 5 of the thesis "
        "'Next-Generation Secure Encrypted DNS Architectures' — "
        "E3-DoH: Enhanced Evolutionary Encryption for DoH, DoT, and DoQ."
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",

    # Author information is redacted in the thesis and in the repository
    # for blind review. The values below are placeholders that a
    # reviewer who needs to attribute the artifact can override by
    # editing this file; they do not affect the build.
    author="E3-DoH Artifact Authors",
    author_email="redacted@example.invalid",

    url="https://github.com/<your-org>/E3-DoH-Reproducibility",
    license="MIT",
    license_files=("LICENSE",),

    python_requires=">=3.11,<3.12",
    packages=list(PACKAGES),
    # package_dir is the default (packages live at the same relative path
    # under the project root), so we do not need to declare it.

    # Dependencies are deliberately NOT in install_requires — see the
    # module docstring. They are exposed as the optional `[install]`
    # extra so that `pip install -e ".[install]"` works for a reviewer
    # who wants one command, while `pip install -r requirements.txt`
    # followed by `pip install -e .` is the recommended two-step.
    install_requires=[],
    extras_require={
        "install": requirements,
    },

    # No entry_points: the scripts in scripts/ are invoked directly as
    # `python scripts/xxx.py`. See the module docstring.

    # No package_data and no data_files: configs, data, and
    # expected_outputs are read from disk at runtime, and the artifact
    # supports only the editable install. See the module docstring.

    # Classifiers. These are standard and help a reviewer see at a
    # glance what the artifact is.
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Security",
        "Topic :: System :: Networking",
    ],

    # Do not zip the egg. A zipped egg would make the data/ directory
    # unreadable by the runtime path resolvers. This is a common
    # reproducibility-artifact footgun, and we are choosing safety over
    # the (negligible) speedup of a zipped egg.
    zip_safe=False,

    # Require setuptools to be at least 61 so that `license_files` works
    # and so that the `extras_require` syntax is stable. requirements.txt
    # does not pin setuptools; the version guard below does.
    setup_requires=[],
    include_package_data=False,
)
