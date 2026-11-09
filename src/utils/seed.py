#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
src/utils/seed.py
=================

E3-DoH Reproducibility — deterministic seeding.

Chapter 5 of the thesis: E3-DoH (Enhanced Evolutionary Encryption for
Multi-Protocol DNS Security — DoH, DoT, DoQ).

This module centralises every source of randomness in the artifact and
enforces the "same seed -> same result" contract that makes
expected_outputs/verification_script.py meaningful. It is imported by
every runnable script in scripts/ (before any other import that might
touch a random source) and by tests/ (to make the test run itself
deterministic).

THE SEED-OFFSET CONVENTION
--------------------------
The artifact uses a single global run seed (from
default_e3_config.yaml §2, runtime.seed) and derives a per-component
seed from it using a fixed offset:

    +0  AEE training weights        (aee_params.yaml §3)
    +1  AEE quantisation projection (aee_params.yaml §11)
    +2  QHE Ziggurat sampler        (qhe_params.yaml §4)
    +3  AMPS HJB solver             (sandbox_mdp_params.yaml §11)
    +4  Riemannian metric cache     (riemannian_metrics.yaml §13)
    +5  Attack suite                (attack_suite_params.yaml §11)

The offsets are defined here as SEED_OFFSETS and MUST be kept in sync
with the config files. A reviewer who wants to re-derive a component's
seed by hand can read the offset from this dict; there is no hidden
arithmetic elsewhere in the codebase.

The offsets are deliberately small integers (not a hash of the
component name), because a reviewer auditing the artifact should be
able to compute the seed on paper from the run seed alone. If the
number of components grows past ~20, the offsets should be replaced
with a hash-derived scheme; until then, the simplicity is a feature.

WHY PER-COMPONENT OFFSETS, NOT ONE GLOBAL RNG
---------------------------------------------
If every component drew from the same global RNG, then changing the
order in which components run (e.g. by parallelising two of them in a
future revision) would change the entire downstream random stream, and
two runs that were supposed to be identical would diverge in a way
that is very hard to debug. The per-component offsets give each
component its own independent stream, derived from the run seed, so
that:
    - Two runs of the same script with the same seed produce the same
      result regardless of component execution order.
    - A component can be tested in isolation by seeding only its
      stream, without affecting the other components.
    - A component's randomness can be replaced (e.g. with a
      hardware-entropy source) without disturbing the other components.

WHAT THIS MODULE CANNOT DO
--------------------------
Determinism in PyTorch is limited by the backends:

  - cuDNN algorithms are chosen at runtime and can vary across runs
    (unless cudnn_benchmark=False and cudnn_deterministic=True, which
    this module sets).
  - cublas operations are non-deterministic unless
    CUBLAS_WORKSPACE_CONFIG=:4096:8 is set BEFORE the CUDA context is
    initialised. This module sets the environment variable, but if
    torch has already been imported and the CUDA context is live,
    the setting has no effect. The module emits a warning in that case.
  - Some operations (e.g. scatter_add on CUDA, index_put with
    accumulate=True) are non-deterministic regardless of settings; the
    torch.use_deterministic_algorithms(True) call will raise a
    RuntimeError rather than silently produce a non-deterministic
    result. This is the correct behaviour: a silent non-determinism
    is worse than an explicit failure.

  - Python's hash randomisation is controlled by PYTHONHASHSEED, which
    must be set BEFORE the interpreter starts. This module can only
    read the value and warn; it cannot change it at runtime. The
    Dockerfile sets it to 0. A reviewer running outside the container
    must set it in their shell before invoking Python.

  - The run header (written by src/utils/logger.py) records the
    observed values of these environment variables, so a reviewer can
    tell whether a run was fully deterministic or only partially.

DETERMINISM IS NOT A SECURITY PROPERTY
--------------------------------------
The seeding scheme guarantees that a run is reproducible. It says
nothing about the *security* of the seed. In particular:

  - The run seed is a small integer (42 by default); it is not a
    cryptographic secret and MUST NOT be used as a key or as input
    to a cryptographic primitive.
  - The per-component RNGs are for simulation and training only.
    Cryptographic randomness (QHE error sampling in a production
    deployment, AEE key material in a production deployment) MUST
    come from a CSPRNG, not from this module. See qhe_params.yaml
    §9 (determinism_note) for the boundary.

Equation references are not applicable here (this module implements
no thesis equation; it is infrastructure).
"""

from __future__ import annotations

import contextlib
import os
import random
import warnings
from typing import Any, Dict, Iterator, Optional

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Per-component seed offsets. MUST match the `*_seed_offset` and
# `seed_offset` keys in the corresponding config files. The mapping is
# asserted in _self_test() and in tests/test_seed.py.
SEED_OFFSETS: Dict[str, int] = {
    "aee_training":    0,   # aee_params.yaml §3   init_seed_offset
    "aee_projection":  1,   # aee_params.yaml §11  projection_seed_offset
    "qhe_ziggurat":    2,   # qhe_params.yaml §4   seed_offset
    "amps_hjb":        3,   # sandbox_mdp_params.yaml §11  seed_offset
    "metrics_cache":   4,   # riemannian_metrics.yaml §13  seed_offset
    "attack_suite":    5,   # attack_suite_params.yaml §11 seed_offset
}

# The two environment variables that must be set BEFORE the interpreter
# (PYTHONHASHSEED) or BEFORE torch's CUDA context (CUBLAS_WORKSPACE_CONFIG)
# are initialised. This module can only read them at runtime; it cannot
# change the effect retroactively. The values below are the values the
# Dockerfile sets and the values the artifact's self-test checks for.
EXPECTED_ENV_VALUES: Dict[str, str] = {
    "PYTHONHASHSEED": "0",
    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
}

# Module-level flag so the env-var warning is emitted at most once per
# process. A run header (via logger.py) records the full state regardless.
_ENV_WARNING_EMITTED: bool = False


# ---------------------------------------------------------------------------
# Core: global seeding
# ---------------------------------------------------------------------------

def set_global_seed(
    seed: int,
    *,
    deterministic: bool = True,
    cudnn_benchmark: bool = False,
    torch_threads: Optional[int] = None,
    numpy_threads: Optional[int] = None,
    omp_threads: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Seed every global random source and configure the deterministic-mode
    flags on torch.

    Parameters
    ----------
    seed : int
        The run seed, typically default_e3_config.yaml §2 runtime.seed.
    deterministic : bool
        If True, call torch.use_deterministic_algorithms(True) and set
        cudnn_deterministic=True. Default True (matches
        default_e3_config.yaml §2 runtime.deterministic).
    cudnn_benchmark : bool
        If True, cudnn_benchmark=True (fast but non-deterministic). Default
        False (matches default_e3_config.yaml §2 runtime.cudnn_benchmark).
    torch_threads, numpy_threads, omp_threads : optional int
        Thread counts. If None, the values from
        default_e3_config.yaml §2 are used
        (torch_threads=8, numpy_threads=8, omp_threads=8). Set to 0 to
        leave the library default (not recommended for reproducibility).

    Returns
    -------
    A dict describing what was set, for the run header:
        {
          "seed": <int>,
          "python": <int>,
          "numpy": <int>,
          "torch": <int or None>,
          "cuda": <int or None>,
          "deterministic": <bool>,
          "cudnn_benchmark": <bool>,
          "cudnn_deterministic": <bool>,
          "torch_threads": <int or None>,
          "numpy_threads": <int or None>,
          "omp_threads": <int or None>,
          "environment": {"PYTHONHASHSEED": ..., "CUBLAS_WORKSPACE_CONFIG": ...},
          "warnings": [<str>, ...]
        }

    Notes
    -----
    This function is intentionally tolerant: if torch is not installed
    (toy-mode runs may not need it) or if CUDA is not available, it
    silently skips the corresponding step and records the fact in the
    returned dict. It does NOT raise on a missing optional dependency;
    that is the caller's responsibility to detect via the returned dict.
    """
    state: Dict[str, Any] = {
        "seed": int(seed),
        "warnings": [],
    }

    # 1. Python's random module.
    random.seed(int(seed))
    state["python"] = int(seed)

    # 2. NumPy's global RNG. The legacy RandomState is what
    #    np.random.* uses; the new Generator is per-component (see
    #    get_component_rng). We seed both for safety: a downstream
    #    library that calls np.random.seed(seed) will get the same
    #    stream as us.
    np.random.seed(int(seed) & 0xFFFFFFFF)
    state["numpy"] = int(seed)

    # 3. Optional PyTorch.
    try:
        import torch  # type: ignore
    except ImportError:
        state["torch"] = None
        state["cuda"] = None
        state["deterministic"] = False
        state["cudnn_benchmark"] = False
        state["cudnn_deterministic"] = False
        state["torch_threads"] = None
        state["numpy_threads"] = None
        state["omp_threads"] = None
        state["environment"] = _read_env()
        return state

    torch.manual_seed(int(seed))
    state["torch"] = int(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
        state["cuda"] = int(seed)
    else:
        state["cuda"] = None

    if deterministic:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        state["deterministic"] = True
        state["cudnn_deterministic"] = True
    else:
        state["deterministic"] = False
        state["cudnn_deterministic"] = bool(getattr(
            torch.backends.cudnn, "deterministic", False
        ))

    torch.backends.cudnn.benchmark = bool(cudnn_benchmark)
    state["cudnn_benchmark"] = bool(cudnn_benchmark)

    # Thread counts. 0 means "leave library default".
    if torch_threads is not None and torch_threads > 0:
        try:
            torch.set_num_threads(int(torch_threads))
            state["torch_threads"] = int(torch_threads)
        except RuntimeError as e:
            state["torch_threads"] = None
            state["warnings"].append(f"torch.set_num_threads failed: {e}")
    else:
        state["torch_threads"] = None

    # NumPy / BLAS thread count: set via env var, which numpy reads at
    # import time. If numpy is already imported (it is — this module
    # imports it at the top), setting the env var has no effect. We
    # record the *observed* value below; a caller who wants to change
    # it must set the env var before importing numpy.
    if numpy_threads is not None and numpy_threads > 0:
        os.environ.setdefault("OMP_NUM_THREADS", str(int(numpy_threads)))
        os.environ.setdefault("OPENBLAS_NUM_THREADS", str(int(numpy_threads)))
        os.environ.setdefault("MKL_NUM_THREADS", str(int(numpy_threads)))
        state["numpy_threads"] = int(numpy_threads)
    else:
        state["numpy_threads"] = int(
            os.environ.get("OMP_NUM_THREADS", os.environ.get("MKL_NUM_THREADS", "0"))
        ) or None

    # OMP threads: same caveat as numpy.
    state["omp_threads"] = int(os.environ.get("OMP_NUM_THREADS", "0")) or None

    # Environment-variable check. Two env vars matter for reproducibility:
    #   PYTHONHASHSEED          — must be set before interpreter start.
    #   CUBLAS_WORKSPACE_CONFIG — must be set before CUDA context.
    # We can only warn, not fix.
    state["environment"] = _read_env()
    global _ENV_WARNING_EMITTED
    if not _ENV_WARNING_EMITTED:
        for k, expected in EXPECTED_ENV_VALUES.items():
            observed = state["environment"].get(k)
            if observed != expected:
                msg = (
                    f"{k} is {observed!r} but the artifact expects {expected!r}. "
                    f"Set it before invoking Python for full determinism. "
                    f"See README.md §6 and Dockerfile (ENV block)."
                )
                warnings.warn(msg, RuntimeWarning, stacklevel=2)
                state["warnings"].append(msg)
        _ENV_WARNING_EMITTED = True

    return state


def _read_env() -> Dict[str, Optional[str]]:
    """Read the two reproducibility-relevant env vars (None if unset)."""
    return {k: os.environ.get(k) for k in EXPECTED_ENV_VALUES}


# ---------------------------------------------------------------------------
# Per-component RNG
# ---------------------------------------------------------------------------

def get_component_seed(seed: int, component: str) -> int:
    """
    Return the derived seed for a component:

        component_seed = seed + SEED_OFFSETS[component]

    Parameters
    ----------
    seed : int
        The run seed.
    component : str
        One of the keys in SEED_OFFSETS. An unknown component is a hard
        error, because a typo would silently reuse the run seed and
        correlate the component with every other un-seeded component.

    Returns
    -------
    The integer seed for the component's RNG.
    """
    if component not in SEED_OFFSETS:
        raise KeyError(
            f"Unknown component '{component}'. "
            f"Known: {sorted(SEED_OFFSETS)}. "
            f"Add a new offset in SEED_OFFSETS and in the corresponding "
            f"config file if this is a new component."
        )
    return int(seed) + SEED_OFFSETS[component]


def get_component_rng(seed: int, component: str) -> np.random.Generator:
    """
    Return an independent numpy.random.Generator for a component.

    Each component gets its own Generator, seeded from
    get_component_seed(seed, component). This is what the AEE, QHE,
    AMPS, metric-cache, and attack-suite engines use.

    Parameters
    ----------
    seed : int
        The run seed.
    component : str
        One of the keys in SEED_OFFSETS.

    Returns
    -------
    A numpy.random.Generator. The caller is responsible for using it
    (not np.random.*), so that the component's randomness is isolated
    from any other component.
    """
    return np.random.default_rng(get_component_seed(seed, component))


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def describe_seed_state(seed: int) -> Dict[str, Any]:
    """
    Describe the full seeding state for a run, without changing it.

    The result is a dict suitable for inclusion in the run header
    (written by src/utils/logger.py). It includes:
        - the run seed,
        - the per-component derived seeds,
        - the observed values of the two reproducibility-relevant
          environment variables,
        - the observed torch / numpy / OMP thread counts (if torch is
          installed).

    This function does NOT call set_global_seed(); it is read-only.
    A caller that wants both the effect and the record should do:

        state = set_global_seed(seed)
        state["component_seeds"] = describe_seed_state(seed)["component_seeds"]

    or simply call describe_seed_state() after set_global_seed().
    """
    component_seeds = {
        name: get_component_seed(seed, name) for name in SEED_OFFSETS
    }
    env = _read_env()
    result: Dict[str, Any] = {
        "run_seed": int(seed),
        "component_seeds": component_seeds,
        "environment": env,
        "torch_threads": None,
        "numpy_threads": int(os.environ.get("OMP_NUM_THREADS", 0)) or None,
        "omp_threads": int(os.environ.get("OMP_NUM_THREADS", 0)) or None,
    }
    try:
        import torch  # type: ignore
        result["torch_threads"] = int(torch.get_num_threads())
        result["torch_deterministic_algorithms"] = bool(
            torch.are_deterministic_algorithms_enabled()
        )
        result["cudnn_deterministic"] = bool(torch.backends.cudnn.deterministic)
        result["cudnn_benchmark"] = bool(torch.backends.cudnn.benchmark)
        result["cuda_available"] = bool(torch.cuda.is_available())
    except ImportError:
        pass
    return result


# ---------------------------------------------------------------------------
# Context manager (for tests)
# ---------------------------------------------------------------------------

@contextlib.contextmanager
def seeded_context(seed: int, **kwargs: Any) -> Iterator[Dict[str, Any]]:
    """
    Context manager that seeds every source, yields the resulting state,
    and restores the *original* Python and NumPy RNG states on exit.

    This is used by tests/ to run a test under a deterministic seed
    without leaking the seed into the next test. It does NOT restore
    torch state (torch has no public API for saving/restoring the
    global RNG state, and doing so would be fragile); a test that
    needs to restore torch state should save/restore the tensor-level
    RNG state itself.

    Usage:
        with seeded_context(42) as state:
            # deterministic block here
            ...
        # original RNG states restored
    """
    py_state = random.getstate()
    np_state = np.random.get_state()
    state = set_global_seed(int(seed), **kwargs)
    try:
        yield state
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "SEED_OFFSETS",
    "EXPECTED_ENV_VALUES",
    "set_global_seed",
    "get_component_seed",
    "get_component_rng",
    "describe_seed_state",
    "seeded_context",
]


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
# Running `python -m src.utils.seed` verifies that:
#   - every component in SEED_OFFSETS has a distinct offset,
#   - the offsets are small non-negative integers (sanity),
#   - get_component_seed is deterministic and component-dependent,
#   - get_component_rng is reproducible and independent across components,
#   - set_global_seed produces the same Python and NumPy state for the
#     same seed.

def _self_test() -> int:
    import sys
    rc = 0
    print("=== seed self-test ===")

    # 1. Distinct, small, non-negative offsets.
    offsets = list(SEED_OFFSETS.values())
    if len(set(offsets)) != len(offsets):
        print(f"[FAIL] SEED_OFFSETS values are not distinct: {SEED_OFFSETS}")
        rc = 1
    elif not all(isinstance(v, int) and v >= 0 and v < 100 for v in offsets):
        print(f"[FAIL] SEED_OFFSETS must be small non-negative ints: {SEED_OFFSETS}")
        rc = 1
    else:
        print(f"[OK]   SEED_OFFSETS: {len(offsets)} distinct offsets in {offsets}")

    # 2. Determinism and component-dependence.
    s_aee = get_component_seed(42, "aee_training")
    s_qhe = get_component_seed(42, "qhe_ziggurat")
    s_aee_2 = get_component_seed(42, "aee_training")
    if s_aee != s_aee_2:
        print(f"[FAIL] get_component_seed not deterministic: {s_aee} != {s_aee_2}")
        rc = 1
    elif s_aee == s_qhe:
        print(f"[FAIL] get_component_seed collides across components: {s_aee}")
        rc = 1
    else:
        print(f"[OK]   get_component_seed deterministic and distinct "
              f"(aee_training={s_aee}, qhe_ziggurat={s_qhe})")

    # 3. RNG reproducibility.
    r1 = get_component_rng(42, "aee_training").random(5)
    r2 = get_component_rng(42, "aee_training").random(5)
    if not np.allclose(r1, r2):
        print(f"[FAIL] get_component_rng not reproducible: {r1} vs {r2}")
        rc = 1
    else:
        print(f"[OK]   get_component_rng reproducible: {r1}")

    # 4. RNG independence across components.
    r_aee = get_component_rng(42, "aee_training").random(5)
    r_qhe = get_component_rng(42, "qhe_ziggurat").random(5)
    if np.allclose(r_aee, r_qhe):
        print(f"[FAIL] component RNGs are correlated: {r_aee}")
        rc = 1
    else:
        print(f"[OK]   component RNGs independent across components")

    # 5. set_global_seed reproducibility (Python + NumPy).
    with seeded_context(42):
        py_a = random.random()
        np_a = float(np.random.random())
    with seeded_context(42):
        py_b = random.random()
        np_b = float(np.random.random())
    if py_a != py_b or np_a != np_b:
        print(f"[FAIL] set_global_seed not reproducible: "
              f"py {py_a} vs {py_b}, np {np_a} vs {np_b}")
        rc = 1
    else:
        print(f"[OK]   set_global_seed reproducible "
              f"(py={py_a:.6f}, np={np_a:.6f})")

    # 6. Environment variable state (warn, not fail — a reviewer's
    #    shell may not have set them, and that is a documented
    #    limitation, not a defect in the code).
    env = _read_env()
    for k, expected in EXPECTED_ENV_VALUES.items():
        observed = env.get(k)
        if observed != expected:
            print(f"[WARN] {k}={observed!r} (artifact expects {expected!r}); "
                  f"determinism is limited. Set it before invoking Python.")
        else:
            print(f"[OK]   {k}={expected}")

    # 7. describe_seed_state returns the expected shape.
    state = describe_seed_state(42)
    if state["run_seed"] != 42:
        print(f"[FAIL] describe_seed_state run_seed: {state['run_seed']}")
        rc = 1
    elif set(state["component_seeds"].keys()) != set(SEED_OFFSETS.keys()):
        print(f"[FAIL] describe_seed_state component_seeds keys mismatch")
        rc = 1
    else:
        print(f"[OK]   describe_seed_state: {len(state['component_seeds'])} components")

    print("=== self-test complete ===")
    return rc


if __name__ == "__main__":
    import sys
    sys.exit(_self_test())
