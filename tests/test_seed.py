#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tests/test_seed.py
==================

E3-DoH Reproducibility — unit tests for src/utils/seed.py.

These tests verify the seeding module's public contract:

    - SEED_OFFSETS is well-formed (distinct, small, non-negative)
    - get_component_seed is deterministic and component-distinct
    - get_component_rng is reproducible and independent across components
    - set_global_seed is reproducible (Python + NumPy)
    - describe_seed_state returns the expected shape
    - seeded_context restores the original RNG state on exit

The tests do NOT verify the exact value of any derived seed — that
would couple the test to the specific offsets and make a legitimate
config change look like a regression. They verify the *invariants* the
rest of the artifact relies on.

Run with:
    pytest tests/test_seed.py -v
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.utils.seed import (
    EXPECTED_ENV_VALUES,
    SEED_OFFSETS,
    describe_seed_state,
    get_component_rng,
    get_component_seed,
    seeded_context,
    set_global_seed,
)


# =============================================================================
# SEED_OFFSETS invariant
# =============================================================================

class TestSeedOffsets:
    def test_offsets_are_distinct(self):
        values = list(SEED_OFFSETS.values())
        assert len(set(values)) == len(values), (
            f"SEED_OFFSETS has duplicate values: {SEED_OFFSETS}"
        )

    def test_offsets_are_small_non_negative(self):
        for name, offset in SEED_OFFSETS.items():
            assert isinstance(offset, int), f"{name}: offset is not int"
            assert 0 <= offset < 100, f"{name}: offset {offset} out of range"

    def test_offsets_are_contiguous(self):
        # The offsets should be a contiguous range starting at 0 so that
        # adding a new component requires only adding one new offset.
        values = sorted(SEED_OFFSETS.values())
        assert values == list(range(len(values))), (
            f"SEED_OFFSETS is not a contiguous range starting at 0: {values}"
        )

    def test_known_component_names(self):
        # The component names the config files reference.
        expected = {
            "aee_training", "aee_projection", "qhe_ziggurat",
            "amps_hjb", "metrics_cache", "attack_suite",
        }
        assert set(SEED_OFFSETS.keys()) == expected


# =============================================================================
# get_component_seed
# =============================================================================

class TestGetComponentSeed:
    def test_deterministic(self):
        for name in SEED_OFFSETS:
            s1 = get_component_seed(42, name)
            s2 = get_component_seed(42, name)
            assert s1 == s2

    def test_distinct_across_components(self):
        seeds = [get_component_seed(42, name) for name in SEED_OFFSETS]
        assert len(set(seeds)) == len(seeds)

    def test_changes_with_run_seed(self):
        s42 = get_component_seed(42, "aee_training")
        s43 = get_component_seed(43, "aee_training")
        assert s42 != s43

    def test_unknown_component_raises(self):
        with pytest.raises(KeyError):
            get_component_seed(42, "not_a_component")

    def test_arithmetic(self):
        # The function should be exactly `seed + offset`.
        for name, offset in SEED_OFFSETS.items():
            assert get_component_seed(100, name) == 100 + offset


# =============================================================================
# get_component_rng
# =============================================================================

class TestGetComponentRng:
    def test_reproducible(self):
        for name in SEED_OFFSETS:
            r1 = get_component_rng(42, name).random(10)
            r2 = get_component_rng(42, name).random(10)
            np.testing.assert_array_equal(r1, r2)

    def test_independent_across_components(self):
        names = list(SEED_OFFSETS.keys())
        draws = [get_component_rng(42, name).random(10) for name in names]
        # No two components should produce the same 10-value sequence.
        for i in range(len(draws)):
            for j in range(i + 1, len(draws)):
                assert not np.allclose(draws[i], draws[j]), (
                    f"{names[i]} and {names[j]} produced identical draws"
                )

    def test_changes_with_run_seed(self):
        r42 = get_component_rng(42, "aee_training").random(10)
        r43 = get_component_rng(43, "aee_training").random(10)
        assert not np.allclose(r42, r43)


# =============================================================================
# set_global_seed
# =============================================================================

class TestSetGlobalSeed:
    def test_python_deterministic(self):
        set_global_seed(42)
        a = random.random()
        set_global_seed(42)
        b = random.random()
        assert a == b

    def test_numpy_deterministic(self):
        set_global_seed(42)
        a = float(np.random.random())
        set_global_seed(42)
        b = float(np.random.random())
        assert a == b

    def test_different_seeds_differ(self):
        set_global_seed(42)
        a = random.random()
        set_global_seed(999)
        b = random.random()
        assert a != b

    def test_returns_dict_with_expected_keys(self):
        state = set_global_seed(42)
        for key in ("seed", "python", "numpy", "warnings", "environment"):
            assert key in state, f"missing key: {key}"

    def test_environment_recorded(self):
        state = set_global_seed(42)
        env = state["environment"]
        for key in EXPECTED_ENV_VALUES:
            assert key in env


# =============================================================================
# describe_seed_state
# =============================================================================

class TestDescribeSeedState:
    def test_run_seed_preserved(self):
        state = describe_seed_state(42)
        assert state["run_seed"] == 42

    def test_component_seeds_populated(self):
        state = describe_seed_state(42)
        assert set(state["component_seeds"].keys()) == set(SEED_OFFSETS.keys())

    def test_component_seed_values_match(self):
        state = describe_seed_state(42)
        for name in SEED_OFFSETS:
            assert state["component_seeds"][name] == get_component_seed(42, name)

    def test_environment_recorded(self):
        state = describe_seed_state(42)
        for key in EXPECTED_ENV_VALUES:
            assert key in state["environment"]


# =============================================================================
# seeded_context
# =============================================================================

class TestSeededContext:
    def test_reproducible_block(self):
        with seeded_context(42) as state:
            a_py = random.random()
            a_np = float(np.random.random())
        with seeded_context(42) as state:
            b_py = random.random()
            b_np = float(np.random.random())
        assert a_py == b_py
        assert a_np == b_np

    def test_restores_python_state(self):
        random.seed(999)
        before = random.random()
        random.seed(999)
        with seeded_context(42):
            random.random()  # consume
        after = random.random()
        assert before == after

    def test_restores_numpy_state(self):
        np.random.seed(999)
        before = float(np.random.random())
        np.random.seed(999)
        with seeded_context(42):
            np.random.random()  # consume
        after = float(np.random.random())
        assert before == after

    def test_restores_on_exception(self):
        random.seed(999)
        before = random.random()
        random.seed(999)
        with pytest.raises(RuntimeError):
            with seeded_context(42):
                raise RuntimeError("boom")
        after = random.random()
        assert before == after

    def test_yields_state_dict(self):
        with seeded_context(42) as state:
            assert "seed" in state
            assert state["seed"] == 42


# =============================================================================
# Cross-seed determinism
# =============================================================================

class TestCrossSeedDeterminism:
    def test_full_seed_cycle(self):
        """
        A complete cycle: seed the global sources, derive a component
        RNG, and draw. Repeat with the same seed and check the sequence
        is identical. This is the pattern the engines use.
        """
        def run_once():
            set_global_seed(42)
            rng = get_component_rng(42, "aee_training")
            return {
                "py": random.random(),
                "np_global": float(np.random.random()),
                "np_component": rng.random(5).tolist(),
            }

        a = run_once()
        b = run_once()
        assert a["py"] == b["py"]
        assert a["np_global"] == b["np_global"]
        assert a["np_component"] == b["np_component"]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
