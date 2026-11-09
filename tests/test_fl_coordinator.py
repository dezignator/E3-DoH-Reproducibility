#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tests/test_fl_coordinator.py
============================

E3-DoH Reproducibility — unit tests for src/models/fl_coordinator.py.

These tests verify the FL coordinator's public contract:

    - Construction and client registration (protocol matching)
    - Federated round execution
    - Geometric median (Weiszfeld IRLS) on known points
    - MAD outlier exclusion (Byzantine robustness)
    - Privacy accounting: realized ε monotone in rounds and noise
    - Adaptive clipping

The tests use small model dimensions (12) and small client populations
(10) so they run in well under a second each.

Run with:
    pytest tests/test_fl_coordinator.py -v
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.models.fl_coordinator import (
    ClientUpdate,
    FLAggregationError,
    FLCoordinator,
    FLClient,
    FLConfigError,
    FLError,
    RoundRecord,
)


# =============================================================================
# Shared fixtures
# =============================================================================

def _make_clients(
    n: int,
    protocol: str = "doh",
    dim: int = 12,
    n_samples: int = 20,
    seed: int = 0,
) -> list:
    """Build `n` clients with synthetic local datasets."""
    rng = np.random.default_rng(seed)
    clients = []
    for i in range(n):
        X = rng.normal(size=(n_samples, dim))
        w = rng.normal(size=dim)
        y = X @ w + rng.normal(scale=0.1, size=n_samples)
        clients.append(FLClient(
            client_id=f"client-{i:03d}", protocol=protocol,
            local_data=(X, y), seed=seed + i,
        ))
    return clients


class _FixedClient(FLClient):
    """A client whose update is a constant vector (for reproducibility tests)."""

    def __init__(self, client_id: str, protocol: str, value: np.ndarray):
        # Build a minimal dataset so the parent constructor is happy;
        # compute_update() ignores it.
        super().__init__(
            client_id=client_id, protocol=protocol,
            local_data=(np.zeros((1, len(value))), np.zeros(1)),
        )
        self._fixed_value = np.asarray(value, dtype=np.float64)

    def compute_update(self, global_params, poisoned=False):
        return self._fixed_value.copy()


class _PoisonedClient(FLClient):
    """A client whose update is far from all others (Byzantine)."""

    def __init__(self, client_id: str, protocol: str, dim: int, scale: float = 100.0):
        super().__init__(
            client_id=client_id, protocol=protocol,
            local_data=(np.zeros((1, dim)), np.zeros(1)),
        )
        self._scale = float(scale)
        self._dim = dim

    def compute_update(self, global_params, poisoned=False):
        v = np.zeros(self._dim)
        v[0] = self._scale
        return v


# =============================================================================
# Construction
# =============================================================================

class TestConstruction:
    def test_construct(self, minimal_fl_config):
        coord = FLCoordinator(minimal_fl_config, protocol="doh", model_dim=12, seed=42)
        assert coord.parameters["protocol"] == "doh"
        assert coord.parameters["n_clients"] == 10
        assert coord.parameters["model_dim"] == 12

    def test_unknown_protocol_rejected(self, minimal_fl_config):
        with pytest.raises(FLConfigError):
            FLCoordinator(minimal_fl_config, protocol="foo")

    def test_missing_config_raises(self):
        with pytest.raises(FLConfigError):
            FLCoordinator({}, protocol="doh")

    def test_initial_global_params_zero(self, minimal_fl_config):
        coord = FLCoordinator(minimal_fl_config, protocol="doh", model_dim=12, seed=42)
        np.testing.assert_array_equal(coord.global_params, np.zeros(12))


# =============================================================================
# Client registration
# =============================================================================

class TestClientRegistration:
    def test_register_one(self, minimal_fl_config):
        coord = FLCoordinator(minimal_fl_config, protocol="doh", model_dim=12, seed=42)
        clients = _make_clients(1)
        coord.register_client(clients[0])
        assert len(coord._clients) == 1

    def test_protocol_mismatch_rejected(self, minimal_fl_config):
        coord = FLCoordinator(minimal_fl_config, protocol="doh", model_dim=12, seed=42)
        wrong = FLClient(
            client_id="wrong", protocol="dot",
            local_data=(np.zeros((1, 12)), np.zeros(1)),
        )
        with pytest.raises(FLError):
            coord.register_client(wrong)

    def test_register_many(self, minimal_fl_config):
        coord = FLCoordinator(minimal_fl_config, protocol="doh", model_dim=12, seed=42)
        coord.register_clients(_make_clients(5))
        assert len(coord._clients) == 5

    def test_empty_client_rejected(self):
        with pytest.raises(Exception):
            FLClient(
                client_id="empty", protocol="doh",
                local_data=(np.zeros((0, 12)), np.zeros(0)),
            )


# =============================================================================
# Round execution
# =============================================================================

class TestRound:
    def test_run_round_without_clients_raises(self, minimal_fl_config):
        coord = FLCoordinator(minimal_fl_config, protocol="doh", model_dim=12, seed=42)
        with pytest.raises(FLError):
            coord.run_round()

    def test_run_round_produces_record(self, minimal_fl_config):
        coord = FLCoordinator(minimal_fl_config, protocol="doh", model_dim=12, seed=42)
        coord.register_clients(_make_clients(5))
        record = coord.run_round()
        assert isinstance(record, RoundRecord)
        assert record.round_number == 1
        assert record.n_clients_sampled >= 1
        assert record.n_clients_included >= 1
        assert record.realized_epsilon > 0

    def test_global_params_change_after_round(self, minimal_fl_config):
        coord = FLCoordinator(minimal_fl_config, protocol="doh", model_dim=12, seed=42)
        coord.register_clients(_make_clients(5))
        before = coord.global_params.copy()
        coord.run_round()
        after = coord.global_params
        assert not np.array_equal(before, after)

    def test_run_multiple_rounds(self, minimal_fl_config):
        coord = FLCoordinator(minimal_fl_config, protocol="doh", model_dim=12, seed=42)
        coord.register_clients(_make_clients(5))
        history = coord.run(n_rounds=3)
        assert len(history) == 3


# =============================================================================
# Geometric median
# =============================================================================

class TestGeometricMedian:
    def test_three_points_origin(self, minimal_fl_config):
        # Three points: (0,0), (10,0), (0,10). The geometric median
        # is (0,0).
        coord = FLCoordinator(minimal_fl_config, protocol="doh", model_dim=2, seed=42)
        pts = np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0]])
        w = np.array([1.0, 1.0, 1.0])
        gm = coord._geometric_median(pts, w)
        assert np.linalg.norm(gm) < 0.1

    def test_collinear_points_middle(self, minimal_fl_config):
        # Three collinear points: the geometric median is the middle one.
        coord = FLCoordinator(minimal_fl_config, protocol="doh", model_dim=2, seed=42)
        pts = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
        w = np.array([1.0, 1.0, 1.0])
        gm = coord._geometric_median(pts, w)
        np.testing.assert_allclose(gm, [1.0, 0.0], atol=0.1)


# =============================================================================
# MAD outlier exclusion
# =============================================================================

class TestMADOutlier:
    def test_single_outlier_excluded(self, minimal_fl_config):
        coord = FLCoordinator(minimal_fl_config, protocol="doh", model_dim=12, seed=42)
        # 5 honest clients near +1, 1 poisoned client near -100.
        for i in range(5):
            coord.register_client(_FixedClient(
                f"honest-{i}", "doh", np.ones(12),
            ))
        coord.register_client(_FixedClient(
            "malicious", "doh", np.full(12, -100.0),
        ))
        # Sample all clients.
        coord._client_sample_rate = 1.0
        coord._noise_multiplier = 0.0  # no DP noise, see clearly
        record = coord.run_round()
        # The MAD filter should exclude the poisoned client.
        assert record.n_clients_excluded >= 1
        assert record.n_clients_included <= 5

    def test_all_same_keeps_all(self, minimal_fl_config):
        coord = FLCoordinator(minimal_fl_config, protocol="doh", model_dim=12, seed=42)
        for i in range(3):
            coord.register_client(_FixedClient(
                f"c-{i}", "doh", np.ones(12),
            ))
        coord._client_sample_rate = 1.0
        coord._noise_multiplier = 0.0
        record = coord.run_round()
        assert record.n_clients_excluded == 0
        assert record.n_clients_included == 3


# =============================================================================
# Privacy accounting
# =============================================================================

class TestPrivacyAccounting:
    def test_target_epsilon_from_config(self, minimal_fl_config):
        coord = FLCoordinator(minimal_fl_config, protocol="doh", model_dim=12, seed=42)
        target, realized = coord.privacy_budget()
        assert target == 1.2  # from minimal_fl_config

    def test_realized_epsilon_positive(self, minimal_fl_config):
        coord = FLCoordinator(minimal_fl_config, protocol="doh", model_dim=12, seed=42)
        _, realized = coord.privacy_budget()
        assert realized > 0

    def test_realized_epsilon_monotone_in_rounds(self, minimal_fl_config):
        coord = FLCoordinator(minimal_fl_config, protocol="doh", model_dim=12, seed=42)
        eps_5 = coord.compute_realized_epsilon(n_rounds=5)
        eps_10 = coord.compute_realized_epsilon(n_rounds=10)
        assert eps_10 >= eps_5

    def test_realized_epsilon_monotone_in_noise(self, minimal_fl_config):
        coord_hi = FLCoordinator(minimal_fl_config, protocol="doh", model_dim=12, seed=42)
        coord_hi._noise_multiplier = 1.2
        coord_lo = FLCoordinator(minimal_fl_config, protocol="doh", model_dim=12, seed=42)
        coord_lo._noise_multiplier = 3.0
        eps_hi = coord_hi.compute_realized_epsilon(n_rounds=10)
        eps_lo = coord_lo.compute_realized_epsilon(n_rounds=10)
        assert eps_lo < eps_hi


# =============================================================================
# Adaptive clipping
# =============================================================================

class TestAdaptiveClipping:
    def test_adaptive_clip_bounded(self, minimal_fl_config):
        # Enable adaptive clipping in the config.
        cfg = dict(minimal_fl_config)
        cfg["_includes"] = dict(cfg["_includes"])
        cfg["_includes"]["fl_dp"] = dict(cfg["_includes"]["fl_dp"])
        cfg["_includes"]["fl_dp"]["dp_sgd"] = dict(cfg["_includes"]["fl_dp"]["dp_sgd"])
        cfg["_includes"]["fl_dp"]["dp_sgd"]["adaptive_clipping"] = {
            "enabled": True, "quantile": 0.5,
            "min": 1.0, "max": 8.0,
            "recompute_interval_rounds": 1,
            "quantile_noise_multiplier": 0.5,
        }
        coord = FLCoordinator(cfg, protocol="doh", model_dim=12, seed=42)
        coord.register_clients(_make_clients(5, seed=1))
        coord.run_round()
        # The clip norm should stay within [min, max].
        assert 1.0 <= coord._current_clip_norm <= 8.0


# =============================================================================
# Convergence
# =============================================================================

class TestConvergence:
    def test_run_stops_at_max_rounds(self, minimal_fl_config):
        coord = FLCoordinator(minimal_fl_config, protocol="doh", model_dim=12, seed=42)
        coord.register_clients(_make_clients(5))
        coord._max_rounds = 3
        history = coord.run()
        assert len(history) <= 3

    def test_run_with_eval_fn(self, minimal_fl_config):
        coord = FLCoordinator(minimal_fl_config, protocol="doh", model_dim=12, seed=42)
        coord.register_clients(_make_clients(5))
        coord._max_rounds = 2
        def eval_fn(w):
            return float(np.mean(np.abs(w) < 10.0))
        history = coord.run(eval_fn=eval_fn)
        # Each round should record an accuracy.
        for record in history:
            assert record.global_accuracy is not None


# =============================================================================
# Determinism and reset
# =============================================================================

class TestDeterminism:
    def test_same_seed_same_history(self, minimal_fl_config):
        c1 = FLCoordinator(minimal_fl_config, protocol="doh", model_dim=12, seed=42)
        c2 = FLCoordinator(minimal_fl_config, protocol="doh", model_dim=12, seed=42)
        c1.register_clients(_make_clients(5, seed=7))
        c2.register_clients(_make_clients(5, seed=7))
        h1 = c1.run(n_rounds=3)
        h2 = c2.run(n_rounds=3)
        assert len(h1) == len(h2)
        for r1, r2 in zip(h1, h2):
            assert r1.n_clients_sampled == r2.n_clients_sampled
            assert r1.n_clients_included == r2.n_clients_included

    def test_reset(self, minimal_fl_config):
        coord = FLCoordinator(minimal_fl_config, protocol="doh", model_dim=12, seed=42)
        coord.register_clients(_make_clients(5))
        coord.run(n_rounds=2)
        coord.reset()
        assert len(coord.history) == 0
        np.testing.assert_array_equal(coord.global_params, np.zeros(12))


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
