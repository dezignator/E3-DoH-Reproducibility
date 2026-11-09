#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tests/test_am_sandbox.py
========================

E3-DoH Reproducibility — unit tests for src/models/am_sandbox.py.

These tests verify the AMPS controller's public contract:

    - Construction and parameter loading
    - decide(): benign / attack distribution → CONTINUE / ISOLATE
    - Per-protocol threshold offsets
    - TTL expiry → RELEASE
    - apply_action: bookkeeping-only fallback (no adapter registered)
    - apply_action: adapter dispatch (mock adapter)
    - Reward computation on labelled decisions
    - Flow state transitions
    - Telemetry (mean decision latency)
    - reset()

The tests use a small HJB grid (or skip the HJB precompute) to keep
runtimes short.

Run with:
    pytest tests/test_am_sandbox.py -v
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.models.am_sandbox import (
    AMPSConfigError,
    AMPSController,
    AMPSFlowError,
    SandboxDecision,
    FlowSandboxState,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def am_config() -> dict:
    """
    A minimal AMPS config with a small HJB grid and a cache path that
    points to a temp directory.
    """
    import tempfile
    return {
        "_includes": {
            "sandbox_mdp": {
                "divergence": {
                    "primary": {"threshold": 0.30, "smoothing": 0.01},
                    "secondary": {"threshold": 0.45, "alpha": 2.0, "smoothing": 0.01},
                    "combination": "or",
                    "per_protocol_offsets": {
                        "doh": 1.00, "dot": 1.05, "doq": 0.95,
                    },
                },
                "sandbox": {"isolation_ttl_seconds": 30.0},
                "mdp": {
                    "inverse_temperature": 2.0,
                    "discount_factor": 0.95,
                    "reward": {
                        "security_weight": 1.0, "cost_weight": 0.3,
                        "correct_isolation": 1.0, "incorrect_isolation": -0.5,
                        "miss": -2.0, "correct_no_action": 0.1,
                    },
                },
                "hjb": {
                    "grid_points_per_dim": 8,  # small for speed
                    "time_step": 1e-3,
                    "max_iterations": 20,
                    "convergence_tolerance": 1e-2,
                    "boundary_conditions": {"low": "reflecting", "high": "absorbing"},
                },
                "implementation": {
                    "hjb_cache_path": "/tmp/e3doh_test_hjb.npz",
                },
            },
        },
        "runtime": {"seed": 42, "mode": "toy"},
        "_paths": {"repo_root": "/tmp"},
    }


@pytest.fixture
def benign_dist() -> np.ndarray:
    """A distribution with low entropy that the controller considers benign."""
    return np.array([0.3, 0.2, 0.2, 0.15, 0.05, 0.05, 0.02, 0.01, 0.01, 0.01])


@pytest.fixture
def attack_dist() -> np.ndarray:
    """A distribution far from `benign_dist` (the trigger fires)."""
    return np.array([0.01, 0.01, 0.01, 0.01, 0.05, 0.05, 0.1, 0.1, 0.3, 0.36])


# =============================================================================
# Construction
# =============================================================================

class TestConstruction:
    def test_construct(self, am_config):
        ctrl = AMPSController(am_config, seed=42, precompute_hjb=False)
        assert ctrl.parameters["kl_threshold"] == 0.30
        assert ctrl.parameters["renyi_threshold"] == 0.45
        assert ctrl.parameters["ttl_seconds"] == 30.0

    def test_missing_config_raises(self):
        with pytest.raises(AMPSConfigError):
            AMPSController({}, seed=42, precompute_hjb=False)

    def test_per_protocol_offsets_recorded(self, am_config):
        ctrl = AMPSController(am_config, seed=42, precompute_hjb=False)
        offsets = ctrl.parameters["per_protocol_offsets"]
        assert offsets["dot"] == 1.05
        assert offsets["doq"] == 0.95


# =============================================================================
# Decisions
# =============================================================================

class TestDecisions:
    def test_benign_continues(self, am_config, benign_dist):
        ctrl = AMPSController(am_config, seed=42, precompute_hjb=False)
        rec = ctrl.decide("flow-1", "doh", benign_dist, benign_dist)
        assert rec.decision == SandboxDecision.CONTINUE
        assert rec.kl_divergence < 0.1
        assert rec.triggered_by == "neither"

    def test_attack_isolates(self, am_config, benign_dist, attack_dist):
        ctrl = AMPSController(am_config, seed=42, precompute_hjb=False)
        rec = ctrl.decide("flow-2", "doh", attack_dist, benign_dist)
        assert rec.decision == SandboxDecision.ISOLATE
        assert rec.triggered_by in ("kl", "renyi", "both")

    def test_unknown_protocol_rejected(self, am_config, benign_dist):
        ctrl = AMPSController(am_config, seed=42, precompute_hjb=False)
        with pytest.raises(AMPSConfigError):
            ctrl.decide("flow-1", "foo", benign_dist, benign_dist)

    def test_shape_mismatch_rejected(self, am_config, benign_dist):
        ctrl = AMPSController(am_config, seed=42, precompute_hjb=False)
        with pytest.raises(Exception):  # AMPSError
            ctrl.decide("flow-1", "doh", benign_dist[:5], benign_dist)

    def test_reward_on_benign_continue(self, am_config, benign_dist):
        ctrl = AMPSController(am_config, seed=42, precompute_hjb=False)
        rec = ctrl.decide(
            "flow-1", "doh", benign_dist, benign_dist, true_label="benign",
        )
        # correct_no_action = 0.1
        assert rec.reward == pytest.approx(0.1)

    def test_reward_on_attack_isolate(self, am_config, benign_dist, attack_dist):
        ctrl = AMPSController(am_config, seed=42, precompute_hjb=False)
        rec = ctrl.decide(
            "flow-2", "doh", attack_dist, benign_dist, true_label="attack",
        )
        # correct_isolation = 1.0
        assert rec.reward == pytest.approx(1.0)

    def test_reward_on_missed_attack(self, am_config, benign_dist, attack_dist):
        ctrl = AMPSController(am_config, seed=42, precompute_hjb=False)
        rec = ctrl.decide(
            "flow-3", "doh", benign_dist, benign_dist, true_label="attack",
        )
        # No trigger fired; decision is CONTINUE for an attack → miss = -2.0
        assert rec.reward == pytest.approx(-2.0)


# =============================================================================
# Per-protocol thresholds
# =============================================================================

class TestPerProtocolThresholds:
    def test_dot_threshold_higher(self, am_config, benign_dist):
        ctrl = AMPSController(am_config, seed=42, precompute_hjb=False)
        rec = ctrl.decide("f", "dot", benign_dist, benign_dist)
        # DoT kl_threshold = 0.30 * 1.05 = 0.315
        assert rec.kl_threshold == pytest.approx(0.315, abs=1e-6)

    def test_doq_threshold_lower(self, am_config, benign_dist):
        ctrl = AMPSController(am_config, seed=42, precompute_hjb=False)
        rec = ctrl.decide("f", "doq", benign_dist, benign_dist)
        # DoQ kl_threshold = 0.30 * 0.95 = 0.285
        assert rec.kl_threshold == pytest.approx(0.285, abs=1e-6)


# =============================================================================
# TTL and release
# =============================================================================

class TestTTL:
    def test_ttl_expires_release(self, am_config, benign_dist, attack_dist):
        ctrl = AMPSController(am_config, seed=42, precompute_hjb=False)
        # Isolate.
        rec = ctrl.decide("flow-1", "doh", attack_dist, benign_dist)
        assert rec.decision == SandboxDecision.ISOLATE
        # Advance time past the TTL.
        future = time.perf_counter() + 100.0
        rec2 = ctrl.decide(
            "flow-1", "doh", attack_dist, benign_dist, now_s=future,
        )
        assert rec2.decision == SandboxDecision.RELEASE
        assert rec2.triggered_by == "ttl"

    def test_anomaly_resolved_release(self, am_config, benign_dist, attack_dist):
        ctrl = AMPSController(am_config, seed=42, precompute_hjb=False)
        # Isolate.
        ctrl.decide("flow-1", "doh", attack_dist, benign_dist)
        # Now benign again.
        rec = ctrl.decide("flow-1", "doh", benign_dist, benign_dist)
        assert rec.decision == SandboxDecision.RELEASE


# =============================================================================
# apply_action
# =============================================================================

class TestApplyAction:
    def test_apply_isolate_bookkeeping(self, am_config, benign_dist, attack_dist):
        ctrl = AMPSController(am_config, seed=42, precompute_hjb=False)
        rec = ctrl.decide("flow-1", "doh", attack_dist, benign_dist)
        result = ctrl.apply_action("flow-1", rec.decision)
        assert result["success"] is True
        assert result["action"] == "isolate_doh_stream"

    def test_apply_continue_no_action(self, am_config, benign_dist):
        ctrl = AMPSController(am_config, seed=42, precompute_hjb=False)
        rec = ctrl.decide("flow-1", "doh", benign_dist, benign_dist)
        result = ctrl.apply_action("flow-1", rec.decision)
        assert result["action"] is None
        assert result["success"] is True

    def test_apply_unknown_flow_raises(self, am_config):
        ctrl = AMPSController(am_config, seed=42, precompute_hjb=False)
        with pytest.raises(AMPSFlowError):
            ctrl.apply_action("nonexistent", SandboxDecision.ISOLATE)

    def test_apply_with_registered_adapter(self, am_config, benign_dist, attack_dist):
        ctrl = AMPSController(am_config, seed=42, precompute_hjb=False)

        # A mock adapter that records the calls.
        calls = []
        class MockAdapter:
            def apply_sandbox_action(self, flow, action):
                calls.append(action)
                return SimpleNamespace(
                    success=True, action=action, latency_ms=1.2,
                    detail="migrated",
                )

        ctrl.register_adapter("doq", MockAdapter())
        rec = ctrl.decide("flow-d", "doq", attack_dist, benign_dist)
        assert rec.decision == SandboxDecision.ISOLATE
        mock_flow = SimpleNamespace(
            flow_id="flow-d", protocol="doq", protocol_data={},
        )
        result = ctrl.apply_action("flow-d", rec.decision, flow=mock_flow)
        assert result["success"] is True
        assert result["action"] == "migrate_doq_connection"
        assert calls == ["migrate_doq_connection"]


# =============================================================================
# Flow state
# =============================================================================

class TestFlowState:
    def test_isolated_after_decide(self, am_config, benign_dist, attack_dist):
        ctrl = AMPSController(am_config, seed=42, precompute_hjb=False)
        ctrl.decide("flow-1", "doh", attack_dist, benign_dist)
        state = ctrl.flow_state("flow-1")
        assert state.state == FlowSandboxState.ISOLATED
        assert state.action_taken == "isolate_doh_stream"

    def test_observing_after_release(self, am_config, benign_dist, attack_dist):
        ctrl = AMPSController(am_config, seed=42, precompute_hjb=False)
        ctrl.decide("flow-1", "doh", attack_dist, benign_dist)
        # Advance past TTL, then decide → release.
        future = time.perf_counter() + 100.0
        ctrl.decide("flow-1", "doh", attack_dist, benign_dist, now_s=future)
        state = ctrl.flow_state("flow-1")
        assert state.state == FlowSandboxState.OBSERVING

    def test_unknown_flow_raises(self, am_config):
        ctrl = AMPSController(am_config, seed=42, precompute_hjb=False)
        with pytest.raises(AMPSFlowError):
            ctrl.flow_state("nonexistent")


# =============================================================================
# Telemetry and reset
# =============================================================================

class TestTelemetry:
    def test_mean_latency_positive(self, am_config, benign_dist):
        ctrl = AMPSController(am_config, seed=42, precompute_hjb=False)
        for i in range(5):
            ctrl.decide(f"flow-{i}", "doh", benign_dist, benign_dist)
        assert ctrl.mean_decision_latency_us > 0

    def test_reset(self, am_config, benign_dist):
        ctrl = AMPSController(am_config, seed=42, precompute_hjb=False)
        ctrl.decide("flow-1", "doh", benign_dist, benign_dist)
        assert len(ctrl.history) == 1
        ctrl.reset()
        assert len(ctrl.history) == 0
        assert len(ctrl._flows) == 0


# =============================================================================
# HJB precomputation (small grid)
# =============================================================================

class TestHJB:
    @pytest.mark.slow
    def test_hjb_precomputed(self, am_config):
        ctrl = AMPSController(am_config, seed=42, precompute_hjb=True)
        assert ctrl._hjb_solution is not None
        assert ctrl._hjb_solution.shape == (8, 8)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
