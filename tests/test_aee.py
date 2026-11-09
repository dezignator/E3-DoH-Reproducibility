#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tests/test_aee.py
=================

E3-DoH Reproducibility — unit tests for src/models/aee_engine.py.

These tests verify the AEE engine's two layers independently:

    1. The Module-LWE KEM (keygen / encapsulate / decapsulate).
    2. The SDE-driven key-evolution schedule with the modulation policy.

They also verify the security boundary is *respected*: no test
asserts a security property of the schedule itself; the schedule's
tests are about determinism and the constraint α²+β²+γ²=1, not about
unpredictability.

The tests use a small lattice (n=16) for speed. The production engine
uses n=1024, which would make the O(n²) polynomial multiplication in
the KEM too slow for a unit test.

Run with:
    pytest tests/test_aee.py -v
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.models.aee_engine import (
    AEEConfigError,
    AEEEngine,
    AEEPolynomialError,
    AEEState,
    AEEStateError,
    _hkdf_sha256,
    _poly_mul,
)


# =============================================================================
# Shared fixtures
# =============================================================================

def _make_config(n: int = 16, q: int = 12289, k: int = 2) -> Dict[str, Any]:
    """
    A minimal AEE config with a small lattice. q=12289 is prime and
    2n | q-1, so NTT-based operations would be possible; the engine
    uses direct polynomial multiplication regardless, but the values
    are valid.
    """
    return {
        "_includes": {
            "aee": {
                "lattice": {
                    "ring": {"n": n, "q_baseline": q},
                    "module": {"k_baseline": k, "k_min": 2, "k_max": 4},
                    "distribution": {"sigma_err": 3.19},
                },
                "sde": {
                    "alpha_init": 0.6, "beta_init": 0.6, "gamma_init": 0.529,
                    "wiener": {"dt": 1e-3},
                    "diffusion": {"floor": 1e-3, "ceiling": 1.0},
                    "default_rotation_seconds": 10.0,
                },
                "modulation": {
                    "enabled": True,
                    "entropy_safe": 0.55, "entropy_trigger": 0.85,
                    "downgrade_patience_windows": 10,
                    "steps": [
                        {"name": "baseline", "k": k, "q_bits": 14, "sigma_err": 3.19},
                        {"name": "elevated", "k": k + 1, "q_bits": 15, "sigma_err": 3.19},
                        {"name": "high",     "k": k + 2, "q_bits": 16, "sigma_err": 2.83},
                    ],
                },
                "per_protocol": {
                    "doh": {"parameter_update_time_us": 42.0, "tolerance_us": 1.0},
                    "dot": {"parameter_update_time_us": 38.0, "tolerance_us": 1.0},
                    "doq": {"parameter_update_time_us": 45.0, "tolerance_us": 1.0},
                },
                "key_derivation": {
                    "quantisation": {
                        "output_dim": 8, "bits_per_coordinate": 16,
                        "projection_seed_offset": 1,
                    },
                    "kdf": {
                        "family": "hkdf", "hash": "sha256",
                        "output_bits": 256, "info": "e3doh-aee-session-key-v1",
                    },
                },
            },
        },
        "runtime": {"seed": 42},
    }


# =============================================================================
# Polynomial helpers
# =============================================================================

class TestPolynomialArithmetic:
    def test_poly_mul_identity(self):
        # Multiply by the polynomial [1, 0, 0, ...] — should be identity.
        n = 8
        q = 12289
        a = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.int64)
        one = np.zeros(n, dtype=np.int64); one[0] = 1
        result = _poly_mul(a, one, q)
        np.testing.assert_array_equal(result, a % q)

    def test_poly_mul_x_reduction(self):
        # x * x^(n-1) = x^n ≡ -1 (mod x^n + 1).
        n = 8
        q = 12289
        x_poly = np.zeros(n, dtype=np.int64); x_poly[1] = 1
        x_n_minus_1 = np.zeros(n, dtype=np.int64); x_n_minus_1[n - 1] = 1
        result = _poly_mul(x_poly, x_n_minus_1, q)
        # Result should be -1 mod q at position 0, zero elsewhere.
        expected = np.zeros(n, dtype=np.int64)
        expected[0] = (-1) % q
        np.testing.assert_array_equal(result, expected)

    def test_poly_mul_length_mismatch(self):
        with pytest.raises(AEEPolynomialError):
            _poly_mul(np.zeros(8, dtype=np.int64), np.zeros(4, dtype=np.int64), 12289)


class TestHKDF:
    def test_deterministic(self):
        k1 = _hkdf_sha256(b"input", salt=b"salt", info=b"info", length=32)
        k2 = _hkdf_sha256(b"input", salt=b"salt", info=b"info", length=32)
        assert k1 == k2

    def test_length(self):
        k = _hkdf_sha256(b"input", salt=b"salt", info=b"info", length=32)
        assert len(k) == 32

    def test_different_inputs(self):
        k1 = _hkdf_sha256(b"input-A", salt=b"salt", info=b"info", length=32)
        k2 = _hkdf_sha256(b"input-B", salt=b"salt", info=b"info", length=32)
        assert k1 != k2

    def test_different_info(self):
        k1 = _hkdf_sha256(b"input", salt=b"salt", info=b"info-A", length=32)
        k2 = _hkdf_sha256(b"input", salt=b"salt", info=b"info-B", length=32)
        assert k1 != k2


# =============================================================================
# Construction and config
# =============================================================================

class TestConstruction:
    def test_construct(self):
        engine = AEEEngine(_make_config(), protocol="doh", seed=42)
        assert engine.protocol == "doh"

    def test_missing_config_raises(self):
        with pytest.raises(AEEConfigError):
            AEEEngine({}, protocol="doh")

    def test_unknown_protocol_rejected(self):
        with pytest.raises(AEEConfigError):
            AEEEngine(_make_config(), protocol="foo")

    def test_parameters_expose_lattice(self):
        engine = AEEEngine(_make_config(), protocol="doh", seed=42)
        p = engine.parameters
        assert p["n"] == 16
        assert p["q"] == 12289
        assert p["k"] == 2


# =============================================================================
# KEM
# =============================================================================

class TestKEM:
    def test_round_trip(self):
        engine = AEEEngine(_make_config(), protocol="doh", seed=42)
        pk, sk = engine.keygen()
        message = b"0123456789abcdef0123456789abcdef"  # 32 bytes
        ct, ss_encap = engine.encapsulate(pk, message)
        ss_decap = engine.decapsulate(sk, ct)
        assert ss_encap == ss_decap

    def test_wrong_key_gives_different_secret(self):
        engine = AEEEngine(_make_config(), protocol="doh", seed=42)
        pk, _sk = engine.keygen()
        _pk2, sk2 = engine.keygen()
        message = b"0123456789abcdef0123456789abcdef"
        ct, ss_encap = engine.encapsulate(pk, message)
        ss_wrong = engine.decapsulate(sk2, ct)
        assert ss_wrong != ss_encap

    def test_shared_secret_length(self):
        engine = AEEEngine(_make_config(), protocol="doh", seed=42)
        pk, _sk = engine.keygen()
        _ct, ss = engine.encapsulate(pk, b"x" * 32)
        assert len(ss) == 32  # SHA-256

    def test_message_too_long_rejected(self):
        engine = AEEEngine(_make_config(), protocol="doh", seed=42)
        pk, _sk = engine.keygen()
        # n=16 bits = 2 bytes maximum.
        with pytest.raises(AEEPolynomialError):
            engine.encapsulate(pk, b"x" * 10)


# =============================================================================
# SDE
# =============================================================================

class TestSDE:
    def test_evolution_count_increments(self):
        engine = AEEEngine(_make_config(), protocol="doh", seed=42)
        assert engine.state.evolution_count == 0
        engine.evolve_step(entropy=0.5, threat_score=0.3)
        assert engine.state.evolution_count == 1
        engine.evolve_step(entropy=0.5, threat_score=0.3)
        assert engine.state.evolution_count == 2

    def test_key_vector_changes(self):
        engine = AEEEngine(_make_config(), protocol="doh", seed=42)
        before = engine.state.key_vector.copy()
        engine.evolve_step(entropy=0.5, threat_score=0.3)
        after = engine.state.key_vector
        assert not np.array_equal(before, after)

    def test_simplex_constraint_preserved(self):
        engine = AEEEngine(_make_config(), protocol="doh", seed=42)
        for _ in range(5):
            engine.evolve_step(entropy=0.5, threat_score=0.3)
            s = engine.state
            norm_sq = s.alpha ** 2 + s.beta ** 2 + s.gamma ** 2
            assert abs(norm_sq - 1.0) < 1e-6

    def test_entropy_out_of_range_rejected(self):
        engine = AEEEngine(_make_config(), protocol="doh", seed=42)
        with pytest.raises(AEEStateError):
            engine.evolve_step(entropy=1.5, threat_score=0.3)

    def test_threat_out_of_range_rejected(self):
        engine = AEEEngine(_make_config(), protocol="doh", seed=42)
        with pytest.raises(AEEStateError):
            engine.evolve_step(entropy=0.5, threat_score=-0.1)


# =============================================================================
# Modulation policy
# =============================================================================

class TestModulation:
    def test_escalates_on_trigger(self):
        engine = AEEEngine(_make_config(), protocol="doh", seed=42)
        assert engine.state.modulation_step == "baseline"
        engine.evolve_step(entropy=0.95, threat_score=0.9)
        assert engine.state.modulation_step == "elevated"

    def test_escalates_twice(self):
        engine = AEEEngine(_make_config(), protocol="doh", seed=42)
        engine.evolve_step(entropy=0.95, threat_score=0.9)
        engine.evolve_step(entropy=0.95, threat_score=0.9)
        assert engine.state.modulation_step == "high"

    def test_does_not_escalate_past_max(self):
        engine = AEEEngine(_make_config(), protocol="doh", seed=42)
        for _ in range(5):
            engine.evolve_step(entropy=0.95, threat_score=0.9)
        assert engine.state.modulation_step == "high"

    def test_downgrade_after_patience(self):
        engine = AEEEngine(_make_config(), protocol="doh", seed=42)
        # Escalate to high.
        engine.evolve_step(entropy=0.95, threat_score=0.9)
        engine.evolve_step(entropy=0.95, threat_score=0.9)
        assert engine.state.modulation_step == "high"
        # Now safe for enough windows to downgrade once.
        for _ in range(11):  # patience is 10, +1 to trigger
            engine.evolve_step(entropy=0.1, threat_score=0.1)
        assert engine.state.modulation_step == "elevated"

    def test_modulation_disabled_keeps_baseline(self):
        cfg = _make_config()
        cfg["_includes"]["aee"]["modulation"]["enabled"] = False
        engine = AEEEngine(cfg, protocol="doh", seed=42)
        for _ in range(5):
            engine.evolve_step(entropy=0.95, threat_score=0.9)
        assert engine.state.modulation_step == "baseline"


# =============================================================================
# Session key derivation
# =============================================================================

class TestSessionKey:
    def test_length(self):
        engine = AEEEngine(_make_config(), protocol="doh", seed=42)
        key = engine.derive_session_key(b"\x00" * 32)
        assert len(key) == 32

    def test_deterministic(self):
        engine = AEEEngine(_make_config(), protocol="doh", seed=42)
        k1 = engine.derive_session_key(b"\xab" * 32)
        k2 = engine.derive_session_key(b"\xab" * 32)
        assert k1 == k2

    def test_changes_with_state(self):
        engine = AEEEngine(_make_config(), protocol="doh", seed=42)
        k1 = engine.derive_session_key(b"\xab" * 32)
        engine.evolve_step(entropy=0.5, threat_score=0.5)
        k2 = engine.derive_session_key(b"\xab" * 32)
        assert k1 != k2

    def test_changes_with_shared_secret(self):
        engine = AEEEngine(_make_config(), protocol="doh", seed=42)
        k1 = engine.derive_session_key(b"\xab" * 32)
        k2 = engine.derive_session_key(b"\xcd" * 32)
        assert k1 != k2


# =============================================================================
# Determinism and serialization
# =============================================================================

class TestDeterminism:
    def test_same_seed_same_state(self):
        e1 = AEEEngine(_make_config(), protocol="doh", seed=42)
        e2 = AEEEngine(_make_config(), protocol="doh", seed=42)
        for _ in range(3):
            e1.evolve_step(entropy=0.5, threat_score=0.3)
            e2.evolve_step(entropy=0.5, threat_score=0.3)
        np.testing.assert_allclose(e1.state.key_vector, e2.state.key_vector)

    def test_different_seed_different_state(self):
        e1 = AEEEngine(_make_config(), protocol="doh", seed=42)
        e2 = AEEEngine(_make_config(), protocol="doh", seed=999)
        for _ in range(3):
            e1.evolve_step(entropy=0.5, threat_score=0.3)
            e2.evolve_step(entropy=0.5, threat_score=0.3)
        assert not np.allclose(e1.state.key_vector, e2.state.key_vector)


class TestSerialization:
    def test_round_trip(self):
        e1 = AEEEngine(_make_config(), protocol="doh", seed=42)
        for _ in range(3):
            e1.evolve_step(entropy=0.5, threat_score=0.3)
        payload = e1.to_dict()
        e2 = AEEEngine.from_dict(_make_config(), payload)
        np.testing.assert_allclose(e2.state.key_vector, e1.state.key_vector)
        assert e2.state.modulation_step == e1.state.modulation_step


# =============================================================================
# Timing
# =============================================================================

class TestTiming:
    def test_update_time_positive(self):
        engine = AEEEngine(_make_config(), protocol="doh", seed=42)
        t = engine.measure_update_time_us(n_repeats=5, n_warmup=2)
        assert 0 < t < 1e6  # less than 1 second per step

    def test_effective_time_includes_overhead(self):
        engine = AEEEngine(_make_config(), protocol="doh", seed=42)
        raw = engine.measure_update_time_us(n_repeats=5, n_warmup=2)
        effective = engine.effective_update_time_us(n_repeats=5)
        assert effective >= raw  # per-protocol overhead is positive

    def test_per_protocol_overhead_dot_lowest(self):
        # The DoT transport overhead is the lowest of the three (thesis §5.4.4).
        e_doh = AEEEngine(_make_config(), protocol="doh", seed=42)
        e_dot = AEEEngine(_make_config(), protocol="dot", seed=42)
        e_doq = AEEEngine(_make_config(), protocol="doq", seed=42)
        assert e_dot._transport_overhead_us < e_doh._transport_overhead_us
        assert e_doq._transport_overhead_us > e_doh._transport_overhead_us


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
