#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tests/test_hardware_simulation.py
=================================

E3-DoH Reproducibility — unit tests for the two hardware emulators.

These tests verify the A100 and Loihi emulators' public contracts:

    - Construction with the `emulated` backend
    - `real` backend raises the documented exception
    - Benchmark produces a report with the expected keys
    - Emulated time = CPU time / calibration factor (invertible)
    - Convenience functions return dicts with the `*_emulated` tag

The tests use the smallest benchmark parameters that keep the runtime
under a second; the emulators' actual full benchmarks are exercised by
the table scripts, not by these unit tests.

Run with:
    pytest tests/test_hardware_simulation.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.hardware_simulation.a100_emulation import (
    A100_CALIBRATION_FACTOR,
    A100_SPEC,
    A100BackendNotAvailable,
    A100ConfigError,
    A100Emulator,
    benchmark_a100_emulation,
)
from src.hardware_simulation.loihi_mock import (
    LOIHI_CALIBRATION_FACTOR,
    LOIHI_SPEC,
    LoihiBackendNotAvailable,
    LoihiConfigError,
    LoihiEmulator,
    benchmark_loihi_emulation,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def emulated_config() -> dict:
    return {
        "hardware": {
            "a100": {"backend": "emulated"},
            "loihi": {"backend": "emulated"},
        },
        "runtime": {"seed": 42},
    }


@pytest.fixture
def a100_config() -> dict:
    return {"hardware": {"a100": {"backend": "emulated"}}, "runtime": {"seed": 42}}


@pytest.fixture
def loihi_config() -> dict:
    return {"hardware": {"loihi": {"backend": "emulated"}}, "runtime": {"seed": 42}}


# =============================================================================
# A100 emulator
# =============================================================================

class TestA100Emulator:
    def test_construct(self, a100_config):
        emu = A100Emulator(a100_config, seed=42)
        assert emu.calibration_factor == A100_CALIBRATION_FACTOR
        assert emu.parameters["polynomial_degree"] == 256

    def test_spec_reference(self, a100_config):
        emu = A100Emulator(a100_config, seed=42)
        # The thesis's A100 lattice throughput.
        assert emu.parameters["reference_a100_tops"] == 2.5

    def test_real_backend_raises(self):
        with pytest.raises(A100BackendNotAvailable):
            A100Emulator(
                {"hardware": {"a100": {"backend": "real"}}},
                seed=42,
            )

    def test_invalid_backend_raises(self):
        with pytest.raises(A100ConfigError):
            A100Emulator(
                {"hardware": {"a100": {"backend": "quantum"}}},
                seed=42,
            )

    def test_calibration_override(self):
        cfg = {
            "hardware": {"a100": {"backend": "emulated", "calibration_factor": 100.0}},
            "runtime": {"seed": 42},
        }
        emu = A100Emulator(cfg, seed=42)
        assert emu.calibration_factor == 100.0

    def test_benchmark_keys(self, a100_config):
        emu = A100Emulator(a100_config, seed=42)
        report = emu.benchmark(n_mul_repeats=2, n_window_repeats=1)
        d = report.to_dict()
        for key in (
            "measured_cpu_mul_time_us", "measured_cpu_window_time_us",
            "emulated_a100_mul_time_us", "emulated_a100_window_time_us",
            "calibration_factor", "reference_a100_tops", "hardware",
        ):
            assert key in d, f"missing key: {key}"
        assert d["hardware"] == "a100_emulated"

    def test_invertibility(self, a100_config):
        emu = A100Emulator(a100_config, seed=42)
        report = emu.benchmark(n_mul_repeats=2, n_window_repeats=1)
        inverted = report.emulated_a100_mul_time_us * report.calibration_factor
        assert abs(inverted - report.measured_cpu_mul_time_us) < 1e-6

    def test_convenience_function(self, a100_config):
        d = benchmark_a100_emulation(a100_config, seed=42)
        assert d["hardware"] == "a100_emulated"


# =============================================================================
# Loihi emulator
# =============================================================================

class TestLoihiEmulator:
    def test_construct(self, loihi_config):
        emu = LoihiEmulator(loihi_config, seed=42)
        assert emu.calibration_factor == LOIHI_CALIBRATION_FACTOR

    def test_spec_reference(self, loihi_config):
        emu = LoihiEmulator(loihi_config, seed=42)
        # Thesis reports 0.8 ms latency and 0.12 mJ energy.
        assert emu.parameters["loihi_spec"]["thesis_inference_latency_ms"] == 0.8
        assert emu.parameters["loihi_spec"]["thesis_energy_per_inference_mj"] == 0.12

    def test_real_backend_raises(self):
        with pytest.raises(LoihiBackendNotAvailable):
            LoihiEmulator(
                {"hardware": {"loihi": {"backend": "real"}}},
                seed=42,
            )

    def test_invalid_backend_raises(self):
        with pytest.raises(LoihiConfigError):
            LoihiEmulator(
                {"hardware": {"loihi": {"backend": "quantum"}}},
                seed=42,
            )

    def test_benchmark_keys(self, loihi_config):
        emu = LoihiEmulator(loihi_config, seed=42)
        report = emu.benchmark(n_repeats=1)
        d = report.to_dict()
        for key in (
            "measured_cpu_forward_time_us",
            "measured_cpu_forward_time_scaled_us",
            "emulated_loihi_forward_time_us",
            "calibration_factor",
            "reference_loihi_latency_ms",
            "reference_loihi_energy_mj",
            "spike_count",
            "n_neurons",
            "hardware",
        ):
            assert key in d, f"missing key: {key}"
        assert d["hardware"] == "loihi_emulated"

    def test_invertibility(self, loihi_config):
        emu = LoihiEmulator(loihi_config, seed=42)
        report = emu.benchmark(n_repeats=1)
        # emulated * factor / chip_scale = measured
        from src.hardware_simulation.loihi_mock import _CHIP_SCALE_FACTOR
        inverted = (
            report.emulated_loihi_forward_time_us
            * report.calibration_factor
            / _CHIP_SCALE_FACTOR
        )
        assert abs(inverted - report.measured_cpu_forward_time_us) < 1e-4

    def test_convenience_function(self, loihi_config):
        d = benchmark_loihi_emulation(loihi_config, seed=42)
        assert d["hardware"] == "loihi_emulated"


# =============================================================================
# Cross-emulator invariants
# =============================================================================

class TestCrossEmulator:
    def test_both_tags_distinct(self, a100_config, loihi_config):
        a = A100Emulator(a100_config, seed=42).benchmark(
            n_mul_repeats=1, n_window_repeats=1,
        )
        l = LoihiEmulator(loihi_config, seed=42).benchmark(n_repeats=1)
        assert a.to_dict()["hardware"] != l.to_dict()["hardware"]

    def test_calibration_factors_documented(self):
        # A100 = 250, Loihi = 650 (250 × 2.6 from thesis §4.24).
        assert A100_CALIBRATION_FACTOR == 250.0
        assert LOIHI_CALIBRATION_FACTOR == 650.0


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
