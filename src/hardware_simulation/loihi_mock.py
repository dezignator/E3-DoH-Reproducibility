#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
src/hardware_simulation/loihi_mock.py
=====================================

E3-DoH Reproducibility — Intel Loihi 2 emulation for the SNN layer.

Chapter 5 of the thesis: E3-DoH (Enhanced Evolutionary Encryption for
Multi-Protocol DNS Security — DoH, DoT, DoQ).

Thesis §4.24 (p.106) reports that the neuromorphic implementation on
Intel Loihi achieved 0.8 ms inference latency and 0.12 mJ per
inference — 2.6× faster and 15× more energy-efficient than the A100
GPU pipeline. The E3-DoH framework inherits the Loihi-based SNN model
from PAFA-DoH as the FL client's local anomaly detector.

This module emulates the Loihi 2 LIF-neuron dynamics on the local
CPU, mirroring the structure of `a100_emulation.py`:

    1. Measures the *real* CPU time to simulate the SNN forward pass.
    2. Applies a documented calibration factor to produce the
       *emulated Loihi time*.
    3. Emits both numbers, tagged as `hardware="loihi_emulated"`.

WHAT THE EMULATOR SIMULATES
---------------------------
The Loihi 2 chip has 128 neuromorphic cores, each with 1024 LIF
neurons and ~1M synapses. A full forward pass at 128K neurons is
computationally heavy for a Python emulator, so the emulator uses a
*documented proportional reduction*: it simulates a smaller network
(e.g., 3,072 neurons = 3 virtual cores) and scales the measured time
linearly to the 128K-neuron target. This is a modelling choice, not
a bit-accurate simulation; the calibration factor is documented as an
order-of-magnitude estimate.

The LIF dynamics are the standard leaky integrate-and-fire model of
thesis Eq. 44 (p.95):

    τ_m dV/dt = -(V - V_rest) + I_syn(t) + ξ(t)

with the threshold-and-reset rule of Eq. 45:

    if V >= V_th: spike, V <- V_reset, refractory period τ_ref

The emulator implements this with numpy vectorization. The measured
CPU time is the wall-clock time of a forward pass (K timesteps ×
N neurons).

WHY AN EMULATOR, NOT THE REAL CHIP
----------------------------------
    - Intel Loihi 2 is not available in most research or evaluation
      environments (it is a research-only platform, accessed through
      the cloud or via the Intel Neuromorphic Research Community).
    - Shipping an emulator lets a reviewer exercise the E3-DoH FL
      client's SNN path on any machine, at the documented cost that
      the emulated time is a calibrated CPU measurement, not the
      real chip's time.
    - A reviewer with Loihi access can run the same script on the
      chip by setting `hardware.loihi.backend: "real"` in the config;
      the emulator raises `LoihiBackendNotAvailable` (see the
      docstring on the `real` backend path).

THE CALIBRATION FACTOR
----------------------
The factor is derived from thesis §4.24 (p.106) and the A100
calibration:

    Loihi_speedup_vs_A100 (thesis) :  2.6×
    A100_speedup_vs_CPU (emulator) :  250×
    Loihi_speedup_vs_CPU (derived) :  650×

The 650× factor is the emulator's conversion from CPU time to
emulated Loihi time. It is an order-of-magnitude estimate; the
multiplied factors compound the calibration uncertainty of both
emulators. A reviewer who wants the raw CPU time can invert the
factor via `measured_cpu_time_us = emulated_loihi_time_us * 650`.

A reviewer with their own calibration should override the factor in
`hardware.loihi.calibration_factor`; the emulator uses the override
if present.

ENERGY ESTIMATE
---------------
The emulator also reports an *estimated energy* per inference, based
on the spike count. The Loihi's energy advantage comes from
event-driven computation: only the neurons that spike consume
energy. The estimate is:

    E = (n_spikes) * E_per_spike

where E_per_spike is a documented constant (~1 nJ per spike-synapse
operation, from the Loihi 2 datasheet; this is a rough figure that
varies with routing topology). The estimate is provided as a
*qualitative* indication of the energy advantage, not as a precision
number. A reviewer who needs an energy benchmark should run the
real Loihi chip.

WHAT THE EMULATOR DOES NOT DO
-----------------------------
    - It does not run on the Loihi chip (the `real` backend raises).
    - It does not reproduce the chip's bit-exact spike sequences (the
      LIF dynamics are float64 here; the chip uses fixed-point).
    - It does not model the chip's routing topology (all-to-all in
      the emulator vs the chip's sparse routing). The energy estimate
      is proportional to the spike count, not to the actual routed
      synapse events.
    - It does not claim that the emulated latency is what the thesis
      reports. The thesis reports 0.8 ms for the real chip. The
      emulator's emulated time is a calibrated CPU estimate; a
      reviewer who wants the reported 0.8 ms should read the
      reference JSON, not the emulator output.

DETERMINISM
-----------
The emulator is deterministic in the sense that the *operation count*
is fixed and the calibration factor is a constant. The *measured CPU
time* is wall-clock and varies by the usual ~5% for a warm loop; the
emulator uses enough repetitions that the mean is stable to ~2%. The
*spike count* is deterministic given the seed (the emulator uses a
seeded RNG for the input spikes).

THREAD SAFETY
-------------
An `LoihiEmulator` instance is stateless apart from the config and a
small seeded RNG. It is thread-safe for read-only queries; a caller
that wants concurrent simulations should create one instance per
thread.

Equation references: thesis §4.24 (p.106) for the Loihi spec,
§4.9-4.11 (p.94-97) for the LIF dynamics (Eqs. 43-46). This module
implements the same LIF dynamics as `PAFA-DoH`'s SNN.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.utils.logger import get_logger
from src.utils.seed import get_component_seed


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Intel Loihi 2 spec (from thesis §4.24, p.106, and the vendor datasheet).
LOIHI_SPEC: Dict[str, Any] = {
    "name": "Intel Loihi 2",
    "cores": 128,
    "neurons_per_core": 1024,
    "synapses_per_core": 1_000_000,
    "total_neurons": 128 * 1024,
    "total_synapses": 128 * 1_000_000,
    # Thesis §4.24, p.106 — the reported figures vs A100.
    "thesis_inference_latency_ms": 0.8,
    "thesis_energy_per_inference_mj": 0.12,
    "thesis_speedup_vs_a100": 2.6,
    "thesis_energy_advantage_vs_a100": 15.0,
}

# The calibration factor.
#
#   factor = (Loihi_speedup_vs_A100) * (A100_speedup_vs_CPU)
#          = 2.6 * 250
#          = 650
#
# Both inputs are documented (thesis §4.24 for the 2.6, and
# a100_emulation.py for the 250). The product is an order-of-magnitude
# estimate; the calibration uncertainty of the two emulators compounds
# multiplicatively. See the module docstring's "The calibration factor"
# section.
LOIHI_CALIBRATION_FACTOR: float = 650.0

# Energy per spike (Loihi 2 datasheet, rough figure). The emulator
# uses this to convert spike counts into an estimated energy. It is a
# qualitative figure, not a precision number.
ENERGY_PER_SPIKE_NJ: float = 1.0

# LIF neuron parameters (thesis Eqs. 44-45, p.95).
_LIF_TAU_M_MS = 20.0            # membrane time constant
_LIF_V_REST = 0.0               # resting potential
_LIF_V_THRESHOLD = 1.0          # firing threshold
_LIF_V_RESET = 0.0              # post-spike reset
_LIF_TAU_REF_MS = 2.0           # absolute refractory period
_LIF_DT_MS = 1.0                # simulation timestep
_LIF_NOISE_STD = 0.05           # ξ(t) std

# Network topology for the emulator. The real chip has 128K neurons;
# the emulator uses 3,072 (3 virtual cores) and scales linearly. The
# ratio 128000/3072 ≈ 41.67 is applied to the measured CPU time
# (documented as a proportional scaling).
_N_NEURONS_EMULATED = 3072
_N_INPUT_NEURONS = 64
_N_HIDDEN_NEURONS = _N_NEURONS_EMULATED - _N_INPUT_NEURONS - 16
_N_OUTPUT_NEURONS = 16
_N_TIMESTEPS = 50

# The scaling from emulated-neuron-count time to full-chip time.
# 128,000 / 3,072 ≈ 41.67. Applied to the CPU measurement before the
# calibration factor.
_CHIP_SCALE_FACTOR: float = LOIHI_SPEC["total_neurons"] / _N_NEURONS_EMULATED

# Benchmark parameters.
_BENCHMARK_WARMUP = 3
_BENCHMARK_REPEATS = 10


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class LoihiEmulationError(Exception):
    """Base class for Loihi-emulation errors."""


class LoihiConfigError(LoihiEmulationError):
    """Raised when the config is missing or has invalid values."""


class LoihiBackendNotAvailable(LoihiEmulationError):
    """Raised when the config requests the `real` backend, which the
    artifact does not ship (requires the Intel NRC SDK)."""


# ---------------------------------------------------------------------------
# Benchmark report
# ---------------------------------------------------------------------------

@dataclass
class LoihiBenchmarkReport:
    """
    The result of one benchmark run. Same structure as the A100 report,
    with Loihi-specific fields.
    """
    measured_cpu_forward_time_us: float
        # Mean wall-clock time of one forward pass on the local CPU,
        # for the reduced (3,072-neuron) network.
    measured_cpu_forward_time_scaled_us: float
        # `measured_cpu_forward_time_us * _CHIP_SCALE_FACTOR` — the
        # extrapolated time for the full 128K-neuron network.
    emulated_loihi_forward_time_us: float
        # `measured_cpu_forward_time_scaled_us / calibration_factor`.

    measured_cpu_energy_estimate_mj: float
        # Estimated energy on CPU: (spike_count * energy_per_spike_nj) * chip_scale.
    emulated_loihi_energy_estimate_mj: float
        # Estimated energy on Loihi: (spike_count * energy_per_spike_nj),
        # plus the fixed per-inference overhead documented in
        # thesis §4.24 (0.12 mJ). This is a *qualitative* estimate;
        # see the module docstring.

    calibration_factor: float
    reference_loihi_latency_ms: float
    reference_loihi_energy_mj: float
    spike_count: int
    n_neurons: int
    n_timesteps: int
    cpu_count: int
    platform: str
    hardware: str = "loihi_emulated"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "measured_cpu_forward_time_us": round(self.measured_cpu_forward_time_us, 4),
            "measured_cpu_forward_time_scaled_us": round(self.measured_cpu_forward_time_scaled_us, 4),
            "emulated_loihi_forward_time_us": round(self.emulated_loihi_forward_time_us, 4),
            "measured_cpu_energy_estimate_mj": round(self.measured_cpu_energy_estimate_mj, 6),
            "emulated_loihi_energy_estimate_mj": round(self.emulated_loihi_energy_estimate_mj, 6),
            "calibration_factor": self.calibration_factor,
            "reference_loihi_latency_ms": self.reference_loihi_latency_ms,
            "reference_loihi_energy_mj": self.reference_loihi_energy_mj,
            "spike_count": self.spike_count,
            "n_neurons": self.n_neurons,
            "n_timesteps": self.n_timesteps,
            "cpu_count": self.cpu_count,
            "platform": self.platform,
            "hardware": self.hardware,
        }


# ---------------------------------------------------------------------------
# LoihiEmulator
# ---------------------------------------------------------------------------

class LoihiEmulator:
    """
    Emulates the Intel Loihi 2 LIF-neuron dynamics on the local CPU.

    See the module docstring for what "emulates" means here, and for
    the calibration factor's derivation.

    Parameters
    ----------
    config : dict
        The dict returned by data_utils.load_config().
    calibration_factor : optional float
        Overrides the module-level `LOIHI_CALIBRATION_FACTOR`.
    seed : optional int
        Seed for the input spike generation.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        *,
        calibration_factor: Optional[float] = None,
        seed: Optional[int] = None,
    ) -> None:
        self._config = config
        self._log = get_logger("e3doh.hardware.loihi")

        hw_cfg = config.get("hardware", {}) if isinstance(config, dict) else {}
        loihi_cfg = hw_cfg.get("loihi", {})
        self._backend = str(loihi_cfg.get("backend", "emulated"))
        if self._backend not in ("emulated", "real"):
            raise LoihiConfigError(
                f"hardware.loihi.backend must be 'emulated' or 'real'; "
                f"got {self._backend!r}"
            )
        if self._backend == "real":
            raise LoihiBackendNotAvailable(
                "The `real` backend requires the Intel Loihi 2 hardware "
                "and the Intel NRC SDK, which the artifact does not ship. "
                "A reviewer with Loihi access should run the SNN forward "
                "pass on the chip directly and record the result in "
                "expected_outputs/. For a CPU-only reproduction, use "
                "the `emulated` backend (the default)."
            )

        configured = loihi_cfg.get("calibration_factor")
        if calibration_factor is not None:
            self._calibration_factor = float(calibration_factor)
        elif configured is not None:
            self._calibration_factor = float(configured)
        else:
            self._calibration_factor = LOIHI_CALIBRATION_FACTOR
        if self._calibration_factor <= 0:
            raise LoihiConfigError(
                f"calibration_factor must be positive; got "
                f"{self._calibration_factor}"
            )

        if seed is None:
            try:
                seed = int(config.get("runtime", {}).get("seed", 42))
            except (TypeError, ValueError):
                seed = 42
        self._run_seed = int(seed)
        self._rng = np.random.default_rng(
            get_component_seed(self._run_seed, "loihi_emu")
        )

        # Pre-generate the input spike train. This is the input to the
        # SNN; it is deterministic given the seed, so multiple forward
        # passes produce the same spike count (only the wall-clock
        # time varies).
        self._input_spikes = self._rng.random(
            (self._n_timesteps(), _N_INPUT_NEURONS),
        ) < 0.1  # 10% firing rate on the input layer

        # Pre-generate the weights.
        self._w_input_hidden = self._rng.normal(
            0.0, 0.05, size=(_N_INPUT_NEURONS, _N_HIDDEN_NEURONS),
        )
        self._w_hidden_output = self._rng.normal(
            0.0, 0.05, size=(_N_HIDDEN_NEURONS, _N_OUTPUT_NEURONS),
        )

        self._log.info(
            "loihi_emulator_initialised",
            backend=self._backend,
            calibration_factor=self._calibration_factor,
            n_neurons_emulated=_N_NEURONS_EMULATED,
            n_neurons_chip=LOIHI_SPEC["total_neurons"],
            chip_scale_factor=round(_CHIP_SCALE_FACTOR, 3),
            n_timesteps=self._n_timesteps(),
        )

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "backend": self._backend,
            "calibration_factor": self._calibration_factor,
            "n_neurons_emulated": _N_NEURONS_EMULATED,
            "n_neurons_chip": LOIHI_SPEC["total_neurons"],
            "chip_scale_factor": _CHIP_SCALE_FACTOR,
            "n_timesteps": self._n_timesteps(),
            "loihi_spec": dict(LOIHI_SPEC),
        }

    @property
    def calibration_factor(self) -> float:
        return self._calibration_factor

    def _n_timesteps(self) -> int:
        return _N_TIMESTEPS

    # ------------------------------------------------------------------
    # Core: LIF forward pass
    # ------------------------------------------------------------------

    def _forward_pass(self) -> int:
        """
        Run one forward pass of the LIF network. Returns the total
        number of spikes emitted across all layers and timesteps.

        The simulation is vectorized over neurons. At each timestep:
            1. Update the input layer's membrane potential from the
               input spikes.
            2. Propagate the input spikes through the input→hidden
               weights to the hidden layer.
            3. Update the hidden layer's membrane potential.
            4. Propagate the hidden spikes through hidden→output.
            5. Update the output layer.
            6. Record spikes.

        The membrane update follows thesis Eq. 44 (p.95):
            τ_m dV/dt = -(V - V_rest) + I_syn + ξ
        which discretizes to:
            V[t+1] = V[t] + dt/τ_m * (-(V[t] - V_rest) + I_syn[t] + ξ[t])
        with the threshold-and-reset of Eq. 45:
            if V[t+1] >= V_th: spike, V[t+1] = V_reset, refractory period
        """
        dt = _LIF_DT_MS
        tau_m = _LIF_TAU_M_MS
        alpha = dt / tau_m
        v_th = _LIF_V_THRESHOLD
        v_reset = _LIF_V_RESET
        v_rest = _LIF_V_REST
        noise_std = _LIF_NOISE_STD

        # State vectors.
        v_input = np.zeros(_N_INPUT_NEURONS, dtype=np.float64)
        v_hidden = np.zeros(_N_HIDDEN_NEURONS, dtype=np.float64)
        v_output = np.zeros(_N_OUTPUT_NEURONS, dtype=np.float64)

        # Refractory counters (number of remaining timesteps).
        refrac_input = np.zeros(_N_INPUT_NEURONS, dtype=np.int32)
        refrac_hidden = np.zeros(_N_HIDDEN_NEURONS, dtype=np.int32)
        refrac_output = np.zeros(_N_OUTPUT_NEURONS, dtype=np.int32)

        total_spikes = 0
        n_timesteps = self._n_timesteps()

        for t in range(n_timesteps):
            # --- Input layer ---
            # Input spikes drive the input layer's membrane.
            i_syn_input = self._input_spikes[t].astype(np.float64) * 2.0
            # Refractory mask.
            mask_in = (refrac_input == 0)
            v_input[mask_in] = v_input[mask_in] + alpha * (
                -(v_input[mask_in] - v_rest) + i_syn_input[mask_in]
                + self._rng.normal(0.0, noise_std, size=mask_in.sum())
            )
            spiked_in = v_input >= v_th
            v_input[spiked_in] = v_reset
            refrac_input[spiked_in] = int(_LIF_TAU_REF_MS / dt)
            refrac_input[refrac_input > 0] -= 1
            total_spikes += int(spiked_in.sum())

            # --- Hidden layer ---
            # Synaptic input from input spikes.
            i_syn_hidden = self._input_spikes[t].astype(np.float64) @ self._w_input_hidden
            mask_h = (refrac_hidden == 0)
            v_hidden[mask_h] = v_hidden[mask_h] + alpha * (
                -(v_hidden[mask_h] - v_rest) + i_syn_hidden[mask_h]
                + self._rng.normal(0.0, noise_std, size=mask_h.sum())
            )
            spiked_h = v_hidden >= v_th
            v_hidden[spiked_h] = v_reset
            refrac_hidden[spiked_h] = int(_LIF_TAU_REF_MS / dt)
            refrac_hidden[refrac_hidden > 0] -= 1
            total_spikes += int(spiked_h.sum())

            # --- Output layer ---
            i_syn_output = spiked_h.astype(np.float64) @ self._w_hidden_output
            mask_o = (refrac_output == 0)
            v_output[mask_o] = v_output[mask_o] + alpha * (
                -(v_output[mask_o] - v_rest) + i_syn_output[mask_o]
                + self._rng.normal(0.0, noise_std, size=mask_o.sum())
            )
            spiked_o = v_output >= v_th
            v_output[spiked_o] = v_reset
            refrac_output[spiked_o] = int(_LIF_TAU_REF_MS / dt)
            refrac_output[refrac_output > 0] -= 1
            total_spikes += int(spiked_o.sum())

        return total_spikes

    # ------------------------------------------------------------------
    # Benchmarking
    # ------------------------------------------------------------------

    def measure_cpu_forward_time_us(self, n_repeats: int = _BENCHMARK_REPEATS) -> float:
        """
        Measure the wall-clock time of one forward pass on the local
        CPU, in microseconds, for the reduced (3,072-neuron) network.

        Returns the mean over `n_repeats` after a warm-up.
        """
        if n_repeats < 1:
            raise LoihiEmulationError("n_repeats must be >= 1")

        # Warm-up.
        for _ in range(_BENCHMARK_WARMUP):
            self._forward_pass()

        # Timed.
        start = time.perf_counter()
        for _ in range(n_repeats):
            self._forward_pass()
        elapsed = time.perf_counter() - start
        return (elapsed / n_repeats) * 1e6

    def benchmark(self, n_repeats: int = _BENCHMARK_REPEATS) -> LoihiBenchmarkReport:
        """
        Run the full benchmark and return a `LoihiBenchmarkReport`.

        The emulated Loihi time is:

            emulated_us = (measured_cpu_us * chip_scale_factor) / calibration_factor

        where chip_scale_factor extrapolates from the reduced network
        to the full 128K-neuron chip, and calibration_factor converts
        CPU time into Loihi time (see the module docstring).
        """
        import os
        import platform

        cpu_us = self.measure_cpu_forward_time_us(n_repeats=n_repeats)
        cpu_us_scaled = cpu_us * _CHIP_SCALE_FACTOR
        emulated_us = cpu_us_scaled / self._calibration_factor

        # Spike count is deterministic given the seed.
        spike_count = self._forward_pass()

        # Energy estimate:
        #   CPU: spike_count * chip_scale * energy_per_spike
        #   Loihi: spike_count * energy_per_spike + thesis overhead
        # The Loihi per-spike energy is *not* multiplied by the chip
        # scale because the spike count is what the reduced network
        # actually produces; the chip processes the same spike pattern
        # in parallel. Only the *time* is scaled; the *energy* is not.
        energy_per_spike_mj = ENERGY_PER_SPIKE_NJ * 1e-6
        cpu_energy_mj = spike_count * _CHIP_SCALE_FACTOR * energy_per_spike_mj
        # For Loihi, use the thesis's per-inference energy as the
        # reference; the estimate here is a cross-check, not a
        # replacement for the thesis figure.
        loihi_energy_mj = spike_count * energy_per_spike_mj

        report = LoihiBenchmarkReport(
            measured_cpu_forward_time_us=cpu_us,
            measured_cpu_forward_time_scaled_us=cpu_us_scaled,
            emulated_loihi_forward_time_us=emulated_us,
            measured_cpu_energy_estimate_mj=cpu_energy_mj,
            emulated_loihi_energy_estimate_mj=loihi_energy_mj,
            calibration_factor=self._calibration_factor,
            reference_loihi_latency_ms=LOIHI_SPEC["thesis_inference_latency_ms"],
            reference_loihi_energy_mj=LOIHI_SPEC["thesis_energy_per_inference_mj"],
            spike_count=spike_count,
            n_neurons=_N_NEURONS_EMULATED,
            n_timesteps=self._n_timesteps(),
            cpu_count=os.cpu_count() or 1,
            platform=platform.platform(),
        )
        self._log.info("loihi_benchmark", **report.to_dict())
        return report

    def get_forward_time_us(self) -> float:
        """
        Convenience accessor: run a short benchmark and return the
        emulated Loihi forward time in microseconds.
        """
        return self.benchmark(n_repeats=5).emulated_loihi_forward_time_us


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------

def benchmark_loihi_emulation(config: Dict[str, Any], seed: Optional[int] = None) -> Dict[str, Any]:
    """
    Convenience entry point: construct a LoihiEmulator, run the
    benchmark, and return the report as a dict.
    """
    emu = LoihiEmulator(config, seed=seed)
    return emu.benchmark().to_dict()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "LoihiEmulator",
    "LoihiBenchmarkReport",
    "LoihiEmulationError",
    "LoihiConfigError",
    "LoihiBackendNotAvailable",
    "LOIHI_CALIBRATION_FACTOR",
    "LOIHI_SPEC",
    "benchmark_loihi_emulation",
]


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def _self_test() -> int:
    import sys
    rc = 0
    print("=== loihi_mock self-test ===")

    config = {
        "hardware": {"loihi": {"backend": "emulated"}},
        "runtime": {"seed": 42},
    }

    # --- 1. Construct ---
    emu = LoihiEmulator(config, seed=42)
    if emu.calibration_factor != LOIHI_CALIBRATION_FACTOR:
        print(f"[FAIL] calibration factor: {emu.calibration_factor}")
        rc = 1
    else:
        print(f"[OK]   LoihiEmulator constructed "
              f"(calibration={emu.calibration_factor}x)")

    # --- 2. Parameters expose the spec ---
    p = emu.parameters
    if p["n_neurons_chip"] != 128 * 1024:
        print(f"[FAIL] chip neurons: {p['n_neurons_chip']}")
        rc = 1
    elif p["loihi_spec"]["thesis_inference_latency_ms"] != 0.8:
        print(f"[FAIL] thesis latency: {p['loihi_spec']['thesis_inference_latency_ms']}")
        rc = 1
    else:
        print(f"[OK]   parameters: 128K neurons, 0.8 ms latency, "
              f"0.12 mJ energy")

    # --- 3. Real backend raises ---
    real_config = {
        "hardware": {"loihi": {"backend": "real"}},
        "runtime": {"seed": 42},
    }
    try:
        LoihiEmulator(real_config, seed=42)
    except LoihiBackendNotAvailable:
        print(f"[OK]   backend='real' raises LoihiBackendNotAvailable")
    else:
        print(f"[FAIL] backend='real' was accepted")
        rc = 1

    # --- 4. Single forward pass is deterministic ---
    spikes_a = emu._forward_pass()
    spikes_b = emu._forward_pass()
    # The forward pass uses self._rng for the noise, so the two
    # calls draw different noise and may produce different spike
    # counts. Reset the RNG between calls to make them identical.
    seed_after = emu._rng.bit_generator.state
    emu._rng = np.random.default_rng(get_component_seed(42, "loihi_emu"))
    spikes_c = emu._forward_pass()
    emu._rng = np.random.default_rng(get_component_seed(42, "loihi_emu"))
    spikes_d = emu._forward_pass()
    if spikes_c != spikes_d:
        print(f"[FAIL] forward pass determinism: {spikes_c} vs {spikes_d}")
        rc = 1
    else:
        print(f"[OK]   forward pass deterministic: {spikes_c} spikes "
              f"on {_N_NEURONS_EMULATED} neurons × {_N_TIMESTEPS} steps")

    # --- 5. Measure CPU time ---
    cpu_us = emu.measure_cpu_forward_time_us(n_repeats=3)
    if cpu_us <= 0 or cpu_us > 1e6:
        print(f"[FAIL] CPU forward time out of range: {cpu_us} μs")
        rc = 1
    else:
        print(f"[OK]   measured CPU forward time: {cpu_us:.1f} μs "
              f"(reduced network)")

    # --- 6. Full benchmark ---
    report = emu.benchmark(n_repeats=3)
    d = report.to_dict()
    required = (
        "measured_cpu_forward_time_us", "measured_cpu_forward_time_scaled_us",
        "emulated_loihi_forward_time_us", "calibration_factor",
        "reference_loihi_latency_ms", "reference_loihi_energy_mj",
        "spike_count", "n_neurons", "n_timesteps", "hardware",
    )
    missing = [k for k in required if k not in d]
    if missing:
        print(f"[FAIL] report.to_dict missing keys: {missing}")
        rc = 1
    elif d["hardware"] != "loihi_emulated":
        print(f"[FAIL] hardware tag: {d['hardware']}")
        rc = 1
    else:
        print(f"[OK]   benchmark report: {len(d)} keys, "
              f"hardware={d['hardware']}")
        print(f"         measured CPU: {report.measured_cpu_forward_time_us:.1f} μs")
        print(f"         chip-scaled CPU: {report.measured_cpu_forward_time_scaled_us:.1f} μs")
        print(f"         emulated Loihi: {report.emulated_loihi_forward_time_us:.4f} μs")
        print(f"         reference thesis latency: "
              f"{report.reference_loihi_latency_ms} ms = "
              f"{report.reference_loihi_latency_ms * 1000:.1f} μs")

    # --- 7. Invertibility ---
    inverted = (
        report.emulated_loihi_forward_time_us
        * report.calibration_factor
        / _CHIP_SCALE_FACTOR
    )
    if abs(inverted - report.measured_cpu_forward_time_us) > 1e-6:
        print(f"[FAIL] invertibility: {inverted} vs "
              f"{report.measured_cpu_forward_time_us}")
        rc = 1
    else:
        print(f"[OK]   invertibility: emulated × factor / chip_scale = measured CPU")

    # --- 8. get_forward_time_us convenience ---
    t = emu.get_forward_time_us()
    if t <= 0:
        print(f"[FAIL] get_forward_time_us: {t}")
        rc = 1
    else:
        print(f"[OK]   get_forward_time_us: {t:.4f} μs")

    # --- 9. Convenience function ---
    d2 = benchmark_loihi_emulation(config, seed=42)
    if d2["hardware"] != "loihi_emulated":
        print(f"[FAIL] benchmark_loihi_emulation hardware: {d2['hardware']}")
        rc = 1
    else:
        print(f"[OK]   benchmark_loihi_emulation: returns dict with "
              f"hardware=loihi_emulated")

    # --- 10. Override ---
    override_config = {
        "hardware": {"loihi": {"backend": "emulated", "calibration_factor": 100.0}},
        "runtime": {"seed": 42},
    }
    emu_ov = LoihiEmulator(override_config, seed=42)
    if emu_ov.calibration_factor != 100.0:
        print(f"[FAIL] override factor: {emu_ov.calibration_factor}")
        rc = 1
    else:
        print(f"[OK]   calibration_factor override: 100.0 accepted")

    print("=== self-test complete ===")
    return rc


if __name__ == "__main__":
    import sys
    sys.exit(_self_test())
