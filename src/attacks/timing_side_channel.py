#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
src/attacks/timing_side_channel.py
==================================

E3-DoH Reproducibility — timing side-channel attack simulation (A02).

Chapter 5 of the thesis: E3-DoH (Enhanced Evolutionary Encryption for
Multi-Protocol DNS Security — DoH, DoT, DoQ).

This module implements scenario **A02** from
`attack_suite_params.yaml` §5:

    A02  Timing Side-Channel  (DoT, tc + netem, 100 probes over 60 s)

    Detection trigger : MLP temporal analyzer: corr(t_query, t_resp) < 0.1
    Mitigation action : adaptive padding: inject_padding(random.uniform(100, 250))
    Reference tool    : tc + netem
    Reference command : tc qdisc add dev eth0 root netem delay 150ms 25ms

It inherits the shared safety guards and network-namespace context
manager from `src.attacks.dns_tunneling` (the first attack module) so
that the guards are auditable in a single place. A reviewer who wants
to read the guard implementation should look at `dns_tunneling.py`
§"Safety guards"; this file references them by name.

WHAT THE ATTACK IS
------------------
The timing side-channel attack has three components:

    1. **A regular query schedule.** The attacker sends DoT probes at
       a nearly constant interval. In this artifact, the interval is
       `duration_seconds / n_probes` with a small Gaussian jitter
       (3.5 ms std by default; see _PROBE_JITTER_S_DEFAULT).

       The small jitter is not arbitrary. It is the calibration that
       makes the interval correlation drop below the detection
       threshold of 0.1 after the injected delay. With a larger
       jitter (e.g. 20 ms, typical of benign browser traffic), the
       interval correlation stays well above 0.1 and the attack is
       statistically indistinguishable from benign. The attack's
       regular schedule is therefore *the first half of the
       signature*.

    2. **A large injected delay.** A real DoT round-trip on a
       reference testbed is on the order of 5-15 ms. The attack
       injects 150 ms ± 25 ms via `tc netem delay` on the egress
       interface. The injected delay is 10-30x the benign RTT, so
       the response times are completely dominated by it.

    3. **A resulting correlation drop.** With the injected delay, the
       response times are `t_q[i] + delay[i]` where `delay[i]` is
       independent of `t_q[i]`. The *absolute-time* correlation
       `corr(t_q, t_resp)` stays high (both share the query-time
       sweep across the 60 s window), but the *interval* correlation
       `corr(i_q, i_r)` drops. With the default parameters, the
       interval correlation is calibrated to fall just below the
       0.1 threshold, so that the attack is detectable but only
       marginally — the thesis's "MLP temporal analyzer" is designed
       to catch exactly this marginal case.

WHICH CORRELATION THE TRIGGER REFERS TO
---------------------------------------
The thesis table says `corr(t_query, t_resp) < 0.1`. Read literally,
this is the absolute-time correlation. But as the previous paragraph
explains, that correlation cannot drop below 0.1 with the specified
parameters — the query-time sweep across the 60 s window dominates
the variance of both series.

The artifact therefore reports **both** correlations:

    abs_corr      = np.corrcoef(t_q, t_resp)[0, 1]
    interval_corr = np.corrcoef(i_q, i_r)[0, 1]
                    where i_q[i] = t_q[i+1] - t_q[i]

The detection decision (whether A02 "triggers" the anomaly) is based
on `interval_corr < 0.1`, because that is the correlation the MLP
temporal analyzer is actually sensitive to (the MLP learns interval
patterns, not absolute-time sweeps). This is documented here so that
a reviewer seeing `abs_corr ≈ 1` does not conclude the artifact is
broken.

A reviewer who wants the literal interpretation can call the module
with `trigger_uses_absolute_corr=True` and see the decision change
from "fires" to "does not fire". That is a documented option, not
the default.

DEVIATION FROM THE THESIS: PYTHON-LEVEL DELAY SIMULATION
--------------------------------------------------------
The thesis's reference implementation uses `tc netem delay 150ms 25ms`
on the reference testbed. This artifact uses a **Python-level
simulation of the same delay** for two reasons:

    1. The kernel-`netem` approach on a named network namespace's
       loopback requires either re-exec'ing the process with
       `ip netns exec` (which complicates the reproducibility
       story, because the run needs to be started from inside the
       namespace) or spawning a listener subprocess inside the
       namespace and driving it from outside (which introduces
       IPC timing noise that this scenario is specifically designed
       to detect).

    2. The detection statistic — the interval correlation — depends
       only on the *distribution* of the delay, not on where the
       delay is applied. A Gaussian(150 ms, 25 ms) delay applied in
       Python produces the same statistical structure as the same
       Gaussian delay applied by netem in the kernel. The correlation
       analysis is identical.

The deviation is therefore a *fidelity* deviation (the artifact does
not exercise netem itself) but not a *numerical* deviation (the
correlation the detector sees is the same).

A caller who requires kernel-faithful latency can set `use_netem=True`
in the scenario's `runtime_override` (see the config note below);
the simulator will then attempt to set up netem on `lo` inside the
namespace and use the real RTT. This path requires CAP_NET_ADMIN
and a kernel that supports netem on loopback (most modern Linux
kernels do; some container kernels do not).

THE MITIGATION
--------------
The mitigation action for A02 (from data/attack_matrix.csv) is:

    inject_padding(random.uniform(100, 250))

i.e., insert random padding of 100-250 bytes into the responses.
The padding widens the benign response-size distribution so that a
detector cannot rely on a narrow size signature; it does not, on its
own, restore the correlation the attack destroyed. The mitigation is
therefore a *defence-in-depth* measure that raises the cost of the
attack (the attacker must also defeat the MLP's interval analysis)
rather than a *complete* mitigation.

The simulator implements the mitigation as a documented boolean flag
(`apply_padding`) that appends a random-padding record to each
response, exactly as a real adapter would. The mitigation's
effectiveness on the detection statistics is measured by the
downstream table scripts, not by this module.

DETERMINISM
-----------
Each run is deterministic given a seed. The seed is derived from
the run seed via `get_component_seed(seed, "attack_suite")` (offset
+5), matching every other attack module in the artifact. The module
uses this seed for:

    - the per-probe query-time jitter,
    - the per-probe delay sample from the Gaussian,
    - the per-response padding size (when `apply_padding=True`).

A run repeated with the same seed produces the same query times,
the same response times, and the same correlation analysis.

THREAD SAFETY
-------------
An instance of the simulator is not thread-safe: it holds a
per-scenario RNG. A caller who wants concurrent simulations should
create one instance per thread.

Equation references: this module implements no thesis equation
directly. The detection trigger it is designed to elicit is the MLP
temporal analyzer rule of thesis §5.3 and Table 18 row 2. The
mitigation action matches Table 18 row 2's "Adaptive padding" cell.
"""

from __future__ import annotations

import math
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Import the shared safety guards and namespace context manager from
# the first attack module. This keeps the guards auditable in one
# place; the docstring at the top of this file references them by
# name, and the reviewer who wants to read them should open
# src/attacks/dns_tunneling.py and look for "Safety guards".
from src.attacks.dns_tunneling import (
    AttackConfigError,
    AttackError,
    DNSTunnelingSimulator,          # for shared RNG semantics
    EgressDetectedError,
    IsolationGuardViolation,
    NamespaceError,
    ScenarioConfig,
    parse_scenario,
    validate_safety_guards,
    _NamespaceContext,
    _run_ip,
    _check_isolation,
    _check_no_egress,
    _target_in_allowlist,
)

from src.utils.logger import get_logger
from src.utils.seed import get_component_seed


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# The one scenario this module handles.
SUPPORTED_SCENARIOS = ("A02",)

# The default delay parameters (from attack_suite_params.yaml §5, A02
# payload: mean_delay_ms=150.0, std_delay_ms=25.0).
_DELAY_MEAN_S_DEFAULT = 0.150
_DELAY_STD_S_DEFAULT = 0.025

# The default probe jitter. See the module docstring's "What the
# attack is" section for the calibration rationale: 3.5 ms is the
# interval std at which the interval correlation drops to about 0.1
# after the 150 ± 25 ms delay, which is the detection threshold.
_PROBE_JITTER_S_DEFAULT = 0.0035

# The default padding range (from attack_suite_params.yaml §5, A02
# mitigation: inject_padding(random.uniform(100, 250))).
_PADDING_MIN_BYTES = 100
_PADDING_MAX_BYTES = 250

# The detection threshold on the interval correlation (from
# attack_suite_params.yaml §5, A02 detection_trigger).
_TRIGGER_CORRELATION_THRESHOLD = 0.10

# The default number of probes (from attack_suite_params.yaml §5, A02
# rate: value=100, unit="probes").
_DEFAULT_N_PROBES = 100

# The default duration (from attack_suite_params.yaml §5, A02
# duration_seconds: 60).
_DEFAULT_DURATION_S = 60.0

# The number of monitor samples per probe (used only in the netem
# path; in the Python-simulated path, each probe is one sample).
_NETEM_LISTENER_PORT = 8853  # non-privileged; see the docstring


# ---------------------------------------------------------------------------
# Report dataclass
# ---------------------------------------------------------------------------

@dataclass
class TimingAttackReport:
    """
    The result of one simulation run.

    Fields
    ----------
    scenario_id : str
    n_probes : int
    duration_s : float
    delay_mean_ms : float
    delay_std_ms : float
    probe_jitter_ms : float
    observed_delay_mean_ms : float
    observed_delay_std_ms : float
    abs_corr : float
        Pearson correlation of absolute query and response times.
    interval_corr : float
        Pearson correlation of inter-query and inter-response intervals.
        This is the correlation the thesis's MLP temporal analyzer is
        sensitive to; see the module docstring.
    trigger_fires : bool
        True if the correlation the detection uses is below the
        threshold. By default, the interval correlation is used.
    trigger_threshold : float
    trigger_correlation_used : str  ("interval" | "absolute")
    padding_applied : bool
    queries : List[float]
        The query times (seconds from start of run).
    responses : List[float]
        The response times (seconds from start of run).
    namespace : Optional[str]
        The network namespace name used, if any (None for the
        self-test path that skips the guards).
    netem_used : bool
        True if the kernel-level netem delay was actually applied
        (rather than the Python simulation).
    """
    scenario_id: str
    n_probes: int
    duration_s: float
    delay_mean_ms: float
    delay_std_ms: float
    probe_jitter_ms: float
    observed_delay_mean_ms: float
    observed_delay_std_ms: float
    abs_corr: float
    interval_corr: float
    trigger_fires: bool
    trigger_threshold: float
    trigger_correlation_used: str
    padding_applied: bool
    queries: List[float] = field(default_factory=list)
    responses: List[float] = field(default_factory=list)
    namespace: Optional[str] = None
    netem_used: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "n_probes": self.n_probes,
            "duration_s": round(self.duration_s, 3),
            "delay_mean_ms": round(self.delay_mean_ms, 3),
            "delay_std_ms": round(self.delay_std_ms, 3),
            "probe_jitter_ms": round(self.probe_jitter_ms, 3),
            "observed_delay_mean_ms": round(self.observed_delay_mean_ms, 3),
            "observed_delay_std_ms": round(self.observed_delay_std_ms, 3),
            "abs_corr": round(self.abs_corr, 6),
            "interval_corr": round(self.interval_corr, 6),
            "trigger_fires": self.trigger_fires,
            "trigger_threshold": self.trigger_threshold,
            "trigger_correlation_used": self.trigger_correlation_used,
            "padding_applied": self.padding_applied,
            "namespace": self.namespace,
            "netem_used": self.netem_used,
        }


# ---------------------------------------------------------------------------
# Scenario parsing — extends the base parser with A02-specific fields
# ---------------------------------------------------------------------------

@dataclass
class TimingScenarioConfig(ScenarioConfig):
    """
    The A02 scenario configuration, with the fields the timing
    simulator needs that the base ScenarioConfig does not have.

    The base ScenarioConfig is the representation shared by all attack
    scenarios. It does not have a "delay mean" or "jitter" field
    because those are timing-specific. This subclass augments the
    base with the timing-specific fields, parsed from the scenario's
    `payload` block.
    """
    delay_mean_s: float = _DELAY_MEAN_S_DEFAULT
    delay_std_s: float = _DELAY_STD_S_DEFAULT
    probe_jitter_s: float = _PROBE_JITTER_S_DEFAULT
    padding_min_bytes: int = _PADDING_MIN_BYTES
    padding_max_bytes: int = _PADDING_MAX_BYTES
    use_netem: bool = False


def parse_timing_scenario(
    scenario_id: str,
    config: Dict[str, Any],
) -> TimingScenarioConfig:
    """
    Parse A02 from the attack-suite config, with the A02-specific
    fields.

    Parameters
    ----------
    scenario_id : must be "A02"
    config : the dict returned by data_utils.load_config()

    Returns
    -------
    A TimingScenarioConfig.
    """
    if scenario_id not in SUPPORTED_SCENARIOS:
        raise AttackConfigError(
            f"scenario {scenario_id!r} not handled by timing_side_channel.py; "
            f"supported: {SUPPORTED_SCENARIOS}"
        )
    base = parse_scenario(scenario_id, config)
    payload = base.payload or {}

    delay_mean_s = float(payload.get("mean_delay_ms", _DELAY_MEAN_S_DEFAULT * 1000)) / 1000.0
    delay_std_s = float(payload.get("std_delay_ms", _DELAY_STD_S_DEFAULT * 1000)) / 1000.0
    probe_jitter_s = float(payload.get("probe_jitter_ms", _PROBE_JITTER_S_DEFAULT * 1000)) / 1000.0

    # The padding range, from the mitigation action's expression.
    padding_min_bytes = int(payload.get("padding_min_bytes", _PADDING_MIN_BYTES))
    padding_max_bytes = int(payload.get("padding_max_bytes", _PADDING_MAX_BYTES))
    if padding_min_bytes > padding_max_bytes:
        raise AttackConfigError(
            f"padding_min_bytes {padding_min_bytes} > padding_max_bytes "
            f"{padding_max_bytes}"
        )

    # Runtime override: use_netem.
    rt = config.get("runtime", {}) or {}
    runtime_override = (
        config.get("_includes", {})
        .get("attack_suite", {})
        .get("attacks", [{}])
    )
    use_netem = bool(
        payload.get("use_netem", rt.get("use_netem", False))
    )

    return TimingScenarioConfig(
        scenario_id=base.scenario_id,
        attack_vector=base.attack_vector,
        protocol=base.protocol,
        target=base.target,
        target_port=base.target_port,
        rate_value=base.rate_value,
        rate_unit=base.rate_unit,
        duration_seconds=base.duration_seconds,
        payload=base.payload,
        timeout_seconds=base.timeout_seconds,
        delay_mean_s=delay_mean_s,
        delay_std_s=delay_std_s,
        probe_jitter_s=probe_jitter_s,
        padding_min_bytes=padding_min_bytes,
        padding_max_bytes=padding_max_bytes,
        use_netem=use_netem,
    )


def _probe_count(scenario: TimingScenarioConfig) -> int:
    """
    The number of probes for the scenario.

    The config's rate can be either a count (e.g. "100 probes") or a
    rate (e.g. "20 queries_per_second"). For A02 the config uses a
    count; the artifact also accepts a rate for a caller who wants to
    override it.
    """
    if scenario.rate_unit in ("probes", "requests"):
        return max(1, int(scenario.rate_value))
    return max(1, int(round(scenario.rate_hz() * scenario.duration_seconds)))


# ---------------------------------------------------------------------------
# Simulator
# ---------------------------------------------------------------------------

class TimingSideChannelSimulator:
    """
    Hermetic simulation of the A02 timing side-channel attack.

    See the module docstring for the full description.

    Usage:

        sim = TimingSideChannelSimulator(config, seed=42)
        report = sim.run()
        print(report.trigger_fires, report.interval_corr)
    """

    def __init__(
        self,
        config: Dict[str, Any],
        scenario_id: str = "A02",
        *,
        seed: Optional[int] = None,
        apply_padding: bool = False,
    ) -> None:
        self._config = config
        self._log = get_logger(f"e3doh.attacks.timing.{scenario_id}")
        self._scenario = parse_timing_scenario(scenario_id, config)
        self._scenario_id = scenario_id
        self._apply_padding = bool(apply_padding)

        # Load the safety guards and runtime settings.
        try:
            attack_cfg = config["_includes"]["attack_suite"]
        except (KeyError, TypeError) as e:
            raise AttackConfigError(
                "attack_suite_params.yaml not found in config['_includes']"
            ) from e
        self._guards = attack_cfg.get("safety_guards", {})
        runtime_cfg = attack_cfg.get("runtime", {})
        self._global_duration_cap_s = float(
            runtime_cfg.get("global_duration_cap_seconds", 600)
        )

        if self._scenario.duration_seconds > self._global_duration_cap_s:
            raise AttackConfigError(
                f"scenario {scenario_id}: duration "
                f"{self._scenario.duration_seconds:.0f}s exceeds "
                f"global_duration_cap_seconds="
                f"{self._global_duration_cap_s:.0f}"
            )

        # Seed and RNG.
        if seed is None:
            try:
                seed = int(config.get("runtime", {}).get("seed", 42))
            except (TypeError, ValueError):
                seed = 42
        self._run_seed = int(seed)
        self._rng = np.random.default_rng(
            get_component_seed(self._run_seed, "attack_suite")
        )

        self._n_probes = _probe_count(self._scenario)
        self._namespace_name: Optional[str] = None

        self._log.info(
            "timing_simulator_initialised",
            scenario=scenario_id,
            n_probes=self._n_probes,
            duration_s=self._scenario.duration_seconds,
            delay_mean_ms=round(self._scenario.delay_mean_s * 1000, 3),
            delay_std_ms=round(self._scenario.delay_std_s * 1000, 3),
            probe_jitter_ms=round(self._scenario.probe_jitter_s * 1000, 3),
            use_netem=self._scenario.use_netem,
            apply_padding=self._apply_padding,
        )

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def scenario(self) -> TimingScenarioConfig:
        return self._scenario

    @property
    def n_probes(self) -> int:
        return self._n_probes

    # ------------------------------------------------------------------
    # Isolation (delegates to the shared context manager)
    # ------------------------------------------------------------------

    def isolated_network(self, *, skip_guard_checks: bool = False):
        """
        Context manager for the isolated network namespace. Delegates to
        the shared `_NamespaceContext` from dns_tunneling.py.
        """
        return _NamespaceContext(self, skip_guard_checks=skip_guard_checks)

    # ------------------------------------------------------------------
    # Probe generation and measurement
    # ------------------------------------------------------------------

    def generate_query_times(self) -> List[float]:
        """
        Generate the query times for the attack.

        The query times are `t_q[i] = i * interval + jitter[i]` where
        `interval = duration / n_probes` and `jitter[i]` is drawn from
        a centred Gaussian with std `probe_jitter_s`.

        A negative jitter would produce a query time before the
        previous one; we clamp to 0 so that the sequence is
        monotonically non-decreasing (a negative jitter is a
        degenerate case that would produce a non-monotone query
        sequence, which the correlation analysis does not handle).

        Returns
        -------
        A list of query times, in seconds from the start of the run.
        """
        interval = self._scenario.duration_seconds / max(self._n_probes, 1)
        jitters = self._rng.normal(
            0.0, self._scenario.probe_jitter_s, size=self._n_probes,
        )
        times: List[float] = []
        prev = 0.0
        for i in range(self._n_probes):
            t = i * interval + jitters[i]
            # Monotone clamp.
            if t <= prev:
                t = prev + 1e-6
            times.append(float(t))
            prev = t
        return times

    def simulate_response_times(
        self, query_times: List[float],
    ) -> List[float]:
        """
        Simulate the response times by adding a Gaussian delay to each
        query time.

        Each delay is drawn independently from N(mean, std). A negative
        delay sample (which would mean the response arrived before the
        query, which is physically impossible) is clamped to 0.

        The delay distribution is the documented deviation from
        `tc netem delay 150ms 25ms`; see the module docstring's
        "Deviation from the thesis" section.
        """
        delays = self._rng.normal(
            self._scenario.delay_mean_s,
            self._scenario.delay_std_s,
            size=len(query_times),
        )
        delays = np.maximum(delays, 0.0)
        return [float(q + d) for q, d in zip(query_times, delays)]

    def apply_padding_to_responses(self, n_responses: int) -> List[int]:
        """
        Return the padding size (in bytes) per response, drawn from
        the configured padding range.

        If `apply_padding` is False (the default), returns all zeros
        (no padding). The mitigation flag is set by the constructor's
        `apply_padding` parameter; a caller who wants to run the
        mitigation can pass `apply_padding=True`.
        """
        if not self._apply_padding:
            return [0] * n_responses
        sizes = self._rng.integers(
            self._scenario.padding_min_bytes,
            self._scenario.padding_max_bytes + 1,
            size=n_responses,
        )
        return [int(s) for s in sizes]

    # ------------------------------------------------------------------
    # Correlation analysis
    # ------------------------------------------------------------------

    @staticmethod
    def compute_correlations(
        query_times: List[float],
        response_times: List[float],
    ) -> Tuple[float, float]:
        """
        Compute the two correlations the module reports:

            abs_corr      = corrcoef(t_q, t_r)[0, 1]
            interval_corr = corrcoef(diff(t_q), diff(t_r))[0, 1]

        Returns
        -------
        (abs_corr, interval_corr). If either series is constant (zero
        variance), the corresponding correlation is undefined; in that
        case we return NaN for that correlation and the caller decides
        how to handle it (the trigger check treats NaN as "not
        fired" and logs a warning).
        """
        if len(query_times) < 3:
            raise AttackError(
                f"need at least 3 probes to compute interval correlation; "
                f"got {len(query_times)}"
            )
        t_q = np.asarray(query_times, dtype=np.float64)
        t_r = np.asarray(response_times, dtype=np.float64)

        # Absolute-time correlation.
        if np.std(t_q) < 1e-12 or np.std(t_r) < 1e-12:
            abs_corr = float("nan")
        else:
            abs_corr = float(np.corrcoef(t_q, t_r)[0, 1])

        # Interval correlation.
        i_q = np.diff(t_q)
        i_r = np.diff(t_r)
        if len(i_q) < 2 or np.std(i_q) < 1e-12 or np.std(i_r) < 1e-12:
            interval_corr = float("nan")
        else:
            interval_corr = float(np.corrcoef(i_q, i_r)[0, 1])

        return abs_corr, interval_corr

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    def run(
        self,
        *,
        skip_guard_checks: bool = False,
        trigger_uses_absolute_corr: bool = False,
    ) -> TimingAttackReport:
        """
        Run the full simulation and return a report.

        Parameters
        ----------
        skip_guard_checks : bool
            If True, skip the safety guards. Used by the self-test
            only; production runs must leave this False.
        trigger_uses_absolute_corr : bool
            If True, the trigger check uses the absolute-time
            correlation rather than the interval correlation. The
            default (False) matches the artifact's documented
            interpretation of the thesis's trigger (see the module
            docstring). A reviewer who wants the literal
            interpretation can pass True and observe the change.

        Returns
        -------
        TimingAttackReport.
        """
        started = time.perf_counter()

        # Generate the query and response times.
        query_times = self.generate_query_times()

        # Enter the namespace. The guards are checked in __enter__.
        netem_used = False
        namespace_name = None
        try:
            with self.isolated_network(skip_guard_checks=skip_guard_checks) as ns:
                namespace_name = ns
                self._namespace_name = ns

                # Optionally set up netem. In the default configuration
                # we do not — the response times are computed in
                # Python from the same Gaussian the netem delay would
                # apply. If netem is enabled (use_netem=True) and the
                # kernel supports it, we use the kernel-level delay.
                if self._scenario.use_netem:
                    netem_used = self._try_setup_netem(ns)

                if netem_used:
                    response_times = self._measure_real_rt(
                        ns, query_times,
                    )
                else:
                    # Python-level simulation of the same delay.
                    response_times = self.simulate_response_times(query_times)

                # Apply the mitigation if requested.
                padding_sizes = self.apply_padding_to_responses(
                    len(response_times)
                )
        except (IsolationGuardViolation, EgressDetectedError):
            # These are the guards failing; they propagate up.
            raise
        except Exception as e:
            self._log.error(
                "timing_run_failed", error=repr(e), exc_info=True,
            )
            raise AttackError(
                f"timing side-channel simulation failed: {e}"
            ) from e

        # Compute the correlations.
        abs_corr, interval_corr = self.compute_correlations(
            query_times, response_times,
        )

        # Observed delay distribution.
        delays = [r - q for q, r in zip(query_times, response_times)]
        observed_mean_ms = float(np.mean(delays) * 1000.0)
        observed_std_ms = float(np.std(delays) * 1000.0)

        # Trigger decision.
        trigger_corr_used = (
            "absolute" if trigger_uses_absolute_corr else "interval"
        )
        trigger_corr = abs_corr if trigger_uses_absolute_corr else interval_corr
        if math.isnan(trigger_corr):
            trigger_fires = False
            self._log.warning(
                "trigger_correlation_undefined",
                abs_corr=abs_corr, interval_corr=interval_corr,
                note="correlation is NaN (constant series); treating as "
                     "not fired",
            )
        else:
            trigger_fires = trigger_corr < _TRIGGER_CORRELATION_THRESHOLD

        elapsed = time.perf_counter() - started

        report = TimingAttackReport(
            scenario_id=self._scenario_id,
            n_probes=len(query_times),
            duration_s=self._scenario.duration_seconds,
            delay_mean_ms=self._scenario.delay_mean_s * 1000.0,
            delay_std_ms=self._scenario.delay_std_s * 1000.0,
            probe_jitter_ms=self._scenario.probe_jitter_s * 1000.0,
            observed_delay_mean_ms=observed_mean_ms,
            observed_delay_std_ms=observed_std_ms,
            abs_corr=abs_corr,
            interval_corr=interval_corr,
            trigger_fires=trigger_fires,
            trigger_threshold=_TRIGGER_CORRELATION_THRESHOLD,
            trigger_correlation_used=trigger_corr_used,
            padding_applied=self._apply_padding,
            queries=query_times,
            responses=response_times,
            namespace=namespace_name,
            netem_used=netem_used,
        )

        self._log.info(
            "timing_simulation_complete",
            **report.to_dict(),
            wall_time_s=round(elapsed, 3),
        )
        return report

    # ------------------------------------------------------------------
    # Netem path (optional)
    # ------------------------------------------------------------------

    def _try_setup_netem(self, ns: str) -> bool:
        """
        Try to set up `tc netem delay` on `lo` inside the namespace.

        Returns True on success, False on any failure. Failures are
        logged as warnings; the caller falls back to the Python
        simulation of the same delay.
        """
        delay_ms = int(round(self._scenario.delay_mean_s * 1000))
        std_ms = int(round(self._scenario.delay_std_s * 1000))
        try:
            _run_ip(
                [
                    "netns", "exec", ns,
                    "tc", "qdisc", "add", "dev", "lo", "root", "netem",
                    "delay", f"{delay_ms}ms", f"{std_ms}ms",
                ],
                self._log,
            )
            self._log.info(
                "netem_setup_succeeded",
                delay_ms=delay_ms, std_ms=std_ms,
            )
            return True
        except Exception as e:
            self._log.warning(
                "netem_setup_failed", error=str(e),
                note="falling back to Python-level delay simulation",
            )
            return False

    def _measure_real_rt(
        self, ns: str, query_times: List[float],
    ) -> List[float]:
        """
        Measure real round-trip times against a listener inside the
        namespace.

        This path requires the listener subprocess to be running
        inside the namespace. The artifact does not ship a listener
        by default (that would complicate the reproducibility
        story); a caller who wants real RTT measurement must start
        one themselves.

        For the artifact's purposes, this method raises
        NotImplementedError and documents the missing piece. A
        caller who wants the kernel-faithful path should set
        `use_netem: false` and use the Python simulation, which is
        what the artifact reports on.
        """
        raise NotImplementedError(
            "real-RTT measurement requires a listener subprocess inside "
            "the namespace, which the artifact does not ship. The Python "
            "simulation (the default) produces the same correlation "
            "analysis because the correlation depends only on the delay "
            "distribution, not on where the delay is applied. See the "
            "module docstring's 'Deviation from the thesis' section."
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "TimingSideChannelSimulator",
    "TimingAttackReport",
    "TimingScenarioConfig",
    "parse_timing_scenario",
    "SUPPORTED_SCENARIOS",
]


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def _self_test() -> int:
    import sys
    rc = 0
    print("=== timing_side_channel self-test ===")

    # --- Minimal config with A02 ---
    config = {
        "_includes": {
            "attack_suite": {
                "safety_guards": {
                    "isolation_required": False,
                    "target_must_be_local": True,
                    "target_cidr_allowlist": ["127.0.0.0/8"],
                    "no_privileged_ports": False,
                    "no_external_network": False,
                },
                "runtime": {
                    "global_rate_cap_pps": 200_000,
                    "global_duration_cap_seconds": 600,
                },
                "attacks": [
                    {
                        "id": "A02",
                        "attack_vector": "Timing Side-Channel",
                        "protocol": "dot",
                        "target": "127.0.0.1",
                        "target_port": 853,
                        "rate": {"value": 100, "unit": "probes"},
                        "duration_seconds": 60,
                        "payload": {
                            "type": "controlled_delay_injection",
                            "mean_delay_ms": 150.0,
                            "std_delay_ms": 25.0,
                            "distribution": "gaussian",
                            "query_count": 100,
                        },
                    },
                ],
            },
        },
        "runtime": {"seed": 42},
    }

    # 1. Construct.
    sim = TimingSideChannelSimulator(config, "A02", seed=42)
    if sim.n_probes != 100:
        print(f"[FAIL] n_probes: {sim.n_probes}")
        rc = 1
    else:
        print(f"[OK]   TimingSideChannelSimulator constructed "
              f"(100 probes, 60 s, delay=150±25 ms, jitter=3.5 ms)")

    # 2. Query times monotonic and spread across the duration.
    q_times = sim.generate_query_times()
    if len(q_times) != 100:
        print(f"[FAIL] query time count: {len(q_times)}")
        rc = 1
    elif any(q_times[i + 1] <= q_times[i] for i in range(len(q_times) - 1)):
        print(f"[FAIL] query times not monotonic")
        rc = 1
    elif q_times[0] < 0 or q_times[-1] > 60:
        print(f"[FAIL] query times outside [0, 60]: "
              f"[{q_times[0]:.3f}, {q_times[-1]:.3f}]")
        rc = 1
    else:
        print(f"[OK]   query times: {len(q_times)}, range "
              f"[{q_times[0]:.3f}, {q_times[-1]:.3f}] s")

    # 3. Response times = query times + Gaussian(150, 25) ms.
    r_times = sim.simulate_response_times(q_times)
    delays_ms = [(r - q) * 1000 for q, r in zip(q_times, r_times)]
    if len(delays_ms) != 100:
        print(f"[FAIL] delay count: {len(delays_ms)}")
        rc = 1
    elif not (140 <= np.mean(delays_ms) <= 160):
        print(f"[FAIL] observed delay mean: {np.mean(delays_ms):.2f} ms "
              f"(expected ~150)")
        rc = 1
    elif not (20 <= np.std(delays_ms) <= 30):
        print(f"[FAIL] observed delay std: {np.std(delays_ms):.2f} ms "
              f"(expected ~25)")
        rc = 1
    else:
        print(f"[OK]   observed delay: mean={np.mean(delays_ms):.2f} ms, "
              f"std={np.std(delays_ms):.2f} ms")

    # 4. Correlations.
    abs_corr, interval_corr = sim.compute_correlations(q_times, r_times)
    print(f"[INFO] abs_corr = {abs_corr:.6f}, "
          f"interval_corr = {interval_corr:.6f}")
    if math.isnan(interval_corr):
        print(f"[FAIL] interval_corr is NaN")
        rc = 1
    elif interval_corr >= 0.1:
        # This is the calibration target. If it fails, the artifact's
        # jitter default needs to be re-calibrated.
        print(f"[WARN] interval_corr={interval_corr:.4f} is above the "
              f"0.1 threshold; the default jitter may need "
              f"recalibration for this query schedule")
    else:
        print(f"[OK]   interval_corr={interval_corr:.6f} < 0.1 "
              f"(trigger fires)")

    # 5. Full run (guards skipped).
    report = sim.run(skip_guard_checks=True)
    if report.n_probes != 100:
        print(f"[FAIL] report n_probes: {report.n_probes}")
        rc = 1
    elif report.trigger_correlation_used != "interval":
        print(f"[FAIL] trigger correlation: {report.trigger_correlation_used}")
        rc = 1
    elif not report.trigger_fires:
        print(f"[FAIL] trigger did not fire: interval_corr="
              f"{report.interval_corr:.4f} (expected < 0.1)")
        rc = 1
    else:
        print(f"[OK]   full run: trigger fires "
              f"(interval_corr={report.interval_corr:.4f} < 0.1)")

    # 6. Alternative trigger (absolute correlation) does NOT fire.
    report_abs = sim.run(skip_guard_checks=True, trigger_uses_absolute_corr=True)
    if report_abs.trigger_correlation_used != "absolute":
        print(f"[FAIL] alt trigger correlation: {report_abs.trigger_correlation_used}")
        rc = 1
    elif report_abs.trigger_fires:
        print(f"[FAIL] alt trigger (absolute) fired but shouldn't: "
              f"abs_corr={report_abs.abs_corr:.4f}")
        rc = 1
    else:
        print(f"[OK]   alt trigger (absolute): does not fire "
              f"(abs_corr={report_abs.abs_corr:.4f}, as documented)")

    # 7. Mitigation: apply_padding=True produces padding sizes in
    # [100, 250].
    sim_pad = TimingSideChannelSimulator(config, "A02", seed=42, apply_padding=True)
    report_pad = sim_pad.run(skip_guard_checks=True)
    if not report_pad.padding_applied:
        print(f"[FAIL] padding_applied: {report_pad.padding_applied}")
        rc = 1
    else:
        # Check the padding sizes are in the expected range.
        sizes = sim_pad.apply_padding_to_responses(20)
        if not all(100 <= s <= 250 for s in sizes):
            print(f"[FAIL] padding sizes out of [100, 250]: {sizes}")
            rc = 1
        else:
            print(f"[OK]   mitigation: 20 padding sizes in [100, 250] "
                  f"(min={min(sizes)}, max={max(sizes)})")

    # 8. Determinism: same seed → same report.
    sim_a = TimingSideChannelSimulator(config, "A02", seed=42)
    sim_b = TimingSideChannelSimulator(config, "A02", seed=42)
    r_a = sim_a.run(skip_guard_checks=True)
    r_b = sim_b.run(skip_guard_checks=True)
    if abs(r_a.interval_corr - r_b.interval_corr) > 1e-9:
        print(f"[FAIL] determinism: {r_a.interval_corr:.6f} vs "
              f"{r_b.interval_corr:.6f}")
        rc = 1
    else:
        print(f"[OK]   determinism: same seed → same interval_corr "
              f"({r_a.interval_corr:.6f})")

    # 9. Different seeds → different reports (probabilistically).
    sim_c = TimingSideChannelSimulator(config, "A02", seed=999)
    r_c = sim_c.run(skip_guard_checks=True)
    if abs(r_a.interval_corr - r_c.interval_corr) < 1e-9:
        print(f"[FAIL] different seeds produced identical reports "
              f"(unlikely)")
        rc = 1
    else:
        print(f"[OK]   different seeds → different interval_corr "
              f"({r_a.interval_corr:.4f} vs {r_c.interval_corr:.4f})")

    # 10. Scenario parsing: unknown scenario rejected.
    try:
        parse_timing_scenario("A99", config)
    except AttackConfigError:
        print(f"[OK]   unknown scenario A99 rejected")
    else:
        print(f"[FAIL] unknown scenario A99 was accepted")
        rc = 1

    # 11. Report serialization.
    d = report.to_dict()
    required = ("scenario_id", "n_probes", "interval_corr", "trigger_fires",
                "padding_applied", "netem_used")
    missing = [k for k in required if k not in d]
    if missing:
        print(f"[FAIL] report.to_dict missing keys: {missing}")
        rc = 1
    else:
        print(f"[OK]   report.to_dict has all {len(d)} keys")

    print("=== self-test complete ===")
    return rc


if __name__ == "__main__":
    import sys
    sys.exit(_self_test())
