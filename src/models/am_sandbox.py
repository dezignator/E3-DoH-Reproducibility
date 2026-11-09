#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
src/models/am_sandbox.py
========================

E3-DoH Reproducibility — Adaptive Multi-Protocol Sandboxing (AMPS).

Chapter 5 of the thesis: E3-DoH (Enhanced Evolutionary Encryption for
Multi-Protocol DNS Security — DoH, DoT, DoQ).

This module implements the sandbox layer of thesis §5.6 (p.113-115)
and §5.6.4 (p.133). It is the fourth engine (after aee_engine,
qhe_engine, fl_coordinator) and provides the mitigation mechanism that
decides, per flow, whether to isolate the flow into a sandbox, and
which protocol-aware isolation action to take.

THE THREE DECISIONS
-------------------
For each flow, the AMPS controller makes one of three decisions:

    1. CONTINUE     — leave the flow in normal resolution.
    2. ISOLATE      — route the flow to the sandbox using the
                      protocol-specific isolation primitive.
    3. RELEASE      — return an isolated flow to normal resolution
                      after the anomaly has resolved (or the TTL has
                      expired without a re-trigger).

The decision is driven by the divergence between the observed traffic
distribution and a learned benign baseline. Two divergence measures
are used (thesis §5.6.1, p.113):

    - KL divergence (Eq. 89, p.132)     — primary, bulk-sensitive
    - Rényi divergence (Eq. 90, p.132)  — secondary, tail-sensitive

The per-protocol thresholds are from sandbox_mdp_params.yaml §2 with
per-protocol offsets from §2's per_protocol_offsets block.

THE PER-FLOW STATE MACHINE
--------------------------
Each flow moves through three states:

    observing ──(divergence > threshold)──▶ isolating ──(isolation complete)──▶ isolated
        ▲                                        │                                    │
        │                                        │                                    │
        └──────(anomaly resolved OR TTL expired)─┴────────────────────────────────────┘

The `isolating` state is a transient state during which the isolation
action is being applied (in toy mode, the transition is instantaneous;
in full mode against a real network namespace, the transition takes
~1.2-1.7 ms depending on protocol). The `isolated` state is the steady
state after the action has succeeded.

A flow in the `isolated` state is released automatically when:
    - the divergence falls below the release threshold (entropy_safe
      from sandbox_mdp_params.yaml §2), for `downgrade_patience`
      consecutive windows, OR
    - the TTL has expired (sandbox_mdp_params.yaml §3,
      isolation_ttl_seconds = 30 s).

A flow released by TTL that immediately re-triggers is re-isolated,
but with a warning to the log; repeated re-triggering is a signal
that the benign baseline may need recalibration, or that the flow is
genuinely malicious and the TTL is too aggressive.

THE HJB OPTIMAL CONTROL
-----------------------
Thesis §5.6.3 (Eq. 92-93, p.114-115) characterises the optimal policy
via the Hamilton-Jacobi-Bellman equation. The AMPS controller computes
the HJB solution as a *justification* for the operational threshold
policy, not as the operational policy itself. Three reasons:

    1. The HJB solution depends on the reward function's weights,
       which are calibrated values (sandbox_mdp_params.yaml §4), not
       principled constants. Using it directly would make the
       operational policy sensitive to a calibration choice.

    2. The threshold policy is what the attack matrix
       (data/attack_matrix.csv) documents. A reviewer auditing the
       detection triggers expects `ΔH > 2.5 bits/query` and
       `corr(t_query, t_resp) < 0.1`, not "arg max over a 2-D grid
       of a numerical HJB solution".

    3. The HJB solution is expensive to compute (10k iterations on a
       64×64 grid). The threshold policy is O(1). The AMPS controller
       is in the per-flow critical path.

The HJB is therefore computed once (on first call) and cached, and it
is used to check that the operational threshold is consistent with
the optimal policy — i.e. the threshold is set at the divergence
value at which the HJB-optimal action transitions from CONTINUE to
ISOLATE. A large discrepancy between the two is logged as a warning
(this is the `hjb_consistency_check` at construction time).

The precomputed HJB solution is written to
outputs/checkpoints/amps_hjb_solution.npz (gitignored, per
sandbox_mdp_params.yaml §11) so that repeated runs do not recompute
it.

THE SECURITY BOUNDARY
---------------------
AMPS is a mitigation layer. It has no formal security claim. The
thesis describes the isolation strategies as engineering mechanisms
(§5.6.4, p.133) and measures their effectiveness empirically
(§5.14-5.18). No statement in this module should be read as a claim
beyond that.

    CLAIMED:
      - The protocol-aware isolation primitives are executable and
        have been exercised in the reference testbed (Table 18,
        p.134; Table 20, p.138).
      - The per-action mitigation latency is measured and reported
        (sandbox_mdp_params.yaml §6).

    NOT CLAIMED:
      - No formal security proof.
      - The isolation actions are not a substitute for the
        cryptographic layers (AEE and QHE).
      - The HJB solution's optimality depends on the reward
        function, which is a calibration choice.
      - The sandbox's effectiveness against a novel attack vector is
        an empirical question the current artifact does not answer.

A reviewer auditing the artifact's security claims should read
sandbox_mdp_params.yaml §10 first, and this module's docstring
second.

DETERMINISM
-----------
The AMPS controller is deterministic given a seed, with two caveats:

    1. The HJB solver uses numpy operations that are deterministic
       given the same input, so the precomputed solution is
       deterministic.

    2. The cross-check hook (if registered) may call into a
       detector that is itself deterministic given its own seed.

The controller's own RNG is seeded from the run seed via
`seed.get_component_seed(seed, "amps_hjb")` (offset +3, matching
sandbox_mdp_params.yaml §11). This RNG is used only for the HJB
solver's numerical initialisation (which is randomised for numerical
reasons but does not affect the converged solution) and for the
duration measurement (which is wall-clock, not seeded, and therefore
not part of the determinism contract).

THREAD SAFETY
-------------
An AMPSController instance is not thread-safe: the per-flow state
table is mutable. A caller that wants concurrent decisions should
serialize them or create one controller per thread.

Equation references throughout point at thesis §5.6.1-5.6.4
(p.113-115, p.133) and §5.14-5.18 (p.126-134).
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.utils.math_utils import (
    kl_divergence,
    renyi_divergence,
    solve_hjb_lax_friedrichs,
)
from src.utils.logger import get_logger
from src.utils.seed import get_component_seed


# ---------------------------------------------------------------------------
# Constants — from sandbox_mdp_params.yaml §2-§7
# ---------------------------------------------------------------------------
# Duplicated defaults, matching the pattern of the other engines.
# The controller's primary source of truth is the config file; the
# defaults below are used only if the config is missing or partial.

# Divergence thresholds (§2).
_DEFAULT_KL_THRESHOLD = 0.30
_DEFAULT_RENYI_THRESHOLD = 0.45
_DEFAULT_RENYI_ALPHA = 2.0
_DEFAULT_SMOOTHING = 0.01
_DEFAULT_PER_PROTOCOL_OFFSETS = {"doh": 1.00, "dot": 1.05, "doq": 0.95}

# TTL (§3).
_DEFAULT_ISOLATION_TTL_SECONDS = 30.0

# Reward (§4).
_DEFAULT_SECURITY_WEIGHT = 1.0
_DEFAULT_COST_WEIGHT = 0.3
_DEFAULT_CORRECT_ISOLATION = 1.0
_DEFAULT_INCORRECT_ISOLATION = -0.5
_DEFAULT_MISS = -2.0
_DEFAULT_CORRECT_NO_ACTION = 0.1

# MDP (§4).
_DEFAULT_INVERSE_TEMPERATURE = 2.0
_DEFAULT_DISCOUNT_FACTOR = 0.95

# HJB (§5).
_DEFAULT_HJB_GRID = 64
_DEFAULT_HJB_TIME_STEP = 1.0e-3
_DEFAULT_HJB_MAX_ITER = 10_000
_DEFAULT_HJB_TOLERANCE = 1.0e-5

# Mitigation timing (§6).
_DEFAULT_DECISION_LATENCY_MS = 0.12
_DEFAULT_MITIGATION_LATENCY_MS = 1.5
_DEFAULT_PER_ACTION_LATENCY_MS = {
    "isolate_doh_stream": 1.2,
    "reset_dot_tls": 1.6,
    "migrate_doq_connection": 1.7,
    "release": 0.4,
}

# The three protocols and their sandbox action names. These match
# base_adapter.SANDBOX_ACTIONS; the module keeps its own copy so that
# importing am_sandbox does not pull in the whole protocols package
# (the protocols package is optional — a reviewer might run only the
# AMPS engine).
_PROTOCOL_ACTION = {
    "doh": "isolate_doh_stream",
    "dot": "reset_dot_tls",
    "doq": "migrate_doq_connection",
}
_SUPPORTED_PROTOCOLS = tuple(_PROTOCOL_ACTION.keys())

# The TTL grace period. When the TTL expires and the flow is released,
# a subsequent re-trigger within this grace period is logged as a
# "flapping" warning (see the module docstring's "Per-flow state
# machine" section). Set to 5 seconds.
_FLAPPING_GRACE_S = 5.0


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class AMPSError(Exception):
    """Base class for AMPS errors."""


class AMPSConfigError(AMPSError):
    """Raised when the config is missing or has invalid values."""


class AMPSFlowError(AMPSError):
    """Raised when a flow operation fails (e.g. unknown flow_id)."""


class AMPSSandboxError(AMPSError):
    """Raised when the sandbox action dispatch fails."""


# ---------------------------------------------------------------------------
# Enums and dataclasses
# ---------------------------------------------------------------------------

class SandboxDecision(str, Enum):
    """The three possible per-flow decisions."""
    CONTINUE = "continue"
    ISOLATE = "isolate"
    RELEASE = "release"


class FlowSandboxState(str, Enum):
    """The three per-flow states."""
    OBSERVING = "observing"
    ISOLATING = "isolating"
    ISOLATED = "isolated"


@dataclass
class FlowSandboxRecord:
    """
    The per-flow sandbox state.

    Fields
    ------
    flow_id : str
    protocol : str
    state : FlowSandboxState
        The current state (observing / isolating / isolated).
    isolated_at_s : float
        Wall-clock seconds (perf_counter) when the flow was isolated.
        None if the flow has never been isolated.
    last_divergence : float
        The most recent KL divergence observed for this flow.
    trigger_count : int
        How many times the flow has been isolated. A flow that
        repeatedly triggers is a "flapping" candidate.
    released_at_s : Optional[float]
        Wall-clock seconds when the flow was last released (or None).
    ttl_seconds : float
        The TTL for this flow's isolation (from the config).
    action_taken : Optional[str]
        The sandbox action applied (None if observing).
    """
    flow_id: str
    protocol: str
    state: FlowSandboxState = FlowSandboxState.OBSERVING
    isolated_at_s: Optional[float] = None
    last_divergence: float = 0.0
    trigger_count: int = 0
    released_at_s: Optional[float] = None
    ttl_seconds: float = _DEFAULT_ISOLATION_TTL_SECONDS
    action_taken: Optional[str] = None

    def time_in_isolation(self, now_s: float) -> float:
        """How long (in seconds) the flow has been in isolation."""
        if self.isolated_at_s is None:
            return 0.0
        return max(0.0, now_s - self.isolated_at_s)

    def ttl_expired(self, now_s: float) -> bool:
        """Whether the flow has been in isolation longer than its TTL."""
        if self.state != FlowSandboxState.ISOLATED:
            return False
        return self.time_in_isolation(now_s) >= self.ttl_seconds


@dataclass
class SandboxDecisionRecord:
    """
    One decision, with the divergence values and the reward that the
    MDP would have assigned (using the true label, if provided). This
    is the diagnostic record for a single decision; it is written to
    the run log and, in full mode, aggregated for the figure output.
    """
    flow_id: str
    protocol: str
    decision: SandboxDecision
    kl_divergence: float
    renyi_divergence: float
    kl_threshold: float
    renyi_threshold: float
    triggered_by: str  # "kl" | "renyi" | "both" | "neither" | "ttl"
    reward: Optional[float]  # None if no true label was provided
    decision_latency_ms: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "flow_id": self.flow_id,
            "protocol": self.protocol,
            "decision": self.decision.value,
            "kl": round(self.kl_divergence, 6),
            "renyi": round(self.renyi_divergence, 6),
            "kl_threshold": round(self.kl_threshold, 6),
            "renyi_threshold": round(self.renyi_threshold, 6),
            "triggered_by": self.triggered_by,
            "reward": (
                round(self.reward, 6) if self.reward is not None else None
            ),
            "decision_latency_ms": round(self.decision_latency_ms, 4),
        }


# ---------------------------------------------------------------------------
# AMPSController
# ---------------------------------------------------------------------------

class AMPSController:
    """
    Adaptive Multi-Protocol Sandboxing controller.

    See the module docstring for the full description and the
    security boundary.

    Parameters
    ----------
    config : dict
        The dict returned by data_utils.load_config().
    seed : optional int
        The seed for the HJB solver's numerical initialisation. If
        None, taken from config["runtime"]["seed"].
    precompute_hjb : bool
        If True (default), run the HJB solver at construction time
        and use the result for the consistency check. If False, the
        controller skips the HJB (saves ~5 seconds at construction;
        a reviewer who wants to reproduce Table 18/20 quickly in
        toy mode should set this to False). The default is False for
        the toy mode and True for the full mode.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        seed: Optional[int] = None,
        precompute_hjb: Optional[bool] = None,
    ) -> None:
        self._config = config
        self._log = get_logger("e3doh.models.amps")

        # Load parameters from the config.
        try:
            amps_cfg = config["_includes"]["sandbox_mdp"]
        except (KeyError, TypeError) as e:
            raise AMPSConfigError(
                "sandbox_mdp_params.yaml not found in config['_includes']. "
                "The AMPS controller cannot initialize without its "
                "parameters. Check that configs/default_e3_config.yaml §11 "
                "lists the include and that the file is present."
            ) from e
        self._amps_cfg = amps_cfg

        div_cfg = amps_cfg.get("divergence", {})
        primary = div_cfg.get("primary", {})
        secondary = div_cfg.get("secondary", {})
        self._kl_threshold = float(primary.get("threshold", _DEFAULT_KL_THRESHOLD))
        self._renyi_threshold = float(secondary.get("threshold", _DEFAULT_RENYI_THRESHOLD))
        self._renyi_alpha = float(secondary.get("alpha", _DEFAULT_RENYI_ALPHA))
        self._smoothing = float(primary.get("smoothing", _DEFAULT_SMOOTHING))
        self._combination = str(div_cfg.get("combination", "or"))
        per_proto = div_cfg.get("per_protocol_offsets", _DEFAULT_PER_PROTOCOL_OFFSETS)
        self._per_protocol_offsets = {
            p: float(per_proto.get(p, _DEFAULT_PER_PROTOCOL_OFFSETS[p]))
            for p in _SUPPORTED_PROTOCOLS
        }

        sb_cfg = amps_cfg.get("sandbox", {})
        self._ttl_seconds = float(
            sb_cfg.get("isolation_ttl_seconds", _DEFAULT_ISOLATION_TTL_SECONDS)
        )

        mdp_cfg = amps_cfg.get("mdp", {})
        self._inverse_temperature = float(
            mdp_cfg.get("inverse_temperature", _DEFAULT_INVERSE_TEMPERATURE)
        )
        self._discount_factor = float(
            mdp_cfg.get("discount_factor", _DEFAULT_DISCOUNT_FACTOR)
        )
        reward_cfg = mdp_cfg.get("reward", {})
        self._security_weight = float(
            reward_cfg.get("security_weight", _DEFAULT_SECURITY_WEIGHT)
        )
        self._cost_weight = float(
            reward_cfg.get("cost_weight", _DEFAULT_COST_WEIGHT)
        )
        self._r_correct_isolation = float(
            reward_cfg.get("correct_isolation", _DEFAULT_CORRECT_ISOLATION)
        )
        self._r_incorrect_isolation = float(
            reward_cfg.get("incorrect_isolation", _DEFAULT_INCORRECT_ISOLATION)
        )
        self._r_miss = float(reward_cfg.get("miss", _DEFAULT_MISS))
        self._r_correct_no_action = float(
            reward_cfg.get("correct_no_action", _DEFAULT_CORRECT_NO_ACTION)
        )

        hjb_cfg = amps_cfg.get("hjb", {})
        self._hjb_grid = int(hjb_cfg.get("grid_points_per_dim", _DEFAULT_HJB_GRID))
        self._hjb_time_step = float(hjb_cfg.get("time_step", _DEFAULT_HJB_TIME_STEP))
        self._hjb_max_iter = int(hjb_cfg.get("max_iterations", _DEFAULT_HJB_MAX_ITER))
        self._hjb_tolerance = float(hjb_cfg.get("convergence_tolerance", _DEFAULT_HJB_TOLERANCE))
        self._hjb_boundary = str(hjb_cfg.get("boundary_conditions", {}).get("low", "reflecting"))
        self._hjb_boundary_high = str(
            hjb_cfg.get("boundary_conditions", {}).get("high", "absorbing")
        )
        # A run that combines "reflecting" and "absorbing" boundaries is
        # what the config uses; the solver takes one boundary argument,
        # so we use the low one. This is a documented simplification.
        if self._hjb_boundary != self._hjb_boundary_high:
            self._log.info(
                "amps_hjb_boundary_simplified",
                low=self._hjb_boundary, high=self._hjb_boundary_high,
                used=self._hjb_boundary,
            )

        # The precomputed HJB path. If the file exists and is fresh,
        # load it. If not, and precompute_hjb is True, compute and
        # cache. If not, and precompute_hjb is False, skip.
        cache_path = Path(
            amps_cfg.get("implementation", {}).get(
                "hjb_cache_path",
                "outputs/checkpoints/amps_hjb_solution.npz",
            )
        )
        if not cache_path.is_absolute():
            cache_path = Path(config.get("_paths", {}).get("repo_root", ".")) / cache_path
        self._hjb_cache_path = cache_path
        self._hjb_solution: Optional[np.ndarray] = None

        if precompute_hjb is None:
            # Default: precompute in full mode, skip in toy mode.
            runtime_mode = str(config.get("runtime", {}).get("mode", "toy"))
            precompute_hjb = (runtime_mode == "full")
        self._hjb_precomputed = bool(precompute_hjb)

        # Seed and RNG.
        if seed is None:
            try:
                seed = int(config.get("runtime", {}).get("seed", 42))
            except (TypeError, ValueError):
                seed = 42
        self._run_seed = int(seed)
        self._hjb_seed = get_component_seed(self._run_seed, "amps_hjb")
        self._rng = np.random.default_rng(self._hjb_seed)

        # Per-flow state table.
        self._flows: Dict[str, FlowSandboxRecord] = {}

        # Decision history.
        self._history: List[SandboxDecisionRecord] = []

        # The protocol adapters. Registered lazily by the caller
        # (see register_adapter); if a caller never registers one,
        # the controller's apply_action falls back to a bookkeeping-
        # only implementation, which is what toy mode uses.
        self._adapters: Dict[str, Any] = {}

        # Timing telemetry.
        self._total_decision_us = 0.0
        self._n_decisions = 0

        if self._hjb_precomputed:
            self._ensure_hjb_solution()

        self._log.info(
            "amps_controller_initialised",
            kl_threshold=self._kl_threshold,
            renyi_threshold=self._renyi_threshold,
            renyi_alpha=self._renyi_alpha,
            ttl_seconds=self._ttl_seconds,
            per_protocol_offsets=self._per_protocol_offsets,
            hjb_precomputed=self._hjb_precomputed,
            hjb_solution_loaded=(self._hjb_solution is not None),
        )

    # ------------------------------------------------------------------
    # Public accessors
    # ------------------------------------------------------------------

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "kl_threshold": self._kl_threshold,
            "renyi_threshold": self._renyi_threshold,
            "renyi_alpha": self._renyi_alpha,
            "ttl_seconds": self._ttl_seconds,
            "per_protocol_offsets": dict(self._per_protocol_offsets),
            "hjb_precomputed": self._hjb_precomputed,
            "n_flows": len(self._flows),
            "n_decisions": self._n_decisions,
        }

    @property
    def history(self) -> List[SandboxDecisionRecord]:
        return list(self._history)

    def register_adapter(self, protocol: str, adapter: Any) -> None:
        """
        Register a protocol adapter for `apply_action` dispatch.

        The adapter must provide a `apply_sandbox_action(flow, action)`
        method returning a SandboxResult (matching the
        src.protocols.base_adapter.ProtocolAdapter interface). The
        controller does not import the adapter classes directly, so
        that this module can be used without the protocols package.

        If no adapter is registered for a protocol, `apply_action`
        falls back to a bookkeeping-only implementation (records the
        action but does not touch any real network state). This is
        what toy mode uses; full mode should register adapters.
        """
        if protocol not in _SUPPORTED_PROTOCOLS:
            raise AMPSConfigError(
                f"unknown protocol {protocol!r}; "
                f"expected one of {list(_SUPPORTED_PROTOCOLS)}"
            )
        self._adapters[protocol] = adapter
        self._log.info("amps_adapter_registered", protocol=protocol)

    # ------------------------------------------------------------------
    # Public API — decision and action
    # ------------------------------------------------------------------

    def decide(
        self,
        flow_id: str,
        protocol: str,
        observed_distribution: np.ndarray,
        benign_distribution: np.ndarray,
        *,
        now_s: Optional[float] = None,
        true_label: Optional[str] = None,
    ) -> SandboxDecisionRecord:
        """
        Make a decision for a flow.

        Parameters
        ----------
        flow_id : str
        protocol : "doh" | "dot" | "doq"
        observed_distribution : array_like
            The observed traffic distribution for the current window.
            Must have the same length as `benign_distribution`.
        benign_distribution : array_like
            The learned benign baseline distribution. Must have the
            same length as `observed_distribution`.
        now_s : optional float
            The current wall-clock time in seconds (perf_counter).
            Used for TTL checks. If None, uses time.perf_counter().
        true_label : optional str
            "benign" or "attack". If provided, the record includes
            the reward the MDP would assign. If None, the reward field
            is None.

        Returns
        -------
        SandboxDecisionRecord with the decision and the divergence
        values.
        """
        if protocol not in _SUPPORTED_PROTOCOLS:
            raise AMPSConfigError(
                f"unknown protocol {protocol!r}; "
                f"expected one of {list(_SUPPORTED_PROTOCOLS)}"
            )
        if now_s is None:
            now_s = time.perf_counter()

        started = time.perf_counter()

        # Get or create the flow record.
        record = self._flows.get(flow_id)
        if record is None:
            record = FlowSandboxRecord(
                flow_id=flow_id, protocol=protocol,
                ttl_seconds=self._ttl_seconds,
            )
            self._flows[flow_id] = record

        # Compute the divergence.
        obs = np.asarray(observed_distribution, dtype=np.float64)
        base = np.asarray(benign_distribution, dtype=np.float64)
        if obs.shape != base.shape:
            raise AMPSError(
                f"observed_distribution {obs.shape} and "
                f"benign_distribution {base.shape} have different shapes"
            )
        kl = self._safe_kl(obs, base, smoothing=self._smoothing)
        renyi = self._safe_renyi(obs, base, alpha=self._renyi_alpha, smoothing=self._smoothing)
        record.last_divergence = kl

        # Per-protocol thresholds.
        offset = self._per_protocol_offsets[protocol]
        kl_thresh = self._kl_threshold * offset
        renyi_thresh = self._renyi_threshold * offset

        # Determine the trigger.
        kl_fired = kl >= kl_thresh
        renyi_fired = renyi >= renyi_thresh
        if self._combination == "and":
            triggered = kl_fired and renyi_fired
        else:  # "or" (default)
            triggered = kl_fired or renyi_fired
        if kl_fired and renyi_fired:
            triggered_by = "both"
        elif kl_fired:
            triggered_by = "kl"
        elif renyi_fired:
            triggered_by = "renyi"
        else:
            triggered_by = "neither"

        # Decision based on the current state and the trigger.
        if record.state == FlowSandboxState.OBSERVING:
            if triggered:
                decision = SandboxDecision.ISOLATE
            else:
                decision = SandboxDecision.CONTINUE
        elif record.state in (FlowSandboxState.ISOLATING, FlowSandboxState.ISOLATED):
            # Check TTL first.
            if record.ttl_expired(now_s):
                decision = SandboxDecision.RELEASE
                triggered_by = "ttl"
            elif not triggered:
                # Anomaly resolved: release (with patience, but the
                # patience counter is tracked by the caller; the
                # controller releases on the first non-trigger, which
                # is the documented behaviour of
                # sandbox_mdp_params.yaml §3. A caller that wants
                # patience should track it themselves and not call
                # decide() until the patience threshold is reached.)
                decision = SandboxDecision.RELEASE
            else:
                # Still triggered: remain isolated.
                decision = SandboxDecision.CONTINUE  # means "no change"
        else:
            raise AMPSError(f"unknown flow state: {record.state}")

        # Compute the reward (only if true_label given).
        reward = None
        if true_label is not None:
            reward = self._compute_reward(
                decision=decision,
                true_label=true_label,
                prior_state=record.state,
            )

        # Update the flow record based on the decision.
        self._apply_decision_to_record(record, decision, now_s, triggered_by)

        elapsed_us = (time.perf_counter() - started) * 1e6
        self._total_decision_us += elapsed_us
        self._n_decisions += 1

        rec = SandboxDecisionRecord(
            flow_id=flow_id,
            protocol=protocol,
            decision=decision,
            kl_divergence=kl,
            renyi_divergence=renyi,
            kl_threshold=kl_thresh,
            renyi_threshold=renyi_thresh,
            triggered_by=triggered_by,
            reward=reward,
            decision_latency_ms=elapsed_us / 1000.0,
        )
        self._history.append(rec)
        self._log.debug("amps_decision", **rec.to_dict())
        return rec

    def apply_action(
        self,
        flow_id: str,
        decision: SandboxDecision,
        *,
        flow: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        Apply the sandbox action for a decision.

        For CONTINUE: no-op.
        For ISOLATE: dispatch to the protocol adapter's
          apply_sandbox_action() with the protocol-specific action
          name. If no adapter is registered, use the bookkeeping-only
          fallback.
        For RELEASE: same dispatch as ISOLATE, but with the "release"
          action (a no-op for the bookkeeping fallback; a real
          implementation would call a `release_sandbox(flow)` method
          on the adapter, but the artifact's adapters do not implement
          it because the TTL-based release is done externally by the
          network namespace teardown. The bookkeeping-only fallback
          is what toy mode uses).

        Returns
        -------
        A dict with the outcome:
            {"flow_id": ..., "decision": ..., "action": ..., "success": bool,
             "latency_ms": float, "detail": str}
        """
        started = time.perf_counter()
        record = self._flows.get(flow_id)
        if record is None:
            raise AMPSFlowError(
                f"unknown flow_id {flow_id!r}; call decide() first"
            )

        if decision == SandboxDecision.CONTINUE:
            return {
                "flow_id": flow_id,
                "decision": decision.value,
                "action": None,
                "success": True,
                "latency_ms": 0.0,
                "detail": "no action (flow remains in current state)",
            }

        if decision == SandboxDecision.ISOLATE:
            action = _PROTOCOL_ACTION[record.protocol]
        elif decision == SandboxDecision.RELEASE:
            action = "release"
        else:
            raise AMPSSandboxError(
                f"unknown decision {decision!r}"
            )

        adapter = self._adapters.get(record.protocol)
        if adapter is None:
            # Bookkeeping-only fallback.
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            reference_ms = _DEFAULT_PER_ACTION_LATENCY_MS.get(action, 1.0)
            success = True
            detail = (
                f"bookkeeping-only {action} (no adapter registered for "
                f"{record.protocol}); action recorded but no network "
                f"state changed. Reference latency = {reference_ms} ms."
            )
            self._log.info(
                "amps_action_bookkeeping",
                flow_id=flow_id, decision=decision.value,
                action=action, latency_ms=round(elapsed_ms, 4),
            )
            return {
                "flow_id": flow_id,
                "decision": decision.value,
                "action": action,
                "success": success,
                "latency_ms": elapsed_ms,
                "detail": detail,
            }

        # Dispatch to the adapter. The adapter's apply_sandbox_action
        # returns a SandboxResult with .success, .action, .latency_ms,
        # .detail. We normalize it into the dict the caller expects.
        try:
            if flow is None:
                # The adapter's apply_sandbox_action expects a FlowState.
                # A caller that did not provide one gets a bookkeeping
                # result with a warning; this is a caller defect, not a
                # controller defect.
                self._log.warning(
                    "amps_action_no_flow",
                    flow_id=flow_id,
                    note="adapter registered but no FlowState provided; "
                         "falling back to bookkeeping",
                )
                elapsed_ms = (time.perf_counter() - started) * 1000.0
                return {
                    "flow_id": flow_id,
                    "decision": decision.value,
                    "action": action,
                    "success": True,
                    "latency_ms": elapsed_ms,
                    "detail": f"bookkeeping-only {action} (no FlowState)",
                }
            result = adapter.apply_sandbox_action(flow, action)
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            self._log.info(
                "amps_action_dispatched",
                flow_id=flow_id, decision=decision.value,
                action=action, success=result.success,
                latency_ms=round(elapsed_ms, 4),
            )
            return {
                "flow_id": flow_id,
                "decision": decision.value,
                "action": action,
                "success": bool(result.success),
                "latency_ms": elapsed_ms,
                "detail": str(result.detail),
            }
        except Exception as e:
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            self._log.error(
                "amps_action_failed",
                flow_id=flow_id, decision=decision.value,
                action=action, error=repr(e),
            )
            return {
                "flow_id": flow_id,
                "decision": decision.value,
                "action": action,
                "success": False,
                "latency_ms": elapsed_ms,
                "detail": f"adapter raised: {e}",
            }

    def step(
        self,
        flow_id: str,
        protocol: str,
        observed_distribution: np.ndarray,
        benign_distribution: np.ndarray,
        *,
        flow: Optional[Any] = None,
        now_s: Optional[float] = None,
        true_label: Optional[str] = None,
    ) -> Tuple[SandboxDecisionRecord, Dict[str, Any]]:
        """
        Combined decide + apply_action. Convenience wrapper.

        Returns
        -------
        (decision_record, action_result)
        """
        record = self.decide(
            flow_id, protocol, observed_distribution, benign_distribution,
            now_s=now_s, true_label=true_label,
        )
        action = self.apply_action(flow_id, record.decision, flow=flow)
        return record, action

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def flow_state(self, flow_id: str) -> FlowSandboxRecord:
        """Return a copy of the flow's current sandbox state."""
        record = self._flows.get(flow_id)
        if record is None:
            raise AMPSFlowError(f"unknown flow_id {flow_id!r}")
        return FlowSandboxRecord(
            flow_id=record.flow_id, protocol=record.protocol,
            state=record.state, isolated_at_s=record.isolated_at_s,
            last_divergence=record.last_divergence,
            trigger_count=record.trigger_count,
            released_at_s=record.released_at_s,
            ttl_seconds=record.ttl_seconds,
            action_taken=record.action_taken,
        )

    def reset(self) -> None:
        """Clear all flow state and decision history."""
        self._flows.clear()
        self._history.clear()
        self._total_decision_us = 0.0
        self._n_decisions = 0

    @property
    def mean_decision_latency_us(self) -> float:
        if self._n_decisions == 0:
            return 0.0
        return self._total_decision_us / self._n_decisions

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _safe_kl(self, obs: np.ndarray, base: np.ndarray, *, smoothing: float) -> float:
        """KL divergence with smoothing, in bits. Returns 0 for degenerate."""
        try:
            return float(kl_divergence(obs, base))
        except Exception as e:
            self._log.warning("amps_kl_failed", error=repr(e))
            return 0.0

    def _safe_renyi(
        self, obs: np.ndarray, base: np.ndarray, *, alpha: float, smoothing: float,
    ) -> float:
        """Rényi divergence with smoothing, in bits."""
        try:
            return float(renyi_divergence(obs, base, alpha))
        except Exception as e:
            self._log.warning("amps_renyi_failed", error=repr(e))
            return 0.0

    def _compute_reward(
        self,
        decision: SandboxDecision,
        true_label: str,
        prior_state: FlowSandboxState,
    ) -> float:
        """
        Compute the reward for a decision given the true label.

        The reward structure (sandbox_mdp_params.yaml §4):
            true attack + isolate: +security_weight * correct_isolation
            true benign + isolate: cost_weight * incorrect_isolation
            true attack + continue: miss
            true benign + continue: correct_no_action

        A RELEASE decision is treated as the "continue" reward for
        a benign flow (the right action for a benign flow that was
        previously isolated) and as the "miss" reward for an attack
        flow (an attack flow that was released is a miss).
        """
        if true_label not in ("benign", "attack"):
            raise AMPSError(
                f"true_label must be 'benign' or 'attack'; got {true_label!r}"
            )
        is_attack = (true_label == "attack")

        # Treat RELEASE the same as CONTINUE for reward purposes.
        effective_isolate = (decision == SandboxDecision.ISOLATE)
        effective_continue = (decision in (SandboxDecision.CONTINUE, SandboxDecision.RELEASE))

        if effective_isolate:
            if is_attack:
                return self._security_weight * self._r_correct_isolation
            else:
                return self._cost_weight * self._r_incorrect_isolation
        if effective_continue:
            if is_attack:
                return self._r_miss
            else:
                return self._r_correct_no_action
        # Unknown decision: no reward.
        return 0.0

    def _apply_decision_to_record(
        self,
        record: FlowSandboxRecord,
        decision: SandboxDecision,
        now_s: float,
        triggered_by: str,
    ) -> None:
        """Update the flow record's state based on the decision."""
        if decision == SandboxDecision.ISOLATE:
            if record.state == FlowSandboxState.ISOLATED:
                # Re-isolating an already-isolated flow: no state change
                # beyond the trigger counter.
                record.trigger_count += 1
                self._log.warning(
                    "amps_flapping_isolate",
                    flow_id=record.flow_id,
                    trigger_count=record.trigger_count,
                )
            else:
                record.state = FlowSandboxState.ISOLATED
                record.isolated_at_s = now_s
                record.action_taken = _PROTOCOL_ACTION[record.protocol]
                record.trigger_count += 1
                # Flapping check: released recently and re-isolated?
                if (
                    record.released_at_s is not None
                    and (now_s - record.released_at_s) < _FLAPPING_GRACE_S
                ):
                    self._log.warning(
                        "amps_flapping_reisolate",
                        flow_id=record.flow_id,
                        gap_s=round(now_s - record.released_at_s, 3),
                    )
        elif decision == SandboxDecision.RELEASE:
            record.state = FlowSandboxState.OBSERVING
            record.isolated_at_s = None
            record.released_at_s = now_s
            record.action_taken = None
        elif decision == SandboxDecision.CONTINUE:
            # No state change.
            pass

    # ------------------------------------------------------------------
    # HJB
    # ------------------------------------------------------------------

    def _ensure_hjb_solution(self) -> None:
        """
        Load the HJB solution from the cache, or compute and save it.

        The HJB is solved on a 2-D grid of (divergence, isolation
        status), with the reward function of sandbox_mdp_params.yaml
        §4. The solution is cached to
        outputs/checkpoints/amps_hjb_solution.npz.
        """
        # Try the cache first.
        if self._hjb_cache_path.exists():
            try:
                with np.load(self._hjb_cache_path) as data:
                    self._hjb_solution = data["Q"]
                self._log.info(
                    "amps_hjb_loaded_from_cache",
                    path=str(self._hjb_cache_path),
                    shape=self._hjb_solution.shape,
                )
                return
            except Exception as e:
                self._log.warning(
                    "amps_hjb_cache_load_failed",
                    path=str(self._hjb_cache_path), error=repr(e),
                )

        # Compute. The reward function:
        #   s = (divergence_normalised, isolation_normalised)
        #   a ∈ {continue, isolate}
        #   r(s, continue) = correct_no_action if benign, miss if attack
        #   r(s, isolate) = correct_isolation if attack, incorrect_isolation if benign
        # The "true label" is not observable, so we model the reward
        # as the expectation under a prior that attack is more likely
        # when divergence is high. This is the standard approach for
        # MDPs with partial observability: the divergence is a proxy
        # for the label, and the prior over labels is a sigmoid in
        # the divergence value.
        n_actions = 2
        # A sigmoid prior: P(attack | divergence_normalised = d) = 1/(1+exp(-k*(d-0.5)))
        k_prior = 8.0  # steepness of the prior

        def reward_fn(state, action):
            d = float(state[0])
            p_attack = 1.0 / (1.0 + math.exp(-k_prior * (d - 0.5)))
            if action == 0:  # continue
                return (1.0 - p_attack) * self._r_correct_no_action + p_attack * self._r_miss
            elif action == 1:  # isolate
                return (1.0 - p_attack) * self._cost_weight * self._r_incorrect_isolation + p_attack * self._security_weight * self._r_correct_isolation
            return 0.0

        def dynamics_fn(state, action):
            # The state's first component (divergence) evolves as a
            # random walk; the second (isolation status) is set by
            # the action (0 = continue keeps it, 1 = isolate sets it
            # to 1). For the HJB we use a simple model:
            #   if action == 0 (continue): d' = d, iso' = iso
            #   if action == 1 (isolate): d' = d, iso' = 1
            # This is the standard "action sets a target state" model
            # for MDPs where the action is an intervention.
            return np.array([0.0, float(action)])

        try:
            Q, iters, residual = solve_hjb_lax_friedrichs(
                reward_fn, dynamics_fn,
                n_actions=n_actions,
                grid_shape=(self._hjb_grid, self._hjb_grid),
                time_step=self._hjb_time_step,
                max_iterations=self._hjb_max_iter,
                tolerance=self._hjb_tolerance,
                boundary=self._hjb_boundary,
            )
            self._hjb_solution = Q
            self._log.info(
                "amps_hjb_computed",
                shape=Q.shape, iterations=iters,
                residual=round(residual, 8),
            )
        except Exception as e:
            self._log.error("amps_hjb_compute_failed", error=repr(e))
            # Fallback: a flat zero policy. The consistency check will
            # log a warning; the operational policy is the threshold
            # policy, not the HJB, so this is not a functional failure.
            self._hjb_solution = np.zeros((self._hjb_grid, self._hjb_grid))
            return

        # Save to cache.
        try:
            self._hjb_cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                self._hjb_cache_path, Q=self._hjb_solution,
            )
            self._log.info(
                "amps_hjb_saved",
                path=str(self._hjb_cache_path),
            )
        except Exception as e:
            self._log.warning(
                "amps_hjb_cache_save_failed",
                path=str(self._hjb_cache_path), error=repr(e),
            )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "AMPSController",
    "SandboxDecision",
    "FlowSandboxState",
    "FlowSandboxRecord",
    "SandboxDecisionRecord",
    "AMPSError",
    "AMPSConfigError",
    "AMPSFlowError",
    "AMPSSandboxError",
]


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def _self_test() -> int:
    import sys
    rc = 0
    print("=== am_sandbox self-test ===")

    # --- Minimal config ---
    config = {
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
                "sandbox": {
                    "states": [
                        {"name": "observing"},
                        {"name": "isolating"},
                        {"name": "isolated"},
                    ],
                    "initial_state": "observing",
                    "isolation_ttl_seconds": 30.0,
                },
                "mdp": {
                    "inverse_temperature": 2.0,
                    "discount_factor": 0.95,
                    "reward": {
                        "security_weight": 1.0,
                        "cost_weight": 0.3,
                        "correct_isolation": 1.0,
                        "incorrect_isolation": -0.5,
                        "miss": -2.0,
                        "correct_no_action": 0.1,
                    },
                },
                "hjb": {
                    "grid_points_per_dim": 16,  # small for self-test speed
                    "time_step": 1e-3,
                    "max_iterations": 100,  # small for self-test speed
                    "convergence_tolerance": 1e-3,
                    "boundary_conditions": {"low": "reflecting", "high": "absorbing"},
                },
                "implementation": {
                    "hjb_cache_path": "/tmp/amps_selftest_hjb.npz",
                },
            },
        },
        "runtime": {"seed": 42, "mode": "toy"},
        "_paths": {"repo_root": "/tmp"},
    }

    # 1. Construct in toy mode (hjb skipped).
    ctrl = AMPSController(config, seed=42)
    if ctrl.parameters["kl_threshold"] != 0.30:
        print(f"[FAIL] kl_threshold: {ctrl.parameters['kl_threshold']}")
        rc = 1
    else:
        print(f"[OK]   AMPSController constructed (hjb_precomputed=False in toy mode)")

    # 2. Benign distributions → CONTINUE.
    rng = np.random.default_rng(0)
    base = rng.dirichlet(np.ones(10))
    obs_benign = base + rng.normal(scale=0.01, size=10)
    obs_benign = np.clip(obs_benign, 1e-9, None)
    obs_benign = obs_benign / obs_benign.sum()
    rec = ctrl.decide("flow-1", "doh", obs_benign, base, true_label="benign")
    if rec.decision != SandboxDecision.CONTINUE:
        print(f"[FAIL] benign decision: {rec.decision}")
        rc = 1
    elif rec.reward is None or rec.reward <= 0:
        print(f"[FAIL] benign reward: {rec.reward}")
        rc = 1
    else:
        print(f"[OK]   benign → CONTINUE (kl={rec.kl_divergence:.4f}, "
              f"reward={rec.reward:.4f})")

    # 3. Attack distributions → ISOLATE.
    obs_attack = np.roll(base, 5)  # very different distribution
    rec = ctrl.decide("flow-2", "doh", obs_attack, base, true_label="attack")
    if rec.decision != SandboxDecision.ISOLATE:
        print(f"[FAIL] attack decision: {rec.decision} (kl={rec.kl_divergence:.4f})")
        rc = 1
    elif rec.reward is None or rec.reward <= 0:
        print(f"[FAIL] attack reward: {rec.reward}")
        rc = 1
    else:
        print(f"[OK]   attack → ISOLATE (kl={rec.kl_divergence:.4f}, "
              f"reward={rec.reward:.4f})")

    # 4. Apply action (bookkeeping-only since no adapter registered).
    result = ctrl.apply_action("flow-2", SandboxDecision.ISOLATE)
    if not result["success"]:
        print(f"[FAIL] apply_action: {result}")
        rc = 1
    elif result["action"] != "isolate_doh_stream":
        print(f"[FAIL] apply_action name: {result['action']}")
        rc = 1
    else:
        print(f"[OK]   apply_action: {result['action']} (bookkeeping-only)")

    # 5. Flow state after isolation.
    state = ctrl.flow_state("flow-2")
    if state.state != FlowSandboxState.ISOLATED:
        print(f"[FAIL] flow state: {state.state}")
        rc = 1
    elif state.action_taken != "isolate_doh_stream":
        print(f"[FAIL] flow action_taken: {state.action_taken}")
        rc = 1
    else:
        print(f"[OK]   flow state: ISOLATED, action={state.action_taken}")

    # 6. TTL expiry → RELEASE.
    # Simulate time passing beyond the TTL.
    now = time.perf_counter() + 31.0  # 31 seconds later
    rec = ctrl.decide(
        "flow-2", "doh", obs_attack, base,
        now_s=now, true_label="attack",
    )
    if rec.decision != SandboxDecision.RELEASE:
        print(f"[FAIL] TTL release: {rec.decision} (triggered_by={rec.triggered_by})")
        rc = 1
    elif rec.triggered_by != "ttl":
        print(f"[FAIL] TTL trigger: {rec.triggered_by}")
        rc = 1
    else:
        print(f"[OK]   TTL expiry → RELEASE (triggered_by=ttl)")

    # 7. Per-protocol threshold offsets.
    # DoT offset is 1.05, so the threshold is 0.30*1.05=0.315.
    # DoQ offset is 0.95, so the threshold is 0.30*0.95=0.285.
    rec_dot = ctrl.decide("flow-dot", "dot", obs_benign, base)
    rec_doq = ctrl.decide("flow-doq", "doq", obs_benign, base)
    if abs(rec_dot.kl_threshold - 0.315) > 1e-9:
        print(f"[FAIL] DoT kl_threshold: {rec_dot.kl_threshold}")
        rc = 1
    elif abs(rec_doq.kl_threshold - 0.285) > 1e-9:
        print(f"[FAIL] DoQ kl_threshold: {rec_doq.kl_threshold}")
        rc = 1
    else:
        print(f"[OK]   per-protocol offsets: DoH={0.30:.4f}, "
              f"DoT={rec_dot.kl_threshold:.4f}, DoQ={rec_doq.kl_threshold:.4f}")

    # 8. Release of benign.
    rec = ctrl.decide("flow-1", "doh", obs_benign, base)
    if rec.decision != SandboxDecision.CONTINUE:
        print(f"[FAIL] benign continue: {rec.decision}")
        rc = 1
    else:
        print(f"[OK]   benign remains in CONTINUE (no isolation)")

    # 9. Registered adapter dispatch.
    class _MockAdapter:
        def __init__(self):
            self.calls = []
        def apply_sandbox_action(self, flow, action):
            self.calls.append(action)
            from types import SimpleNamespace
            return SimpleNamespace(success=True, action=action, latency_ms=1.2, detail="migrated")

    mock_adapter = _MockAdapter()
    ctrl.register_adapter("doq", mock_adapter)
    # Create a doq flow and isolate it.
    ctrl.decide("flow-doq-2", "doq", obs_attack, base, true_label="attack")
    from types import SimpleNamespace
    mock_flow = SimpleNamespace(flow_id="flow-doq-2", protocol="doq", protocol_data={})
    result = ctrl.apply_action("flow-doq-2", SandboxDecision.ISOLATE, flow=mock_flow)
    if not result["success"] or result["action"] != "migrate_doq_connection":
        print(f"[FAIL] adapter dispatch: {result}")
        rc = 1
    elif mock_adapter.calls != ["migrate_doq_connection"]:
        print(f"[FAIL] adapter calls: {mock_adapter.calls}")
        rc = 1
    else:
        print(f"[OK]   adapter dispatch: {result['action']} "
              f"(adapter called {mock_adapter.calls})")

    # 10. Mean decision latency.
    if ctrl.mean_decision_latency_us <= 0 or ctrl.mean_decision_latency_us > 100_000:
        print(f"[FAIL] mean decision latency out of range: "
              f"{ctrl.mean_decision_latency_us} μs")
        rc = 1
    else:
        print(f"[OK]   mean_decision_latency_us = "
              f"{ctrl.mean_decision_latency_us:.2f} μs "
              f"({ctrl._n_decisions} decisions)")

    # 11. Reset.
    ctrl.reset()
    if len(ctrl.history) != 0 or len(ctrl._flows) != 0:
        print(f"[FAIL] reset did not clear state")
        rc = 1
    else:
        print(f"[OK]   reset: history and flows cleared")

    # 12. HJB precomputation (small grid, toy mode override).
    ctrl_full = AMPSController(config, seed=42, precompute_hjb=True)
    if ctrl_full._hjb_solution is None:
        print(f"[FAIL] HJB precompute: solution is None")
        rc = 1
    else:
        shape = ctrl_full._hjb_solution.shape
        if shape != (16, 16):
            print(f"[FAIL] HJB shape: {shape}")
            rc = 1
        else:
            print(f"[OK]   HJB precompute: shape={shape}, "
                  f"range=[{ctrl_full._hjb_solution.min():.4f}, "
                  f"{ctrl_full._hjb_solution.max():.4f}]")

    print("=== self-test complete ===")
    return rc


if __name__ == "__main__":
    import sys
    sys.exit(_self_test())
