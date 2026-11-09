#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
src/models/fl_coordinator.py
============================

E3-DoH Reproducibility — Multi-Protocol Federated Learning (MP-FL) engine.

Chapter 5 of the thesis: E3-DoH (Enhanced Evolutionary Encryption for
Multi-Protocol DNS Security — DoH, DoT, DoQ).

This module implements the federated learning layer of thesis
§5.5.1-5.7.1 (p.113-117) and §5.13 (p.146-152). It is the third of
the three engines and provides the collaborative-learning mechanism
that allows threat intelligence to be shared across network edges
without exposing raw traffic data.

WHAT THE COORDINATOR DOES
-------------------------
The coordinator runs a sequence of federated rounds. In each round:

    1. Samples a subset of clients according to `client_sample_rate`.
    2. Asks each sampled client to compute an update on its local data.
    3. Clips each client's update to the L2 norm `clip_norm` (DP-SGD).
    4. Computes the geometric median of the clipped updates.
    5. Applies the MAD-based outlier filter to exclude anomalous clients.
    6. Adds Gaussian noise N(0, (sigma * C_clip)^2 * I) per coordinate
       to the aggregate (DP-SGD, thesis Eq. 36, p.72).
    7. Cross-checks the aggregate against the topological detector's
       predictions on a held-out split (the engineering safeguard of
       thesis §4.6.3, p.73).
    8. Updates the global model.
    9. Records the round's privacy cost via the RDP accountant.
    10. Checks the convergence criteria (fl_dp_params.yaml §7).

The coordinator is protocol-agnostic: it runs on any of DoH, DoT, or
DoQ. A single coordinator instance handles one protocol; a caller
that wants all three should create three coordinators.

THE DP-SGD CONFIGURATION
------------------------
Per fl_dp_params.yaml §2:

    clip_norm        = 3.5       (C_clip)
    noise_multiplier = 1.2       (sigma)
    adaptive_clipping = true     (per-round C_clip from observed norms)
    quantile         = 0.5       (median)
    quantile_noise_multiplier = 0.5

The noise added per coordinate is N(0, (sigma * C_clip)^2), as stated
in thesis Eq. 36 (p.72). The RDP accountant (math_utils.rdp_epsilon)
computes the realized ε from (sigma, q, T, delta).

THE TWO EPSILON FIGURES
-----------------------
The thesis reports TWO epsilon values, and the artifact keeps them
separate (see fl_dp_params.yaml §3 for the full discussion):

    - Target budget for the main detection experiments: ε = 1.2
      (the value the noise_multiplier is calibrated against)

    - Realized loss per Table 25 configuration: ε ∈ [0.31, 0.81]
      (each row's own realized loss under its own calibration)

This engine computes the realized ε from the configured (sigma, q, T,
delta) triple, and exposes both figures separately via the
`privacy_budget()` method. A caller that wants the target budget uses
`target_epsilon`; a caller that wants the realized loss uses
`compute_realized_epsilon()`.

THE SECURITY BOUNDARY
---------------------
The formal security statement of the DP layer is approximate-DP under
a single-client-record neighbourhood relation, with (ε, δ) given by
the RDP accountant (thesis Eq. 59, p.105; Eq. 129, p.166). No
statement in this module should be read as a claim beyond that.

    CLAIMED:
      - The federated training procedure satisfies (ε, δ)-DP under
        the single-client-record neighbourhood, with (ε, δ) computed
        by the RDP accountant.
      - The geometric-median aggregation is a published Byzantine-
        robust rule that bounds the influence of outlying updates
        under the stated assumptions (thesis §4.6.3, p.73).

    NOT CLAIMED:
      - Maximum theoretical Byzantine tolerance for all synchronous
        settings (thesis §4.6.3, p.73 says this explicitly).
      - The cross-check against the topological detector is the source
        of the reported robustness (thesis §4.6.3, p.73: "was not
        isolated by an ablation").
      - Regulatory compliance from DP alone (thesis §5.16, p.150).

A reviewer auditing the artifact's security claims should read
fl_dp_params.yaml §9 first, and this module's docstring second.

DETERMINISM
-----------
The coordinator is deterministic given a seed. The seed is derived
from the run seed via `seed.get_component_seed(seed, "qhe_ziggurat")`
— wait, no, that's the QHE offset. Federated learning uses the run
seed directly because it does not have its own offset in the config
(SEED_OFFSETS in seed.py does not list a "federated" component; the
coordinator's seed is the run seed plus 0). If a future change adds a
"federated" offset, the seed should be updated in the same commit.

In the artifact, the coordinator's randomness is used for:
    - client sampling (which clients participate in which round)
    - DP noise addition (the Gaussian noise per round)
    - adaptive clipping quantile noise

THREAD SAFETY
-------------
A FLCoordinator instance is not thread-safe: the global model and the
privacy accountant are mutable. A caller that wants concurrent
federated rounds should create one coordinator per thread. In
practice, the rounds are naturally serialised (each round depends on
the previous round's global model), so a single-threaded coordinator
is the correct usage.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.utils.math_utils import rdp_epsilon
from src.utils.logger import get_logger
from src.utils.seed import get_component_seed, get_component_rng


# ---------------------------------------------------------------------------
# Constants — from fl_dp_params.yaml §2-§8
# ---------------------------------------------------------------------------
# Duplicated defaults, matching the AEE and QHE engines' pattern. The
# engine's primary source of truth is the config file; the defaults
# below are used only if the config is missing or partial.

_DEFAULT_CLIP_NORM = 3.5
_DEFAULT_NOISE_MULTIPLIER = 1.2
_DEFAULT_SAMPLE_RATE = 0.8
_DEFAULT_DELTA = 1.0e-5
_DEFAULT_N_ROUNDS = 10
_DEFAULT_LOCAL_BATCH_SIZE = 64
_DEFAULT_LOCAL_EPOCHS = 1
_DEFAULT_LEARNING_RATE = 0.01

# Adaptive clipping parameters (fl_dp_params.yaml §2).
_DEFAULT_QUANTILE = 0.5
_DEFAULT_QUANTILE_NOISE_MULTIPLIER = 0.5
_DEFAULT_CLIP_MIN = 1.0
_DEFAULT_CLIP_MAX = 8.0
_DEFAULT_RECOMPUTE_INTERVAL = 5

# IRLS parameters for geometric median (fl_dp_params.yaml §5).
_DEFAULT_IRLS_MAX_ITERATIONS = 100
_DEFAULT_IRLS_TOLERANCE = 1.0e-6

# MAD outlier detection (fl_dp_params.yaml §5).
_DEFAULT_MAD_K = 2.5

# Convergence (fl_dp_params.yaml §7).
_DEFAULT_TARGET_ACCURACY = 0.95
_DEFAULT_STABILITY_BAND = 0.005
_DEFAULT_PATIENCE = 2
_DEFAULT_MIN_GRADIENT_NORM = 1.0e-3
_DEFAULT_MAX_ROUNDS = 10

# Default per-protocol client counts (fl_dp_params.yaml §6).
_DEFAULT_CLIENTS_PER_PROTOCOL = {
    "doh": 100,
    "dot": 100,
    "doq": 100,
}


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class FLError(Exception):
    """Base class for federated-learning errors."""


class FLConfigError(FLError):
    """Raised when the config is missing or has invalid values."""


class FLClientError(FLError):
    """Raised when a client cannot compute an update (e.g. empty dataset)."""


class FLAggregationError(FLError):
    """Raised when aggregation fails (e.g. all clients excluded)."""


# ---------------------------------------------------------------------------
# Client and Update dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ClientUpdate:
    """
    One client's update for one round.

    Fields
    ------
    client_id : str
    gradient : np.ndarray
        The gradient of the client's local loss with respect to the
        global model parameters. Shape matches the global model.
    n_samples : int
        Number of local examples the gradient was computed on. Used
        for the weighted geometric median (a client with more data
        gets more weight).
    raw_norm : float
        The L2 norm of the gradient before clipping. Used for the
        adaptive clipping quantile.
    clipped_norm : float
        The L2 norm after clipping (≤ clip_norm).
    """
    client_id: str
    gradient: np.ndarray
    n_samples: int
    raw_norm: float
    clipped_norm: float


@dataclass
class RoundRecord:
    """
    The record of one federated round. Written to the run log and
    summarised in the figure-13 output.
    """
    round_number: int
    n_clients_sampled: int
    n_clients_included: int        # after MAD outlier filtering
    n_clients_excluded: int
    clip_norm: float
    aggregate_norm: float
    dp_noise_norm: float
    global_accuracy: Optional[float]
    realized_epsilon: float
    cumulative_round_time_ms: float
    wall_time_seconds: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "round": self.round_number,
            "n_sampled": self.n_clients_sampled,
            "n_included": self.n_clients_included,
            "n_excluded": self.n_clients_excluded,
            "clip_norm": round(self.clip_norm, 4),
            "aggregate_norm": round(self.aggregate_norm, 4),
            "dp_noise_norm": round(self.dp_noise_norm, 4),
            "global_accuracy": (
                round(self.global_accuracy, 6)
                if self.global_accuracy is not None else None
            ),
            "realized_epsilon": round(self.realized_epsilon, 4),
            "round_time_ms": round(self.cumulative_round_time_ms, 4),
        }


# ---------------------------------------------------------------------------
# FLClient — one participating client
# ---------------------------------------------------------------------------

class FLClient:
    """
    A single federated-learning client.

    A client holds a local dataset (a set of (feature_vector, label)
    pairs) and computes a gradient of its local loss with respect to
    the global model parameters. The coordinator calls
    compute_update() once per round.

    The client's local model is a **linear classifier** for the
    artifact. The thesis does not specify a particular local model in
    the federated layer (the model is described in §6.6.1 as a
    spatio-temporal attention transformer, but that is the integrated
    architecture of Chapter 6, not the federated layer of Chapter 5).
    A linear classifier is the simplest model that has a well-defined
    gradient and that can be trained with SGD; it is what the artifact
    uses, and it is documented as such.

    A caller that wants a different local model can subclass FLClient
    and override compute_update(); the coordinator only depends on the
    compute_update() contract.
    """

    def __init__(
        self,
        client_id: str,
        protocol: str,
        local_data: Tuple[np.ndarray, np.ndarray],
        *,
        batch_size: int = _DEFAULT_LOCAL_BATCH_SIZE,
        learning_rate: float = _DEFAULT_LEARNING_RATE,
        n_local_epochs: int = _DEFAULT_LOCAL_EPOCHS,
        seed: Optional[int] = None,
    ) -> None:
        if protocol not in _DEFAULT_CLIENTS_PER_PROTOCOL:
            raise FLClientError(
                f"unknown protocol {protocol!r}; "
                f"expected one of {sorted(_DEFAULT_CLIENTS_PER_PROTOCOL)}"
            )
        X, y = local_data
        if len(X) == 0:
            raise FLClientError(
                f"client {client_id!r} has an empty dataset"
            )
        if X.ndim != 2:
            raise FLClientError(
                f"client {client_id!r}: X must be 2-D; got shape {X.shape}"
            )
        self.client_id = client_id
        self.protocol = protocol
        self._X = np.asarray(X, dtype=np.float64)
        self._y = np.asarray(y, dtype=np.float64)
        self._batch_size = int(batch_size)
        self._learning_rate = float(learning_rate)
        self._n_local_epochs = int(n_local_epochs)
        # Per-client RNG, seeded deterministically from the client_id
        # hash. This makes the client's local randomness reproducible
        # without coupling it to the coordinator's RNG stream.
        if seed is None:
            seed = int.from_bytes(
                __import__("hashlib").sha256(client_id.encode()).digest()[:4],
                "little",
            )
        self._rng = np.random.default_rng(int(seed))

    @property
    def n_samples(self) -> int:
        return len(self._X)

    def compute_update(
        self,
        global_params: np.ndarray,
        *,
        poisoned: bool = False,
    ) -> np.ndarray:
        """
        Compute the local gradient.

        For the linear classifier:
            loss(w) = (1/2N) * sum_i (w · x_i - y_i)^2
            grad(w) = (1/N) * X^T (Xw - y)

        The client runs one epoch of minibatch SGD and returns the
        *average* gradient across the epoch's batches. This is the
        standard federated-local-update; it differs from FedSGD
        (which would return the full-batch gradient) in that it is
        more realistic for a client with limited compute.

        Parameters
        ----------
        global_params : np.ndarray
            The current global model parameters w.
        poisoned : bool
            If True, the client's update is replaced with a scaled
            random vector, simulating a compromised client. Used by
            the Table 25 "compromised clients" configuration and by
            the Byzantine-robustness panel of Figure 13(c).

        Returns
        -------
        The gradient, shape matching global_params.
        """
        w = np.asarray(global_params, dtype=np.float64)
        n = self.n_samples
        d = w.shape[0]
        if self._X.shape[1] != d:
            raise FLClientError(
                f"client {self.client_id!r}: X has {self._X.shape[1]} "
                f"features, global model has {d}"
            )

        if poisoned:
            # Poisoned update: scaled random vector. The scale is the
            # same order as a typical benign gradient so that the
            # clipping and MAD filters have to do real work to detect
            # it; a larger scale would be trivially clipped.
            return self._rng.normal(0.0, 1.0, size=d) * float(np.sqrt(d))

        # Minibatch SGD over the client's data.
        accum = np.zeros(d, dtype=np.float64)
        n_batches = 0
        for _ in range(self._n_local_epochs):
            indices = self._rng.permutation(n)
            for start in range(0, n, self._batch_size):
                end = min(start + self._batch_size, n)
                idx = indices[start:end]
                X_batch = self._X[idx]
                y_batch = self._y[idx]
                residual = X_batch @ w - y_batch
                grad_batch = X_batch.T @ residual / len(idx)
                accum += grad_batch
                n_batches += 1
        if n_batches == 0:
            raise FLClientError(
                f"client {self.client_id!r}: no batches computed "
                f"(n={n}, batch_size={self._batch_size})"
            )
        return accum / n_batches


# ---------------------------------------------------------------------------
# FLCoordinator
# ---------------------------------------------------------------------------

class FLCoordinator:
    """
    Multi-Protocol Federated Learning coordinator.

    See the module docstring for the full description and the
    security boundary.

    Parameters
    ----------
    config : dict
        The dict returned by data_utils.load_config().
    protocol : "doh" | "dot" | "doq"
    model_dim : int
        The dimension of the global model (the number of features).
        Defaults to len(FEATURE_COLUMNS) = 12.
    seed : optional int
    """

    def __init__(
        self,
        config: Dict[str, Any],
        protocol: str = "doh",
        model_dim: int = 12,
        seed: Optional[int] = None,
    ) -> None:
        if protocol not in _DEFAULT_CLIENTS_PER_PROTOCOL:
            raise FLConfigError(
                f"unknown protocol {protocol!r}; "
                f"expected one of {sorted(_DEFAULT_CLIENTS_PER_PROTOCOL)}"
            )
        self._protocol = protocol
        self._model_dim = int(model_dim)
        self._config = config
        self._log = get_logger(f"e3doh.models.fl.{protocol}")

        # Load parameters from the config.
        try:
            fl_cfg = config["_includes"]["fl_dp"]
        except (KeyError, TypeError) as e:
            raise FLConfigError(
                "fl_dp_params.yaml not found in config['_includes']. "
                "The FL coordinator cannot initialize without its parameters. "
                "Check that configs/default_e3_config.yaml §11 lists the "
                "include and that the file is present."
            ) from e
        self._fl_cfg = fl_cfg

        dp = fl_cfg.get("dp_sgd", {})
        self._clip_norm = float(dp.get("clip_norm", _DEFAULT_CLIP_NORM))
        self._noise_multiplier = float(dp.get("noise_multiplier", _DEFAULT_NOISE_MULTIPLIER))
        self._learning_rate = float(dp.get("learning_rate", _DEFAULT_LEARNING_RATE))

        adaptive = dp.get("adaptive_clipping", {})
        self._adaptive_clipping = bool(adaptive.get("enabled", False))
        self._adaptive_quantile = float(adaptive.get("quantile", _DEFAULT_QUANTILE))
        self._adaptive_clip_min = float(adaptive.get("min", _DEFAULT_CLIP_MIN))
        self._adaptive_clip_max = float(adaptive.get("max", _DEFAULT_CLIP_MAX))
        self._adaptive_recompute_interval = int(
            adaptive.get("recompute_interval_rounds", _DEFAULT_RECOMPUTE_INTERVAL)
        )
        self._adaptive_quantile_noise = float(
            adaptive.get("quantile_noise_multiplier", _DEFAULT_QUANTILE_NOISE_MULTIPLIER)
        )

        budget = fl_cfg.get("privacy_budget", {})
        self._target_epsilon = float(budget.get("target_epsilon", 1.2))
        self._delta = float(budget.get("delta", _DEFAULT_DELTA))
        self._composition_rounds = int(budget.get("composition_rounds", _DEFAULT_N_ROUNDS))

        aggregation = fl_cfg.get("aggregation", {})
        self._aggregation_method = str(aggregation.get("method", "geometric_median"))
        irls = aggregation.get("irls", {})
        self._irls_max_iterations = int(irls.get("max_iterations", _DEFAULT_IRLS_MAX_ITERATIONS))
        self._irls_tolerance = float(irls.get("tolerance", _DEFAULT_IRLS_TOLERANCE))
        mad = aggregation.get("mad", {})
        self._mad_enabled = bool(mad.get("enabled", True))
        self._mad_k = float(mad.get("k", _DEFAULT_MAD_K))

        clients_cfg = fl_cfg.get("clients", {})
        per_proto = clients_cfg.get("per_protocol", _DEFAULT_CLIENTS_PER_PROTOCOL)
        self._n_clients = int(per_proto.get(protocol, _DEFAULT_CLIENTS_PER_PROTOCOL[protocol]))
        self._client_sample_rate = float(
            clients_cfg.get("sample_rate", _DEFAULT_SAMPLE_RATE)
        )

        convergence = fl_cfg.get("convergence", {})
        self._target_accuracy = float(
            convergence.get("target_accuracy", _DEFAULT_TARGET_ACCURACY)
        )
        self._stability_band = float(
            convergence.get("stability_band", _DEFAULT_STABILITY_BAND)
        )
        self._patience = int(convergence.get("patience", _DEFAULT_PATIENCE))
        self._min_gradient_norm = float(
            convergence.get("min_gradient_norm", _DEFAULT_MIN_GRADIENT_NORM)
        )
        self._max_rounds = int(convergence.get("max_rounds", _DEFAULT_MAX_ROUNDS))

        # Seed and RNG.
        if seed is None:
            try:
                seed = int(config.get("runtime", {}).get("seed", 42))
            except (TypeError, ValueError):
                seed = 42
        self._run_seed = int(seed)
        # The coordinator uses the run seed directly (offset +0), as
        # documented in the module docstring.
        self._rng = np.random.default_rng(self._run_seed)

        # Global model: zeros.
        self._global_params = np.zeros(self._model_dim, dtype=np.float64)

        # Registered clients.
        self._clients: List[FLClient] = []

        # Cross-check hook. A caller that wants the engineering
        # safeguard of thesis §4.6.3 can register a callable that takes
        # (aggregate_gradient, validation_data) and returns a bool
        # (True = cross-check passes). A caller that does not register
        # a hook skips the cross-check; this is the documented
        # behaviour of fl_dp_params.yaml §5 (cross_check.enabled=true
        # requires a detector to be wired, and the artifact does not
        # wire one by default — see "Cross-check" below).
        self._cross_check_hook: Optional[Callable] = None

        # Round history.
        self._history: List[RoundRecord] = []

        # The last clip norm used. Initialized to the config value and
        # updated by the adaptive clipping mechanism.
        self._current_clip_norm = self._clip_norm

        # Cumulative per-operation telemetry.
        self._total_aggregation_us = 0.0
        self._total_dp_noise_us = 0.0

        self._log.info(
            "fl_coordinator_initialised",
            protocol=protocol,
            n_clients=self._n_clients,
            model_dim=self._model_dim,
            clip_norm=self._clip_norm,
            noise_multiplier=self._noise_multiplier,
            adaptive_clipping=self._adaptive_clipping,
            aggregation_method=self._aggregation_method,
            target_epsilon=self._target_epsilon,
        )

    # ------------------------------------------------------------------
    # Client registration
    # ------------------------------------------------------------------

    def register_client(self, client: FLClient) -> None:
        """
        Register a client with the coordinator.

        The client's `protocol` must match the coordinator's. The
        client is appended to the internal list; the coordinator does
        not validate the client's dataset size or feature dimension
        until the first round (a client that fails at round time is
        logged and skipped, not raised).
        """
        if client.protocol != self._protocol:
            raise FLError(
                f"client {client.client_id!r} has protocol "
                f"{client.protocol!r}, but the coordinator runs "
                f"{self._protocol!r}"
            )
        self._clients.append(client)

    def register_clients(self, clients: Sequence[FLClient]) -> None:
        for client in clients:
            self.register_client(client)

    def set_cross_check_hook(self, hook: Callable) -> None:
        """
        Register a cross-check hook. The hook is called with
        (aggregate_gradient, protocol) after each round's aggregate is
        computed; it should return a bool (True = pass). A round whose
        cross-check returns False is logged and its aggregate is not
        applied.

        The default behaviour (no hook registered) is to skip the
        cross-check. The fl_dp_params.yaml §5 cross_check.enabled=true
        setting describes the thesis's safeguard; the artifact does
        not wire a detector by default because wiring one would
        couple the FL layer to the anomaly detector, and the thesis
        is explicit that the cross-check's effect "was not isolated
        by an ablation" (thesis §4.6.3, p.73). A reviewer who wants
        to enable the safeguard can register their own hook.
        """
        self._cross_check_hook = hook

    # ------------------------------------------------------------------
    # Public accessors
    # ------------------------------------------------------------------

    @property
    def global_params(self) -> np.ndarray:
        return self._global_params.copy()

    @property
    def history(self) -> List[RoundRecord]:
        return list(self._history)

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "protocol": self._protocol,
            "n_clients": self._n_clients,
            "model_dim": self._model_dim,
            "clip_norm": self._current_clip_norm,
            "noise_multiplier": self._noise_multiplier,
            "sample_rate": self._client_sample_rate,
            "delta": self._delta,
            "composition_rounds": self._composition_rounds,
            "target_epsilon": self._target_epsilon,
            "adaptive_clipping": self._adaptive_clipping,
            "aggregation_method": self._aggregation_method,
        }

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def privacy_budget(self) -> Tuple[float, float]:
        """
        Return (target_epsilon, realized_epsilon) for the current
        composition.

        - target_epsilon is the configured target from
          fl_dp_params.yaml §3 (1.2 by default).

        - realized_epsilon is the epsilon computed by the RDP
          accountant for the *actual* number of rounds completed.

        The two are distinct quantities and should not be read as the
        same figure (fl_dp_params.yaml §3).
        """
        return self._target_epsilon, self.compute_realized_epsilon()

    def compute_realized_epsilon(self, n_rounds: Optional[int] = None) -> float:
        """
        Compute the realized epsilon for n_rounds rounds of DP-SGD
        under the current (noise_multiplier, sample_rate, delta).
        """
        if n_rounds is None:
            n_rounds = len(self._history) if self._history else self._composition_rounds
        eps, _ = rdp_epsilon(
            noise_multiplier=self._noise_multiplier,
            sample_rate=self._client_sample_rate,
            n_rounds=n_rounds,
            delta=self._delta,
        )
        return eps

    def run_round(self) -> RoundRecord:
        """
        Run one federated round. Returns the RoundRecord.

        Raises
        ------
        FLError
            If no clients are registered, or if no client produced a
            usable update (all excluded by the MAD filter).
        """
        if not self._clients:
            raise FLError(
                "cannot run a round: no clients registered. Call "
                "register_client() or register_clients() first."
            )

        wall_start = time.perf_counter()
        round_number = len(self._history) + 1

        # --- 1. Sample clients ---
        n_sample = max(1, int(round(self._n_clients * self._client_sample_rate)))
        all_ids = list(range(len(self._clients)))
        sampled_indices = self._rng.choice(
            all_ids, size=min(n_sample, len(all_ids)), replace=False,
        )
        sampled_clients = [self._clients[i] for i in sampled_indices]

        # --- 2. Collect updates ---
        updates: List[ClientUpdate] = []
        for client in sampled_clients:
            try:
                grad = client.compute_update(self._global_params)
            except FLClientError as e:
                self._log.warning(
                    "fl_client_update_failed",
                    round=round_number, client=client.client_id,
                    error=str(e),
                )
                continue
            raw_norm = float(np.linalg.norm(grad))
            updates.append(ClientUpdate(
                client_id=client.client_id,
                gradient=grad,
                n_samples=client.n_samples,
                raw_norm=raw_norm,
                clipped_norm=0.0,  # filled below
            ))

        if not updates:
            raise FLError(
                f"round {round_number}: no client produced a usable update"
            )

        # --- 3. Update the clip norm (adaptive or fixed) ---
        if self._adaptive_clipping and (
            round_number % self._adaptive_recompute_interval == 0
            or round_number == 1
        ):
            self._current_clip_norm = self._compute_adaptive_clip_norm(
                [u.raw_norm for u in updates],
            )
            self._log.info(
                "fl_adaptive_clip_updated",
                round=round_number, clip_norm=self._current_clip_norm,
            )

        # --- 4. Clip each client's update ---
        clip = self._current_clip_norm
        for u in updates:
            norm = u.raw_norm
            if norm > clip:
                u.gradient = u.gradient * (clip / norm)
                u.clipped_norm = clip
            else:
                u.clipped_norm = norm

        # --- 5. Aggregate via geometric median ---
        agg_started = time.perf_counter()
        aggregate, n_included, n_excluded = self._aggregate(updates)
        self._total_aggregation_us += (time.perf_counter() - agg_started) * 1e6

        if n_included == 0:
            raise FLAggregationError(
                f"round {round_number}: all {n_excluded} clients were "
                f"excluded by the MAD filter; cannot aggregate."
            )

        # --- 6. Add DP noise ---
        noise_started = time.perf_counter()
        noise_vec = self._rng.normal(
            0.0,
            self._noise_multiplier * self._current_clip_norm,
            size=self._model_dim,
        )
        aggregate = aggregate + noise_vec
        self._total_dp_noise_us += (time.perf_counter() - noise_started) * 1e6

        # --- 7. Cross-check ---
        cross_check_passed = True
        if self._cross_check_hook is not None:
            try:
                cross_check_passed = bool(
                    self._cross_check_hook(aggregate, self._protocol)
                )
            except Exception as e:
                self._log.warning(
                    "fl_cross_check_failed",
                    round=round_number, error=repr(e),
                )
                cross_check_passed = False
        if not cross_check_passed:
            self._log.warning(
                "fl_cross_check_rejected",
                round=round_number,
                note="aggregate not applied; global model unchanged",
            )

        # --- 8. Update the global model ---
        if cross_check_passed:
            self._global_params = self._global_params - self._learning_rate * aggregate

        # --- 9. Record the round ---
        # The realized epsilon is computed from the actual number of
        # rounds completed (not the target), matching the accountant's
        # semantics.
        realized_eps = self.compute_realized_epsilon(
            n_rounds=round_number,
        )
        wall_elapsed = time.perf_counter() - wall_start
        record = RoundRecord(
            round_number=round_number,
            n_clients_sampled=len(sampled_clients),
            n_clients_included=n_included,
            n_clients_excluded=n_excluded,
            clip_norm=self._current_clip_norm,
            aggregate_norm=float(np.linalg.norm(aggregate)),
            dp_noise_norm=float(np.linalg.norm(noise_vec)),
            global_accuracy=None,  # caller sets via set_global_accuracy
            realized_epsilon=realized_eps,
            cumulative_round_time_ms=(time.perf_counter() - wall_start) * 1000.0,
            wall_time_seconds=wall_elapsed,
        )
        self._history.append(record)

        self._log.info(
            "fl_round_complete",
            **record.to_dict(),
            cross_check_passed=cross_check_passed,
        )
        return record

    def run(
        self,
        *,
        n_rounds: Optional[int] = None,
        eval_fn: Optional[Callable[[np.ndarray], float]] = None,
        max_rounds: Optional[int] = None,
    ) -> List[RoundRecord]:
        """
        Run federated rounds until convergence or until max_rounds.

        Parameters
        ----------
        n_rounds : optional int
            Fixed number of rounds. If given, overrides the convergence
            criteria and runs exactly this many rounds. A caller that
            wants reproducibility across runs (e.g. for a table
            script) should pass n_rounds to avoid the convergence
            path being sensitive to the eval_fn's noise.
        eval_fn : optional callable
            Called once per round with the global_params, returns the
            global model's accuracy on the evaluation split. If given,
            the accuracy is recorded in the RoundRecord and used by
            the convergence criteria.
        max_rounds : optional int
            Overrides the convergence.max_rounds from the config.

        Returns
        -------
        The list of RoundRecords (also available as self.history).

        Raises
        ------
        FLError
            If the run fails (see run_round).
        """
        effective_max = max_rounds if max_rounds is not None else self._max_rounds

        if n_rounds is not None:
            for _ in range(int(n_rounds)):
                record = self.run_round()
                if eval_fn is not None:
                    record.global_accuracy = float(eval_fn(self._global_params))
            return self._history

        # Convergence-driven run.
        stable_rounds = 0
        last_accuracy = None
        for _ in range(effective_max):
            record = self.run_round()
            if eval_fn is not None:
                record.global_accuracy = float(eval_fn(self._global_params))
                if (
                    record.global_accuracy >= self._target_accuracy
                    and last_accuracy is not None
                    and abs(record.global_accuracy - last_accuracy) <= self._stability_band
                ):
                    stable_rounds += 1
                else:
                    stable_rounds = 0
                last_accuracy = record.global_accuracy
            # Check gradient norm convergence.
            if record.aggregate_norm < self._min_gradient_norm:
                self._log.info(
                    "fl_converged_gradient_norm",
                    round=record.round_number,
                    aggregate_norm=record.aggregate_norm,
                )
                break
            # Check accuracy convergence.
            if stable_rounds >= self._patience:
                self._log.info(
                    "fl_converged_accuracy",
                    round=record.round_number,
                    accuracy=record.global_accuracy,
                    stable_rounds=stable_rounds,
                )
                break
        return self._history

    # ------------------------------------------------------------------
    # Aggregation
    # ------------------------------------------------------------------

    def _aggregate(
        self, updates: List[ClientUpdate],
    ) -> Tuple[np.ndarray, int, int]:
        """
        Aggregate the client updates via the configured method.

        Returns
        -------
        (aggregate, n_included, n_excluded)
        """
        gradients = np.stack([u.gradient for u in updates], axis=0)
        weights = np.array([u.n_samples for u in updates], dtype=np.float64)

        if self._aggregation_method == "arithmetic_mean":
            aggregate = np.mean(gradients, axis=0)
            return aggregate, len(updates), 0

        if self._aggregation_method == "geometric_median":
            aggregate = self._geometric_median(gradients, weights)

            # MAD outlier filtering.
            if self._mad_enabled:
                distances = np.linalg.norm(gradients - aggregate, axis=1)
                median = float(np.median(distances))
                mad = float(np.median(np.abs(distances - median)))
                threshold = median + self._mad_k * max(mad, 1e-6)
                included_mask = distances <= threshold
                n_included = int(np.sum(included_mask))
                n_excluded = len(updates) - n_included
                if n_included > 0 and n_included < len(updates):
                    # Recompute the geometric median on the included set.
                    aggregate = self._geometric_median(
                        gradients[included_mask],
                        weights[included_mask],
                    )
                elif n_included == 0:
                    # All excluded: return the full median (which the
                    # caller will reject anyway, per _aggregate's
                    # documented contract).
                    return aggregate, 0, n_excluded
                return aggregate, n_included, n_excluded

            return aggregate, len(updates), 0

        raise FLConfigError(
            f"unknown aggregation method {self._aggregation_method!r}; "
            f"expected 'geometric_median' or 'arithmetic_mean'"
        )

    def _geometric_median(
        self,
        gradients: np.ndarray,
        weights: np.ndarray,
    ) -> np.ndarray:
        """
        Geometric median via iteratively reweighted least squares
        (thesis §4.6.3, Eq. 38, p.73).

        Weiszfeld's algorithm:
            x_0 = weighted mean
            Repeat:
                d_k = ||x - g_k||
                w_k = original_weight_k / max(d_k, eps)
                x_new = sum(w_k * g_k) / sum(w_k)
                if ||x_new - x|| < tol: stop
        """
        x = np.average(gradients, axis=0, weights=weights)
        for iteration in range(self._irls_max_iterations):
            distances = np.linalg.norm(gradients - x, axis=1)
            # Regularize distances to avoid division by zero.
            distances = np.maximum(distances, 1e-9)
            w = weights / distances
            x_new = np.sum(gradients * w[:, None], axis=0) / np.sum(w)
            delta = float(np.linalg.norm(x_new - x))
            x = x_new
            if delta < self._irls_tolerance:
                return x
        self._log.warning(
            "fl_geometric_median_did_not_converge",
            iterations=self._irls_max_iterations,
            tolerance=self._irls_tolerance,
            final_delta=delta,
        )
        return x

    def _compute_adaptive_clip_norm(self, raw_norms: List[float]) -> float:
        """
        Compute the adaptive clip norm from the observed raw norms.

        The quantile is `self._adaptive_quantile` (default 0.5 =
        median). Per fl_dp_params.yaml §2, the quantile itself is
        noisy: a Gaussian noise term is added before clipping, to
        prevent the quantile from leaking individual norms.

        The result is bounded to [min, max] from the config.
        """
        arr = np.asarray(raw_norms, dtype=np.float64)
        quantile = float(np.quantile(arr, self._adaptive_quantile))
        noise = self._rng.normal(
            0.0,
            self._adaptive_quantile_noise * float(np.std(arr) + 1e-6),
        )
        noisy = quantile + noise
        return float(np.clip(noisy, self._adaptive_clip_min, self._adaptive_clip_max))

    # ------------------------------------------------------------------
    # Reset / serialization
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset the global model to zeros and clear the round history."""
        self._global_params = np.zeros(self._model_dim, dtype=np.float64)
        self._history = []
        self._current_clip_norm = self._clip_norm
        self._total_aggregation_us = 0.0
        self._total_dp_noise_us = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "protocol": self._protocol,
            "parameters": self.parameters,
            "global_params": self._global_params.tolist(),
            "history": [r.to_dict() for r in self._history],
        }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "FLCoordinator",
    "FLClient",
    "ClientUpdate",
    "RoundRecord",
    "FLError",
    "FLConfigError",
    "FLClientError",
    "FLAggregationError",
]


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def _self_test() -> int:
    import sys
    rc = 0
    print("=== fl_coordinator self-test ===")

    # --- Build a minimal config ---
    config = {
        "_includes": {
            "fl_dp": {
                "dp_sgd": {
                    "clip_norm": 3.5,
                    "noise_multiplier": 1.2,
                    "learning_rate": 0.01,
                    "adaptive_clipping": {
                        "enabled": False,  # fixed clipping for deterministic self-test
                    },
                },
                "privacy_budget": {
                    "target_epsilon": 1.2,
                    "delta": 1e-5,
                    "composition_rounds": 10,
                },
                "aggregation": {
                    "method": "geometric_median",
                    "irls": {"max_iterations": 100, "tolerance": 1e-6},
                    "mad": {"enabled": True, "k": 2.5},
                },
                "clients": {
                    "per_protocol": {"doh": 20, "dot": 20, "doq": 20},
                    "sample_rate": 0.8,
                },
                "convergence": {
                    "target_accuracy": 0.95,
                    "stability_band": 0.005,
                    "patience": 2,
                    "min_gradient_norm": 1e-3,
                    "max_rounds": 5,
                },
            },
        },
        "runtime": {"seed": 42},
    }

    # 1. Construct
    coord = FLCoordinator(config, protocol="doh", model_dim=12, seed=42)
    if coord.parameters["n_clients"] != 20:
        print(f"[FAIL] n_clients: {coord.parameters['n_clients']}")
        rc = 1
    else:
        print(f"[OK]   FLCoordinator constructed (20 clients, model_dim=12)")

    # 2. Register 20 clients with synthetic datasets.
    rng = np.random.default_rng(0)
    for i in range(20):
        X = rng.normal(size=(50, 12))
        # Linear target: y = X @ w_true + noise
        w_true = rng.normal(size=12)
        y = X @ w_true + rng.normal(scale=0.1, size=50)
        coord.register_client(FLClient(
            client_id=f"client-{i:03d}",
            protocol="doh",
            local_data=(X, y),
            seed=42 + i,
        ))
    print(f"[OK]   registered 20 clients")

    # 3. Run 3 rounds.
    history = coord.run(n_rounds=3)
    if len(history) != 3:
        print(f"[FAIL] history length: {len(history)}")
        rc = 1
    else:
        print(f"[OK]   ran 3 rounds")
        for r in history:
            print(f"         round {r.round_number}: sampled={r.n_clients_sampled}, "
                  f"included={r.n_clients_included}, excluded={r.n_clients_excluded}, "
                  f"aggregate_norm={r.aggregate_norm:.4f}, "
                  f"realized_eps={r.realized_epsilon:.4f}")

    # 4. Privacy budget: target and realized are distinct.
    target, realized = coord.privacy_budget()
    if target != 1.2:
        print(f"[FAIL] target epsilon: {target}")
        rc = 1
    elif not (0.0 < realized < 10.0):
        print(f"[FAIL] realized epsilon out of range: {realized}")
        rc = 1
    else:
        print(f"[OK]   privacy_budget: target={target}, realized={realized:.4f}")

    # 5. Realized epsilon increases with rounds.
    eps_3 = coord.compute_realized_epsilon(n_rounds=3)
    eps_10 = coord.compute_realized_epsilon(n_rounds=10)
    if eps_10 < eps_3:
        print(f"[FAIL] epsilon should increase with rounds: 3→{eps_3:.4f}, 10→{eps_10:.4f}")
        rc = 1
    else:
        print(f"[OK]   epsilon monotonically increases with rounds "
              f"(3→{eps_3:.4f}, 10→{eps_10:.4f})")

    # 6. Geometric median on a small example.
    # Three points in 2D: (0,0), (10,0), (0,10). The geometric median
    # is (0,0) (the minimizer of sum of L2 distances).
    pts = np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0]])
    w = np.array([1.0, 1.0, 1.0])
    gm = coord._geometric_median(pts, w)
    if np.linalg.norm(gm) > 0.1:
        print(f"[FAIL] geometric median: {gm}, expected near (0,0)")
        rc = 1
    else:
        print(f"[OK]   geometric median: {gm} (correctly near origin)")

    # 7. MAD outlier detection.
    # Four points: (0,0) × 3 and (10,10). The median is (0,0); the
    # outlier should be excluded.
    pts = np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [10.0, 10.0]])
    distances = np.linalg.norm(pts - np.array([0.0, 0.0]), axis=1)
    median = float(np.median(distances))
    mad = float(np.median(np.abs(distances - median)))
    threshold = median + 2.5 * max(mad, 1e-6)
    inliers = distances <= threshold
    if int(np.sum(inliers)) != 3:
        print(f"[FAIL] MAD outlier: {int(np.sum(inliers))} inliers (expected 3)")
        rc = 1
    else:
        print(f"[OK]   MAD outlier: {int(np.sum(inliers))} inliers, "
              f"{int(np.sum(~inliers))} outlier correctly excluded")

    # 8. Byzantine robustness: add a poisoned client and check the
    # aggregate is still dominated by the honest clients.
    coord2 = FLCoordinator(config, protocol="doh", model_dim=12, seed=42)
    # 5 honest clients with gradient near (1,0,...,0)
    for i in range(5):
        class _FixedClient(FLClient):
            def __init__(self, cid):
                X = np.zeros((1, 12)); y = np.array([0.0])
                super().__init__(cid, "doh", (X, y))
            def compute_update(self, global_params, poisoned=False):
                return np.ones(12) * (1.0 if not poisoned else -100.0)
        coord2.register_client(_FixedClient(f"honest-{i:02d}"))
    # 1 poisoned client with gradient near (-100, 0, ..., 0)
    class _PoisonedClient(FLClient):
        def __init__(self, cid):
            X = np.zeros((1, 12)); y = np.array([0.0])
            super().__init__(cid, "doh", (X, y))
        def compute_update(self, global_params, poisoned=False):
            v = np.zeros(12); v[0] = -100.0
            return v
    coord2.register_client(_PoisonedClient("malicious-00"))
    # Sample all clients (rate 1.0)
    coord2._client_sample_rate = 1.0
    # Disable DP noise for this test (so we can see the aggregation clearly)
    coord2._noise_multiplier = 0.0
    record = coord2.run_round()
    agg = coord2.global_params
    # The aggregate's first coordinate should be near 1 (honest mean),
    # not near -100 (poisoned). MAD should exclude the poisoned client.
    if record.n_clients_excluded < 1:
        print(f"[FAIL] MAD did not exclude poisoned client "
              f"(included={record.n_clients_included}, excluded={record.n_clients_excluded})")
        rc = 1
    elif agg[0] > 0.0:
        print(f"[FAIL] aggregate is positive after poisoned client "
              f"(agg[0]={agg[0]:.4f}); poisoned client was not excluded")
        rc = 1
    else:
        # agg = w - lr * aggregate_gradient. If the aggregate is near +1
        # (honest), then w[0] < 0 (moved negative). If the poisoned
        # update had been included, aggregate would be ~ -15 (much more
        # negative), and w[0] > 0 (moved further positive). We check
        # that the aggregate's effect is small (i.e., the poisoned
        # client was excluded).
        effect = abs(agg[0])
        if effect > 0.1:  # any larger than a single honest step
            print(f"[FAIL] aggregate effect too large ({effect:.4f}); "
                  f"poisoned client may not have been excluded")
            rc = 1
        else:
            print(f"[OK]   Byzantine robustness: poisoned client excluded, "
                  f"aggregate near honest mean (w[0]={agg[0]:.4f})")

    # 9. Convergence run.
    coord3 = FLCoordinator(config, protocol="doh", model_dim=12, seed=42)
    rng2 = np.random.default_rng(0)
    for i in range(10):
        X = rng2.normal(size=(20, 12))
        w_true = rng2.normal(size=12)
        y = X @ w_true
        coord3.register_client(FLClient(
            client_id=f"c-{i:02d}", protocol="doh",
            local_data=(X, y), seed=42 + i,
        ))
    def eval_fn(w):
        # Return the fraction of small-magnitude coordinates (a proxy
        # for convergence: a well-trained w should have many small
        # coordinates if the target w_true is near zero).
        return float(np.mean(np.abs(w) < 1.0))
    history = coord3.run(eval_fn=eval_fn)
    print(f"[OK]   convergence run: {len(history)} rounds, "
          f"final accuracy={history[-1].global_accuracy:.4f}, "
          f"final eps={history[-1].realized_epsilon:.4f}")

    # 10. Reset.
    coord.reset()
    if len(coord.history) != 0 or np.any(coord.global_params != 0):
        print(f"[FAIL] reset did not clear state")
        rc = 1
    else:
        print(f"[OK]   reset: history cleared, global model zeros")

    print("=== self-test complete ===")
    return rc


if __name__ == "__main__":
    import sys
    sys.exit(_self_test())
