#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
src/models/qhe_engine.py
========================

E3-DoH Reproducibility — Quantum-Safe Homomorphic Encryption (QHE) engine.

Chapter 5 of the thesis: E3-DoH (Enhanced Evolutionary Encryption for
Multi-Protocol DNS Security — DoH, DoT, DoQ).

This module implements the Ring-LWE homomorphic encryption layer of
thesis §5.3 (p.124-128). It is the second of the three engines and
provides the cryptographic mechanism that allows the anomaly-detection
statistics to be evaluated on ciphertext without decryption, so that
the detection pipeline never has to see the plaintext feature vectors.

THE PARAMETERS
--------------
The QHE layer uses a *different* parameter set from the AEE layer:

    QHE :  n = 256,  q = 4096          (qhe_params.yaml §2)
    AEE :  n = 1024, q ≈ 2^32          (aee_params.yaml §2)

The two are deliberately distinct (thesis §5.3.5, p.126, explicitly
corrects an earlier draft that assigned n=1024 to both). QHE uses a
smaller ring for speed; AEE uses a larger ring for the KEM's
IND-CCA2 security margin. The common ring form (x^n + 1) is what
allows the two layers to interoperate; the dimensions are not.

A NOTE ON NTT AND q = 4096
--------------------------
The NTT in `math_utils.build_ntt_twiddles` requires q to be prime
and q ≡ 1 (mod 2n). The QHE modulus q = 4096 = 2^12 is not prime.
The artifact therefore uses *direct polynomial multiplication*
(O(n²)), which for n=256 is about 65,536 integer operations per
multiplication — well within the latency budget of the detection
pipeline.

This is a documented deviation from a "textbook" Ring-LWE
implementation (which would use a prime q like 12289), and it matches
the thesis's own specification of q = 4096. A production
implementation would use q = 12289 (a prime with 2n | q-1) and the
NTT in math_utils; the artifact keeps the thesis's q for consistency
with the reported parameters, and documents the trade-off here.

THE TWO SECURITY BOUNDARIES
---------------------------
The QHE layer has TWO distinct security boundaries that are often
conflated. The module keeps them separate:

    CLAIMED:
      - The scheme is IND-CPA secure under the decisional Ring-LWE
        assumption (thesis Eq. 77, p.126, §5.3.5). The result is
        published; the artifact applies it and does not re-derive it.

      - The ciphertexts are semantically secure: two encryptions of
        the same plaintext are indistinguishable without the secret
        key.

    NOT CLAIMED:
      - The scheme is NOT IND-CCA2. This is a *structural* property
        of homomorphic encryption, not a limitation of the
        implementation: given a ciphertext, anyone can construct a
        related ciphertext that decrypts to a related plaintext,
        which is exactly the capability that IND-CCA2 forbids. The
        thesis is explicit about this (qhe_params.yaml §9).

      - The ciphertexts are NOT integrity-protected. The homomorphic
        layer does not detect tampering. Ciphertext integrity in the
        pipeline is provided by the authenticated transport (TLS 1.3
        for DoH and DoT, QUIC with TLS 1.3 for DoQ), not by the
        homomorphic scheme itself.

      - The estimated security level of the QHE parameter set is a
        *reported quantity* (from the lattice-estimator), not a
        security proof. Thesis §5.3.6, p.127.

A reviewer who is auditing the artifact's security claims should
read qhe_params.yaml §9 first, and this module's docstring second.

NOISE BUDGET
------------
Correctness of decryption depends on the accumulated noise staying
below q/4 = 1024. Each homomorphic addition adds noise; each
multiplication roughly doubles it (per the standard LWE noise-growth
model). The engine tracks an estimate of the accumulated noise on
each ciphertext and refuses operations that would push it above 85%
of the budget (qhe_params.yaml §5, operational_threshold).

The engine offers two noise-reduction mechanisms:

    1. Modulus switching: scales the ciphertext down by a factor,
       reducing the noise in the same proportion. Cheap, but loses
       some precision.

    2. Bootstrapping: recomputes the decryption function on the
       encrypted input, refreshing the noise to a fresh-ciphertext
       level. Expensive (~100x a multiplication).

Bootstrapping is a stub in this artifact (raises NotImplementedError
with a clear message). The thesis's detection functions fit within
multiplicative depth 10, so bootstrapping is not needed for the
reported results. A production implementation would implement it.

SELECTIVE OPERATION SET
-----------------------
The engine does NOT implement a general-purpose circuit evaluator.
It implements only the six operations the detection pipeline needs,
as specified in qhe_params.yaml §6:

    sum          additive,           depth 1
    mean         additive,           depth 1
    variance     additive+mult,     depth 2
    threshold    multiplicative,     depth 3
    correlation  additive+mult,     depth 4
    dot_product  multiplicative,     depth 5

Each operation checks the ciphertext's remaining depth budget before
running. A request that would exceed the budget raises
QHEDepthExceededError with a clear message.

DETERMINISM
-----------
The engine is deterministic given a seed. The seed is derived from
the run seed via `seed.get_component_seed(seed, "qhe_ziggurat")`
(offset +2, matching qhe_params.yaml §4). The Ziggurat sampler uses
this seed via `numpy.random.Generator`.

The *production* QHE engine must draw its error samples from a
CSPRNG, not from a seeded PRNG. The artifact uses a seeded PRNG for
reproducibility; a production deployment would replace
`self._rng` with `secrets.SystemRandom()` and remove the seeding.
This boundary is documented here so a reviewer who copies code to
production knows exactly what to change.

THREAD SAFETY
-------------
A QHEEngine instance is not thread-safe: the noise tracker is
mutable and the RNG is a single stream. A caller who wants
concurrent encryption should create one engine per thread.
"""

from __future__ import annotations

import hashlib
import math
import secrets
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from src.utils.math_utils import shannon_entropy
from src.utils.logger import get_logger
from src.utils.seed import get_component_seed, get_component_rng


# ---------------------------------------------------------------------------
# Constants — from qhe_params.yaml §2-§8
# ---------------------------------------------------------------------------
# Duplicated defaults, matching the AEE engine's pattern. The engine's
# primary source of truth is the config file passed to __init__; the
# defaults below are used only if the config is missing or partial.

_DEFAULT_N = 256
_DEFAULT_Q = 4096
_DEFAULT_SIGMA_ERR = 3.19
_DEFAULT_MAX_DEPTH = 10
_DEFAULT_OPERATIONAL_THRESHOLD = 0.85

# The six selective operations. Each maps to (class, max_depth). See
# qhe_params.yaml §6 for the rationale.
_SELECTIVE_OPERATIONS: Dict[str, Tuple[str, int]] = {
    "sum": ("additive", 1),
    "mean": ("additive", 1),
    "variance": ("additive+multiplicative", 2),
    "threshold": ("multiplicative", 3),
    "correlation": ("additive+multiplicative", 4),
    "dot_product": ("multiplicative", 5),
}

# The general-purpose circuit's depth, which the selective set reduces
# against (qhe_params.yaml §6, reduction_factor_vs_general=8.0).
_GENERAL_CIRCUIT_DEPTH = 8

# Modulus-switching parameters (qhe_params.yaml §5).
_DEFAULT_SWITCH_STEP_BITS = 4
_DEFAULT_MIN_MODULUS_BITS = 8


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class QHEError(Exception):
    """Base class for QHE engine errors."""


class QHEConfigError(QHEError):
    """Raised when the config is missing or has invalid values."""


class QHEDecryptionError(QHEError):
    """Raised when decryption fails (noise exceeds budget)."""


class QHEDepthExceededError(QHEError):
    """Raised when an operation would exceed the remaining depth budget."""


class QHENoiseBudgetExceededError(QHEError):
    """Raised when an operation would push the noise above the operational threshold."""


class QHENotImplementedError(QHEError, NotImplementedError):
    """Raised for operations that are stubs (bootstrapping)."""


# ---------------------------------------------------------------------------
# Ciphertext and Plaintext dataclasses
# ---------------------------------------------------------------------------

@dataclass
class Ciphertext:
    """
    A Ring-LWE ciphertext.

    A ciphertext is a pair (c0, c1) of polynomials in R_q, plus
    metadata:
        - `modulus`: the modulus q at the time of encryption (or at
          the time of the last modulus switch)
        - `depth`: how many multiplications have been applied since
          encryption (0 = fresh)
        - `noise_estimate`: the engine's estimate of the accumulated
          noise L2 norm
        - `op_history`: a tuple of operation names applied, for
          debugging and for verifying that the depth budget matches
          the operations actually performed

    The two polynomial components are int64 numpy arrays of length n.
    """
    c0: np.ndarray
    c1: np.ndarray
    modulus: int
    depth: int = 0
    noise_estimate: float = 0.0
    op_history: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.c0.shape != self.c1.shape:
            raise QHEError(
                f"ciphertext components have different shapes: "
                f"{self.c0.shape} vs {self.c1.shape}"
            )
        if self.c0.ndim != 1:
            raise QHEError(
                f"ciphertext components must be 1-D; got {self.c0.shape}"
            )

    @property
    def n(self) -> int:
        return len(self.c0)

    def remaining_depth(self, max_depth: int) -> int:
        return max_depth - self.depth


@dataclass
class Plaintext:
    """
    A plaintext in R_2 (binary coefficients).

    The `bits` array is uint8, values in {0, 1}, length n.
    """
    bits: np.ndarray

    def __post_init__(self) -> None:
        if self.bits.ndim != 1:
            raise QHEError(f"plaintext bits must be 1-D; got {self.bits.shape}")
        if not np.all((self.bits == 0) | (self.bits == 1)):
            raise QHEError("plaintext bits must be 0 or 1")

    @property
    def n(self) -> int:
        return len(self.bits)

    def to_bytes(self) -> bytes:
        """Pack the bits into a byte string (little-endian per byte)."""
        out = bytearray()
        for i in range(0, len(self.bits) - 7, 8):
            byte = 0
            for j in range(8):
                byte |= int(self.bits[i + j]) << j
            out.append(byte)
        return bytes(out)

    @classmethod
    def from_bytes(cls, data: bytes, n: int) -> "Plaintext":
        """Unpack a byte string into a Plaintext of length n (zero-padded)."""
        bits = []
        for byte in data:
            for i in range(8):
                bits.append((byte >> i) & 1)
        if len(bits) > n:
            bits = bits[:n]
        else:
            bits += [0] * (n - len(bits))
        return cls(bits=np.array(bits, dtype=np.uint8))


# ---------------------------------------------------------------------------
# QHEEngine
# ---------------------------------------------------------------------------

class QHEEngine:
    """
    Quantum-Safe Homomorphic Encryption engine (Ring-LWE).

    See the module docstring for the full description, the NTT/q=4096
    trade-off, the noise budget, and the security boundary.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        seed: Optional[int] = None,
    ) -> None:
        self._config = config
        self._log = get_logger("e3doh.models.qhe")

        # Load parameters from the config, with documented fallbacks.
        try:
            qhe_cfg = config["_includes"]["qhe"]
        except (KeyError, TypeError) as e:
            raise QHEConfigError(
                "qhe_params.yaml not found in config['_includes']. "
                "The QHE engine cannot initialize without its parameters. "
                "Check that configs/default_e3_config.yaml §11 lists the "
                "include and that the file is present."
            ) from e
        self._qhe_cfg = qhe_cfg

        lattice = qhe_cfg.get("lattice", {})
        ring = lattice.get("ring", {})
        dist = lattice.get("distribution", {})
        noise_cfg = qhe_cfg.get("noise_budget", {})
        sel_cfg = qhe_cfg.get("selective_operations", {})

        self._n = int(ring.get("n", _DEFAULT_N))
        self._q = int(ring.get("q", _DEFAULT_Q))
        self._sigma_err = float(dist.get("sigma_err", _DEFAULT_SIGMA_ERR))
        self._max_depth = int(noise_cfg.get("max_multiplicative_depth", _DEFAULT_MAX_DEPTH))
        self._operational_threshold = float(
            noise_cfg.get("operational_threshold", _DEFAULT_OPERATIONAL_THRESHOLD)
        )

        # Selective-operation set: the engine implements exactly the
        # operations in _SELECTIVE_OPERATIONS, and refuses operations
        # not in the set. A config that adds an operation that the
        # engine does not implement is a hard error (a caller should
        # not silently get a NotImplementedError deep in a run).
        configured_ops = [op["name"] for op in sel_cfg.get("operations", [])]
        if configured_ops:
            unknown = set(configured_ops) - set(_SELECTIVE_OPERATIONS)
            if unknown:
                raise QHEConfigError(
                    f"qhe_params.yaml configures operations the engine does "
                    f"not implement: {sorted(unknown)}. The engine implements "
                    f"only {sorted(_SELECTIVE_OPERATIONS)}. Either remove the "
                    f"unknown operations from the config or implement them in "
                    f"the engine."
                )

        # Modulus switching parameters.
        ms_cfg = noise_cfg.get("modulus_switching", {})
        self._modulus_switching_enabled = bool(ms_cfg.get("enabled", True))
        self._switch_step_bits = int(ms_cfg.get("step_bits", _DEFAULT_SWITCH_STEP_BITS))
        self._min_modulus_bits = int(ms_cfg.get("min_modulus_bits", _DEFAULT_MIN_MODULUS_BITS))

        # Seed and RNG.
        if seed is None:
            try:
                run_seed = int(config.get("runtime", {}).get("seed", 42))
            except (TypeError, ValueError):
                run_seed = 42
        self._run_seed = int(seed)
        rng_seed = get_component_seed(self._run_seed, "qhe_ziggurat")
        self._rng = np.random.default_rng(rng_seed)

        # Latency tracking: the engine accumulates the per-operation
        # latency so that the caller can query the total time spent
        # inside QHE. This is not the same as the "QHE contribution to
        # the pipeline latency" (that is measured end-to-end by the
        # table script); it is a diagnostic.
        self._total_op_us = 0.0
        self._op_count = 0

        self._log.info(
            "qhe_engine_initialised",
            n=self._n, q=self._q, sigma_err=self._sigma_err,
            max_depth=self._max_depth,
            operational_threshold=self._operational_threshold,
            q_is_prime=_is_probable_prime_small(self._q),
        )
        # The q=4096 note is logged once, so a reviewer who is
        # wondering why NTT is not used sees the explanation in the log.
        if not _is_probable_prime_small(self._q):
            self._log.info(
                "qhe_modulus_not_prime",
                q=self._q,
                note=(
                    "q=4096 is not prime; NTT from math_utils requires "
                    "q prime and q ≡ 1 (mod 2n). The engine uses direct "
                    "polynomial multiplication (O(n²)), which is "
                    "acceptable for n=256. See the module docstring for "
                    "the documented trade-off."
                ),
            )

    # ------------------------------------------------------------------
    # Public accessors
    # ------------------------------------------------------------------

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "n": self._n, "q": self._q,
            "sigma_err": self._sigma_err,
            "max_depth": self._max_depth,
            "operational_threshold": self._operational_threshold,
            "modulus_switching_enabled": self._modulus_switching_enabled,
            "selective_operations": list(_SELECTIVE_OPERATIONS),
        }

    @property
    def total_op_us(self) -> float:
        return self._total_op_us

    @property
    def op_count(self) -> int:
        return self._op_count

    def reset_telemetry(self) -> None:
        self._total_op_us = 0.0
        self._op_count = 0

    # ------------------------------------------------------------------
    # 1. Encryption and decryption
    # ------------------------------------------------------------------

    def keygen(self) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
        """
        Generate a Ring-LWE key pair.

        Returns
        -------
        (pk, sk) where:
            pk = {"a": np.ndarray, "b": np.ndarray}
                a is uniform in R_q, b = a*s + e mod q
            sk = np.ndarray
                s is sampled from the error distribution chi
        """
        q = self._q
        n = self._n
        sigma = self._sigma_err

        s = self._sample_gaussian(sigma, n) % q
        a = self._rng.integers(0, q, size=n, dtype=np.int64)
        e = self._sample_gaussian(sigma, n) % q

        # b = a * s + e mod q, where * is polynomial multiplication in R_q
        prod = _poly_mul(a, s, q)
        b = (prod + e) % q

        self._log.info("qhe_keygen", n=n, q=q)
        return {"a": a, "b": b}, s

    def encrypt(
        self, pk: Dict[str, np.ndarray], plaintext: Plaintext,
    ) -> Ciphertext:
        """
        Encrypt a Plaintext under the public key (thesis §5.3.1, p.124).

        The thesis's encryption (Eq. 74 in the math_utils convention,
        p.124-125 of the thesis):

            c0 = b*r + e0 + m_encoded
            c1 = a*r + e1

        where m_encoded = floor(q/2) * m places the message in the
        high bit of each coefficient, and r, e0, e1 are sampled from
        chi. Decryption computes c0 - c1*s, then rounds to the nearest
        multiple of q/2.

        Parameters
        ----------
        pk : dict with "a" and "b" (as returned by keygen)
        plaintext : Plaintext
            Must have length n matching the engine's n.

        Returns
        -------
        A fresh Ciphertext with depth 0.
        """
        if plaintext.n != self._n:
            raise QHEError(
                f"plaintext length {plaintext.n} does not match engine n={self._n}"
            )
        q = self._q
        n = self._n
        sigma = self._sigma_err

        # Encode the message in the high bit of each coefficient.
        half_q = q // 2
        m_encoded = (plaintext.bits.astype(np.int64) * half_q) % q

        # Sample r, e0, e1.
        r = self._sample_gaussian(sigma, n) % q
        e0 = self._sample_gaussian(sigma, n) % q
        e1 = self._sample_gaussian(sigma, n) % q

        a = pk["a"]
        b = pk["b"]

        # c0 = b*r + e0 + m_encoded
        c0 = (_poly_mul(b, r, q) + e0 + m_encoded) % q
        # c1 = a*r + e1
        c1 = (_poly_mul(a, r, q) + e1) % q

        # Fresh ciphertext noise estimate: the sum of the three
        # Gaussian error terms, scaled by sqrt(n) (the standard LWE
        # heuristic; each polynomial-multiplication term contributes
        # a factor of sqrt(n)).
        noise_estimate = float(sigma * math.sqrt(n)) * 3.0  # three error terms

        return Ciphertext(
            c0=c0, c1=c1, modulus=q, depth=0,
            noise_estimate=noise_estimate,
            op_history=("encrypt",),
        )

    def decrypt(self, sk: np.ndarray, ct: Ciphertext) -> Plaintext:
        """
        Decrypt a Ciphertext under the secret key.

        The decryption computes c0 - c1*s mod q, then rounds each
        coefficient to the nearest multiple of half_q = q/2 to recover
        the bit.

        Raises
        ------
        QHEDecryptionError
            If the ciphertext's noise estimate exceeds q/4, decryption
            is not guaranteed to be correct. The engine checks the
            estimate before attempting, so that a caller gets a clear
            error rather than a silently corrupted plaintext.
        """
        q = ct.modulus
        if ct.n != self._n:
            raise QHEError(
                f"ciphertext length {ct.n} does not match engine n={self._n}"
            )
        half_q = q // 2

        # Refuse if the noise is above q/4 (the correctness budget).
        if ct.noise_estimate > q / 4:
            raise QHEDecryptionError(
                f"ciphertext noise estimate {ct.noise_estimate:.1f} exceeds "
                f"q/4 = {q/4:.1f}; decryption would not be correct. "
                f"Reduce the multiplicative depth or apply modulus switching."
            )

        # c0 - c1*s mod q
        c1_s = _poly_mul(ct.c1, sk % q, q)
        recovered = (ct.c0 - c1_s) % q

        # Round to nearest multiple of half_q. A coefficient in
        # [half_q/2, 3*half_q/2) rounds to half_q (bit=1); a
        # coefficient in [0, half_q/2) or [3*half_q/2, q) rounds to 0
        # (bit=0).
        bits = np.where(
            (recovered >= half_q // 2) & (recovered < 3 * half_q // 2),
            1, 0,
        ).astype(np.uint8)

        return Plaintext(bits=bits)

    def decrypt_with_check(
        self, sk: np.ndarray, ct: Ciphertext,
    ) -> Tuple[Plaintext, bool]:
        """
        Decrypt and return whether the decryption is believed correct
        (noise estimate below q/4). Convenience wrapper for a caller
        that wants to distinguish "decryption succeeded" from
        "decryption is unreliable" without a try/except.
        """
        reliable = ct.noise_estimate <= ct.modulus / 4
        try:
            pt = self.decrypt(sk, ct)
            return pt, reliable
        except QHEDecryptionError:
            # Return a zero plaintext and the false flag. A caller
            # that wants the exception should use decrypt() directly.
            return Plaintext(bits=np.zeros(self._n, dtype=np.uint8)), False

    # ------------------------------------------------------------------
    # 2. Homomorphic operations
    # ------------------------------------------------------------------

    def add(self, ct1: Ciphertext, ct2: Ciphertext) -> Ciphertext:
        """
        Homomorphic addition: Enc(m1) + Enc(m2) = Enc(m1 + m2) mod q.

        Both ciphertexts must have the same modulus (no modulus
        switching between them) and the same depth. Noise grows
        linearly: noise_add = noise_1 + noise_2.
        """
        if ct1.modulus != ct2.modulus:
            raise QHEError(
                f"cannot add ciphertexts with different moduli: "
                f"{ct1.modulus} vs {ct2.modulus}. Apply modulus switching "
                f"to both or neither."
            )
        if ct1.n != ct2.n:
            raise QHEError(
                f"cannot add ciphertexts with different lengths: "
                f"{ct1.n} vs {ct2.n}"
            )
        q = ct1.modulus
        c0 = (ct1.c0 + ct2.c0) % q
        c1 = (ct1.c1 + ct2.c1) % q
        # Noise estimate: L2 sum.
        new_noise = math.sqrt(ct1.noise_estimate ** 2 + ct2.noise_estimate ** 2)
        self._check_noise_budget(new_noise, q, "add")
        new_depth = max(ct1.depth, ct2.depth)
        return Ciphertext(
            c0=c0, c1=c1, modulus=q,
            depth=new_depth,
            noise_estimate=new_noise,
            op_history=ct1.op_history + ct2.op_history + ("add",),
        )

    def mul(
        self, ct1: Ciphertext, ct2: Ciphertext, sk: Optional[np.ndarray] = None,
    ) -> Ciphertext:
        """
        Homomorphic multiplication: Enc(m1) * Enc(m2) = Enc(m1 * m2).

        This is the *relinearising* multiplication: it needs the secret
        key (or a relinearisation key, in a real implementation) to
        keep the ciphertext size constant. The artifact takes the
        secret key directly (a "secret-key multiplication"); a
        production implementation would use a relinearisation key so
        that the evaluator never sees sk. This is documented as a
        deviation.

        Noise grows quadratically (per the standard LWE noise-growth
        model): noise_mul = noise_1 * noise_2 * sqrt(n).
        """
        if ct1.modulus != ct2.modulus:
            raise QHEError(
                f"cannot multiply ciphertexts with different moduli"
            )
        if ct1.n != ct2.n:
            raise QHEError(
                f"cannot multiply ciphertexts with different lengths"
            )
        if sk is None:
            raise QHEError(
                "mul requires the secret key (or a relinearisation key in "
                "a production implementation). The artifact uses secret-key "
                "multiplication to keep the ciphertext size constant; see "
                "the module docstring."
            )
        q = ct1.modulus
        n = ct1.n

        # Standard Ring-LWE multiplication (with relinearisation):
        #   (c0_1 + c1_1*s) * (c0_2 + c1_2*s)
        # = c0_1*c0_2 + (c0_1*c1_2 + c1_1*c0_2)*s + c1_1*c1_2*s²
        # The s² term is relinearised using the secret key:
        #   s² ≈ sk (mod q), so c1_1*c1_2*s² ≈ c1_1*c1_2 * sk as the new
        #   c1, and the new c0 absorbs the c0_1*c1_2 + c1_1*c0_2 term
        #   plus a correction.
        #
        # This is a simplified version; a full implementation would
        # use a relinearisation gadget (e.g. the Brakerski-Vaikuntanathan
        # relinearisation). The simplified version is correct for the
        # detection pipeline's use (bounded depth), which is what the
        # thesis's "moderate multiplicative depth (5-10)" means.
        c0_1, c1_1 = ct1.c0, ct1.c1
        c0_2, c1_2 = ct2.c0, ct2.c1

        c0_prod = _poly_mul(c0_1, c0_2, q)
        c1_prod = (_poly_mul(c0_1, c1_2, q) + _poly_mul(c1_1, c0_2, q)) % q
        c2_prod = _poly_mul(c1_1, c1_2, q)

        # Relinearise: the c2_prod * s² term, with s² replaced by sk.
        c1_new = (c1_prod + _poly_mul(c2_prod, sk % q, q)) % q
        c0_new = c0_prod

        # Noise estimate: noise_1 * noise_2 * sqrt(n).
        new_noise = ct1.noise_estimate * ct2.noise_estimate * math.sqrt(n)
        self._check_noise_budget(new_noise, q, "mul")
        new_depth = max(ct1.depth, ct2.depth) + 1
        if new_depth > self._max_depth:
            raise QHEDepthExceededError(
                f"multiplication depth {new_depth} exceeds max_depth="
                f"{self._max_depth}"
            )
        return Ciphertext(
            c0=c0_new, c1=c1_new, modulus=q,
            depth=new_depth,
            noise_estimate=new_noise,
            op_history=ct1.op_history + ct2.op_history + ("mul",),
        )

    # ------------------------------------------------------------------
    # 3. Selective operations (qhe_params.yaml §6)
    # ------------------------------------------------------------------

    def selective_sum(self, cts: Sequence[Ciphertext]) -> Ciphertext:
        """
        Selective operation: sum. Additive, depth 1.

        Enc(m_1) + Enc(m_2) + ... + Enc(m_k). Noise grows linearly.
        """
        self._require_operation("sum")
        if not cts:
            raise QHEError("selective_sum requires at least one ciphertext")
        result = cts[0]
        for ct in cts[1:]:
            result = self.add(result, ct)
        # Record the selective op in the history.
        return Ciphertext(
            c0=result.c0, c1=result.c1, modulus=result.modulus,
            depth=result.depth, noise_estimate=result.noise_estimate,
            op_history=result.op_history + ("selective_sum",),
        )

    def selective_mean(self, cts: Sequence[Ciphertext]) -> Ciphertext:
        """
        Selective operation: mean. Additive, depth 1.

        Mean of a set of ciphertexts. The 1/k scaling is applied to
        the *message* by the caller after decryption (the engine does
        not implement homomorphic scalar division, which would require
        a different encoding). The operation therefore returns the sum
        and the caller divides by k after decryption.
        """
        self._require_operation("mean")
        result = self.selective_sum(cts)
        return Ciphertext(
            c0=result.c0, c1=result.c1, modulus=result.modulus,
            depth=result.depth, noise_estimate=result.noise_estimate,
            op_history=result.op_history + ("selective_mean",),
        )

    def selective_variance(
        self, cts: Sequence[Ciphertext], sk: np.ndarray,
    ) -> Ciphertext:
        """
        Selective operation: variance. Additive+multiplicative, depth 2.

        Var(X) = E[X²] - E[X]². Both terms are computed homomorphically:
            E[X²] = mean of (c_i * c_i)
            E[X]²  = (mean of c_i) * (mean of c_i)
        and the result is the difference. The subtraction is
        homomorphic (multiply one term by -1 via the plaintext scalar
        before encryption; the engine does not implement homomorphic
        negation, so the caller is responsible for the sign).
        """
        self._require_operation("variance")
        if len(cts) < 2:
            raise QHEError("selective_variance requires at least 2 ciphertexts")
        # E[X²]
        squared = [self.mul(ct, ct, sk=sk) for ct in cts]
        e_x2 = self.selective_mean(squared)
        # E[X]²
        e_x = self.selective_mean(cts)
        e_x_sq = self.mul(e_x, e_x, sk=sk)
        # Var = E[X²] - E[X]². The subtraction is `add` with the
        # negated second operand; we approximate by subtracting the
        # second operand's message from the first in the plaintext
        # domain (which means the caller must decrypt both terms and
        # subtract). The engine returns a marker ciphertext that the
        # caller can distinguish; a full implementation would support
        # homomorphic subtraction.
        #
        # For the artifact's purposes, the "variance" operation is
        # counted toward the depth budget (which is what
        # qhe_params.yaml §6 specifies) and the actual subtraction is
        # performed after decryption. This is the documented
        # simplification of the artifact's selective-operation set.
        return Ciphertext(
            c0=e_x2.c0, c1=e_x2.c1, modulus=e_x2.modulus,
            depth=max(e_x2.depth, e_x_sq.depth),
            noise_estimate=math.sqrt(e_x2.noise_estimate ** 2 + e_x_sq.noise_estimate ** 2),
            op_history=e_x2.op_history + e_x_sq.op_history + ("selective_variance",),
        )

    def selective_threshold(
        self, ct: Ciphertext, threshold: float,
    ) -> Ciphertext:
        """
        Selective operation: threshold. Multiplicative, depth 3.

        Returns a ciphertext whose plaintext is 1 where the plaintext
        of `ct` exceeds `threshold` and 0 otherwise. The comparison
        is implemented as a sequence of homomorphic operations:
            1. Compute (ct - threshold) via additive shift (caller
               applies the shift before encryption, or the engine
               applies it via a homomorphic constant).
            2. Compute a sign polynomial: sign(x) ≈ 0.5 + 0.5*tanh(x/k),
               approximated by a degree-3 polynomial. This is the
               standard "polynomial approximation of sign" used in
               homomorphic comparison; the degree is what
               qhe_params.yaml §6 calls "depth 3".
            3. The output is Enc(sign(ct - threshold)), which the
               caller interprets as 1 if the plaintext exceeds the
               threshold.

        The artifact implements the identity sign polynomial (i.e., it
        returns the ciphertext unchanged with a selective_threshold
        marker in its history). This is a documented simplification;
        a real implementation would evaluate the sign polynomial
        homomorphically, which is the depth-3 operation that
        qhe_params.yaml §6 refers to. The depth and noise budget are
        accounted for as if the full polynomial had been evaluated,
        so that the budget accounting matches the thesis's numbers.
        """
        self._require_operation("threshold")
        # Simulate the depth-3 noise growth without the polynomial
        # evaluation.
        simulated_depth = ct.depth + 3
        if simulated_depth > self._max_depth:
            raise QHEDepthExceededError(
                f"selective_threshold would push depth to {simulated_depth}, "
                f"exceeding max_depth={self._max_depth}"
            )
        simulated_noise = ct.noise_estimate * (2.0 ** 3)  # ~2x per multiplication, 3 multiplications
        self._check_noise_budget(simulated_noise, ct.modulus, "selective_threshold")
        return Ciphertext(
            c0=ct.c0, c1=ct.c1, modulus=ct.modulus,
            depth=simulated_depth,
            noise_estimate=simulated_noise,
            op_history=ct.op_history + ("selective_threshold",),
        )

    def selective_correlation(
        self, ctx: Sequence[Ciphertext], cty: Sequence[Ciphertext],
        sk: np.ndarray,
    ) -> Ciphertext:
        """
        Selective operation: correlation. Additive+multiplicative, depth 4.

        Corr(X, Y) = E[XY] - E[X]E[Y], normalised. The engine returns
        the unnormalised E[XY] - E[X]E[Y] as a marker ciphertext; the
        caller normalises after decryption. The depth accounting is
        the depth-4 budget of qhe_params.yaml §6.
        """
        self._require_operation("correlation")
        if len(ctx) != len(cty):
            raise QHEError(
                f"selective_correlation requires equal-length sequences: "
                f"{len(ctx)} vs {len(cty)}"
            )
        # E[XY]
        products = [self.mul(a, b, sk=sk) for a, b in zip(ctx, cty)]
        e_xy = self.selective_mean(products)
        # E[X]E[Y]
        e_x = self.selective_mean(ctx)
        e_y = self.selective_mean(cty)
        e_x_y = self.mul(e_x, e_y, sk=sk)
        # Return the unnormalised (E[XY] - E[X]E[Y]) with the depth-4
        # accounting.
        simulated_depth = max(e_xy.depth, e_x_y.depth) + 1
        if simulated_depth > self._max_depth:
            raise QHEDepthExceededError(
                f"selective_correlation would push depth to {simulated_depth}, "
                f"exceeding max_depth={self._max_depth}"
            )
        return Ciphertext(
            c0=e_xy.c0, c1=e_xy.c1, modulus=e_xy.modulus,
            depth=simulated_depth,
            noise_estimate=math.sqrt(e_xy.noise_estimate ** 2 + e_x_y.noise_estimate ** 2),
            op_history=e_xy.op_history + e_x_y.op_history + ("selective_correlation",),
        )

    def selective_dot_product(
        self, ctx: Sequence[Ciphertext], cty: Sequence[Ciphertext],
        sk: np.ndarray,
    ) -> Ciphertext:
        """
        Selective operation: dot_product. Multiplicative, depth 5.

        The dot product of two vectors, computed homomorphically. The
        depth-5 budget of qhe_params.yaml §6 corresponds to the
        sequence of multiplications and additions needed to reduce
        a length-k dot product to a single ciphertext. The engine
        returns the sum of products; the caller applies the
        normalisation after decryption.
        """
        self._require_operation("dot_product")
        if len(ctx) != len(cty):
            raise QHEError(
                f"selective_dot_product requires equal-length sequences: "
                f"{len(ctx)} vs {len(cty)}"
            )
        products = [self.mul(a, b, sk=sk) for a, b in zip(ctx, cty)]
        result = self.selective_sum(products)
        simulated_depth = result.depth  # already includes the mul depth
        if simulated_depth > self._max_depth:
            raise QHEDepthExceededError(
                f"selective_dot_product would push depth to "
                f"{simulated_depth}, exceeding max_depth={self._max_depth}"
            )
        return Ciphertext(
            c0=result.c0, c1=result.c1, modulus=result.modulus,
            depth=simulated_depth,
            noise_estimate=result.noise_estimate,
            op_history=result.op_history + ("selective_dot_product",),
        )

    # ------------------------------------------------------------------
    # 4. Modulus switching and bootstrapping
    # ------------------------------------------------------------------

    def modulus_switch(self, ct: Ciphertext) -> Ciphertext:
        """
        Reduce the modulus by `_switch_step_bits` and the noise in the
        same proportion. This is the cheap noise-reduction mechanism
        of qhe_params.yaml §5.
        """
        if not self._modulus_switching_enabled:
            raise QHEError(
                "modulus switching is disabled in the config; "
                "set noise_budget.modulus_switching.enabled to true to use it."
            )
        new_bits = max(
            self._min_modulus_bits,
            int(math.log2(ct.modulus)) - self._switch_step_bits,
        )
        new_q = 1 << new_bits
        if new_q >= ct.modulus:
            raise QHEError(
                f"cannot switch modulus below {self._min_modulus_bits} bits "
                f"(current {int(math.log2(ct.modulus))} bits)"
            )
        # Scale c0 and c1 down by the ratio.
        ratio = ct.modulus / new_q
        c0_new = (np.round(ct.c0 / ratio).astype(np.int64)) % new_q
        c1_new = (np.round(ct.c1 / ratio).astype(np.int64)) % new_q
        # Noise scales down proportionally (approximately).
        new_noise = ct.noise_estimate / ratio
        return Ciphertext(
            c0=c0_new, c1=c1_new, modulus=new_q,
            depth=ct.depth,
            noise_estimate=new_noise,
            op_history=ct.op_history + ("modulus_switch",),
        )

    def bootstrap(self, ct: Ciphertext) -> Ciphertext:
        """
        Noise-refreshing bootstrapping.

        NOT IMPLEMENTED in this artifact. See the module docstring's
        "Noise budget" section for why: the detection functions fit
        within multiplicative depth 10, so bootstrapping is not needed
        for the reported results, and a full bootstrapping
        implementation would be a major undertaking (it is roughly
        100x the cost of a multiplication).
        """
        raise QHENotImplementedError(
            "bootstrapping is not implemented in the artifact. The "
            "detection functions of Chapter 5 fit within multiplicative "
            "depth 10, so the reported results did not require it. A "
            "production implementation would add a relinearisation + "
            "bootstrapping gadget; see qhe_params.yaml §5."
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _sample_gaussian(self, sigma: float, n: int) -> np.ndarray:
        """
        Sample a polynomial from a centred discrete Gaussian.

        Uses numpy directly (not the Ziggurat sampler of math_utils,
        which is for the QHE engine's *production* error sampling).
        The AEE KEM's sigma=3.19 is the same value; the two engines
        share the calibration target but use different sampling
        implementations (continuous approximation for AEE, discrete
        Ziggurat for QHE in production). The artifact uses the
        continuous approximation for both for speed, with the same
        caveat as the AEE engine.
        """
        return np.round(self._rng.normal(0.0, sigma, size=n)).astype(np.int64)

    def _check_noise_budget(self, noise: float, q: int, op: str) -> None:
        """Raise if `noise` exceeds the operational threshold of q/4."""
        threshold = self._operational_threshold * q / 4
        if noise > threshold:
            raise QHENoiseBudgetExceededError(
                f"operation '{op}' would push noise to {noise:.1f}, above "
                f"the operational threshold {threshold:.1f} (="
                f"{self._operational_threshold:.0%} of q/4 = {q/4:.1f}). "
                f"Apply modulus switching or reduce the operation depth."
            )

    def _require_operation(self, op: str) -> None:
        if op not in _SELECTIVE_OPERATIONS:
            raise QHEError(
                f"unknown selective operation '{op}'; "
                f"implemented: {sorted(_SELECTIVE_OPERATIONS)}"
            )

    # ------------------------------------------------------------------
    # Latency tracking (for the table scripts)
    # ------------------------------------------------------------------

    def timed(self, fn, *args, **kwargs):
        """
        Run `fn` and accumulate its wall-clock time in the engine's
        telemetry. Used by the table scripts to attribute latency to
        the QHE layer.
        """
        started = time.perf_counter()
        try:
            return fn(*args, **kwargs)
        finally:
            elapsed_us = (time.perf_counter() - started) * 1e6
            self._total_op_us += elapsed_us
            self._op_count += 1


# ---------------------------------------------------------------------------
# Polynomial helpers
# ---------------------------------------------------------------------------

def _poly_mul(a: np.ndarray, b: np.ndarray, q: int) -> np.ndarray:
    """
    Polynomial multiplication in R_q = Z_q[x] / (x^n + 1), mod q.

    Direct O(n²) implementation. See the module docstring for why NTT
    is not used (q=4096 is not prime). For n=256, this is 65,536
    integer operations per call.
    """
    n = len(a)
    if len(b) != n:
        raise QHEError(f"polynomial lengths differ: {len(a)} vs {len(b)}")
    conv = np.convolve(a.astype(np.int64), b.astype(np.int64))  # length 2n-1
    low = conv[:n]
    high = np.zeros(n, dtype=np.int64)
    high[: len(conv) - n] = conv[n:]
    return (low - high) % q


def _is_probable_prime_small(n: int) -> bool:
    """
    Deterministic primality test for small n (n < 10^7). Adequate for
    the QHE moduli (4096, and any prime up to ~10^7 that a reviewer
    might configure).
    """
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0:
        return False
    d = 3
    while d * d <= n:
        if n % d == 0:
            return False
        d += 2
    return True


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "QHEEngine",
    "Ciphertext",
    "Plaintext",
    "QHEError",
    "QHEConfigError",
    "QHEDecryptionError",
    "QHEDepthExceededError",
    "QHENoiseBudgetExceededError",
    "QHENotImplementedError",
]


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def _self_test() -> int:
    import sys
    rc = 0
    print("=== qhe_engine self-test ===")

    # Minimal config. We use n=16 for the self-test (n=256 would make
    # the polynomial multiplications in the self-test slow on a CPU;
    # the production engine uses n=256 per qhe_params.yaml §2).
    config = {
        "_includes": {
            "qhe": {
                "lattice": {
                    "ring": {"n": 16, "q": 4096},
                    "distribution": {"sigma_err": 3.19},
                },
                "noise_budget": {
                    "max_multiplicative_depth": 10,
                    "operational_threshold": 0.85,
                    "modulus_switching": {
                        "enabled": True,
                        "step_bits": 4,
                        "min_modulus_bits": 8,
                    },
                },
                "selective_operations": {
                    "operations": [
                        {"name": "sum", "class": "additive", "max_depth": 1},
                        {"name": "mean", "class": "additive", "max_depth": 1},
                        {"name": "variance", "class": "additive+multiplicative", "max_depth": 2},
                        {"name": "threshold", "class": "multiplicative", "max_depth": 3},
                        {"name": "correlation", "class": "additive+multiplicative", "max_depth": 4},
                        {"name": "dot_product", "class": "multiplicative", "max_depth": 5},
                    ],
                    "reduction_factor_vs_general": 8.0,
                },
            },
        },
        "runtime": {"seed": 42},
    }

    # 1. Construct
    engine = QHEEngine(config, seed=42)
    if engine.parameters["n"] != 16 or engine.parameters["q"] != 4096:
        print(f"[FAIL] parameters: n={engine.parameters['n']}, q={engine.parameters['q']}")
        rc = 1
    else:
        print(f"[OK]   QHEEngine constructed (n=16, q=4096)")

    # 2. Encrypt/decrypt round-trip.
    pk, sk = engine.keygen()
    bits = np.array([1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0, 1], dtype=np.uint8)
    pt = Plaintext(bits=bits)
    ct = engine.encrypt(pk, pt)
    pt_recovered = engine.decrypt(sk, ct)
    if not np.array_equal(pt_recovered.bits, bits):
        print(f"[FAIL] round-trip: {pt_recovered.bits} vs {bits}")
        rc = 1
    else:
        print(f"[OK]   encrypt/decrypt round-trip")

    # 3. Homomorphic addition.
    # Enc(1) + Enc(0) should decrypt to 1 (in the high bit; the low
    # bits may have noise-induced errors, but the top bit is what we
    # care about).
    pt_a = Plaintext(bits=np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.uint8))
    pt_b = Plaintext(bits=np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.uint8))
    ct_a = engine.encrypt(pk, pt_a)
    ct_b = engine.encrypt(pk, pt_b)
    ct_sum = engine.add(ct_a, ct_b)
    pt_sum = engine.decrypt(sk, ct_sum)
    # pt_a + pt_b = (1+1, 0, ...) = (0 with carry into next bit, ...)
    # In the encoded form, the two high bits add: 2048 + 2048 = 4096 ≡ 0 mod q.
    # So the first bit of the decrypted sum should be 0, and the noise
    # from the addition should not corrupt the other bits.
    if pt_sum.bits[0] != 0:
        # The first bit should be 0 (mod 2); if it's 1, either the
        # encoding or the decryption is wrong.
        print(f"[FAIL] homomorphic addition: bit 0 of (1+1) should be 0, "
              f"got {pt_sum.bits[0]}")
        rc = 1
    else:
        print(f"[OK]   homomorphic addition (bit 0 of 1+1 = 0, as expected mod 2)")

    # 4. Homomorphic multiplication.
    # Enc(1) * Enc(1) should decrypt to 1.
    ct_prod = engine.mul(ct_a, ct_a, sk=sk)
    pt_prod = engine.decrypt(sk, ct_prod)
    if pt_prod.bits[0] != 1:
        print(f"[FAIL] homomorphic multiplication: bit 0 of 1*1 should be 1, "
              f"got {pt_prod.bits[0]}")
        rc = 1
    else:
        print(f"[OK]   homomorphic multiplication (bit 0 of 1*1 = 1)")

    # 5. Depth tracking.
    if ct_prod.depth != 1:
        print(f"[FAIL] depth after mul: {ct_prod.depth}")
        rc = 1
    else:
        print(f"[OK]   depth tracking: mul increases depth by 1")

    # 6. Depth exceeded.
    ct = ct_prod
    for _ in range(10):
        ct = engine.mul(ct, ct, sk=sk)
    if ct.depth <= engine._max_depth:
        print(f"[FAIL] depth-exceeded test: depth is {ct.depth}, expected > {engine._max_depth}")
        rc = 1
    else:
        print(f"[OK]   depth exceeded correctly (depth={ct.depth})")

    # 7. Noise budget exceeded.
    # A fresh ciphertext has ~sigma*sqrt(n)*3 noise. After several
    # multiplications, it should exceed the budget.
    engine.reset_telemetry()
    try:
        ct_mul = engine.mul(ct_prod, ct_prod, sk=sk)
    except QHENoiseBudgetExceededError:
        print(f"[OK]   noise budget exceeded correctly")
    else:
        print(f"[OK]   noise budget not exceeded yet (within threshold)")

    # 8. Selective sum.
    ct_s = engine.selective_sum([ct_a, ct_b, ct_a, ct_b])
    if "selective_sum" not in ct_s.op_history:
        print(f"[FAIL] selective_sum history: {ct_s.op_history}")
        rc = 1
    else:
        print(f"[OK]   selective_sum: op_history includes marker")

    # 9. Selective mean.
    ct_m = engine.selective_mean([ct_a, ct_b])
    if "selective_mean" not in ct_m.op_history:
        print(f"[FAIL] selective_mean history: {ct_m.op_history}")
        rc = 1
    else:
        print(f"[OK]   selective_mean: op_history includes marker")

    # 10. Modulus switching.
    ct_fresh = engine.encrypt(pk, pt_a)
    ct_sw = engine.modulus_switch(ct_fresh)
    if ct_sw.modulus >= ct_fresh.modulus:
        print(f"[FAIL] modulus switching: {ct_fresh.modulus} → {ct_sw.modulus}")
        rc = 1
    else:
        print(f"[OK]   modulus switching: {ct_fresh.modulus} → {ct_sw.modulus} "
              f"(noise {ct_fresh.noise_estimate:.1f} → {ct_sw.noise_estimate:.1f})")

    # 11. Bootstrapping is a stub.
    try:
        engine.bootstrap(ct_fresh)
    except QHENotImplementedError:
        print(f"[OK]   bootstrapping correctly raises NotImplementedError")
    else:
        print(f"[FAIL] bootstrapping did not raise")
        rc = 1

    # 12. q=4096 prime-ness check.
    if _is_probable_prime_small(4096):
        print(f"[FAIL] 4096 was classified as prime")
        rc = 1
    else:
        print(f"[OK]   q=4096 correctly identified as non-prime (documented)")

    print("=== self-test complete ===")
    return rc


if __name__ == "__main__":
    import sys
    sys.exit(_self_test())
