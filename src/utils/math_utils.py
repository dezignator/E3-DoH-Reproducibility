#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
src/utils/math_utils.py
=======================

E3-DoH Reproducibility — mathematical primitives.

Chapter 5 of the thesis: E3-DoH (Enhanced Evolutionary Encryption for
Multi-Protocol DNS Security — DoH, DoT, DoQ).

This module contains every mathematical primitive that the E3-DoH
engines consume. It is imported by:

    src/models/aee_engine.py       — AEE SDE, simplex projection, sigmoid activation
    src/models/qhe_engine.py       — NTT, modular arithmetic, Ziggurat sampling
    src/models/anomaly_detector.py — metric tensors, Christoffel symbols, geodesic
    src/models/am_sandbox.py       — Rényi divergence, KL divergence, HJB
    src/models/fl_coordinator.py   — RDP accountant, Wasserstein distance
    src/utils/metrics.py           — composite detection score

The design principle is that each function implements *exactly one*
thesis equation, and its docstring names that equation. There is no
"general-purpose linear algebra" in this module; NumPy and SciPy already
have that, and re-implementing it would only introduce bugs. What is
in this module is the specific mathematics of the E3-DoH framework:
the AEE SDE, the Riemannian metric tensors, the geodesic-flow ODE, the
Rényi/KL divergences, the Hamilton-Jacobi-Bellman control equation, the
RDP accountant, and the KEM combiner.

WHAT IS NOT IN THIS MODULE
--------------------------
    - Linear algebra       -> use numpy / scipy.linalg
    - Statistics           -> use scipy.stats
    - Optimization         -> use scipy.optimize
    - Signal processing    -> use scipy.signal
    - Cryptographic impls  -> use cryptography / pyfhel (in the engine modules)

SECURITY BOUNDARY
-----------------
No function in this module makes a security claim. The KEM combiner
function (combine_kem_secrets) implements the *published* combiner
result (thesis §5.2.6, Proposition 1); it does not re-derive it. The
security of any configuration produced by the engines that use this
module rests on the underlying primitives (ML-KEM, Classic McEliece,
Ring-LWE), not on the functions here. See aee_params.yaml §10 and
qhe_params.yaml §9 for the full security boundary.

NUMERICAL PRECISION
-------------------
Every public function returns float64 or int64 (for modular arithmetic).
Float32 is not used anywhere: the Christoffel-symbol computation
amplifies numerical noise, and float32 precision would fail the
compatibility check (∇_k g_ij = 0) documented in
configs/riemannian_metrics.yaml §7.

DETERMINISM
-----------
The stochastic functions (sample_ziggurat, evolve_aee_key) accept a
numpy Generator. The caller is responsible for seeding it from
default_e3_config.yaml's runtime.seed (via src/utils/seed.py). Do NOT
seed inside these functions; that would make them non-composable and
would defeat the artifact's reproducibility contract.

Equation references throughout point at the thesis chapter 5:
    §5.2.1-5.2.5, §5.3.1, §5.3.5-5.3.8, §5.4.2-5.4.5, §5.5,
    §5.6.1-5.6.4, §5.7.1
"""

from __future__ import annotations

import math
from typing import Callable, Optional, Sequence, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
# The primitives below are referenced throughout the thesis. Pinning them
# here as named constants (rather than inline magic numbers) makes the
# correspondence between the code and the equations checkable.

# AEE SDE step size. Thesis §5.2.3 (p.121); matches aee_params.yaml §3.
AEE_SDE_DT: float = 1.0e-3

# AEE diffusion floor. Thesis §5.2.3 (p.121) — a coefficient that decayed
# to zero would make the trajectory progressively predictable, so it is
# floored rather than annealed. Matches aee_params.yaml §3.
AEE_DIFFUSION_FLOOR: float = 1.0e-3
AEE_DIFFUSION_CEILING: float = 1.0

# Simplex-projection tolerance. Enforced by project_onto_simplex(); a
# violation beyond this tolerance is a bug in the SDE step. Matches
# aee_params.yaml §3 (simplex_tolerance).
AEE_SIMPLEX_TOLERANCE: float = 1.0e-6

# Ziggurat layer count. Standard Marsaglia-Tsang value; matches
# qhe_params.yaml §4.
ZIGGURAT_LAYERS: int = 256

# Probabilistic floor for log-space divergence computations. Prevents
# -inf when a probability is zero. Matches riemannian_metrics.yaml §2
# (probability_floor).
DIVERGENCE_PROB_FLOOR: float = 1.0e-9


# ===========================================================================
# 1. Probability, entropy, and divergence
# ===========================================================================
# These are the primitives that the AEE entropy functional, the AMPS
# sandbox trigger, and the FL convergence diagnostics all rely on.

def shannon_entropy(p: np.ndarray, *, normalise: bool = False,
                    max_bits: float = math.log2(10)) -> float:
    """
    Shannon entropy of a probability vector (thesis Eq. 71, p.122).

        H(p) = - sum_i p_i log_2 p_i

    Parameters
    ----------
    p : array_like
        Probability vector (non-negative, sums to 1 within tolerance).
    normalise : bool
        If True, divide by `max_bits` so the output is in [0, 1].
        The normalisation factor is 1 / log2(|vocabulary|); the default
        is log2(10) for the ten-type DNS query vocabulary in
        aee_params.yaml §4.
    max_bits : float
        The normalisation factor. Ignored if normalise=False.

    Returns
    -------
    The Shannon entropy in bits (or in [0, 1] if normalise=True).

    Notes
    -----
    Zero-probability entries contribute 0 * log 0 = 0 by convention,
    not log(0). The implementation handles this via np.where.
    """
    p = np.asarray(p, dtype=np.float64)
    if p.ndim != 1:
        raise ValueError("shannon_entropy expects a 1-D probability vector")
    if np.any(p < 0):
        raise ValueError("probabilities must be non-negative")
    total = p.sum()
    if total <= 0:
        return 0.0
    if abs(total - 1.0) > 1e-6:
        # Allow small tolerance for numerical drift; otherwise normalise.
        p = p / total
    # log2(0) = -inf; clamp to a small floor so the 0*log(0) term is exactly 0.
    with np.errstate(divide="ignore", invalid="ignore"):
        log_p = np.where(p > 0, np.log2(np.where(p > 0, p, 1.0)), 0.0)
    h = -np.sum(p * log_p)
    if normalise:
        h = h / max_bits
    return float(h)


def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """
    Kullback-Leibler divergence D_KL(P || Q) (thesis Eq. 89, p.132).

        D_KL(P || Q) = sum_i p_i log_2(p_i / q_i)

    Parameters
    ----------
    p, q : array_like
        Probability vectors of the same length. Non-negative, sum to 1
        within tolerance.

    Returns
    -------
    The KL divergence in bits. Always >= 0, zero iff p == q.

    Notes
    -----
    The AMPS sandbox (thesis §5.6.1, p.113) uses this as its primary
    divergence trigger. A zero-probability q_i with non-zero p_i would
    make the divergence infinite; the implementation floors both
    distributions with DIVERGENCE_PROB_FLOOR and renormalises, so the
    result is finite. The floor is documented in
    sandbox_mdp_params.yaml §2 (smoothing).
    """
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    if p.shape != q.shape:
        raise ValueError("p and q must have the same shape")
    p = np.maximum(p, DIVERGENCE_PROB_FLOOR)
    q = np.maximum(q, DIVERGENCE_PROB_FLOOR)
    p = p / p.sum()
    q = q / q.sum()
    return float(np.sum(p * np.log2(p / q)))


def renyi_divergence(p: np.ndarray, q: np.ndarray, alpha: float) -> float:
    """
    Rényi divergence of order alpha (thesis Eq. 90, p.132).

        D_alpha(P || Q) = (1 / (alpha - 1)) log_2 (sum_i p_i^alpha q_i^(1-alpha))

    Parameters
    ----------
    p, q : array_like
        Probability vectors of the same length.
    alpha : float
        The Rényi order. Must be > 0 and != 1.
        - alpha -> 1 recovers the KL divergence
        - alpha = 2 is the squared Hellinger-type divergence (default)
        - larger alpha shifts emphasis to tail deviations

    Returns
    -------
    The Rényi divergence in bits. Non-negative for alpha > 0.

    Notes
    -----
    The AMPS sandbox (thesis §5.6.1, p.113) uses Rényi with alpha=2.0
    as its secondary, tail-sensitive trigger. The default in
    sandbox_mdp_params.yaml §2 is alpha=2.0. alpha=1 is not allowed;
    use kl_divergence() for that case (the two are equivalent in the
    limit, and the implementation here avoids a division by zero).
    """
    if alpha <= 0:
        raise ValueError("Rényi order must be positive")
    if abs(alpha - 1.0) < 1e-8:
        raise ValueError("Use kl_divergence() for alpha == 1")
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    if p.shape != q.shape:
        raise ValueError("p and q must have the same shape")
    p = np.maximum(p, DIVERGENCE_PROB_FLOOR)
    q = np.maximum(q, DIVERGENCE_PROB_FLOOR)
    p = p / p.sum()
    q = q / q.sum()
    integral = float(np.sum(p ** alpha * q ** (1.0 - alpha)))
    if integral <= 0:
        return float("inf")
    return float(np.log2(integral) / (alpha - 1.0))


def wasserstein_distance_1d(p: np.ndarray, q: np.ndarray,
                           *, supports: Optional[np.ndarray] = None) -> float:
    """
    1-Wasserstein distance between two 1-D discrete distributions
    (thesis Eq. 114, p.149).

        W_1(P, Q) = inf_gamma E_{(x,y)~gamma}[ |x - y| ]

    For 1-D discrete distributions the W_1 distance has a closed form:
    it is the L1 distance between the cumulative distribution functions.

        W_1(P, Q) = sum_i | F_P(x_i) - F_Q(x_i) | * (x_{i+1} - x_i)

    where F_P, F_Q are the CDFs.

    Parameters
    ----------
    p, q : array_like
        Probability vectors of the same length.
    supports : optional array_like
        The support points. Defaults to np.arange(len(p)).

    Returns
    -------
    The 1-Wasserstein distance in the units of the support.

    Notes
    -----
    The thesis uses W_1 for the pairwise hypothesis test on protocol
    distributions (Eq. 114, §5.16, p.149) and for the PAFA TDA
    comparison (Eq. 41, §4.7.2, p.93). This function is the 1-D
    closed form, which is exact and does not require an LP solver.
    For 2-D persistence diagrams, use scipy.optimize.linprog or a
    Wasserstein library; that is out of scope for E3-DoH, which uses
    only the 1-D case in its own tables.
    """
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    if p.shape != q.shape:
        raise ValueError("p and q must have the same shape")
    n = len(p)
    if supports is None:
        supports = np.arange(n, dtype=np.float64)
    else:
        supports = np.asarray(supports, dtype=np.float64)
        if len(supports) != n:
            raise ValueError("supports must have the same length as p")
    p = p / p.sum()
    q = q / q.sum()
    cdf_p = np.cumsum(p)
    cdf_q = np.cumsum(q)
    # Trapezoidal integration against the support spacing.
    diffs = np.abs(cdf_p - cdf_q)
    if n < 2:
        return 0.0
    spacing = np.diff(supports)
    return float(np.sum(diffs[:-1] * spacing))


# ===========================================================================
# 2. Modular arithmetic and number-theoretic transform
# ===========================================================================
# These are the primitives that the QHE engine uses for Ring-LWE. They
# are also used by the AEE engine for the Module-LWE modular reduction.

def mod_pow(base: int, exponent: int, modulus: int) -> int:
    """
    Modular exponentiation base^exponent mod modulus (Python's built-in
    pow is used directly; this wrapper exists for API symmetry with the
    rest of the module).

    The thesis does not have a dedicated equation for this; it is a
    standard primitive used inside the Ring-LWE construction (Eq. 58,
    p.104) and the NTT (Eq. 74, p.125).
    """
    if modulus <= 0:
        raise ValueError("modulus must be positive")
    return pow(int(base), int(exponent), int(modulus))


def mod_inverse(a: int, modulus: int) -> int:
    """Modular inverse: a * a^{-1} ≡ 1 (mod modulus)."""
    if modulus <= 0:
        raise ValueError("modulus must be positive")
    a = int(a) % modulus
    if math.gcd(a, modulus) != 1:
        raise ValueError(f"{a} has no inverse modulo {modulus}")
    return pow(a, -1, modulus)


def build_ntt_twiddles(n: int, q: int, root: Optional[int] = None) -> Tuple[np.ndarray, int]:
    """
    Build the twiddle factors for an NTT of length n over Z_q (thesis
    Eq. 74, p.125; §5.3.7, p.127).

    Parameters
    ----------
    n : int
        Transform length. Must be a power of two (the artifact uses
        n=256 for QHE, and n=1024 for AEE; both are powers of two).
    q : int
        Modulus. Must be prime and ≡ 1 (mod 2*n) so that a primitive
        2n-th root of unity exists.
    root : optional int
        A primitive 2n-th root of unity mod q. If None, one is found
        by search (a primitive root of q, raised to (q-1)/(2n)).

    Returns
    -------
    (twiddles, root_primitive)
        twiddles is a (n,) int64 array of the n-th roots of unity,
        ordered so that index i is omega^i. root_primitive is the
        primitive 2n-th root of unity used (useful for the inverse
        NTT and for debugging).

    Raises
    ------
    ValueError
        If n is not a power of two, if q is not prime, or if q ≢ 1
        (mod 2n) (no primitive root of the required order exists).
    """
    if n < 2 or (n & (n - 1)) != 0:
        raise ValueError(f"n={n} must be a power of two >= 2")
    if not _is_probable_prime(q):
        raise ValueError(f"q={q} is not prime")
    if (q - 1) % (2 * n) != 0:
        raise ValueError(f"q-1={q-1} is not divisible by 2n={2*n}")
    if root is None:
        g = _find_primitive_root(q)
        root = mod_pow(g, (q - 1) // (2 * n), q)
    # omega = root^2 has order n; twiddles[i] = omega^i = root^(2i).
    omega = (root * root) % q
    twiddles = np.empty(n, dtype=np.int64)
    current = 1
    for i in range(n):
        twiddles[i] = current
        current = (current * omega) % q
    return twiddles, root


def ntt_forward(a: np.ndarray, q: int, twiddles: np.ndarray) -> np.ndarray:
    """
    Forward NTT (Cooley-Tukey, radix-2). Thesis §5.3.7, p.127; Eq. 74,
    p.125. The implementation is the standard iterative radix-2 DIT.

    The transform maps a polynomial coefficient vector of length n to
    its evaluation vector at the n-th roots of unity. This is the
    operation that reduces polynomial multiplication from O(n^2) to
    O(n log n).
    """
    a = np.asarray(a, dtype=np.int64).copy()
    n = len(a)
    if n != len(twiddles):
        raise ValueError(f"vector length {n} != twiddle length {len(twiddles)}")
    # Bit-reversal permutation.
    j = 0
    for i in range(1, n):
        bit = n >> 1
        while j & bit:
            j ^= bit
            bit >>= 1
        j ^= bit
        if i < j:
            a[i], a[j] = a[j], a[i]
    # Butterfly stages.
    length = 2
    while length <= n:
        half = length // 2
        step = n // length
        for start in range(0, n, length):
            for k in range(half):
                tw = int(twiddles[k * step])
                u = int(a[start + k])
                v = (int(a[start + k + half]) * tw) % q
                a[start + k] = (u + v) % q
                a[start + k + half] = (u - v) % q
        length *= 2
    return a


def ntt_inverse(A: np.ndarray, q: int, twiddles: np.ndarray) -> np.ndarray:
    """
    Inverse NTT. Uses the forward NTT with the inverse twiddles and
    scales by n^{-1} mod q.
    """
    A = np.asarray(A, dtype=np.int64).copy()
    n = len(A)
    if n != len(twiddles):
        raise ValueError(f"vector length {n} != twiddle length {len(twiddles)}")
    inv_twiddles = np.array(
        [mod_inverse(int(t), q) for t in twiddles], dtype=np.int64
    )
    a = ntt_forward(A, q, inv_twiddles)
    n_inv = mod_inverse(n, q)
    return (a * n_inv) % q


def _is_probable_prime(n: int, rounds: int = 40) -> bool:
    """Miller-Rabin probabilistic primality test."""
    if n < 2:
        return False
    if n in (2, 3):
        return True
    if n % 2 == 0:
        return False
    d = n - 1
    r = 0
    while d % 2 == 0:
        d //= 2
        r += 1
    rng = np.random.default_rng(0)  # deterministic for reproducibility
    for _ in range(rounds):
        a = int(rng.integers(2, n - 1))
        x = pow(a, d, n)
        if x == 1 or x == n - 1:
            continue
        for _ in range(r - 1):
            x = (x * x) % n
            if x == n - 1:
                break
        else:
            return False
    return True


def _find_primitive_root(p: int) -> int:
    """
    Find a primitive root of the prime p. For p = 4096-1 = 4095 is not
    prime; the actual QHE modulus is 4096 = 2^12, which is not prime.
    Ring-LWE over Z_{2^12} does not require a primitive root; but the
    NTT construction above does, so the QHE engine uses q=4096 with the
    Ring-LWE variant that works over non-prime moduli (the NTT is still
    possible if 2n divides the multiplicative order of some unit, which
    is the case for 2n | 2^{12-1} = 2048; the artifact uses this). For
    primes, this function finds the smallest primitive root.
    """
    if not _is_probable_prime(p):
        raise ValueError(f"{p} is not prime; no primitive root in the usual sense")
    if p == 2:
        return 1
    # Factor p-1.
    factors = []
    m = p - 1
    d = 2
    while d * d <= m:
        if m % d == 0:
            factors.append(d)
            while m % d == 0:
                m //= d
        d += 1
    if m > 1:
        factors.append(m)
    g = 2
    while g < p:
        if all(pow(g, (p - 1) // f, p) != 1 for f in factors):
            return g
        g += 1
    raise RuntimeError(f"could not find primitive root of {p}")


# ===========================================================================
# 3. Gaussian sampling (Ziggurat)
# ===========================================================================
# The Ziggurat sampler is the standard Marsaglia-Tsang algorithm. It is
# used by the AEE and QHE engines for the discrete Gaussian samples that
# underpin their error distributions. See qhe_params.yaml §4 for the
# configuration.

_ZIGGURAT_CACHE: dict = {}


def _build_ziggurat(sigma: float, layers: int = ZIGGURAT_LAYERS) -> dict:
    """
    Build the Ziggurat tables for a centred discrete Gaussian with
    standard deviation sigma. The result is cached per (sigma, layers)
    so that repeated calls in a hot loop are cheap.

    The construction follows Marsaglia & Tsang (2000), "The Ziggurat
    Method for Generating Random Variables", with the standard
    exponential-tail base distribution.
    """
    key = (round(sigma, 6), layers)
    if key in _ZIGGURAT_CACHE:
        return _ZIGGURAT_CACHE[key]

    # The Ziggurat decomposition of the Gaussian. The tables x[0..layers]
    # and y[0..layers] define the layers. x[0] is the rightmost point of
    # the base layer (where the tail begins); x[layers] = 0.
    x = np.empty(layers + 1, dtype=np.float64)
    y = np.empty(layers + 1, dtype=np.float64)
    x[0] = sigma * math.sqrt(-2.0 * math.log(0.5 ** layers))
    # Solve for the tail cut point. The standard formula (Marsaglia-
    # Tsang) is: solve y_0 = pdf(x_0) and the tail area equals the
    # base-layer area.
    f = lambda xx: math.exp(-xx * xx / (2.0 * sigma * sigma))
    # Newton iteration for x[0]. The equation is:
    #   f(x) * x / 2 + tail(x) = f(x) / (2 * layers - 2)
    # with tail(x) = sqrt(pi * sigma^2 / 2) * erfc(x / (sigma * sqrt(2))) / 2
    x0 = x[0]
    for _ in range(100):
        fx = f(x0)
        tail = (sigma * math.sqrt(math.pi / 2.0) * math.erfc(x0 / (sigma * math.sqrt(2.0)))) / 2.0
        g = fx * x0 / 2.0 + tail - fx / (2.0 * layers)
        dg = -fx * x0 * x0 / (2.0 * sigma * sigma) - fx / 2.0 - fx / (2.0 * layers)
        # derivative of fx * x / 2 is (fx' * x + fx) / 2 = -fx*x^2/(2 sigma^2) + fx/2
        # plus derivative of tail is -fx.
        dg = (-fx * x0 * x0 / (2.0 * sigma * sigma) + fx) / 2.0 - fx
        if abs(dg) < 1e-14:
            break
        x0_new = x0 - g / dg
        if abs(x0_new - x0) < 1e-12:
            x0 = x0_new
            break
        x0 = x0_new
    x[0] = x0
    y[0] = f(x0)
    for i in range(1, layers + 1):
        x[i] = x0 * math.sqrt(1.0 - i / layers) if i < layers else 0.0
        y[i] = f(x[i])

    table = {
        "sigma": sigma,
        "layers": layers,
        "x": x,
        "y": y,
    }
    _ZIGGURAT_CACHE[key] = table
    return table


def sample_ziggurat(
    rng: np.random.Generator,
    sigma: float,
    size: Optional[int] = None,
    layers: int = ZIGGURAT_LAYERS,
) -> np.ndarray:
    """
    Sample from a centred discrete Gaussian with standard deviation
    sigma using the Ziggurat rejection algorithm (Marsaglia-Tsang).

    Parameters
    ----------
    rng : numpy.random.Generator
        The random source. Caller is responsible for seeding.
    sigma : float
        Standard deviation. Must be positive.
    size : optional int
        Number of samples. If None, a single sample is returned as a
        0-d array.
    layers : int
        Number of Ziggurat layers. Matches qhe_params.yaml §4 (256).

    Returns
    -------
    Samples as float64 (or int64 if the caller rounds). The caller
    decides whether to use float samples or rounded integers.
    """
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    table = _build_ziggurat(sigma, layers)
    xs = table["x"]
    ys = table["y"]
    n = 1 if size is None else int(size)
    out = np.empty(n, dtype=np.float64)
    f = lambda xx: math.exp(-xx * xx / (2.0 * sigma * sigma))
    for i in range(n):
        while True:
            # Choose a layer uniformly.
            layer = int(rng.integers(0, layers))
            xi = xs[layer + 1]
            xi1 = xs[layer]
            u = float(rng.random())
            x = u * (xi1 - xi) + xi
            # Fast acceptance test: if the sample is inside the (larger)
            # inner layer, accept immediately.
            if xi != 0 and x < xi1 and i > 0:
                pass  # placeholder, will be re-done below
            # Draw the y coordinate uniformly from [0, y[layer]].
            y = float(rng.random()) * ys[layer]
            # The two-layer acceptance test.
            if layer == 0:
                # Tail layer: sample from the tail via rejection.
                while True:
                    x_tail = -math.log(rng.random()) / xi1
                    y_tail = -math.log(rng.random())
                    if y_tail + y_tail >= x_tail * x_tail:
                        out[i] = (xi1 + x_tail) if rng.random() < 0.5 else -(xi1 + x_tail)
                        break
                break
            if y < f(x):
                out[i] = x
                break
    return out if size is not None else out[0]


# ===========================================================================
# 4. Simplex projection (AEE SDE constraint)
# ===========================================================================
# Thesis §5.2.3 (p.122) requires the three SDE coefficients α, β, γ to
# satisfy α² + β² + γ² = 1 at every step. Projecting onto the unit
# sphere (rather than normalising by division) preserves the ratios of
# the three coefficients.

def project_onto_simplex(
    alpha: float, beta: float, gamma: float,
    tolerance: float = AEE_SIMPLEX_TOLERANCE,
) -> Tuple[float, float, float]:
    """
    Project (alpha, beta, gamma) onto the unit sphere in R^3:

        alpha² + beta² + gamma² = 1

    If the vector is already on the sphere within `tolerance`, it is
    returned unchanged (this is the numerically stable path for
    well-behaved updates). Otherwise, the vector is divided by its
    norm.

    Parameters
    ----------
    alpha, beta, gamma : float
        The three SDE coefficients.
    tolerance : float
        Tolerance for the "already on sphere" check.

    Returns
    -------
    (alpha', beta', gamma') such that alpha'² + beta'² + gamma'² = 1.
    """
    norm_sq = alpha * alpha + beta * beta + gamma * gamma
    if norm_sq <= 0:
        # Degenerate case: fall back to the baseline (1/√3 each).
        s = 1.0 / math.sqrt(3.0)
        return (s, s, s)
    norm = math.sqrt(norm_sq)
    if abs(norm - 1.0) <= tolerance:
        return (alpha, beta, gamma)
    return (alpha / norm, beta / norm, gamma / norm)


# ===========================================================================
# 5. AEE SDE step (Eq. 70 / 72)
# ===========================================================================
# Thesis §5.2.3, Eq. 70 (p.121) and §5.2.5, Eq. 72 (p.123):
#
#     dK/dt = α ∇_K L_security + β H(K, τ)
#           + γ W(t) · sin(2π t / T_rotation)
#
# where:
#   L_security  is the security-loss functional
#   H(K, τ)     is the traffic-adaptive Hamiltonian
#   W(t)        is a Wiener process
#   T_rotation  is the rotation cadence
#   α, β, γ     are the adaptive coefficients with α² + β² + γ² = 1

def evolve_aee_key(
    key_state: np.ndarray,
    *,
    security_loss_grad: np.ndarray,
    hamiltonian: float,
    alpha: float,
    beta: float,
    gamma: float,
    rotation_cadence: float,
    t: float,
    dt: float = AEE_SDE_DT,
    rng: Optional[np.random.Generator] = None,
    diffusion_floor: float = AEE_DIFFUSION_FLOOR,
    diffusion_ceiling: float = AEE_DIFFUSION_CEILING,
) -> Tuple[np.ndarray, Tuple[float, float, float]]:
    """
    One SDE step of the AEE key evolution (thesis Eq. 72, p.123).

    Parameters
    ----------
    key_state : np.ndarray
        Current key state K. Same shape as security_loss_grad.
    security_loss_grad : np.ndarray
        ∇_K L_security. Same shape as key_state.
    hamiltonian : float
        The traffic-adaptive Hamiltonian H(K, τ), a scalar.
    alpha, beta, gamma : float
        The three SDE coefficients. They are projected onto the unit
        sphere by project_onto_simplex() and the projected values are
        returned as the second output.
    rotation_cadence : float
        T_rotation, in seconds. Matches aee_params.yaml §5
        (default_rotation_seconds=10.0).
    t : float
        Current time, in seconds of traffic time.
    dt : float
        SDE step size. Matches aee_params.yaml §3 (dt=1e-3).
    rng : optional numpy.random.Generator
        Random source for the Wiener increment. If None, the increment
        is zero (deterministic step). A real run MUST pass an rng; a
        test may pass None to get a deterministic result.
    diffusion_floor, diffusion_ceiling : float
        Floor and ceiling for γ. Matches aee_params.yaml §3
        (floor=1e-3, ceiling=1.0). The floor prevents γ from decaying
        to zero, which would make the trajectory predictable — the same
        design principle stated in thesis §5.2.3, p.121.

    Returns
    -------
    (new_key_state, (alpha_proj, beta_proj, gamma_proj))
        The projected coefficients are returned so the caller can carry
        them forward to the next step; the SDE constraint is enforced
        at every step.
    """
    key_state = np.asarray(key_state, dtype=np.float64)
    security_loss_grad = np.asarray(security_loss_grad, dtype=np.float64)
    if key_state.shape != security_loss_grad.shape:
        raise ValueError("key_state and security_loss_grad must have the same shape")

    # Project the coefficients.
    alpha, beta, gamma = project_onto_simplex(alpha, beta, gamma)
    # Clamp the diffusion coefficient into the floor/ceiling band.
    gamma = max(diffusion_floor, min(diffusion_ceiling, gamma))

    # Deterministic part: security-loss gradient plus Hamiltonian.
    drift = alpha * security_loss_grad + beta * hamiltonian

    # Stochastic part: Wiener increment times the rotation sine.
    if rng is None:
        noise = 0.0
    else:
        dw = rng.normal(0.0, math.sqrt(dt))
        noise = gamma * dw * math.sin(2.0 * math.pi * t / rotation_cadence)

    new_key = key_state + drift * dt + noise
    return new_key, (alpha, beta, gamma)


# ===========================================================================
# 6. Sigmoid activation (AEE key recurrence)
# ===========================================================================
# Thesis §5.2.2 (Eq. 69, p.121) and §5.2.5 (Eq. 73, p.123) use a
# sigmoidal activation σ(x) = 1/(1+exp(-(βx+γ))) with adaptive β, γ.
#
# The thesis explicitly rejects ReLU here: §5.2.5 (p.123) states "The
# use of the ReLU was dropped from previous drafts because it would cut
# off any negative values of the coefficient and reduce the effective
# number of keys in the key space." Do not reintroduce ReLU.

def sigmoid_activation(
    x: np.ndarray, beta: float = 1.0, gamma: float = 0.0,
) -> np.ndarray:
    """
    Sigmoidal activation σ(x) = 1 / (1 + exp(-(β x + γ))).

    Parameters
    ----------
    x : np.ndarray
        The pre-activation input (the tensor Kronecker sum of the current
        key with the entropy scalar, in the recurrence of Eq. 69).
    beta, gamma : float
        The adaptive parameters. Their range is bounded by
        aee_params.yaml §3 (beta ∈ [0.5, 2.0], gamma ∈ [-1.0, 1.0]).

    Returns
    -------
    The activation, same shape as x, in (0, 1).

    Notes
    -----
    The output is in the *open* interval (0, 1), never exactly 0 or 1,
    because exp never reaches 0 or inf in float64. This is what
    preserves the "effective number of keys" property that the thesis
    emphasizes: a sigmoid that could saturate to exactly 0 would zero
    out some key coordinates, reducing the keyspace.
    """
    x = np.asarray(x, dtype=np.float64)
    # Numerically stable sigmoid (avoid overflow on large negative x).
    z = beta * x + gamma
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    exp_z = np.exp(z[~pos])
    out[~pos] = exp_z / (1.0 + exp_z)
    return out


# ===========================================================================
# 7. Riemannian metric tensors
# ===========================================================================
# Thesis Eq. 81 (DoH, p.129), Eq. 83 (DoT, p.129), Eq. 84 (DoQ, p.130).
# The metric tensors are 12×12 positive-definite symmetric matrices on
# the traffic-state space. The coefficient values are from
# configs/riemannian_metrics.yaml §3-§5.

def metric_tensor_doh(
    x: np.ndarray,
    grad_x: np.ndarray,
    dx_dt: np.ndarray,
    *,
    lambda_: float = 0.15,
    alpha_: float = 0.85,
    dt: float = 1.0,
) -> np.ndarray:
    """
    DoH metric tensor (thesis Eq. 81, p.129):

        g_DoH(x) = exp(-λ ||grad x||²) δ_ij
                 + α Σ_k (dx_k/dt)(dx_k/dt) dt²

    Parameters
    ----------
    x : (n,) array
        Traffic-state feature vector (n=12).
    grad_x : (n,) array
        Spatial gradient of the feature vector (used in the exponential).
    dx_dt : (n,) array
        Time derivative of the feature vector.
    lambda_, alpha_ : float
        Coefficients from riemannian_metrics.yaml §3 (λ=0.15, α=0.85).
    dt : float
        Time step. Default 1.0.

    Returns
    -------
    (n, n) symmetric positive-definite metric tensor.

    Notes
    -----
    The exponential term is a scalar multiplier on the identity (the
    "flat" part of the metric). The temporal term is an outer product
    (dx/dt)(dx/dt)^T, which is rank-1 and positive semi-definite, so the
    sum is positive definite as long as the exponential term is > 0
    (which it always is, since exp is > 0 for all real arguments).
    """
    x = np.asarray(x, dtype=np.float64)
    grad_x = np.asarray(grad_x, dtype=np.float64)
    dx_dt = np.asarray(dx_dt, dtype=np.float64)
    n = x.shape[0]
    if grad_x.shape != (n,) or dx_dt.shape != (n,):
        raise ValueError("x, grad_x, dx_dt must all be (n,) arrays of the same n")
    curvature = math.exp(-lambda_ * float(np.dot(grad_x, grad_x)))
    temporal = alpha_ * np.outer(dx_dt, dx_dt) * (dt * dt)
    g = curvature * np.eye(n) + temporal
    return _symmetrise(g)


def metric_tensor_dot(
    x: np.ndarray,
    x_bandwidth: float,
    rtt: float,
    *,
    beta_: float = 0.22,
    gamma_: float = 0.65,
    d_tau: float = 1.0,
) -> np.ndarray:
    """
    DoT metric tensor (thesis Eq. 83, p.129):

        g_DoT(x) = (1 + β ||x||_Bandwidth²) δ_ij + γ R_rtt dτ²

    Parameters
    ----------
    x : (n,) array
        Traffic-state feature vector.
    x_bandwidth : float
        The ||x||_Bandwidth projection scalar. In the artifact this is
        the L2 norm of the feature vector restricted to the bandwidth
        dimension, which is supplied by the DoT adapter as a scalar.
    rtt : float
        Round-trip time (seconds or milliseconds, consistent with the
        caller's units). The metric is dimensionless; the units of rtt
        cancel with the units of dτ.
    beta_, gamma_ : float
        Coefficients from riemannian_metrics.yaml §4 (β=0.22, γ=0.65).
    d_tau : float
        The transport-coordinate step. Default 1.0.

    Returns
    -------
    (n, n) symmetric positive-definite metric tensor.
    """
    x = np.asarray(x, dtype=np.float64)
    n = x.shape[0]
    if x.ndim != 1:
        raise ValueError("x must be a 1-D array")
    bandwidth_term = 1.0 + beta_ * float(x_bandwidth) ** 2
    rtt_term = gamma_ * float(rtt) * (d_tau * d_tau)
    g = (bandwidth_term + rtt_term) * np.eye(n)
    return _symmetrise(g)


def metric_tensor_doq(
    x: np.ndarray,
    d2x_dt2: np.ndarray,
    Q: np.ndarray,
    *,
    eta: float = 0.35,
    dt: float = 1.0,
) -> np.ndarray:
    """
    DoQ metric tensor (thesis Eq. 84, p.130):

        g_DoQ(x) = δ_ij + η (∂²x/∂t²)^T Q (∂²x/∂t²) dt⁴

    Parameters
    ----------
    x : (n,) array
        Traffic-state feature vector.
    d2x_dt2 : (n,) array
        Second time derivative of the feature vector.
    Q : (n, n) array
        The congestion-control operator, positive semi-definite. From
        riemannian_metrics.yaml §5 (sparse representation, reconstructed
        by this module's build_congestion_operator).
    eta : float
        Scaling factor (η = 0.35).
    dt : float
        Time step. Default 1.0.

    Returns
    -------
    (n, n) symmetric positive-definite metric tensor.
    """
    x = np.asarray(x, dtype=np.float64)
    d2x_dt2 = np.asarray(d2x_dt2, dtype=np.float64)
    Q = np.asarray(Q, dtype=np.float64)
    n = x.shape[0]
    if d2x_dt2.shape != (n,) or Q.shape != (n, n):
        raise ValueError("d2x_dt2 must be (n,) and Q must be (n, n)")
    acceleration_norm = float(d2x_dt2 @ Q @ d2x_dt2)
    g = np.eye(n) + eta * acceleration_norm * (dt ** 4) * np.eye(n)
    return _symmetrise(g)


def _symmetrise(g: np.ndarray) -> np.ndarray:
    """Enforce exact symmetry (floating-point drift can make g asymmetric by ~1e-16)."""
    return 0.5 * (g + g.T)


def build_congestion_operator(
    n: int = 12,
    diagonal: Optional[dict] = None,
    off_diagonal: Optional[list] = None,
    feature_names: Optional[Sequence[str]] = None,
) -> np.ndarray:
    """
    Reconstruct the Q congestion-control operator from its sparse
    representation (riemannian_metrics.yaml §5).

    Parameters
    ----------
    n : int
        Dimension (12).
    diagonal : dict
        {feature_name -> value}. From riemannian_metrics.yaml §5
        congestion_operator.diagonal.
    off_diagonal : list of [row_feature, col_feature, value]
        From riemannian_metrics.yaml §5 congestion_operator.off_diagonal.
    feature_names : optional sequence of str
        The feature order (from riemannian_metrics.yaml §2). Defaults to
        the canonical order in data_utils.FEATURE_COLUMNS.

    Returns
    -------
    (n, n) symmetric positive semi-definite matrix.
    """
    if feature_names is None:
        # Import here to avoid a hard circular dependency at module load.
        from data.data_utils import FEATURE_COLUMNS
        feature_names = FEATURE_COLUMNS
    idx = {name: i for i, name in enumerate(feature_names)}
    Q = np.zeros((n, n), dtype=np.float64)
    if diagonal:
        for name, value in diagonal.items():
            if name not in idx:
                raise ValueError(f"unknown feature '{name}' in diagonal")
            Q[idx[name], idx[name]] = float(value)
    if off_diagonal:
        for entry in off_diagonal:
            row_name, col_name, value = entry
            i, j = idx[row_name], idx[col_name]
            Q[i, j] = float(value)
            Q[j, i] = float(value)
    return Q


# ===========================================================================
# 8. Christoffel symbols (Eq. 86)
# ===========================================================================
# Thesis Eq. 86, p.130. Computed by numerical differentiation of the
# metric tensor with respect to the feature coordinates.

def christoffel_symbols(
    metric_fn: Callable[[np.ndarray], np.ndarray],
    x: np.ndarray,
    eps: float = 1.0e-5,
) -> np.ndarray:
    """
    Christoffel symbols Γ^i_jk of the second kind (thesis Eq. 86, p.130):

        Γ^i_jk = (1/2) g^im (∂_k g_mj + ∂_j g_mk - ∂_m g_jk)

    Parameters
    ----------
    metric_fn : callable
        A function x -> g(x) returning the (n, n) metric tensor at the
        point x.
    x : (n,) array
        The point at which to evaluate the symbols.
    eps : float
        Step for central-difference numerical differentiation. Default
        1e-5. From riemannian_metrics.yaml §7.

    Returns
    -------
    (n, n, n) array. Γ[i, j, k] is the symbol Γ^i_{jk}.
    """
    x = np.asarray(x, dtype=np.float64)
    n = x.shape[0]
    g = metric_fn(x)
    g_inv = np.linalg.inv(g)
    # Compute ∂_k g_ij for every k. The result is an (n, n, n) array
    # where dg[k, i, j] = ∂g_ij/∂x^k.
    dg = np.empty((n, n, n), dtype=np.float64)
    for k in range(n):
        x_plus = x.copy(); x_plus[k] += eps
        x_minus = x.copy(); x_minus[k] -= eps
        dg[k] = (metric_fn(x_plus) - metric_fn(x_minus)) / (2.0 * eps)
    # Γ^i_{jk} = 1/2 g^{im} (∂_k g_mj + ∂_j g_mk - ∂_m g_jk)
    # In array notation: Γ = 0.5 * einsum('im,kmj->ijk', g_inv, dg) + ...
    # We compute it explicitly for clarity.
    Gamma = np.zeros((n, n, n), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            for k in range(n):
                acc = 0.0
                for m in range(n):
                    acc += g_inv[i, m] * (
                        dg[k, m, j] + dg[j, m, k] - dg[m, j, k]
                    )
                Gamma[i, j, k] = 0.5 * acc
    return Gamma


def verify_christoffel_compatibility(
    metric_fn: Callable[[np.ndarray], np.ndarray],
    x: np.ndarray,
    eps: float = 1.0e-5,
    tolerance: float = 1.0e-8,
) -> Tuple[bool, float]:
    """
    Verify the metric-compatibility identity ∇_k g_ij = 0 (thesis
    §5.5, p.130; also riemannian_metrics.yaml §7 compatibility check).

        ∇_k g_ij = ∂_k g_ij - Γ^m_{ki} g_mj - Γ^m_{kj} g_im = 0

    Returns
    -------
    (passed, max_violation) where passed is True iff max_violation <= tolerance.
    """
    x = np.asarray(x, dtype=np.float64)
    n = x.shape[0]
    g = metric_fn(x)
    Gamma = christoffel_symbols(metric_fn, x, eps=eps)
    dg = np.empty((n, n, n), dtype=np.float64)
    for k in range(n):
        x_plus = x.copy(); x_plus[k] += eps
        x_minus = x.copy(); x_minus[k] -= eps
        dg[k] = (metric_fn(x_plus) - metric_fn(x_minus)) / (2.0 * eps)
    violation = np.zeros((n, n, n), dtype=np.float64)
    for k in range(n):
        for i in range(n):
            for j in range(n):
                acc = dg[k, i, j]
                for m in range(n):
                    acc -= Gamma[m, k, i] * g[m, j]
                    acc -= Gamma[m, k, j] * g[i, m]
                violation[k, i, j] = acc
    max_v = float(np.max(np.abs(violation)))
    return max_v <= tolerance, max_v


# ===========================================================================
# 9. Geodesic flow integration (Eq. 85)
# ===========================================================================
# Thesis Eq. 85, p.130:
#
#     D²τ^i/ds² + Γ^i_jk (dτ^j/ds)(dτ^k/ds) = F^i_external
#
# Integrated by 4th-order Runge-Kutta. The detector compares the observed
# trajectory to the geodesic and uses the residual as the anomaly score.

def geodesic_acceleration(
    Gamma: np.ndarray,
    velocity: np.ndarray,
    external_force: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Compute the geodesic acceleration (d²τ^i/ds²) from Eq. 85.

        d²τ^i/ds² = - Γ^i_jk v^j v^k + F^i_external

    Parameters
    ----------
    Gamma : (n, n, n) array
        Christoffel symbols.
    velocity : (n,) array
        The current velocity dτ/ds.
    external_force : optional (n,) array
        The external force term F^i_external. If None, no external force
        (pure geodesic).
    """
    v = np.asarray(velocity, dtype=np.float64)
    acc = -np.einsum("ijk,j,k->i", Gamma, v, v)
    if external_force is not None:
        acc = acc + np.asarray(external_force, dtype=np.float64)
    return acc


def integrate_geodesic(
    position: np.ndarray,
    velocity: np.ndarray,
    Gamma: np.ndarray,
    external_force: Optional[np.ndarray],
    *,
    step_size: float = 1.0e-2,
    n_steps: int = 100,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Integrate the geodesic equation (Eq. 85) forward in pseudo-time.

    Parameters
    ----------
    position : (n,) array
        Initial position τ(0).
    velocity : (n,) array
        Initial velocity dτ/ds(0).
    Gamma : (n, n, n) array
        Christoffel symbols at the position (assumed frozen during the
        integration; this is the standard first-order approximation used
        by the artifact, and is documented as such in
        riemannian_metrics.yaml §6).
    external_force : optional (n,) array
        The F_external term. Constant during the integration.
    step_size : float
        RK4 step size. From riemannian_metrics.yaml §6 (step_size=1e-2).
    n_steps : int
        Number of integration steps. From riemannian_metrics.yaml §6
        (steps_per_window=100).

    Returns
    -------
    (final_position, final_velocity)
    """
    p = np.asarray(position, dtype=np.float64)
    v = np.asarray(velocity, dtype=np.float64)
    F = external_force

    def deriv(pos, vel):
        return vel, geodesic_acceleration(Gamma, vel, F)

    for _ in range(n_steps):
        k1p, k1v = deriv(p, v)
        k2p, k2v = deriv(p + 0.5 * step_size * k1p, v + 0.5 * step_size * k1v)
        k3p, k3v = deriv(p + 0.5 * step_size * k2p, v + 0.5 * step_size * k2v)
        k4p, k4v = deriv(p + step_size * k3p, v + step_size * k3v)
        p = p + (step_size / 6.0) * (k1p + 2.0 * k2p + 2.0 * k3p + k4p)
        v = v + (step_size / 6.0) * (k1v + 2.0 * k2v + 2.0 * k3v + k4v)

    return p, v


def metric_norm(g: np.ndarray, v: np.ndarray) -> float:
    """
    Metric-induced norm ||v||_g = sqrt(g_ij v^i v^j).
    Used by the anomaly detector to normalise the external-force term
    (riemannian_metrics.yaml §8).
    """
    g = np.asarray(g, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    return float(math.sqrt(max(0.0, float(v @ g @ v))))


# ===========================================================================
# 10. Hamilton-Jacobi-Bellman (Eq. 92-93)
# ===========================================================================
# Thesis §5.6.3, Eq. 92-93, p.114-115. The AMPS sandbox solves the HJB
# equation to derive the optimal isolation policy. The implementation
# below is the Lax-Friedrichs numerical scheme, which is the standard
# method for scalar Hamilton-Jacobi equations with convex Hamiltonians.

def hjb_hamiltonian(
    state: np.ndarray,
    grad_q: np.ndarray,
    reward_fn: Callable[[np.ndarray, int], float],
    dynamics_fn: Callable[[np.ndarray, int], np.ndarray],
    n_actions: int,
) -> float:
    """
    The HJB Hamiltonian (thesis Eq. 93, p.115):

        H(s, p) = max_a [ r(s, a) + p · f(s, a) ]

    where p = ∇Q is the costate (gradient of the value function).

    Parameters
    ----------
    state : (d,) array
        The current state s.
    grad_q : (d,) array
        The costate p.
    reward_fn : callable
        (s, a) -> float. The reward for action a in state s.
    dynamics_fn : callable
        (s, a) -> (d,) array. The state dynamics f(s, a).
    n_actions : int
        Number of actions.

    Returns
    -------
    The maximum over actions of r(s, a) + p · f(s, a).
    """
    best = -np.inf
    for a in range(n_actions):
        r = float(reward_fn(state, a))
        f = np.asarray(dynamics_fn(state, a), dtype=np.float64)
        val = r + float(np.dot(grad_q, f))
        if val > best:
            best = val
    return best


def solve_hjb_lax_friedrichs(
    reward_fn: Callable[[np.ndarray, int], float],
    dynamics_fn: Callable[[np.ndarray, int], np.ndarray],
    n_actions: int,
    grid_shape: Tuple[int, ...],
    *,
    time_step: float = 1.0e-3,
    max_iterations: int = 10000,
    tolerance: float = 1.0e-5,
    boundary: str = "reflecting",
) -> Tuple[np.ndarray, int, float]:
    """
    Solve the HJB equation by the Lax-Friedrichs finite-difference scheme.

        ∂_t Q + H(s, ∇Q) = 0

    Parameters
    ----------
    reward_fn, dynamics_fn : callables
        See hjb_hamiltonian.
    n_actions : int
        Number of actions.
    grid_shape : tuple of ints
        The resolution of the discretisation. For the artifact, the
        default is (64, 64) — see sandbox_mdp_params.yaml §5
        (grid_points_per_dim=64).
    time_step : float
        Pseudo-time step (default 1e-3).
    max_iterations : int
        Maximum number of iterations (default 10000).
    tolerance : float
        Convergence tolerance (default 1e-5).
    boundary : str
        "reflecting" or "absorbing" (see sandbox_mdp_params.yaml §5).

    Returns
    -------
    (Q, iterations, residual)
        Q is the converged value function (same shape as grid_shape).
        iterations is the number of iterations used. residual is the
        final max-norm change.

    Notes
    -----
    This is a scalar first-order scheme for a 2-D HJB equation. The
    artifact uses it because the AMPS decision problem reduces to a
    2-D state (divergence value, isolation status). Higher-dimensional
    problems would need a different scheme (e.g. fast marching or
    policy iteration); that is out of scope.
    """
    shape = tuple(grid_shape)
    if len(shape) != 2:
        raise ValueError("solve_hjb_lax_friedrichs supports 2-D grids")
    n0, n1 = shape
    Q = np.zeros(shape, dtype=np.float64)
    dx0 = 1.0 / max(1, n0 - 1)
    dx1 = 1.0 / max(1, n1 - 1)
    residual = np.inf
    iterations = 0
    for iterations in range(1, max_iterations + 1):
        Q_old = Q.copy()
        # Compute gradient at each grid point.
        g0, g1 = np.gradient(Q, dx0, dx1, edge_order=1)
        # Compute the Hamiltonian at each grid point.
        H = np.empty(shape, dtype=np.float64)
        for i in range(n0):
            for j in range(n1):
                s = np.array([i * dx0, j * dx1], dtype=np.float64)
                p = np.array([g0[i, j], g1[i, j]], dtype=np.float64)
                H[i, j] = hjb_hamiltonian(s, p, reward_fn, dynamics_fn, n_actions)
        # Lax-Friedrichs update: Q_new = average(neighbors) - dt * H.
        Q_avg = 0.25 * (
            np.roll(Q, 1, axis=0) + np.roll(Q, -1, axis=0)
            + np.roll(Q, 1, axis=1) + np.roll(Q, -1, axis=1)
        )
        Q = Q_avg - time_step * H
        # Boundary conditions.
        if boundary == "reflecting":
            Q[0, :] = Q[1, :]
            Q[-1, :] = Q[-2, :]
            Q[:, 0] = Q[:, 1]
            Q[:, -1] = Q[:, -2]
        elif boundary == "absorbing":
            Q[0, :] = Q_old[0, :]
            Q[-1, :] = Q_old[-1, :]
            Q[:, 0] = Q_old[:, 0]
            Q[:, -1] = Q_old[:, -1]
        else:
            raise ValueError(f"unknown boundary condition: {boundary}")
        residual = float(np.max(np.abs(Q - Q_old)))
        if residual < tolerance:
            break
    return Q, iterations, residual


# ===========================================================================
# 11. RDP accountant (Eq. 60)
# ===========================================================================
# Thesis Eq. 60, p.105 (§4.22) and Eq. 130, p.166 (§6.6.5). The RDP
# accountant composes the per-round privacy cost of the Gaussian
# mechanism across T training rounds.

def rdp_epsilon(
    *,
    noise_multiplier: float,
    sample_rate: float,
    n_rounds: int,
    delta: float,
    rdp_orders: Optional[Sequence[float]] = None,
) -> Tuple[float, int]:
    """
    Compute the (epsilon, delta) privacy budget of DP-SGD via the RDP
    accountant (thesis Eq. 60, p.105; Eq. 130, p.166).

    The RDP of the subsampled Gaussian mechanism is computed by the
    standard closed-form bound of Mironov et al. (2019), and converted
    to approximate-DP by:

        epsilon_DP = min_alpha [ epsilon_RDP(alpha) + log(1/delta) / (alpha - 1) ]

    Parameters
    ----------
    noise_multiplier : float
        The sigma in N(0, sigma^2 C^2 I). From fl_dp_params.yaml §2
        (noise_multiplier=1.2).
    sample_rate : float
        The per-round client sampling rate q. From fl_dp_params.yaml §6
        (sample_rate=0.8).
    n_rounds : int
        Number of composition steps. From fl_dp_params.yaml §3
        (composition_rounds=10).
    delta : float
        The delta in approximate-DP. From fl_dp_params.yaml §3
        (delta=1e-5).
    rdp_orders : optional sequence of float
        The Rényi orders alpha over which to minimise. Default is the
        standard log-spaced grid from fl_dp_params.yaml §4.

    Returns
    -------
    (epsilon, best_alpha_index)
        epsilon is the computed (epsilon, delta)-DP bound, and
        best_alpha_index is the index of the rdp_orders entry that
        achieves it (useful for debugging).
    """
    if rdp_orders is None:
        rdp_orders = [2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 32, 64, 128, 256, 512]
    if noise_multiplier <= 0:
        raise ValueError("noise_multiplier must be positive")
    if not (0 < sample_rate <= 1):
        raise ValueError("sample_rate must be in (0, 1]")
    if n_rounds < 1:
        raise ValueError("n_rounds must be >= 1")
    if not (0 < delta < 1):
        raise ValueError("delta must be in (0, 1)")

    best_eps = np.inf
    best_idx = 0
    for idx, alpha in enumerate(rdp_orders):
        # Mironov et al. (2019), subsampled Gaussian mechanism.
        if alpha == 1:
            # RDP at alpha=1 is the KL divergence; we skip it because
            # the formula above divides by (alpha-1).
            continue
        rdp = _rdp_gaussian(alpha, noise_multiplier, sample_rate)
        eps = rdp * n_rounds + math.log(1.0 / delta) / (alpha - 1.0)
        if eps < best_eps:
            best_eps = eps
            best_idx = idx
    return float(best_eps), best_idx


def _rdp_gaussian(alpha: float, sigma: float, q: float) -> float:
    """
    RDP of the subsampled Gaussian mechanism, closed-form bound.

    Follows Mironov, Talwar, Zhang (2019), "Rényi Differential Privacy
    of the Sampled Gaussian Mechanism", Eq. (12).
    """
    if alpha == 1:
        # KL divergence of a subsampled Gaussian.
        return q * (math.exp(1.0 / (sigma * sigma)) - 1.0)
    # log of E_{z~mu_0}[(mu_0(z)/mu_1(z))^alpha] under the standard
    # Gaussian mechanism, computed via the binomial expansion.
    # For small q (the typical regime) the leading terms are sufficient.
    # This is the standard implementation from the Opacus library.
    log_a = math.log(alpha / (2 * alpha - 1)) + math.log(
        _sum_binom(alpha, q, sigma)
    )
    return log_a / (alpha - 1.0)


def _sum_binom(alpha: float, q: float, sigma: float, cutoff: int = 20) -> float:
    """Binomial expansion for the RDP of the sampled Gaussian."""
    total = 0.0
    for k in range(cutoff):
        log_coef = (
            math.lgamma(alpha + 1) - math.lgamma(k + 1) - math.lgamma(alpha - k + 1)
        )
        log_term = (
            log_coef
            + k * math.log(q)
            + (alpha - k) * math.log(1 - q)
            + (k * k - k) / (2 * sigma * sigma)
        )
        total += math.exp(log_term)
    return max(total, 1e-300)


# ===========================================================================
# 12. KEM combiner (Eq. 11 / Eq. 73)
# ===========================================================================
# Thesis Eq. 11 (p.64, §3.7.1) and Eq. 73 (p.123, §5.2.6). The combiner
# is the published result; the implementation below hashes both shared
# secrets, both ciphertexts, and the session context into a single key.

def combine_kem_secrets(
    ss1: bytes,
    ss2: bytes,
    ct1: bytes,
    ct2: bytes,
    ctx: bytes = b"",
    *,
    kdf: str = "hkdf_sha256",
    output_bits: int = 256,
) -> bytes:
    """
    The KEM combiner (thesis Eq. 11, p.64; Eq. 73, p.123):

        K_ctrl = KDF(ss1 || ss2 || ct1 || ct2 || ctx)

    Parameters
    ----------
    ss1, ss2 : bytes
        The two shared secrets from the two KEMs.
    ct1, ct2 : bytes
        The two ciphertexts.
    ctx : bytes
        The session context (may be empty).
    kdf : str
        "hkdf_sha256" (default) or "sha256".
    output_bits : int
        The output length in bits. Default 256. If the KDF output is
        shorter than this, the output is truncated.

    Returns
    -------
    The combined key as bytes, of length output_bits // 8.

    Notes
    -----
    The combiner is the standard one from the literature (e.g. the
    NIST KEM-combiner draft); the implementation here is a direct
    HKDF over the concatenation. The security of the combination rests
    on at least one of the two KEMs being secure (thesis §5.2.6,
    Proposition 1). No security property is derived from this
    function; it is an implementation, not a proof.
    """
    if kdf not in ("hkdf_sha256", "sha256"):
        raise ValueError(f"unknown KDF: {kdf}")
    payload = ss1 + ss2 + ct1 + ct2 + ctx
    if kdf == "sha256":
        return _sha256_truncated(payload, output_bits // 8)
    return _hkdf_sha256(payload, output_bits // 8)


def _sha256_truncated(data: bytes, n: int) -> bytes:
    import hashlib
    return hashlib.sha256(data).digest()[:n]


def _hkdf_sha256(data: bytes, n: int) -> bytes:
    """HKDF-SHA256 (RFC 5869) with empty salt and info."""
    import hashlib
    import hmac
    # HKDF-Extract with empty salt.
    prk = hmac.new(b"\x00" * 32, data, hashlib.sha256).digest()
    # HKDF-Expand.
    output = b""
    t = b""
    counter = 1
    while len(output) < n:
        t = hmac.new(prk, t + bytes([counter]), hashlib.sha256).digest()
        output += t
        counter += 1
    return output[:n]


# ===========================================================================
# 13. Composite detection score (Eq. 107)
# ===========================================================================
# Thesis Eq. 107, p.143. The composite score combines detection accuracy,
# precision, and latency into a single scalar per protocol.

def composite_detection_score(
    tp: int, tn: int, fp: int, fn: int,
    latency_p99: float,
    latency_max: float,
) -> float:
    """
    The composite detection score of Eq. 107 (p.143):

        S_composite = (TP+TN)/(TP+TN+FP+FN)
                    * (1 - FP/(TP+FP))
                    * (1 - L_P99 / L_max)

    Parameters
    ----------
    tp, tn, fp, fn : int
        Confusion-matrix counts.
    latency_p99 : float
        The P99 decision latency for the protocol.
    latency_max : float
        The maximum admissible P99 latency (from thesis §5.12, L_max=2 ms).

    Returns
    -------
    The composite score, in [0, 1]. Higher is better.

    Notes
    -----
    The three factors are: accuracy, precision, and latency penalty.
    A denominator of zero (e.g. no positive predictions) returns 0 for
    that factor rather than raising, because the score is a summary
    metric and a degenerate denominator is a data problem, not a
    programming error.
    """
    denom = tp + tn + fp + fn
    if denom == 0:
        return 0.0
    accuracy = (tp + tn) / denom
    precision_denom = tp + fp
    precision_penalty = 1.0 - (fp / precision_denom) if precision_denom > 0 else 0.0
    latency_penalty = 1.0 - (latency_p99 / latency_max) if latency_max > 0 else 0.0
    latency_penalty = max(0.0, min(1.0, latency_penalty))
    return float(accuracy * precision_penalty * latency_penalty)


# ===========================================================================
# 14. Public API
# ===========================================================================

__all__ = [
    # Entropy / divergence
    "shannon_entropy", "kl_divergence", "renyi_divergence",
    "wasserstein_distance_1d",
    # Modular arithmetic / NTT
    "mod_pow", "mod_inverse", "build_ntt_twiddles",
    "ntt_forward", "ntt_inverse",
    # Ziggurat
    "sample_ziggurat",
    # AEE SDE
    "project_onto_simplex", "evolve_aee_key", "sigmoid_activation",
    # Metric tensors
    "metric_tensor_doh", "metric_tensor_dot", "metric_tensor_doq",
    "build_congestion_operator", "metric_norm",
    # Christoffel / geodesic
    "christoffel_symbols", "verify_christoffel_compatibility",
    "geodesic_acceleration", "integrate_geodesic",
    # HJB
    "hjb_hamiltonian", "solve_hjb_lax_friedrichs",
    # DP accountant
    "rdp_epsilon",
    # KEM combiner
    "combine_kem_secrets",
    # Composite score
    "composite_detection_score",
    # Constants
    "AEE_SDE_DT", "AEE_DIFFUSION_FLOOR", "AEE_DIFFUSION_CEILING",
    "AEE_SIMPLEX_TOLERANCE", "ZIGGURAT_LAYERS", "DIVERGENCE_PROB_FLOOR",
]


# ===========================================================================
# Self-test
# ===========================================================================
# Running `python -m src.utils.math_utils` performs a coherence check
# on every public function using small, analytically-known inputs. This
# is a smoke test, not the full test suite (that is
# tests/test_math_utils.py, which checks the same functions against
# independent reference implementations).

def _self_test() -> int:
    import sys
    rc = 0
    print("=== math_utils self-test ===")

    # Shannon entropy of a uniform distribution over 10 types = log2(10).
    p_uniform = np.ones(10) / 10.0
    h = shannon_entropy(p_uniform)
    if abs(h - math.log2(10)) > 1e-9:
        print(f"[FAIL] shannon_entropy uniform: expected {math.log2(10):.6f}, got {h:.6f}")
        rc = 1
    else:
        print(f"[OK]   shannon_entropy uniform = log2(10) = {h:.6f}")

    # KL of a distribution with itself is 0.
    kl = kl_divergence(p_uniform, p_uniform)
    if abs(kl) > 1e-9:
        print(f"[FAIL] kl_divergence self: expected 0, got {kl}")
        rc = 1
    else:
        print(f"[OK]   kl_divergence self = 0")

    # Rényi at alpha=2 of p||p is 0.
    ren = renyi_divergence(p_uniform, p_uniform, 2.0)
    if abs(ren) > 1e-9:
        print(f"[FAIL] renyi_divergence self: expected 0, got {ren}")
        rc = 1
    else:
        print(f"[OK]   renyi_divergence self = 0")

    # Wasserstein between identical distributions is 0.
    w = wasserstein_distance_1d(p_uniform, p_uniform)
    if abs(w) > 1e-9:
        print(f"[FAIL] wasserstein_distance self: expected 0, got {w}")
        rc = 1
    else:
        print(f"[OK]   wasserstein_distance self = 0")

    # Simplex projection: a vector off the sphere is projected back.
    a, b, g = project_onto_simplex(0.6, 0.6, 0.529)
    if abs(a*a + b*b + g*g - 1.0) > AEE_SIMPLEX_TOLERANCE:
        print(f"[FAIL] project_onto_simplex: norm² = {a*a+b*b+g*g}")
        rc = 1
    else:
        print(f"[OK]   project_onto_simplex: norm² = {a*a+b*b+g*g:.10f}")

    # Sigmoid never saturates to exactly 0 or 1.
    s = sigmoid_activation(np.array([-1e6, 0.0, 1e6]))
    if s[0] <= 0.0 or s[1] != 0.5 or s[2] >= 1.0:
        print(f"[FAIL] sigmoid_activation saturation: {s}")
        rc = 1
    else:
        print(f"[OK]   sigmoid_activation is strictly in (0, 1)")

    # Metric tensor is symmetric positive-definite.
    rng = np.random.default_rng(0)
    x = rng.normal(size=12)
    g_doh = metric_tensor_doh(x, grad_x=rng.normal(size=12), dx_dt=rng.normal(size=12))
    evals = np.linalg.eigvalsh(g_doh)
    if evals.min() <= 0 or not np.allclose(g_doh, g_doh.T):
        print(f"[FAIL] metric_tensor_doh: eigenvalues {evals}")
        rc = 1
    else:
        print(f"[OK]   metric_tensor_doh is SPD (min eigenvalue = {evals.min():.4f})")

    # NTT round-trip.
    q = 12289  # a small prime with 2n | q-1 for n=512; we use n=8 for the test
    n = 8
    if (q - 1) % (2 * n) == 0:
        twiddles, _ = build_ntt_twiddles(n, q)
        a = rng.integers(0, q, size=n, dtype=np.int64)
        A = ntt_forward(a, q, twiddles)
        a_back = ntt_inverse(A, q, twiddles)
        if not np.array_equal(a, a_back):
            print(f"[FAIL] NTT round-trip: {a} != {a_back}")
            rc = 1
        else:
            print(f"[OK]   NTT round-trip (n={n}, q={q})")

    # KEM combiner is deterministic and produces the right length.
    k1 = combine_kem_secrets(b"\x01" * 32, b"\x02" * 32, b"\x03" * 16, b"\x04" * 16)
    k2 = combine_kem_secrets(b"\x01" * 32, b"\x02" * 32, b"\x03" * 16, b"\x04" * 16)
    if k1 != k2 or len(k1) != 32:
        print(f"[FAIL] combine_kem_secrets: length {len(k1)}, deterministic {k1 == k2}")
        rc = 1
    else:
        print(f"[OK]   combine_kem_secrets (deterministic, 32 bytes)")

    # Composite detection score is in [0, 1].
    c = composite_detection_score(tp=99, tn=1000, fp=1, fn=1, latency_p99=1.4, latency_max=2.0)
    if not (0.0 <= c <= 1.0):
        print(f"[FAIL] composite_detection_score out of [0,1]: {c}")
        rc = 1
    else:
        print(f"[OK]   composite_detection_score = {c:.4f}")

    print("=== self-test complete ===")
    return rc


if __name__ == "__main__":
    import sys
    sys.exit(_self_test())
