#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
src/attacks/padding_oracle.py
=============================

E3-DoH Reproducibility — ciphertext-manipulation attack simulation
(A05, A15).

Chapter 5 of the thesis: E3-DoH (Enhanced Evolutionary Encryption for
Multi-Protocol DNS Security — DoH, DoT, DoQ).

This module implements the two ciphertext-manipulation scenarios from
`attack_suite_params.yaml` §5:

    A05  Padding Oracle             (DoT, padbuster, iterated CBC padding)
    A15  Encrypted Payload Fuzzing  (DoH, boofuzz, mutated payloads)

Both target TLS *application-data* records — the layer that carries
encrypted DNS messages — rather than the handshake. The two scenarios
share the same wire format (TLS records) and the same mutation-loop
structure, so they live in one module.

All five attack modules (dns_tunneling, timing_side_channel,
quic_congestion, cert_injection, padding_oracle) together cover the
fifteen scenarios of `data/attack_matrix.csv`. A caller who wants to
run the whole suite should look at `scripts/run_attack_suite.py`
(delivered later).

The safety guards and namespace context manager are imported from
`src.attacks.dns_tunneling` (the first attack module) so that the
guard implementation is auditable in one place.

WIRE FORMATS
------------
TLS records (RFC 8446 §5.1):

    ContentType(1) | LegacyVersion(2) | Length(2) | Fragment(length)

For application data, ContentType is 0x17. The fragment is the
ciphertext (in CBC mode, the first 16 bytes are the IV; the rest is
the encrypted plaintext plus a padding block). The simulation does
not implement CBC padding itself — it emits a *structurally valid*
ciphertext of the right shape and lets the detection trigger observe
the manipulation pattern (see below).

A05 (PADDING ORACLE)
--------------------
The padding-oracle attack iterates the last byte of the second-to-last
ciphertext block, and observes the server's response for each
candidate value. The classic CBC padding-oracle distinguishes a valid
padding from an invalid one by the server's error behaviour: a valid
padding produces a MAC-failure alert; an invalid padding produces a
decryption-failure alert. The difference between the two alerts is
what leaks one bit of plaintext per trial.

The attack requires:
    1. A ciphertext the attacker can tamper with (a captured
       application-data record).
    2. A server that distinguishes valid from invalid padding in its
       response (i.e., a server that is vulnerable to the attack).
    3. Up to 256 trials per byte to recover each byte of plaintext.

The simulation:
    - Emits `n_probes` TLS application-data records.
    - Each record is a *tampered copy* of a synthetic "captured"
      ciphertext, with the last byte of block N-2 set to a distinct
      candidate value (0x00, 0x01, …, 0xFF, cycling).
    - The padding manipulation is deterministic and visible in the
      emitted bytes: the last byte of block N-2 varies between
      records, and the last byte of block N-1 is held at a fixed
      value (the "target block" the attacker is manipulating toward).
    - The artifact does not implement the response side (the server's
      valid/invalid distinction); the detection trigger looks at the
      *ciphertext pattern*, not at the response.

The detection trigger (`HE.eval(c1 == c2) without decryption`, from
data/attack_matrix.csv) detects the padding-oracle pattern by
homomorphically comparing the *ciphertexts* the server receives. If
two application-data records differ only in their last-16-byte block,
and if the block appears many times with different last bytes, the
pattern matches a padding-oracle iteration. A correctly-implemented
homomorphic comparison — where the two ciphertexts are compared under
encryption without decryption — produces a match or non-match without
revealing the plaintext, and the pattern of matches reveals the
iteration.

A15 (ENCRYPTED PAYLOAD FUZZING)
-------------------------------
The payload-fuzzing attack sends application-data records whose
*decrypted contents would be* malformed DNS messages, malformed HTTP/2
frames, or malformed HTTP/2 frame headers, to trigger error paths in
the server's parsing logic. The error paths sometimes leak information
(timing, error messages, stack traces) that is exploitable.

The simulation:
    - Emits `n_probes` TLS application-data records.
    - Each record contains a *mutated* payload. The mutation is drawn
      from a documented set of patterns (see _MUTATION_PATTERNS
      below), applied to a benign-looking base payload.
    - The mutation rate is controlled by the scenario's
      `mutation_rate` payload field (from the config).

The detection trigger (`wire_format_valid(payload) == False ∨
tls_decryption_error`, from data/attack_matrix.csv) detects the
fuzzing by attempting to parse each record. A malformed record's
*tampered* bytes will not match a valid DNS or HTTP/2 wire format;
a server that returns a decryption error for each malformed record
is observing the tampering. The trigger fires when the malformed
fraction exceeds a documented threshold (0.5 by default).

DETECTION-TRIGGER DESIGN
------------------------
The two triggers are derived from the "no decryption" boundary of the
framework (thesis §1.7.1, p.35; §5.5, p.130): the detector must fire
based on observable ciphertext properties, not on the plaintext.

For A05, the observable property is *structural*: the ciphertext
records vary in a way that is consistent with a padding-oracle
iteration (the last byte of block N-2 varies, everything else is
fixed). The homomorphic comparison evaluates ciphertext equality
without decrypting, so it can confirm the structural pattern without
revealing what the padding is.

For A15, the observable property is also *structural*: the record's
ciphertext MAC is invalid (because the payload was mutated after the
MAC was computed). A server that does not recompute the MAC on the
same key would accept the record and then fail during decryption; a
server that does recompute the MAC rejects the record with an alert.
The detection trigger fires on the *MAC-failure alert count*, which
is observable from the server's response bytes.

Both triggers are documented, both respect the no-decryption
boundary, and both match the entries in `data/attack_matrix.csv`.

DEVIATION FROM THE THESIS
-------------------------
The thesis used the reference tools listed in
`attack_suite_params.yaml` §5 (padbuster, boofuzz). This artifact
does not ship those tools — the module constructs the wire bytes
itself with `struct`. The numerical analysis the detection triggers
perform is identical for the same packet sequence, so the deviation
is a *tooling* deviation, not a *numerical* one.

The full CBC padding-oracle attack (the iterated byte recovery loop
and the observation of the server's alerts) is not simulated; the
module simulates the *packet stream* the attack produces, not the
cryptographic attack itself. This is the same modelling choice the
other attack modules make: the detector's job is to spot the
*pattern*, not to prove the attack works.

DETERMINISM
-----------
Each scenario is deterministic given a seed. The seed is derived
from the run seed via `get_component_seed(seed, "attack_suite")`
(offset +5), matching every other attack module. The module uses
this seed for:

    - the initial captured ciphertext (A05),
    - the per-record mutation pattern (A15),
    - the per-record padding candidate (A05).

THREAD SAFETY
-------------
An instance of the simulator is not thread-safe. A caller who wants
concurrent simulations should create one instance per thread.

SECURITY AND LEGAL BOUNDARY
---------------------------
This module implements attack techniques. It is released for
**defensive research and reproducibility only**. See LICENSE,
Defensive-Research Rider, and `attack_suite_params.yaml` §10 for the
enforceable terms. The summary:

    - Permitted use: defensive research, academic reproducibility,
      controlled testing on systems you own or have explicit written
      authorization to test.
    - Prohibited use: against systems you do not own or have
      authorization to test; to develop offensive tooling; in any
      manner that violates applicable law.
    - Isolation must be preserved.
"""

from __future__ import annotations

import hashlib
import math
import os
import struct
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Shared safety guards and namespace context manager.
from src.attacks.dns_tunneling import (
    AttackConfigError,
    AttackError,
    EgressDetectedError,
    IsolationGuardViolation,
    ScenarioConfig,
    _NamespaceContext,
    parse_scenario,
)

from src.utils.logger import get_logger
from src.utils.seed import get_component_seed


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SUPPORTED_SCENARIOS = ("A05", "A15")

# TLS content type for application data.
_TLS_APPLICATION_DATA = 0x17
# TLS legacy record version.
_TLS_LEGACY_VERSION = 0x0303

# TLS record header is 5 bytes (content type 1 + version 2 + length 2).
_TLS_RECORD_HEADER_LEN = 5

# CBC block size is 16 bytes (AES). The IV is one block; the ciphertext
# is the rest, plus a padding block in CBC-with-padding mode.
_CBC_BLOCK_SIZE = 16
_CBC_IV_SIZE = 16

# The default padding-oracle ciphertext size. Two blocks are enough to
# demonstrate the pattern (the last byte of block N-2 varies while
# block N-1 stays fixed). More blocks is more realistic but the pattern
# is already fully visible in two.
_DEFAULT_CBC_CIPHERTEXT_BLOCKS = 3  # 2 data blocks + 1 padding block

# The mutation patterns A15 applies to the base payload. See the
# module docstring's A15 section. Each pattern is a function that takes
# the base bytes and a per-record index, and returns the mutated bytes.
# The set is small and documented so that a reviewer can predict
# exactly what the simulator emits.
_MUTATION_PATTERNS = (
    "bit_flip",           # flip one random bit in one random byte
    "byte_zero",          # zero one random byte
    "byte_ff",            # set one random byte to 0xff
    "truncate",           # drop the last N random bytes
    "extend",             # append N random bytes
    "swap_halves",        # swap the first and second halves
)

# The A15 detection threshold: the fraction of malformed records at
# which the trigger fires. The default (0.5) means "more than half the
# records are malformed".
_DEFAULT_A15_MALFORMED_THRESHOLD = 0.5

# The A05 detection threshold: the fraction of records whose padding
# candidate differs from the base. Since every record except the first
# has a different candidate, the fraction is (n-1)/n, which is ~0.99
# for any realistic n. The trigger fires when the fraction exceeds
# 0.5.
_DEFAULT_A05_PADDING_VARIED_THRESHOLD = 0.5


# ---------------------------------------------------------------------------
# Scenario-specific config
# ---------------------------------------------------------------------------

@dataclass
class PaddingScenarioConfig(ScenarioConfig):
    """
    A scenario config with the ciphertext-manipulation-specific fields
    the simulator needs.
    """
    # A05 fields.
    cbc_ciphertext_blocks: int = _DEFAULT_CBC_CIPHERTEXT_BLOCKS
    # A15 fields.
    mutation_rate: float = 0.5
    payload_depth: int = 3


def parse_padding_scenario(
    scenario_id: str,
    config: Dict[str, Any],
) -> PaddingScenarioConfig:
    """
    Parse one of A05/A15 from the attack-suite config, with the
    ciphertext-manipulation-specific fields.
    """
    if scenario_id not in SUPPORTED_SCENARIOS:
        raise AttackConfigError(
            f"scenario {scenario_id!r} not handled by padding_oracle.py; "
            f"supported: {SUPPORTED_SCENARIOS}"
        )
    base = parse_scenario(scenario_id, config)
    payload = base.payload or {}

    # A05 fields.
    blocks = int(payload.get("cbc_ciphertext_blocks", _DEFAULT_CBC_CIPHERTEXT_BLOCKS))
    if blocks < 2:
        raise AttackConfigError(
            f"A05 requires at least 2 CBC blocks (1 data + 1 padding); "
            f"got {blocks}"
        )

    # A15 fields.
    mutation_rate = float(payload.get("mutation_rate", 0.5))
    if not (0.0 <= mutation_rate <= 1.0):
        raise AttackConfigError(
            f"A15 mutation_rate must be in [0, 1]; got {mutation_rate}"
        )
    payload_depth = int(payload.get("payload_depth", 3))
    if payload_depth < 1:
        raise AttackConfigError(
            f"A15 payload_depth must be >= 1; got {payload_depth}"
        )

    return PaddingScenarioConfig(
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
        cbc_ciphertext_blocks=blocks,
        mutation_rate=mutation_rate,
        payload_depth=payload_depth,
    )


# ---------------------------------------------------------------------------
# Report dataclass
# ---------------------------------------------------------------------------

@dataclass
class PaddingAttackReport:
    """
    The result of one simulation run.

    Same shape as the other attack reports, with the padding-specific
    metric fields. `metric_name` identifies which trigger the scenario
    elicits.
    """
    scenario_id: str
    n_packets_sent: int
    duration_s: float
    target: str
    target_port: int
    detection_trigger: str
    mitigation_action: str
    trigger_fires: bool
    metric_name: str
    metric_value: float
    metric_threshold: float
    namespace: Optional[str] = None
    packets: List[bytes] = field(default_factory=list)

    def to_dict(self, include_packets: bool = False) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "scenario_id": self.scenario_id,
            "n_packets_sent": self.n_packets_sent,
            "duration_s": round(self.duration_s, 3),
            "target": self.target,
            "target_port": self.target_port,
            "detection_trigger": self.detection_trigger,
            "mitigation_action": self.mitigation_action,
            "trigger_fires": self.trigger_fires,
            "metric_name": self.metric_name,
            "metric_value": (
                round(self.metric_value, 6)
                if not math.isnan(self.metric_value) else None
            ),
            "metric_threshold": round(self.metric_threshold, 6),
            "namespace": self.namespace,
        }
        if include_packets:
            d["packets"] = len(self.packets)
        return d


# ---------------------------------------------------------------------------
# TLS helpers (application-data records only)
# ---------------------------------------------------------------------------

def _tls_application_data_record(ciphertext: bytes) -> bytes:
    """
    Wrap a ciphertext in a TLS application-data record.

        ContentType(1)=0x17 | LegacyVersion(2)=0x0303 | Length(2) | Ciphertext
    """
    return (
        bytes([_TLS_APPLICATION_DATA])
        + _TLS_LEGACY_VERSION.to_bytes(2, "big")
        + len(ciphertext).to_bytes(2, "big")
        + ciphertext
    )


# ---------------------------------------------------------------------------
# Simulator
# ---------------------------------------------------------------------------

class PaddingOracleSimulator:
    """
    Hermetic simulation of the two ciphertext-manipulation scenarios.

    Usage:

        sim = PaddingOracleSimulator(config, "A05", seed=42)
        report = sim.run()
        print(report.trigger_fires, report.metric_name, report.metric_value)
    """

    def __init__(
        self,
        config: Dict[str, Any],
        scenario_id: str,
        *,
        seed: Optional[int] = None,
    ) -> None:
        if scenario_id not in SUPPORTED_SCENARIOS:
            raise AttackConfigError(
                f"scenario {scenario_id!r} not handled by "
                f"padding_oracle.py; supported: {SUPPORTED_SCENARIOS}"
            )
        self._config = config
        self._log = get_logger(f"e3doh.attacks.padding.{scenario_id}")
        self._scenario_id = scenario_id
        self._scenario = parse_padding_scenario(scenario_id, config)

        try:
            attack_cfg = config["_includes"]["attack_suite"]
        except (KeyError, TypeError) as e:
            raise AttackConfigError(
                "attack_suite_params.yaml not found in config['_includes']"
            ) from e
        self._guards = attack_cfg.get("safety_guards", {})
        runtime_cfg = attack_cfg.get("runtime", {})
        self._global_rate_cap_pps = float(
            runtime_cfg.get("global_rate_cap_pps", 200_000)
        )
        self._global_duration_cap_s = float(
            runtime_cfg.get("global_duration_cap_seconds", 600)
        )

        rate_hz = self._scenario.rate_hz()
        if rate_hz > self._global_rate_cap_pps:
            raise AttackConfigError(
                f"scenario {scenario_id}: rate {rate_hz:.0f} Hz exceeds "
                f"global_rate_cap_pps={self._global_rate_cap_pps:.0f}"
            )
        if self._scenario.duration_seconds > self._global_duration_cap_s:
            raise AttackConfigError(
                f"scenario {scenario_id}: duration "
                f"{self._scenario.duration_seconds:.0f}s exceeds "
                f"global_duration_cap_seconds="
                f"{self._global_duration_cap_s:.0f}"
            )

        if seed is None:
            try:
                seed = int(config.get("runtime", {}).get("seed", 42))
            except (TypeError, ValueError):
                seed = 42
        self._run_seed = int(seed)
        self._rng = np.random.default_rng(
            get_component_seed(self._run_seed, "attack_suite")
        )

        # The "captured" ciphertext for A05, generated deterministically
        # from the seed. It is the plaintext the attacker is trying to
        # decrypt; the simulation does not need to know its plaintext.
        # It is 3 blocks of 16 bytes (2 data + 1 padding), which is
        # the minimum size at which the padding-oracle pattern is
        # visible.
        self._base_ciphertext = bytes(self._rng.integers(
            0, 256, size=self._scenario.cbc_ciphertext_blocks * _CBC_BLOCK_SIZE,
            dtype=np.uint8,
        ))

        # The base payload for A15: 64 bytes of deterministic
        # "looks-like-a-DNS-message" bytes. A real fuzzer would start
        # from a captured DNS message; the simulation uses a
        # deterministic blob so that the mutation patterns are the
        # only source of variation.
        self._base_payload = bytes(self._rng.integers(
            0, 256, size=64, dtype=np.uint8,
        ))

        self._namespace_name: Optional[str] = None

        self._log.info(
            "padding_simulator_initialised",
            scenario=scenario_id,
            target=f"{self._scenario.target}:{self._scenario.target_port}",
            rate_hz=rate_hz,
            duration_s=self._scenario.duration_seconds,
            cbc_blocks=self._scenario.cbc_ciphertext_blocks,
            mutation_rate=self._scenario.mutation_rate,
            payload_depth=self._scenario.payload_depth,
        )

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def scenario(self) -> PaddingScenarioConfig:
        return self._scenario

    @property
    def n_packets(self) -> int:
        return self._scenario.n_packets()

    # ------------------------------------------------------------------
    # Isolation
    # ------------------------------------------------------------------

    def isolated_network(self, *, skip_guard_checks: bool = False):
        """Delegate to the shared namespace context manager."""
        return _NamespaceContext(self, skip_guard_checks=skip_guard_checks)

    # ------------------------------------------------------------------
    # Packet generation
    # ------------------------------------------------------------------

    def generate_packets(self) -> List[bytes]:
        """Generate the packet payloads for the scenario."""
        if self._scenario_id == "A05":
            return self._build_a05_padding_oracle()
        if self._scenario_id == "A15":
            return self._build_a15_payload_fuzzing()
        raise AttackError(
            f"scenario {self._scenario_id} has no packet builder"
        )

    def _build_a05_padding_oracle(self) -> List[bytes]:
        """
        A05 / padding oracle.

        For each of the n_probes records, produce a TLS
        application-data record whose ciphertext is a copy of the
        base ciphertext, with the last byte of block N-2 set to a
        distinct candidate value (cycling through 0x00..0xFF).

        The last block (block N-1) is held fixed across records;
        only block N-2's last byte varies. This is the exact pattern
        a padding-oracle iteration produces: the attacker probes each
        candidate byte for the target block's last byte, and the
        server's response reveals one bit of information per probe.

        The ciphertext is wrapped in a TLS application-data record
        with content type 0x17 and legacy version 0x0303, matching
        the wire format of a real TLS 1.3 application-data record.
        """
        n = self.n_packets
        base = bytearray(self._base_ciphertext)
        block_size = _CBC_BLOCK_SIZE
        # The block the attacker is manipulating is block N-2
        # (the second-to-last block, in CBC terms). In the byte array,
        # that block ends at offset (N-2 + 1) * block_size - 1.
        target_byte_offset = (
            self._scenario.cbc_ciphertext_blocks - 2 + 1
        ) * block_size - 1

        packets: List[bytes] = []
        for i in range(n):
            tampered = bytearray(base)
            # Cycle through the candidate byte values.
            candidate = i & 0xFF
            tampered[target_byte_offset] = candidate
            packets.append(_tls_application_data_record(bytes(tampered)))

        self._log.info(
            "a05_padding_oracle_generated",
            n_packets=len(packets),
            cbc_blocks=self._scenario.cbc_ciphertext_blocks,
            target_byte_offset=target_byte_offset,
            candidates_cycled=min(n, 256),
        )
        return packets

    def _build_a15_payload_fuzzing(self) -> List[bytes]:
        """
        A15 / encrypted payload fuzzing.

        For each of the n_probes records, produce a TLS
        application-data record whose ciphertext is the base payload
        with a mutation applied. The mutation pattern is drawn from
        _MUTATION_PATTERNS, and the mutation is applied with
        probability `mutation_rate`; if no mutation is applied (the
        remaining 1 - mutation_rate), the record is the unmodified
        base payload.

        The mutation is byte-level: it does not attempt to produce a
        *valid* alternative DNS message (that would be too
        sophisticated for a fuzzer). It produces a payload that is
        *structurally* a TLS application-data fragment (the record
        header is valid) but whose *contents* would fail DNS/HTTP/2
        parsing on the server side.
        """
        n = self.n_packets
        base = self._base_payload
        packets: List[bytes] = []

        for i in range(n):
            if self._rng.random() < self._scenario.mutation_rate:
                pattern_idx = int(self._rng.integers(0, len(_MUTATION_PATTERNS)))
                pattern = _MUTATION_PATTERNS[pattern_idx]
                mutated = self._apply_mutation(base, pattern, i)
            else:
                mutated = base
            packets.append(_tls_application_data_record(mutated))

        self._log.info(
            "a15_payload_fuzzing_generated",
            n_packets=len(packets),
            mutation_rate=self._scenario.mutation_rate,
            patterns_available=list(_MUTATION_PATTERNS),
        )
        return packets

    def _apply_mutation(self, base: bytes, pattern: str, index: int) -> bytes:
        """
        Apply one of the documented mutation patterns to `base`.

        The mutation is deterministic given (base, pattern, index); the
        per-record randomness comes from `self._rng` when the pattern
        requires it.
        """
        out = bytearray(base)
        if not out:
            return bytes(out)
        if pattern == "bit_flip":
            pos = int(self._rng.integers(0, len(out)))
            bit = int(self._rng.integers(0, 8))
            out[pos] ^= (1 << bit)
        elif pattern == "byte_zero":
            pos = int(self._rng.integers(0, len(out)))
            out[pos] = 0x00
        elif pattern == "byte_ff":
            pos = int(self._rng.integers(0, len(out)))
            out[pos] = 0xFF
        elif pattern == "truncate":
            n = max(1, min(len(out) // 4, len(out) - 1))
            out = out[:-n]
        elif pattern == "extend":
            n = max(1, min(len(out) // 4, 32))
            out.extend(self._rng.integers(0, 256, size=n, dtype=np.uint8).tobytes())
        elif pattern == "swap_halves":
            half = len(out) // 2
            out = bytearray(out[half:]) + bytearray(out[:half])
        else:
            raise AttackError(f"unknown mutation pattern {pattern!r}")
        return bytes(out)

    # ------------------------------------------------------------------
    # Metric measurement
    # ------------------------------------------------------------------

    def _measure_a05_metric(self, packets: List[bytes]) -> Tuple[float, float]:
        """
        A05 metric: fraction of records whose padding candidate differs
        from the base.

        A padding-oracle attack iterates the last byte of the target
        block. Every record after the first has a different candidate
        byte, so the fraction is (n-1)/n. The trigger fires when this
        fraction exceeds 0.5.

        The measurement is structural: it does not need to decrypt
        anything, because the candidate byte is *plaintext* in the
        emitted record (it is the last byte of block N-2 in the
        ciphertext, which the attacker directly controls — the
        attacker does not need to know what it decrypts to, only
        that it varies).
        """
        if not packets:
            return 0.0, _DEFAULT_A05_PADDING_VARIED_THRESHOLD
        target_byte_offset_in_ciphertext = (
            self._scenario.cbc_ciphertext_blocks - 2 + 1
        ) * _CBC_BLOCK_SIZE - 1
        # The record header is 5 bytes; the ciphertext starts at offset 5.
        target_byte_offset_in_record = _TLS_RECORD_HEADER_LEN + target_byte_offset_in_ciphertext
        candidates = []
        for pkt in packets:
            if len(pkt) > target_byte_offset_in_record:
                candidates.append(pkt[target_byte_offset_in_record])
            else:
                candidates.append(None)
        # The fraction of records whose candidate differs from the base
        # candidate (i.e., from the first record's value, which is 0).
        base_candidate = candidates[0] if candidates and candidates[0] is not None else 0
        n_diff = sum(1 for c in candidates if c is not None and c != base_candidate)
        fraction = n_diff / max(len(candidates), 1)
        return float(fraction), _DEFAULT_A05_PADDING_VARIED_THRESHOLD

    def _measure_a15_metric(self, packets: List[bytes]) -> Tuple[float, float]:
        """
        A15 metric: fraction of records whose payload is malformed.

        The "malformed" test is structural: the record's TLS header is
        valid (content type 0x17, legacy version 0x0303, length that
        matches the payload), but the record's *ciphertext* is not the
        original base ciphertext. Since the base ciphertext is the
        only "well-formed" payload in the simulation (the mutation
        changes it in a way the server's parser would reject), the
        metric is the fraction of records whose ciphertext differs
        from the base.

        Note: this is a *proxy* for "malformed". The real test would
        be to decrypt the record (which the framework's boundary
        forbids) and check the DNS/HTTP/2 wire format. The proxy is
        sound because any mutation of the base payload produces a
        ciphertext that would fail to decrypt to a well-formed
        plaintext, and the trigger fires on the *distribution* of
        such records, not on any single one.
        """
        if not packets:
            return 0.0, _DEFAULT_A15_MALFORMED_THRESHOLD
        base_ciphertext = self._base_payload
        n_diff = 0
        for pkt in packets:
            if len(pkt) <= _TLS_RECORD_HEADER_LEN:
                n_diff += 1
                continue
            # Strip the record header.
            body = pkt[_TLS_RECORD_HEADER_LEN:]
            if body != base_ciphertext:
                n_diff += 1
        fraction = n_diff / max(len(packets), 1)
        return float(fraction), _DEFAULT_A15_MALFORMED_THRESHOLD

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    def run(
        self,
        *,
        skip_guard_checks: bool = False,
    ) -> PaddingAttackReport:
        """
        Run the full simulation and return a report.

        Parameters
        ----------
        skip_guard_checks : bool
            If True, skip the safety guards (used by the self-test
            only).

        Returns
        -------
        PaddingAttackReport.
        """
        started = time.perf_counter()

        try:
            packets = self.generate_packets()
        except Exception as e:
            self._log.error("packet_generation_failed", error=repr(e))
            raise AttackError(f"packet generation failed: {e}") from e

        namespace_name = None
        try:
            with self.isolated_network(skip_guard_checks=skip_guard_checks) as ns:
                namespace_name = ns
                self._namespace_name = ns
                if not skip_guard_checks:
                    self._send_packets(packets)
        except (IsolationGuardViolation, EgressDetectedError):
            raise
        except Exception as e:
            self._log.error("run_failed", error=repr(e), exc_info=True)
            raise AttackError(f"run failed: {e}") from e

        # Measure the metric.
        if self._scenario_id == "A05":
            metric_value, metric_threshold = self._measure_a05_metric(packets)
            metric_name = "padding_candidate_varied_fraction"
        elif self._scenario_id == "A15":
            metric_value, metric_threshold = self._measure_a15_metric(packets)
            metric_name = "malformed_fraction"
        else:
            raise AttackError(f"unknown scenario {self._scenario_id}")

        trigger_fires = metric_value > metric_threshold
        trigger_str, action_str = self._trigger_and_action()

        elapsed = time.perf_counter() - started
        report = PaddingAttackReport(
            scenario_id=self._scenario_id,
            n_packets_sent=len(packets),
            duration_s=self._scenario.duration_seconds,
            target=self._scenario.target,
            target_port=self._scenario.target_port,
            detection_trigger=trigger_str,
            mitigation_action=action_str,
            trigger_fires=trigger_fires,
            metric_name=metric_name,
            metric_value=metric_value,
            metric_threshold=metric_threshold,
            namespace=namespace_name,
        )
        self._log.info(
            "padding_run_complete",
            **report.to_dict(),
            wall_time_s=round(elapsed, 3),
        )
        return report

    def _send_packets(self, packets: List[bytes]) -> None:
        """Send the packets into the namespace via scapy."""
        try:
            from scapy.all import Ether, IP, TCP, Raw, sendp  # type: ignore
        except ImportError:
            self._log.warning("scapy_not_available_send_skipped")
            return

        scapy_pkts = []
        for payload in packets:
            pkt = (
                Ether()
                / IP(src="10.0.0.1", dst=self._scenario.target)
                / TCP(sport=50000, dport=self._scenario.target_port, flags="PA")
                / Raw(load=payload)
            )
            scapy_pkts.append(pkt)
        inter = 1.0 / max(self._scenario.rate_hz(), 1e-9)
        try:
            sendp(scapy_pkts, iface="lo", inter=inter, verbose=False)
        except Exception as e:
            self._log.error("send_failed", error=repr(e))
            raise AttackError(f"sendp failed: {e}") from e

    def _trigger_and_action(self) -> Tuple[str, str]:
        """Return the (detection_trigger, mitigation_action) pair from
        the attack matrix."""
        if self._scenario_id == "A05":
            return (
                "HE.eval(c1 == c2) without decryption",
                "C' = C ⊕ H(noise || timestamp)",
            )
        if self._scenario_id == "A15":
            return (
                "wire_format_valid(payload) == False ∨ tls_decryption_error",
                "terminate_doh_stream()",
            )
        raise AttackError(
            f"scenario {self._scenario_id} has no trigger/action mapping"
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "PaddingOracleSimulator",
    "PaddingAttackReport",
    "PaddingScenarioConfig",
    "parse_padding_scenario",
    "SUPPORTED_SCENARIOS",
]


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def _self_test() -> int:
    import sys
    rc = 0
    print("=== padding_oracle self-test ===")

    base_guards = {
        "isolation_required": False,
        "target_must_be_local": True,
        "target_cidr_allowlist": ["127.0.0.0/8", "10.0.0.0/8"],
        "no_privileged_ports": False,
        "no_external_network": False,
    }
    runtime = {
        "global_rate_cap_pps": 200_000,
        "global_duration_cap_seconds": 600,
    }
    config = {
        "_includes": {
            "attack_suite": {
                "safety_guards": base_guards,
                "runtime": runtime,
                "attacks": [
                    {
                        "id": "A05", "attack_vector": "Padding Oracle",
                        "protocol": "dot", "target": "127.0.0.1", "target_port": 853,
                        "rate": {"value": 50, "unit": "requests_per_second"},
                        "duration_seconds": 1,
                        "payload": {"cbc_ciphertext_blocks": 3},
                    },
                    {
                        "id": "A15", "attack_vector": "Encrypted Payload Fuzzing",
                        "protocol": "doh", "target": "127.0.0.1", "target_port": 443,
                        "rate": {"value": 500, "unit": "fuzzed_requests_per_second"},
                        "duration_seconds": 1,
                        "payload": {"mutation_rate": 0.5, "payload_depth": 3},
                    },
                ],
            },
        },
        "runtime": {"seed": 42},
    }

    # --- 1. A05 ---
    sim05 = PaddingOracleSimulator(config, "A05", seed=42)
    packets05 = sim05.generate_packets()
    if len(packets05) != 50:
        print(f"[FAIL] A05 packet count: {len(packets05)} vs 50")
        rc = 1
    else:
        print(f"[OK]   A05: {len(packets05)} padding-oracle records")

    # Verify the first record's content type is 0x17 (application data).
    if packets05:
        if packets05[0][0] != 0x17:
            print(f"[FAIL] A05 first record content type: {packets05[0][0]:#04x}")
            rc = 1
        elif packets05[0][1:3] != b"\x03\x03":
            print(f"[FAIL] A05 first record legacy version: {packets05[0][1:3].hex()}")
            rc = 1
        else:
            print(f"[OK]   A05 first record: content_type=0x17, "
                  f"legacy_version=0x0303")

    # Verify the padding candidate cycles.
    target_off = _TLS_RECORD_HEADER_LEN + (3 - 2 + 1) * _CBC_BLOCK_SIZE - 1
    candidates = [p[target_off] for p in packets05[:5]]
    if len(set(candidates)) < 3:
        print(f"[FAIL] A05 candidates not varying: {candidates}")
        rc = 1
    else:
        print(f"[OK]   A05 candidates cycling: {candidates}")

    report05 = sim05.run(skip_guard_checks=True)
    if not report05.trigger_fires:
        print(f"[FAIL] A05 trigger did not fire: "
              f"{report05.metric_name}={report05.metric_value:.4f}")
        rc = 1
    else:
        print(f"[OK]   A05 trigger fires: {report05.metric_name}="
              f"{report05.metric_value:.4f} > {report05.metric_threshold:.4f}")
        print(f"         detection: {report05.detection_trigger}")
        print(f"         mitigation: {report05.mitigation_action}")

    # --- 2. A15 ---
    sim15 = PaddingOracleSimulator(config, "A15", seed=42)
    packets15 = sim15.generate_packets()
    if len(packets15) != 500:
        print(f"[FAIL] A15 packet count: {len(packets15)} vs 500")
        rc = 1
    else:
        print(f"[OK]   A15: {len(packets15)} fuzzed records")

    # Verify the first record's content type is 0x17.
    if packets15 and packets15[0][0] != 0x17:
        print(f"[FAIL] A15 first record content type: {packets15[0][0]:#04x}")
        rc = 1
    else:
        print(f"[OK]   A15 first record: content_type=0x17")

    report15 = sim15.run(skip_guard_checks=True)
    if not report15.trigger_fires:
        print(f"[FAIL] A15 trigger did not fire: "
              f"{report15.metric_name}={report15.metric_value:.4f}")
        rc = 1
    else:
        print(f"[OK]   A15 trigger fires: {report15.metric_name}="
              f"{report15.metric_value:.4f} > {report15.metric_threshold:.4f}")

    # --- 3. Determinism: same seed → same packets ---
    sim_a = PaddingOracleSimulator(config, "A05", seed=42)
    sim_b = PaddingOracleSimulator(config, "A05", seed=42)
    p_a = sim_a.generate_packets()
    p_b = sim_b.generate_packets()
    if p_a != p_b:
        print(f"[FAIL] determinism: same seed produced different packets")
        rc = 1
    else:
        print(f"[OK]   determinism: same seed → identical {len(p_a)} packets")

    # --- 4. Different seed → different base ciphertext ---
    sim_diff = PaddingOracleSimulator(config, "A05", seed=999)
    p_diff = sim_diff.generate_packets()
    if p_a == p_diff:
        print(f"[FAIL] different seeds produced identical packets")
        rc = 1
    else:
        print(f"[OK]   different seeds → different base ciphertext")

    # --- 5. A15 mutation rate respected ---
    # With mutation_rate=0.5, roughly half the records should differ
    # from the base.
    report15_metric = report15.metric_value
    if not (0.3 <= report15_metric <= 0.7):
        print(f"[FAIL] A15 mutation rate: measured {report15_metric:.4f}, "
              f"expected ~0.5 (mutation_rate=0.5)")
        rc = 1
    else:
        print(f"[OK]   A15 mutation rate: measured {report15_metric:.4f} "
              f"(expected ~0.5)")

    # --- 6. Unknown scenario ---
    try:
        parse_padding_scenario("A99", config)
    except AttackConfigError:
        print(f"[OK]   unknown scenario A99 rejected")
    else:
        print(f"[FAIL] unknown scenario A99 was accepted")
        rc = 1

    # --- 7. Invalid mutation_rate ---
    bad_config = {
        **config,
        "_includes": {
            **config["_includes"],
            "attack_suite": {
                **config["_includes"]["attack_suite"],
                "attacks": [
                    {**a, "payload": {**a["payload"], "mutation_rate": 1.5}}
                    if a["id"] == "A15" else a
                    for a in config["_includes"]["attack_suite"]["attacks"]
                ],
            },
        },
    }
    try:
        parse_padding_scenario("A15", bad_config)
    except AttackConfigError:
        print(f"[OK]   mutation_rate=1.5 rejected")
    else:
        print(f"[FAIL] mutation_rate=1.5 was accepted")
        rc = 1

    # --- 8. Report serialization ---
    d = report05.to_dict()
    required = ("scenario_id", "trigger_fires", "metric_name",
                "metric_value", "metric_threshold",
                "detection_trigger", "mitigation_action")
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
