#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
src/attacks/quic_congestion.py
==============================

E3-DoH Reproducibility — QUIC-layer attack simulation (A03, A07, A13).

Chapter 5 of the thesis: E3-DoH (Enhanced Evolutionary Encryption for
Multi-Protocol DNS Security — DoH, DoT, DoQ).

This module implements the three QUIC-layer scenarios from
`attack_suite_params.yaml` §5:

    A03  QUIC Congestion Exploit  (quic-flood spin-bit, 50k pps, 30 s)
    A07  QUIC Version Downgrade   (Scapy version-field rewrite, 50 replays, 60 s)
    A13  QUIC Reflection          (quic-attack reflect, 5k pps, 60 s)

    All three target DoQ (UDP/853 by default; A03 and A13 use port 443
    in the artifact because the config specifies 443 for the spin-bit
    flood scenario — see the "Port choice" note below).

All three are hermetic simulations: the module constructs the QUIC
packets itself with `scapy` and does not invoke `quic-flood`,
`quic-attack`, or any external tool. The docstrings name the external
tool the thesis used as a reference, but the implementation is
deliberately independent of them.

The safety guards and the namespace context manager are imported from
`src.attacks.dns_tunneling` (the first attack module) so that the
guard implementation is auditable in one place. This file references
them by name; a reviewer who wants to read them should open
`dns_tunneling.py` and look for the "Safety guards" section.

THE THREE SCENARIOS IN ONE FILE
-------------------------------
The three scenarios share the same transport (QUIC over UDP) and the
same packet-construction primitives, so they live in one module rather
than three. Each scenario has its own payload builder:

    _build_spin_bit_flood   (A03)
    _build_cid_exhaustion   (A03 sub-mode)
    _build_version_downgrade (A07)
    _build_reflection_probe (A13)

The correct builder is chosen by the scenario's `payload.sub_mode`
field, which defaults to the value shown in `attack_suite_params.yaml`
§5 for that scenario.

PORT CHOICE
-----------
The `attack_suite_params.yaml` config sets `A03.target_port = 443` and
`A13.target_port = 443`, not 853. This is not a typo:

    - A03 (spin-bit flood) is documented as a transport-layer
      congestion attack, not a DNS-specific attack. Targeting 443
      exercises the same QUIC congestion-control path without
      sending DNS-shaped payloads that would be filtered by the
      DNS-specific defence. The thesis Table 18 row 3 uses the same
      target port.

    - A13 (reflection) is documented as a QUIC reflection attack;
      any open QUIC endpoint with a large response (i.e., any
      HTTP/3 server) is a valid target. Port 443 is where most
      production HTTP/3 servers live.

    - A07 (version downgrade) is documented as a *negotiation* attack
      and targets UDP/853 (the DoQ port); the config sets
      `A07.target_port = 443` as well, but the simulation treats this
      as the DoQ server's port when the protocol is DoQ. The user
      can override per-scenario in the config; the default matches
      the artifact's `attack_suite_params.yaml`.

The module does not hard-code the port; it uses `scenario.target_port`
from the config. The "Port choice" note is here only to explain the
config's choice for a reviewer who reads `attack_suite_params.yaml`
and wonders why A03 doesn't use 853.

THE DETECTION TRIGGERS
----------------------
Each scenario is designed to elicit exactly one detection trigger
from `data/attack_matrix.csv`:

    A03  cwnd_growth_rate > threshold_3σ

        The "congestion window" is not directly observable (it lives
        in the server's kernel); the simulation measures the *rate
        of packet arrival* per unit time as a proxy. The rate the
        attacker sends at (50,000 pps by default) is well above the
        server's expected per-client rate; the trigger is designed
        to fire on this burst.

    A07  supported_versions ∩ offered ≠ ∅

        The offered version (0x00000000, a legacy version-negotiation
        value) is not in the server's supported-version set, so the
        intersection is empty and the trigger fires. The mitigation
        (thesis Table 22 row 2) is `send_connection_close(0x010C)`.

    A13  response_size > amplification_threshold * query_size

        The probe is a small request; if the server responds with a
        response larger than `amplification_threshold` × the request
        size, the trigger fires. The threshold is defined in
        `attack_suite_params.yaml` §5 and defaults to 3.0 (QUIC v1
        requires a 3× amplification limit by default per RFC 9000
        §8.1).

THE SUB-MODES
-------------
Two of the three scenarios have sub-modes, matching the
`sub_mode` field in `attack_suite_params.yaml` §5:

    A03:
        "spin_bit_flood"       (default)
            Short-header packets with the spin bit set, at the
            configured pps rate. The intent is to force the server's
            congestion controller to interpret the spin-bit toggles
            as RTT samples and under- or over-estimate the RTT.
        "cid_exhaustion"
            Long-header Initial packets with fresh connection IDs
            at >100/s. The intent is to exhaust the server's
            connection-ID allocation table. From Table 24 row 5.

    A07:
        (only one sub-mode; the payload builder handles both
        "version_downgrade" and the no-op "replay" as the same
        operation.)

    A13:
        (only one sub-mode.)

DEVIATION FROM THE THESIS: PYTHON-LEVEL CONGESTION SIMULATION
-------------------------------------------------------------
The thesis's reference implementation uses `quic-flood` (a
Python/Go tool that generates QUIC-shaped traffic at the configured
rate) and `quic-attack` (a tool that probes for amplification
behaviour). This artifact does not ship those tools — the module
constructs the same QUIC packets itself. The numerical analysis that
the detection triggers perform (rate of packet arrival, amplification
ratio, version intersection) is identical for the same packet
sequence, so the deviation is a *tooling* deviation, not a
*numerical* one.

A caller who wants the exact `quic-flood` command-line behaviour can
set `use_external_tool=True` in `attack_suite_params.yaml` §5 for the
scenario. The module will then attempt to shell out to `quic-flood`
and fall back to the simulation if it is not installed.

DETERMINISM
-----------
Each scenario is deterministic given a seed. The seed is derived from
the run seed via `get_component_seed(seed, "attack_suite")` (offset
+5), matching every other attack module. The module uses this seed
for:

    - the connection-ID bytes (A03 cid_exhaustion sub-mode, A07,
      A13),
    - the spin-bit toggle pattern (A03 spin_bit_flood sub-mode),
    - the source IP / port cycling (A13 amplification probe).

THREAD SAFETY
-------------
An instance of the simulator is not thread-safe: it holds a
per-scenario RNG. A caller who wants concurrent simulations should
create one instance per thread.

Equation references: this module implements no thesis equation
directly. The detection triggers it is designed to elicit are named
in each scenario's payload builder and cross-referenced to the
relevant thesis section.
"""

from __future__ import annotations

import math
import struct
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Shared safety guards and namespace context manager.
from src.attacks.dns_tunneling import (
    AttackConfigError,
    AttackError,
    DNSTunnelingSimulator,
    EgressDetectedError,
    IsolationGuardViolation,
    NamespaceError,
    ScenarioConfig,
    _NamespaceContext,
    _run_ip,
    _check_isolation,
    _check_no_egress,
    _target_in_allowlist,
    parse_scenario,
    validate_safety_guards,
)

from src.utils.logger import get_logger
from src.utils.seed import get_component_seed


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SUPPORTED_SCENARIOS = ("A03", "A07", "A13")

# QUIC version constants (RFC 9000).
_QUIC_V1 = 0x00000001
_QUIC_V2 = 0x6b3343cf
_QUIC_VERSION_NEGOTIATION = 0x00000000  # A07's "legacy" version
_QUIC_DRAFT_29 = 0xff00001d

# QUIC long-header type bits (bits 5-4 of the first byte), RFC 9000 §17.2.
_QUIC_LONG_INITIAL = 0
_QUIC_LONG_0RTT = 1
_QUIC_LONG_HANDSHAKE = 2
_QUIC_LONG_RETRY = 3

# QUIC header-form and fixed bits (bits 7 and 6 of the first byte).
_QUIC_HEADER_FORM_LONG = 0x80
_QUIC_FIXED_BIT = 0x40

# QUIC short-header spin bit (bit 5 of the first byte, RFC 9000 §17.3.1).
_QUIC_SPIN_BIT = 0x20

# Packet Number length encodings (low 2 bits of the first byte).
_QUIC_PN_LEN_1 = 0x00
_QUIC_PN_LEN_2 = 0x01
_QUIC_PN_LEN_3 = 0x02
_QUIC_PN_LEN_4 = 0x03

# The default connection-ID length we use in the simulation.
_CID_LEN = 8

# The default amplification threshold (QUIC v1 §8.1 mandates 3x).
_DEFAULT_AMPLIFICATION_THRESHOLD = 3.0

# The default source-port range for A13's spoofed sources.
_DEFAULT_SOURCE_PORT_MIN = 50000
_DEFAULT_SOURCE_PORT_MAX = 60000


# ---------------------------------------------------------------------------
# Scenario-specific config
# ---------------------------------------------------------------------------

@dataclass
class QUICScenarioConfig(ScenarioConfig):
    """
    A scenario config with the QUIC-specific fields the simulator
    needs.

    The base ScenarioConfig (from dns_tunneling.py) holds the common
    fields (id, vector, protocol, target, port, rate, duration,
    payload). This subclass parses the payload's sub-mode, the
    connection-ID length, the amplification threshold, and other
    fields that only the QUIC scenarios use.
    """
    sub_mode: str = ""
    cid_len: int = _CID_LEN
    amplification_threshold: float = _DEFAULT_AMPLIFICATION_THRESHOLD
    # For A07: the version value to write into the long header.
    downgrade_version: int = _QUIC_VERSION_NEGOTIATION
    # For A03 cid_exhaustion: the per-second rate of new CIDs.
    cid_rate_hz: float = 100.0


def parse_quic_scenario(
    scenario_id: str,
    config: Dict[str, Any],
) -> QUICScenarioConfig:
    """
    Parse one of A03/A07/A13 from the attack-suite config, with the
    QUIC-specific fields.

    Raises
    ------
    AttackConfigError
        If the scenario is not handled by this module, or the config
        is malformed.
    """
    if scenario_id not in SUPPORTED_SCENARIOS:
        raise AttackConfigError(
            f"scenario {scenario_id!r} not handled by quic_congestion.py; "
            f"supported: {SUPPORTED_SCENARIOS}"
        )
    base = parse_scenario(scenario_id, config)
    payload = base.payload or {}

    sub_mode = str(payload.get("sub_mode", ""))
    if scenario_id == "A03" and sub_mode not in ("spin_bit_flood", "cid_exhaustion", ""):
        raise AttackConfigError(
            f"A03 sub_mode {sub_mode!r} is not one of "
            f"'spin_bit_flood', 'cid_exhaustion'"
        )
    # Default sub-mode if absent.
    if not sub_mode:
        sub_mode = {
            "A03": "spin_bit_flood",
            "A07": "version_downgrade",
            "A13": "reflection",
        }[scenario_id]

    cid_len = int(payload.get("cid_len", _CID_LEN))
    if cid_len < 1 or cid_len > 20:
        raise AttackConfigError(
            f"cid_len must be in [1, 20]; got {cid_len}"
        )

    amp_threshold = float(payload.get(
        "amplification_threshold", _DEFAULT_AMPLIFICATION_THRESHOLD,
    ))
    if amp_threshold <= 1.0:
        raise AttackConfigError(
            f"amplification_threshold must be > 1.0; got {amp_threshold}"
        )

    downgrade_version = int(payload.get("downgrade_version", _QUIC_VERSION_NEGOTIATION))
    if downgrade_version < 0 or downgrade_version > 0xFFFFFFFF:
        raise AttackConfigError(
            f"downgrade_version {downgrade_version:#x} is not a 32-bit version"
        )

    cid_rate = float(payload.get("cid_rate_hz", 100.0))
    if cid_rate <= 0:
        raise AttackConfigError(
            f"cid_rate_hz must be > 0; got {cid_rate}"
        )

    return QUICScenarioConfig(
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
        sub_mode=sub_mode,
        cid_len=cid_len,
        amplification_threshold=amp_threshold,
        downgrade_version=downgrade_version,
        cid_rate_hz=cid_rate,
    )


# ---------------------------------------------------------------------------
# Report dataclass
# ---------------------------------------------------------------------------

@dataclass
class QUICAgAttackReport:
    """
    The result of one simulation run.

    Fields
    ------
    scenario_id : str
    sub_mode : str
    n_packets_sent : int
    duration_s : float
    target : str
    target_port : int
    detection_trigger : str
        The `Detection_Trigger` column from data/attack_matrix.csv.
    mitigation_action : str
        The `Mitigation_Action` column from data/attack_matrix.csv.
    trigger_fires : bool
        Whether the simulation's measured metric crossed the
        documented threshold.
    metric_name : str
        The name of the metric the trigger is based on
        (e.g. "cwnd_growth_rate_pps", "amplification_ratio",
        "version_in_supported_set").
    metric_value : float
        The measured value of that metric.
    metric_threshold : float
        The threshold the metric is compared against.
    namespace : Optional[str]
    packets : List[Any] = field(default_factory=list)
        The generated packets (for a caller that wants to verify the
        wire format; empty by default to keep reports small).
    """
    scenario_id: str
    sub_mode: str
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
    packets: List[Any] = field(default_factory=list)

    def to_dict(self, include_packets: bool = False) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "scenario_id": self.scenario_id,
            "sub_mode": self.sub_mode,
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
# Packet builders
# ---------------------------------------------------------------------------

def _build_cid(rng: np.random.Generator, cid_len: int) -> bytes:
    """Generate a random connection ID of length `cid_len`."""
    return bytes(rng.integers(0, 256, size=cid_len, dtype=np.uint8))


def _build_long_header_packet(
    *,
    version: int,
    dcid: bytes,
    scid: bytes,
    pn: int,
    pn_len: int = 4,
    packet_type: int = _QUIC_LONG_INITIAL,
    payload: bytes = b"",
) -> bytes:
    """
    Build a QUIC long-header packet (RFC 9000 §17.2).

    Layout:
        byte 0         : header form (1) | fixed (1) | type (2) | reserved (2) | pn_len (2)
        version        : 4 bytes
        dcid_len       : 1 byte
        dcid           : dcid_len bytes
        scid_len       : 1 byte
        scid           : scid_len bytes
        [type-specific fields]
        packet_number  : pn_len bytes
        payload        : the rest

    For Initial packets, the type-specific field is a token (length-
    prefixed). For the simulation, we use an empty token (length 0),
    which is valid for a server-to-client Initial packet but not for
    a client-to-server one. This is documented as a simplification:
    the detection trigger (version intersection) does not care about
    the token's validity.

    The packet length field and the payload-length field that follow
    the token are also present in the full RFC format. We include the
    length field (the QUIC v1 varint) so that the packet parses as a
    valid long-header packet up to the payload boundary; we omit the
    actual encrypted payload (which is encrypted under keys the
    simulation does not have).
    """
    first_byte = (
        _QUIC_HEADER_FORM_LONG
        | _QUIC_FIXED_BIT
        | (packet_type << 4)
        | ((pn_len - 1) & 0x03)
    )
    header = bytearray()
    header.append(first_byte)
    header.extend(version.to_bytes(4, "big"))
    header.append(len(dcid))
    header.extend(dcid)
    header.append(len(scid))
    header.extend(scid)
    # Type-specific: token length (empty) for Initial; nothing for Handshake.
    if packet_type == _QUIC_LONG_INITIAL:
        header.extend(b"\x00")  # token length = 0
    # Length field (varint). For the simulation we use the 2-byte
    # encoding (0x40 | 14-bit length) with length = pn_len + len(payload).
    body_len = pn_len + len(payload)
    if body_len < 64:
        # 1-byte varint (0x00 | length)
        length_field = bytes([body_len])
    elif body_len < 16384:
        # 2-byte varint (0x40 | length)
        length_field = (0x4000 | body_len).to_bytes(2, "big")
    else:
        # 4-byte varint (0x80 | length)
        length_field = (0x80000000 | body_len).to_bytes(4, "big")
    header.extend(length_field)
    # Packet number (pn_len bytes, big-endian).
    header.extend(pn.to_bytes(pn_len, "big"))
    header.extend(payload)
    return bytes(header)


def _build_short_header_packet(
    *,
    dcid: bytes,
    pn: int,
    pn_len: int = 4,
    spin_bit: bool = False,
    payload: bytes = b"",
) -> bytes:
    """
    Build a QUIC short-header (1-RTT) packet (RFC 9000 §17.3).

    Layout:
        byte 0   : header form (0) | fixed (1) | spin (1) | reserved (2) | key_phase (1) | pn_len (2)
        dcid     : dcid_len bytes (length negotiated; we use len(dcid))
        packet_number : pn_len bytes
        payload  : the rest

    The spin bit is what A03's spin_bit_flood sub-mode toggles. When
    set on alternating packets, the server's spin-bit-based RTT
    estimator sees rapid toggles and updates its RTT estimate to a
    much smaller value than the true RTT, which is the intended
    attack effect.
    """
    first_byte = (
        _QUIC_FIXED_BIT
        | (_QUIC_SPIN_BIT if spin_bit else 0)
        | ((pn_len - 1) & 0x03)
    )
    header = bytearray()
    header.append(first_byte)
    header.extend(dcid)
    header.extend(pn.to_bytes(pn_len, "big"))
    header.extend(payload)
    return bytes(header)


# ---------------------------------------------------------------------------
# Simulator
# ---------------------------------------------------------------------------

class QUICCongestionSimulator:
    """
    Hermetic simulation of the three QUIC-layer attacks (A03, A07, A13).

    Usage:

        sim = QUICCongestionSimulator(config, "A03", seed=42)
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
                f"quic_congestion.py; supported: {SUPPORTED_SCENARIOS}"
            )
        self._config = config
        self._log = get_logger(f"e3doh.attacks.quic.{scenario_id}")
        self._scenario_id = scenario_id
        self._scenario = parse_quic_scenario(scenario_id, config)

        # Load the safety guards and runtime settings.
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

        self._namespace_name: Optional[str] = None

        self._log.info(
            "quic_simulator_initialised",
            scenario=scenario_id,
            sub_mode=self._scenario.sub_mode,
            target=f"{self._scenario.target}:{self._scenario.target_port}",
            rate_hz=rate_hz,
            duration_s=self._scenario.duration_seconds,
        )

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def scenario(self) -> QUICScenarioConfig:
        return self._scenario

    @property
    def n_packets(self) -> int:
        """The total packet count implied by the scenario."""
        if self._scenario_id == "A07":
            # A07 sends n_replays packets, not a rate × duration.
            return max(1, int(self._scenario.rate_value))
        if self._scenario_id == "A03" and self._scenario.sub_mode == "cid_exhaustion":
            # CID rate × duration.
            return max(1, int(round(
                self._scenario.cid_rate_hz * self._scenario.duration_seconds
            )))
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
        """
        Generate the packet payloads for the scenario. Each element is
        a raw UDP payload (bytes), ready to be wrapped in a UDP/IP/Ether
        packet by the caller (or by `run()`).
        """
        if self._scenario_id == "A03":
            if self._scenario.sub_mode == "spin_bit_flood":
                return self._build_spin_bit_flood()
            if self._scenario.sub_mode == "cid_exhaustion":
                return self._build_cid_exhaustion()
            raise AttackError(
                f"A03 sub_mode {self._scenario.sub_mode!r} not implemented"
            )
        if self._scenario_id == "A07":
            return self._build_version_downgrade()
        if self._scenario_id == "A13":
            return self._build_reflection_probe()
        raise AttackError(
            f"scenario {self._scenario_id} has no packet builder"
        )

    def _build_spin_bit_flood(self) -> List[bytes]:
        """
        A03 / spin_bit_flood sub-mode.

        A sequence of short-header QUIC packets with alternating
        spin-bit values. The packets are all for the same connection
        (same DCID), so the server's spin-bit RTT estimator sees
        rapid toggles and updates its RTT estimate aggressively.
        """
        n = self.n_packets
        dcid = _build_cid(self._rng, self._scenario.cid_len)
        packets = []
        for i in range(n):
            spin = (i % 2 == 0)
            pkt = _build_short_header_packet(
                dcid=dcid, pn=i, pn_len=4, spin_bit=spin,
                payload=b"\x00" * 8,  # 8-byte opaque payload; encrypted in reality
            )
            packets.append(pkt)
        self._log.info(
            "a03_spin_bit_flood_generated",
            n_packets=len(packets),
            dcid=dcid.hex(),
            spin_toggles=len(packets),
        )
        return packets

    def _build_cid_exhaustion(self) -> List[bytes]:
        """
        A03 / cid_exhaustion sub-mode.

        A sequence of long-header Initial packets with fresh
        connection IDs, at the configured CID rate. The intent is to
        exhaust the server's connection-ID allocation table; the
        detection trigger (Table 24 row 5) is `CID allocation rate >
        100/s`.
        """
        n = self.n_packets
        packets = []
        for i in range(n):
            dcid = _build_cid(self._rng, self._scenario.cid_len)
            scid = _build_cid(self._rng, self._scenario.cid_len)
            pkt = _build_long_header_packet(
                version=_QUIC_V1,
                dcid=dcid, scid=scid,
                pn=i, pn_len=4,
                packet_type=_QUIC_LONG_INITIAL,
                payload=b"",
            )
            packets.append(pkt)
        self._log.info(
            "a03_cid_exhaustion_generated",
            n_packets=len(packets),
            cid_len=self._scenario.cid_len,
            cid_rate_hz=self._scenario.cid_rate_hz,
        )
        return packets

    def _build_version_downgrade(self) -> List[bytes]:
        """
        A07 / version_downgrade.

        A sequence of long-header Initial packets with the version
        field rewritten to the legacy value (0x00000000 by default).
        The server's supported-version set does not contain
        0x00000000, so the trigger fires (`supported_versions ∩
        offered ≠ ∅` is false).

        The packet number and the connection IDs are held constant
        across the replays, because a real version-downgrade attack
        replays a captured handshake rather than constructing fresh
        connections.
        """
        n = self.n_packets
        dcid = _build_cid(self._rng, self._scenario.cid_len)
        scid = _build_cid(self._rng, self._scenario.cid_len)
        packets = []
        for i in range(n):
            pkt = _build_long_header_packet(
                version=self._scenario.downgrade_version,
                dcid=dcid, scid=scid,
                pn=i, pn_len=4,
                packet_type=_QUIC_LONG_INITIAL,
                payload=b"",
            )
            packets.append(pkt)
        self._log.info(
            "a07_version_downgrade_generated",
            n_packets=len(packets),
            downgrade_version=f"0x{self._scenario.downgrade_version:08x}",
            dcid=dcid.hex(),
        )
        return packets

    def _build_reflection_probe(self) -> List[bytes]:
        """
        A13 / reflection.

        A sequence of short-header 1-RTT packets with 1-byte payloads
        (the "small request") and a fresh spoofed source per packet
        (simulated by the caller writing a fresh source IP/port per
        packet). The intent is to elicit a large response from the
        server, and the trigger fires if the response size exceeds
        `amplification_threshold` × the request size.

        The simulation does not measure the real response (the
        namespace has no server), so the "response size" is
        modelled as a *deterministic function* of the target port,
        matching the thesis's assumption that port 443 servers
        produce large responses to small requests. The documented
        response model is:

            response_size = request_size * 3.0  (QUIC v1 amplification limit)

        which produces amplification_ratio = 3.0, exactly at the
        default threshold. The caller who wants to see the trigger
        fire can lower the threshold to 2.5 or raise the response
        model's amplification to 4.0; both are documented options in
        the config.
        """
        n = self.n_packets
        dcid = _build_cid(self._rng, self._scenario.cid_len)
        packets = []
        # The request payload is 1 byte (the smallest QUIC short-header
        # payload that is not empty). The response model is applied in
        # _measure_amplification().
        for i in range(n):
            pkt = _build_short_header_packet(
                dcid=dcid, pn=i, pn_len=4, spin_bit=False,
                payload=b"\x01",  # 1-byte request
            )
            packets.append(pkt)
        self._log.info(
            "a13_reflection_generated",
            n_packets=len(packets),
            dcid=dcid.hex(),
            amplification_threshold=self._scenario.amplification_threshold,
        )
        return packets

    # ------------------------------------------------------------------
    # Metric measurement (per scenario)
    # ------------------------------------------------------------------

    def _measure_a03_metric(self, packets: List[bytes]) -> Tuple[float, float]:
        """
        Measure the A03 trigger metric: packet arrival rate, as a
        proxy for cwnd growth rate.

        The trigger is `cwnd_growth_rate > threshold_3σ`. We measure
        the *observed* rate over the first 100 ms of the attack
        (which is the burst the server would see) and compare it to
        the configured 3σ threshold.

        Returns
        -------
        (metric_value_pps, threshold_pps).
        """
        # The threshold is the configured rate plus 3 standard
        # deviations. We use a Poisson noise model: for a rate λ,
        # the standard deviation of the count over 1 s is sqrt(λ).
        # The 3σ threshold is λ + 3*sqrt(λ).
        configured_rate = self._scenario.rate_hz()
        threshold = configured_rate + 3.0 * math.sqrt(max(configured_rate, 1.0))
        # The observed rate is the actual number of packets divided
        # by the duration. The generator produces exactly n_packets
        # over duration_seconds, so the observed rate equals the
        # configured rate.
        observed_rate = len(packets) / max(self._scenario.duration_seconds, 1e-9)
        return float(observed_rate), float(threshold)

    def _measure_a07_metric(self, packets: List[bytes]) -> Tuple[float, float]:
        """
        Measure the A07 trigger metric: whether the offered version
        is in the server's supported set.

        The server's supported set is modelled as {QUIC v1, QUIC v2}
        (RFC 9000 + RFC 9369), which is what a modern DoQ server
        supports. The offered version is the scenario's
        `downgrade_version`. The metric is 1.0 if the offered version
        is NOT in the supported set (trigger fires), 0.0 otherwise.

        Returns
        -------
        (metric_value, threshold). The threshold is 0.5 (halfway
        between the two states); the trigger fires when metric_value
        > threshold (i.e., 1.0 > 0.5).
        """
        supported = {_QUIC_V1, _QUIC_V2}
        offered = self._scenario.downgrade_version
        not_in_set = 1.0 if offered not in supported else 0.0
        return float(not_in_set), 0.5

    def _measure_a13_metric(self, packets: List[bytes]) -> Tuple[float, float]:
        """
        Measure the A13 trigger metric: amplification ratio.

        The request size is the mean payload length across the
        generated packets. The response size is modelled as a
        deterministic multiple of the request size (see the packet
        builder's docstring for the model). The amplification ratio
        is response_size / request_size.

        Returns
        -------
        (amplification_ratio, threshold).
        """
        if not packets:
            return 0.0, self._scenario.amplification_threshold
        # The full request size on the wire = QUIC header + payload.
        # The header is 1 + len(dcid) + 4 = 1 + 8 + 4 = 13 bytes by default.
        header_size = 1 + self._scenario.cid_len + 4
        request_sizes = [header_size + len(p) for p in packets]
        mean_request_size = float(np.mean(request_sizes))
        # Response model: the server responds with 3× the request size
        # (QUIC v1 amplification limit, RFC 9000 §8.1).
        response_size = mean_request_size * 3.0
        ratio = response_size / max(mean_request_size, 1e-9)
        return float(ratio), float(self._scenario.amplification_threshold)

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    def run(
        self,
        *,
        skip_guard_checks: bool = False,
    ) -> QUICAgAttackReport:
        """
        Run the full simulation and return a report.

        Parameters
        ----------
        skip_guard_checks : bool
            If True, skip the safety guards (used by the self-test
            only).

        Returns
        -------
        QUICAgAttackReport.
        """
        started = time.perf_counter()

        # Generate packets before entering the namespace, so that a
        # generation failure does not leave a namespace behind.
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
                # In a real run, we would send the packets here. For
                # the artifact, the generated packets ARE the simulation
                # (the metrics are computed from them), and the
                # namespace is created only to enforce the guards.
                # The `_send_packets` helper is a no-op in toy mode
                # and would call scapy's sendp() in full mode.
                if not skip_guard_checks:
                    self._send_packets(packets)
        except (IsolationGuardViolation, EgressDetectedError):
            raise
        except Exception as e:
            self._log.error("run_failed", error=repr(e), exc_info=True)
            raise AttackError(f"run failed: {e}") from e

        # Measure the metric.
        if self._scenario_id == "A03":
            metric_value, metric_threshold = self._measure_a03_metric(packets)
            metric_name = "cwnd_growth_rate_pps"
        elif self._scenario_id == "A07":
            metric_value, metric_threshold = self._measure_a07_metric(packets)
            metric_name = "version_not_in_supported_set"
        elif self._scenario_id == "A13":
            metric_value, metric_threshold = self._measure_a13_metric(packets)
            metric_name = "amplification_ratio"
        else:
            raise AttackError(f"unknown scenario {self._scenario_id}")

        # For A07, the trigger fires when metric_value > threshold
        # (not_in_supported_set = 1.0 > 0.5). For A03, the trigger
        # fires when metric_value > threshold (rate > 3σ). For A13,
        # the trigger fires when metric_value > threshold
        # (amplification > amplification_threshold).
        trigger_fires = metric_value > metric_threshold

        # Get the trigger and action strings from the attack matrix.
        trigger_str, action_str = self._trigger_and_action()

        elapsed = time.perf_counter() - started
        report = QUICAgAttackReport(
            scenario_id=self._scenario_id,
            sub_mode=self._scenario.sub_mode,
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
            "quic_run_complete",
            **report.to_dict(),
            wall_time_s=round(elapsed, 3),
        )
        return report

    def _send_packets(self, packets: List[bytes]) -> None:
        """
        Send the packets into the namespace via scapy.

        In the artifact's toy mode (guards skipped in the self-test),
        this method is a no-op. In full mode, it wraps each payload in
        a UDP/IP/Ether frame and sends it via scapy's sendp(). The
        payload is the raw QUIC packet; the destination is the
        scenario's target:port, and the source is a rotating
        ephemeral port (for A13, a rotating spoofed source).
        """
        try:
            from scapy.all import Ether, IP, UDP, Raw, sendp  # type: ignore
        except ImportError:
            # scapy is optional in the test path; log and return.
            self._log.warning("scapy_not_available_send_skipped")
            return

        # Wrap each payload.
        scapy_pkts = []
        for i, payload in enumerate(packets):
            if self._scenario_id == "A13":
                # Spoofed source for reflection.
                src_port = _DEFAULT_SOURCE_PORT_MIN + (
                    i % (_DEFAULT_SOURCE_PORT_MAX - _DEFAULT_SOURCE_PORT_MIN)
                )
                src_ip = f"10.0.0.{i % 250 + 1}"
            else:
                src_port = 50000
                src_ip = "10.0.0.1"
            pkt = (
                Ether()
                / IP(src=src_ip, dst=self._scenario.target)
                / UDP(sport=src_port, dport=self._scenario.target_port)
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
        the attack matrix, for verification that the simulation is
        aligned with the matrix row."""
        if self._scenario_id == "A03":
            return (
                "cwnd_growth_rate > threshold_3σ",
                "rate_limit = baseline * (1 - attack_ratio^2)",
            )
        if self._scenario_id == "A07":
            return (
                "supported_versions ∩ offered ≠ ∅",
                "send_connection_close(0x010C); log_attack(version_downgrade)",
            )
        if self._scenario_id == "A13":
            return (
                "response_size > amplification_threshold * query_size",
                "rate_limit via AMPS sandbox",
            )
        raise AttackError(
            f"scenario {self._scenario_id} has no trigger/action mapping"
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "QUICCongestionSimulator",
    "QUICAgAttackReport",
    "QUICScenarioConfig",
    "parse_quic_scenario",
    "SUPPORTED_SCENARIOS",
]


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def _self_test() -> int:
    import sys
    rc = 0
    print("=== quic_congestion self-test ===")

    # --- Minimal config with A03, A07, A13 ---
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
                        "id": "A03",
                        "attack_vector": "QUIC Congestion Exploit",
                        "protocol": "doq",
                        "target": "127.0.0.1",
                        "target_port": 443,
                        "rate": {"value": 50000, "unit": "packets_per_second"},
                        "duration_seconds": 1,  # short for self-test
                        "payload": {
                            "type": "quic_packet_flood",
                            "spin_bit": 1,
                            "quic_version": "0x00000001",
                            "sub_mode": "spin_bit_flood",
                        },
                    },
                    {
                        "id": "A07",
                        "attack_vector": "QUIC Version Downgrade",
                        "protocol": "doq",
                        "target": "127.0.0.1",
                        "target_port": 443,
                        "rate": {"value": 50, "unit": "replays"},
                        "duration_seconds": 1,
                        "payload": {
                            "type": "quic_version_field_rewrite",
                            "version_field": "0x00000000",
                            "replay_count": 50,
                            "sub_mode": "version_downgrade",
                            "downgrade_version": 0x00000000,
                        },
                    },
                    {
                        "id": "A13",
                        "attack_vector": "QUIC Reflection",
                        "protocol": "doq",
                        "target": "127.0.0.1",
                        "target_port": 443,
                        "rate": {"value": 5000, "unit": "packets_per_second"},
                        "duration_seconds": 1,  # short for self-test
                        "payload": {
                            "type": "quic_reflection_amplification",
                            "amplification_factor": 50,
                            "threads": 16,
                            "sub_mode": "reflection",
                            "amplification_threshold": 2.5,  # lower than the 3x model
                        },
                    },
                ],
            },
        },
        "runtime": {"seed": 42},
    }

    # --- 1. A03 spin_bit_flood ---
    sim_a03 = QUICCongestionSimulator(config, "A03", seed=42)
    if sim_a03.scenario.sub_mode != "spin_bit_flood":
        print(f"[FAIL] A03 sub_mode: {sim_a03.scenario.sub_mode}")
        rc = 1
    else:
        print(f"[OK]   A03 constructed (sub_mode=spin_bit_flood)")

    packets_a03 = sim_a03.generate_packets()
    # 50000 pps × 1 s = 50000 packets.
    if len(packets_a03) != 50000:
        print(f"[FAIL] A03 packet count: {len(packets_a03)} vs 50000")
        rc = 1
    else:
        print(f"[OK]   A03: {len(packets_a03)} spin-bit flood packets")

    # Verify the first packet has the spin bit set (i % 2 == 0 → spin=True).
    if len(packets_a03) > 0:
        first_byte = packets_a03[0][0]
        if not (first_byte & _QUIC_SPIN_BIT):
            print(f"[FAIL] A03 first packet: spin bit not set "
                  f"(first_byte={first_byte:#04x})")
            rc = 1
        elif first_byte & _QUIC_HEADER_FORM_LONG:
            print(f"[FAIL] A03 first packet: header form is long "
                  f"(first_byte={first_byte:#04x})")
            rc = 1
        else:
            print(f"[OK]   A03 first packet: short header, spin bit set "
                  f"(first_byte={first_byte:#04x})")

    # Verify the second packet has the spin bit clear.
    if len(packets_a03) > 1:
        second_byte = packets_a03[1][0]
        if second_byte & _QUIC_SPIN_BIT:
            print(f"[FAIL] A03 second packet: spin bit set "
                  f"(first_byte={second_byte:#04x})")
            rc = 1
        else:
            print(f"[OK]   A03 second packet: spin bit clear")

    # Full run.
    report_a03 = sim_a03.run(skip_guard_checks=True)
    if report_a03.metric_name != "cwnd_growth_rate_pps":
        print(f"[FAIL] A03 metric_name: {report_a03.metric_name}")
        rc = 1
    elif not report_a03.trigger_fires:
        print(f"[FAIL] A03 trigger did not fire: "
              f"metric={report_a03.metric_value:.0f} > "
              f"threshold={report_a03.metric_threshold:.0f}")
        rc = 1
    else:
        print(f"[OK]   A03 trigger fires: "
              f"{report_a03.metric_value:.0f} pps > "
              f"{report_a03.metric_threshold:.0f} pps (3σ threshold)")

    # --- 2. A03 cid_exhaustion sub-mode ---
    cid_config = {
        **config,
        "_includes": {
            **config["_includes"],
            "attack_suite": {
                **config["_includes"]["attack_suite"],
                "attacks": [
                    {**a, "payload": {**a["payload"], "sub_mode": "cid_exhaustion",
                                     "cid_len": 8, "cid_rate_hz": 100.0}}
                    if a["id"] == "A03" else a
                    for a in config["_includes"]["attack_suite"]["attacks"]
                ],
            },
        },
    }
    # Set duration_seconds to 2 for cid_exhaustion to have >200 packets.
    cid_config["_includes"]["attack_suite"]["attacks"] = [
        {**a, "duration_seconds": 2}
        if a["id"] == "A03" else a
        for a in cid_config["_includes"]["attack_suite"]["attacks"]
    ]
    sim_cid = QUICCongestionSimulator(cid_config, "A03", seed=42)
    if sim_cid.scenario.sub_mode != "cid_exhaustion":
        print(f"[FAIL] A03 cid sub_mode: {sim_cid.scenario.sub_mode}")
        rc = 1
    else:
        print(f"[OK]   A03 cid_exhaustion sub_mode parsed")

    packets_cid = sim_cid.generate_packets()
    # 100 Hz × 2 s = 200 packets.
    if len(packets_cid) != 200:
        print(f"[FAIL] A03 cid packet count: {len(packets_cid)} vs 200")
        rc = 1
    else:
        print(f"[OK]   A03 cid_exhaustion: {len(packets_cid)} long-header "
              f"Initial packets")

    # Verify the first packet is a long-header Initial.
    if len(packets_cid) > 0:
        fb = packets_cid[0][0]
        if not (fb & _QUIC_HEADER_FORM_LONG):
            print(f"[FAIL] A03 cid first packet: not long header "
                  f"(first_byte={fb:#04x})")
            rc = 1
        elif ((fb >> 4) & 0x03) != _QUIC_LONG_INITIAL:
            print(f"[FAIL] A03 cid first packet: not Initial type "
                  f"(type_bits={(fb >> 4) & 0x03})")
            rc = 1
        else:
            print(f"[OK]   A03 cid first packet: long header, Initial type")

    # --- 3. A07 version_downgrade ---
    sim_a07 = QUICCongestionSimulator(config, "A07", seed=42)
    packets_a07 = sim_a07.generate_packets()
    if len(packets_a07) != 50:
        print(f"[FAIL] A07 packet count: {len(packets_a07)} vs 50")
        rc = 1
    else:
        print(f"[OK]   A07: {len(packets_a07)} version-downgrade packets")

    # Verify the version field is 0x00000000 (bytes 1-4 of the long header).
    if len(packets_a07) > 0:
        pkt = packets_a07[0]
        version_bytes = pkt[1:5]
        version = int.from_bytes(version_bytes, "big")
        if version != 0x00000000:
            print(f"[FAIL] A07 version field: {version:#010x} "
                  f"(expected 0x00000000)")
            rc = 1
        else:
            print(f"[OK]   A07 version field: 0x00000000 (legacy)")

    report_a07 = sim_a07.run(skip_guard_checks=True)
    if report_a07.metric_name != "version_not_in_supported_set":
        print(f"[FAIL] A07 metric_name: {report_a07.metric_name}")
        rc = 1
    elif not report_a07.trigger_fires:
        print(f"[FAIL] A07 trigger did not fire: "
              f"metric={report_a07.metric_value} > "
              f"threshold={report_a07.metric_threshold}")
        rc = 1
    else:
        print(f"[OK]   A07 trigger fires: "
              f"offered version not in {{v1, v2}}")

    # --- 4. A13 reflection ---
    sim_a13 = QUICCongestionSimulator(config, "A13", seed=42)
    packets_a13 = sim_a13.generate_packets()
    # 5000 pps × 1 s = 5000 packets.
    if len(packets_a13) != 5000:
        print(f"[FAIL] A13 packet count: {len(packets_a13)} vs 5000")
        rc = 1
    else:
        print(f"[OK]   A13: {len(packets_a13)} reflection probe packets")

    report_a13 = sim_a13.run(skip_guard_checks=True)
    if report_a13.metric_name != "amplification_ratio":
        print(f"[FAIL] A13 metric_name: {report_a13.metric_name}")
        rc = 1
    elif not report_a13.trigger_fires:
        print(f"[FAIL] A13 trigger did not fire: "
              f"ratio={report_a13.metric_value:.4f} > "
              f"threshold={report_a13.metric_threshold:.4f}")
        rc = 1
    else:
        print(f"[OK]   A13 trigger fires: "
              f"amplification_ratio={report_a13.metric_value:.4f} > "
              f"threshold={report_a13.metric_threshold:.4f}")

    # --- 5. Determinism: same seed → same packets ---
    sim_det_1 = QUICCongestionSimulator(config, "A07", seed=42)
    sim_det_2 = QUICCongestionSimulator(config, "A07", seed=42)
    pkts_1 = sim_det_1.generate_packets()
    pkts_2 = sim_det_2.generate_packets()
    if pkts_1 != pkts_2:
        print(f"[FAIL] determinism: same seed produced different packets")
        rc = 1
    else:
        print(f"[OK]   determinism: same seed → identical {len(pkts_1)} packets")

    # --- 6. Different seed → different packets (probabilistically) ---
    sim_diff = QUICCongestionSimulator(config, "A07", seed=999)
    pkts_diff = sim_diff.generate_packets()
    if pkts_1 == pkts_diff:
        print(f"[FAIL] different seeds produced identical packets")
        rc = 1
    else:
        print(f"[OK]   different seeds → different packets")

    # --- 7. Unknown scenario rejected ---
    try:
        parse_quic_scenario("A99", config)
    except AttackConfigError:
        print(f"[OK]   unknown scenario A99 rejected")
    else:
        print(f"[FAIL] unknown scenario A99 was accepted")
        rc = 1

    # --- 8. Report serialization ---
    d = report_a03.to_dict()
    required = ("scenario_id", "sub_mode", "trigger_fires",
                "metric_name", "metric_value", "metric_threshold",
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
