#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tests/test_attacks.py
=====================

E3-DoH Reproducibility — unit tests for the five attack-simulation
modules.

The tests verify each simulator's public contract:

    - Construction with a valid config
    - Rejection of an unknown scenario
    - Rejection of a config that violates the safety guards
    - Packet/payload generation (correct count, well-formed bytes)
    - Determinism (same seed → same packets)
    - Different seeds → different packets
    - Trigger/action strings match data/attack_matrix.csv
    - Metric measurement produces the expected value

The tests that require packet generation skip gracefully when `scapy`
is not installed (the dependency is in requirements.txt tier 3, but a
reviewer running only the test suite without the full install would
hit the missing dependency otherwise).

Run with:
    pytest tests/test_attacks.py -v
"""

from __future__ import annotations

import struct
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from data.data_utils import load_attack_matrix
from src.attacks.dns_tunneling import (
    AttackConfigError,
    DNSTunnelingSimulator,
    EgressDetectedError,
    IsolationGuardViolation,
    SUPPORTED_SCENARIOS as DNS_SCENARIOS,
    parse_scenario as parse_dns_scenario,
)
from src.attacks.timing_side_channel import (
    SUPPORTED_SCENARIOS as TIMING_SCENARIOS,
    TimingSideChannelSimulator,
    parse_timing_scenario,
)
from src.attacks.quic_congestion import (
    QUICCongestionSimulator,
    SUPPORTED_SCENARIOS as QUIC_SCENARIOS,
    parse_quic_scenario,
)
from src.attacks.cert_injection import (
    CertInjectionSimulator,
    SUPPORTED_SCENARIOS as CERT_SCENARIOS,
    parse_cert_scenario,
)
from src.attacks.padding_oracle import (
    PaddingOracleSimulator,
    SUPPORTED_SCENARIOS as PADDING_SCENARIOS,
    parse_padding_scenario,
)


# =============================================================================
# Shared fixtures
# =============================================================================

_GUARDS_OFF = {
    "isolation_required": False,
    "target_must_be_local": True,
    "target_cidr_allowlist": ["127.0.0.0/8", "10.0.0.0/8"],
    "no_privileged_ports": False,
    "no_external_network": False,
}

_RUNTIME = {
    "global_rate_cap_pps": 200_000,
    "global_duration_cap_seconds": 600,
    "capture_pcap": False,
}


def _make_config(**overrides: Any) -> Dict[str, Any]:
    """
    A minimal attack-suite config with all fifteen scenarios, each in
    a small toy-sized form (short duration, small rate) so the tests
    run fast.
    """
    attacks = [
        # A01: DNS tunneling (port 53, the documented exception).
        {"id": "A01", "attack_vector": "DNS Tunneling via DoH",
         "protocol": "doh", "target": "127.0.0.1", "target_port": 53,
         "rate": {"value": 10, "unit": "queries_per_second"},
         "duration_seconds": 1,
         "payload": {"attack_zone": "attack.com",
                     "source_file": "/tmp/test_exfil_A01.bin",
                     "source_size_bytes": 64}},
        # A02: timing side-channel (DoT).
        {"id": "A02", "attack_vector": "Timing Side-Channel",
         "protocol": "dot", "target": "127.0.0.1", "target_port": 853,
         "rate": {"value": 20, "unit": "probes"},
         "duration_seconds": 1,
         "payload": {"mean_delay_ms": 150.0, "std_delay_ms": 25.0}},
        # A03: QUIC congestion.
        {"id": "A03", "attack_vector": "QUIC Congestion Exploit",
         "protocol": "doq", "target": "127.0.0.1", "target_port": 443,
         "rate": {"value": 100, "unit": "packets_per_second"},
         "duration_seconds": 1,
         "payload": {"sub_mode": "spin_bit_flood"}},
        # A04: cert injection.
        {"id": "A04", "attack_vector": "Certificate Injection",
         "protocol": "doh", "target": "127.0.0.1", "target_port": 443,
         "rate": {"value": 5, "unit": "handshakes_per_second"},
         "duration_seconds": 1,
         "payload": {"sni": "client.example.com",
                     "forged_cn": "evil.example.com", "alpn": "h2"}},
        # A05: padding oracle.
        {"id": "A05", "attack_vector": "Padding Oracle",
         "protocol": "dot", "target": "127.0.0.1", "target_port": 853,
         "rate": {"value": 20, "unit": "requests_per_second"},
         "duration_seconds": 1,
         "payload": {"cbc_ciphertext_blocks": 3}},
        # A06: slow-drip exfiltration.
        {"id": "A06", "attack_vector": "Slow-Drip Exfiltration",
         "protocol": "doh", "target": "127.0.0.1", "target_port": 443,
         "rate": {"value": 1, "unit": "queries_per_second"},
         "duration_seconds": 2,
         "payload": {"inter_query_delay_s": [1.0, 4.0],
                     "source_file": "/tmp/test_exfil_A06.bin",
                     "source_size_bytes": 64,
                     "attack_zone": "attack.com"}},
        # A07: version downgrade.
        {"id": "A07", "attack_vector": "QUIC Version Downgrade",
         "protocol": "doq", "target": "127.0.0.1", "target_port": 443,
         "rate": {"value": 10, "unit": "replays"},
         "duration_seconds": 1,
         "payload": {"sub_mode": "version_downgrade",
                     "downgrade_version": 0}},
        # A08: HTTP/2 frame storm.
        {"id": "A08", "attack_vector": "HTTP/2 Frame Storm",
         "protocol": "doh", "target": "127.0.0.1", "target_port": 443,
         "rate": {"value": 100, "unit": "frames_per_second"},
         "duration_seconds": 1,
         "payload": {"sub_mode": "frame_storm"}},
        # A09: ECH poisoning.
        {"id": "A09", "attack_vector": "Encrypted SNI Poisoning",
         "protocol": "doh", "target": "127.0.0.1", "target_port": 443,
         "rate": {"value": 5, "unit": "injections_per_second"},
         "duration_seconds": 1,
         "payload": {"injected_sni": "malicious.example.com",
                     "sub_mode": "sni_poisoning"}},
        # A10: adaptive chaff.
        {"id": "A10", "attack_vector": "Adaptive Chaff Injection",
         "protocol": "doh", "target": "127.0.0.1", "target_port": 443,
         "rate": {"value": 20, "unit": "queries_per_second"},
         "duration_seconds": 1,
         "payload": {"sub_mode": "adaptive_chaff",
                     "attack_zone": "attack.com"}},
        # A11: query flood.
        {"id": "A11", "attack_vector": "DNS Query Flood",
         "protocol": "doh", "target": "127.0.0.1", "target_port": 443,
         "rate": {"value": 100, "unit": "queries_per_second"},
         "duration_seconds": 1,
         "payload": {"clients": 10, "attack_zone": "attack.com"}},
        # A12: handshake abuse.
        {"id": "A12", "attack_vector": "TLS Handshake Abuse",
         "protocol": "dot", "target": "127.0.0.1", "target_port": 853,
         "rate": {"value": 10, "unit": "handshakes_per_second"},
         "duration_seconds": 1,
         "payload": {"sub_mode": "handshake_abuse"}},
        # A13: reflection.
        {"id": "A13", "attack_vector": "QUIC Reflection",
         "protocol": "doq", "target": "127.0.0.1", "target_port": 443,
         "rate": {"value": 50, "unit": "packets_per_second"},
         "duration_seconds": 1,
         "payload": {"sub_mode": "reflection"}},
        # A14: continuation flood.
        {"id": "A14", "attack_vector": "HTTP/2 Continuation Flood",
         "protocol": "doh", "target": "127.0.0.1", "target_port": 443,
         "rate": {"value": 100, "unit": "requests"},
         "duration_seconds": 1,
         "payload": {"max_concurrent_streams": 100}},
        # A15: payload fuzzing.
        {"id": "A15", "attack_vector": "Encrypted Payload Fuzzing",
         "protocol": "doh", "target": "127.0.0.1", "target_port": 443,
         "rate": {"value": 20, "unit": "fuzzed_requests_per_second"},
         "duration_seconds": 1,
         "payload": {"mutation_rate": 0.5}},
    ]
    cfg = {
        "_includes": {
            "attack_suite": {
                "safety_guards": dict(_GUARDS_OFF),
                "runtime": dict(_RUNTIME),
                "attacks": attacks,
            },
        },
        "runtime": {"seed": 42},
    }
    for key, value in overrides.items():
        # Support dotted overrides like "safety_guards.isolation_required".
        parts = key.split(".")
        cursor = cfg
        for p in parts[:-1]:
            cursor = cursor.setdefault(p, {})
        cursor[parts[-1]] = value
    return cfg


def _require_scapy() -> None:
    try:
        import scapy  # noqa: F401
    except ImportError:
        pytest.skip("scapy not installed; skipping packet-generation tests")


# =============================================================================
# Scenario-set coverage
# =============================================================================

class TestScenarioCoverage:
    def test_all_fifteen_scenarios_covered(self):
        all_supported = (
            set(DNS_SCENARIOS) | set(TIMING_SCENARIOS) | set(QUIC_SCENARIOS)
            | set(CERT_SCENARIOS) | set(PADDING_SCENARIOS)
        )
        expected = {f"A{i:02d}" for i in range(1, 16)}
        assert all_supported == expected

    def test_no_module_overlaps(self):
        # Each scenario should be handled by exactly one module.
        from collections import Counter
        counts = Counter()
        for s in (DNS_SCENARIOS, TIMING_SCENARIOS, QUIC_SCENARIOS,
                  CERT_SCENARIOS, PADDING_SCENARIOS):
            counts.update(s)
        duplicates = [s for s, n in counts.items() if n > 1]
        assert not duplicates, f"scenarios handled by multiple modules: {duplicates}"


# =============================================================================
# Attack matrix consistency
# =============================================================================

class TestAttackMatrixConsistency:
    """
    Every simulator's `_trigger_and_action()` should return a pair that
    matches the corresponding row in data/attack_matrix.csv.
    """

    @pytest.mark.parametrize("scenario_id", sorted(
        DNS_SCENARIOS + TIMING_SCENARIOS + QUIC_SCENARIOS
        + CERT_SCENARIOS + PADDING_SCENARIOS
    ))
    def test_trigger_and_action_matches_matrix(self, scenario_id):
        # Load the matrix.
        try:
            matrix = load_attack_matrix()
        except Exception as e:
            pytest.skip(f"attack matrix not loadable: {e}")
        row = matrix[matrix["Attack_ID"] == scenario_id].iloc[0]
        expected_trigger = row["Detection_Trigger"]
        expected_action = row["Mitigation_Action"]

        # Build the simulator and read its trigger/action.
        config = _make_config()
        if scenario_id in DNS_SCENARIOS:
            sim = DNSTunnelingSimulator(config, scenario_id, seed=42)
        elif scenario_id in TIMING_SCENARIOS:
            sim = TimingSideChannelSimulator(config, scenario_id, seed=42)
        elif scenario_id in QUIC_SCENARIOS:
            sim = QUICCongestionSimulator(config, scenario_id, seed=42)
        elif scenario_id in CERT_SCENARIOS:
            sim = CertInjectionSimulator(config, scenario_id, seed=42)
        elif scenario_id in PADDING_SCENARIOS:
            sim = PaddingOracleSimulator(config, scenario_id, seed=42)
        else:
            pytest.skip(f"scenario {scenario_id} not covered")

        trigger, action = sim._trigger_and_action()
        assert trigger == expected_trigger, (
            f"{scenario_id}: trigger mismatch\n"
            f"  simulator: {trigger!r}\n"
            f"  matrix:    {expected_trigger!r}"
        )
        assert action == expected_action, (
            f"{scenario_id}: action mismatch\n"
            f"  simulator: {action!r}\n"
            f"  matrix:    {expected_action!r}"
        )


# =============================================================================
# Scenario parsing
# =============================================================================

class TestScenarioParsing:
    def test_parse_valid(self):
        config = _make_config()
        for sid in DNS_SCENARIOS:
            sc = parse_dns_scenario(sid, config)
            assert sc.scenario_id == sid

    def test_unknown_scenario_rejected(self):
        config = _make_config()
        with pytest.raises(AttackConfigError):
            parse_dns_scenario("A99", config)

    def test_invalid_duration_rejected(self):
        config = _make_config()
        config["_includes"]["attack_suite"]["attacks"][0]["duration_seconds"] = 0
        with pytest.raises(AttackConfigError):
            parse_dns_scenario("A01", config)


# =============================================================================
# Safety guards
# =============================================================================

class TestSafetyGuards:
    def test_target_outside_allowlist_rejected(self):
        # Change A01's target to a public address.
        config = _make_config()
        config["_includes"]["attack_suite"]["attacks"][0]["target"] = "8.8.8.8"
        # Enable the guard.
        config["_includes"]["attack_suite"]["safety_guards"]["target_must_be_local"] = True
        from src.attacks.dns_tunneling import validate_safety_guards
        sim = DNSTunnelingSimulator(config, "A01", seed=42)
        with pytest.raises(IsolationGuardViolation):
            validate_safety_guards(sim.scenario, sim._guards, sim._log)

    def test_privileged_port_rejected_for_a06(self):
        config = _make_config()
        config["_includes"]["attack_suite"]["safety_guards"]["no_privileged_ports"] = True
        config["_includes"]["attack_suite"]["safety_guards"]["min_port"] = 1024
        config["_includes"]["attack_suite"]["attacks"][5]["target_port"] = 53  # A06
        from src.attacks.dns_tunneling import validate_safety_guards
        sim = DNSTunnelingSimulator(config, "A06", seed=42)
        with pytest.raises(IsolationGuardViolation):
            validate_safety_guards(sim.scenario, sim._guards, sim._log)

    def test_privileged_port_allowed_for_a01(self):
        # A01's port 53 is the documented DNS port exception.
        config = _make_config()
        config["_includes"]["attack_suite"]["safety_guards"]["no_privileged_ports"] = True
        config["_includes"]["attack_suite"]["safety_guards"]["min_port"] = 1024
        from src.attacks.dns_tunneling import validate_safety_guards
        sim = DNSTunnelingSimulator(config, "A01", seed=42)
        # Should NOT raise.
        validate_safety_guards(sim.scenario, sim._guards, sim._log)


# =============================================================================
# Per-module tests
# =============================================================================

class TestDNSTunneling:
    def test_construct(self):
        sim = DNSTunnelingSimulator(_make_config(), "A01", seed=42)
        assert sim.scenario.scenario_id == "A01"

    def test_unknown_scenario_rejected(self):
        with pytest.raises(AttackConfigError):
            DNSTunnelingSimulator(_make_config(), "A02", seed=42)  # wrong module

    def test_packet_generation(self):
        _require_scapy()
        sim = DNSTunnelingSimulator(_make_config(), "A01", seed=42)
        packets = sim.generate_packets()
        # 10 qps × 1s = 10 packets.
        assert len(packets) == 10
        # Each packet should be a valid DNS query (starts with 0x1234 ID).
        first = bytes(packets[0].payload)
        assert len(first) >= 12
        dns_id = struct.unpack("!H", first[:2])[0]
        assert dns_id == 0x1234

    def test_determinism(self):
        _require_scapy()
        sim1 = DNSTunnelingSimulator(_make_config(), "A01", seed=42)
        sim2 = DNSTunnelingSimulator(_make_config(), "A01", seed=42)
        p1 = sim1.generate_packets()
        p2 = sim2.generate_packets()
        # Same seed → same packet count and same first packet bytes.
        assert len(p1) == len(p2)
        assert bytes(p1[0].payload) == bytes(p2[0].payload)

    def test_different_seeds_differ(self):
        _require_scapy()
        sim1 = DNSTunnelingSimulator(_make_config(), "A01", seed=42)
        sim2 = DNSTunnelingSimulator(_make_config(), "A01", seed=999)
        p1 = sim1.generate_packets()
        p2 = sim2.generate_packets()
        # The DNS ID and the base64 subdomain differ.
        assert bytes(p1[0].payload) != bytes(p2[0].payload)

    def test_full_run_skipping_guards(self):
        _require_scapy()
        sim = DNSTunnelingSimulator(_make_config(), "A01", seed=42)
        report = sim.run(skip_guard_checks=True)
        assert report["scenario_id"] == "A01"
        assert report["n_packets_sent"] == 10


class TestTimingSideChannel:
    def test_construct(self):
        sim = TimingSideChannelSimulator(_make_config(), "A02", seed=42)
        assert sim.scenario.scenario_id == "A02"

    def test_query_times_monotonic(self):
        sim = TimingSideChannelSimulator(_make_config(), "A02", seed=42)
        times = sim.generate_query_times()
        for i in range(len(times) - 1):
            assert times[i + 1] > times[i]

    def test_response_times_have_delay(self):
        sim = TimingSideChannelSimulator(_make_config(), "A02", seed=42)
        q = sim.generate_query_times()
        r = sim.simulate_response_times(q)
        delays_ms = [(rj - qj) * 1000 for qj, rj in zip(q, r)]
        assert 140 < np.mean(delays_ms) < 160  # ~150 ms ± noise

    def test_run_produces_report(self):
        sim = TimingSideChannelSimulator(_make_config(), "A02", seed=42)
        report = sim.run(skip_guard_checks=True)
        assert report.scenario_id == "A02"
        assert report.n_probes == 20

    def test_determinism(self):
        sim1 = TimingSideChannelSimulator(_make_config(), "A02", seed=42)
        sim2 = TimingSideChannelSimulator(_make_config(), "A02", seed=42)
        r1 = sim1.run(skip_guard_checks=True)
        r2 = sim2.run(skip_guard_checks=True)
        assert abs(r1.interval_corr - r2.interval_corr) < 1e-9


class TestQUICCongestion:
    def test_construct(self):
        sim = QUICCongestionSimulator(_make_config(), "A03", seed=42)
        assert sim.scenario.scenario_id == "A03"

    def test_spin_bit_flood_packet_count(self):
        sim = QUICCongestionSimulator(_make_config(), "A03", seed=42)
        packets = sim.generate_packets()
        assert len(packets) == 100

    def test_first_packet_has_spin_bit(self):
        sim = QUICCongestionSimulator(_make_config(), "A03", seed=42)
        packets = sim.generate_packets()
        first_byte = packets[0][0]
        # Spin bit is 0x20; header form (bit 7) should be 0 for short header.
        assert first_byte & 0x80 == 0  # short header
        assert first_byte & 0x20 != 0  # spin bit set on even packets

    def test_version_downgrade_version_field(self):
        sim = QUICCongestionSimulator(_make_config(), "A07", seed=42)
        packets = sim.generate_packets()
        # Version is bytes 1-4 of the long-header packet.
        version = struct.unpack("!I", packets[0][1:5])[0]
        assert version == 0  # the downgrade value

    def test_determinism(self):
        sim1 = QUICCongestionSimulator(_make_config(), "A07", seed=42)
        sim2 = QUICCongestionSimulator(_make_config(), "A07", seed=42)
        assert sim1.generate_packets() == sim2.generate_packets()


class TestCertInjection:
    def test_construct(self):
        sim = CertInjectionSimulator(_make_config(), "A04", seed=42)
        assert sim.scenario.scenario_id == "A04"

    def test_certificate_has_handshake_record_type(self):
        sim = CertInjectionSimulator(_make_config(), "A04", seed=42)
        packets = sim.generate_packets()
        # Each packet is a ClientHello + Certificate; the first byte is
        # the TLS handshake content type 0x16.
        assert packets[0][0] == 0x16

    def test_a08_frames_have_valid_h2_header(self):
        sim = CertInjectionSimulator(_make_config(), "A08", seed=42)
        packets = sim.generate_packets()
        # Each frame is 9-byte H2 header + payload. The length field is
        # the first 3 bytes.
        p = packets[0]
        length = int.from_bytes(p[:3], "big")
        assert len(p) == 9 + length

    def test_a14_continuation_frames_have_right_type(self):
        sim = CertInjectionSimulator(_make_config(), "A14", seed=42)
        packets = sim.generate_packets()
        # Frame type is byte 3 (after the 3-byte length); should be
        # 0x09 (CONTINUATION).
        assert packets[0][3] == 0x09

    def test_trigger_fires_for_a04(self):
        sim = CertInjectionSimulator(_make_config(), "A04", seed=42)
        report = sim.run(skip_guard_checks=True)
        assert report.trigger_fires is True
        assert report.metric_name == "sni_cert_cn_mismatch"


class TestPaddingOracle:
    def test_construct(self):
        sim = PaddingOracleSimulator(_make_config(), "A05", seed=42)
        assert sim.scenario.scenario_id == "A05"

    def test_padding_oracle_content_type(self):
        sim = PaddingOracleSimulator(_make_config(), "A05", seed=42)
        packets = sim.generate_packets()
        # Each record is TLS application-data (content_type 0x17).
        assert packets[0][0] == 0x17

    def test_padding_candidates_cycle(self):
        sim = PaddingOracleSimulator(_make_config(), "A05", seed=42)
        packets = sim.generate_packets()
        # The padding candidate cycles through 0x00, 0x01, ..., 0xFF.
        # Extract the target byte (offset = header(5) + (n_blocks-2+1)*16 - 1).
        target_off = 5 + (3 - 2 + 1) * 16 - 1
        candidates = [p[target_off] for p in packets[:5]]
        # Should be 0, 1, 2, 3, 4.
        assert candidates == [0, 1, 2, 3, 4]

    def test_trigger_fires_for_a05(self):
        sim = PaddingOracleSimulator(_make_config(), "A05", seed=42)
        report = sim.run(skip_guard_checks=True)
        assert report.trigger_fires is True

    def test_a15_mutation_rate(self):
        # With mutation_rate=0.5, the fraction of mutated records is
        # approximately 0.5.
        sim = PaddingOracleSimulator(_make_config(), "A15", seed=42)
        report = sim.run(skip_guard_checks=True)
        assert 0.2 < report.metric_value < 0.8


# =============================================================================
# Determinism across modules
# =============================================================================

class TestCrossModuleDeterminism:
    def test_same_seed_same_packets_each_module(self):
        _require_scapy()
        config = _make_config()
        for sid, ctor in [
            ("A01", DNSTunnelingSimulator),
            ("A03", QUICCongestionSimulator),
            ("A05", PaddingOracleSimulator),
            ("A08", CertInjectionSimulator),
        ]:
            s1 = ctor(config, sid, seed=42)
            s2 = ctor(config, sid, seed=42)
            assert s1.generate_packets() == s2.generate_packets(), (
                f"{sid}: non-deterministic across same-seed runs"
            )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
