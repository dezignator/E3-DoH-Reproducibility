#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tests/test_protocols.py
=======================

E3-DoH Reproducibility — unit tests for the protocol adapters.

These tests verify the three concrete adapters (DoH, DoT, DoQ) against
the abstract base class contract, and against each other where the
contract requires consistency (the twelve-feature schema, the sandbox
action registry, the metric-tensor dispatch).

Run with:
    pytest tests/test_protocols.py -v

The tests cover:
    - Base class invariants (feature split, sandbox registry)
    - Construction of each adapter with a valid config
    - Rejection of invalid config (wrong transport, wrong TLS version,
      wrong QUIC version, unknown protocol)
    - Extraction of the three protocol-specific features
    - Assembly of the twelve-feature vector (order + key set)
    - The nine standard features (analytical checks on a synthetic
      flow)
    - Metric-tensor dispatch: returns a SPD matrix for each protocol
    - Sandbox action dispatch: correct action per protocol, idempotence
    - Cross-protocol action rejection
    - The adapter factory returns the correct class
    - Consistency between the online and offline feature extractors
      (adapter vs data/preprocess_pcap.py helpers)
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from data.data_utils import FEATURE_COLUMNS, META_COLUMNS, PROTOCOLS
from src.protocols.base_adapter import (
    AdapterConfigError,
    FeatureExtractionError,
    FlowState,
    PROTOCOL_FEATURES,
    ProtocolAdapter,
    ProtocolNotSupportedError,
    SANDBOX_ACTIONS,
    SandboxActionError,
    STANDARD_FEATURES,
    compute_standard_features,
    get_adapter,
)
from src.protocols.doh_adapter import DoHAdapter
from src.protocols.dot_adapter import DoTAdapter
from src.protocols.doq_adapter import DoQAdapter


# =============================================================================
# Shared fixtures
# =============================================================================

def _make_metrics_include() -> Dict[str, Any]:
    """
    A minimal `riemannian_metrics` include that satisfies the adapter's
    config validation. The values match configs/riemannian_metrics.yaml
    §3-§5 but are self-contained so the tests do not depend on the file.
    """
    return {
        "doh": {"lambda": 0.15, "alpha": 0.85},
        "dot": {"beta": 0.22, "gamma": 0.65},
        "doq": {
            "eta": 0.35,
            "congestion_operator": {
                "diagonal": {name: 0.5 for name in FEATURE_COLUMNS},
                "off_diagonal": [],
            },
        },
    }


def _make_config(protocol: str = "doh", **protocol_kwargs: Any) -> Dict[str, Any]:
    """A config dict that the adapters accept."""
    protocols_block = {protocol: protocol_kwargs} if protocol_kwargs else {protocol: {}}
    return {
        "protocols": protocols_block,
        "_includes": {"riemannian_metrics": _make_metrics_include()},
        "runtime": {"seed": 42},
    }


def _make_flow(protocol: str, n_queries: int = 3, n_responses: int = 2) -> FlowState:
    """A synthetic FlowState with the given number of packets."""
    flow = FlowState(
        flow_id=f"test-{protocol}-flow",
        protocol=protocol,
        client_ip="10.0.0.1", client_port=50000,
        server_ip="10.0.0.2",
        server_port={"DoH": 443, "DoT": 853, "DoQ": 853}[protocol],
    )
    for i in range(n_queries):
        flow.add_query(0.1 * i, 60 + 10 * i)
    for i in range(n_responses):
        flow.add_response(0.1 * i + 0.005, 120 + 10 * i)
    # Adapter-specific protocol_data.
    if protocol == "DoH":
        flow.protocol_data = {
            "tls_record_types": [0x16, 0x17, 0x17],
            "first_payload": b"\x16\x03\x03\x00\x05" + b"\x00" * 100,
            "n_data_records": 2,
        }
    elif protocol == "DoT":
        flow.protocol_data = {
            "tls_record_types": [0x16, 0x17, 0x17, 0x17],
            "first_payload": b"\x16\x03\x03\x00\x05" + b"\x00" * 100,
            "n_data_records": 3,
        }
    elif protocol == "DoQ":
        flow.protocol_data = {
            "quic_packet_types": [0, 3, 0, 0, 0],
            "first_payload": b"\xc0" + b"\x00" * 200,
            "n_short_header_packets": 3,
        }
    return flow


# =============================================================================
# Base class invariants
# =============================================================================

class TestBaseInvariants:
    def test_feature_split_matches_canonical(self):
        assert set(STANDARD_FEATURES) | set(PROTOCOL_FEATURES) == set(FEATURE_COLUMNS)

    def test_feature_split_disjoint(self):
        assert set(STANDARD_FEATURES).isdisjoint(set(PROTOCOL_FEATURES))

    def test_sandbox_registry_has_three_protocols(self):
        assert set(SANDBOX_ACTIONS.keys()) == set(PROTOCOLS)

    def test_sandbox_action_names(self):
        assert SANDBOX_ACTIONS["doh"] == "isolate_doh_stream"
        assert SANDBOX_ACTIONS["dot"] == "reset_dot_tls"
        assert SANDBOX_ACTIONS["doq"] == "migrate_doq_connection"


# =============================================================================
# compute_standard_features — analytical checks
# =============================================================================

class TestComputeStandardFeatures:
    def test_known_flow(self):
        # Three queries at t=0, 0.1, 0.3 with sizes 60, 80, 100.
        # IATs: 0.1, 0.2. Mean = 0.15. Var = 0.0025. Std = 0.05.
        # Query sizes: mean 80, std sqrt(((60-80)^2 + 0 + (100-80)^2)/3)
        # = sqrt(800/3) ≈ 16.33.
        flow = FlowState(
            flow_id="x", protocol="DoH",
            client_ip="10.0.0.1", client_port=50000,
            server_ip="10.0.0.2", server_port=443,
        )
        flow.add_query(0.0, 60)
        flow.add_query(0.1, 80)
        flow.add_query(0.3, 100)
        flow.add_response(0.005, 120)
        flow.add_response(0.105, 140)
        features = compute_standard_features(flow)
        assert features["inter_arrival_time_mean"] == pytest.approx(0.15, abs=1e-9)
        assert features["inter_arrival_time_std"] == pytest.approx(0.05, abs=1e-9)
        assert features["query_size_mean"] == pytest.approx(80.0, abs=1e-9)
        assert features["query_size_std"] == pytest.approx(math.sqrt(800/3), abs=1e-6)
        assert features["response_size_mean"] == pytest.approx(130.0, abs=1e-9)
        assert features["session_duration"] == pytest.approx(0.3, abs=1e-9)

    def test_burstiness_zero_when_mean_zero(self):
        # Two queries at the same instant: IAT mean = 0, burstiness = 0.
        flow = FlowState(
            flow_id="x", protocol="DoH",
            client_ip="a", client_port=1, server_ip="b", server_port=2,
        )
        flow.add_query(0.0, 60)
        flow.add_query(0.0, 60)
        features = compute_standard_features(flow)
        assert features["burstiness"] == 0.0

    def test_no_queries_raises(self):
        flow = FlowState(
            flow_id="x", protocol="DoH",
            client_ip="a", client_port=1, server_ip="b", server_port=2,
        )
        with pytest.raises(FeatureExtractionError):
            compute_standard_features(flow)

    def test_no_responses_gives_zero_ratio(self):
        flow = FlowState(
            flow_id="x", protocol="DoH",
            client_ip="a", client_port=1, server_ip="b", server_port=2,
        )
        flow.add_query(0.0, 60)
        features = compute_standard_features(flow)
        assert features["response_size_mean"] == 0.0
        assert features["query_response_ratio"] == 0.0


# =============================================================================
# DoHAdapter
# =============================================================================

class TestDoHAdapter:
    def test_construct(self):
        adapter = DoHAdapter(_make_config("doh", transport="http2"))
        assert adapter.protocol == "doh"

    def test_wrong_transport_rejected(self):
        with pytest.raises(FeatureExtractionError):
            DoHAdapter(_make_config("doh", transport="http3"))

    def test_default_transport_is_http2(self):
        # No transport key: the adapter defaults to http2.
        adapter = DoHAdapter(_make_config("doh"))
        assert adapter._transport == "http2"

    def test_extract_protocol_features(self):
        adapter = DoHAdapter(_make_config("doh"))
        flow = _make_flow("DoH")
        features = adapter.extract_protocol_specific_features(flow)
        assert set(features.keys()) == set(PROTOCOL_FEATURES)
        # 2 app_data out of 3 records → 2/3 ≈ 0.667.
        assert features["protocol_specific_feature"] == pytest.approx(2/3, abs=1e-9)

    def test_full_feature_vector_order(self):
        adapter = DoHAdapter(_make_config("doh"))
        flow = _make_flow("DoH")
        features = adapter.extract_features(flow)
        assert list(features.keys()) == list(FEATURE_COLUMNS)

    def test_sandbox_action(self):
        adapter = DoHAdapter(_make_config("doh"))
        flow = _make_flow("DoH")
        result = adapter.apply_sandbox_action(flow, "isolate_doh_stream")
        assert result.success is True
        assert result.action == "isolate_doh_stream"
        assert result.protocol == "doh"

    def test_double_action_is_idempotent(self):
        adapter = DoHAdapter(_make_config("doh"))
        flow = _make_flow("DoH")
        r1 = adapter.apply_sandbox_action(flow, "isolate_doh_stream")
        r2 = adapter.apply_sandbox_action(flow, "isolate_doh_stream")
        assert r1.success is True
        assert r2.success is False  # already terminated

    def test_cross_protocol_action_rejected(self):
        adapter = DoHAdapter(_make_config("doh"))
        flow = _make_flow("DoH")
        with pytest.raises(SandboxActionError):
            adapter.apply_sandbox_action(flow, "reset_dot_tls")

    def test_metric_tensor_spd(self):
        adapter = DoHAdapter(_make_config("doh"))
        rng = np.random.default_rng(0)
        x = rng.normal(size=len(FEATURE_COLUMNS))
        g = adapter.get_metric_tensor(x)
        assert g.shape == (len(FEATURE_COLUMNS), len(FEATURE_COLUMNS))
        assert np.allclose(g, g.T)
        assert np.linalg.eigvalsh(g).min() > 0

    def test_metric_tensor_wrong_shape_rejected(self):
        adapter = DoHAdapter(_make_config("doh"))
        with pytest.raises(FeatureExtractionError):
            adapter.get_metric_tensor(np.zeros(5))


# =============================================================================
# DoTAdapter
# =============================================================================

class TestDoTAdapter:
    def test_construct(self):
        adapter = DoTAdapter(_make_config("dot", tls_min_version="1.3"))
        assert adapter.protocol == "dot"

    def test_wrong_tls_version_rejected(self):
        with pytest.raises(FeatureExtractionError):
            DoTAdapter(_make_config("dot", tls_min_version="1.2"))

    def test_extract_protocol_features(self):
        adapter = DoTAdapter(_make_config("dot"))
        flow = _make_flow("DoT")
        features = adapter.extract_protocol_specific_features(flow)
        # 3 app_data out of 4 records.
        assert features["protocol_specific_feature"] == pytest.approx(0.75, abs=1e-9)

    def test_sandbox_action(self):
        adapter = DoTAdapter(_make_config("dot"))
        flow = _make_flow("DoT")
        result = adapter.apply_sandbox_action(flow, "reset_dot_tls")
        assert result.success is True
        assert result.action == "reset_dot_tls"
        assert result.protocol == "dot"

    def test_cross_protocol_action_rejected(self):
        adapter = DoTAdapter(_make_config("dot"))
        flow = _make_flow("DoT")
        with pytest.raises(SandboxActionError):
            adapter.apply_sandbox_action(flow, "isolate_doh_stream")

    def test_metric_tensor_spd(self):
        adapter = DoTAdapter(_make_config("dot"))
        rng = np.random.default_rng(0)
        g = adapter.get_metric_tensor(rng.normal(size=len(FEATURE_COLUMNS)))
        assert np.linalg.eigvalsh(g).min() > 0

    def test_metric_kwargs_at_baseline(self):
        # At the DoT baseline, the kwargs should be (1.0, 1.0).
        from src.protocols.dot_adapter import _DOT_BASELINE
        adapter = DoTAdapter(_make_config("dot"))
        kwargs = adapter._metric_tensor_kwargs(np.array(_DOT_BASELINE))
        assert kwargs["x_bandwidth"] == pytest.approx(1.0, abs=1e-9)
        assert kwargs["rtt"] == pytest.approx(1.0, abs=1e-9)


# =============================================================================
# DoQAdapter
# =============================================================================

class TestDoQAdapter:
    def test_construct(self):
        adapter = DoQAdapter(_make_config("doq", quic_version=1))
        assert adapter.protocol == "doq"

    def test_wrong_quic_version_rejected(self):
        with pytest.raises(FeatureExtractionError):
            DoQAdapter(_make_config("doq", quic_version=2))

    def test_extract_protocol_features(self):
        adapter = DoQAdapter(_make_config("doq"))
        flow = _make_flow("DoQ")
        features = adapter.extract_protocol_specific_features(flow)
        # 3 short-header (category 0) out of 5 packets.
        assert features["protocol_specific_feature"] == pytest.approx(3/5, abs=1e-9)

    def test_sandbox_action(self):
        adapter = DoQAdapter(_make_config("doq"))
        flow = _make_flow("DoQ")
        result = adapter.apply_sandbox_action(flow, "migrate_doq_connection")
        assert result.success is True
        assert result.action == "migrate_doq_connection"
        assert result.protocol == "doq"
        # Detail should mention the session being preserved.
        assert "preserved" in result.detail.lower()

    def test_double_action_increments_migration_count(self):
        adapter = DoQAdapter(_make_config("doq"))
        flow = _make_flow("DoQ")
        adapter.apply_sandbox_action(flow, "migrate_doq_connection")
        r2 = adapter.apply_sandbox_action(flow, "migrate_doq_connection")
        assert r2.success is False  # already migrated
        assert flow.protocol_data["sandbox_migration_count"] == 2

    def test_metric_tensor_spd(self):
        adapter = DoQAdapter(_make_config("doq"))
        rng = np.random.default_rng(0)
        g = adapter.get_metric_tensor(rng.normal(size=len(FEATURE_COLUMNS)))
        assert np.linalg.eigvalsh(g).min() > 0


# =============================================================================
# Factory
# =============================================================================

class TestGetAdapter:
    def test_doh(self):
        a = get_adapter("doh", _make_config("doh"))
        assert isinstance(a, DoHAdapter)

    def test_dot(self):
        a = get_adapter("dot", _make_config("dot"))
        assert isinstance(a, DoTAdapter)

    def test_doq(self):
        a = get_adapter("doq", _make_config("doq"))
        assert isinstance(a, DoQAdapter)

    def test_unknown_protocol(self):
        with pytest.raises(ProtocolNotSupportedError):
            get_adapter("foo", _make_config("doh"))

    def test_factory_returns_new_instance(self):
        a1 = get_adapter("doh", _make_config("doh"))
        a2 = get_adapter("doh", _make_config("doh"))
        assert a1 is not a2


# =============================================================================
# Cross-adapter consistency
# =============================================================================

class TestCrossAdapter:
    @pytest.mark.parametrize("protocol,adapter_cls", [
        ("doh", DoHAdapter),
        ("dot", DoTAdapter),
        ("doq", DoQAdapter),
    ])
    def test_all_adapters_produce_twelve_features(self, protocol, adapter_cls):
        adapter = adapter_cls(_make_config(protocol))
        flow = _make_flow({"doh": "DoH", "dot": "DoT", "doq": "DoQ"}[protocol])
        features = adapter.extract_features(flow)
        assert len(features) == len(FEATURE_COLUMNS)
        assert list(features.keys()) == list(FEATURE_COLUMNS)

    @pytest.mark.parametrize("protocol,adapter_cls", [
        ("doh", DoHAdapter),
        ("dot", DoTAdapter),
        ("doq", DoQAdapter),
    ])
    def test_all_adapters_have_distinct_sandbox_action(self, protocol, adapter_cls):
        adapter = adapter_cls(_make_config(protocol))
        assert adapter.get_sandbox_action() == SANDBOX_ACTIONS[protocol]

    @pytest.mark.parametrize("protocol,adapter_cls", [
        ("doh", DoHAdapter),
        ("dot", DoTAdapter),
        ("doq", DoQAdapter),
    ])
    def test_all_adapters_reject_other_action(self, protocol, adapter_cls):
        adapter = adapter_cls(_make_config(protocol))
        flow = _make_flow({"doh": "DoH", "dot": "DoT", "doq": "DoQ"}[protocol])
        for other_protocol, other_action in SANDBOX_ACTIONS.items():
            if other_protocol != protocol:
                with pytest.raises(SandboxActionError):
                    adapter.apply_sandbox_action(flow, other_action)

    @pytest.mark.parametrize("protocol,adapter_cls", [
        ("doh", DoHAdapter),
        ("dot", DoTAdapter),
        ("doq", DoQAdapter),
    ])
    def test_to_feature_row_has_all_columns(self, protocol, adapter_cls):
        adapter = adapter_cls(_make_config(protocol))
        flow = _make_flow({"doh": "DoH", "dot": "DoT", "doq": "DoQ"}[protocol])
        row = adapter.to_feature_row(flow)
        for col in FEATURE_COLUMNS + META_COLUMNS:
            assert col in row


# =============================================================================
# Consistency with data/preprocess_pcap.py
# =============================================================================

class TestOfflineConsistency:
    """
    The adapters and the offline preprocessor (data/preprocess_pcap.py)
    implement the same twelve features. A drift between them is a
    defect; these tests catch it.

    The test compares the two *helper* functions (Shannon entropy, L1
    distance) on identical inputs. The full feature extraction is
    harder to compare because the offline path returns one row per
    flow and the online path returns one dict per flow, but the helpers
    are the shared code that matters.
    """

    def test_shannon_entropy_bytes_agrees(self):
        try:
            from data.preprocess_pcap import _shannon_entropy_bytes as offline
        except ImportError:
            pytest.skip("data.preprocess_pcap not importable")
        from src.protocols.doh_adapter import _shannon_entropy_bytes as online
        rng = np.random.default_rng(0)
        for _ in range(5):
            data = bytes(rng.integers(0, 256, size=200, dtype=np.uint8))
            assert online(data) == pytest.approx(offline(data), abs=1e-12)

    def test_l1_distance_agrees(self):
        try:
            from data.preprocess_pcap import _l1_distance_uniform as offline
        except ImportError:
            pytest.skip("data.preprocess_pcap not importable")
        from src.protocols.doh_adapter import _l1_distance_uniform as online
        rng = np.random.default_rng(1)
        for _ in range(5):
            values = [int(x) for x in rng.integers(0, 4, size=20)]
            assert online(values, 4) == pytest.approx(offline(values, 4), abs=1e-12)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
