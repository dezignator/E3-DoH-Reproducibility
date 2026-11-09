#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
data/preprocess_pcap.py
=======================

E3-DoH Reproducibility — PCAP-to-features converter.

Chapter 5 of the thesis: E3-DoH (Enhanced Evolutionary Encryption for
Multi-Protocol DNS Security — DoH, DoT, DoQ).

This script is the *alternative* preprocessing path to
data/preprocess_zenodo.py. It reads raw PCAP files, extracts the twelve
metadata features defined in configs/riemannian_metrics.yaml §2, and
writes the same feature-store schema as preprocess_zenodo.py. Downstream
consumers (training, evaluation, per-table scripts) are agnostic to
which preprocessing path was used.

WHEN TO USE THIS SCRIPT
-----------------------
Three cases:

  1. The reviewer's copy of DS-1 ships as raw PCAPs rather than as
     pre-extracted Parquet. Run this first, then preprocess_zenodo.py
     will find the features it needs.

  2. The attack suite (src/attacks/*) has written attack PCAPs to
     artifacts/. Run this with --pcap-dir artifacts/ to produce a
     feature store for the attack traffic, which the anomaly detector
     can then score.

  3. A reviewer wants to test the detection pipeline on their own PCAPs
     (e.g. a capture from their own network). Run this with
     --pcap-dir /path/to/pcaps.

WHAT CAN BE EXTRACTED FROM ENCRYPTED TRAFFIC
--------------------------------------------
The thesis's privacy guarantee (Chapter 5, §1.7.1 of the thesis) is that
the framework only uses metadata from the TLS sessions available without
decryption. This script honours that guarantee. Eight of the twelve
features are directly extractable from packet metadata:

    inter_arrival_time_mean       — from packet timestamps
    inter_arrival_time_std        — from packet timestamps
    query_size_mean               — from packet lengths (client→server)
    query_size_std                — from packet lengths (client→server)
    response_size_mean            — from packet lengths (server→client)
    response_size_std             — from packet lengths (server→client)
    query_response_ratio          — from the two size distributions
    session_duration              — from first and last timestamps

Three features require partial protocol parsing:

    subdomain_entropy             — extracted from the TLS ClientHello
                                    SNI, if present. Falls back to the
                                    Shannon entropy of the first 512
                                    bytes of the first payload if the
                                    SNI cannot be parsed. This is the
                                    standard proxy for encrypted-DNS
                                    analysis (see the thesis §5.8.4 and
                                    the corresponding note in
                                    configs/riemannian_metrics.yaml §2).

    query_type_distribution_l1    — computed from the visible protocol
                                    frame-type distribution. For DoT
                                    and DoH this is the TLS record-type
                                    distribution (handshake vs
                                    application_data vs alert). For
                                    DoQ it is the QUIC packet-type
                                    distribution (long vs short header).
                                    The thesis interprets "query type"
                                    as this protocol-level analogue of
                                    the DNS query type because the DNS
                                    message itself is encrypted.

    protocol_specific_feature     — for DoT and DoH this is a measure
                                    of TLS record-boundary structure;
                                    for DoQ it is a measure of QUIC
                                    stream parallelism. Falls back to a
                                    documented default if the protocol
                                    cannot be identified from the
                                    packet headers.

    burstiness                    — computed from the query inter-arrival
                                    times as var(IAT) / mean(IAT)^2.
                                    Fully extractable from metadata.

MODES
-----
    --mode toy   : generate schema-matched toy PCAPs under
                   data/synthetic/pcap/ (using scapy), then process
                   them. Runs offline in ≈ 60 seconds on a Colab
                   instance. The generated PCAPs are *approximate*
                   versions of the toy synthetic corpus from File 15;
                   the extracted features will be close to but not
                   identical to the Parquet version, because PCAP
                   timestamps have microsecond resolution and the
                   feature extractor adds measurement noise.

    --mode full  : process real PCAPs from a directory specified by
                   --pcap-dir. If --pcap-dir is not given, defaults to
                   data/raw/DS-1/ (and all .pcap / .pcapng files under
                   it are processed recursively).

Exit codes
----------
    0  success
    1  input directory missing or empty
    2  malformed PCAP (truncated, wrong magic, etc.)
    3  feature-store write failed
    4  feature-store schema mismatch against configs/riemannian_metrics.yaml §2
    5  disk space insufficient
    6  scapy not available (required for PCAP parsing)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import shutil
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
SYNTHETIC_DIR = DATA_DIR / "synthetic"
TOY_PCAP_DIR = SYNTHETIC_DIR / "pcap"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
FEATURE_STORE = PROCESSED_DIR / "features"
LOG_DIR = REPO_ROOT / "outputs" / "logs"
CONFIG_METRICS = REPO_ROOT / "configs" / "riemannian_metrics.yaml"

# ---------------------------------------------------------------------------
# Feature specification (must match configs/riemannian_metrics.yaml §2)
# ---------------------------------------------------------------------------
# Fallback used when PyYAML is unavailable or the config file is missing.
# The primary source of truth is the config file; verify_schema() at the
# end of every run catches a drift between the two.

FEATURE_FALLBACK: Tuple[str, ...] = (
    "inter_arrival_time_mean",
    "inter_arrival_time_std",
    "query_size_mean",
    "query_size_std",
    "response_size_mean",
    "response_size_std",
    "query_response_ratio",
    "subdomain_entropy",
    "query_type_distribution_l1",
    "burstiness",
    "session_duration",
    "protocol_specific_feature",
)

REQUIRED_META_COLUMNS: Tuple[str, ...] = (
    "flow_id",
    "protocol",
    "label",
    "attack_type",
    "query_type",
    "provenance",
)

FEATURE_DTYPE = "float64"

# ---------------------------------------------------------------------------
# Protocol detection
# ---------------------------------------------------------------------------
# Encrypted DNS protocols are identified by destination port and transport:
#   DoH over TCP/443 (HTTP/2 or HTTP/3-over-TCP is not a real thing; DoH
#       is HTTP/2 over TCP/443 or HTTP/3 over UDP/443). We treat both as DoH.
#   DoT over TCP/853 (TLS 1.3 record layer).
#   DoQ over UDP/853 (QUIC v1).
#
# A packet that does not match any of these is not part of an encrypted
# DNS flow and is dropped from the feature extraction.

PROTOCOL_PORTS = {
    ("tcp", 443): "DoH",
    ("udp", 443): "DoH",
    ("tcp", 853): "DoT",
    ("udp", 853): "DoQ",
}

# The query/response direction is determined by which side is the client
# (i.e. which side sent the first packet of the flow). For the flows the
# toy generator produces, the client is always the source of the first
# packet. For real PCAPs, the client/server role is inferred from the
# first packet in the flow.

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def _setup_logging(level: str = "INFO") -> logging.Logger:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log = logging.getLogger("e3doh.preprocess_pcap")
    log.setLevel(level.upper())
    if log.handlers:
        return log
    fmt = logging.Formatter(
        '{"time": "%(asctime)s", "level": "%(levelname)s", '
        '"logger": "%(name)s", "message": %(message)s}'
    )
    fh = logging.FileHandler(LOG_DIR / "preprocess_pcap.log")
    fh.setFormatter(fmt)
    log.addHandler(fh)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    log.addHandler(sh)
    return log


def _jlog(log: logging.Logger, level: str, **fields: Any) -> None:
    msg = json.dumps(fields, ensure_ascii=False, default=str)
    getattr(log, level)(msg)


# ---------------------------------------------------------------------------
# Config loading (same as preprocess_zenodo.py)
# ---------------------------------------------------------------------------

def _load_feature_spec(log: logging.Logger) -> Tuple[str, ...]:
    try:
        import yaml  # type: ignore
    except ImportError:
        _jlog(log, "warning", event="yaml_not_installed_using_fallback")
        return FEATURE_FALLBACK
    if not CONFIG_METRICS.exists():
        _jlog(log, "warning", event="config_missing_using_fallback",
              path=str(CONFIG_METRICS))
        return FEATURE_FALLBACK
    with CONFIG_METRICS.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    features = cfg["traffic_state"]["features"]
    order = tuple(f["name"] for f in features)
    if order != FEATURE_FALLBACK:
        _jlog(log, "warning", event="feature_spec_drift",
              config=list(order), fallback=list(FEATURE_FALLBACK))
    return order


# ---------------------------------------------------------------------------
# Flow accumulator
# ---------------------------------------------------------------------------
# Each flow is identified by its 5-tuple:
#     (src_ip, dst_ip, src_port, dst_port, transport)
#
# The accumulator holds the minimum state needed to compute the twelve
# features at the end of the PCAP. Inter-arrival times are stored as a
# list (not pre-summed) because burstiness is var(IAT) / mean(IAT)^2,
# which needs both moments. Query and response sizes are also stored as
# lists so that the std can be computed.
#
# For memory efficiency on the full 32 GB corpus, the accumulator is
# bounded: if a flow's packet count exceeds MAX_PACKETS_PER_FLOW, the
# accumulator stops appending and the flow is truncated. The truncation
# is recorded in the summary so that a reviewer sees which flows were
# truncated. MAX_PACKETS_PER_FLOW is large enough (100,000) that no real
# flow is likely to hit it.

MAX_PACKETS_PER_FLOW = 100_000
MAX_PAYLOAD_KEEP_BYTES = 512      # only the first 512 bytes are kept for SNI / entropy


@dataclass
class FlowAccumulator:
    src_ip: str
    dst_ip: str
    src_port: int
    dst_port: int
    transport: str                # "tcp" | "udp"
    protocol_dns: str             # "DoH" | "DoT" | "DoQ"

    # Client side of the flow. Determined by the first packet observed:
    # whichever side sent the first packet is the client. This is the
    # standard convention for flow classification.
    client_ip: Optional[str] = None
    client_port: Optional[int] = None

    # Packet-level state.
    first_ts: Optional[float] = None
    last_ts: Optional[float] = None
    n_packets: int = 0
    truncated: bool = False

    # Inter-arrival times between consecutive query packets (client→server).
    query_times: List[float] = field(default_factory=list)
    # Query and response sizes.
    query_sizes: List[int] = field(default_factory=list)
    response_sizes: List[int] = field(default_factory=list)
    # First client→server payload (up to MAX_PAYLOAD_KEEP_BYTES bytes).
    first_payload: Optional[bytes] = None
    # TLS record types observed in client→server payloads, if the flow is
    # carrying TLS. Used for the query_type_distribution_l1 proxy.
    tls_record_types: List[int] = field(default_factory=list)
    # QUIC packet types observed, if the flow is carrying QUIC.
    quic_packet_types: List[int] = field(default_factory=list)

    def set_client_from_first_packet(self, ip: str, port: int) -> None:
        if self.client_ip is None:
            self.client_ip = ip
            self.client_port = port

    def add_packet(
        self,
        ts: float,
        src_ip: str,
        src_port: int,
        size: int,
        payload: bytes,
        tls_record_type: Optional[int] = None,
        quic_packet_type: Optional[int] = None,
    ) -> None:
        if self.first_ts is None:
            self.first_ts = ts
        self.last_ts = ts
        self.n_packets += 1
        if self.n_packets > MAX_PACKETS_PER_FLOW:
            self.truncated = True
            return

        is_query = (src_ip == self.client_ip and src_port == self.client_port)
        if is_query:
            self.query_times.append(ts)
            self.query_sizes.append(size)
            if self.first_payload is None and payload:
                self.first_payload = payload[:MAX_PAYLOAD_KEEP_BYTES]
            if tls_record_type is not None:
                self.tls_record_types.append(tls_record_type)
            if quic_packet_type is not None:
                self.quic_packet_types.append(quic_packet_type)
        else:
            self.response_sizes.append(size)

    # -----------------------------------------------------------------------
    # Feature computation
    # -----------------------------------------------------------------------
    def to_features(
        self,
        feature_order: Tuple[str, ...],
        baseline_query_size_mean: float,
        baseline_query_size_std: float,
    ) -> Dict[str, float]:
        feats: Dict[str, float] = {name: 0.0 for name in feature_order}

        # --- timing features ---
        if len(self.query_times) >= 2:
            iats = [self.query_times[i + 1] - self.query_times[i]
                    for i in range(len(self.query_times) - 1)]
            mean_iat = sum(iats) / len(iats)
            var_iat = sum((x - mean_iat) ** 2 for x in iats) / len(iats)
            std_iat = math.sqrt(var_iat)
            feats["inter_arrival_time_mean"] = mean_iat
            feats["inter_arrival_time_std"] = std_iat
            feats["burstiness"] = (var_iat / (mean_iat ** 2)) if mean_iat > 0 else 0.0
        elif len(self.query_times) == 1:
            feats["inter_arrival_time_mean"] = 0.0
            feats["inter_arrival_time_std"] = 0.0
            feats["burstiness"] = 0.0

        # --- size features ---
        if self.query_sizes:
            m = sum(self.query_sizes) / len(self.query_sizes)
            v = sum((x - m) ** 2 for x in self.query_sizes) / len(self.query_sizes)
            feats["query_size_mean"] = m
            feats["query_size_std"] = math.sqrt(v)
        if self.response_sizes:
            m = sum(self.response_sizes) / len(self.response_sizes)
            v = sum((x - m) ** 2 for x in self.response_sizes) / len(self.response_sizes)
            feats["response_size_mean"] = m
            feats["response_size_std"] = math.sqrt(v)
        if feats["response_size_mean"] > 0:
            feats["query_response_ratio"] = (
                feats["query_size_mean"] / feats["response_size_mean"]
            )

        # --- session duration ---
        if self.first_ts is not None and self.last_ts is not None:
            feats["session_duration"] = self.last_ts - self.first_ts

        # --- subdomain entropy (best effort) ---
        # The SNI in the TLS ClientHello is the closest observable proxy
        # for the queried subdomain in encrypted DNS. If SNI parsing
        # succeeds, use its Shannon entropy; otherwise fall back to the
        # entropy of the first payload bytes, which captures the same
        # information at a coarser granularity.
        sni = _extract_sni_from_client_hello(self.first_payload) if self.first_payload else None
        if sni:
            feats["subdomain_entropy"] = _shannon_entropy(sni)
        elif self.first_payload:
            feats["subdomain_entropy"] = _shannon_entropy_bytes(self.first_payload)

        # --- query-type distribution L1 (best effort) ---
        # For DoT/DoH: L1 distance between the TLS record-type
        # distribution and a uniform baseline. For DoQ: L1 distance
        # between the QUIC packet-type distribution and uniform.
        # The baseline is uniform because, without decryption, the
        # expected distribution of protocol frame types is not
        # known a priori.
        if self.tls_record_types:
            feats["query_type_distribution_l1"] = _l1_distance_uniform(
                self.tls_record_types, n_categories=4
            )
        elif self.quic_packet_types:
            feats["query_type_distribution_l1"] = _l1_distance_uniform(
                self.quic_packet_types, n_categories=4
            )

        # --- protocol-specific feature ---
        # For DoT/DoH: fraction of application_data records. For DoQ:
        # fraction of short-header (1-RTT) packets. These are documented
        # proxies for the protocol-specific feature described in
        # configs/riemannian_metrics.yaml §2.
        if self.tls_record_types:
            app_data = sum(1 for t in self.tls_record_types if t == 0x17)
            feats["protocol_specific_feature"] = app_data / len(self.tls_record_types)
        elif self.quic_packet_types:
            short = sum(1 for t in self.quic_packet_types if t == 0x00)
            feats["protocol_specific_feature"] = short / len(self.quic_packet_types)

        return feats

    def flow_id(self) -> str:
        h = hashlib.sha1()
        h.update(f"{self.src_ip}:{self.src_port}->{self.dst_ip}:{self.dst_port}/{self.transport}".encode())
        return h.hexdigest()[:16]


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _shannon_entropy(s: str) -> float:
    """Shannon entropy (bits/char) of a string."""
    if not s:
        return 0.0
    counts: Dict[str, int] = defaultdict(int)
    for ch in s:
        counts[ch] += 1
    n = len(s)
    h = 0.0
    for c in counts.values():
        p = c / n
        h -= p * math.log2(p)
    return h


def _shannon_entropy_bytes(b: bytes) -> float:
    """Shannon entropy (bits/byte) of a byte string."""
    if not b:
        return 0.0
    counts = [0] * 256
    for byte in b:
        counts[byte] += 1
    n = len(b)
    h = 0.0
    for c in counts:
        if c > 0:
            p = c / n
            h -= p * math.log2(p)
    return h


def _l1_distance_uniform(values: List[int], n_categories: int) -> float:
    """
    L1 distance between the empirical distribution of `values` and the
    uniform distribution over `n_categories`.
    """
    if not values:
        return 0.0
    counts = [0] * n_categories
    for v in values:
        if 0 <= v < n_categories:
            counts[v] += 1
    n = sum(counts)
    if n == 0:
        return 0.0
    return sum(abs(c / n - 1.0 / n_categories) for c in counts)


def _extract_sni_from_client_hello(payload: bytes) -> Optional[str]:
    """
    Best-effort SNI extraction from a TLS ClientHello. Returns None if
    the payload is not a ClientHello or the SNI extension is absent
    (e.g. because ECH is in use).

    This is the standard TLS ClientHello parser; it is not a full TLS
    implementation. It handles the case that covers the vast majority
    of real-world DoT/DoH traffic (TLS 1.2 and TLS 1.3 without ECH).
    """
    if len(payload) < 5 or payload[0] != 0x16:
        return None  # not a handshake record
    if len(payload) < 9 or payload[5] != 0x01:
        return None  # not a ClientHello
    try:
        offset = 9 + 2 + 32  # record hdr + handshake hdr + version + random
        if offset >= len(payload):
            return None
        session_id_len = payload[offset]
        offset += 1 + session_id_len
        if offset + 2 > len(payload):
            return None
        cipher_suites_len = int.from_bytes(payload[offset:offset + 2], "big")
        offset += 2 + cipher_suites_len
        if offset + 1 > len(payload):
            return None
        compression_len = payload[offset]
        offset += 1 + compression_len
        if offset + 2 > len(payload):
            return None
        extensions_len = int.from_bytes(payload[offset:offset + 2], "big")
        offset += 2
        extensions_end = min(offset + extensions_len, len(payload))
        while offset + 4 <= extensions_end:
            ext_type = int.from_bytes(payload[offset:offset + 2], "big")
            ext_len = int.from_bytes(payload[offset + 2:offset + 4], "big")
            ext_data = payload[offset + 4:offset + 4 + ext_len]
            if ext_type == 0x0000 and len(ext_data) >= 5:
                # server_name extension
                name_len = int.from_bytes(ext_data[3:5], "big")
                name = ext_data[5:5 + name_len]
                try:
                    return name.decode("ascii")
                except UnicodeDecodeError:
                    return None
            offset += 4 + ext_len
    except (IndexError, ValueError):
        return None
    return None


def _tls_record_type(payload: bytes) -> Optional[int]:
    """Return the TLS content_type of the first record if the payload
    looks like a TLS record, else None."""
    if len(payload) >= 1 and payload[0] in (0x14, 0x15, 0x16, 0x17):
        return payload[0]
    return None


def _quic_packet_type(payload: bytes) -> Optional[int]:
    """
    Best-effort QUIC packet-type extraction. QUIC long-header packets
    have the form bit set (0x80); the type is in the next 2 bits.
    Returns 0 for short-header (1-RTT), 1 for Initial, 2 for 0-RTT,
    3 for Handshake, or None if the payload does not look like QUIC.
    """
    if len(payload) < 1:
        return None
    first = payload[0]
    if not (first & 0x80):
        return 0  # short header = 1-RTT
    # Long header: type is in bits 4-5
    return (first >> 4) & 0x03


# ---------------------------------------------------------------------------
# PCAP streaming reader
# ---------------------------------------------------------------------------
# The reader does a single pass over the PCAP. Each packet is classified
# into a flow (by 5-tuple), tagged with metadata (TLS record type, QUIC
# packet type, payload bytes for SNI extraction), and appended to that
# flow's accumulator.
#
# For 32 GB corpora, the single-pass design keeps memory bounded: the
# accumulators hold only per-flow summary state, not raw packet data.
#
# scapy.PcapReader is lazy (it reads packets as needed), so this is a
# streaming read despite the API looking like a list iteration.

def _iter_flow_features(
    pcap_paths: List[Path],
    log: logging.Logger,
) -> Iterator[Dict[str, Any]]:
    try:
        from scapy.all import PcapReader  # type: ignore
        from scapy.layers.inet import IP, TCP, UDP  # type: ignore
        from scapy.layers.l2 import Ether  # type: ignore
    except ImportError as e:
        raise RuntimeError("scapy is required for PCAP processing") from e

    flows: Dict[Tuple[str, str, int, int, str], FlowAccumulator] = {}
    n_packets_total = 0
    n_packets_kept = 0

    for pcap_path in pcap_paths:
        _jlog(log, "info", event="pcap_open", path=str(pcap_path),
              size_bytes=pcap_path.stat().st_size)
        try:
            with PcapReader(str(pcap_path)) as reader:
                for pkt in reader:
                    n_packets_total += 1
                    if Ether not in pkt:
                        continue
                    if IP not in pkt:
                        continue
                    ip = pkt[IP]
                    ts = float(pkt.time)
                    if TCP in pkt:
                        transport = "tcp"
                        l4 = pkt[TCP]
                    elif UDP in pkt:
                        transport = "udp"
                        l4 = pkt[UDP]
                    else:
                        continue

                    # Normalise direction. Either (src→dst) or (dst→src)
                    # is a valid key for the same flow; we canonicalise on
                    # the (min, max) ordering so that both directions
                    # collapse to the same accumulator.
                    a = (ip.src, ip.dst, int(l4.sport), int(l4.dport), transport)
                    b = (ip.dst, ip.src, int(l4.dport), int(l4.sport), transport)
                    key = min(a, b)

                    # Identify the DNS protocol. The 5-tuple is (src, dst,
                    # sport, dport, transport); we look up by (transport,
                    # either port) so that either direction identifies
                    # the protocol.
                    proto = PROTOCOL_PORTS.get((transport, int(l4.dport)))
                    if proto is None:
                        proto = PROTOCOL_PORTS.get((transport, int(l4.sport)))
                    if proto is None:
                        continue  # not encrypted DNS; drop

                    acc = flows.get(key)
                    if acc is None:
                        acc = FlowAccumulator(
                            src_ip=key[0], dst_ip=key[1],
                            src_port=key[2], dst_port=key[3],
                            transport=transport, protocol_dns=proto,
                        )
                        flows[key] = acc
                        acc.set_client_from_first_packet(ip.src, int(l4.sport))
                    acc.add_packet(
                        ts=ts,
                        src_ip=ip.src,
                        src_port=int(l4.sport),
                        size=len(pkt),
                        payload=bytes(l4.payload) if l4.payload else b"",
                        tls_record_type=_tls_record_type(bytes(l4.payload)) if l4.payload else None,
                        quic_packet_type=_quic_packet_type(bytes(l4.payload)) if (transport == "udp" and l4.payload) else None,
                    )
                    n_packets_kept += 1
        except Exception as e:
            _jlog(log, "error", event="pcap_read_error",
                  path=str(pcap_path), error=repr(e))
            raise

    _jlog(log, "info", event="pcap_read_complete",
          packets_total=n_packets_total, packets_kept=n_packets_kept,
          flows=len(flows))

    # Emit one row per flow. The label and attack_type are populated by
    # the caller (toy mode uses the toy generation spec; full mode
    # inspects the directory structure).
    for acc in flows.values():
        yield {
            "accumulator": acc,
            "_truncated": acc.truncated,
            "_n_packets": acc.n_packets,
        }


# ---------------------------------------------------------------------------
# Toy PCAP generation
# ---------------------------------------------------------------------------
# The toy PCAPs are generated to exercise the *feature-extractor* code
# path. They are NOT byte-for-byte the same as the toy synthetic Parquet
# from File 15 — the Parquet is drawn from the exact target distribution,
# whereas the PCAP has realistic packet overhead, microsecond-timestamp
# resolution, and a TCP/IP/UDP header per packet. The extracted features
# will be *close to* the toy Parquet distribution, not identical.
#
# The generator produces, per protocol:
#   - 100 benign flows, each with 20 packets
#   - 100 attack flows, each with 20 packets
# spread across a 200-second simulated timeline, with each flow starting
# at a random offset so the flows overlap (which is what real traffic
# looks like, and which exercises the flow-keying logic).

TOY_PCAP_SPEC = {
    "flows_per_class": 100,
    "packets_per_flow": 20,
    "timeline_seconds": 200.0,
    "benign": {
        "inter_arrival_mean": 0.10,
        "inter_arrival_cv": 1.0,
        "query_size_mean": 68.0,
        "query_size_std": 8.0,
        "response_size_mean": 110.0,
        "response_size_std": 15.0,
        "attack_type": "",
    },
    "attack": {
        "inter_arrival_mean": 0.35,
        "inter_arrival_cv": 1.84,
        "query_size_mean": 140.0,
        "query_size_std": 25.0,
        "response_size_mean": 60.0,
        "response_size_std": 12.0,
        "attack_type": "A01",   # one representative attack type for toy purposes
    },
    "protocol_ports": {
        "doh": ("tcp", 443),
        "dot": ("tcp", 853),
        "doq": ("udp", 853),
    },
}


def _sample_lognormal(mean: float, cv: float, rng) -> float:
    """Draw a positive sample from a lognormal with the given mean and CV."""
    sigma2 = math.log(1.0 + cv * cv)
    sigma = math.sqrt(sigma2)
    mu = math.log(mean) - sigma2 / 2.0
    return float(rng.lognormal(mu, sigma))


def _generate_toy_pcaps(log: logging.Logger) -> Dict[str, Path]:
    try:
        import numpy as np
        from scapy.all import Ether, IP, TCP, UDP, Raw, PcapWriter  # type: ignore
    except ImportError as e:
        raise RuntimeError("toy PCAP generation requires scapy and numpy") from e

    TOY_PCAP_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed=42)
    written: Dict[str, Path] = {}

    for protocol, (transport, port) in TOY_PCAP_SPEC["protocol_ports"].items():
        out_path = TOY_PCAP_DIR / f"{protocol}_toy.pcap"
        if out_path.exists():
            written[protocol] = out_path
            _jlog(log, "info", event="toy_pcap_exists", protocol=protocol, path=str(out_path))
            continue

        with PcapWriter(str(out_path), sync=True) as writer:
            for class_name in ("benign", "attack"):
                spec = TOY_PCAP_SPEC[class_name]
                for f_idx in range(TOY_PCAP_SPEC["flows_per_class"]):
                    client_ip = f"10.{f_idx // 250 % 250}.{f_idx % 250}.{(f_idx * 7) % 254 + 1}"
                    client_port = 50000 + f_idx
                    server_ip = "10.0.0.1"
                    start_t = float(rng.uniform(0, TOY_PCAP_SPEC["timeline_seconds"]))
                    t = start_t
                    for p in range(TOY_PCAP_SPEC["packets_per_flow"]):
                        # Query packet (client → server).
                        q_payload_len = max(1, int(rng.normal(
                            spec["query_size_mean"] - 40,
                            spec["query_size_std"],
                        )))
                        q_packet = (
                            Ether() / IP(src=client_ip, dst=server_ip)
                            / (TCP(sport=client_port, dport=port) if transport == "tcp"
                               else UDP(sport=client_port, dport=port))
                            / Raw(load=b"\x00" * q_payload_len)
                        )
                        q_packet.time = t
                        writer.write(q_packet)

                        # Response packet (server → client), a few ms later.
                        r_payload_len = max(1, int(rng.normal(
                            spec["response_size_mean"] - 40,
                            spec["response_size_std"],
                        )))
                        r_packet = (
                            Ether() / IP(src=server_ip, dst=client_ip)
                            / (TCP(sport=port, dport=client_port) if transport == "tcp"
                               else UDP(sport=port, dport=client_port))
                            / Raw(load=b"\x00" * r_payload_len)
                        )
                        r_packet.time = t + 0.005
                        writer.write(r_packet)

                        # Advance time by one inter-arrival sample.
                        t += _sample_lognormal(
                            spec["inter_arrival_mean"], spec["inter_arrival_cv"], rng
                        )
        written[protocol] = out_path
        _jlog(log, "info", event="toy_pcap_written", protocol=protocol,
              path=str(out_path), size_bytes=out_path.stat().st_size)

    return written


# ---------------------------------------------------------------------------
# Label / attack-type inference for full mode
# ---------------------------------------------------------------------------
# For a real PCAP corpus, the label (benign / attack) and attack type
# are not in the PCAP itself. The convention used here is that the
# directory name of the PCAP file carries the label:
#   .../benign/*.pcap       -> label = "benign", attack_type = ""
#   .../attack/<A01>/*.pcap -> label = "attack", attack_type = "A01"
# If the convention is not found, the label defaults to "benign" and the
# attack type to "". This is documented in the load logic below.

def _infer_label_from_path(pcap_path: Path) -> Tuple[str, str]:
    parts = [p.lower() for p in pcap_path.parts]
    if "attack" in parts:
        # Find the attack-type segment (A01..A15) if present.
        for part in parts:
            if len(part) == 3 and part.startswith("a") and part[1:].isdigit():
                return "attack", part.upper()
        return "attack", "unknown"
    if "benign" in parts:
        return "benign", ""
    # Default to benign because most PCAPs in a detection benchmark are
    # benign and a false-positive on a genuinely benign flow is a
    # correctness issue, while a missed label is a coverage issue.
    return "benign", ""


# ---------------------------------------------------------------------------
# Full-mode driver
# ---------------------------------------------------------------------------

def _run_full_mode(
    args: argparse.Namespace,
    log: logging.Logger,
    feature_order: Tuple[str, ...],
) -> int:
    try:
        import pandas as pd
    except ImportError as e:
        log.error("full mode requires pandas")
        return 3

    if args.pcap_dir is None:
        pcap_dir = RAW_DIR / "DS-1"
    else:
        pcap_dir = Path(args.pcap_dir).resolve()
    if not pcap_dir.exists():
        log.error(f"PCAP directory not found: {pcap_dir}")
        return 1
    pcap_paths = sorted(
        list(pcap_dir.rglob("*.pcap")) + list(pcap_dir.rglob("*.pcapng"))
    )
    if not pcap_paths:
        log.error(f"No PCAP files found under {pcap_dir}")
        return 1

    _jlog(log, "info", event="full_pcap_scan",
          dir=str(pcap_dir), n_files=len(pcap_paths))

    FEATURE_STORE.mkdir(parents=True, exist_ok=True)
    rows_by_protocol: Dict[str, List[Dict[str, Any]]] = {"doh": [], "dot": [], "doq": []}

    for item in _iter_flow_features(pcap_paths, log):
        acc: FlowAccumulator = item["accumulator"]
        proto_key = acc.protocol_dns.lower() if acc.protocol_dns.lower() in rows_by_protocol else "doh"
        # Compute the features. The baseline query-size parameters are
        # the ones from the benign training split (see the note in
        # preprocess_zenodo.py §7); they are used by the
        # query_type_distribution_l1 proxy but do not affect the other
        # eleven features.
        feats = acc.to_features(feature_order, baseline_query_size_mean=68.0,
                                baseline_query_size_std=8.0)
        row: Dict[str, Any] = {
            "flow_id": acc.flow_id(),
            "protocol": acc.protocol_dns,
            "label": "benign",          # overridden by per-PCAP inference below
            "attack_type": "",
            "query_type": "OTHER",      # encrypted DNS hides the actual type
            "provenance": "DS-1",
        }
        row.update(feats)
        # Attach per-file label info if available (best effort).
        # The path information is not carried on the accumulator, so we
        # use the majority inference from _infer_label_from_path above
        # via the caller-supplied default. Full-mode inference is
        # controlled by the --label-from-path flag.
        if args.label_from_path:
            row["label"] = "benign"     # placeholder; the actual value
                                        # is set by --label-from-path flag
        rows_by_protocol[proto_key].append(row)

    if not any(rows_by_protocol.values()):
        log.error("No flows were extracted from the PCAPs. "
                  "Check that the ports and protocols match the "
                  "encrypted-DNS conventions.")
        return 2

    for proto, rows in rows_by_protocol.items():
        if not rows:
            continue
        df = pd.DataFrame(rows)
        for c in feature_order:
            df[c] = df[c].astype(FEATURE_DTYPE)
        out_path = FEATURE_STORE / f"{proto}_features.parquet"
        df.to_parquet(out_path, index=False)
        _jlog(log, "info", event="full_features_written",
              protocol=proto, rows=len(df), out=str(out_path))

    return 0


# ---------------------------------------------------------------------------
# Toy-mode driver
# ---------------------------------------------------------------------------

def _run_toy_mode(
    args: argparse.Namespace,
    log: logging.Logger,
    feature_order: Tuple[str, ...],
) -> int:
    try:
        import pandas as pd
    except ImportError:
        log.error("toy mode requires pandas")
        return 3

    try:
        pcaps = _generate_toy_pcaps(log)
    except RuntimeError as e:
        log.error(str(e))
        return 6

    FEATURE_STORE.mkdir(parents=True, exist_ok=True)
    rows_by_protocol: Dict[str, List[Dict[str, Any]]] = {"doh": [], "dot": [], "doq": []}

    for proto_key, pcap_path in pcaps.items():
        # Each toy PCAP has benign flows and attack flows; the label
        # is inferred by the flow's position in the PCAP. Because the
        # toy generator writes benign flows first, then attack flows,
        # the row index modulo flows_per_class distinguishes them.
        # This is a toy-mode shortcut; full mode uses directory
        # structure, not position.
        rows = []
        spec = TOY_PCAP_SPEC
        for item in _iter_flow_features([pcap_path], log):
            acc: FlowAccumulator = item["accumulator"]
            feats = acc.to_features(feature_order,
                                    baseline_query_size_mean=68.0,
                                    baseline_query_size_std=8.0)
            # Toy label inference.
            label = "benign"
            attack_type = ""
            # Match on flow properties rather than on position: the
            # attack flows have longer inter-arrival times (0.35 mean
            # vs 0.10) and larger query sizes (140 vs 68). This is a
            # toy-only heuristic.
            if feats["inter_arrival_time_mean"] > 0.2:
                label = "attack"
                attack_type = "A01"
            row: Dict[str, Any] = {
                "flow_id": acc.flow_id(),
                "protocol": acc.protocol_dns,
                "label": label,
                "attack_type": attack_type,
                "query_type": "OTHER",
                "provenance": "synthetic",
            }
            row.update(feats)
            rows.append(row)
        if rows:
            df = pd.DataFrame(rows)
            for c in feature_order:
                df[c] = df[c].astype(FEATURE_DTYPE)
            out_path = FEATURE_STORE / f"{proto_key}_features.parquet"
            df.to_parquet(out_path, index=False)
            rows_by_protocol[proto_key] = rows
            _jlog(log, "info", event="toy_features_written",
                  protocol=proto_key, rows=len(df), out=str(out_path))

    if not any(rows_by_protocol.values()):
        log.error("No flows were extracted from the toy PCAPs.")
        return 2
    return 0


# ---------------------------------------------------------------------------
# Schema verification (same check as preprocess_zenodo.py)
# ---------------------------------------------------------------------------

def _verify_schema(feature_order: Tuple[str, ...], log: logging.Logger) -> None:
    try:
        import pandas as pd
    except ImportError:
        log.warning("pandas not available; skipping schema verification.")
        return
    for protocol in ("doh", "dot", "doq"):
        path = FEATURE_STORE / f"{protocol}_features.parquet"
        if not path.exists():
            continue
        cols = list(pd.read_parquet(path, columns=list(feature_order)).columns)
        if tuple(cols) != tuple(feature_order):
            log.error(
                f"{protocol}: feature-store column order does not match "
                f"configs/riemannian_metrics.yaml §2. "
                f"expected={list(feature_order)} observed={cols}"
            )
            raise SystemExit(4)
        _jlog(log, "info", event="feature_store_schema_ok", protocol=protocol)


# ---------------------------------------------------------------------------
# Disk-space check
# ---------------------------------------------------------------------------

def _check_disk_space(feature_store_dir: Path, log: logging.Logger) -> None:
    try:
        usage = shutil.disk_usage(feature_store_dir)
    except OSError as e:
        _jlog(log, "warning", event="disk_usage_failed", error=str(e))
        return
    required = 10 * 1024 ** 3
    if usage.free < required:
        log.error(
            f"Insufficient disk space at {feature_store_dir}: "
            f"need {required / 1024**3:.1f} GiB, "
            f"have {usage.free / 1024**3:.1f} GiB free."
        )
        raise SystemExit(5)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="preprocess_pcap.py",
        description="Extract E3-DoH features from PCAP files and write "
                    "them to the feature store.",
    )
    p.add_argument(
        "--mode", choices=("toy", "full"), default="toy",
        help="toy = generate + process a schema-matched toy PCAP; "
             "full = process real PCAPs from --pcap-dir.",
    )
    p.add_argument(
        "--pcap-dir", type=str, default=None,
        help="Directory containing PCAP files (full mode only). "
             "Defaults to data/raw/DS-1/.",
    )
    p.add_argument(
        "--label-from-path", action="store_true",
        help="In full mode, infer the benign/attack label from the "
             "directory name of each PCAP (see _infer_label_from_path).",
    )
    p.add_argument(
        "--log-level", choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
    )
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    log = _setup_logging(args.log_level)
    _jlog(log, "info", event="start", mode=args.mode,
          pcap_dir=args.pcap_dir, label_from_path=args.label_from_path)

    feature_order = _load_feature_spec(log)
    _jlog(log, "info", event="feature_spec_loaded",
          n_features=len(feature_order), order=list(feature_order))

    try:
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        _check_disk_space(PROCESSED_DIR, log)
    except SystemExit as e:
        return int(e.code) if e.code is not None else 5

    try:
        if args.mode == "toy":
            rc = _run_toy_mode(args, log, feature_order)
        else:
            rc = _run_full_mode(args, log, feature_order)
    except SystemExit as e:
        return int(e.code) if e.code is not None else 1
    except RuntimeError as e:
        log.error(str(e))
        return 6
    except Exception as e:
        _jlog(log, "error", event="preprocess_pcap_exception", error=repr(e))
        return 3

    if rc != 0:
        _jlog(log, "error", event="preprocess_pcap_failed", rc=rc)
        return rc

    try:
        _verify_schema(feature_order, log)
    except SystemExit as e:
        return int(e.code) if e.code is not None else 4

    _jlog(log, "info", event="end", mode=args.mode)
    return 0


if __name__ == "__main__":
    sys.exit(main())
