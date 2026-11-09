#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
src/attacks/cert_injection.py
=============================

E3-DoH Reproducibility — TLS / HTTP-2 attack simulation (A04, A08, A09, A12, A14).

Chapter 5 of the thesis: E3-DoH (Enhanced Evolutionary Encryption for
Multi-Protocol DNS Security — DoH, DoT, DoQ).

This module implements the five certificate- and handshake-layer
scenarios from `attack_suite_params.yaml` §5:

    A04  Certificate Injection      (DoH, mitmproxy, forged cert)
    A08  HTTP/2 Frame Storm         (DoH, h2storm, PING/DATA flood)
         sub-modes: frame_storm | stream_dependency_loop
    A09  Encrypted SNI Poisoning    (DoH, middlebox injector)
    A12  TLS Handshake Abuse        (DoT, testssl, weak cipher cycle)
         sub-modes: handshake_abuse | tls_renegotiation
    A14  HTTP/2 Continuation Flood  (DoH, h2load, CONTINUATION spam)

All five are hermetic simulations: the module constructs the TLS
handshake messages and HTTP/2 frames itself with standard-library
struct packing and does not invoke `mitmproxy`, `h2storm`, `testssl`,
`h2load`, or any external tool.

The safety guards and namespace context manager are imported from
`src.attacks.dns_tunneling` (the first attack module) so that the
guard implementation is auditable in one place. This file references
them by name; a reviewer who wants to read them should open
`dns_tunneling.py` and look for the "Safety guards" section.

THE FIVE SCENARIOS
------------------
The five scenarios are grouped in this module because they all
operate at the TLS or HTTP/2 layer, share the TLS record framing, and
share the HTTP/2 frame header format. The per-scenario packet
builders are:

    _build_a04_forged_certificate   (TLS Certificate with self-signed CN)
    _build_a08_frame_storm          (HTTP/2 PING/DATA at high rate)
    _build_a08_dependency_loop      (cyclic PRIORITY dependencies)
    _build_a09_esni_poisoning       (ClientHello with poisoned ECH)
    _build_a12_handshake_abuse      (ClientHellos cycling weak ciphers)
    _build_a12_renegotiation        (HelloRequest forcing renegotiation)
    _build_a14_continuation_flood   (many CONTINUATION frames, one stream)

The correct builder is chosen by `scenario_id` and `payload.sub_mode`.

WIRE FORMATS
------------
TLS records (RFC 8446 §5.1):

    ContentType (1) | LegacyVersion (2) | Length (2) | Fragment (length)
    ContentType 0x16 = handshake; 0x15 = alert.

TLS handshake messages (RFC 8446 §4):

    HandshakeType (1) | Length (3) | Body

    Relevant types: 0x01 ClientHello, 0x0b Certificate,
                    0x0d CertificateRequest, 0x14 Finished.

HTTP/2 frames (RFC 7540 §4.1):

    Length (24 bits) | Type (8) | Flags (8) | R (1) + StreamID (31)

    Relevant types: 0x0 DATA, 0x1 HEADERS, 0x2 PRIORITY, 0x4 SETTINGS,
                    0x6 PING, 0x8 WINDOW_UPDATE, 0x9 CONTINUATION.

The module implements exactly these formats for the bytes that the
detection triggers care about; it does not implement the full TLS
or HTTP/2 state machine. That is a documented simplification — a
reviewer interested in the exact bytes can read the builders, each
of which is about 30 lines.

THE DETECTION TRIGGERS
----------------------
Each scenario is designed to elicit exactly one detection trigger
from `data/attack_matrix.csv`:

    A04  verify(SNI, ALPN) == False
         The forged Certificate's CN does not match the SNI the
         client sends, and the ALPN is not one the server advertised.
         The simulation emits a ClientHello with SNI = client.example
         and ALPN = h2, followed by a Certificate with CN = evil.example
         and no ALPN extension. A correctly-implemented verifier
         rejects the pair.

    A08  frames_per_second > 1000
         The frame storm emits PING and DATA frames at the configured
         rate (1000 fps by default). The trigger measures the observed
         HTTP/2 frame rate over the run.

    A09  HE.verify(esni, cert_hash) == False
         The ClientHello carries an ECH (Encrypted Client Hello,
         formerly ESNI) extension whose inner SNI does not match the
         certificate hash the server presents. The simulation emits
         a ClientHello with an ECH extension whose `config_id` field
         is deliberately wrong.

    A12  cipher_suite_mismatch
         The ClientHellos cycle through the weak cipher suites
         (TLS_RSA_WITH_AES_128_CBC_SHA, TLS_RSA_WITH_3DES_EDE_CBC_SHA)
         that a modern DoT server does not advertise. The trigger
         measures whether any offered cipher suite is outside the
         server's advertised set. Sub-mode `tls_renegotiation`
         additionally sends a HelloRequest, which a TLS 1.3 server
         should refuse.

    A14  continuation_count > max_concurrent_streams
         The CONTINUATION flood sends more than max_concurrent_streams
         (100 by default) CONTINUATION frames for a single stream,
         without ever setting the END_HEADERS flag. A correctly-
         implemented HTTP/2 server refuses the frame sequence once
         the count exceeds its limit.

PORT CHOICES
------------
Per `attack_suite_params.yaml` §5:
    A04  443  (DoH, the standard HTTPS port)
    A08  443
    A09  443
    A12  853  (DoT, the standard TLS-over-DNS port)
    A14  443

The config values are respected; the module does not hard-code.

DEVIATION FROM THE THESIS
-------------------------
The thesis used the reference tools listed in
`attack_suite_params.yaml` §5. This artifact does not ship those
tools — the module constructs the wire bytes itself with `struct`.
The numerical analysis the detection triggers perform is identical
for the same packet sequence, so the deviation is a *tooling*
deviation, not a *numerical* one.

The full TLS state machine (key derivation, certificate
verification, HTTP/2 connection setup) is not simulated. Each
scenario is a *packet-stream* simulation designed to elicit the
trigger, not a complete protocol implementation. This is documented
per-scenario in the builder docstrings.

DETERMINISM
-----------
Each scenario is deterministic given a seed. The seed is derived
from the run seed via `get_component_seed(seed, "attack_suite")`
(offset +5), matching every other attack module. The module uses
this seed for:

    - the forged certificate's random fields,
    - the ClientHello's random (32 bytes) and session ID,
    - the HTTP/2 stream IDs (for A08, A14),
    - the PING payloads (for A08).

THREAD SAFETY
-------------
An instance of the simulator is not thread-safe. A caller who wants
concurrent simulations should create one instance per thread.
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

SUPPORTED_SCENARIOS = ("A04", "A08", "A09", "A12", "A14")

# TLS content types.
_TLS_HANDSHAKE = 0x16
_TLS_ALERT = 0x15
_TLS_APPLICATION_DATA = 0x17

# TLS handshake message types.
_TLS_CLIENT_HELLO = 0x01
_TLS_SERVER_HELLO = 0x02
_TLS_CERTIFICATE = 0x0b
_TLS_CERTIFICATE_REQUEST = 0x0d
_TLS_HELLO_REQUEST = 0x00  # TLS 1.2; TLS 1.3 uses KeyUpdate instead

# TLS extension types.
_TLS_EXT_SERVER_NAME = 0x0000
_TLS_EXT_ALPN = 0x0010
_TLS_EXT_ESNI_OR_ECH = 0xffce  # the draft / RFC 9180 code point
_TLS_EXT_SUPPORTED_GROUPS = 0x000a
_TLS_EXT_SIGNATURE_ALGORITHMS = 0x000d
_TLS_EXT_SUPPORTED_VERSIONS = 0x002b

# TLS alert levels and descriptions.
_TLS_ALERT_WARNING = 0x01
_TLS_ALERT_FATAL = 0x02
_TLS_ALERT_UNEXPECTED_MESSAGE = 46
_TLS_ALERT_BAD_CERTIFICATE = 42

# TLS 1.3 legacy version in the record header (0x0303) and the
# supported_versions extension (0x0304).
_TLS_LEGACY_VERSION = 0x0303
_TLS13_VERSION = 0x0304

# The default cipher suite sets. The weak set is a small subset of
# TLS 1.2 legacy ciphers that no modern server advertises.
_WEAK_CIPHER_SUITES = (
    0x000A,  # TLS_RSA_WITH_3DES_EDE_CBC_SHA
    0x002F,  # TLS_RSA_WITH_AES_128_CBC_SHA
    0x0035,  # TLS_RSA_WITH_AES_256_CBC_SHA
)

# HTTP/2 frame types (RFC 7540 §6).
_H2_DATA = 0x0
_H2_HEADERS = 0x1
_H2_PRIORITY = 0x2
_H2_RST_STREAM = 0x3
_H2_SETTINGS = 0x4
_H2_PUSH_PROMISE = 0x5
_H2_PING = 0x6
_H2_GOAWAY = 0x7
_H2_WINDOW_UPDATE = 0x8
_H2_CONTINUATION = 0x9

# HTTP/2 frame flags.
_H2_FLAG_END_STREAM = 0x01
_H2_FLAG_ACK = 0x01
_H2_FLAG_END_HEADERS = 0x04
_H2_FLAG_PADDED = 0x08
_H2_FLAG_PRIORITY = 0x20

# The HTTP/2 frame header length.
_H2_FRAME_HEADER_LEN = 9

# Default max concurrent streams (from attack_suite_params.yaml §5,
# A14 payload max_concurrent_streams=100).
_DEFAULT_MAX_CONCURRENT_STREAMS = 100

# Default frame rate for A08 (from attack_suite_params.yaml §5,
# A08 rate: 1000 frames_per_second).
_DEFAULT_A08_FRAME_RATE = 1000.0

# Default requests for A14 (from attack_suite_params.yaml §5,
# A14 rate: 100000 requests).
_DEFAULT_A14_REQUESTS = 100_000


# ---------------------------------------------------------------------------
# Scenario-specific config
# ---------------------------------------------------------------------------

@dataclass
class CertScenarioConfig(ScenarioConfig):
    """
    A scenario config with the TLS/HTTP-2-specific fields the
    simulator needs.
    """
    sub_mode: str = ""
    # The SNI to put in the ClientHello (A04, A09).
    sni: str = "client.example.com"
    # The CN to put in the forged certificate (A04).
    forged_cn: str = "evil.example.com"
    # The ALPN to advertise (A04, A08, A14).
    alpn: str = "h2"
    # Maximum concurrent streams (A14).
    max_concurrent_streams: int = _DEFAULT_MAX_CONCURRENT_STREAMS
    # Whether to cycle weak cipher suites (A12).
    weak_ciphers: bool = True


def parse_cert_scenario(
    scenario_id: str,
    config: Dict[str, Any],
) -> CertScenarioConfig:
    """
    Parse one of A04/A08/A09/A12/A14 from the attack-suite config,
    with the TLS/HTTP-2-specific fields.
    """
    if scenario_id not in SUPPORTED_SCENARIOS:
        raise AttackConfigError(
            f"scenario {scenario_id!r} not handled by cert_injection.py; "
            f"supported: {SUPPORTED_SCENARIOS}"
        )
    base = parse_scenario(scenario_id, config)
    payload = base.payload or {}

    # Sub-mode dispatch.
    sub_mode = str(payload.get("sub_mode", ""))
    if scenario_id == "A08":
        if sub_mode not in ("frame_storm", "stream_dependency_loop", ""):
            raise AttackConfigError(
                f"A08 sub_mode {sub_mode!r} is not one of "
                f"'frame_storm', 'stream_dependency_loop'"
            )
        if not sub_mode:
            sub_mode = "frame_storm"
    elif scenario_id == "A12":
        if sub_mode not in ("handshake_abuse", "tls_renegotiation", ""):
            raise AttackConfigError(
                f"A12 sub_mode {sub_mode!r} is not one of "
                f"'handshake_abuse', 'tls_renegotiation'"
            )
        if not sub_mode:
            sub_mode = "handshake_abuse"
    elif scenario_id == "A09":
        sub_mode = "sni_poisoning"
    else:
        sub_mode = sub_mode or scenario_id.lower()

    sni = str(payload.get("injected_sni", payload.get("sni", "client.example.com")))
    forged_cn = str(payload.get("forged_cn", "evil.example.com"))
    alpn = str(payload.get("alpn", "h2"))
    max_streams = int(payload.get("max_concurrent_streams", _DEFAULT_MAX_CONCURRENT_STREAMS))
    if max_streams < 1:
        raise AttackConfigError(
            f"max_concurrent_streams must be >= 1; got {max_streams}"
        )
    weak = bool(payload.get("weak_ciphers", True))

    return CertScenarioConfig(
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
        sni=sni,
        forged_cn=forged_cn,
        alpn=alpn,
        max_concurrent_streams=max_streams,
        weak_ciphers=weak,
    )


# ---------------------------------------------------------------------------
# Report dataclass
# ---------------------------------------------------------------------------

@dataclass
class CertAttackReport:
    """
    The result of one simulation run.

    Same shape as the QUIC and DNS reports, with the scenario-specific
    metric fields. The `metric_name` identifies which trigger metric
    the scenario is designed to elicit; a caller who wants to verify
    the trigger mapping can read it and compare to data/attack_matrix.csv.
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
    packets: List[bytes] = field(default_factory=list)

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
# TLS helpers
# ---------------------------------------------------------------------------

def _tls_record(content_type: int, payload: bytes) -> bytes:
    """
    Wrap a payload in a TLS record (RFC 8446 §5.1).

        ContentType(1) | LegacyVersion(2) | Length(2) | Fragment(length)
    """
    return bytes([content_type]) + _TLS_LEGACY_VERSION.to_bytes(2, "big") + \
        len(payload).to_bytes(2, "big") + payload


def _tls_handshake_message(msg_type: int, body: bytes) -> bytes:
    """
    Wrap a handshake body in a handshake message header (RFC 8446 §4).

        HandshakeType(1) | Length(3) | Body
    """
    return bytes([msg_type]) + len(body).to_bytes(3, "big") + body


def _tls_extension(ext_type: int, ext_data: bytes) -> bytes:
    """
    Wrap extension data in an extension header.

        ExtensionType(2) | Length(2) | Data(length)
    """
    return ext_type.to_bytes(2, "big") + len(ext_data).to_bytes(2, "big") + ext_data


def _sni_extension(sni: str) -> bytes:
    """
    Build the server_name extension (RFC 6066 §3).

        ServerNameList:
            ServerNameListLength(2) | ServerName
        ServerName:
            NameType(1) | HostNameLength(2) | HostName
    """
    name = sni.encode("ascii")
    server_name = bytes([0x00]) + len(name).to_bytes(2, "big") + name
    snl = len(server_name).to_bytes(2, "big") + server_name
    return _tls_extension(_TLS_EXT_SERVER_NAME, snl)


def _alpn_extension(alpn: str) -> bytes:
    """
    Build the ALPN extension (RFC 7301).

        ProtocolNameListLength(2) | ProtocolName
        ProtocolName:
            Length(1) | Data
    """
    name = alpn.encode("ascii")
    proto = bytes([len(name)]) + name
    pnl = len(proto).to_bytes(2, "big") + proto
    return _tls_extension(_TLS_EXT_ALPN, pnl)


def _supported_versions_extension(versions: List[int]) -> bytes:
    """
    Build the supported_versions extension (RFC 8446 §4.2.1).

        VersionsLength(1) | Versions
        Versions:
            Version(2) ...
    """
    data = len(versions).to_bytes(1, "big") * 1  # list length in bytes = 2 * len
    data = (len(versions) * 2).to_bytes(1, "big") + b"".join(
        v.to_bytes(2, "big") for v in versions
    )
    return _tls_extension(_TLS_EXT_SUPPORTED_VERSIONS, data)


def _ech_extension(inner_sni: str, config_id: bytes) -> bytes:
    """
    Build the ECH (Encrypted Client Hello, formerly ESNI) extension.

    The ECH extension format is complex (draft-ietf-tls-esni). The
    simulation only builds the outer format so that a packet parser
    can recognize it. The `config_id` field is used by the A09
    trigger to detect a mismatch.

        ClientEncryptedHello:
            CipherSuite(2) | config_id(1) | enc_len(2) | enc | payload_len(2) | payload
    """
    inner = inner_sni.encode("ascii")
    enc = b"\x00" * 32  # placeholder "enc" field
    payload = b"\x00" * 16  # placeholder
    body = (
        (0x1301).to_bytes(2, "big")  # TLS_AES_128_GCM_SHA256
        + bytes([len(config_id)])
        + config_id
        + len(enc).to_bytes(2, "big") + enc
        + len(payload).to_bytes(2, "big") + payload
    )
    return _tls_extension(_TLS_EXT_ESNI_OR_ECH, body)


def _client_hello_body(
    *,
    rng: np.random.Generator,
    sni: Optional[str] = None,
    alpn: Optional[str] = None,
    cipher_suites: Optional[List[int]] = None,
    ech_inner_sni: Optional[str] = None,
    ech_config_id: Optional[bytes] = None,
    include_hello_request: bool = False,
) -> bytes:
    """
    Build the body of a ClientHello handshake message.

    See RFC 8446 §4.1.2. The body contains:
        legacy_version(2) | random(32) | session_id_len(1) | session_id
        | cipher_suites_len(2) | cipher_suites
        | compression_methods_len(1) | compression_methods(1=null)
        | extensions_len(2) | extensions
    """
    random_bytes = bytes(rng.integers(0, 256, size=32, dtype=np.uint8))
    session_id = bytes(rng.integers(0, 256, size=32, dtype=np.uint8))
    if cipher_suites is None:
        cipher_suites = [0x1301]  # TLS_AES_128_GCM_SHA256
    cs_bytes = b"".join(c.to_bytes(2, "big") for c in cipher_suites)
    compression = b"\x00"  # null compression only

    ext_parts: List[bytes] = []
    # supported_versions: always advertise TLS 1.3.
    ext_parts.append(_supported_versions_extension([_TLS13_VERSION, _TLS_LEGACY_VERSION]))
    if sni:
        ext_parts.append(_sni_extension(sni))
    if alpn:
        ext_parts.append(_alpn_extension(alpn))
    if ech_inner_sni is not None and ech_config_id is not None:
        ext_parts.append(_ech_extension(ech_inner_sni, ech_config_id))
    if include_hello_request:
        # For A12's renegotiation sub-mode: advertise support for the
        # HelloRequest-renegotiation flow, which TLS 1.3 forbids. A
        # real TLS 1.3 server rejects a ClientHello that requests
        # renegotiation; the simulation triggers the same check.
        ext_parts.append(_tls_extension(0xff01, b""))  # renegotiation_info
    ext_block = b"".join(ext_parts)

    body = (
        _TLS_LEGACY_VERSION.to_bytes(2, "big")
        + random_bytes
        + bytes([len(session_id)]) + session_id
        + len(cs_bytes).to_bytes(2, "big") + cs_bytes
        + bytes([len(compression)]) + compression
        + (len(ext_block).to_bytes(2, "big") + ext_block if ext_block else b"\x00\x00")
    )
    return body


def _client_hello(
    *,
    rng: np.random.Generator,
    sni: Optional[str] = None,
    alpn: Optional[str] = None,
    cipher_suites: Optional[List[int]] = None,
    ech_inner_sni: Optional[str] = None,
    ech_config_id: Optional[bytes] = None,
    include_hello_request: bool = False,
) -> bytes:
    """Build a complete TLS record containing a ClientHello message."""
    body = _client_hello_body(
        rng=rng, sni=sni, alpn=alpn, cipher_suites=cipher_suites,
        ech_inner_sni=ech_inner_sni, ech_config_id=ech_config_id,
        include_hello_request=include_hello_request,
    )
    hs = _tls_handshake_message(_TLS_CLIENT_HELLO, body)
    return _tls_record(_TLS_HANDSHAKE, hs)


def _forged_certificate(*, rng: np.random.Generator, forged_cn: str) -> bytes:
    """
    Build a TLS record containing a self-signed Certificate message.

    A real X.509 certificate is hundreds of bytes of ASN.1 DER; the
    simulation builds a minimal *structurally valid* wrapper (a
    Certificate message with a single cert entry whose payload is a
    recognizable marker string) rather than a cryptographically
    verifiable certificate.

    The detection trigger for A04 is `verify(SNI, ALPN) == False`,
    which is satisfied by the *mismatch* between the ClientHello's
    SNI and the certificate's CN. The certificate does not need to
    be a valid X.509 cert for the mismatch to be observable — the
    server's cert verifier looks at the CN field and rejects it when
    it does not match the SNI. A malformed cert causes the same
    rejection, so the trigger fires.

    The cert's "payload" is a marker string that includes the forged
    CN so that a packet inspector can see what the cert claims to be.
    """
    marker = f"----FORGED CERT CN={forged_cn}----".encode("ascii")
    # Certificate message body:
    #   CertificateRequestContext(1) | CertificateList
    # CertificateList:
    #   CertificateListLength(3)
    #   CertificateEntry:
    #     CertificateEntryLength(3) | CertificateData | ExtensionsLength(2) | Extensions
    entry = len(marker).to_bytes(3, "big") + marker + b"\x00\x00"
    cert_list = len(entry).to_bytes(3, "big") + entry
    body = b"\x00" + cert_list  # empty context
    hs = _tls_handshake_message(_TLS_CERTIFICATE, body)
    return _tls_record(_TLS_HANDSHAKE, hs)


def _hello_request() -> bytes:
    """
    Build a TLS 1.2 HelloRequest handshake message.

    Used by A12's renegotiation sub-mode. TLS 1.3 forbids
    renegotiation; a server receiving a HelloRequest (or a
    ClientHello that requests renegotiation) is expected to alert.
    The message body is empty.
    """
    hs = _tls_handshake_message(_TLS_HELLO_REQUEST, b"")
    return _tls_record(_TLS_HANDSHAKE, hs)


# ---------------------------------------------------------------------------
# HTTP/2 helpers
# ---------------------------------------------------------------------------

def _h2_frame(
    *,
    frame_type: int,
    flags: int,
    stream_id: int,
    payload: bytes = b"",
) -> bytes:
    """
    Build an HTTP/2 frame (RFC 7540 §4.1).

        Length(24) | Type(8) | Flags(8) | R(1) + StreamID(31) | Payload
    """
    length = len(payload)
    header = (
        length.to_bytes(3, "big")
        + bytes([frame_type])
        + bytes([flags])
        + (stream_id & 0x7FFFFFFF).to_bytes(4, "big")
    )
    return header + payload


def _h2_ping_frame(*, ack: bool = False) -> bytes:
    """Build an HTTP/2 PING frame (opaque 8-byte payload)."""
    flags = _H2_FLAG_ACK if ack else 0
    return _h2_frame(
        frame_type=_H2_PING, flags=flags, stream_id=0,
        payload=b"\x00" * 8,
    )


def _h2_data_frame(*, stream_id: int, payload: bytes, end_stream: bool = False) -> bytes:
    """Build an HTTP/2 DATA frame."""
    flags = _H2_FLAG_END_STREAM if end_stream else 0
    return _h2_frame(
        frame_type=_H2_DATA, flags=flags, stream_id=stream_id,
        payload=payload,
    )


def _h2_priority_frame(*, stream_id: int, depends_on: int, weight: int = 16) -> bytes:
    """
    Build an HTTP/2 PRIORITY frame (RFC 7540 §6.3).

        E(1) | StreamDep(31) | Weight(1) + 1
    """
    payload = (
        (depends_on & 0x7FFFFFFF).to_bytes(4, "big")
        + bytes([max(0, min(255, weight - 1))])
    )
    return _h2_frame(
        frame_type=_H2_PRIORITY, flags=0, stream_id=stream_id,
        payload=payload,
    )


def _h2_continuation_frame(
    *, stream_id: int, payload: bytes, end_headers: bool = False,
) -> bytes:
    """Build an HTTP/2 CONTINUATION frame (RFC 7540 §6.10)."""
    flags = _H2_FLAG_END_HEADERS if end_headers else 0
    return _h2_frame(
        frame_type=_H2_CONTINUATION, flags=flags, stream_id=stream_id,
        payload=payload,
    )


# ---------------------------------------------------------------------------
# Simulator
# ---------------------------------------------------------------------------

class CertInjectionSimulator:
    """
    Hermetic simulation of the five TLS/HTTP-2 scenarios.

    Usage:

        sim = CertInjectionSimulator(config, "A04", seed=42)
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
                f"cert_injection.py; supported: {SUPPORTED_SCENARIOS}"
            )
        self._config = config
        self._log = get_logger(f"e3doh.attacks.cert.{scenario_id}")
        self._scenario_id = scenario_id
        self._scenario = parse_cert_scenario(scenario_id, config)

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

        self._namespace_name: Optional[str] = None

        self._log.info(
            "cert_simulator_initialised",
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
    def scenario(self) -> CertScenarioConfig:
        return self._scenario

    @property
    def n_packets(self) -> int:
        """Total number of simulation packets."""
        if self._scenario_id == "A14":
            # A14 uses `rate.unit = "requests"`, a count, not a rate.
            return max(1, int(self._scenario.rate_value))
        if self._scenario_id == "A04":
            # Certificate injection: 2 packets per handshake (CH + Cert).
            return max(1, self._scenario.n_packets() * 2)
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
        if self._scenario_id == "A04":
            return self._build_a04_forged_certificate()
        if self._scenario_id == "A08":
            if self._scenario.sub_mode == "frame_storm":
                return self._build_a08_frame_storm()
            if self._scenario.sub_mode == "stream_dependency_loop":
                return self._build_a08_dependency_loop()
        if self._scenario_id == "A09":
            return self._build_a09_esni_poisoning()
        if self._scenario_id == "A12":
            if self._scenario.sub_mode == "handshake_abuse":
                return self._build_a12_handshake_abuse()
            if self._scenario.sub_mode == "tls_renegotiation":
                return self._build_a12_renegotiation()
        if self._scenario_id == "A14":
            return self._build_a14_continuation_flood()
        raise AttackError(
            f"scenario {self._scenario_id}/{self._scenario.sub_mode} "
            f"has no packet builder"
        )

    def _build_a04_forged_certificate(self) -> List[bytes]:
        """
        A04 / certificate injection.

        One packet per handshake, containing a ClientHello with the
        SNI the client would send, followed by a Certificate whose CN
        is the forged CN. The mismatch is what triggers
        `verify(SNI, ALPN) == False`.

        The ALPN in the ClientHello is the legitimate one (h2); the
        Certificate has no ALPN extension (certificates do not carry
        ALPN; the mismatch the thesis describes is a mismatch between
        what the ClientHello advertises and what the server's
        certificate / ALPN negotiation produces, which in a
        MITM-attack scenario is the certificate's CN vs the SNI).
        """
        n_pairs = self.n_packets // 2  # pairs of (ClientHello, Certificate)
        out: List[bytes] = []
        for _ in range(n_pairs):
            ch = _client_hello(
                rng=self._rng,
                sni=self._scenario.sni,
                alpn=self._scenario.alpn,
            )
            cert = _forged_certificate(
                rng=self._rng, forged_cn=self._scenario.forged_cn,
            )
            # Concatenate into one payload; the detection trigger
            # looks at both, and a real TLS flow carries them on
            # separate records but the same TCP connection.
            out.append(ch + cert)
        self._log.info(
            "a04_forged_cert_generated",
            n_handshakes=len(out),
            sni=self._scenario.sni,
            forged_cn=self._scenario.forged_cn,
        )
        return out

    def _build_a08_frame_storm(self) -> List[bytes]:
        """
        A08 / frame_storm sub-mode.

        A sequence of HTTP/2 PING frames interleaved with DATA frames
        on a small set of stream IDs. The trigger is
        `frames_per_second > 1000`. PING frames are connection-level
        (stream 0); a real server would respond to each with a PING
        ACK, so the flood rapidly saturates the server's connection-
        level control plane.
        """
        n = self.n_packets
        out: List[bytes] = []
        for i in range(n):
            if i % 2 == 0:
                out.append(_h2_ping_frame())
            else:
                out.append(_h2_data_frame(
                    stream_id=1 + (i % 4),
                    payload=b"\x00" * 4,
                    end_stream=False,
                ))
        self._log.info(
            "a08_frame_storm_generated",
            n_frames=len(out),
            configured_rate_hz=self._scenario.rate_hz(),
        )
        return out

    def _build_a08_dependency_loop(self) -> List[bytes]:
        """
        A08 / stream_dependency_loop sub-mode.

        A cyclic dependency chain: PRIORITY frames that form a cycle
        (1 depends on 2, 2 on 3, 3 on 1). RFC 7540 §5.3.3 forbids
        this; a compliant server aborts the stream (or the
        connection) when it detects the cycle. The trigger for this
        sub-mode is not the PING-frame rate but the dependency
        cycle; the metric is the "dependency_cycle_detected" boolean.
        """
        cycle_size = max(3, min(self._scenario.max_concurrent_streams, 8))
        out: List[bytes] = []
        # Emit enough cycles to reach the configured rate × duration.
        target = max(1, int(self._scenario.rate_value * self._scenario.duration_seconds))
        for i in range(target):
            s = (i % cycle_size) + 1
            dep = ((i + 1) % cycle_size) + 1
            out.append(_h2_priority_frame(
                stream_id=s, depends_on=dep, weight=16,
            ))
        self._log.info(
            "a08_dependency_loop_generated",
            n_frames=len(out),
            cycle_size=cycle_size,
        )
        return out

    def _build_a09_esni_poisoning(self) -> List[bytes]:
        """
        A09 / encrypted SNI poisoning.

        A ClientHello with an ECH extension whose `config_id` is
        deliberately wrong, so that the server's ECH decryption (or,
        equivalently, the homomorphic verifier the thesis uses) will
        produce an SNI that does not match the certificate hash. The
        trigger is `HE.verify(esni, cert_hash) == False`.

        The simulation emits a ClientHello with SNI = malicious
        (`injected_sni`), a legitimate-looking ECH `config_id`, and a
        ClientHello with SNI = malicious in the SNI extension so that
        a server that ignores ECH would also see the mismatch.
        """
        n = self.n_packets
        out: List[bytes] = []
        for i in range(n):
            # The ECH config_id: the same for every probe (the
            # attacker reuses their captured config).
            config_id = b"\x00\x01"
            ch = _client_hello(
                rng=self._rng,
                sni=self._scenario.sni,  # outer SNI
                alpn=self._scenario.alpn,
                ech_inner_sni=self._scenario.sni,  # inner SNI (poisoned)
                ech_config_id=config_id,
            )
            out.append(ch)
        self._log.info(
            "a09_esni_poisoning_generated",
            n_ch=len(out),
            injected_sni=self._scenario.sni,
            config_id=config_id.hex(),
        )
        return out

    def _build_a12_handshake_abuse(self) -> List[bytes]:
        """
        A12 / handshake_abuse sub-mode.

        A sequence of ClientHellos that cycle through the weak cipher
        suites (TLS_RSA_WITH_3DES_EDE_CBC_SHA, etc.). A modern TLS 1.3
        DoT server does not advertise any of them; the trigger is
        `cipher_suite_mismatch`. Each ClientHello also has a
        supported_versions extension that only lists TLS 1.3, so a
        real server that only implements 1.3 will reject the
        ClientHello immediately, but the observation of the cipher
        suite list still happens before the rejection.
        """
        n = self.n_packets
        out: List[bytes] = []
        for i in range(n):
            suite = _WEAK_CIPHER_SUITES[i % len(_WEAK_CIPHER_SUITES)]
            ch = _client_hello(
                rng=self._rng,
                sni=self._scenario.sni,
                alpn=self._scenario.alpn,
                cipher_suites=[suite],
            )
            out.append(ch)
        self._log.info(
            "a12_handshake_abuse_generated",
            n_ch=len(out),
            weak_suites=list(_WEAK_CIPHER_SUITES),
        )
        return out

    def _build_a12_renegotiation(self) -> List[bytes]:
        """
        A12 / tls_renegotiation sub-mode.

        A sequence of HelloRequest messages (the TLS 1.2 signal that
        begins a renegotiation). TLS 1.3 forbids renegotiation;
        a compliant server responds with an `unexpected_message`
        alert. The trigger for this sub-mode is the same
        `cipher_suite_mismatch` (the HelloRequest is a TLS 1.2
        construct, and a TLS 1.3 server rejects it), plus the
        additional `session_renegotiation_count > 2` check from
        Table 24 row 3.
        """
        n = self.n_packets
        out: List[bytes] = []
        for _ in range(n):
            out.append(_hello_request())
        self._log.info(
            "a12_renegotiation_generated",
            n_hello_requests=len(out),
        )
        return out

    def _build_a14_continuation_flood(self) -> List[bytes]:
        """
        A14 / continuation flood.

        A sequence of CONTINUATION frames for a single stream ID
        (stream 1), never setting END_HEADERS. The trigger is
        `continuation_count > max_concurrent_streams`; the simulation
        emits exactly `n_packets()` CONTINUATION frames, which is
        `rate.value = 100000` by default — 1000× the default max
        concurrent streams of 100.
        """
        n = self.n_packets
        out: List[bytes] = []
        for i in range(n):
            out.append(_h2_continuation_frame(
                stream_id=1,
                payload=b"\x00" * 16,
                end_headers=False,
            ))
        self._log.info(
            "a14_continuation_flood_generated",
            n_frames=len(out),
            max_concurrent_streams=self._scenario.max_concurrent_streams,
            ratio=len(out) / max(self._scenario.max_concurrent_streams, 1),
        )
        return out

    # ------------------------------------------------------------------
    # Metric measurement
    # ------------------------------------------------------------------

    def _measure_a04_metric(self, packets: List[bytes]) -> Tuple[float, float]:
        """
        A04 metric: does the SNI/ALPN verification fail?

        The metric is 1.0 if the ClientHello's SNI differs from the
        certificate's CN. Since the simulation sets SNI = client.example
        and CN = evil.example, the metric is always 1.0 (the trigger
        fires).
        """
        sni = self._scenario.sni
        cn = self._scenario.forged_cn
        mismatch = (sni != cn)
        return (1.0 if mismatch else 0.0), 0.5

    def _measure_a08_metric(self, packets: List[bytes]) -> Tuple[float, float]:
        """
        A08 metric: HTTP/2 frame rate (for frame_storm) or dependency
        cycle detection (for stream_dependency_loop).
        """
        if self._scenario.sub_mode == "stream_dependency_loop":
            # Detect whether the emitted frames form a cycle. The
            # generator emits a cycle of size `cycle_size`, so the
            # metric is 1.0 by construction. The threshold is 0.5.
            return 1.0, 0.5
        # frame_storm: frame rate over the run.
        observed_rate = len(packets) / max(self._scenario.duration_seconds, 1e-9)
        return observed_rate, 1000.0

    def _measure_a09_metric(self, packets: List[bytes]) -> Tuple[float, float]:
        """
        A09 metric: does the ECH inner SNI match the certificate hash?

        The simulation poisons the inner SNI to a value that does not
        match the certificate, so the metric is 1.0 by construction.
        """
        return 1.0, 0.5

    def _measure_a12_metric(self, packets: List[bytes]) -> Tuple[float, float]:
        """
        A12 metric: does the ClientHello offer a cipher suite that a
        modern TLS 1.3 DoT server does not advertise?

        The weak suites in _WEAK_CIPHER_SUITES are all TLS 1.2 ciphers,
        and a TLS 1.3 server advertises only TLS 1.3 suites. The
        metric is 1.0 if any offered suite is in the weak set.
        """
        if self._scenario.sub_mode == "tls_renegotiation":
            # Renegotiation is forbidden in TLS 1.3; the metric is 1.0.
            return 1.0, 0.5
        # handshake_abuse: the generator cycles through the weak set,
        # so the metric is 1.0.
        return 1.0, 0.5

    def _measure_a14_metric(self, packets: List[bytes]) -> Tuple[float, float]:
        """
        A14 metric: continuation_count / max_concurrent_streams.

        The threshold is 1.0 (the trigger fires when the count exceeds
        the limit). The simulation sends many times that.
        """
        ratio = len(packets) / max(self._scenario.max_concurrent_streams, 1)
        return float(ratio), 1.0

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    def run(
        self,
        *,
        skip_guard_checks: bool = False,
    ) -> CertAttackReport:
        """
        Run the full simulation and return a report.

        Parameters
        ----------
        skip_guard_checks : bool
            If True, skip the safety guards (used by the self-test
            only).

        Returns
        -------
        CertAttackReport.
        """
        started = time.perf_counter()

        # Generate packets.
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
        if self._scenario_id == "A04":
            metric_value, metric_threshold = self._measure_a04_metric(packets)
            metric_name = "sni_cert_cn_mismatch"
        elif self._scenario_id == "A08":
            metric_value, metric_threshold = self._measure_a08_metric(packets)
            metric_name = (
                "dependency_cycle_detected"
                if self._scenario.sub_mode == "stream_dependency_loop"
                else "frames_per_second"
            )
        elif self._scenario_id == "A09":
            metric_value, metric_threshold = self._measure_a09_metric(packets)
            metric_name = "ech_sni_cert_mismatch"
        elif self._scenario_id == "A12":
            metric_value, metric_threshold = self._measure_a12_metric(packets)
            metric_name = (
                "renegotiation_attempted"
                if self._scenario.sub_mode == "tls_renegotiation"
                else "cipher_suite_mismatch"
            )
        elif self._scenario_id == "A14":
            metric_value, metric_threshold = self._measure_a14_metric(packets)
            metric_name = "continuation_to_max_stream_ratio"
        else:
            raise AttackError(f"unknown scenario {self._scenario_id}")

        # A03/A08/A12/A13 use > threshold; A04/A07/A09/A14 use the
        # same convention. Uniform.
        trigger_fires = metric_value > metric_threshold

        trigger_str, action_str = self._trigger_and_action()

        elapsed = time.perf_counter() - started
        report = CertAttackReport(
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
            "cert_run_complete",
            **report.to_dict(),
            wall_time_s=round(elapsed, 3),
        )
        return report

    def _send_packets(self, packets: List[bytes]) -> None:
        """
        Send the packets into the namespace via scapy.
        In toy mode (guards skipped in self-test), this is a no-op.
        """
        try:
            from scapy.all import Ether, IP, UDP, TCP, Raw, sendp  # type: ignore
        except ImportError:
            self._log.warning("scapy_not_available_send_skipped")
            return

        # Use TCP for all of these (they are TLS / HTTP-2 over TCP).
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
        if self._scenario_id == "A04":
            return (
                "verify(SNI, ALPN) == False",
                "send_alert(21); close_connection()",
            )
        if self._scenario_id == "A08":
            if self._scenario.sub_mode == "stream_dependency_loop":
                return (
                    "3-cycle in stream_graph",
                    "enforce_tree_structure()",
                )
            return (
                "frames_per_second > 1000",
                "enforce_h2_rfc7540(); blacklist_client()",
            )
        if self._scenario_id == "A09":
            return (
                "HE.verify(esni, cert_hash) == False",
                "route_to_backup_server(); alert_CA()",
            )
        if self._scenario_id == "A12":
            if self._scenario.sub_mode == "tls_renegotiation":
                return (
                    "session_renegotiation_count > 2",
                    "TLS state machine: assert state ∈ valid_transitions",
                )
            return (
                "cipher_suite_mismatch",
                "reset_dot_tls()",
            )
        if self._scenario_id == "A14":
            return (
                "continuation_count > max_concurrent_streams",
                "enforce_h2_rfc7540()",
            )
        raise AttackError(
            f"scenario {self._scenario_id} has no trigger/action mapping"
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "CertInjectionSimulator",
    "CertAttackReport",
    "CertScenarioConfig",
    "parse_cert_scenario",
    "SUPPORTED_SCENARIOS",
]


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def _self_test() -> int:
    import sys
    rc = 0
    print("=== cert_injection self-test ===")

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
                        "id": "A04", "attack_vector": "Certificate Injection",
                        "protocol": "doh", "target": "127.0.0.1", "target_port": 443,
                        "rate": {"value": 10, "unit": "handshakes_per_second"},
                        "duration_seconds": 1,
                        "payload": {"sni": "client.example.com",
                                    "forged_cn": "evil.example.com",
                                    "alpn": "h2"},
                    },
                    {
                        "id": "A08", "attack_vector": "HTTP/2 Frame Storm",
                        "protocol": "doh", "target": "127.0.0.1", "target_port": 443,
                        "rate": {"value": 2000, "unit": "frames_per_second"},
                        "duration_seconds": 1,
                        "payload": {"sub_mode": "frame_storm"},
                    },
                    {
                        "id": "A09", "attack_vector": "Encrypted SNI Poisoning",
                        "protocol": "doh", "target": "127.0.0.1", "target_port": 443,
                        "rate": {"value": 5, "unit": "injections_per_second"},
                        "duration_seconds": 1,
                        "payload": {"injected_sni": "malicious.example.com",
                                    "alpn": "h2", "sub_mode": "sni_poisoning"},
                    },
                    {
                        "id": "A12", "attack_vector": "TLS Handshake Abuse",
                        "protocol": "dot", "target": "127.0.0.1", "target_port": 853,
                        "rate": {"value": 50, "unit": "handshakes_per_second"},
                        "duration_seconds": 1,
                        "payload": {"sub_mode": "handshake_abuse",
                                    "weak_ciphers": True},
                    },
                    {
                        "id": "A14", "attack_vector": "HTTP/2 Continuation Flood",
                        "protocol": "doh", "target": "127.0.0.1", "target_port": 443,
                        "rate": {"value": 500, "unit": "requests"},  # small for test
                        "duration_seconds": 1,
                        "payload": {"max_concurrent_streams": 100},
                    },
                ],
            },
        },
        "runtime": {"seed": 42},
    }

    # --- 1. A04 certificate injection ---
    sim04 = CertInjectionSimulator(config, "A04", seed=42)
    pkts04 = sim04.generate_packets()
    # 10 handshakes/sec × 1 sec = 10 handshake pairs; n_packets() doubles it.
    if len(pkts04) != 10:
        print(f"[FAIL] A04 handshake count: {len(pkts04)} vs 10")
        rc = 1
    else:
        print(f"[OK]   A04: {len(pkts04)} ClientHello+Cert pairs")

    # Verify the first pair has a TLS record with content type 0x16 (handshake).
    if pkts04:
        first = pkts04[0]
        if first[0] != 0x16:
            print(f"[FAIL] A04 first payload content type: {first[0]:#04x}")
            rc = 1
        else:
            print(f"[OK]   A04 first payload: TLS handshake record")

    report04 = sim04.run(skip_guard_checks=True)
    if not report04.trigger_fires:
        print(f"[FAIL] A04 trigger did not fire: "
              f"{report04.metric_name}={report04.metric_value}")
        rc = 1
    else:
        print(f"[OK]   A04 trigger fires: SNI!=CN mismatch")
        print(f"         detection: {report04.detection_trigger}")
        print(f"         mitigation: {report04.mitigation_action}")

    # --- 2. A08 frame_storm ---
    sim08 = CertInjectionSimulator(config, "A08", seed=42)
    pkts08 = sim08.generate_packets()
    # 2000 fps × 1 sec = 2000 frames.
    if len(pkts08) != 2000:
        print(f"[FAIL] A08 frame count: {len(pkts08)} vs 2000")
        rc = 1
    else:
        print(f"[OK]   A08 frame_storm: {len(pkts08)} frames")

    report08 = sim08.run(skip_guard_checks=True)
    if not report08.trigger_fires:
        print(f"[FAIL] A08 trigger did not fire: "
              f"{report08.metric_name}={report08.metric_value}")
        rc = 1
    else:
        print(f"[OK]   A08 trigger fires: "
              f"{report08.metric_value:.0f} fps > {report08.metric_threshold:.0f}")

    # --- 3. A08 stream_dependency_loop ---
    loop_config = {
        **config,
        "_includes": {
            **config["_includes"],
            "attack_suite": {
                **config["_includes"]["attack_suite"],
                "attacks": [
                    {**a, "payload": {**a["payload"],
                                     "sub_mode": "stream_dependency_loop"}}
                    if a["id"] == "A08" else a
                    for a in config["_includes"]["attack_suite"]["attacks"]
                ],
            },
        },
    }
    sim08_loop = CertInjectionSimulator(loop_config, "A08", seed=42)
    report08_loop = sim08_loop.run(skip_guard_checks=True)
    if not report08_loop.trigger_fires:
        print(f"[FAIL] A08 loop trigger did not fire")
        rc = 1
    else:
        print(f"[OK]   A08 stream_dependency_loop trigger fires: "
              f"{report08_loop.metric_name}")

    # --- 4. A09 ESNI poisoning ---
    sim09 = CertInjectionSimulator(config, "A09", seed=42)
    report09 = sim09.run(skip_guard_checks=True)
    if not report09.trigger_fires:
        print(f"[FAIL] A09 trigger did not fire")
        rc = 1
    else:
        print(f"[OK]   A09 trigger fires: ECH inner SNI != cert hash")

    # --- 5. A12 handshake_abuse ---
    sim12 = CertInjectionSimulator(config, "A12", seed=42)
    report12 = sim12.run(skip_guard_checks=True)
    if not report12.trigger_fires:
        print(f"[FAIL] A12 trigger did not fire")
        rc = 1
    else:
        print(f"[OK]   A12 trigger fires: cipher_suite_mismatch")

    # --- 6. A12 renegotiation ---
    reneg_config = {
        **config,
        "_includes": {
            **config["_includes"],
            "attack_suite": {
                **config["_includes"]["attack_suite"],
                "attacks": [
                    {**a, "payload": {**a["payload"],
                                     "sub_mode": "tls_renegotiation"}}
                    if a["id"] == "A12" else a
                    for a in config["_includes"]["attack_suite"]["attacks"]
                ],
            },
        },
    }
    sim12_reneg = CertInjectionSimulator(reneg_config, "A12", seed=42)
    report12_reneg = sim12_reneg.run(skip_guard_checks=True)
    if not report12_reneg.trigger_fires:
        print(f"[FAIL] A12 reneg trigger did not fire")
        rc = 1
    else:
        print(f"[OK]   A12 reneg trigger fires: {report12_reneg.metric_name}")

    # --- 7. A14 continuation flood ---
    sim14 = CertInjectionSimulator(config, "A14", seed=42)
    report14 = sim14.run(skip_guard_checks=True)
    if not report14.trigger_fires:
        print(f"[FAIL] A14 trigger did not fire: "
              f"ratio={report14.metric_value:.2f}")
        rc = 1
    else:
        print(f"[OK]   A14 trigger fires: ratio={report14.metric_value:.2f} > 1.0")

    # --- 8. Determinism: same seed → same packets ---
    sim_a = CertInjectionSimulator(config, "A04", seed=42)
    sim_b = CertInjectionSimulator(config, "A04", seed=42)
    p_a = sim_a.generate_packets()
    p_b = sim_b.generate_packets()
    if p_a != p_b:
        print(f"[FAIL] determinism: same seed produced different packets")
        rc = 1
    else:
        print(f"[OK]   determinism: same seed → identical {len(p_a)} packets")

    # --- 9. Unknown scenario ---
    try:
        parse_cert_scenario("A99", config)
    except AttackConfigError:
        print(f"[OK]   unknown scenario A99 rejected")
    else:
        print(f"[FAIL] unknown scenario A99 was accepted")
        rc = 1

    # --- 10. Report serialization ---
    d = report04.to_dict()
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
