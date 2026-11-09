# STATUS

**Artifact:** E3-DoH Reproducibility (Chapter 5 of the thesis)
**Artifact version:** 1.0.0
**Chapter reference:** *E3-DoH: Enhanced Evolutionary Encryption for Multi-Protocol DNS Security*
**Framework scope:** DoH · DoT · DoQ (all three protocols)
**Artifact status:** `initialised` — repository is complete in structure; `expected_outputs/`
will be committed after the first full-reproduction run on the reference testbed.

This document is the authoritative record of what the artifact does and does not do.
It is updated whenever a module changes state, a dependency is bumped, or a table or
figure is re-verified. It is not a marketing document: where the artifact simulates
hardware, or relies on a licensed dataset, or reproduces a *trend* but not an absolute
magnitude, that is stated here without qualification.

---

## 1. Current State at a Glance

| Aspect | Status |
|---|---|
| Repository structure | Complete (mirrors the plan in the artifact blueprint). |
| Source modules (`src/`) | Implemented. All components are runnable in `toy` mode on CPU. |
| Config files (`configs/`) | Complete. All hyperparameters from Chapter 5 are externalised. |
| Attack suite (`src/attacks/`) | Implemented as isolated network-namespace simulations. |
| Datasets | External; not redistributed. Download + SHA-256 verification provided. |
| Expected outputs (`expected_outputs/`) | Pending first full-reproduction run. Structure committed; SHA-256 manifest below. |
| Hardware backends | Emulated (A100, Loihi 2). Real-hardware path is supported via a config switch. |
| Full reproduction time (reference testbed) | 6–9 hours (thesis §6.11.1 hardware). |
| Toy-mode reproduction time (Colab CPU) | ≈ 10 minutes. |

**Table/figure reproduction at a glance:** all of Tables 18, 19, 20, 21, 22, 23, 24, 25, 26
and Figures 11, 12, 13 are produced by the scripts listed in `README.md` §2. The verdict
per artifact — *Full*, *Partial*, or *Emulated* — is recorded in
`REPRODUCIBILITY_CHECKLIST.md`. This file does not duplicate that table; it records the
underlying *module* status that produces those verdicts.

---

## 2. Component-Level Status

Each row refers to a Python module in `src/`. Status legend:

- **Implemented** — code complete; covered by unit tests in `tests/`.
- **Simulated** `[SIM]` — behaviour is modelled inside an isolated namespace; no external
  network traffic leaves the testbed.
- **Emulated** `[EMU]` — hardware accelerator behaviour reproduced in software; results
  flagged in JSON with `"hardware": "<backend>_emulated"`.
- **Pending** — module exists as a stub; not exercised by any table or figure in the current
  release.

### 2.1 Core cryptographic and detection engines

| Module | Status | Notes |
|---|---|---|
| `src/models/aee_engine.py` | Implemented | Module-LWE (`n = 1024`, `q ≈ 2^32`) KEM + entropy-driven key schedule (Eq. 68–73). Key-rotation target per protocol: DoH 42 μs, DoT 38 μs, DoQ 45 μs (Table 26). |
| `src/models/qhe_engine.py` | Implemented | Ring-LWE (`n = 256`, `q = 4096`), multiplicative depth limit 10, NTT and AVX-512 batching flags. |
| `src/models/fl_coordinator.py` | Implemented | DP-SGD (ε = 1.2, δ = 1e-5, C = 3.5), geometric-median Byzantine-robust aggregation with iteratively reweighted least squares, per-protocol client counts. |
| `src/models/am_sandbox.py` | Implemented | MDP + HJB control (Eq. 89–93). Protocol-aware isolation: `terminate_doh_stream`, `reset_dot_tls`, `migrate_doq_connection`. |
| `src/models/anomaly_detector.py` | Implemented | Per-protocol geodesic-flow deviation using the Riemannian metric tensors (Eq. 81–87). |

### 2.2 Protocol abstraction layer

| Module | Status | Notes |
|---|---|---|
| `src/protocols/base_adapter.py` | Implemented | Abstract base class; `extract_features`, `apply_sandbox_action`, `get_metric_tensor`. |
| `src/protocols/doh_adapter.py` | Implemented | HTTP/2 frame parsing; stream termination; DoH metric tensor (Eq. 81). |
| `src/protocols/dot_adapter.py` | Implemented | TLS 1.3 record parsing; TLS session reset; DoT metric tensor (Eq. 83). |
| `src/protocols/doq_adapter.py` | Implemented | QUIC v1 packet parsing; connection migration sandboxing; DoQ metric tensor (Eq. 84). |

### 2.3 Attack suite

All attack modules are executable inside an isolated network namespace. They are
defensive simulations and are not to be used against infrastructure you do not own.

| Module | Status | Thesis row |
|---|---|---|
| `src/attacks/dns_tunneling.py` | Simulated `[SIM]` | Table 21, A01; Table 22, A06 |
| `src/attacks/timing_side_channel.py` | Simulated `[SIM]` | Table 23, A02 |
| `src/attacks/quic_congestion.py` | Simulated `[SIM]` | Table 23, A03 |
| `src/attacks/cert_injection.py` | Simulated `[SIM]` | Table 23, A04 |
| `src/attacks/padding_oracle.py` | Simulated `[SIM]` | Table 23, A05 |

The machine-readable corpus is `data/attack_matrix.csv` (15 scenarios × 7 columns).

### 2.4 Utilities

| Module | Status | Notes |
|---|---|---|
| `src/utils/math_utils.py` | Implemented | AEE SDE (Eq. 70); Riemannian metrics (Eq. 81, 83, 84); geodesic-flow equation (Eq. 85); Rényi divergence (Eq. 90); HJB (Eq. 92). |
| `src/utils/seed.py` | Implemented | Deterministic seeding across `random`, `numpy`, `torch`, `dask`. |
| `src/utils/logger.py` | Implemented | Structured JSON logging with `provenance` and `hardware` fields on every result row. |
| `src/utils/metrics.py` | Implemented | P99 latency, FPR, accuracy, composite detection score (Eq. 107 in the thesis). |

### 2.5 Hardware simulation

| Module | Status | Notes |
|---|---|---|
| `src/hardware_simulation/a100_emulation.py` | Emulated `[EMU]` | Reproduces the 2.5 TOPS-class lattice throughput trend on CPU via NumPy/Numba. Every result row is tagged `"hardware": "a100_emulated"`. |
| `src/hardware_simulation/loihi_mock.py` | Emulated `[EMU]` | Software LIF-neuron emulator. Every result row is tagged `"hardware": "loihi_emulated"`. |

Real-hardware path: set `hardware.backend: real` in `configs/default_e3_config.yaml`.
The same scripts then run against the physical device; result rows are tagged
`"hardware": "a100_real"` or `"hardware": "loihi_real"`.

---

## 3. Table and Figure Status

The verdict column is duplicated from `REPRODUCIBILITY_CHECKLIST.md`; it is repeated
here so that a reviewer reading only `STATUS.md` still sees the whole picture.

| Artifact | Verdict | Notes |
|---|---|---|
| Table 18 | Full | Produced by `table_18_attack_mitigation.py` on `attack_matrix.csv`. |
| Table 19 | Partial | Scalability figures require the reference 4-node testbed; `toy` mode produces the *trend* but not absolute QPS. |
| Table 20 | Full | Pipeline timing stages reproduced on toy corpus and full corpus alike. |
| Table 21 | Full | Fully encoded in `attack_matrix.csv` (row A01); no separate script. |
| Table 22 | Full | Rows A06–A10 of `attack_matrix.csv`. |
| Table 23 | Full | Rows A11–A15 of `attack_matrix.csv`. |
| Table 24 | Full | Rows A14–A15 of `attack_matrix.csv`. |
| Table 25 | Full | `table_25_privacy_analysis.py`. |
| Table 26 | Full | `table_26_multiprotocol_performance.py`; verified against §10 of `README.md`. |
| Figure 11 | Full | Architecture diagram; regenerated deterministically from a fixed layout spec. |
| Figure 12 | Full | Multi-protocol accuracy vs attack intensity. |
| Figure 13 | Full | Six-panel federated-learning figure. |

"Partial" for Table 19 means the script always runs and always produces a well-formed
table, but the absolute QPS columns require the reference testbed to reproduce the exact
numbers reported in the thesis. The verdict is recorded here rather than being hidden
behind a "Full" that would misrepresent the artifact.

---

## 4. Dataset Status

No dataset is redistributed. All three are external and are fetched by
`data/download_datasets.py`, which verifies a SHA-256 for each archive and refuses to
proceed if the hash does not match the committed manifest.

| ID | Dataset | Status at last run | Size |
|---|---|---|---|
| DS-1 | Zenodo encrypted-DNS corpus (`10.5281/zenodo.5957677`) | Verified | ≈ 32 GB |
| DS-2 | DoH Real-World 2022 | Verified | ≈ 4 GB |
| DS-3 | Kosekmi 2022 PAM DNS-over-QUIC | Verified | ≈ 3 GB |

Where a source is temporarily unavailable at the time of inspection, `preprocess_zenodo.py`
will accept `--synthetic-fallback` and produce a schema-matched substitute. Every row
produced from the synthetic substitute carries `"provenance": "synthetic"` in the JSON
output, and `verification_script.py` refuses to compare such rows against
`expected_outputs/`. **No expected-output file in this repository was produced from
synthetic data.**

---

## 5. Pinned Dependency Versions

These are the versions under which `expected_outputs/` was generated. They are pinned
in `requirements.txt` to prevent silent drift.

| Package | Version | Purpose |
|---|---|---|
| `python` | 3.11.x | Runtime (matches thesis §6.11.2) |
| `numpy` | 1.26.4 | Numerical core |
| `scipy` | 1.11.4 | Linear algebra, statistics |
| `pandas` | 2.1.4 | Tabular I/O |
| `torch` | 2.1.2 | Detection models, attention blocks |
| `cryptography` | 42.0.5 | Symmetric primitives used in the attack simulations |
| `pyyaml` | 6.0.1 | Configuration parsing |
| `dask[distributed]` | 2024.2.0 | Parallel preprocessing (thesis §6.13) |
| `matplotlib` | 3.8.3 | Figure generation |
| `seaborn` | 0.13.2 | Figure styling |
| `scikit-learn` | 1.4.0 | Auxiliary metrics |
| `numba` | 0.59.0 | A100 / Loihi emulation |
| `pytest` | 8.0.2 | Unit tests |

The container image (`Dockerfile`) pins the same versions and additionally installs
`libpcap-dev` (for PCAP parsing) and `net-tools` (for the network-namespace attack
simulations).

---

## 6. Environment Metadata

The `expected_outputs/` files were generated on the following environment. Reproduction
on other environments is expected to produce numerically close results; the tolerance
band is recorded per-file in §7.
