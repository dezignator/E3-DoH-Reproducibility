# E3-DoH Reproducibility Artifact

**Enhanced Evolutionary Encryption for Multi-Protocol DNS Security (DoH, DoT, DoQ)**

This repository is the runnable reproducibility artifact for **Chapter 5** of the thesis
*"Next-Generation Secure Encrypted DNS Architectures: Four Complementary Frameworks for
Adaptive, Privacy-Preserving, and Quantum-Resilient Defense."*

The artifact is scoped strictly to the **E3-DoH framework**. It does not contain, and does
not claim, results from MIRAGE-DoH (Chapter 3), PAFA-DoH (Chapter 4), or A³E-DNS (Chapter 6).
Each of those frameworks has its own separate artifact.

---

## 1. What This Artifact Reproduces

The E3-DoH framework combines four components over three encrypted DNS protocols:

| Component | Full Name | Role |
|---|---|---|
| **AEE** | Adaptive Evolutionary Encryption | Module-LWE key encapsulation whose key schedule is evolved by a stochastic differential equation driven by traffic entropy (Eq. 68–73). |
| **QHE** | Quantum-Safe Homomorphic Encryption | Ring-LWE (n = 256, q = 4096) evaluation of anomaly-detection statistics on ciphertext without decryption (Eq. 74–81). |
| **MP-FL** | Multi-Protocol Federated Learning | Per-protocol local models, DP-SGD (ε = 1.2), geometric-median Byzantine-robust aggregation (Eq. 88). |
| **AMPS** | Adaptive Multi-Protocol Sandboxing | MDP + Hamilton-Jacobi-Bellman control; protocol-aware isolation (DoH stream termination, DoT TLS reset, DoQ connection migration); triggered by KL / Rényi divergence (Eq. 89–93). |

A common **protocol-abstraction layer** (Section 5.1) normalises DoH (HTTP/2 or HTTP/3),
DoT (TLS 1.3), and DoQ (QUIC v1) into a single metadata representation so that the same
downstream detection and cryptographic layers can operate on all three.

---

## 2. Table and Figure Map

Every table and figure reported in Chapter 5 is produced by a named script. The mapping is
authoritative — no table is hand-edited, and no number is entered manually.

| Thesis artifact | Content | Script | Expected output |
|---|---|---|---|
| **Table 18** | Attack injection methods and detection mechanisms (5 vectors) | `scripts/reproduce_tables/table_18_attack_mitigation.py` | `expected_outputs/table_18_expected.json` |
| **Table 19** | Scalability analysis under load (1M–12M QPS target) | `scripts/reproduce_tables/table_19_scalability.py` | `expected_outputs/table_19_expected.json` |
| **Table 20** | Real-time attack processing pipeline (capture → mitigation) | `scripts/reproduce_tables/table_20_pipeline.py` | `expected_outputs/table_20_expected.json` |
| **Table 21** | dns2tcp exfiltration parameter configuration | `data/attack_matrix.csv` (row A01) | embedded in matrix |
| **Table 22** | Advanced attack scenarios with E3-DoH response | `data/attack_matrix.csv` (rows A06–A10) | embedded in matrix |
| **Table 23** | Network attack injection commands | `data/attack_matrix.csv` (rows A11–A15) | embedded in matrix |
| **Table 24** | Attack injection techniques and defences | `data/attack_matrix.csv` (rows A14–A15 + schema) | embedded in matrix |
| **Table 25** | Privacy-preserving FL configurations (ε / utility / robustness / cost) | `scripts/reproduce_tables/table_25_privacy_analysis.py` | `expected_outputs/table_25_expected.json` |
| **Table 26** | Multi-protocol performance comparison (DoH / DoT / DoQ) | `scripts/reproduce_tables/table_26_multiprotocol_performance.py` | `expected_outputs/table_26_expected.json` |
| **Figure 11** | E3-DoH architecture and data-flow diagram | `scripts/reproduce_figures/figure_11_architecture.py` | `expected_outputs/figure_11_expected.svg` |
| **Figure 12** | Detection accuracy vs attack intensity (3 protocol panels) | `scripts/reproduce_figures/figure_12_performance.py` | `expected_outputs/figure_12_expected.png` |
| **Figure 13** | Federated learning: convergence, DP trade-off, Byzantine robustness, scalability, aggregation, privacy-under-attack (6 panels) | `scripts/reproduce_figures/figure_13_federated_privacy.py` | `expected_outputs/figure_13_expected.png` |

The complete attack scenario corpus (Tables 18, 21, 22, 23, 24 distilled into one machine-readable
table) is stored in **`data/attack_matrix.csv`**. This is the "huge tabular data" the thesis refers
to: 15 distinct attack scenarios × 7 columns (Attack_ID, Attack_Vector, Protocol, Tool,
Command_Syntax, Detection_Trigger, Mitigation_Action).

---

## 3. Repository Layout
