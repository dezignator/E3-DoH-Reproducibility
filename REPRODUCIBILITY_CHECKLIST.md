# REPRODUCIBILITY_CHECKLIST

**Artifact:** E3-DoH Reproducibility (Chapter 5 of the thesis)
**Artifact version:** 1.0.0
**Reviewer scope:** Chapter 5 only — Tables 18, 19, 20, 21, 22, 23, 24, 25, 26;
Figures 11, 12, 13.

This file records, for every table and every figure in Chapter 5, the **formal verdict**
on whether this artifact reproduces it, the **exact procedure** to reproduce it, the
**tolerance** within which a fresh run is considered a match, and the **estimated time**
of that run in `toy` and `full` mode.

A reviewer should be able to read this file top-to-bottom, run the commands, and know
exactly what conclusion is licensed. The verdicts are not all "Full" — a checklist that
says "Full" on every line and never says "Partial" is not a checklist, it is a
marketing document. This one distinguishes.

---

## 1. Verdict Legend

Every table and figure carries exactly one verdict, from a fixed set of four.

| Verdict | Meaning |
|---|---|
| **Full** | Reproduced end-to-end from committed data, configuration, and code. A fresh run on the reference environment reproduces the thesis numbers within the stated tolerance. |
| **Partial** | The script runs end-to-end and reproduces the *trend* on any environment, but the *absolute magnitude* of the reported number requires the reference testbed or the licensed/embargoed dataset. The reason is recorded in the per-artifact section. |
| **Emulated** `[EMU]` | The underlying hardware accelerator (NVIDIA A100 or Intel Loihi 2) is emulated in software. A fresh run reproduces the reported values within the stated tolerance on the emulated backend. The real-hardware path exists and is supported; it has not been re-run to produce a second `expected_outputs/` set. |
| **Simulated** `[SIM]` | The behaviour being measured is a network-isolated simulation (attack injection, adaptive sandbox isolation). The simulation is the intended measurement — no external traffic leaves the testbed. |

A table that is **Emulated** is still reproduced — it is a stronger statement than
**Partial**, because the emulated result *is* the reported number. A table that is
**Partial** reports the reported number but a fresh run may deviate from it for reasons
outside the artifact's control (testbed hardware, licence terms).

---

## 2. Summary Table

| Artifact | Verdict | Reference file | Tolerance | Toy time | Full time |
|---|---|---|---|---|---|
| Table 18 | **Full** | `table_18_expected.json` | exact | < 1 min | ≈ 12 min |
| Table 19 | **Partial** | `table_19_expected.json` | trend only (toy); exact (full) | < 1 min | ≈ 90 min |
| Table 20 | **Full** | `table_20_expected.json` | ± 5 % per stage | < 1 min | ≈ 20 min |
| Table 21 | **Full** | `data/attack_matrix.csv` (A01) | schema + value | < 30 s | < 30 s |
| Table 22 | **Full** | `data/attack_matrix.csv` (A06–A10) | schema + value | < 30 s | < 30 s |
| Table 23 | **Full** | `data/attack_matrix.csv` (A11–A15) | schema + value | < 30 s | < 30 s |
| Table 24 | **Full** | `data/attack_matrix.csv` (A14–A15) | schema + value | < 30 s | < 30 s |
| Table 25 | **Full** | `table_25_expected.json` | ± 0.05 pp (accuracy); exact (ε) | < 2 min | ≈ 45 min |
| Table 26 | **Full** | `table_26_expected.json` | per-column (see §5) | < 3 min | ≈ 2 h |
| Figure 11 | **Full** | `figure_11_expected.svg` | deterministic | < 10 s | < 10 s |
| Figure 12 | **Full** | `figure_12_expected.png` | ± 0.3 pp per panel | < 3 min | ≈ 40 min |
| Figure 13 | **Full** | `figure_13_expected.png` | ± 0.5 pp per panel | < 5 min | ≈ 60 min |

Times are measured on the reference environment described in `STATUS.md` §6.
"Toy" means `--mode toy` on the reference environment; "full" means `--mode full`
with the three datasets present and verified.

---

## 3. Global Procedure

The commands below are the same for every artifact, so they are stated once here rather
than repeated 12 times.

### 3.1 One-time setup

```bash
git clone https://github.com/<your-org>/E3-DoH-Reproducibility.git
cd E3-DoH-Reproducibility
pip install -r requirements.txt
python setup.py develop
