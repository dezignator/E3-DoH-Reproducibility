#!/usr/bin/env bash
# =============================================================================
# E3-DoH Reproducibility — Full Reproduction Driver
# =============================================================================
#
# Chapter 5 of the thesis: E3-DoH (Enhanced Evolutionary Encryption for
# Multi-Protocol DNS Security — DoH, DoT, DoQ).
#
# This shell script runs the entire reproduction pipeline in the
# correct order, from the dataset acquisition through the per-table
# and per-figure scripts, and finally through verification.
#
# USAGE
# -----
#     # Toy mode: synthetic corpus, fast (~10 min on CPU)
#     bash scripts/reproduce_all.sh --mode toy
#
#     # Full mode: DS-1 corpus, slow (6-9 hours on the reference testbed)
#     bash scripts/reproduce_all.sh --mode full
#
#     # Re-run only the table/figure scripts (assumes data is already
#     # preprocessed)
#     bash scripts/reproduce_all.sh --mode toy --skip-data
#
#     # Verify against the committed reference after the run
#     bash scripts/reproduce_all.sh --mode toy --verify
#
# OPTIONS
# -------
#     --mode {toy,full}     Run mode (default: toy)
#     --skip-data           Skip the download and preprocessing steps
#     --skip-train          Skip the training orchestrator (train_e3.py)
#     --skip-tables         Skip the per-table scripts
#     --skip-figures        Skip the per-figure scripts
#     --verify              Run verification_script.py --all after
#     --continue-on-error   Keep going if a step fails (default: stop)
#
# EXIT CODES
# ----------
#     0  all steps succeeded (and verification passed, if requested)
#     1  a step failed (--continue-on-error keeps going)
#     2  bad argument
#
# =============================================================================

set -u

# Colour output (green for OK, red for FAIL, yellow for skip).
if [ -t 1 ]; then
    OK="\033[0;32m"
    FAIL="\033[0;31m"
    SKIP="\033[0;33m"
    NC="\033[0m"
else
    OK=""
    FAIL=""
    SKIP=""
    NC=""
fi

# Repo root is the parent of the directory this script is in.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# Defaults.
MODE="toy"
SKIP_DATA=0
SKIP_TRAIN=0
SKIP_TABLES=0
SKIP_FIGURES=0
VERIFY=0
CONTINUE_ON_ERROR=0

# Parse arguments.
while [ $# -gt 0 ]; do
    case "$1" in
        --mode)               MODE="$2"; shift 2 ;;
        --skip-data)          SKIP_DATA=1; shift ;;
        --skip-train)         SKIP_TRAIN=1; shift ;;
        --skip-tables)        SKIP_TABLES=1; shift ;;
        --skip-figures)       SKIP_FIGURES=1; shift ;;
        --verify)             VERIFY=1; shift ;;
        --continue-on-error)  CONTINUE_ON_ERROR=1; shift ;;
        -h|--help)
            grep '^# ' "$0" | sed 's/^# \?//' | head -40
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            exit 2
            ;;
    esac
done

case "${MODE}" in
    toy|full) ;;
    *)
        echo "Invalid mode: ${MODE} (expected toy or full)" >&2
        exit 2
        ;;
esac

# Failures counter.
FAILED=0
TOTAL=0

run_step() {
    local name="$1"; shift
    local cmd=("$@")
    TOTAL=$((TOTAL + 1))
    echo ""
    echo "─── [${TOTAL}] ${name} ───"
    echo "    ${cmd[*]}"
    local start
    start=$(date +%s)
    if "${cmd[@]}"; then
        local elapsed
        elapsed=$(($(date +%s) - start))
        printf "${OK}[OK]${NC}   %s (%.1fs)\n" "${name}" "${elapsed}"
        return 0
    else
        local rc=$?
        printf "${FAIL}[FAIL]${NC} %s (exit %d)\n" "${name}" "${rc}"
        FAILED=$((FAILED + 1))
        if [ "${CONTINUE_ON_ERROR}" -eq 0 ]; then
            echo ""
            echo "Stopping on first failure. Re-run with --continue-on-error to keep going." >&2
            exit 1
        fi
        return ${rc}
    fi
}

skip_step() {
    local name="$1"
    TOTAL=$((TOTAL + 1))
    printf "${SKIP}[SKIP]${NC} %s\n" "${name}"
}

# ---- Data acquisition and preprocessing ---------------------------------
if [ "${SKIP_DATA}" -eq 1 ]; then
    skip_step "data/download_datasets.py"
    skip_step "data/preprocess_zenodo.py"
else
    run_step "data/download_datasets.py (mode=${MODE})" \
        python data/download_datasets.py --mode "${MODE}"
    run_step "data/preprocess_zenodo.py (mode=${MODE})" \
        python data/preprocess_zenodo.py --mode "${MODE}"
fi

# ---- Training orchestrator ----------------------------------------------
if [ "${SKIP_TRAIN}" -eq 1 ]; then
    skip_step "scripts/train_e3.py"
else
    run_step "scripts/train_e3.py (mode=${MODE})" \
        python scripts/train_e3.py --mode "${MODE}"
fi

# ---- Per-table scripts ---------------------------------------------------
if [ "${SKIP_TABLES}" -eq 1 ]; then
    skip_step "scripts/reproduce_tables/*.py"
else
    # Table 18 depends on the attack suite, which requires namespace
    # isolation. In toy mode, the scripts use --skip-guards via
    # default; in full mode, the reviewer must run inside a namespace
    # with CAP_NET_ADMIN.
    run_step "table 18 (attack mitigation)" \
        python scripts/reproduce_tables/table_18_attack_mitigation.py --mode "${MODE}"
    run_step "table 19 (scalability)" \
        python scripts/reproduce_tables/table_19_scalability.py --mode "${MODE}"
    run_step "table 20 (pipeline stages)" \
        python scripts/reproduce_tables/table_20_pipeline.py --mode "${MODE}" --protocol doh
    run_step "table 25 (privacy analysis)" \
        python scripts/reproduce_tables/table_25_privacy_analysis.py --mode "${MODE}"
    run_step "table 26 (multi-protocol performance)" \
        python scripts/reproduce_tables/table_26_multiprotocol_performance.py --mode "${MODE}"
fi

# ---- Per-figure scripts --------------------------------------------------
if [ "${SKIP_FIGURES}" -eq 1 ]; then
    skip_step "scripts/reproduce_figures/*.py"
else
    run_step "figure 11 (architecture)" \
        python scripts/reproduce_figures/figure_11_architecture.py
    run_step "figure 12 (detection performance)" \
        python scripts/reproduce_figures/figure_12_performance.py --mode "${MODE}"
    run_step "figure 13 (FL and privacy)" \
        python scripts/reproduce_figures/figure_13_federated_privacy.py --mode "${MODE}"
fi

# ---- Verification --------------------------------------------------------
if [ "${VERIFY}" -eq 1 ]; then
    run_step "verification_script.py --all" \
        python expected_outputs/verification_script.py --all
else
    skip_step "verification_script.py (use --verify to run)"
fi

# ---- Summary -------------------------------------------------------------
echo ""
echo "========================================"
echo "Reproduction summary (mode=${MODE}):"
echo "  Steps:     ${TOTAL}"
echo "  Failed:    ${FAILED}"
if [ "${FAILED}" -eq 0 ]; then
    echo "  Status:    ${OK}PASS${NC}"
    echo ""
    echo "Outputs are in: outputs/"
    echo "Reference:      expected_outputs/"
    echo "Run verification: python expected_outputs/verification_script.py --all"
    exit 0
else
    echo "  Status:    ${FAIL}FAIL${NC}"
    exit 1
fi
