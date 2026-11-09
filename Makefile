# =============================================================================
# E3-DoH Reproducibility — Makefile
# =============================================================================
#
# Chapter 5 of the thesis: E3-DoH (Enhanced Evolutionary Encryption for
# Multi-Protocol DNS Security — DoH, DoT, DoQ).
#
# Convenience targets for the most common reviewer / developer tasks.
# Every target is a thin wrapper around a command documented in
# README.md or STATUS.md; nothing here does anything the raw command
# would not.
#
# USAGE
# -----
#     make help           List every target
#     make install        Install runtime + dev dependencies + editable
#     make test           Run fast tests (skips slow)
#     make test-all       Run every test (requires --run-slow)
#     make lint           Lint with ruff
#     make format         Format with black + ruff --fix
#     make toy            Full toy pipeline (~10 min)
#     make full           Full pipeline (~6-9 h on reference testbed)
#     make verify         Run the main verification script
#     make verify-supp    Run the supplementary verification for Tables 21-24
#     make verify-all     Run both verification scripts
#     make manifest       Regenerate expected_outputs/MANIFEST.sha256
#     make clean          Remove outputs, features, synthetic data
#     make clean-all      Also remove virtualenv, caches, etc.
#     make schema-check   Validate the master config against configs/schema.json
#
# The `make toy` target is what a reviewer runs first after `make install`.
#
# =============================================================================

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

PYTHON ?= python
PIP    ?= pip
PYTEST ?= pytest
RUFF   ?= ruff
BLACK  ?= black

MODE ?= toy

# Colours (no-op if not on a TTY).
ifeq ($(shell test -t 1 && echo yes),yes)
    OK   := \033[0;32m
    WARN := \033[0;33m
    ERR  := \033[0;31m
    NC   := \033[0m
else
    OK   :=
    WARN :=
    ERR  :=
    NC   :=
endif

# -----------------------------------------------------------------------------
# Default target
# -----------------------------------------------------------------------------

.DEFAULT_GOAL := help

.PHONY: help
help:  ## List every target
	@echo "E3-DoH Reproducibility — available targets"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  $(OK)%-20s$(NC) %s\n", $$1, $$2}'
	@echo ""
	@echo "Variables:"
	@echo "  MODE=toy|full        (default: toy)"
	@echo "  PYTHON=python3.11    (default: python)"

# -----------------------------------------------------------------------------
# Setup
# -----------------------------------------------------------------------------

.PHONY: install
install:  ## Install runtime + dev dependencies and the artifact (editable)
	$(PIP) install -r requirements.txt
	$(PIP) install -r requirements-dev.txt
	$(PIP) install -e .
	@echo -e "$(OK)Install complete$(NC)"

.PHONY: install-runtime
install-runtime:  ## Install only runtime dependencies + the artifact
	$(PIP) install -r requirements.txt
	$(PIP) install -e .
	@echo -e "$(OK)Runtime install complete$(NC)"

.PHONY: check-setup
check-setup:  ## Run setup.py's --check-only coherence checks
	$(PYTHON) setup.py --check-only

# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------

.PHONY: test
test:  ## Run fast tests (slow tests are skipped)
	$(PYTEST) tests/ -v -m "not slow"

.PHONY: test-all
test-all:  ## Run every test, including slow (requires the toy dataset)
	$(PYTEST) tests/ -v --run-slow

.PHONY: test-math
test-math:  ## Only the math_utils test suite
	$(PYTEST) tests/test_math_utils.py -v

.PHONY: test-protocols
test-protocols:  ## Only the protocol adapter test suite
	$(PYTEST) tests/test_protocols.py -v

.PHONY: coverage
coverage:  ## Run tests with coverage and produce a report
	$(PYTEST) tests/ --cov=src --cov=data --cov-report=term-missing --cov-report=html

# -----------------------------------------------------------------------------
# Lint and format
# -----------------------------------------------------------------------------

.PHONY: lint
lint:  ## Lint with ruff
	$(RUFF) check src/ data/ scripts/ tests/

.PHONY: format
format:  ## Format with black then fix with ruff
	$(BLACK) src/ data/ scripts/ tests/
	$(RUFF) check --fix src/ data/ scripts/ tests/

.PHONY: typecheck
typecheck:  ## Type-check with mypy (src/ and scripts/)
	mypy src/ scripts/

# -----------------------------------------------------------------------------
# Reproduction
# -----------------------------------------------------------------------------

.PHONY: toy
toy:  ## Run the full toy pipeline (synthetic corpus, ~10 min)
	MODE=toy bash scripts/reproduce_all.sh --mode toy
	@echo -e "$(OK)Toy pipeline complete. Run 'make verify' next.$(NC)"

.PHONY: full
full:  ## Run the full pipeline on the reference testbed (~6-9 h)
	MODE=full bash scripts/reproduce_all.sh --mode full
	@echo -e "$(OK)Full pipeline complete. Run 'make verify-all' next.$(NC)"

.PHONY: data
data:  ## Download + preprocess datasets only (no training, no tables)
	$(PYTHON) data/download_datasets.py --mode $(MODE)
	$(PYTHON) data/preprocess_zenodo.py --mode $(MODE)

.PHONY: train
train:  ## Run just the training orchestrator
	$(PYTHON) scripts/train_e3.py --mode $(MODE)

.PHONY: attacks
attacks:  ## Run the attack suite (toy mode, guards skipped)
	$(PYTHON) scripts/run_attack_suite.py --all --skip-guards

# -----------------------------------------------------------------------------
# Verification
# -----------------------------------------------------------------------------

.PHONY: verify
verify:  ## Run the main verification script
	$(PYTHON) expected_outputs/verification_script.py --all

.PHONY: verify-supp
verify-supp:  ## Run the supplementary verification for Tables 21-24
	$(PYTHON) expected_outputs/verify_supplementary.py --all

.PHONY: verify-all
verify-all: verify verify-supp  ## Run both verification scripts
	@echo -e "$(OK)All verifications complete$(NC)"

# -----------------------------------------------------------------------------
# Manifest and schema
# -----------------------------------------------------------------------------

.PHONY: manifest
manifest:  ## Regenerate expected_outputs/MANIFEST.sha256
	$(PYTHON) scripts/generate_manifest.py \
		--outputs expected_outputs/ \
		--manifest expected_outputs/MANIFEST.sha256 \
		--verbose

.PHONY: schema-check
schema-check:  ## Validate the master config against configs/schema.json
	$(PYTHON) -c "import json, yaml, sys; \
		cfg = yaml.safe_load(open('configs/default_e3_config.yaml')); \
		schema = json.load(open('configs/schema.json')); \
		import jsonschema; \
		jsonschema.validate(cfg, schema); \
		print('config validates against schema.json')"

# -----------------------------------------------------------------------------
# Cleanup
# -----------------------------------------------------------------------------

.PHONY: clean
clean:  ## Remove outputs, features, synthetic data (keeps .git)
	rm -rf outputs/tables/* outputs/figures/* outputs/logs/* outputs/checkpoints/* outputs/attacks/* 2>/dev/null || true
	rm -rf data/processed/features/* 2>/dev/null || true
	rm -rf data/synthetic/* 2>/dev/null || true
	rm -rf artifacts/* 2>/dev/null || true
	@echo -e "$(OK)Cleaned outputs/, data/processed/features/, data/synthetic/, artifacts/$(NC)"

.PHONY: clean-all
clean-all: clean  ## Also remove .venv, caches, and the .git directory is preserved
	rm -rf .venv venv 2>/dev/null || true
	rm -rf .pytest_cache .ruff_cache .mypy_cache .numba_cache 2>/dev/null || true
	rm -rf htmlcov .coverage .coverage.* coverage.xml 2>/dev/null || true
	rm -rf **/__pycache__ 2>/dev/null || true
	rm -f expected_outputs/MANIFEST.sha256 2>/dev/null || true
	@echo -e "$(WARN)Removed virtualenv, caches, coverage, and MANIFEST.sha256$(NC)"
	@echo -e "$(WARN)Restore MANIFEST.sha256 with 'make manifest' before committing$(NC)"

# -----------------------------------------------------------------------------
# Diagnostics
# -----------------------------------------------------------------------------

.PHONY: state
state:  ## Print the artifact's coherence state (from data_utils self-test)
	$(PYTHON) data/data_utils.py

.PHONY: info
info:  ## Print environment info (Python, dependency versions, hardware)
	@$(PYTHON) -c "import sys, platform, importlib.metadata as m; \
		print('Python:', sys.version.split()[0]); \
		print('Platform:', platform.platform()); \
		pkgs = ['numpy', 'scipy', 'pandas', 'torch', 'scikit-learn', 'opacus', 'scapy', 'dask']; \
		print('Packages:'); \
		[print(f'  {p}=={m.version(p)}') if _safe_version(p, m) else print(f'  {p} (not installed)') for p in pkgs]"

# Helper used by `info` — defined in Python because Make cannot inline it cleanly.
define _safe_version_check
python -c "import importlib.metadata as m; m.version('$(1)')" >/dev/null 2>&1 && echo yes || echo no
endef
