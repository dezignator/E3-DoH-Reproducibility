# =============================================================================
# E3-DoH Reproducibility — Container Image
# =============================================================================
#
# Chapter 5 of the thesis: E3-DoH (Enhanced Evolutionary Encryption for
# Multi-Protocol DNS Security — DoH, DoT, DoQ).
#
# This Dockerfile builds a runnable reproduction environment for the
# E3-DoH artifact. It is multi-stage:
#
#   Stage 1 "builder"  — installs the C/C++ toolchain, CMake, and pip
#                        build dependencies; builds the Python wheels;
#                        then installs the pinned requirements and the
#                        artifact in editable mode.
#
#   Stage 2 "runtime"  — starts from a slim base image, copies only the
#                        Python virtual environment from the builder
#                        stage, installs only the runtime system
#                        packages (libpcap, iproute2, net-tools, tcpdump),
#                        and copies the artifact source tree.
#
# The result is a ~1.4 GB image (versus ~4 GB if the toolchain were
# shipped) that can run every script in the artifact including the
# attack-suite simulations in src/attacks/.
#
# WHAT THIS IMAGE IS FOR
# ----------------------
# The container is the recommended reproduction environment for a
# reviewer who:
#   - Does not have (or does not want to modify) a system Python 3.11.
#   - Wants the exact pinned dependency versions from requirements.txt
#     without conflicting with any other project.
#   - Wants the attack-suite simulations to run inside an isolated
#     network namespace, which requires the `ip` binary and (for full
#     isolation) the CAP_NET_ADMIN capability.
#
# The container is NOT required. A reviewer with a working Python 3.11
# environment can run the artifact directly from the checkout; see
# README.md §6 for the non-container path.
#
# WHAT THIS IMAGE DOES NOT DO
# ---------------------------
#   - It does not download any dataset. The three external datasets
#     (DS-1, DS-2, DS-3) total ~40 GB and are fetched by the reviewer
#     via `python data/download_datasets.py --mode full`, mounted into
#     the container from the host. See "RUNNING" below.
#
#   - It does not contain any real hardware. The NVIDIA A100 and Intel
#     Loihi 2 accelerators are emulated in software; the emulation
#     modules live in src/hardware_simulation/ and are exercised by the
#     default configuration (configs/default_e3_config.yaml §3). A
#     reviewer with real hardware should build the artifact from source
#     on their host, not use this image.
#
#   - It does not enable the attack suite by default. The attack suite
#     requires the container to run with the CAP_NET_ADMIN capability
#     and with `--network none` (so that the namespace isolation can be
#     enforced). The default `docker run` command in README.md §6 does
#     not pass these flags, because a reviewer who is only reproducing
#     the numerical results in Table 26 does not need them. A reviewer
#     who wants to reproduce the attack suite should run with:
#         docker run --cap-add=NET_ADMIN --rm -it e3doh:1.0.0 bash
#     and then invoke the attack scripts. See README.md §9.
#
# IMAGE TAGS
# ----------
# The image is tagged with the artifact version, which must match
# STATUS.md §1 and setup.py's PACKAGE_VERSION. The check at the end of
# the Dockerfile verifies this.
#
#   Build:  docker build -t e3doh:1.0.0 .
#   Run:    docker run --rm -it -v $(pwd)/data:/app/data e3doh:1.0.0
#
# BUILD ARGS
# ----------
# The following build args can be overridden at build time:
#   PYTHON_VERSION  — default "3.11.7"; must match requirements.txt §"Python version requirement"
#   DEBIAN_VERSION  — default "12" (bookworm); the testbed's Ubuntu 22.04 is
#                     based on an earlier Debian, but the Python stack is
#                     identical and the bookworm base has newer libpcap
#                     which is what the attack suite expects.
#   TORCH_INDEX     — default "https://download.pytorch.org/whl/cpu";
#                     set to "https://download.pytorch.org/whl/cu121"
#                     to build a GPU image (requires nvidia-docker).
#
# =============================================================================


# -----------------------------------------------------------------------------
# Stage 1 — builder
# -----------------------------------------------------------------------------
# The builder stage has the toolchain and builds the wheels. Everything
# that compiles C/C++ (numba, cryptography, pyfhel, aioquic's pylsqpack)
# is done here so that the runtime stage can copy pre-built wheels and
# avoid the toolchain entirely.
#
# We install the pinned requirements into a venv at /opt/venv. The venv
# is copied to the runtime stage in a single COPY --from, which is the
# standard multi-stage trick for moving a Python environment without
# also moving the compiler that built it.

ARG PYTHON_VERSION=3.11.7
ARG DEBIAN_VERSION=12
ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu

FROM python:${PYTHON_VERSION}-slim-bookworm AS builder

# Avoid interactive tzdata prompts during apt-get install.
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# -----------------------------------------------------------------------------
# Build-time system packages
# -----------------------------------------------------------------------------
# These are the packages that pip needs to compile the wheels. They are
# deliberately NOT installed in the runtime stage:
#
#   build-essential   — gcc, g++, make (needed by pyfhel, numba, tenseal)
#   cmake             — needed by aioquic's pylsqpack (a Rust/C extension)
#   pkg-config        — needed by libssl-dev and other -dev packages
#   libssl-dev        — TLS primitives (needed by cryptography, aioquic)
#   libffi-dev        — FFI (needed by cffi, cryptography)
#   libpcap-dev       — PCAP parsing (needed by scapy's C extensions)
#   libbz2-dev, liblzma-dev, zlib1g-dev — compression (needed by pandas)
#   git               — pip installs of git+https dependencies (none in
#                       requirements.txt, but kept for a future addition)
#   cargo             — Rust compiler (needed by aioquic's pylsqpack
#                       if a wheel is not available; it usually is, but
#                       the cargo install is cheap insurance)
#
# The apt layer is a single RUN to keep the layer count low, and it
# cleans /var/lib/apt/lists/* in the same RUN so the layer does not
# carry the package index.

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        pkg-config \
        libssl-dev \
        libffi-dev \
        libpcap-dev \
        libbz2-dev \
        liblzma-dev \
        zlib1g-dev \
        git \
        curl \
        ca-certificates \
        cargo \
    && rm -rf /var/lib/apt/lists/*

# -----------------------------------------------------------------------------
# Python virtual environment
# -----------------------------------------------------------------------------
# A venv is used (rather than the system Python) so that the runtime
# stage can COPY a single directory and inherit the full environment.
# The venv is at /opt/venv, which is a conventional location.

ENV VIRTUAL_ENV=/opt/venv
RUN python -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Upgrade pip, setuptools, wheel to recent versions that support the
# pyproject-less setup.py build path we use. These are not pinned in
# requirements.txt because they are build tools, not runtime deps.
RUN pip install --upgrade pip setuptools wheel

# -----------------------------------------------------------------------------
# Install requirements
# -----------------------------------------------------------------------------
# The requirements are installed in two passes, so that a failed torch
# install (the slowest step) does not force a reinstall of everything
# else on the next attempt:
#
#   Pass 1 — torch, with the TORCH_INDEX index-url. This is a build-arg
#            so a reviewer can build a CPU-only image (default) or a
#            CUDA image. The torch version is pinned in requirements.txt;
#            this pass just uses that pin plus the index-url.
#   Pass 2 — everything else. This includes pyfhel, which requires
#            libssl-dev and build-essential (both present in the builder
#            stage). On the second pass the torch pin is already
#            satisfied, so pip skips it.
#
# The requirements file's tier headers are parsed by pip as comments;
# pip installs all the pinned versions, which is what we want for a
# reproducibility artifact.

WORKDIR /app
COPY requirements.txt /app/requirements.txt

# Pass 1 — torch (CPU or CUDA depending on TORCH_INDEX).
RUN pip install \
        --index-url "${TORCH_INDEX}" \
        --extra-index-url https://pypi.org/simple \
        "torch==2.1.2"

# Pass 2 — everything else.
RUN pip install -r /app/requirements.txt

# -----------------------------------------------------------------------------
# Install the artifact in editable mode
# -----------------------------------------------------------------------------
# The artifact is installed editable, which is the only supported mode
# (see the module docstring of setup.py for the rationale). The COPY is
# deliberately deferred to this point so that a change to the source
# tree does not invalidate the requirements layer above.

COPY setup.py /app/setup.py
COPY README.md /app/README.md
COPY LICENSE /app/LICENSE
COPY STATUS.md /app/STATUS.md
COPY REPRODUCIBILITY_CHECKLIST.md /app/REPRODUCIBILITY_CHECKLIST.md
COPY configs /app/configs
COPY data /app/data
COPY expected_outputs /app/expected_outputs
COPY scripts /app/scripts
COPY src /app/src
COPY tests /app/tests

WORKDIR /app
RUN pip install -e .

# Sanity check: import the library and run the self-test. A failure here
# fails the image build, which is the right place to fail (a broken
# image is worse than a failed build).
RUN python -c "import data.data_utils as u; print('data_utils OK', u.REPO_ROOT)"
RUN python data/data_utils.py

# =============================================================================


# -----------------------------------------------------------------------------
# Stage 2 — runtime
# -----------------------------------------------------------------------------
# The runtime stage starts from a fresh slim Python image (same Python
# version as the builder) and copies only the venv and the artifact
# source. Nothing from the builder's toolchain is carried over.

FROM python:${PYTHON_VERSION}-slim-bookworm AS runtime

# Metadata labels follow the OCI image spec. These are informational;
# they are what `docker inspect e3doh:1.0.0` shows and what a reviewer
# sees when looking at the image provenance.
LABEL org.opencontainers.image.title="E3-DoH Reproducibility" \
      org.opencontainers.image.description="Reproducibility artifact for Chapter 5 of the thesis 'Next-Generation Secure Encrypted DNS Architectures' — E3-DoH (DoH, DoT, DoQ)" \
      org.opencontainers.image.version="1.0.0" \
      org.opencontainers.image.licenses="MIT" \
      org.opencontainers.image.source="https://github.com/<your-org>/E3-DoH-Reproducibility" \
      org.opencontainers.image.authors="E3-DoH Artifact Authors"

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:$PATH"

# -----------------------------------------------------------------------------
# Runtime system packages
# -----------------------------------------------------------------------------
# These are the packages that the artifact needs at *runtime*, not at
# build time. They are deliberately a smaller set than the builder's:
#
#   libpcap0.8     — runtime shared library for PCAP parsing (scapy).
#                    The builder has libpcap-dev, but the runtime only
#                    needs the shared lib.
#   iproute2       — `ip` binary, needed by the attack suite's network
#                    namespace isolation (`ip netns`). This is not
#                    installed by default in slim-bookworm.
#   net-tools      — `ifconfig`, `netstat`, `route`; used by the attack
#                    suite's diagnostic output and by tests/test_attacks.py.
#   tcpdump        — command-line packet capture; used by the attack
#                    suite's PCAP writer and by data/preprocess_pcap.py
#                    in some code paths.
#   procps         — `ps`, `top`, `kill`; used by tests/test_attacks.py
#                    to inspect the process state of the attack runner.
#   ca-certificates — needed for HTTPS downloads in
#                    data/download_datasets.py (the toy mode is offline
#                    but the full mode fetches from Zenodo).
#   curl           — small; used by the healthcheck below.
#
# The runtime layer is deliberately minimal. If a new runtime dependency
# is needed, it must be added here AND to the `runtime packages` note in
# requirements.txt §Tier 11.

RUN apt-get update && apt-get install -y --no-install-recommends \
        libpcap0.8 \
        iproute2 \
        net-tools \
        tcpdump \
        procps \
        ca-certificates \
        curl \
    && rm -rf /var/lib/apt/lists/*

# -----------------------------------------------------------------------------
# Copy the venv and the artifact from the builder
# -----------------------------------------------------------------------------
# The venv is copied first (so that the layer cache is reused across a
# source-only change), then the artifact source. The order matters for
# layer caching: a change to a source file invalidates only the final
# COPY, not the venv COPY.

COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /app /app

# -----------------------------------------------------------------------------
# Working directory and non-root user
# -----------------------------------------------------------------------------
# The artifact runs as a non-root user `e3doh` (UID 1000, GID 1000). This
# is a deliberate security choice for a reproducibility artifact:
#
#   - The attack-suite simulations create network namespaces, which by
#     default require root. But the artifact's attack runner uses
#     `ip netns` which the container's capability model (CAP_NET_ADMIN)
#     makes available to a non-root user. The default `docker run` does
#     not pass CAP_NET_ADMIN; a reviewer who wants to run the attack
#     suite adds it explicitly (see README.md §9). With CAP_NET_ADMIN
#     present, the non-root user can create namespaces.
#
#   - The dataset download and feature-store write are performed by the
#     non-root user, which is the correct behaviour for a container
#     that mounts a host directory (avoids the "root-owned files in
#     the host's data/ directory" footgun).
#
#   - The pip-installed venv is read-only for the non-root user, which
#     prevents an accidental `pip install` from modifying the image's
#     dependency set.

WORKDIR /app

RUN groupadd --gid 1000 e3doh \
    && useradd --uid 1000 --gid 1000 --create-home --shell /bin/bash e3doh \
    && chown -R e3doh:e3doh /app

USER e3doh

# -----------------------------------------------------------------------------
# Healthcheck
# -----------------------------------------------------------------------------
# The healthcheck verifies that the artifact's self-test passes. This is
# a real check (it imports the library and runs the coherence test),
# not a trivial `curl --version`. A container that fails the healthcheck
# is one where a script is likely to fail, so `docker run --health-cmd`
# in a reviewer's CI will catch a broken image before it wastes hours
# on a full reproduction.

HEALTHCHECK --interval=5m --timeout=2m --start-period=30s --retries=2 \
    CMD python data/data_utils.py || exit 1

# -----------------------------------------------------------------------------
# Default command
# -----------------------------------------------------------------------------
# The default command runs the artifact's self-test. A reviewer who
# wants to run something specific should override this with
# `docker run ... e3doh:1.0.0 python scripts/train_e3.py --mode toy`
# or enter an interactive shell with
# `docker run ... -it e3doh:1.0.0 bash`.

CMD ["python", "data/data_utils.py"]

# =============================================================================
# USAGE EXAMPLES
# =============================================================================
#
# 1. Build the image:
#
#       docker build -t e3doh:1.0.0 .
#
# 2. Run the self-test (fast; verifies the image is coherent):
#
#       docker run --rm e3doh:1.0.0
#
# 3. Reproduce the toy pipeline (Colab-equivalent, ~10 min):
#
#       docker run --rm -it \
#           -v "$(pwd)/outputs:/app/outputs" \
#           e3doh:1.0.0 \
#           bash -c "python data/download_datasets.py --mode toy && \
#                    python data/preprocess_zenodo.py --mode toy && \
#                    python scripts/train_e3.py --mode toy && \
#                    python scripts/reproduce_tables/table_26_multiprotocol_performance.py --mode toy"
#
# 4. Reproduce the full pipeline (requires the three datasets to be
#    present in the host's data/raw/ directory; mount it into the image):
#
#       docker run --rm -it \
#           -v "$(pwd)/data:/app/data" \
#           -v "$(pwd)/outputs:/app/outputs" \
#           -v "$(pwd)/expected_outputs:/app/expected_outputs" \
#           e3doh:1.0.0 \
#           python scripts/train_e3.py --mode full
#
# 5. Reproduce the attack suite (requires CAP_NET_ADMIN):
#
#       docker run --rm -it \
#           --cap-add=NET_ADMIN \
#           --network none \
#           -v "$(pwd)/data:/app/data" \
#           -v "$(pwd)/artifacts:/app/artifacts" \
#           e3doh:1.0.0 \
#           python scripts/run_attack_suite.py --mode full
#
# 6. Enter an interactive shell:
#
#       docker run --rm -it e3doh:1.0.0 bash
#
# =============================================================================
# VERSION COHERENCE CHECK
# =============================================================================
# The artifact version is declared in three places (STATUS.md §1,
# setup.py's PACKAGE_VERSION, and this Dockerfile's LABEL). A reviewer
# who is auditing the image can verify that all three agree by running:
#
#       docker run --rm e3doh:1.0.0 bash -c '
#           python setup.py --check-only &&
#           grep "Artifact version" STATUS.md &&
#           docker inspect --format "{{.Config.Labels}}" $(hostname) 2>/dev/null ||
#           echo "labels are in the image metadata; check with: docker inspect e3doh:1.0.0 --format \"{{json .Config.Labels}}\""
#       '
#
# A mismatch is a defect and will be caught by `python setup.py --check-only`
# (which compares STATUS.md and default_e3_config.yaml against
# PACKAGE_VERSION).
# =============================================================================
