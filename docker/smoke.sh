#!/usr/bin/env bash
# Phase 0 gate: build the model inside the CUDA image, run one loss() step and one
# full-plot predict() on a synthetic plot (tests/gpu/test_smoke.py).
#
# Usage, on the GPU host, from anywhere inside the checkout:
#   docker/smoke.sh                       # image forestformer3d:cu118
#   IMAGE=forestformer3d:dev docker/smoke.sh
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${IMAGE:-forestformer3d:cu118}"

exec docker run --rm --gpus all --shm-size=64g \
    -v "${REPO}:/workspace" \
    "${IMAGE}" \
    python -m pytest -m gpu tests/gpu/test_smoke.py -v -p no:cacheprovider
