#!/usr/bin/env bash
# Phase 0 gate: build the model inside the CUDA image, run one loss() step and one
# full-plot predict() on a synthetic plot (tests/gpu/test_smoke.py).
#
# Usage, on the GPU host, from anywhere inside the checkout:
#   docker/smoke.sh                       # image forestformer3d:cu118
#   FF3D_IMAGE=forestformer3d:dev docker/smoke.sh     # IMAGE= also still works
#   FF3D_GPU=3 docker/smoke.sh                        # which card to pin
#   FF3D_DOCKER=podman docker/smoke.sh                # rootless runtime
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# Through ff3d_docker rather than a hand-rolled `docker run`, for two reasons: it honours
# $FF3D_DOCKER, so this works under podman (which does not accept `--gpus`), and it pins
# ONE card instead of the `--gpus all` this used to pass, which reserves every GPU on a
# shared host even when the job needs one.
source "$REPO/benchmark/common.sh"
FF3D_IMAGE="${IMAGE:-$FF3D_IMAGE}"         # IMAGE= kept as the older spelling

ff3d_docker python -m pytest -m gpu tests/gpu/test_smoke.py -v -p no:cacheprovider
