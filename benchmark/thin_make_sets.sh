#!/usr/bin/env bash
# Thin the labelled ForAINetV2 test plots to an airborne-laser density.
#
#   bash benchmark/thin_make_sets.sh                 # 25 pts/m2, both modes, seed 0
#   FF3D_THIN_DENSITY=10 bash benchmark/thin_make_sets.sh
#   FF3D_DRY_RUN=1 bash benchmark/thin_make_sets.sh  # print the docker commands only
#
# Produces, for each mode (uniform -> letter u, canopy -> letter c):
#   data/ForAINetV2/test_data_thin<D><letter>/<stem>_thin<D><letter>.ply   (28 files)
#   work_dirs/logs/thin/<mode>/scan_list.txt                              (the 28 new stems)
#
# It never touches data/ForAINetV2/meta_data/test_list.txt: the thinned stems go into a
# private scan list under work_dirs/, which benchmark/thin_eval.sh then passes to
# batch_load/create_data/tools/test.py.
#
# No GPU is needed (thin_plots.py is numpy + plyfile + shapely), so this uses
# `--entrypoint python` to bypass the image's entrypoint, which asserts CUDA.
# See docs/benchmarks/2026-09-23-als-density-eval.md.
set -euo pipefail
source "$(dirname "$0")/common.sh"

FF3D_THIN_DENSITY="${FF3D_THIN_DENSITY:-25}"
FF3D_THIN_SEED="${FF3D_THIN_SEED:-0}"
FF3D_THIN_LIST="${FF3D_THIN_LIST:-data/ForAINetV2/meta_data/test_list.txt}"

# Same mount as ff3d_docker, minus --gpus and with the entrypoint bypassed.
thin_cpu_docker() {
  ff3d_run $FF3D_DOCKER run --rm --entrypoint python \
    -e PYTHONPATH=/workspace -w /workspace \
    -v "$FF3D_ROOT":/workspace "$FF3D_IMAGE" "$@"
}

for pair in "uniform u" "canopy c"; do
  # shellcheck disable=SC2086
  set -- $pair
  mode="$1" letter="$2"
  dst="data/ForAINetV2/test_data_thin${FF3D_THIN_DENSITY}${letter}"
  ff3d_log "thinning -> ${dst} (${mode}, ${FF3D_THIN_DENSITY} pts/m2, seed ${FF3D_THIN_SEED})"
  ff3d_run mkdir -p "$FF3D_ROOT/work_dirs/logs/thin/${mode}"
  thin_cpu_docker benchmark/thin_plots.py \
    --src data/ForAINetV2/test_data --dst "$dst" \
    --density "$FF3D_THIN_DENSITY" --mode "$mode" --seed "$FF3D_THIN_SEED" \
    --list "$FF3D_THIN_LIST" \
    --out-list "work_dirs/logs/thin/${mode}/scan_list.txt"
done
ff3d_log "thinning done"
