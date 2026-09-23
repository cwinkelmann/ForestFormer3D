#!/usr/bin/env bash
# Per-GPU queue: bash berlin-run-gpu.sh <gpu> <tile stem>...
#
# Tracked copy of the script used for the Berlin km-tile production runs; the
# live copy on carrot is work_dirs/logs/berlin-run-gpu.sh (identical). One
# invocation owns one GPU and walks its tiles sequentially: split -> run ->
# merge -> masks. Paths are carrot's and absolute on purpose.
#
#   nohup bash benchmark/berlin_run_gpu.sh 5 3dm_33_381_5829_1_be 3dm_33_381_5830_1_be \
#     > work_dirs/logs/berlin-gpu5-$(date +%Y%m%d-%H%M%S).log 2>&1 &
#
# See .claude/skills/ff3d-inference-km-tiles/SKILL.md.
set -uo pipefail
GPU="$1"; shift
cd /raid/cwinkelmann/ForestFormer3D
source /raid/cwinkelmann/ff3d-geo-venv/bin/activate
CK=work_dirs/clean_forestformer/epoch_3000_fix.pth
for T in "$@"; do
  echo "=== $(date +%FT%T) $T: split ==="
  python -m ff3d_geo split --las inputs/berlin/$T.las --out inputs/berlin/sub/$T > work_dirs/logs/split-$T.txt || { echo "!!! $T split failed"; continue; }
  echo "=== $(date +%FT%T) $T: run ($(wc -l < work_dirs/logs/split-$T.txt) sub-tiles) ==="
  S=$(date +%s)
  python -m ff3d_geo run --las inputs/berlin/sub/$T/*.las --checkpoint $CK --out work_dirs/berlin-$T --gpu $GPU || echo "!!! $T run reported failures (see above)"
  R=$(( $(date +%s) - S ))
  echo "=== $(date +%FT%T) $T: merge ==="
  python -m ff3d_geo merge --las work_dirs/berlin-$T/*_100m.las --gpkg work_dirs/berlin-$T/*_100m_trees.gpkg \
    --out-las work_dirs/berlin-$T/$T.las --out-gpkg work_dirs/berlin-$T/${T}_trees.gpkg \
    --report-json work_dirs/berlin-$T/${T}_report.json --report-md work_dirs/berlin-$T/${T}_report.md --runtime-s $R \
    || { echo "!!! $T merge failed"; continue; }
  python -m ff3d_geo masks --las work_dirs/berlin-$T/$T.las --out work_dirs/berlin-$T || echo "!!! $T masks failed"
  echo "=== $(date +%FT%T) $T: tile done (run ${R}s) ==="
done
echo "=== $(date +%FT%T) block finished ==="
