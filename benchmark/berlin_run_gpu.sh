#!/usr/bin/env bash
# Per-GPU queue: bash berlin-run-gpu.sh <gpu> <tile stem>...
#
# Tracked copy of the script used for the Berlin km-tile production runs; the
# live copy on carrot is work_dirs/logs/berlin-run-gpu.sh (identical). One
# invocation owns one GPU and walks its tiles sequentially: split -> run.
# Splitting uses a 20 m halo (--buffer 20) plus whichever of the eight
# surrounding km tiles already exist under inputs/berlin/, so the halo can be
# filled from the neighbours' own points; the sub-tiles are inferred as one
# batch (run), same as before. Merging into a seamless km tile with
# mosaic-wide tree ids is NOT done here any more -- run benchmark/berlin_stitch.sh
# once every tile of the mosaic (across all GPUs) has finished its run.
#
#   nohup bash benchmark/berlin_run_gpu.sh 5 3dm_33_381_5829_1_be 3dm_33_381_5830_1_be \
#     > work_dirs/logs/berlin-gpu5-$(date +%Y%m%d-%H%M%S).log 2>&1 &
#
# See .claude/skills/ff3d-inference-km-tiles/SKILL.md.
set -uo pipefail
GPU="$1"; shift
cd /raid/cwinkelmann/ForestFormer3D || exit 1
source /raid/cwinkelmann/ff3d-geo-venv/bin/activate
CK=work_dirs/clean_forestformer/epoch_3000_fix.pth
for T in "$@"; do
  echo "=== $(date +%FT%T) $T: split ==="
  # The neighbour lookup below reads E/N out of the stem, so a stem in any other
  # naming would silently find no neighbours ($((E + dE)) on an empty operand) and
  # the tile would be inferred halo-starved on its km borders with no message.
  if [[ ! "$T" =~ ^3dm_33_[0-9]+_[0-9]+_1_be$ ]]; then
    echo "!!! $T: not a 3dm_33_<E>_<N>_1_be km-tile stem; neighbour lookup impossible" >&2
    exit 2
  fi
  IFS=_ read -r P1 P2 E N P5 P6 <<< "$T"
  NB=()
  for dE in -1 0 1; do
    for dN in -1 0 1; do
      if [ "$dE" -eq 0 ] && [ "$dN" -eq 0 ]; then
        continue
      fi
      CAND="inputs/berlin/${P1}_${P2}_$((E + dE))_$((N + dN))_${P5}_${P6}.las"
      if [ -f "$CAND" ]; then
        NB+=("$CAND")
      fi
    done
  done
  python -m ff3d_geo split --las inputs/berlin/$T.las --out inputs/berlin/sub/$T --buffer 20 \
    ${NB[@]+--neighbours "${NB[@]}"} > work_dirs/logs/split-$T.txt \
    || { echo "!!! $T split failed"; continue; }
  echo "=== $(date +%FT%T) $T: run ($(wc -l < work_dirs/logs/split-$T.txt) sub-tiles) ==="
  S=$(date +%s)
  python -m ff3d_geo run --las inputs/berlin/sub/$T/*.las --checkpoint $CK --out work_dirs/berlin-$T --gpu $GPU || echo "!!! $T run reported failures (see above)"
  R=$(( $(date +%s) - S ))
  echo "$R" >> work_dirs/logs/runtime-$T.txt
  echo "=== $(date +%FT%T) $T: tile done (run ${R}s) ==="
done
echo "=== $(date +%FT%T) block finished ==="
