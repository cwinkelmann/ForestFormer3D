#!/usr/bin/env bash
# AMS3D over km tiles on carrot, the like-for-like way: bash ams3d_run_cpu.sh <tile stem>...
#
# One invocation walks its tiles sequentially, each tile as one `ff3d_geo ams3d --subtiles`
# run over the SAME haloed 100 m sub-tiles the ForestFormer3D and SegmentAnyTree production
# runs used (inputs/berlin/sub/<T>/, written by `split --buffer 20`). Every sub-tile result
# keeps every point in input order, halo included, so benchmark/ams3d_stitch.sh can unify
# ids over the shared halo exactly as for the other two methods. CPU only: no GPU, no Docker.
#
# carrot has 224 cores. One run at --workers 48 took 19-25 min per km tile on the old 10 m
# buffer; the 20 m halo carries ~1.4x the points. Launch 4 queues in parallel (4 x 48 =
# 192 workers) and expect ~30 min per tile, ~4-5 h for 33 tiles. Detached on purpose:
# never run this inside a live ssh `bash -c` (see the potree-viewer-on-carrot note).
#
#   nohup bash benchmark/ams3d_run_cpu.sh 3dm_33_375_5827_1_be 3dm_33_375_5828_1_be \
#     > work_dirs/logs/ams3d/ams3d-q0-$(date +%Y%m%d-%H%M%S).log 2>&1 < /dev/null &
#
# Environment: FF3D_ROOT, GEO_VENV, AMS3D_WORKERS (48), AMS3D_CONFIG (C), AMS3D_SUB (berlin/sub).
set -uo pipefail
FF3D_ROOT="${FF3D_ROOT:-/raid/cwinkelmann/ForestFormer3D}"
GEO_VENV="${GEO_VENV:-/raid/cwinkelmann/ff3d-geo-venv}"
AMS3D_WORKERS="${AMS3D_WORKERS:-48}"
AMS3D_CONFIG="${AMS3D_CONFIG:-C}"
AMS3D_SUB="${AMS3D_SUB:-berlin/sub}"
cd "$FF3D_ROOT" || exit 1
source "$GEO_VENV/bin/activate"
mkdir -p work_dirs/logs/ams3d
for T in "$@"; do
  if [[ ! "$T" =~ ^3dm_33_[0-9]+_[0-9]+_1_be$ ]]; then
    echo "!!! $T: not a 3dm_33_<E>_<N>_1_be km-tile stem" >&2; exit 2
  fi
  IN="inputs/$AMS3D_SUB/$T"
  OUT="work_dirs/ams3d33-$T/sub"
  if [ ! -f "$IN/split_manifest.json" ]; then echo "!!! $T: no haloed split under $IN"; continue; fi
  N=$(ls "$IN"/*.las 2>/dev/null | wc -l)
  echo "=== $(date +%FT%T) $T: ams3d --subtiles ($N sub-tiles, $AMS3D_WORKERS workers, config $AMS3D_CONFIG) ==="
  S=$(date +%s)
  python -m ff3d_geo ams3d --subtiles "$IN" --out "$OUT" --workers "$AMS3D_WORKERS" --config "$AMS3D_CONFIG" \
    > "work_dirs/logs/ams3d/subtiles-$T.txt" 2>&1 || { echo "!!! $T ams3d failed (see work_dirs/logs/ams3d/subtiles-$T.txt)"; continue; }
  R=$(( $(date +%s) - S ))
  M=$(ls "$OUT"/*.las 2>/dev/null | wc -l)
  [ "$M" -eq "$N" ] || echo "!!! $T: $M of $N sub-tile results written"
  echo "$R" >> "work_dirs/logs/ams3d/runtime-$T.txt"
  echo "=== $(date +%FT%T) $T: tile done (run ${R}s, $M results) ==="
done
echo "=== $(date +%FT%T) block finished ==="
