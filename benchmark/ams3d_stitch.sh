#!/usr/bin/env bash
# Mosaic-wide stitch of the AMS3D sub-tile results: bash ams3d_stitch.sh <tile stem>...
#
# Runs after every tile has finished benchmark/ams3d_run_cpu.sh. ONE `ff3d_geo stitch`
# over all given tiles -- the same manifests the ForestFormer3D and SegmentAnyTree stitches
# used, so a tree straddling a sub-tile or km-tile border gets the SAME id everywhere --
# then per tile border-check (seam metrics) and masks (instance/semantic GeoTIFFs, crowns).
# Output: work_dirs/ams3d-mosaic/<T>.las, _trees.gpkg, _report.json/.md, _border.json,
# _crowns.gpkg, _instance_50cm.tif, _semantic_50cm.tif, plus stitch.json.
#
#   nohup bash benchmark/ams3d_stitch.sh $(ls -d inputs/berlin/sub/*/ | xargs -n1 basename) \
#     > work_dirs/logs/ams3d/stitch-$(date +%Y%m%d-%H%M%S).log 2>&1 < /dev/null &
set -uo pipefail
FF3D_ROOT="${FF3D_ROOT:-/raid/cwinkelmann/ForestFormer3D}"
GEO_VENV="${GEO_VENV:-/raid/cwinkelmann/ff3d-geo-venv}"
cd "$FF3D_ROOT" || exit 1
source "$GEO_VENV/bin/activate"
[ "$#" -gt 0 ] || { echo "usage: bash benchmark/ams3d_stitch.sh <tile stem>..." >&2; exit 2; }
OUT="${AMS3D_MOSAIC_OUT:-work_dirs/ams3d-mosaic}"   # override to keep an earlier mosaic intact
MIN_H="${FF3D_MIN_HEIGHT:-2}"                       # drop instances shorter than this (m); 0 keeps all
MANIFESTS=(); RESULTS=(); TOTAL=0
for T in "$@"; do
  MANIFESTS+=("inputs/berlin/sub/$T/split_manifest.json")
  RESULTS+=("work_dirs/ams3d33-$T/sub")
  R=$(tail -n1 "work_dirs/logs/ams3d/runtime-$T.txt" 2>/dev/null || echo 0)
  case $R in ''|*[!0-9]*) R=0;; esac
  TOTAL=$((TOTAL + R))
done
echo "=== $(date +%FT%T) ams3d mosaic: stitch ($# tiles, runtime ${TOTAL}s) ==="
python -m ff3d_geo stitch --manifest "${MANIFESTS[@]}" --results "${RESULTS[@]}" --out "$OUT" --runtime-s "$TOTAL" --min-height "$MIN_H" \
  || { echo "!!! ams3d stitch failed"; exit 1; }
echo "=== $(date +%FT%T) stitch done ==="
for T in "$@"; do
  echo "=== $(date +%FT%T) $T: border-check ==="
  python -m ff3d_geo border-check --las "$OUT/$T.las" --json "$OUT/${T}_border.json" || echo "!!! $T border-check failed"
  echo "=== $(date +%FT%T) $T: masks ==="
  python -m ff3d_geo masks --las "$OUT/$T.las" --out "$OUT" --prefix "$T" || echo "!!! $T masks failed"
done
echo "=== $(date +%FT%T) ams3d mosaic finished ==="
