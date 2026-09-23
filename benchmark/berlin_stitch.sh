#!/usr/bin/env bash
# Mosaic-wide stitch: bash berlin_stitch.sh <tile stem>...
#
# Runs after every tile of the mosaic has finished benchmark/berlin_run_gpu.sh's
# split -> run (on however many GPUs). It stitches ALL the given tiles' sub-tile
# results in ONE `ff3d_geo stitch` call, so a tree straddling a sub-tile border --
# or a km-tile border -- gets the SAME id everywhere in the mosaic (one manifest
# and one results dir per tile; see ff3d_geo/stitch.py). Then, per tile:
# `border-check` (seam metrics), `masks` (the instance/semantic GeoTIFFs + crown
# polygons) and, when FF3D_BUILDINGS names a footprint GeoPackage, `buildings`
# (the Berlin ALKIS roof mask).
#
#   bash benchmark/berlin_stitch.sh 3dm_33_381_5829_1_be 3dm_33_381_5830_1_be \
#     > work_dirs/logs/berlin-stitch-$(date +%Y%m%d-%H%M%S).log 2>&1
#   FF3D_BUILDINGS=/path/alkis_buildings.gpkg bash benchmark/berlin_stitch.sh ...
#
# See .claude/skills/ff3d-inference-km-tiles/SKILL.md.
set -uo pipefail
cd /raid/cwinkelmann/ForestFormer3D || exit 1
source /raid/cwinkelmann/ff3d-geo-venv/bin/activate
[ "$#" -gt 0 ] || { echo "usage: bash benchmark/berlin_stitch.sh <tile stem>..." >&2; exit 2; }
OUT=work_dirs/berlin-mosaic

MANIFESTS=()
RESULTS=()
TOTAL_RUNTIME=0
for T in "$@"; do
  MANIFESTS+=("inputs/berlin/sub/$T/split_manifest.json")
  RESULTS+=("work_dirs/berlin-$T")
  RFILE="work_dirs/logs/runtime-$T.txt"
  if [ -f "$RFILE" ]; then
    R=$(tail -n1 "$RFILE")
    case $R in
      ''|*[!0-9]*)
        echo "!!! $T: unreadable $RFILE (last line '$R'), its run time is not counted in --runtime-s"
        R=0
        ;;
    esac
    TOTAL_RUNTIME=$((TOTAL_RUNTIME + R))
  else
    echo "!!! $T: no $RFILE, its run time is not counted in --runtime-s"
  fi
done

echo "=== $(date +%FT%T) mosaic: stitch ($# tiles) ==="
python -m ff3d_geo stitch --manifest "${MANIFESTS[@]}" --results "${RESULTS[@]}" --out "$OUT" \
  --runtime-s "$TOTAL_RUNTIME" \
  || { echo "!!! mosaic stitch failed"; exit 1; }
echo "=== $(date +%FT%T) mosaic: stitch done ==="

for T in "$@"; do
  echo "=== $(date +%FT%T) $T: border-check ==="
  python -m ff3d_geo border-check --las $OUT/$T.las --json $OUT/${T}_border.json \
    || echo "!!! $T border-check failed"

  echo "=== $(date +%FT%T) $T: masks ==="
  python -m ff3d_geo masks --las $OUT/$T.las --out $OUT --prefix $T \
    || echo "!!! $T masks failed"

  if [ -n "${FF3D_BUILDINGS:-}" ]; then
    echo "=== $(date +%FT%T) $T: buildings ==="
    python -m ff3d_geo buildings --las $OUT/$T.las --buildings "$FF3D_BUILDINGS" \
      --out $OUT/masked \
      || echo "!!! $T buildings failed"
  fi

  echo "=== $(date +%FT%T) $T: tile done ==="
done
echo "=== $(date +%FT%T) block finished ==="
