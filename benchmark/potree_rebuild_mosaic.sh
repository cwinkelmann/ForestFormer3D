#!/usr/bin/env bash
# Rebuild the carrot Potree site from a re-stitched mosaic, all three methods, then swap it
# into the served directory.  Detached on purpose (nohup, never a live ssh `bash -c`).
#
#   nohup bash benchmark/potree_rebuild_mosaic.sh 44 > work_dirs/logs/potree/rebuild-44-$(date +%Y%m%d-%H%M%S).log 2>&1 < /dev/null &
#
# A re-stitch hands out a new id set to EVERY tree, so every octree and every per-tile
# overlay (tree tops, crowns, instance raster) has to be rebuilt, not only the new tiles'.
# Steps, each skipped when its marker work_dirs/logs/potree/rebuild-<TAG>.<step>.done exists:
#   wait      until benchmark/berlin_extend_mosaic.sh is gone and its agree.done marker is set
#   bytile    work_dirs/<m>-mosaic-<TAG>-by-tile/<T>/ symlink layouts (build_potree_site wants <dir>/<T>/<T>.las)
#   octrees   PotreeConverter per tile and method into work_dirs/logs/potree/out_<m><TAG>/, three methods in parallel
#   site      a fresh site dir next to the served one: viewer files copied from it, the
#             orthophoto overlays of tiles whose GeoTIFF is not on carrot carried over,
#             octrees hard-linked in, ForestFormer3D base build, SAT and AMS3D variants
#   swap      served dir -> <served>_<old tiles>tiles, fresh dir -> served, nginx restarted,
#             a few HTTP checks against the container
#
# Environment: TAG (arg 1), FF3D_ROOT, GEO_VENV, POTREE_SITE (/raid/cwinkelmann/potree/berlin_potree_v2),
# POTREE_DOP21 / POTREE_DOP25 (inputs/berlin_dop/dop_2021_rgb, dop_2025_sommer), POTREE_URL
# (http://10.188.1.1:8080), POTREE_CONTAINER (ff3d-potree).
set -uo pipefail
TAG="${1:?usage: potree_rebuild_mosaic.sh <tag, e.g. 44>}"
FF3D_ROOT="${FF3D_ROOT:-/raid/cwinkelmann/ForestFormer3D}"
GEO_VENV="${GEO_VENV:-/raid/cwinkelmann/ff3d-geo-venv}"
SITE="${POTREE_SITE:-/raid/cwinkelmann/potree/berlin_potree_v2}"
DOP21="${POTREE_DOP21:-inputs/berlin_dop/dop_2021_rgb}"
DOP25="${POTREE_DOP25:-inputs/berlin_dop/dop_2025_sommer}"
URL="${POTREE_URL:-http://10.188.1.1:8080}"
CONTAINER="${POTREE_CONTAINER:-ff3d-potree}"
cd "$FF3D_ROOT" || exit 1
source "$GEO_VENV/bin/activate"
PC=work_dirs/logs/potree/PotreeConverter_linux_x64/PotreeConverter
LOG=work_dirs/logs/potree
mkdir -p "$LOG"
NEW="${SITE}_build_$TAG"
log() { echo "=== $(date +%FT%T) $*"; }
done_marker() { [ -f "$LOG/rebuild-$TAG.$1.done" ]; }
mark() { date +%FT%T > "$LOG/rebuild-$TAG.$1.done"; }

# ------------------------------------------------------------------------------- wait
if ! done_marker wait; then
  P="berlin_extend_mosaic"; P="${P}[.]sh"
  log "waiting for the extend driver and work_dirs/logs/extend/agree.done"
  until [ "$(pgrep -f "$P" | wc -l)" = "0" ] && [ -f work_dirs/logs/extend/agree.done ]; do sleep 600; done
  for M in berlin sat ams3d; do
    n=$(ls "work_dirs/$M-mosaic-$TAG"/3dm_33_*_1_be.las 2>/dev/null | wc -l)
    log "$M-mosaic-$TAG: $n tile LAS"
    [ "$n" -gt 0 ] || { echo "!!! no $M-mosaic-$TAG, aborting"; exit 1; }
  done
  mark wait
fi
TILES=($(ls "work_dirs/berlin-mosaic-$TAG"/3dm_33_*_1_be.las | xargs -n1 basename | sed 's/\.las$//'))
log "${#TILES[@]} tiles"

# ----------------------------------------------------------------------------- bytile
if ! done_marker bytile; then
  for M in berlin sat ams3d; do
    for T in "${TILES[@]}"; do
      D="work_dirs/$M-mosaic-$TAG-by-tile/$T"; mkdir -p "$D"
      for f in "work_dirs/$M-mosaic-$TAG/$T"*; do ln -sfn "$FF3D_ROOT/$f" "$D/$(basename "$f")"; done
    done
  done
  mark bytile
fi

# ---------------------------------------------------------------------------- octrees
convert_all() {   # $1 = mosaic prefix (berlin|sat|ams3d), $2 = out dir name
  for T in "${TILES[@]}"; do
    [ -f "$LOG/$2/$T/metadata.json" ] && continue
    python3 benchmark/potree_convert_tile.py --las "work_dirs/$1-mosaic-$TAG/$T.las" \
      --out "$LOG/$2/$T" --potree-converter "$PC" > "$LOG/conv_$2_$T.log" 2>&1 \
      || echo "!!! $2 $T conversion failed"
  done
}
if ! done_marker octrees; then
  log "octrees: ${#TILES[@]} tiles x 3 methods"
  convert_all berlin "out_v3_$TAG" & P1=$!
  convert_all sat "out_sat$TAG" & P2=$!
  convert_all ams3d "out_ams3d$TAG" & P3=$!
  wait $P1 $P2 $P3
  for O in "out_v3_$TAG" "out_sat$TAG" "out_ams3d$TAG"; do
    log "$O: $(ls "$LOG/$O"/*/metadata.json 2>/dev/null | wc -l)/${#TILES[@]} octrees"
  done
  mark octrees
fi

# ------------------------------------------------------------------------------- site
if ! done_marker site; then
  log "site: assembling $NEW"
  rm -rf "$NEW"; mkdir -p "$NEW/data"
  for f in index.html README.md libs build; do [ -e "$SITE/$f" ] && cp -a "$SITE/$f" "$NEW/"; done
  # orthophoto overlays of the tiles whose GeoTIFFs are not on carrot: carried over
  cp -a "$SITE"/data/*_dop2021.* "$SITE"/data/*_dop2025.* "$NEW/data/" 2>/dev/null
  for pair in "out_v3_$TAG:pointclouds" "out_sat$TAG:pointclouds_sat" "out_ams3d$TAG:pointclouds_ams3d"; do
    IFS=: read -r O D <<< "$pair"
    mkdir -p "$NEW/$D"
    for T in "${TILES[@]}"; do
      [ -f "$LOG/$O/$T/metadata.json" ] && cp -al "$LOG/$O/$T" "$NEW/$D/$T"
    done
    log "$D: $(ls "$NEW/$D" | wc -l) octrees linked"
  done
  python benchmark/build_potree_site.py --site "$NEW" --ff3d-dir "work_dirs/berlin-mosaic-$TAG-by-tile" \
    --dop2021-dir "$DOP21" --dop2025-dir "$DOP25" --jobs 8 > "$LOG/site-$TAG-base.log" 2>&1 \
    || { echo "!!! base site build failed (see $LOG/site-$TAG-base.log)"; exit 1; }
  for V in sat ams3d; do
    python benchmark/build_potree_site.py --site "$NEW" --variant "$V" --variant-dir "work_dirs/$V-mosaic-$TAG-by-tile" \
      --jobs 8 > "$LOG/site-$TAG-$V.log" 2>&1 || echo "!!! $V variant build failed (see $LOG/site-$TAG-$V.log)"
  done
  python - "$NEW/data/tiles.json" <<'EOF'
import json, sys
recs = json.load(open(sys.argv[1]))
recs = recs["tiles"] if isinstance(recs, dict) and "tiles" in recs else recs
n = len(recs); sat = sum("sat" in r.get("variants", {}) for r in recs); ams = sum("ams3d" in r.get("variants", {}) for r in recs)
dop = sum("dop2021" in r.get("layers", {}) for r in recs)
print(f"  manifest: {n} tiles, sat {sat}, ams3d {ams}, dop2021 overlays {dop}")
EOF
  mark site
fi

# ------------------------------------------------------------------------------- swap
if ! done_marker swap; then
  OLD_N=$(ls "$SITE/pointclouds" | wc -l)
  log "swap: $SITE -> ${SITE}_${OLD_N}tiles, $NEW -> $SITE, restart $CONTAINER"
  mv "$SITE" "${SITE}_${OLD_N}tiles" && mv "$NEW" "$SITE" || { echo "!!! swap failed"; exit 1; }
  docker restart "$CONTAINER" > /dev/null && sleep 5
  ok=0; bad=0
  for T in "${TILES[@]:0:3}" "${TILES[@]: -3}"; do
    for path in "pointclouds/$T/metadata.json" "pointclouds_sat/$T/metadata.json" "pointclouds_ams3d/$T/metadata.json" "data/${T}_trees.geojson"; do
      code=$(curl -s -o /dev/null -w '%{http_code}' "$URL/$path"); [ "$code" = "200" ] && ok=$((ok + 1)) || { bad=$((bad + 1)); echo "  !! $code $path"; }
    done
  done
  log "swap: HTTP checks ok=$ok bad=$bad; manifest $(curl -s "$URL/data/tiles.json" | python3 -c 'import json,sys; d=json.load(sys.stdin); d=d["tiles"] if isinstance(d,dict) and "tiles" in d else d; print(len(d), "tiles")')"
  mark swap
fi
log "potree rebuild $TAG: all stages done"
