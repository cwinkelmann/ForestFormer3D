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
#   filter    drop instances under FF3D_MIN_HEIGHT (2 m) from any mosaic stitched without
#             --min-height, recompute border metrics and the agreements (ff3d_geo.filter)
#   octrees   PotreeConverter per tile and method into work_dirs/logs/potree/out_<m><TAG>/, four sets in parallel
#             (ff3d from the building-masked LAS, ff3d_raw from the raw model output, sat, ams3d)
#   site      a fresh site dir next to the served one: viewer files copied from it, the
#             orthophoto overlays of tiles whose GeoTIFF is not on carrot carried over,
#             octrees hard-linked in, ForestFormer3D base build, SAT / AMS3D / ff3d_raw / ptf variants
#   swap      served dir -> <served>_<old tiles>tiles, fresh dir -> served, nginx restarted,
#             a few HTTP checks against the container
#
# Environment: TAG (arg 1), FF3D_MIN_HEIGHT (2), FF3D_ROOT, GEO_VENV, POTREE_SITE (/raid/cwinkelmann/potree/berlin_potree_v2),
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
      # ForestFormer3D: the stitch also writes the building-masked products into
      # masked/ (roof instances dropped, footprint points -> class 3); those are the
      # deliverable, so they override the raw ones (the raw _border.json stays).
      if [ "$M" = berlin ] && [ -f "work_dirs/berlin-mosaic-$TAG/masked/$T.las" ]; then
        for f in "work_dirs/berlin-mosaic-$TAG/masked/$T"*; do ln -sfn "$FF3D_ROOT/$f" "$D/$(basename "$f")"; done
        # ... and the raw model output stays reachable as the ff3d_raw viewer variant
        R="work_dirs/berlin-mosaic-$TAG-by-tile-raw/$T"; mkdir -p "$R"
        for f in "work_dirs/berlin-mosaic-$TAG/$T"*; do ln -sfn "$FF3D_ROOT/$f" "$R/$(basename "$f")"; done
      fi
    done
  done
  mark bytile
fi

# ------------------------------------------------------------------------------ filter
# Instances shorter than $FF3D_MIN_HEIGHT (default 2 m; grass on meadows labelled leaf,
# ff3d_geo.filter) must be gone from every mosaic before octrees and overlays are built.
# A stitch run with --min-height already did it (stitch.json records min_height); a
# mosaic stitched without it is filtered here in place, tile by tile, and its border
# metrics and the three agreements are recomputed since the ids changed.
MIN_H="${FF3D_MIN_HEIGHT:-2}"
if ! done_marker filter && [ "$MIN_H" != "0" ]; then
  refiltered=0
  for M in berlin sat ams3d; do
    D="work_dirs/$M-mosaic-$TAG"
    have=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1])).get('min_height', 0))" "$D/stitch.json" 2>/dev/null || echo 0)
    if [ "$have" != "0" ] && [ "$have" != "0.0" ]; then log "filter: $M-mosaic-$TAG already stitched with min_height $have"; continue; fi
    log "filter: $M-mosaic-$TAG, < $MIN_H m, ${#TILES[@]} tiles"
    printf '%s\n' "${TILES[@]}" | xargs -P 8 -I{} sh -c \
      "python -m ff3d_geo filter --las $D/{}.las --out $D --min-height $MIN_H > $LOG/filter-$TAG-$M-{}.log 2>&1 || echo '!!! filter $M {} failed'; \
       python -m ff3d_geo border-check --las $D/{}.las --json $D/{}_border.json > /dev/null 2>&1 || echo '!!! border-check $M {} failed'"
    python3 - "$D/stitch.json" "$MIN_H" <<'EOF2'
import json, sys
p, mh = sys.argv[1], float(sys.argv[2])
d = json.load(open(p)); d["min_height"] = mh; d["filtered_after_stitch"] = True
json.dump(d, open(p, "w"), indent=2)
EOF2
    refiltered=$((refiltered + 1))
  done
  if [ "$refiltered" -gt 0 ]; then
    log "filter: recomputing the agreements on the filtered mosaics"
    rm -f "work_dirs/logs/extend/agreement-$TAG-"*/*.json
    PIDS=()
    for pair in "berlin:ForestFormer3D:sat:SegmentAnyTree" "berlin:ForestFormer3D:ams3d:AMS3D" "sat:SegmentAnyTree:ams3d:AMS3D"; do
      (
        IFS=: read -r A LA B LB <<< "$pair"
        OUTD="work_dirs/logs/extend/agreement-$TAG-$A-$B"; mkdir -p "$OUTD"
        for T in "${TILES[@]}"; do
          python3 benchmark/instance_agreement.py --a "work_dirs/$A-mosaic-$TAG/$T.las" --label-a "$LA" \
            --b "work_dirs/$B-mosaic-$TAG/$T.las" --label-b "$LB" --json "$OUTD/$T.json" > "$OUTD/$T.txt" 2>&1 \
            || echo "!!! agreement $A vs $B $T failed"
        done
      ) &
      PIDS+=($!)
    done
    wait "${PIDS[@]}"
  fi
  mark filter
fi

# ---------------------------------------------------------------------------- octrees
convert_one() {   # $1 = las dir, $2 = out dir name, $3 = tile
  [ -f "$LOG/$2/$3/metadata.json" ] && return 0
  [ -f "$1/$3.las" ] || { echo "!!! $2 $3: no $1/$3.las"; return 0; }
  python3 benchmark/potree_convert_tile.py --las "$1/$3.las" \
    --out "$LOG/$2/$3" --potree-converter "$PC" > "$LOG/conv_$2_$3.log" 2>&1 \
    || echo "!!! $2 $3 conversion failed"
}
export -f convert_one; export LOG PC
CONVERT_JOBS="${CONVERT_JOBS:-6}"   # converters per method set; four sets run side by side
convert_all() {   # $1 = directory holding <T>.las, $2 = out dir name
  printf '%s\n' "${TILES[@]}" | xargs -P "$CONVERT_JOBS" -I{} bash -c 'convert_one "$1" "$2" "$3"' _ "$1" "$2" {}
}
# ForestFormer3D: the building-masked LAS (masked/) is the deliverable and becomes the
# base "ff3d" octrees; the raw model output becomes the "ff3d_raw" variant so the viewer
# can show where the ALKIS footprint mask overrode the model (semantic 3, no instance).
FF3D_LAS="work_dirs/berlin-mosaic-$TAG"; FF3D_RAW_LAS=""
if [ -d "$FF3D_LAS/masked" ]; then FF3D_RAW_LAS="$FF3D_LAS"; FF3D_LAS="$FF3D_LAS/masked"; fi
if ! done_marker octrees; then
  log "octrees: ${#TILES[@]} tiles x 3 methods (+ ff3d_raw: ${FF3D_RAW_LAS:-none})"
  convert_all "$FF3D_LAS" "out_v3_$TAG" & P1=$!
  convert_all "work_dirs/sat-mosaic-$TAG" "out_sat$TAG" & P2=$!
  convert_all "work_dirs/ams3d-mosaic-$TAG" "out_ams3d$TAG" & P3=$!
  P4=""; [ -n "$FF3D_RAW_LAS" ] && { convert_all "$FF3D_RAW_LAS" "out_ff3draw$TAG" & P4=$!; }
  wait $P1 $P2 $P3 $P4
  for O in "out_v3_$TAG" "out_sat$TAG" "out_ams3d$TAG" "out_ff3draw$TAG"; do
    [ -d "$LOG/$O" ] || continue
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
  # out_ptf holds the PointTreeFormer octrees (15 tiles, built once by add_ptf.sh)
  for pair in "out_v3_$TAG:pointclouds" "out_sat$TAG:pointclouds_sat" "out_ams3d$TAG:pointclouds_ams3d" \
              "out_ff3draw$TAG:pointclouds_ff3d_raw" "out_ptf:pointclouds_ptf"; do
    IFS=: read -r O D <<< "$pair"
    [ -d "$LOG/$O" ] || continue
    mkdir -p "$NEW/$D"
    for T in "${TILES[@]}"; do
      [ -f "$LOG/$O/$T/metadata.json" ] && cp -al "$LOG/$O/$T" "$NEW/$D/$T"
    done
    log "$D: $(ls "$NEW/$D" | wc -l) octrees linked"
  done
  python benchmark/build_potree_site.py --site "$NEW" --ff3d-dir "work_dirs/berlin-mosaic-$TAG-by-tile" \
    --dop2021-dir "$DOP21" --dop2025-dir "$DOP25" --jobs 8 > "$LOG/site-$TAG-base.log" 2>&1 \
    || { echo "!!! base site build failed (see $LOG/site-$TAG-base.log)"; exit 1; }
  for pair in "sat:work_dirs/sat-mosaic-$TAG-by-tile" "ams3d:work_dirs/ams3d-mosaic-$TAG-by-tile" \
              "ff3d_raw:work_dirs/berlin-mosaic-$TAG-by-tile-raw" "ptf:work_dirs/ptf-mosaic"; do
    IFS=: read -r V VD <<< "$pair"
    [ -d "$NEW/pointclouds_$V" ] && [ -d "$VD" ] || { log "variant $V skipped (no octrees or no $VD)"; continue; }
    python benchmark/build_potree_site.py --site "$NEW" --variant "$V" --variant-dir "$VD" \
      --jobs 8 > "$LOG/site-$TAG-$V.log" 2>&1 || echo "!!! $V variant build failed (see $LOG/site-$TAG-$V.log)"
  done
  python - "$NEW/data/tiles.json" <<'EOF'
import json, sys
recs = json.load(open(sys.argv[1]))
recs = recs["tiles"] if isinstance(recs, dict) and "tiles" in recs else recs
n = len(recs)
counts = {v: sum(v in r.get("variants", {}) for r in recs) for v in ("sat", "ams3d", "ff3d_raw", "ptf")}
dop = sum("dop2021" in r.get("layers", {}) for r in recs)
print(f"  manifest: {n} tiles, variants {counts}, dop2021 overlays {dop}")
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
    for path in "pointclouds/$T/metadata.json" "pointclouds_sat/$T/metadata.json" "pointclouds_ams3d/$T/metadata.json" "pointclouds_ff3d_raw/$T/metadata.json" "data/${T}_trees.geojson"; do
      code=$(curl -s -o /dev/null -w '%{http_code}' "$URL/$path"); [ "$code" = "200" ] && ok=$((ok + 1)) || { bad=$((bad + 1)); echo "  !! $code $path"; }
    done
  done
  log "swap: HTTP checks ok=$ok bad=$bad; manifest $(curl -s "$URL/data/tiles.json" | python3 -c 'import json,sys; d=json.load(sys.stdin); d=d["tiles"] if isinstance(d,dict) and "tiles" in d else d; print(len(d), "tiles")')"
  mark swap
fi
log "potree rebuild $TAG: all stages done"
