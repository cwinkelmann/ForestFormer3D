#!/usr/bin/env bash
# Extend the Berlin mosaic by new km tiles with all three methods, then re-stitch EVERYTHING.
#
#   nohup bash benchmark/berlin_extend_mosaic.sh 3dm_33_375_5826_1_be 3dm_33_376_5826_1_be ... \
#     > work_dirs/logs/extend/extend-$(date +%Y%m%d-%H%M%S).log 2>&1 < /dev/null &
#
# Adding a tile to a seamless mosaic is not "run one more tile": tree ids are dense across the
# whole mosaic, so the new tiles are split and inferred, then ONE stitch over all tiles (old
# and new) hands out a fresh id set. The earlier mosaics are left intact; the new ones go to
# work_dirs/{berlin,sat,ams3d}-mosaic-$EXTEND_TAG. Stages, each skipped when its marker
# work_dirs/logs/extend/<stage>.done exists (so a crashed run resumes where it stopped):
#
#   wait     every new tile's LAS under inputs/berlin/, $EXTEND_MARKER (the uploader's "all
#            files complete" flag), and no process matching $EXTEND_WAIT_FOR left (default:
#            the AMS3D CPU queues of the previous mosaic -- 192 workers would starve the GPU
#            runs' host-side preprocessing).
#   ams3d33  once that run is gone: stitch the PREVIOUS mosaic's AMS3D results and compute its
#            agreement with the other two methods, in the background (CPU), so the 33-tile
#            three-way comparison exists before the 44-tile one supersedes it.
#   ff3d     benchmark/berlin_run_gpu.sh (split --buffer 20 + run), one queue per free GPU in
#            $EXTEND_GPUS, tiles round-robin. A GPU with memory in use at launch is skipped.
#   sat+ams  SegmentAnyTree (SAT_STITCH=1, two queues per GPU) and AMS3D (four CPU queues)
#            over the same haloed splits, concurrently.
#   stitch   FF3D (with the building mask when $FF3D_BUILDINGS is set) and SAT in parallel,
#            then AMS3D, each over ALL tiles that have a split under inputs/berlin/sub/.
#   agree    benchmark/instance_agreement.py per tile for the three method pairs.
#
# Environment: EXTEND_TAG (44), EXTEND_GPUS ("3 4 5 6 7"; GPU 1 is never ours), EXTEND_MARKER
# (a file to wait for before starting; empty = no wait), EXTEND_WAIT_FOR (pgrep -f pattern,
# default ams3d_run_cpu[.]sh), FF3D_BUILDINGS (footprint GeoPackage covering ALL tiles),
# FF3D_ROOT, GEO_VENV. Detached on purpose: never run it inside a live ssh `bash -c`.
set -uo pipefail
FF3D_ROOT="${FF3D_ROOT:-/raid/cwinkelmann/ForestFormer3D}"
GEO_VENV="${GEO_VENV:-/raid/cwinkelmann/ff3d-geo-venv}"
EXTEND_TAG="${EXTEND_TAG:-44}"
EXTEND_GPUS="${EXTEND_GPUS:-3 4 5 6 7}"
EXTEND_MARKER="${EXTEND_MARKER:-}"
P="ams3d_run_cpu"; EXTEND_WAIT_FOR="${EXTEND_WAIT_FOR:-${P}[.]sh}"
cd "$FF3D_ROOT" || exit 1
source "$GEO_VENV/bin/activate"
[ "$#" -gt 0 ] || { echo "usage: bash benchmark/berlin_extend_mosaic.sh <new tile stem>..." >&2; exit 2; }
NEW=("$@")
for T in "${NEW[@]}"; do
  [[ "$T" =~ ^3dm_33_[0-9]+_[0-9]+_1_be$ ]] || { echo "!!! $T: not a 3dm_33_<E>_<N>_1_be stem" >&2; exit 2; }
done
LOG=work_dirs/logs/extend
mkdir -p "$LOG" work_dirs/logs/sat work_dirs/logs/ams3d
TS=$(date +%Y%m%d-%H%M%S)
log() { echo "=== $(date +%FT%T) $*"; }
done_marker() { [ -f "$LOG/$1.done" ]; }
mark() { date +%FT%T > "$LOG/$1.done"; }
# the tiles of the previous mosaic: whatever is split BEFORE the new ones are
OLD=($(ls -d inputs/berlin/sub/*/ 2>/dev/null | xargs -n1 basename | grep -v -F -x -f <(printf '%s\n' "${NEW[@]}")))
log "extend: ${#NEW[@]} new tiles on top of ${#OLD[@]} existing, tag $EXTEND_TAG, GPUs [$EXTEND_GPUS]"

# ------------------------------------------------------------------------------- wait
if ! done_marker wait; then
  for T in "${NEW[@]}"; do
    until [ -f "inputs/berlin/$T.las" ]; do sleep 60; done
  done
  if [ -n "$EXTEND_MARKER" ]; then
    log "waiting for $EXTEND_MARKER"
    until [ -f "$EXTEND_MARKER" ]; do sleep 60; done
  fi
  for T in "${NEW[@]}"; do   # a truncated upload would fail inside split, hours later
    python - "inputs/berlin/$T.las" <<'EOF' || { echo "!!! $T: LAS unreadable, aborting"; exit 1; }
import sys, laspy
with laspy.open(sys.argv[1]) as f:
    n = f.header.point_count
    f.read()   # the whole file: a short file raises here
print(f"  {sys.argv[1]}: {n:,} points ok")
EOF
  done
  log "waiting for processes matching '$EXTEND_WAIT_FOR' to finish"
  until [ "$(pgrep -f "$EXTEND_WAIT_FOR" | wc -l)" = "0" ]; do sleep 300; done
  mark wait
fi

# ---------------------------------------------------------------- ams3d33 (background)
AMS33_PID=
if ! done_marker ams3d33 && [ "${#OLD[@]}" -gt 0 ]; then
  (
    log "ams3d33: stitch of the previous ${#OLD[@]}-tile mosaic"
    bash benchmark/ams3d_stitch.sh "${OLD[@]}" > "work_dirs/logs/ams3d/stitch-$TS.log" 2>&1 \
      || echo "!!! ams3d33 stitch failed (work_dirs/logs/ams3d/stitch-$TS.log)"
    for pair in "berlin-mosaic:ForestFormer3D:ams3d-mosaic:AMS3D" "sat-mosaic:SegmentAnyTree:ams3d-mosaic:AMS3D"; do
      IFS=: read -r A LA B LB <<< "$pair"
      OUTD="work_dirs/logs/agreement33-${A%%-*}-${B%%-*}"; mkdir -p "$OUTD"
      for T in "${OLD[@]}"; do
        [ -f "$OUTD/$T.json" ] && continue
        python3 benchmark/instance_agreement.py --a "work_dirs/$A/$T.las" --label-a "$LA" \
          --b "work_dirs/$B/$T.las" --label-b "$LB" --json "$OUTD/$T.json" > "$OUTD/$T.txt" 2>&1 \
          || echo "!!! agreement33 $A vs $B $T failed"
      done
    done
    log "ams3d33: done"
    date +%FT%T > "$LOG/ams3d33.done"
  ) &
  AMS33_PID=$!
fi

# ------------------------------------------------------------------------------- ff3d
free_gpus() {
  for g in $EXTEND_GPUS; do
    m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$g" 2>/dev/null | tr -d ' ')
    [ "${m:-99999}" -lt 2000 ] && echo "$g"
  done
}
if ! done_marker ff3d; then
  GPUS=($(free_gpus))
  until [ "${#GPUS[@]}" -gt 0 ]; do log "no free GPU among [$EXTEND_GPUS], waiting"; sleep 600; GPUS=($(free_gpus)); done
  log "ff3d: ${#NEW[@]} tiles on GPUs ${GPUS[*]}"
  declare -a Q=(); i=0
  for T in "${NEW[@]}"; do Q[i % ${#GPUS[@]}]+=" $T"; i=$((i + 1)); done
  PIDS=()
  for k in "${!GPUS[@]}"; do
    [ -n "${Q[k]:-}" ] || continue
    log "  GPU ${GPUS[k]}:${Q[k]}"
    # shellcheck disable=SC2086
    bash benchmark/berlin_run_gpu.sh "${GPUS[k]}" ${Q[k]} > "work_dirs/logs/berlin-gpu${GPUS[k]}-$TS.log" 2>&1 &
    PIDS+=($!)
  done
  wait "${PIDS[@]}"
  F=$(grep -h '!!!' work_dirs/logs/berlin-gpu*-"$TS".log 2>/dev/null | wc -l)
  log "ff3d: queues finished, $F failure lines"
  for T in "${NEW[@]}"; do
    [ -f "inputs/berlin/sub/$T/split_manifest.json" ] || echo "!!! $T: no split manifest"
    [ -d "work_dirs/berlin-$T" ] || echo "!!! $T: no run output"
  done
  mark ff3d
fi

# ---------------------------------------------------------------------------- sat+ams
if ! done_marker satams; then
  GPUS=($(free_gpus))
  until [ "${#GPUS[@]}" -gt 0 ]; do log "no free GPU for SAT, waiting"; sleep 600; GPUS=($(free_gpus)); done
  NQ=$(( 2 * ${#GPUS[@]} )); [ "$NQ" -gt "${#NEW[@]}" ] && NQ=${#NEW[@]}
  log "sat: ${#NEW[@]} tiles in $NQ queues on GPUs ${GPUS[*]} (two per GPU)"
  declare -a SQ=(); i=0
  for T in "${NEW[@]}"; do SQ[i % NQ]+=" $T"; i=$((i + 1)); done
  PIDS=()
  for k in $(seq 0 $((NQ - 1))); do
    g=${GPUS[k % ${#GPUS[@]}]}
    # shellcheck disable=SC2086
    SAT_STITCH=1 bash benchmark/sat_run_gpu.sh "$g" ${SQ[k]} > "work_dirs/logs/sat/sat-gpu$g-q$k-$EXTEND_TAG-$TS.log" 2>&1 &
    PIDS+=($!)
  done
  log "ams3d: ${#NEW[@]} tiles in 4 CPU queues"
  declare -a AQ=(); i=0
  for T in "${NEW[@]}"; do AQ[i % 4]+=" $T"; i=$((i + 1)); done
  for k in 0 1 2 3; do
    [ -n "${AQ[k]:-}" ] || continue
    # shellcheck disable=SC2086
    bash benchmark/ams3d_run_cpu.sh ${AQ[k]} > "work_dirs/logs/ams3d/ams3d-q$k-$EXTEND_TAG-$TS.log" 2>&1 &
    PIDS+=($!)
  done
  wait "${PIDS[@]}"
  F=$(grep -h '!!!' work_dirs/logs/sat/sat-gpu*-"$EXTEND_TAG"-"$TS".log work_dirs/logs/ams3d/ams3d-q*-"$EXTEND_TAG"-"$TS".log 2>/dev/null | wc -l)
  log "sat+ams3d: finished, $F failure lines"
  for T in "${NEW[@]}"; do
    [ -d "work_dirs/sat-$T/sub" ] || echo "!!! $T: no SAT sub-tile results"
    [ -d "work_dirs/ams3d33-$T/sub" ] || echo "!!! $T: no AMS3D sub-tile results"
  done
  mark satams
fi

# ------------------------------------------------------------------------------ stitch
ALL=($(ls -d inputs/berlin/sub/*/ | xargs -n1 basename))
[ -n "$AMS33_PID" ] && { log "waiting for the ams3d33 background stage"; wait "$AMS33_PID"; }
if ! done_marker stitch; then
  log "stitch: ${#ALL[@]} tiles -> work_dirs/{berlin,sat,ams3d}-mosaic-$EXTEND_TAG"
  FF3D_MOSAIC_OUT="work_dirs/berlin-mosaic-$EXTEND_TAG" bash benchmark/berlin_stitch.sh "${ALL[@]}" \
    > "work_dirs/logs/berlin-stitch-$EXTEND_TAG-$TS.log" 2>&1 &
  P1=$!
  (
    SOUT="work_dirs/sat-mosaic-$EXTEND_TAG"; MAN=(); RES=()
    for T in "${ALL[@]}"; do MAN+=("inputs/berlin/sub/$T/split_manifest.json"); RES+=("work_dirs/sat-$T/sub"); done
    TOTAL=$(grep -h "tile done" work_dirs/logs/sat/sat-gpu*.log 2>/dev/null | sed "s/.*run //;s/s).*//" | grep -E '^[0-9]+$' | paste -sd+ | bc)
    echo "=== $(date +%FT%T) SAT stitch (${#ALL[@]} tiles, runtime ${TOTAL:-0}s) ==="
    python -m ff3d_geo stitch --manifest "${MAN[@]}" --results "${RES[@]}" --out "$SOUT" --runtime-s "${TOTAL:-0}" \
      || { echo "!!! SAT stitch failed"; exit 1; }
    for T in "${ALL[@]}"; do
      python -m ff3d_geo border-check --las "$SOUT/$T.las" --json "$SOUT/${T}_border.json" || echo "!!! $T SAT border-check failed"
      python -m ff3d_geo masks --las "$SOUT/$T.las" --out "$SOUT" --prefix "$T" || echo "!!! $T SAT masks failed"
    done
    echo "=== $(date +%FT%T) SAT mosaic finished ==="
  ) > "work_dirs/logs/sat/stitch-$EXTEND_TAG-$TS.log" 2>&1 &
  P2=$!
  wait "$P1" "$P2"
  AMS3D_MOSAIC_OUT="work_dirs/ams3d-mosaic-$EXTEND_TAG" bash benchmark/ams3d_stitch.sh "${ALL[@]}" \
    > "work_dirs/logs/ams3d/stitch-$EXTEND_TAG-$TS.log" 2>&1 || echo "!!! ams3d stitch failed"
  for M in berlin sat ams3d; do
    n=$(ls "work_dirs/$M-mosaic-$EXTEND_TAG"/3dm_33_*_1_be.las 2>/dev/null | wc -l)
    log "stitch: $M-mosaic-$EXTEND_TAG has $n/${#ALL[@]} tile LAS"
  done
  mark stitch
fi

# ------------------------------------------------------------------------------- agree
if ! done_marker agree; then
  PIDS=()
  for pair in "berlin:ForestFormer3D:sat:SegmentAnyTree" "berlin:ForestFormer3D:ams3d:AMS3D" "sat:SegmentAnyTree:ams3d:AMS3D"; do
    (
      IFS=: read -r A LA B LB <<< "$pair"
      OUTD="$LOG/agreement-$EXTEND_TAG-$A-$B"; mkdir -p "$OUTD"
      for T in "${ALL[@]}"; do
        [ -f "$OUTD/$T.json" ] && continue
        python3 benchmark/instance_agreement.py --a "work_dirs/$A-mosaic-$EXTEND_TAG/$T.las" --label-a "$LA" \
          --b "work_dirs/$B-mosaic-$EXTEND_TAG/$T.las" --label-b "$LB" --json "$OUTD/$T.json" > "$OUTD/$T.txt" 2>&1 \
          || echo "!!! agreement $A vs $B $T failed"
      done
    ) &
    PIDS+=($!)
  done
  wait "${PIDS[@]}"
  mark agree
fi
log "extend: all stages done"
