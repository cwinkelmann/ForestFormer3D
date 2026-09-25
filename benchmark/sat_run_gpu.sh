#!/bin/bash
# SegmentAnyTree over km tiles on carrot: one invocation owns ONE GPU and walks its
# tiles sequentially.  Per tile:  run_inference.sh over the 100 m sub-tiles that
# `ff3d_geo split` cut  ->  benchmark/sat_to_ff3d.py (contract LAS, tree table,
# report, masks)  ->  residue cleanup.  Mirrors benchmark/berlin_run_gpu.sh.
#
#   nohup bash benchmark/sat_run_gpu.sh 7 3dm_33_381_5828_1_be 3dm_33_381_5829_1_be \
#     > work_dirs/logs/sat/sat-gpu7-$(date +%Y%m%d-%H%M%S).log 2>&1 &
#
# The sub-tiles must already exist under inputs/<SAT_SUB>/<T>/ (see ff3d-inference-km-tiles),
# where SAT_SUB defaults to berlin/sub.
#
# Two ways to run, and they need DIFFERENT splits:
#
#   SAT_STITCH=0 (default) -- a halo-free split. benchmark/sat_to_ff3d.py concatenates the
#     per-sub-tile results with each sub-tile's ids offset into a disjoint range, so ids are
#     unique within the km tile but a tree cut by a sub-tile border becomes two trees. On a
#     HALOED split this mode is WRONG: the shared halo points would be merged once per
#     sub-tile that saw them, silently inflating the point count and duplicating trees.
#
#   SAT_STITCH=1 -- the haloed split, for a like-for-like comparison with ForestFormer3D.
#     SegmentAnyTree preserves its input's point count and order, so the per-sub-tile contract
#     LAS (kept via --keep-subtiles, under <out>/sub/) satisfy what `ff3d_geo stitch` needs:
#     core ownership plus IoU matching over the shared halo. Afterwards run ONE stitch over
#     the whole mosaic, exactly as for ForestFormer3D:
#       python -m ff3d_geo stitch --manifest inputs/berlin/sub/*/split_manifest.json \
#         --results work_dirs/sat-*/sub --out work_dirs/sat-mosaic
#     Both methods then share the split, the halo, the core ownership and the stitch, so only
#     the segmentation model differs.
#
# Environment: SAT_ROOT (SegmentAnyTree checkout with model_file/PointGroup-PAPER.pt),
# SAT_IMAGE, FF3D_ROOT, GEO_VENV, SAT_SUB, SAT_STITCH, SAT_KEEP=1 keeps the intermediate files.
set -uo pipefail

GPU="${1:?usage: sat_run_gpu.sh <gpu> <tile>...}"
shift
FF3D_ROOT="${FF3D_ROOT:-/raid/cwinkelmann/ForestFormer3D}"
SAT_ROOT="${SAT_ROOT:-/raid/cwinkelmann/SegmentAnyTree_infer}"
SAT_IMAGE="${SAT_IMAGE:-segment-any-tree:cu118}"
GEO_VENV="${GEO_VENV:-/raid/cwinkelmann/ff3d-geo-venv}"
# Sub-tile set to read, relative to $FF3D_ROOT/inputs. Halo-free unless SAT_STITCH=1.
SAT_SUB="${SAT_SUB:-berlin/sub}"
# 1 = keep the per-sub-tile contract LAS so `ff3d_geo stitch` can unify ids over the halo.
SAT_STITCH="${SAT_STITCH:-0}"
KEEP_SUB=(); [ "$SAT_STITCH" = "1" ] && KEEP_SUB=(--keep-subtiles)

if [ "$(wc -c < "$SAT_ROOT/model_file/PointGroup-PAPER.pt")" -lt 1000000 ]; then
    echo "!!! $SAT_ROOT/model_file/PointGroup-PAPER.pt is an LFS pointer, not the model"; exit 1
fi

for T in "$@"; do
    IN="$FF3D_ROOT/inputs/$SAT_SUB/$T"
    OUT="$FF3D_ROOT/work_dirs/sat-$T"
    RAW="$OUT/sat_raw"
    if [ ! -d "$IN" ]; then echo "!!! $T: no sub-tiles under $IN"; continue; fi
    N=$(ls "$IN"/*.las 2>/dev/null | wc -l)
    mkdir -p "$OUT"
    echo "=== $(date -Is) $T: sat run ($N sub-tiles, GPU $GPU) ==="
    start=$(date +%s)
    # run_inference.sh copies the inputs, shifts each file to local coordinates, runs
    # eval.py once for all files (one model load) and merges the predictions back.
    docker run --rm --name "sat-gpu$GPU-$T" --gpus "device=$GPU" \
        -e NUMBA_CACHE_DIR=/tmp/numba -e OMP_NUM_THREADS=16 --shm-size=64g \
        -v "$SAT_ROOT":/home/nibio/mutable-outside-world \
        -v "$FF3D_ROOT/inputs":/inputs -v "$FF3D_ROOT/work_dirs":/work_dirs \
        --entrypoint bash "$SAT_IMAGE" run_inference.sh "/inputs/$SAT_SUB/$T" "/work_dirs/sat-$T/sat_raw"
    rc=$?
    R=$(( $(date +%s) - start ))
    M=$(ls "$RAW"/final_results/*_out.la? 2>/dev/null | wc -l)
    if [ "$rc" -ne 0 ] || [ "$M" -ne "$N" ]; then
        echo "!!! $T sat run failed (rc=$rc, $M of $N result files) after ${R}s"; continue
    fi
    echo "=== $(date -Is) $T: convert ($M files, run ${R}s) ==="
    "$GEO_VENV/bin/python" "$FF3D_ROOT/benchmark/sat_to_ff3d.py" \
        --sat-las "$RAW"/final_results/*_out.la? --tile "$T" --out "$OUT" --runtime-s "$R" \
        ${KEEP_SUB[@]+"${KEEP_SUB[@]}"} \
        || { echo "!!! $T convert failed"; continue; }
    if [ "${SAT_KEEP:-0}" != "1" ]; then
        # input copies, local-coordinate PLYs, the 665 MB checkpoint copy and the
        # per-file prediction PLYs; final_results/ and eval.log stay.
        rm -rf "$RAW/input_data" "$RAW/utm2local" "$RAW/PointGroup-PAPER.pt" "$RAW"/*.ply
    fi
    echo "=== $(date -Is) $T: tile done (run ${R}s) ==="
done
echo "=== $(date -Is) block finished ==="
