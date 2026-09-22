#!/usr/bin/env bash
# Released checkpoint epoch_3000: old vs fixed inference on the ForAINetV2 test split.
#
#   bash benchmark/run_release_eval.sh                      # daemonizes itself; prints the log path
#   FF3D_FOREGROUND=1 bash benchmark/run_release_eval.sh    # run in the foreground
#   FF3D_DRY_RUN=1 FF3D_FOREGROUND=1 bash benchmark/run_release_eval.sh   # print docker commands only
#
# Stages (each guarded by a marker under work_dirs/bench-release/ so the script can be
# re-run after an interruption without redoing finished stages):
#   1. preprocess all splits once (main checkout; own marker: the test infos pkl)
#   2. prepare epoch_3000_{converted,raw}.pth via ff3d_prepare_checkpoint (own marker: .layout)
#   3. fixed: tools/test.py on the CONVERTED file  -> work_dirs/bench-release-fixed  (.done-fixed)
#   4. old:   tools/test.py on the RAW file        -> work_dirs/bench-release-old    (.done-old)
#      (old test.py permutes in memory; the old predict() writes <work_dir>/<scan>.ply)
#   5. fixed tools/final_eval.py on both directories (.done-eval-fixed / .done-eval-old)
set -euo pipefail
source "$(dirname "$0")/common.sh"
ff3d_daemonize release-eval "$0" "$@"

FIXED_DIR="work_dirs/bench-release-fixed"
OLD_DIR="work_dirs/bench-release-old"
BENCH_DIR="$FF3D_ROOT/work_dirs/bench-release"

# run_test_stage/run_eval_stage: every mutating filesystem command (mkdir, rm, touch) goes
# through common.sh's ff3d_run, exactly like the docker invocations themselves, so under
# FF3D_DRY_RUN=1 it is only printed, never executed -- a dry run must not create the output
# dirs, delete *.ply/evaluation_total_test.txt left by an interrupted real run, or write the
# $BENCH_DIR/$marker files (the docker call was only printed, not actually run, so nothing
# was verified; a marker left behind by a dry run would make a subsequent REAL run silently
# skip every stage). The marker touch is additionally gated behind an explicit `return 0` on
# the dry-run branch, since it only makes sense after a REAL, verified success.
run_test_stage() {  # run_test_stage <old|fixed> <ckpt (root-relative)> <out_dir (root-relative)> <marker>
  local variant="$1" ckpt="$2" out="$3" marker="$4" runner n
  if [ -f "$BENCH_DIR/$marker" ]; then
    ff3d_log "$variant test.py already done ($out) -- skipping (marker $marker present)"
    return 0
  fi
  ff3d_run mkdir -p "$FF3D_ROOT/$out"
  ff3d_run rm -f "$FF3D_ROOT/$out"/*.ply
  if [ "$variant" = "old" ]; then runner=ff3d_docker_old; else runner=ff3d_docker; fi
  ff3d_log "$variant test.py: $ckpt -> $out"
  "$runner" python tools/test.py "$FF3D_CONFIG" "$ckpt" --work-dir "$out"
  if [ "${FF3D_DRY_RUN:-0}" = "1" ]; then
    ff3d_log "$variant test.py dry-run: skipping ply-count check and marker $marker"
    return 0
  fi
  n="$(find "$FF3D_ROOT/$out" -maxdepth 1 -name '*.ply' | wc -l | tr -d ' ')"
  [ "$n" = "$N_TEST" ] || ff3d_die "$variant produced $n ply files, expected $N_TEST in $out"
  ff3d_log "$variant test.py done: $n ply files"
  ff3d_run mkdir -p "$BENCH_DIR"
  ff3d_run touch "$BENCH_DIR/$marker"
}

run_eval_stage() {  # run_eval_stage <out_dir (root-relative)> <marker>
  local out="$1" marker="$2"
  if [ -f "$BENCH_DIR/$marker" ]; then
    ff3d_log "final_eval already done ($out) -- skipping (marker $marker present)"
    return 0
  fi
  ff3d_run rm -f "$FF3D_ROOT/$out/evaluation_total_test.txt"
  ff3d_log "final_eval.py $out"
  ff3d_docker python tools/final_eval.py "$out"
  if [ "${FF3D_DRY_RUN:-0}" = "1" ]; then
    ff3d_log "final_eval dry-run: skipping F1 check and marker $marker"
    return 0
  fi
  grep -q '^Instance Segmentation F1 score:' "$FF3D_ROOT/$out/evaluation_total_test.txt" \
    || ff3d_die "no F1 line in $out/evaluation_total_test.txt"
  grep '^Instance Segmentation F1 score:' "$FF3D_ROOT/$out/evaluation_total_test.txt" | tail -1
  ff3d_run mkdir -p "$BENCH_DIR"
  ff3d_run touch "$BENCH_DIR/$marker"
}

ff3d_log "release eval start (root $FF3D_ROOT, image $FF3D_IMAGE)"
[ -f "$FF3D_RELEASE_CKPT" ] || ff3d_die "missing $FF3D_RELEASE_CKPT (run benchmark/fetch_zenodo.sh)"
[ -f "$FF3D_OLD/tools/test.py" ] || ff3d_die "missing old worktree (run benchmark/setup_old_worktree.sh)"
[ -f "$FF3D_DATA/meta_data/test_list.txt" ] || ff3d_die "missing $FF3D_DATA/meta_data/test_list.txt"
N_TEST="$(grep -c . "$FF3D_DATA/meta_data/test_list.txt")"

ff3d_preprocess

layout="$(ff3d_prepare_checkpoint work_dirs/clean_forestformer/epoch_3000_fix.pth work_dirs/clean_forestformer/epoch_3000)"
# Under FF3D_DRY_RUN=1, ff3d_prepare_checkpoint writes nothing and $layout is its own
# multi-line "DRY: (planned) ..." dump (the layout truly cannot be determined without
# running the container), not a single "raw"/"converted" word -- print it as-is instead of
# wrapping it in an ff3d_log line meant for a single-line real result.
if [ "${FF3D_DRY_RUN:-0}" = "1" ]; then
  printf '%s\n' "$layout"
else
  ff3d_log "Zenodo checkpoint layout: $layout"
fi

run_test_stage fixed work_dirs/clean_forestformer/epoch_3000_converted.pth "$FIXED_DIR" .done-fixed
run_test_stage old   work_dirs/clean_forestformer/epoch_3000_raw.pth       "$OLD_DIR"   .done-old

run_eval_stage "$FIXED_DIR" .done-eval-fixed
run_eval_stage "$OLD_DIR"   .done-eval-old
ff3d_log "release eval finished"
