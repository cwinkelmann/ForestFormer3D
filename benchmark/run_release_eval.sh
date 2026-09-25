#!/usr/bin/env bash
# Released checkpoint epoch_3000: old vs fixed inference on the ForAINetV2 test split.
#
#   bash benchmark/run_release_eval.sh                      # daemonizes itself; prints the log path
#   FF3D_FOREGROUND=1 bash benchmark/run_release_eval.sh    # run in the foreground
#   FF3D_DRY_RUN=1 FF3D_FOREGROUND=1 bash benchmark/run_release_eval.sh   # print docker commands only
#
# Stages (each guarded by a marker INSIDE that stage's own output directory, same layout as
# benchmark/run_train_200.sh, so the script can be re-run after an interruption without
# redoing finished stages):
#   1. preprocess all splits once (main checkout; own marker: the test infos pkl)
#   2. prepare epoch_3000_{converted,raw}.pth via ff3d_prepare_checkpoint (own marker: .layout)
#   3. fixed: tools/test.py on the CONVERTED file  -> work_dirs/bench-release-fixed/.done-test
#   4. old:   tools/test.py on the RAW file        -> work_dirs/bench-release-old/.done-test
#      (old test.py permutes in memory; the old predict() writes <work_dir>/<scan>.ply)
#   5. fixed tools/final_eval.py on both directories (<out>/.done-eval each)
#
# Markers written by runs made BEFORE this change lived in a third directory,
# work_dirs/bench-release/.done-{fixed,old,eval-fixed,eval-old}; they are historical and are
# not read any more (delete or ignore them).
#
# Old variant, two deliberate asymmetries (see docs/benchmarks/RUNBOOK-carrot.md section 7):
#   - a NON-ZERO exit of the old tools/test.py is tolerated: the old evaluator
#     (unified_metric.py @ 6a75c37) always crashes with an IndexError AFTER full-plot
#     inference has written every result .ply. The N_TEST ply-count check below is the real
#     postcondition. A non-zero exit of the FIXED runner is still fatal.
#   - before running a test stage, an output dir that already holds exactly N_TEST .ply files
#     is ADOPTED (marker written, stage skipped) instead of wiped and re-inferred; set
#     FF3D_FORCE=1 to force the rm + re-run. This is what makes a re-run after an SSH drop
#     safe: 1-2.5 h of verified inference is never thrown away by accident.
set -euo pipefail
source "$(dirname "$0")/common.sh"
ff3d_daemonize release-eval "$0" "$@"

FIXED_DIR="work_dirs/bench-release-fixed"
OLD_DIR="work_dirs/bench-release-old"

# run_test_stage/run_eval_stage: every mutating filesystem command (mkdir, rm, touch) goes
# through common.sh's ff3d_run, exactly like the docker invocations themselves, so under
# FF3D_DRY_RUN=1 it is only printed, never executed -- a dry run must not create the output
# dirs, delete *.ply/evaluation_total_test.txt left by an interrupted real run, or write the
# $out/.done-* markers (the docker call was only printed, not actually run, so nothing was
# verified; a marker left behind by a dry run would make a subsequent REAL run silently skip
# every stage). The marker touch is additionally gated behind an explicit `return 0` on the
# dry-run branch, since it only makes sense after a REAL, verified success -- including in
# the adopt branch, which under a dry run only reports the decision it WOULD take.
run_test_stage() {  # run_test_stage <old|fixed> <ckpt (root-relative)> <out_dir (root-relative)>
  local variant="$1" ckpt="$2" out="$3" runner n rc=0
  local marker=".done-test"
  if [ -f "$FF3D_ROOT/$out/$marker" ]; then
    ff3d_log "$variant test.py already done ($out) -- skipping (marker $out/$marker present)"
    return 0
  fi
  # Adopt an interrupted-but-complete run instead of destroying it: the old variant's stage
  # always ends in a crashed evaluator, so the marker is frequently missing while all N_TEST
  # result PLYs are on disk and valid.
  n="$(ff3d_count_ply "$FF3D_ROOT/$out")"
  if [ "$n" = "$N_TEST" ] && [ "${FF3D_FORCE:-0}" != "1" ]; then
    if [ "${FF3D_DRY_RUN:-0}" = "1" ]; then
      ff3d_log "DRY: (would adopt) $n existing ply files in $out -- stage would be skipped (FF3D_FORCE=1 to re-run)"
      return 0
    fi
    ff3d_log "$variant test.py: adopting $n existing PLYs in $out (FF3D_FORCE=1 to re-run)"
    ff3d_run touch "$FF3D_ROOT/$out/$marker"
    return 0
  fi
  ff3d_run mkdir -p "$FF3D_ROOT/$out"
  ff3d_run rm -f "$FF3D_ROOT/$out"/*.ply
  if [ "$variant" = "old" ]; then runner=ff3d_docker_old; else runner=ff3d_docker; fi
  ff3d_log "$variant test.py: $ckpt -> $out"
  "$runner" python tools/test.py "$FF3D_CONFIG" "$ckpt" --work-dir "$out" || rc=$?
  if [ "$rc" != "0" ]; then
    # The old evaluator crashes AFTER writing every ply (review finding #1); the ply count
    # below is the real postcondition. The fixed path has no such excuse.
    [ "$variant" = "old" ] || ff3d_die "$variant tools/test.py failed with exit $rc"
    ff3d_log "WARNING: old tools/test.py exited $rc -- expected (its evaluator crashes after writing all PLYs); checking the ply count instead"
  fi
  if [ "${FF3D_DRY_RUN:-0}" = "1" ]; then
    ff3d_log "$variant test.py dry-run: skipping ply-count check and marker $out/$marker"
    return 0
  fi
  n="$(ff3d_count_ply "$FF3D_ROOT/$out")"
  [ "$n" = "$N_TEST" ] || ff3d_die "$variant produced $n ply files, expected $N_TEST in $out"
  ff3d_log "$variant test.py done: $n ply files"
  ff3d_run touch "$FF3D_ROOT/$out/$marker"
}

run_eval_stage() {  # run_eval_stage <out_dir (root-relative)>
  local out="$1"
  local marker=".done-eval"
  if [ -f "$FF3D_ROOT/$out/$marker" ]; then
    ff3d_log "final_eval already done ($out) -- skipping (marker $out/$marker present)"
    return 0
  fi
  ff3d_run rm -f "$FF3D_ROOT/$out/evaluation_total_test.txt"
  ff3d_log "final_eval.py $out"
  ff3d_docker python tools/final_eval.py "$out"
  if [ "${FF3D_DRY_RUN:-0}" = "1" ]; then
    ff3d_log "final_eval dry-run: skipping F1 check and marker $out/$marker"
    return 0
  fi
  grep -q '^Instance Segmentation F1 score:' "$FF3D_ROOT/$out/evaluation_total_test.txt" \
    || ff3d_die "no F1 line in $out/evaluation_total_test.txt"
  grep '^Instance Segmentation F1 score:' "$FF3D_ROOT/$out/evaluation_total_test.txt" | tail -1
  ff3d_run touch "$FF3D_ROOT/$out/$marker"
}

ff3d_log "release eval start (root $FF3D_ROOT, image $FF3D_IMAGE)"
[ -f "$FF3D_RELEASE_CKPT" ] || ff3d_die "missing $FF3D_RELEASE_CKPT (run benchmark/fetch_zenodo.sh)"
[ -f "$FF3D_OLD/tools/test.py" ] || ff3d_die "missing old worktree (run benchmark/setup_old_worktree.sh)"
[ -f "$FF3D_DATA/meta_data/test_list.txt" ] || ff3d_die "missing $FF3D_DATA/meta_data/test_list.txt"
# `|| true`: grep -c exits 1 on an empty list, which set -e would turn into a silent death.
N_TEST="$(grep -c . "$FF3D_DATA/meta_data/test_list.txt" || true)"
[ "${N_TEST:-0}" -gt 0 ] || ff3d_die "empty $FF3D_DATA/meta_data/test_list.txt (no test scans listed)"

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

run_test_stage fixed work_dirs/clean_forestformer/epoch_3000_converted.pth "$FIXED_DIR"
run_test_stage old   work_dirs/clean_forestformer/epoch_3000_raw.pth       "$OLD_DIR"

run_eval_stage "$FIXED_DIR"
run_eval_stage "$OLD_DIR"
ff3d_log "release eval finished"
