#!/usr/bin/env bash
# 200-epoch training run, old or fixed code, then test-split scoring of epoch_200.pth.
#
#   bash benchmark/wait_idle.sh -- bash benchmark/run_train_200.sh fixed
#   bash benchmark/wait_idle.sh -- bash benchmark/run_train_200.sh old
#
# The script daemonizes itself (nohup) unless FF3D_FOREGROUND=1; the log path is printed.
# Stages with markers, all inside the stage's own output dir: train
# (work_dirs/bench-<variant>-200/.done-train), prepare checkpoint (epoch_<N>.layout), test
# (work_dirs/bench-<variant>-200/test/.done-test), eval (.../test/.done-eval). A test stage
# whose output dir already holds the full set of result PLYs is adopted rather than re-run;
# FF3D_FORCE=1 overrides that and re-infers from scratch. Training is resumable: a re-run after a crash passes --resume auto so
# mmengine continues from the last checkpoint in the work dir (the --resume flag and its
# 'auto' behaviour are identical between the old and fixed tools/train.py -- verified
# against git 6a75c37 and the current tree: both parse --resume the same way and set
# cfg.resume = True when args.resume == 'auto').
#
# Checkpoint layout rule (see also common.sh's ff3d_prepare_checkpoint): a freshly trained
# checkpoint saved by spconv 2.3.6 is RAW (out, k, k, k, in) -- that is what
# fix_spconv_checkpoint.py expects and what ff3d_prepare_checkpoint is expected to report
# (exit 0, "raw"). The script does not rely on that expectation holding: it always feeds
# the fixed tools/test.py the *_converted.pth file and the old tools/test.py (which
# permutes weights in memory) the *_raw.pth file, whichever way ff3d_prepare_checkpoint's
# exit code went, and logs a WARNING if the freshly trained checkpoint was reported as
# anything other than "raw".
#
# Preprocessing: uses common.sh's ff3d_preprocess, whose own marker is the test infos pkl
# (data/ForAINetV2/forainetv2_oneformer3d_infos_test.pkl) -- not a separate .done-preprocess
# file. This is the SAME marker benchmark/run_release_eval.sh's ff3d_preprocess call uses,
# so whichever of the two benchmark scripts runs first does the (one-time) preprocessing
# and the other just sees it already done.
#
# GPU idle guard: FF3D_WAIT_IDLE=1 makes this script block (via benchmark/wait_idle.sh)
# until the configured GPU ($FF3D_GPU) is idle before doing anything else. This is optional
# and off by default; the primary intended usage is still wrapping the whole invocation
# externally, as in the usage examples above.
#
# Warm-up length (model.prepare_epoch): the config's default is 1000 (out of a 3000-epoch
# paper run) -- the instance decoder stays frozen/untrained until
# `epoch > model.prepare_epoch`. A 200-epoch benchmark run at that same default would NEVER
# leave warm-up, so the decoder is never trained and val/test F1 comes back 0.0 for both
# variants (this is exactly what the first real runs hit). --cfg-options
# model.prepare_epoch=<N> is always added, default FF3D_PREPARE_EPOCH=60: the same ~30%
# warm-up ratio as the paper's 1000/3000 (60/200 = 30%). Applies identically to both
# variants: the old ForAINetV2OneFormer3D_XAwarequery.__init__ (git 6a75c37,
# oneformer3d/oneformer3d.py) takes `prepare_epoch` as a constructor arg read straight from
# the config the same way the fixed model class does, so overriding model.prepare_epoch via
# --cfg-options works the same for both. FF3D_EXTRA_CFG_OPTIONS (space-separated
# key=value tokens, e.g. "a.b=1 c.d=2") is appended to the same --cfg-options call for
# future runs that need to override something else, without editing this script.
#
# FF3D_DRY_RUN=1: this script writes NOTHING under work_dirs, same contract as common.sh
# (commits 6738a5f, b7000f8, 5c723ec):
#   - every mutating filesystem command (mkdir, touch, rm) goes through common.sh's
#     ff3d_run, which prints it instead of executing it, exactly like ff3d_docker/
#     ff3d_docker_old already do for the docker invocations themselves.
#   - every file-existence POSTcondition check on an artefact this script's own preceding
#     command was supposed to just produce (epoch_<N>.pth after train.py, the test-split
#     .ply count after test.py, the F1 line in evaluation_total_test.txt after
#     final_eval.py) is skipped with a "DRY: (postcondition skipped) <path>" line instead
#     of ff3d_die -- because under a dry run the preceding docker command was only
#     printed, never actually run, so none of these artefacts exist yet, and (since
#     nothing is written) never will.
# PREconditions on things the user must already have supplied (the config, the data, the
# old worktree) are untouched by any of this and still die for real under a dry run.
# common.sh's ff3d_prepare_checkpoint (stage 2) has matching dry-run behaviour of its own:
# it never touches the filesystem and cannot report a real layout (that requires actually
# running the container), so `layout` here becomes its "DRY: ..." explanation text rather
# than "raw"/"converted", which routes to the (harmless, informational) WARNING branch
# below every dry run -- expected, not a bug.
set -euo pipefail
source "$(dirname "$0")/common.sh"

VARIANT="${1:-}"
case "$VARIANT" in
  old|fixed) ;;
  *) echo "usage: $0 <old|fixed>" >&2; exit 2 ;;
esac
ff3d_daemonize "train-${VARIANT}-200" "$0" "$@"

if [ "${FF3D_WAIT_IDLE:-0}" = "1" ]; then
  bash "$(dirname "$0")/wait_idle.sh" -- true
fi

WORK="work_dirs/bench-${VARIANT}-200"
TEST_OUT="$WORK/test"
EPOCHS="${FF3D_EPOCHS:-200}"
VAL_INTERVAL="${FF3D_VAL_INTERVAL:-20}"
PREPARE_EPOCH="${FF3D_PREPARE_EPOCH:-60}"
CFG_OPTS=(train_cfg.max_epochs="$EPOCHS" train_cfg.val_interval="$VAL_INTERVAL" default_hooks.checkpoint.max_keep_ckpts=2 model.prepare_epoch="$PREPARE_EPOCH")
if [ -n "${FF3D_EXTRA_CFG_OPTIONS:-}" ]; then
  # Intentional word-splitting: FF3D_EXTRA_CFG_OPTIONS is a space-separated list of
  # key=value tokens, each becoming its own --cfg-options item.
  CFG_OPTS+=(${FF3D_EXTRA_CFG_OPTIONS})
fi
if [ "$VARIANT" = "old" ]; then RUNNER=ff3d_docker_old; else RUNNER=ff3d_docker; fi

ff3d_log "train-${VARIANT}-200 start (root $FF3D_ROOT, image $FF3D_IMAGE, epochs $EPOCHS)"
[ "$VARIANT" = "fixed" ] || [ -f "$FF3D_OLD/tools/test.py" ] || ff3d_die "missing old worktree (run benchmark/setup_old_worktree.sh)"
# Preconditions first, then read the test list (run_release_eval.sh does the same): a missing
# file used to die with a raw `grep: ...: No such file` and an EMPTY one made `grep -c` exit 1,
# which set -e turned into a silent death with no message at all.
[ -f "$FF3D_DATA/meta_data/test_list.txt" ] || ff3d_die "missing $FF3D_DATA/meta_data/test_list.txt"
N_TEST="$(grep -c . "$FF3D_DATA/meta_data/test_list.txt" || true)"
[ "${N_TEST:-0}" -gt 0 ] || ff3d_die "empty $FF3D_DATA/meta_data/test_list.txt (no test scans listed)"
ff3d_preprocess
ff3d_run mkdir -p "$FF3D_ROOT/$WORK"

# 1. train
if [ -f "$FF3D_ROOT/$WORK/.done-train" ]; then
  ff3d_log "training already done ($WORK)"
else
  RESUME=()
  if [ -f "$FF3D_ROOT/$WORK/last_checkpoint" ]; then
    RESUME=(--resume auto)
    ff3d_log "resuming from $(cat "$FF3D_ROOT/$WORK/last_checkpoint")"
  fi
  ff3d_log "tools/train.py ($VARIANT) -> $WORK"
  # ${RESUME[@]+...} keeps an empty array legal under set -u on bash < 4.4
  "$RUNNER" python tools/train.py "$FF3D_CONFIG" --work-dir "$WORK" ${RESUME[@]+"${RESUME[@]}"} --cfg-options "${CFG_OPTS[@]}"
  if [ "${FF3D_DRY_RUN:-0}" = "1" ]; then
    echo "DRY: (postcondition skipped) $FF3D_ROOT/$WORK/epoch_${EPOCHS}.pth"
  else
    [ -f "$FF3D_ROOT/$WORK/epoch_${EPOCHS}.pth" ] || ff3d_die "training finished without $WORK/epoch_${EPOCHS}.pth"
  fi
  ff3d_run touch "$FF3D_ROOT/$WORK/.done-train"
  ff3d_log "training done"
fi

# 2. prepare checkpoint layouts (expected: raw, because train.py saves spconv's native layout)
layout="$(ff3d_prepare_checkpoint "$WORK/epoch_${EPOCHS}.pth" "$WORK/epoch_${EPOCHS}")"
ff3d_log "epoch_${EPOCHS}.pth layout: $layout"
[ "$layout" = "raw" ] || ff3d_log "WARNING: freshly trained checkpoint reported as '$layout'; continuing with the derived files"
if [ "$VARIANT" = "old" ]; then CKPT="$WORK/epoch_${EPOCHS}_raw.pth"; else CKPT="$WORK/epoch_${EPOCHS}_converted.pth"; fi

# 3. test split inference through the same path as run_release_eval.sh
# Two old-variant asymmetries, identical to run_release_eval.sh's run_test_stage (see
# docs/benchmarks/RUNBOOK-carrot.md section 7):
#   - a NON-ZERO exit of the old tools/test.py is tolerated (its evaluator @ 6a75c37 always
#     crashes with an IndexError AFTER every result .ply has been written); the N_TEST
#     ply-count check stays the real postcondition. The fixed runner still dies on non-zero.
#   - a $TEST_OUT that already holds exactly N_TEST .ply files is ADOPTED (marker written,
#     stage skipped) instead of wiped and re-inferred; FF3D_FORCE=1 forces the rm + re-run.
if [ -f "$FF3D_ROOT/$TEST_OUT/.done-test" ]; then
  ff3d_log "test.py already done ($TEST_OUT)"
elif [ "$(ff3d_count_ply "$FF3D_ROOT/$TEST_OUT")" = "$N_TEST" ] && [ "${FF3D_FORCE:-0}" != "1" ]; then
  if [ "${FF3D_DRY_RUN:-0}" = "1" ]; then
    ff3d_log "DRY: (would adopt) $N_TEST existing ply files in $TEST_OUT -- stage would be skipped (FF3D_FORCE=1 to re-run)"
  else
    ff3d_log "test.py: adopting $N_TEST existing PLYs in $TEST_OUT (FF3D_FORCE=1 to re-run)"
    ff3d_run touch "$FF3D_ROOT/$TEST_OUT/.done-test"
  fi
else
  ff3d_run mkdir -p "$FF3D_ROOT/$TEST_OUT"
  ff3d_run rm -f "$FF3D_ROOT/$TEST_OUT"/*.ply
  ff3d_log "tools/test.py ($VARIANT) $CKPT -> $TEST_OUT"
  rc=0
  "$RUNNER" python tools/test.py "$FF3D_CONFIG" "$CKPT" --work-dir "$TEST_OUT" || rc=$?
  if [ "$rc" != "0" ]; then
    [ "$VARIANT" = "old" ] || ff3d_die "$VARIANT tools/test.py failed with exit $rc"
    ff3d_log "WARNING: old tools/test.py exited $rc -- expected (its evaluator crashes after writing all PLYs); checking the ply count instead"
  fi
  if [ "${FF3D_DRY_RUN:-0}" = "1" ]; then
    echo "DRY: (postcondition skipped) $FF3D_ROOT/$TEST_OUT/*.ply (expected $N_TEST)"
  else
    n="$(ff3d_count_ply "$FF3D_ROOT/$TEST_OUT")"
    [ "$n" = "$N_TEST" ] || ff3d_die "$VARIANT produced $n ply files, expected $N_TEST in $TEST_OUT"
  fi
  ff3d_run touch "$FF3D_ROOT/$TEST_OUT/.done-test"
fi

# 4. score with the fixed final_eval.py from the main checkout
if [ -f "$FF3D_ROOT/$TEST_OUT/.done-eval" ]; then
  ff3d_log "final_eval already done ($TEST_OUT)"
else
  ff3d_run rm -f "$FF3D_ROOT/$TEST_OUT/evaluation_total_test.txt"
  ff3d_docker python tools/final_eval.py "$TEST_OUT"
  if [ "${FF3D_DRY_RUN:-0}" = "1" ]; then
    echo "DRY: (postcondition skipped) $FF3D_ROOT/$TEST_OUT/evaluation_total_test.txt"
  else
    grep -q '^Instance Segmentation F1 score:' "$FF3D_ROOT/$TEST_OUT/evaluation_total_test.txt" \
      || ff3d_die "no F1 line in $TEST_OUT/evaluation_total_test.txt"
  fi
  ff3d_run touch "$FF3D_ROOT/$TEST_OUT/.done-eval"
fi
if [ "${FF3D_DRY_RUN:-0}" = "1" ]; then
  echo "DRY: (postcondition skipped) $FF3D_ROOT/$TEST_OUT/evaluation_total_test.txt (F1 report line)"
else
  grep '^Instance Segmentation F1 score:' "$FF3D_ROOT/$TEST_OUT/evaluation_total_test.txt" | tail -1
fi
ff3d_log "train-${VARIANT}-200 finished"
