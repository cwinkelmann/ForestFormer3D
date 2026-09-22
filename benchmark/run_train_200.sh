#!/usr/bin/env bash
# 200-epoch training run, old or fixed code, then test-split scoring of epoch_200.pth.
#
#   bash benchmark/wait_idle.sh -- bash benchmark/run_train_200.sh fixed
#   bash benchmark/wait_idle.sh -- bash benchmark/run_train_200.sh old
#
# The script daemonizes itself (nohup) unless FF3D_FOREGROUND=1; the log path is printed.
# Stages with markers: train (.done-train), prepare checkpoint, test (.done-test), eval
# (.done-eval). Training is resumable: a re-run after a crash passes --resume auto so
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
CFG_OPTS=(train_cfg.max_epochs="$EPOCHS" train_cfg.val_interval="$VAL_INTERVAL" default_hooks.checkpoint.max_keep_ckpts=2)
N_TEST="$(grep -c . "$FF3D_DATA/meta_data/test_list.txt")"
if [ "$VARIANT" = "old" ]; then RUNNER=ff3d_docker_old; else RUNNER=ff3d_docker; fi

ff3d_log "train-${VARIANT}-200 start (root $FF3D_ROOT, image $FF3D_IMAGE, epochs $EPOCHS)"
[ "$VARIANT" = "fixed" ] || [ -f "$FF3D_OLD/tools/test.py" ] || ff3d_die "missing old worktree (run benchmark/setup_old_worktree.sh)"
ff3d_preprocess
mkdir -p "$FF3D_ROOT/$WORK"

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
  [ -f "$FF3D_ROOT/$WORK/epoch_${EPOCHS}.pth" ] || ff3d_die "training finished without $WORK/epoch_${EPOCHS}.pth"
  touch "$FF3D_ROOT/$WORK/.done-train"
  ff3d_log "training done"
fi

# 2. prepare checkpoint layouts (expected: raw, because train.py saves spconv's native layout)
layout="$(ff3d_prepare_checkpoint "$WORK/epoch_${EPOCHS}.pth" "$WORK/epoch_${EPOCHS}")"
ff3d_log "epoch_${EPOCHS}.pth layout: $layout"
[ "$layout" = "raw" ] || ff3d_log "WARNING: freshly trained checkpoint reported as '$layout'; continuing with the derived files"
if [ "$VARIANT" = "old" ]; then CKPT="$WORK/epoch_${EPOCHS}_raw.pth"; else CKPT="$WORK/epoch_${EPOCHS}_converted.pth"; fi

# 3. test split inference through the same path as run_release_eval.sh
if [ -f "$FF3D_ROOT/$TEST_OUT/.done-test" ]; then
  ff3d_log "test.py already done ($TEST_OUT)"
else
  mkdir -p "$FF3D_ROOT/$TEST_OUT"
  rm -f "$FF3D_ROOT/$TEST_OUT"/*.ply
  ff3d_log "tools/test.py ($VARIANT) $CKPT -> $TEST_OUT"
  "$RUNNER" python tools/test.py "$FF3D_CONFIG" "$CKPT" --work-dir "$TEST_OUT"
  n="$(find "$FF3D_ROOT/$TEST_OUT" -maxdepth 1 -name '*.ply' | wc -l | tr -d ' ')"
  [ "$n" = "$N_TEST" ] || ff3d_die "$VARIANT produced $n ply files, expected $N_TEST in $TEST_OUT"
  touch "$FF3D_ROOT/$TEST_OUT/.done-test"
fi

# 4. score with the fixed final_eval.py from the main checkout
if [ -f "$FF3D_ROOT/$TEST_OUT/.done-eval" ]; then
  ff3d_log "final_eval already done ($TEST_OUT)"
else
  rm -f "$FF3D_ROOT/$TEST_OUT/evaluation_total_test.txt"
  ff3d_docker python tools/final_eval.py "$TEST_OUT"
  grep -q '^Instance Segmentation F1 score:' "$FF3D_ROOT/$TEST_OUT/evaluation_total_test.txt" \
    || ff3d_die "no F1 line in $TEST_OUT/evaluation_total_test.txt"
  touch "$FF3D_ROOT/$TEST_OUT/.done-eval"
fi
grep '^Instance Segmentation F1 score:' "$FF3D_ROOT/$TEST_OUT/evaluation_total_test.txt" | tail -1
ff3d_log "train-${VARIANT}-200 finished"
