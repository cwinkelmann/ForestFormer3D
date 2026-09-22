#!/usr/bin/env bash
# Shared settings and helpers for the Phase 2 benchmark scripts.
# Source it:  source "$(dirname "$0")/common.sh"
# Every path variable can be overridden from the environment (tests do that).
set -euo pipefail

FF3D_ROOT="${FF3D_ROOT:-/raid/cwinkelmann/ForestFormer3D}"
FF3D_IMAGE="${FF3D_IMAGE:-forestformer3d:cu118}"
FF3D_DOCKER="${FF3D_DOCKER:-docker}"          # set to "sudo docker" if needed
FF3D_SHM="${FF3D_SHM:-64g}"
FF3D_GPU="${FF3D_GPU:-0}"                     # explicit --gpus "device=$FF3D_GPU"
FF3D_OLD_COMMIT="${FF3D_OLD_COMMIT:-6a75c37}"
# The old-code worktree (main @ FF3D_OLD_COMMIT) MUST NOT live under
# $FF3D_ROOT/work_dirs: ff3d_docker_old also bind-mounts $FF3D_ROOT/work_dirs onto
# /workspace/work_dirs, and a worktree inside work_dirs/ would be a self-nesting bind
# mount (the container would see the old worktree inside its own work_dirs mount).
# FF3D_OLD_ROOT is therefore a sibling directory of the checkout, independently
# overridable (it does not derive from FF3D_ROOT).
FF3D_OLD_ROOT="${FF3D_OLD_ROOT:-/raid/cwinkelmann/ff3d-old-main}"
FF3D_OLD="$FF3D_OLD_ROOT"                     # alias: Tasks 3-5 scripts refer to FF3D_OLD
export FF3D_OLD FF3D_OLD_ROOT
FF3D_CONFIG="configs/oneformer3d_qs_radius16_qp300_2many.py"
FF3D_DATA="${FF3D_ROOT}/data/ForAINetV2"
FF3D_CKPT_DIR="${FF3D_ROOT}/work_dirs/clean_forestformer"
FF3D_RELEASE_CKPT="${FF3D_CKPT_DIR}/epoch_3000_fix.pth"
FF3D_LOGS="${FF3D_ROOT}/work_dirs/logs"
FF3D_BENCH_DIR="${FF3D_ROOT}/benchmark"
FF3D_DRY_RUN="${FF3D_DRY_RUN:-0}"             # 1: print docker commands instead of running them

ff3d_log() { echo "$(date '+%Y-%m-%dT%H:%M:%S') $*"; }   # portable (GNU and BSD date)
ff3d_die() { ff3d_log "ERROR: $*" >&2; exit 1; }

# ff3d_run <cmd...>: run <cmd>, or (FF3D_DRY_RUN=1) print it shell-quoted instead.
# Mirrors tools/inference_bluepoint.sh's run().
ff3d_run() {
  if [ "${FF3D_DRY_RUN:-0}" = "1" ]; then
    printf 'DRY:'
    printf ' %q' "$@"
    printf '\n'
  else
    "$@"
  fi
}

# Run <cmd...> inside the image with the main checkout mounted at /workspace.
# --gpus "device=$FF3D_GPU" pins one explicit physical GPU (the server has 8, shared
# with other users); never --gpus all with CUDA_VISIBLE_DEVICES masking, which still
# reserves/locks all devices for this container.
ff3d_docker() {
  ff3d_run $FF3D_DOCKER run --rm --gpus "device=${FF3D_GPU}" --shm-size="$FF3D_SHM" \
    -e PYTHONPATH=/workspace -w /workspace \
    -v "$FF3D_ROOT":/workspace \
    "$FF3D_IMAGE" "$@"
}

# Run <cmd...> inside the SAME image with the old worktree (main @ $FF3D_OLD_COMMIT) at
# /workspace. Three bind mounts on top of it:
#   FF3D_OLD_ROOT   -> /workspace              (the old worktree itself, a sibling
#                                                directory of FF3D_ROOT -- see the
#                                                FF3D_OLD_ROOT comment above)
#   main data/      -> /workspace/data          (preprocessed once by the main checkout)
#   main work_dirs/ -> /workspace/work_dirs     (checkpoints in, results out)
# plus benchmark/old_prelude.sh -> /old_prelude.sh (ro), which patches the old mmengine
# loops.py/base_model.py before exec-ing the real command.
# The image entrypoint copies /workspace/replace_mmdetection_files/transforms_3d.py into
# mmdet3d; because /workspace IS the old worktree here, the old code's own transforms_3d.py
# is applied automatically.
# The bind mount onto /workspace also SHADOWS the image's baked
# `ENTRYPOINT /workspace/docker/entrypoint.sh`: commit $FF3D_OLD_COMMIT predates that
# entrypoint script, so it has no docker/ dir of its own and the container would fail to
# exec. benchmark/setup_old_worktree.sh (Task 3) copies $FF3D_ROOT/docker/entrypoint.sh
# into $FF3D_OLD_ROOT/docker/entrypoint.sh to restore it under the mount; assert that here
# before every run.
ff3d_docker_old() {
  if [ "${FF3D_DRY_RUN:-0}" != "1" ]; then
    [ -f "$FF3D_OLD_ROOT/tools/test.py" ] || ff3d_die "old worktree missing: run benchmark/setup_old_worktree.sh"
    [ -x "$FF3D_OLD_ROOT/docker/entrypoint.sh" ] \
      || ff3d_die "old worktree lacks docker/entrypoint.sh -- run benchmark/setup_old_worktree.sh"
  fi
  ff3d_run $FF3D_DOCKER run --rm --gpus "device=${FF3D_GPU}" --shm-size="$FF3D_SHM" \
    -e PYTHONPATH=/workspace -w /workspace \
    -v "$FF3D_OLD_ROOT":/workspace \
    -v "$FF3D_ROOT/data":/workspace/data \
    -v "$FF3D_ROOT/work_dirs":/workspace/work_dirs \
    -v "$FF3D_BENCH_DIR/old_prelude.sh":/old_prelude.sh:ro \
    "$FF3D_IMAGE" bash /old_prelude.sh "$@"
}

# Preprocess ForAINetV2 (train, val and test splits) once. Marker: the test infos pkl.
ff3d_preprocess() {
  if [ -f "$FF3D_DATA/forainetv2_oneformer3d_infos_test.pkl" ] && [ "${FORCE_PREP:-0}" != "1" ]; then
    ff3d_log "preprocessing present, skipping (FORCE_PREP=1 to redo)"
    return 0
  fi
  [ -d "$FF3D_DATA/train_val_data" ] && [ -d "$FF3D_DATA/test_data" ] \
    || ff3d_die "data missing: run benchmark/fetch_zenodo.sh"
  ff3d_log "preprocessing: batch_load + create_data"
  ff3d_docker bash -c "cd data/ForAINetV2 && python batch_load_ForAINetV2_data.py && cd /workspace && python tools/create_data_forainetv2.py forainetv2"
  [ -f "$FF3D_DATA/forainetv2_oneformer3d_infos_test.pkl" ] || ff3d_die "create_data produced no test pkl"
  ff3d_log "preprocessing done"
}

# ff3d_prepare_checkpoint <in.pth> <out_stem>   (both relative to FF3D_ROOT)
# Produces <out_stem>_converted.pth (for the fixed tools/test.py), <out_stem>_raw.pth (for
# the old tools/test.py, which permutes in memory) and <out_stem>.layout.
# Prints the layout of <in.pth>: "converted" (fix script exited 2) or "raw" (exited 0).
ff3d_prepare_checkpoint() {
  local in="$1" stem="$2" rc
  local conv="${stem}_converted.pth" raw="${stem}_raw.pth" layout="${stem}.layout"
  [ -f "$FF3D_ROOT/$in" ] || ff3d_die "checkpoint missing: $FF3D_ROOT/$in"
  if [ -f "$FF3D_ROOT/$conv" ] && [ -f "$FF3D_ROOT/$raw" ] && [ -f "$FF3D_ROOT/$layout" ]; then
    cat "$FF3D_ROOT/$layout"
    return 0
  fi
  set +e
  ff3d_docker python tools/fix_spconv_checkpoint.py --in-path "$in" --out-path "$conv" >&2
  rc=$?
  set -e
  case "$rc" in
    0)
      ff3d_log "$in is RAW; converted -> $conv" >&2
      cp -f "$FF3D_ROOT/$in" "$FF3D_ROOT/$raw"
      echo raw > "$FF3D_ROOT/$layout"
      ;;
    2)
      ff3d_log "$in is already CONVERTED; deriving raw -> $raw" >&2
      cp -f "$FF3D_ROOT/$in" "$FF3D_ROOT/$conv"
      ff3d_docker python benchmark/unfix_spconv_checkpoint.py --in-path "$in" --out-path "$raw" >&2
      echo converted > "$FF3D_ROOT/$layout"
      ;;
    *) ff3d_die "fix_spconv_checkpoint.py failed with exit $rc" ;;
  esac
  cat "$FF3D_ROOT/$layout"
}

# ff3d_daemonize <name> <script> [args...]
# Re-executes <script> under nohup with FF3D_FOREGROUND=1 and exits. The log path is printed.
ff3d_daemonize() {
  local name="$1"; shift
  if [ "${FF3D_FOREGROUND:-0}" = "1" ]; then
    return 0
  fi
  mkdir -p "$FF3D_LOGS"
  local log="$FF3D_LOGS/${name}-$(date +%Y%m%d-%H%M%S).log"
  FF3D_FOREGROUND=1 nohup bash "$@" > "$log" 2>&1 &
  echo "started $name pid $! -- follow with: tail -f $log"
  exit 0
}
