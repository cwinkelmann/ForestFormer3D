#!/usr/bin/env bash
# Preprocess + inference + final_eval for ONE thinned test set.
#
#   FF3D_GPU=3 bash benchmark/thin_eval.sh uniform u
#   FF3D_GPU=2 bash benchmark/thin_eval.sh canopy  c
#   FF3D_GPU=2 bash benchmark/thin_eval.sh canopy  c -step model.test_cfg.region_step_factor=0.5
#   FF3D_DRY_RUN=1 bash benchmark/thin_eval.sh uniform u     # print the docker commands only
#
#   $1 mode    uniform | canopy (names the scan list written by thin_make_sets.sh)
#   $2 letter  u | c             (names data/ForAINetV2/test_data_thin<D><letter>)
#   $3 suffix  optional tag appended to the output dir, e.g. "-step"
#   $4.. extra --cfg-options key=value pairs appended to tools/test.py
#
# Everything lands under work_dirs/logs/thin/<mode><suffix>/:
#   forainetv2_oneformer3d_infos_test.pkl   private info pkl (create_data --out-dir)
#   out/<stem>.ply                          28 result clouds (tools/test.py --work-dir)
#   out/evaluation_total_test.txt           tools/final_eval.py
#   t_start, t_end                          unix seconds around the inference step
#
# The tracked data/ForAINetV2/meta_data/test_list.txt is never read or written: the scan
# list is work_dirs/logs/thin/<mode>/scan_list.txt and the train/val lists are pointed at an
# empty file, so batch_load only exports the 28 thinned scans. Their exports
# (forainetv2_instance_data/<stem>_thin*_*.npy) are new file names, so the full-density
# exports used by the release benchmark are left untouched.
# See docs/benchmarks/2026-09-23-als-density-eval.md.
set -euo pipefail
source "$(dirname "$0")/common.sh"

[ $# -ge 2 ] || ff3d_die "usage: $0 <mode> <letter> [suffix] [cfg-options...]"
mode="$1"; letter="$2"; suffix="${3:-}"
if [ $# -ge 3 ]; then shift 3; else shift 2; fi

FF3D_THIN_DENSITY="${FF3D_THIN_DENSITY:-25}"
tag="${mode}${suffix}"
here="work_dirs/logs/thin/${tag}"
list="/workspace/work_dirs/logs/thin/${mode}/scan_list.txt"
empty="/workspace/work_dirs/logs/thin/empty_list.txt"

ff3d_run mkdir -p "$FF3D_ROOT/$here"
[ "${FF3D_DRY_RUN:-0}" = "1" ] || : > "$FF3D_ROOT/work_dirs/logs/thin/empty_list.txt"

if [ ! -f "$FF3D_ROOT/${here}/forainetv2_oneformer3d_infos_test.pkl" ]; then
  ff3d_log "preprocess ${tag}"
  ff3d_docker bash -c "cd data/ForAINetV2 && python batch_load_ForAINetV2_data.py \
      --test_scan_names_file ${list} --train_scan_names_file ${empty} --val_scan_names_file ${empty} \
      --test_forainetv2_dir test_data_thin${FF3D_THIN_DENSITY}${letter} \
    && cd /workspace && python tools/create_data_forainetv2.py forainetv2 \
      --test-list ${list} --splits test --out-dir /workspace/${here}"
fi
if [ "${FF3D_DRY_RUN:-0}" != "1" ]; then
  [ -f "$FF3D_ROOT/${here}/forainetv2_oneformer3d_infos_test.pkl" ] \
    || ff3d_die "create_data produced no test pkl for ${tag}"
fi

ff3d_log "inference ${tag} on GPU ${FF3D_GPU}"
[ "${FF3D_DRY_RUN:-0}" = "1" ] || date +%s > "$FF3D_ROOT/${here}/t_start"
ff3d_docker python tools/test.py "$FF3D_CONFIG" \
  work_dirs/clean_forestformer/epoch_3000_converted.pth \
  --work-dir "${here}/out" \
  --cfg-options "test_dataloader.dataset.ann_file=/workspace/${here}/forainetv2_oneformer3d_infos_test.pkl" \
                randomness.seed=0 "$@"
[ "${FF3D_DRY_RUN:-0}" = "1" ] || date +%s > "$FF3D_ROOT/${here}/t_end"

ff3d_log "final_eval ${tag}"
ff3d_docker python tools/final_eval.py "/workspace/${here}/out"
ff3d_log "MODE_DONE ${tag}"
