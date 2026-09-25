#!/usr/bin/env bash
# Two-pass ("bluepoint") inference: run the model, re-run it on the points that
# were left unsegmented, merge, evaluate. Every path and threshold is an
# environment variable; DRY_RUN=1 prints the commands instead of running them.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_DIR="${WORK_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"
CONFIG_FILE="${CONFIG_FILE:-$WORK_DIR/configs/oneformer3d_qs_radius16_qp300_2many.py}"
MODEL_PATH="${MODEL_PATH:-$WORK_DIR/work_dirs/clean_forestformer/epoch_3000_fix.pth}"
ITERATIONS="${ITERATIONS:-2}"
SCORE_TH="${SCORE_TH:-0.4}"
BLUEPOINTS_DIR="${BLUEPOINTS_DIR:-$WORK_DIR/work_dirs/bluepoints}"
# tools/create_data_forainetv2.py defaults --root-path to ./data/ForAINetV2
# relative to its own cwd; overriding DATA_ROOT here does not move where it
# reads/writes the info pkls.
DATA_ROOT="${DATA_ROOT:-$WORK_DIR/data/ForAINetV2}"
TEST_LIST_INIT="${TEST_LIST_INIT:-$DATA_ROOT/meta_data/test_list_initial.txt}"
# Default to a private temp file so the tracked meta_data/test_list.txt is
# never mutated by this script (same rule as "no sed on the tracked
# config"); only the temp file this script itself created is cleaned up on
# exit -- a caller-supplied TEST_LIST is never deleted.
if [[ -z "${TEST_LIST:-}" ]]; then
    TEST_LIST="$(mktemp "${TMPDIR:-/tmp}/ff3d_test_list.XXXXXX")"
    TEST_LIST_IS_TEMP=1
else
    TEST_LIST_IS_TEMP=0
fi
trap '[[ "$TEST_LIST_IS_TEMP" == 1 ]] && rm -f "$TEST_LIST"' EXIT
TEST_DATA_DIR="${TEST_DATA_DIR:-$DATA_ROOT/test_data}"
DRY_RUN="${DRY_RUN:-0}"
export PYTHONPATH="${PYTHONPATH:-$WORK_DIR}"

run() {                       # run "$@" or echo it, shell-quoted
    if [[ "$DRY_RUN" == "1" ]]; then
        printf 'DRY:'
        printf ' %q' "$@"
        printf '\n'
    else
        "$@"
    fi
}

write_list() {                # write_list <name> <file>
    if [[ "$DRY_RUN" == "1" ]]; then
        echo "DRY: echo $1 > $2"
    else
        echo "$1" > "$2"
    fi
}

[[ -f "$TEST_LIST_INIT" ]] || { echo "missing $TEST_LIST_INIT" >&2; exit 1; }

if [[ -d "$DATA_ROOT/forainetv2_instance_data" ]]; then
    run find "$DATA_ROOT/forainetv2_instance_data" -type f -name "*bluepoints*" -delete
fi

while IFS= read -r scan_name || [[ -n "${scan_name:-}" ]]; do
    [[ -z "$scan_name" ]] && continue
    echo "Processing: $scan_name"
    iteration=1
    current_scan_name="$scan_name"

    while [[ "$iteration" -le "$ITERATIONS" ]]; do
        echo "Iteration $iteration for $current_scan_name"
        write_list "$current_scan_name" "$TEST_LIST"

        ( cd "$DATA_ROOT" && run python batch_load_ForAINetV2_data.py --test_scan_names_file "$TEST_LIST" )
        # --test-list/--splits: build the info pkl from the private TEST_LIST only,
        # so neither the tracked meta_data/test_list.txt nor the train/val pkls are
        # involved (create_data would otherwise read meta_data/test_list.txt).
        ( cd "$WORK_DIR" && run python tools/create_data_forainetv2.py forainetv2 \
            --test-list "$TEST_LIST" --splits test )
        ( cd "$WORK_DIR" && CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" run python tools/test.py "$CONFIG_FILE" "$MODEL_PATH" \
            --work-dir "$BLUEPOINTS_DIR" \
            --cfg-options "model.test_cfg.score_th=$SCORE_TH" "model.test_cfg.output_dir=$BLUEPOINTS_DIR" )

        new_pre_path="$BLUEPOINTS_DIR/${scan_name}_${iteration}.ply"
        if [[ ! -f "$new_pre_path" ]]; then
            echo "No prediction file $new_pre_path; ending iterations for $scan_name."
            break
        fi
        bluepoints_file="${scan_name}_bluepoints_${iteration}.ply"
        bluepoints_path="$BLUEPOINTS_DIR/$bluepoints_file"
        if [[ ! -f "$bluepoints_path" ]]; then
            echo "No bluepoints file $bluepoints_path; ending iterations for $scan_name."
            break
        fi
        run cp "$bluepoints_path" "$TEST_DATA_DIR/"
        current_scan_name="${bluepoints_file%.ply}"
        iteration=$((iteration + 1))
    done

    echo "Merging results for $scan_name"
    ( cd "$WORK_DIR" && run python tools/merge_prediction.py "$scan_name" "$BLUEPOINTS_DIR" "$ITERATIONS" )
    echo "Finished processing $scan_name."
done < "$TEST_LIST_INIT"

for ((i = 1; i <= ITERATIONS; i++)); do
    round_dir="$BLUEPOINTS_DIR/round_$i"
    echo "Evaluating results in: $round_dir"
    ( cd "$WORK_DIR" && run python tools/final_eval.py "$round_dir" )
    for noise_dir in "$BLUEPOINTS_DIR"/round_"$i"_after_remove_noise_*; do
        [[ -d "$noise_dir" ]] || continue
        echo "Evaluating results in: $noise_dir"
        ( cd "$WORK_DIR" && run python tools/final_eval.py "$noise_dir" )
    done
done

echo "All test cases processed."
