"""A fake `docker` binary for the benchmark scripts' REAL-run (non-dry-run) tests.

The dry-run tests cover what the scripts would print; they cannot cover the parts that only
happen on a real run: the ply-count postcondition, the marker writes, the "adopt an
already-complete output dir" branch and the tolerated non-zero exit of the old tools/test.py.
Those need the docker invocations to actually "do" something, so FF3D_DOCKER is pointed at
the shell script written by write_fake_docker() below. It understands the four commands the
benchmark scripts run inside the container and fakes exactly their observable side effects:

  tools/fix_spconv_checkpoint.py   exit FAKE_FIX_RC (default 0 == "input is raw")
  tools/train.py                   create <work-dir>/epoch_<max_epochs>.pth, exit FAKE_TRAIN_RC
  tools/test.py                    create FAKE_N_PLY *.ply in <work-dir>, then exit
                                   FAKE_OLD_TEST_RC when the command line goes through
                                   old_prelude.sh (the old variant), else FAKE_TEST_RC
  tools/final_eval.py              write an evaluation_total_test.txt with an F1 line

Every invocation is appended to $FAKE_LOG, one line per call, so a test can assert that a
stage did NOT run at all.

No docker, no GPU, no torch is involved; the script is plain POSIX-ish bash.
"""
from pathlib import Path

FAKE_DOCKER = r"""#!/usr/bin/env bash
# fake docker -- see tests/benchmark_fakes.py
set -u
args=("$@")
all="$*"
[ -n "${FAKE_LOG:-}" ] && printf '%s\n' "$all" >> "$FAKE_LOG"

wd=""
for ((i=0; i<${#args[@]}; i++)); do
  if [ "${args[i]}" = "--work-dir" ]; then wd="${args[i+1]}"; fi
done

case "$all" in
  *unfix_spconv_checkpoint*)
    exit 0
    ;;
  *fix_spconv_checkpoint*)
    exit "${FAKE_FIX_RC:-0}"
    ;;
  *tools/train.py*)
    ep="$(printf '%s\n' $all | sed -n 's/^train_cfg.max_epochs=//p' | head -1)"
    mkdir -p "$FF3D_ROOT/$wd"
    : > "$FF3D_ROOT/$wd/epoch_${ep:-200}.pth"
    exit "${FAKE_TRAIN_RC:-0}"
    ;;
  *tools/test.py*)
    mkdir -p "$FF3D_ROOT/$wd"
    n="${FAKE_N_PLY:-0}"
    i=1
    while [ "$i" -le "$n" ]; do
      printf 'fresh-ply-%s\n' "$i" > "$FF3D_ROOT/$wd/scan_${i}.ply"
      i=$((i+1))
    done
    case "$all" in
      *old_prelude.sh*) exit "${FAKE_OLD_TEST_RC:-0}" ;;
      *) exit "${FAKE_TEST_RC:-0}" ;;
    esac
    ;;
  *tools/final_eval.py*)
    d="${args[$((${#args[@]}-1))]}"
    mkdir -p "$FF3D_ROOT/$d"
    printf 'Instance Segmentation F1 score: 0.9000\n' > "$FF3D_ROOT/$d/evaluation_total_test.txt"
    exit 0
    ;;
esac
exit 0
"""


def write_fake_docker(tmp_path: Path) -> Path:
    """Write the fake docker binary into tmp_path and return its path (set FF3D_DOCKER to it)."""
    p = tmp_path / "fake-docker"
    p.write_text(FAKE_DOCKER)
    p.chmod(0o755)
    return p
