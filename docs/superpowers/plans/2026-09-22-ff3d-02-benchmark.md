# ForestFormer3D Phase 2: Benchmark on carrot Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce `docs/benchmarks/<date>-carrot-ff3d.md` with two tables measured on the H100 server carrot: the released `epoch_3000` checkpoint under old vs fixed inference, and a 200-epoch training run under old vs fixed code, both scored on the ForAINetV2 test split by the fixed `tools/final_eval.py`.

**Architecture:** Six small bash scripts plus one Python collector under `benchmark/`. All GPU work runs inside the Phase 0 image `forestformer3d:cu118`; "old code" is a `git worktree` of `main` (6a75c37) at `work_dirs/old-main` that is mounted at `/workspace` in the same image with the main checkout's `data/` and `work_dirs/` bind-mounted over it, so only code differs and every output lands under the main checkout's `work_dirs/`. Long jobs re-exec themselves under `nohup` and write stage markers so they can be re-run after an SSH drop. `collect.py` is pure Python (numpy only) and is unit-tested on the Mac against synthetic `evaluation_total_test.txt`, `scalars.json` and mmengine log fixtures.

**Tech Stack:** bash (`set -euo pipefail`), curl, unzip, python3 on the carrot host; Docker with NVIDIA runtime; PyTorch 2.0.1+cu118 / mmengine 0.7.3 / spconv-cu118 2.3.6 inside the image; numpy + pytest for the CPU tests.

**Spec:** `docs/superpowers/specs/2026-09-22-ff3d-fixes-benchmark-tegel-design.md` (section 5, with 2, 7, 8). Interface contract: this plan uses the names fixed in the shared contract (Phase 2 line: `benchmark/fetch_zenodo.sh`, `benchmark/run_release_eval.sh`, `benchmark/run_train_200.sh <old|fixed>`, `benchmark/collect.py -> docs/benchmarks/<YYYY-MM-DD>-carrot-ff3d.md`, old code = git worktree of main (6a75c37) at `work_dirs/old-main` inside the same image).

## Global Constraints

- Phases 0 and 1 are merged before this plan starts: image `forestformer3d:cu118` exists on carrot; `docker/entrypoint.sh` copies `/workspace/replace_mmdetection_files/transforms_3d.py` into the mmdet3d site-packages then `exec "$@"`; `tools/test.py` takes `--work-dir` and does NOT permute weights; `tools/fix_spconv_checkpoint.py --in-path --out-path` exits 2 and prints `already converted` on a converted checkpoint; `tools/final_eval.py` accumulates correctly across files; `pyproject.toml` with `[tool.pytest.ini_options] testpaths=["tests"]`, `addopts="-m 'not gpu'"` exists; `tests/conftest.py` exists.
- Repo on carrot: `/raid/cwinkelmann/ForestFormer3D`, a clone of branch `fix/review-findings`, repo mounted at `/workspace`, `PYTHONPATH=/workspace`. Run pattern: `docker run --rm --gpus all --shm-size=64g -v <checkout>:/workspace forestformer3d:cu118 <cmd>`.
- Old code = `main` at commit `6a75c37`. Its `tools/test.py` permutes 5-D `unet*`/`input_conv*` weights with `permute(1, 2, 3, 4, 0)` in memory, saves a temp checkpoint, and sets `cfg.model.test_cfg['output_dir'] = cfg.work_dir` (current checkout `tools/test.py` lines 121-149). Its `predict()` writes `os.path.join(output_path, f"{current_filename}.ply")` (`oneformer3d/oneformer3d.py` line 2572) where `current_filename` is the stem of `lidar_path` and `output_path = self.test_cfg.get('output_dir', ...)` (line 2294). It only takes that branch when `'test' in lidar_path` (line 2270); test scan names end in `_test` (e.g. `NIBIO_NIBIO_plot_1_annotated_test`), so the branch is taken. Its `train_step`/`test_step` take an `epoch` kwarg, so it needs `replace_mmdetection_files/loops.py` and `base_model.py` copied into the mmengine site-packages (old README step 4).
- Zenodo record 16742708 (queried 2026-09-22 via `https://zenodo.org/api/records/16742708`) has exactly three files: `train_val_data.zip` (2,426,597,147 B, md5 `5a63cc1cbe88edd9ebec28ad7e46f79b`), `test_data.zip` (377,105,598 B, md5 `1c00a0f0b89f03b74064432162619136`), `clean_forestformer.zip` (197,823,601 B, md5 `553d67379331966509076f3fbb409e57`). Download URL pattern `https://zenodo.org/api/records/16742708/files/<key>/content`. The scripts still discover files generically.
- Dataset lists: `data/ForAINetV2/meta_data/{train,val,test}_list.txt` = 46 / 15 / 27 scans. Preprocessing: `cd data/ForAINetV2 && python batch_load_ForAINetV2_data.py` then `python tools/create_data_forainetv2.py forainetv2` produce `data/ForAINetV2/forainetv2_instance_data/` and `data/ForAINetV2/forainetv2_oneformer3d_infos_{train,val,test}.pkl`.
- Config `configs/oneformer3d_qs_radius16_qp300_2many.py`: `train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=3000, val_interval=100)`, `default_hooks.checkpoint = dict(type='CheckpointHook', interval=1, max_keep_ckpts=3, save_optimizer=True)`, `default_hooks.logger = dict(type='LoggerHook', interval=20)`, `train_dataloader.batch_size=2` (46 scans -> 23 iterations per epoch, so mmengine logs one train record per epoch at the end of the epoch because `len(dataloader) <= interval`), `vis_backends` include `LocalVisBackend`. The val metric `UnifiedSegMetric` returns keys `mIoU, mIoU_binary, mMWCov, mMUCov, mPrecision, mRecall, F1, mPQ, mSQ, mRQ` with no prefix.
- mmengine 0.7.3 log layout for `--work-dir W`: `W/<timestamp>/<timestamp>.log` (lines `YYYY/MM/DD HH:MM:SS - mmengine - INFO - Epoch(train)  [E][I/N]  lr: ...  time: ...  loss: ...`), `W/<timestamp>/vis_data/scalars.json` (one JSON object per line; train records carry `loss`, `epoch`, `step`, `lr`, `time`, `data_time`, `memory`; val records carry the metric keys plus `step` = epoch), checkpoints `W/epoch_<E>.pth` and `W/last_checkpoint`.
- Weight layouts: spconv 2.3.6 stores `SparseConv3d`/`SubMConv3d`/`SparseInverseConv3d` weights as `(out_channels, kD, kH, kW, in_channels)`; `tools/train.py` saves that unchanged, so a freshly trained checkpoint is in the RAW layout. `fix_spconv_checkpoint.py` applies `permute(1, 2, 3, 4, 0)` -> `(kD, kH, kW, in_channels, out_channels)` = the CONVERTED layout that the fixed `tools/test.py` loads and that `epoch_3000_fix.pth` is expected to have (its name ends in `_fix`; spec 8 says do not assume, decide by the exit code). Inverse: `permute(4, 0, 1, 2, 3)`.
- Every shell script: `#!/usr/bin/env bash`, `set -euo pipefail`, absolute or `$FF3D_ROOT`-relative paths, no `sudo` baked in (`FF3D_DOCKER` variable selects `docker` or `sudo docker`).
- Every commit message ends with a blank line and `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`.
- No CI. CPU tests run on the Mac with `pytest`; carrot steps are the last tasks and need VPN or the `carrot` SSH alias.

---

## File map

| Path | Responsibility |
|------|----------------|
| `benchmark/common.sh` | Paths, image tag, `ff3d_docker`, `ff3d_docker_old`, `ff3d_preprocess`, `ff3d_prepare_checkpoint`, `ff3d_daemonize` |
| `benchmark/old_prelude.sh` | Runs inside the old-code container before the command: copies the old worktree's `loops.py` and `base_model.py` into mmengine |
| `benchmark/unfix_spconv_checkpoint.py` | Inverse of `tools/fix_spconv_checkpoint.py` (`permute(4, 0, 1, 2, 3)`) |
| `benchmark/fetch_zenodo.sh` | Idempotent download + unzip of record 16742708 (`--list` mode) |
| `benchmark/setup_old_worktree.sh` | `git worktree add work_dirs/old-main 6a75c37` + sanity checks |
| `benchmark/wait_idle.sh` | Launch a command only after the GPU has been idle for N consecutive samples |
| `benchmark/run_release_eval.sh` | Preprocess once, released checkpoint under old and fixed `test.py`, `final_eval.py` on both |
| `benchmark/run_train_200.sh <old\|fixed>` | 200-epoch training + test-split scoring of `epoch_200.pth` |
| `benchmark/collect.py` | Parse results and write `docs/benchmarks/<date>-carrot-ff3d.md` |
| `tests/test_benchmark_fetch.py` | Local HTTP fixture exercising `fetch_zenodo.sh` end to end |
| `tests/test_benchmark_unfix.py` | Round trip fix -> unfix on a synthetic state dict (skips without torch) |
| `tests/test_benchmark_collect.py` | Synthetic fixtures for `collect.py` |
| `docs/benchmarks/RUNBOOK-carrot.md` | Ordered commands for carrot |

Output layout on carrot (all under the main checkout, also visible to the old container):

```
work_dirs/clean_forestformer/epoch_3000_fix.pth        # from Zenodo
work_dirs/clean_forestformer/epoch_3000_converted.pth  # what fixed test.py loads
work_dirs/clean_forestformer/epoch_3000_raw.pth        # what old test.py loads
work_dirs/clean_forestformer/epoch_3000.layout         # "converted" or "raw" (layout of the Zenodo file)
work_dirs/bench-release-fixed/<scan>.ply + evaluation_total_test.txt
work_dirs/bench-release-old/<scan>.ply   + evaluation_total_test.txt
work_dirs/bench-old-200/{epoch_200.pth, <ts>/<ts>.log, <ts>/vis_data/scalars.json, test/...}
work_dirs/bench-fixed-200/{...same...}
work_dirs/logs/*.log                                    # nohup logs
work_dirs/old-main/                                     # git worktree of 6a75c37
```

---

### Task 1: Shared shell library, old-code prelude and checkpoint inverse

**Files:**
- Create: `benchmark/common.sh`
- Create: `benchmark/old_prelude.sh`
- Create: `benchmark/unfix_spconv_checkpoint.py`
- Modify: `.gitignore` (append ignore rules for data, downloads, work_dirs)
- Test: `tests/test_benchmark_unfix.py`

**Interfaces:**
- Consumes: `tools/fix_spconv_checkpoint.py --in-path --out-path` (exit 0 = converted written, exit 2 = "already converted"); `docker/entrypoint.sh` from Phase 0.
- Produces (sourced by every later script):
  - variables `FF3D_ROOT`, `FF3D_IMAGE`, `FF3D_DOCKER`, `FF3D_SHM`, `FF3D_OLD`, `FF3D_CONFIG`, `FF3D_DATA`, `FF3D_CKPT_DIR`, `FF3D_RELEASE_CKPT`, `FF3D_LOGS`, `FF3D_OLD_COMMIT`
  - `ff3d_docker <cmd...>`: run `<cmd>` in the image with the main checkout at `/workspace`
  - `ff3d_docker_old <cmd...>`: run `<cmd>` in the image with the old worktree at `/workspace`, main `data/` at `/workspace/data`, main `work_dirs/` at `/workspace/work_dirs`, after `old_prelude.sh`
  - `ff3d_preprocess`: runs batch_load + create_data once (marker: `data/ForAINetV2/forainetv2_oneformer3d_infos_test.pkl`)
  - `ff3d_prepare_checkpoint <in.pth (root-relative)> <out_stem (root-relative)>`: writes `<out_stem>_converted.pth`, `<out_stem>_raw.pth`, `<out_stem>.layout`; prints `converted` or `raw` (the layout of `<in.pth>`)
  - `ff3d_daemonize <name> "$0" "$@"`: re-exec the calling script under `nohup` unless `FF3D_FOREGROUND=1`
  - `benchmark/unfix_spconv_checkpoint.py --in-path X --out-path Y`

- [ ] **Step 1: Write the failing round-trip test**

```python
# tests/test_benchmark_unfix.py
"""fix_spconv_checkpoint.py followed by unfix_spconv_checkpoint.py is the identity.

Needs torch (the checkpoints are torch.save files). On the Mac without torch the
test is skipped; inside the container it runs as a plain CPU test.
"""
import subprocess
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

REPO = Path(__file__).resolve().parents[1]
FIX = REPO / "tools" / "fix_spconv_checkpoint.py"
UNFIX = REPO / "benchmark" / "unfix_spconv_checkpoint.py"


def _raw_state_dict():
    # raw spconv 2.x layout: (out, kD, kH, kW, in); channel counts > 3 like the real model
    # (input_conv in=3 out=32, unet 32..256); plus a non-5D and a non-unet key
    return {
        "unet.blocks.0.conv.weight": torch.arange(8 * 3 * 3 * 3 * 4, dtype=torch.float32).reshape(8, 3, 3, 3, 4),
        "input_conv.0.weight": torch.arange(32 * 3 * 3 * 3 * 3, dtype=torch.float32).reshape(32, 3, 3, 3, 3),
        "unet.blocks.0.i_branch.0.weight": torch.arange(8 * 1 * 1 * 1 * 4, dtype=torch.float32).reshape(8, 1, 1, 1, 4),
        "unet.blocks.0.bn.weight": torch.ones(8),
        "decoder.linear.weight": torch.ones(7, 7),
    }


def test_fix_then_unfix_is_identity(tmp_path):
    raw = tmp_path / "raw.pth"
    conv = tmp_path / "conv.pth"
    back = tmp_path / "back.pth"
    torch.save({"state_dict": _raw_state_dict(), "meta": {"epoch": 1}}, raw)

    r = subprocess.run([sys.executable, str(FIX), "--in-path", str(raw), "--out-path", str(conv)],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    sd = torch.load(conv, map_location="cpu")["state_dict"]
    assert tuple(sd["unet.blocks.0.conv.weight"].shape) == (3, 3, 3, 4, 8)
    assert tuple(sd["input_conv.0.weight"].shape) == (3, 3, 3, 3, 32)

    r = subprocess.run([sys.executable, str(UNFIX), "--in-path", str(conv), "--out-path", str(back)],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    got = torch.load(back, map_location="cpu")
    assert got["meta"] == {"epoch": 1}
    for k, v in _raw_state_dict().items():
        assert torch.equal(got["state_dict"][k], v), k


def test_unfix_refuses_raw_layout(tmp_path):
    raw = tmp_path / "raw.pth"
    out = tmp_path / "out.pth"
    torch.save({"state_dict": _raw_state_dict()}, raw)
    r = subprocess.run([sys.executable, str(UNFIX), "--in-path", str(raw), "--out-path", str(out)],
                       capture_output=True, text=True)
    assert r.returncode == 2
    assert "already raw" in r.stdout + r.stderr
    assert not out.exists()
```

- [ ] **Step 2: Run the test to verify it fails (or skips on the Mac)**

Run: `cd /Users/christian/ForestFormer3D && python3 -m pytest tests/test_benchmark_unfix.py -v`
Expected on the Mac (no torch): `1 skipped` with reason `could not import 'torch'` (the module-level `importorskip` skips the whole file as one item).
Expected inside the container (`ff3d_docker python -m pytest tests/test_benchmark_unfix.py -v`, after Task 8 sets up carrot): `FAILED ... FileNotFoundError` for `benchmark/unfix_spconv_checkpoint.py`.

- [ ] **Step 3: Write `benchmark/unfix_spconv_checkpoint.py`**

```python
#!/usr/bin/env python3
"""Inverse of tools/fix_spconv_checkpoint.py.

fix_spconv_checkpoint.py turns the raw spconv 2.x layout (out, kD, kH, kW, in) of every
5-D ``unet*``/``input_conv*`` weight into (kD, kH, kW, in, out) with permute(1, 2, 3, 4, 0).
This script applies permute(4, 0, 1, 2, 3) to get the raw layout back. The old
tools/test.py (main @ 6a75c37) permutes in memory, so it must be fed the raw layout.

Layout rule (mirrors the shape rule in tools/fix_spconv_checkpoint.py): kernel sizes in
this model are 1, 2 or 3 while channel counts are >= 6 (input_conv) or >= 32 (unet), so a
5-D weight whose dims 1..3 are all <= 3 is raw, one whose dims 0..2 are all <= 3 is
converted.

Exit codes: 0 written, 2 input is already in the raw layout (nothing written), 1 error.
"""
import argparse
import sys

import torch


def is_spconv_weight(name, tensor):
    return (name.startswith('unet') or name.startswith('input_conv')) \
        and name.endswith('weight') and tensor.dim() == 5


def layout_of(tensor):
    s = tuple(tensor.shape)
    if max(s[1:4]) <= 3 and s[0] > 3:
        return 'raw'
    if max(s[0:3]) <= 3 and s[4] > 3:
        return 'converted'
    raise ValueError(f'cannot classify weight shape {s}')


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--in-path', required=True)
    parser.add_argument('--out-path', required=True)
    args = parser.parse_args(argv)

    checkpoint = torch.load(args.in_path, map_location='cpu')
    state_dict = checkpoint['state_dict']
    spconv = [k for k, v in state_dict.items() if is_spconv_weight(k, v)]
    if not spconv:
        print('no spconv weights found', file=sys.stderr)
        return 1
    layouts = {layout_of(state_dict[k]) for k in spconv}
    if layouts == {'raw'}:
        print('already raw: input is in the (out, k, k, k, in) layout; nothing written')
        return 2
    if layouts != {'converted'}:
        print(f'mixed layouts {layouts}; refusing', file=sys.stderr)
        return 1
    for k in spconv:
        state_dict[k] = state_dict[k].permute(4, 0, 1, 2, 3).contiguous()
    torch.save(checkpoint, args.out_path)
    print(f'wrote raw-layout checkpoint {args.out_path} ({len(spconv)} weights permuted)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
```

- [ ] **Step 4: Write `benchmark/common.sh`**

```bash
#!/usr/bin/env bash
# Shared settings and helpers for the Phase 2 benchmark scripts.
# Source it:  source "$(dirname "$0")/common.sh"
# Every path variable can be overridden from the environment (tests do that).
set -euo pipefail

FF3D_ROOT="${FF3D_ROOT:-/raid/cwinkelmann/ForestFormer3D}"
FF3D_IMAGE="${FF3D_IMAGE:-forestformer3d:cu118}"
FF3D_DOCKER="${FF3D_DOCKER:-docker}"          # set to "sudo docker" if needed
FF3D_SHM="${FF3D_SHM:-64g}"
FF3D_GPU="${FF3D_GPU:-0}"                     # CUDA_VISIBLE_DEVICES inside the container
FF3D_OLD_COMMIT="${FF3D_OLD_COMMIT:-6a75c37}"
FF3D_OLD="${FF3D_ROOT}/work_dirs/old-main"
FF3D_CONFIG="configs/oneformer3d_qs_radius16_qp300_2many.py"
FF3D_DATA="${FF3D_ROOT}/data/ForAINetV2"
FF3D_CKPT_DIR="${FF3D_ROOT}/work_dirs/clean_forestformer"
FF3D_RELEASE_CKPT="${FF3D_CKPT_DIR}/epoch_3000_fix.pth"
FF3D_LOGS="${FF3D_ROOT}/work_dirs/logs"
FF3D_BENCH_DIR="${FF3D_ROOT}/benchmark"

ff3d_log() { echo "$(date '+%Y-%m-%dT%H:%M:%S') $*"; }   # portable (GNU and BSD date)
ff3d_die() { ff3d_log "ERROR: $*" >&2; exit 1; }

# Run <cmd...> inside the image with the main checkout mounted at /workspace.
ff3d_docker() {
  $FF3D_DOCKER run --rm --gpus all --shm-size="$FF3D_SHM" \
    -e PYTHONPATH=/workspace -e CUDA_VISIBLE_DEVICES="$FF3D_GPU" -w /workspace \
    -v "$FF3D_ROOT":/workspace \
    "$FF3D_IMAGE" "$@"
}

# Run <cmd...> inside the SAME image with the old worktree (main @ 6a75c37) at /workspace.
# Three bind mounts on top of it:
#   main data/      -> /workspace/data       (preprocessed once by the main checkout)
#   main work_dirs/ -> /workspace/work_dirs  (checkpoints in, results out; the worktree
#                                             itself lives inside it, that nesting is harmless)
#   benchmark/old_prelude.sh -> /old_prelude.sh (copies the old mmengine patches)
# The image entrypoint copies /workspace/replace_mmdetection_files/transforms_3d.py into
# mmdet3d; because /workspace IS the old worktree here, the old code's own transforms_3d.py
# is applied automatically. loops.py and base_model.py (removed from the fixed code in
# Phase 1, still required by the old train_step/test_step signature) are copied by the prelude.
ff3d_docker_old() {
  [ -f "$FF3D_OLD/tools/test.py" ] || ff3d_die "old worktree missing: run benchmark/setup_old_worktree.sh"
  $FF3D_DOCKER run --rm --gpus all --shm-size="$FF3D_SHM" \
    -e PYTHONPATH=/workspace -e CUDA_VISIBLE_DEVICES="$FF3D_GPU" -w /workspace \
    -v "$FF3D_OLD":/workspace \
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
```

- [ ] **Step 5: Write `benchmark/old_prelude.sh`**

```bash
#!/usr/bin/env bash
# Runs INSIDE the container for old-code runs (main @ 6a75c37), after the image entrypoint.
# The old train_step/val_step/test_step take an `epoch` kwarg and need the patched
# mmengine loops.py and base_model.py from the old worktree (README step 4 of that
# commit). transforms_3d.py was already copied by the image entrypoint from
# /workspace/replace_mmdetection_files, which is the old worktree's copy here.
set -euo pipefail
MMENGINE_DIR="$(python -c 'import mmengine, os; print(os.path.dirname(mmengine.__file__))')"
cp /workspace/replace_mmdetection_files/loops.py      "$MMENGINE_DIR/runner/loops.py"
cp /workspace/replace_mmdetection_files/base_model.py "$MMENGINE_DIR/model/base_model/base_model.py"
echo "old_prelude: patched $MMENGINE_DIR/{runner/loops.py,model/base_model/base_model.py}"
exec "$@"
```

- [ ] **Step 6: Append ignore rules to `.gitignore`**

Run:
```bash
cd /Users/christian/ForestFormer3D
for p in 'work_dirs/' 'downloads/' 'data/ForAINetV2/train_val_data/' 'data/ForAINetV2/test_data/' 'data/ForAINetV2/forainetv2_instance_data/' 'data/ForAINetV2/*.pkl'; do
  grep -qxF "$p" .gitignore || echo "$p" >> .gitignore
done
cat .gitignore
```
Expected: the file ends with the six new lines (each present once).

- [ ] **Step 7: Syntax-check the shell files and re-run the test**

Run: `cd /Users/christian/ForestFormer3D && bash -n benchmark/common.sh && bash -n benchmark/old_prelude.sh && chmod +x benchmark/old_prelude.sh benchmark/unfix_spconv_checkpoint.py && FF3D_ROOT=/tmp/x bash -c 'source benchmark/common.sh; echo "$FF3D_OLD $FF3D_CONFIG"' && python3 -m pytest tests/test_benchmark_unfix.py -v`
Expected: `/tmp/x/work_dirs/old-main configs/oneformer3d_qs_radius16_qp300_2many.py` then `1 skipped` on the Mac (`2 passed` inside the container later).

- [ ] **Step 8: Commit**

```bash
cd /Users/christian/ForestFormer3D
git add benchmark/common.sh benchmark/old_prelude.sh benchmark/unfix_spconv_checkpoint.py tests/test_benchmark_unfix.py .gitignore
git commit -m "bench: shared docker wrappers, old-code prelude, checkpoint un-fix

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 2: `benchmark/fetch_zenodo.sh`

**Files:**
- Create: `benchmark/fetch_zenodo.sh`
- Test: `tests/test_benchmark_fetch.py`

**Interfaces:**
- Consumes: `FF3D_ROOT` from `benchmark/common.sh`; env overrides `ZENODO_API` (default `https://zenodo.org/api/records/16742708`) and `FF3D_DOWNLOADS` (default `$FF3D_ROOT/downloads`).
- Produces: `$FF3D_ROOT/data/ForAINetV2/train_val_data/*.ply`, `.../test_data/*.ply`, `$FF3D_ROOT/work_dirs/clean_forestformer/epoch_3000_fix.pth` (any `.pth` from the record), `$FF3D_DOWNLOADS/<key>` archives and `.unpacked-<md5>` markers. `--list` prints `key<TAB>size<TAB>md5<TAB>url` per file and exits.

- [ ] **Step 1: Write the failing end-to-end test with a local HTTP server**

```python
# tests/test_benchmark_fetch.py
"""Runs benchmark/fetch_zenodo.sh against a local HTTP server that mimics the Zenodo
REST record shape (files[].key/size/checksum/links.self). No network, no torch."""
import hashlib
import json
import os
import subprocess
import threading
import zipfile
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "benchmark" / "fetch_zenodo.sh"


def _md5(p: Path) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest()


@pytest.fixture
def zenodo(tmp_path):
    srv_dir = tmp_path / "srv"
    srv_dir.mkdir()
    with zipfile.ZipFile(srv_dir / "train_val_data.zip", "w") as z:
        z.writestr("train_val_data/CULS_plot_2_annotated.ply", b"ply-train")
        z.writestr("train_val_data/CULS_plot_3_annotated.ply", b"ply-val")
    with zipfile.ZipFile(srv_dir / "test_data.zip", "w") as z:
        z.writestr("test_data/NIBIO_NIBIO_plot_1_annotated_test.ply", b"ply-test")
    with zipfile.ZipFile(srv_dir / "clean_forestformer.zip", "w") as z:
        z.writestr("clean_forestformer/epoch_3000_fix.pth", b"not-a-real-checkpoint")

    handler = partial(SimpleHTTPRequestHandler, directory=str(srv_dir))
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    port = httpd.server_address[1]
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()

    files = []
    for name in ("train_val_data.zip", "test_data.zip", "clean_forestformer.zip"):
        p = srv_dir / name
        files.append({"key": name, "size": p.stat().st_size, "checksum": "md5:" + _md5(p),
                      "links": {"self": f"http://127.0.0.1:{port}/{name}"}})
    (srv_dir / "record.json").write_text(json.dumps({"files": files}))
    try:
        yield f"http://127.0.0.1:{port}/record.json"
    finally:
        httpd.shutdown()


def _run(args, root, api):
    env = dict(os.environ, FF3D_ROOT=str(root), ZENODO_API=api)
    return subprocess.run(["bash", str(SCRIPT), *args], env=env, capture_output=True, text=True)


def test_list_prints_one_line_per_file(tmp_path, zenodo):
    r = _run(["--list"], tmp_path / "root", zenodo)
    assert r.returncode == 0, r.stderr
    lines = [l.split("\t") for l in r.stdout.strip().splitlines()]
    assert [l[0] for l in lines] == ["train_val_data.zip", "test_data.zip", "clean_forestformer.zip"]
    assert all(len(l) == 4 and len(l[2]) == 32 for l in lines)


def test_fetch_places_files_and_is_idempotent(tmp_path, zenodo):
    root = tmp_path / "root"
    r = _run([], root, zenodo)
    assert r.returncode == 0, r.stdout + r.stderr
    assert (root / "data/ForAINetV2/train_val_data/CULS_plot_2_annotated.ply").read_bytes() == b"ply-train"
    assert (root / "data/ForAINetV2/train_val_data/CULS_plot_3_annotated.ply").exists()
    assert (root / "data/ForAINetV2/test_data/NIBIO_NIBIO_plot_1_annotated_test.ply").read_bytes() == b"ply-test"
    assert (root / "work_dirs/clean_forestformer/epoch_3000_fix.pth").read_bytes() == b"not-a-real-checkpoint"
    assert "train_val_data: 2 ply" in r.stdout
    assert "test_data: 1 ply" in r.stdout

    r2 = _run([], root, zenodo)
    assert r2.returncode == 0, r2.stdout + r2.stderr
    assert r2.stdout.count("already present") == 3
    assert r2.stdout.count("already unpacked") == 3


def test_md5_mismatch_fails(tmp_path, zenodo):
    root = tmp_path / "root"
    dl = root / "downloads"
    dl.mkdir(parents=True)
    (dl / "test_data.zip").write_bytes(b"corrupt")
    # a corrupt partial file: curl resumes, md5 still wrong -> non-zero exit, clear message
    r = _run([], root, zenodo)
    assert r.returncode != 0
    assert "md5 mismatch" in r.stdout + r.stderr
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cd /Users/christian/ForestFormer3D && python3 -m pytest tests/test_benchmark_fetch.py -v`
Expected: 3 FAILED (`bash: .../benchmark/fetch_zenodo.sh: No such file or directory`, returncode 127).

- [ ] **Step 3: Write `benchmark/fetch_zenodo.sh`**

```bash
#!/usr/bin/env bash
# Idempotent download of Zenodo record 16742708 (ForAINetV2 dataset + epoch_3000_fix.pth).
#
#   bash benchmark/fetch_zenodo.sh          # download, verify md5, unpack, place
#   bash benchmark/fetch_zenodo.sh --list   # only print key<TAB>size<TAB>md5<TAB>url
#
# Layout produced under $FF3D_ROOT:
#   data/ForAINetV2/train_val_data/*.ply   (PLY files from zips whose name does not contain "test")
#   data/ForAINetV2/test_data/*.ply        (PLY files from zips whose name contains "test")
#   work_dirs/clean_forestformer/*.pth     (any .pth from any zip or a bare .pth file)
# Archives stay in $FF3D_DOWNLOADS (default $FF3D_ROOT/downloads) with a
# .unpacked-<md5> marker so a second run is a no-op.
# File names are discovered from the REST API, never hard-coded.
set -euo pipefail
source "$(dirname "$0")/common.sh"

ZENODO_API="${ZENODO_API:-https://zenodo.org/api/records/16742708}"
FF3D_DOWNLOADS="${FF3D_DOWNLOADS:-$FF3D_ROOT/downloads}"

list_files() {
  # one line per file: key \t size \t md5 \t url
  curl -sSL --fail "$ZENODO_API" | python3 -c '
import json, sys
rec = json.load(sys.stdin)
for f in rec["files"]:
    md5 = f["checksum"].split(":", 1)[-1]
    print("\t".join([f["key"], str(f["size"]), md5, f["links"]["self"]]))
'
}

md5_of() {
  if command -v md5sum >/dev/null 2>&1; then md5sum "$1" | cut -d' ' -f1
  else md5 -q "$1"; fi
}

download() {  # download <url> <dest> <md5>
  local url="$1" dest="$2" md5="$3" rc
  if [ -f "$dest" ] && [ "$(md5_of "$dest")" = "$md5" ]; then
    echo "already present: $dest"
    return 0
  fi
  echo "downloading $url -> $dest"
  set +e
  curl -L --fail --retry 5 --retry-delay 15 -C - -o "$dest" "$url" < /dev/null
  rc=$?
  set -e
  # 33 = server does not honour the byte range, 22 = HTTP 416 (range past the end because the
  # file on disk is already complete). Both leave the existing file untouched; md5 decides below.
  if [ "$rc" -ne 0 ] && [ "$rc" -ne 22 ] && [ "$rc" -ne 33 ]; then
    ff3d_die "curl failed with exit $rc for $url"
  fi
  [ -f "$dest" ] || ff3d_die "curl exit $rc and no file at $dest for $url"
  local got
  got="$(md5_of "$dest")"
  if [ "$got" != "$md5" ]; then
    ff3d_die "md5 mismatch for $dest: expected $md5 got $got (delete the file and re-run)"
  fi
  echo "verified md5 $md5: $dest"
}

place_unpacked() {  # place_unpacked <unpack_dir> <key>
  local dir="$1" key="$2" target
  if [[ "$key" == *test* ]]; then target="$FF3D_DATA/test_data"; else target="$FF3D_DATA/train_val_data"; fi
  mkdir -p "$target" "$FF3D_CKPT_DIR"
  find "$dir" -type f -name '*.ply' -not -path '*/__MACOSX/*' -exec mv -f {} "$target"/ \;
  find "$dir" -type f -name '*.pth' -not -path '*/__MACOSX/*' -exec mv -f {} "$FF3D_CKPT_DIR"/ \;
}

unpack() {  # unpack <archive> <key> <md5>
  local archive="$1" key="$2" md5="$3"
  local marker="$archive.unpacked-$md5"
  if [ -f "$marker" ]; then
    echo "already unpacked: $key"
    return 0
  fi
  case "$key" in
    *.zip)
      local tmp="$FF3D_DOWNLOADS/unpack-${key%.zip}"
      rm -rf "$tmp"; mkdir -p "$tmp"
      unzip -q "$archive" -d "$tmp"
      place_unpacked "$tmp" "$key"
      rm -rf "$tmp"
      ;;
    *.pth)
      mkdir -p "$FF3D_CKPT_DIR"
      cp -f "$archive" "$FF3D_CKPT_DIR/$key"
      ;;
    *)
      echo "leaving $key in $FF3D_DOWNLOADS (not a zip or pth)"
      ;;
  esac
  touch "$marker"
  echo "unpacked: $key"
}

main() {
  if [ "${1:-}" = "--list" ]; then
    list_files
    return 0
  fi
  mkdir -p "$FF3D_DOWNLOADS" "$FF3D_DATA"
  local listing
  listing="$(list_files)"
  [ -n "$listing" ] || ff3d_die "record lists no files"
  while IFS=$'\t' read -r key size md5 url; do
    echo "== $key ($size bytes)"
    download "$url" "$FF3D_DOWNLOADS/$key" "$md5"
    unpack "$FF3D_DOWNLOADS/$key" "$key" "$md5"
  done <<< "$listing"
  echo "train_val_data: $(find "$FF3D_DATA/train_val_data" -name '*.ply' 2>/dev/null | wc -l | tr -d ' ') ply"
  echo "test_data: $(find "$FF3D_DATA/test_data" -name '*.ply' 2>/dev/null | wc -l | tr -d ' ') ply"
  echo "checkpoints: $(ls "$FF3D_CKPT_DIR"/*.pth 2>/dev/null | tr '\n' ' ')"
  if [ -f "$FF3D_DATA/meta_data/test_list.txt" ]; then
    local want have
    want="$(grep -c . "$FF3D_DATA/meta_data/test_list.txt")"
    have="$(find "$FF3D_DATA/test_data" -name '*.ply' | wc -l | tr -d ' ')"
    [ "$want" = "$have" ] || echo "WARNING: test_list.txt has $want scans, test_data has $have ply files"
  fi
}

main "$@"
```

- [ ] **Step 4: Syntax-check and run the tests**

Run: `cd /Users/christian/ForestFormer3D && chmod +x benchmark/fetch_zenodo.sh && bash -n benchmark/fetch_zenodo.sh && python3 -m pytest tests/test_benchmark_fetch.py -v`
Expected: `3 passed`.

- [ ] **Step 5: Run `--list` against the real record (needs network)**

Run: `cd /Users/christian/ForestFormer3D && FF3D_ROOT=/tmp/ff3d-list bash benchmark/fetch_zenodo.sh --list`
Expected (exact values as of 2026-09-22):
```
train_val_data.zip	2426597147	5a63cc1cbe88edd9ebec28ad7e46f79b	https://zenodo.org/api/records/16742708/files/train_val_data.zip/content
test_data.zip	377105598	1c00a0f0b89f03b74064432162619136	https://zenodo.org/api/records/16742708/files/test_data.zip/content
clean_forestformer.zip	197823601	553d67379331966509076f3fbb409e57	https://zenodo.org/api/records/16742708/files/clean_forestformer.zip/content
```
If the network is unavailable the command fails with `curl: (6) Could not resolve host`; that is acceptable here, the tests in Step 4 cover the parsing.

- [ ] **Step 6: Commit**

```bash
cd /Users/christian/ForestFormer3D
git add benchmark/fetch_zenodo.sh tests/test_benchmark_fetch.py
git commit -m "bench: idempotent Zenodo fetch for dataset and checkpoint

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 3: `benchmark/setup_old_worktree.sh` and `benchmark/wait_idle.sh`

**Files:**
- Create: `benchmark/setup_old_worktree.sh`
- Create: `benchmark/wait_idle.sh`

**Interfaces:**
- Consumes: `FF3D_ROOT`, `FF3D_OLD`, `FF3D_OLD_COMMIT` from `common.sh`.
- Produces: `$FF3D_OLD` = detached worktree at 6a75c37 with the old `tools/test.py`, `replace_mmdetection_files/{loops,base_model,transforms_3d}.py`; `benchmark/wait_idle.sh -- <cmd...>` execs `<cmd>` once `nvidia-smi --query-compute-apps=pid` is empty, 1-minute load is below `LOAD_MAX` and no `forestformer3d` container runs, for `NEEDED` consecutive samples `INTERVAL` seconds apart.

- [ ] **Step 1: Write `benchmark/setup_old_worktree.sh`**

```bash
#!/usr/bin/env bash
# Create the old-code worktree (main @ 6a75c37) at work_dirs/old-main. Idempotent.
#
# The old README asked for three manual copies into site-packages
# (replace_mmdetection_files/{loops.py,base_model.py,transforms_3d.py}). None is done here:
#   - transforms_3d.py is copied by the image entrypoint from /workspace/replace_mmdetection_files,
#     and /workspace is this worktree when ff3d_docker_old runs, so the OLD file is applied;
#   - loops.py and base_model.py are copied by benchmark/old_prelude.sh (mounted into the
#     old container) because the fixed image entrypoint no longer copies them.
# The old README also asked to run fix_spconv_checkpoint.py before test.py; the benchmark
# instead feeds the old test.py the RAW layout (it permutes in memory), see run_release_eval.sh.
set -euo pipefail
source "$(dirname "$0")/common.sh"

cd "$FF3D_ROOT"
git rev-parse --verify --quiet "${FF3D_OLD_COMMIT}^{commit}" >/dev/null \
  || git fetch origin main
git rev-parse --verify --quiet "${FF3D_OLD_COMMIT}^{commit}" >/dev/null \
  || ff3d_die "commit $FF3D_OLD_COMMIT not found even after fetching origin/main"

if [ -f "$FF3D_OLD/.git" ]; then
  ff3d_log "worktree exists at $FF3D_OLD ($(git -C "$FF3D_OLD" rev-parse --short HEAD))"
else
  mkdir -p "$(dirname "$FF3D_OLD")"
  git worktree add --detach "$FF3D_OLD" "$FF3D_OLD_COMMIT"
fi

head="$(git -C "$FF3D_OLD" rev-parse --short HEAD)"
[ "$head" = "$(git rev-parse --short "$FF3D_OLD_COMMIT")" ] || ff3d_die "worktree HEAD $head != $FF3D_OLD_COMMIT"

# Sanity checks that this really is the old code path the benchmark relies on.
grep -q 'permute(1, 2, 3, 4, 0)' "$FF3D_OLD/tools/test.py" \
  || ff3d_die "old tools/test.py does not permute weights in memory; wrong commit?"
grep -q "test_cfg\['output_dir'\] = cfg.work_dir" "$FF3D_OLD/tools/test.py" \
  || ff3d_die "old tools/test.py does not route output to work_dir; wrong commit?"
for f in loops.py base_model.py transforms_3d.py; do
  [ -f "$FF3D_OLD/replace_mmdetection_files/$f" ] || ff3d_die "old worktree lacks replace_mmdetection_files/$f"
done
ff3d_log "old worktree ready: $FF3D_OLD @ $head"
```

- [ ] **Step 2: Write `benchmark/wait_idle.sh`**

```bash
#!/usr/bin/env bash
# Run a command once the GPU host is genuinely idle.
#
#   bash benchmark/wait_idle.sh -- bash benchmark/run_train_200.sh fixed
#
# "Idle" = all three, for NEEDED consecutive samples INTERVAL seconds apart:
#   no GPU compute processes (nvidia-smi), 1-minute load below LOAD_MAX,
#   no running container from the forestformer3d image.
# One empty sample is not idle: a neighbour between job stages looks empty for a moment.
# The load and container checks matter because preprocessing and final_eval are CPU-only
# phases of our own pipeline during which nvidia-smi already reports zero processes.
set -u
INTERVAL=${INTERVAL:-120}
NEEDED=${NEEDED:-3}
LOAD_MAX=${LOAD_MAX:-8.0}
MAX_WAIT=${MAX_WAIT:-$((48*3600))}
FF3D_DOCKER=${FF3D_DOCKER:-docker}
FF3D_IMAGE_PREFIX=${FF3D_IMAGE_PREFIX:-forestformer3d}

[ "${1:-}" = "--" ] && shift
[ $# -gt 0 ] || { echo "usage: wait_idle.sh -- <command...>" >&2; exit 2; }

streak=0; waited=0
now() { date '+%Y-%m-%dT%H:%M:%S'; }   # portable (GNU and BSD date)
echo "watcher started $(now): need ${NEEDED}x${INTERVAL}s idle (gpu+load+containers)"
while [ "$waited" -lt "$MAX_WAIT" ]; do
  gpu=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c . || true)
  load=$(cut -d' ' -f1 /proc/loadavg 2>/dev/null || echo 0)
  cont=$($FF3D_DOCKER ps --format '{{.Image}}' 2>/dev/null | grep -c "^${FF3D_IMAGE_PREFIX}" || true)
  ok=$(awk -v l="$load" -v m="$LOAD_MAX" 'BEGIN{print (l<m)?1:0}')
  if [ "$gpu" -eq 0 ] && [ "$ok" -eq 1 ] && [ "$cont" -eq 0 ]; then
    streak=$((streak+1))
    echo "$(date +%H:%M) idle ${streak}/${NEEDED} (load $load)"
    if [ "$streak" -ge "$NEEDED" ]; then
      echo "$(now) IDLE -- launching: $*"
      exec "$@"
    fi
  else
    [ "$streak" -gt 0 ] && echo "$(date +%H:%M) busy again (gpu=$gpu load=$load containers=$cont) -- reset"
    streak=0
  fi
  sleep "$INTERVAL"; waited=$((waited+INTERVAL))
done
echo "$(now) gave up after ${MAX_WAIT}s -- never idle"
exit 1
```

- [ ] **Step 3: Syntax-check and dry-run the idle guard on the Mac**

Run: `cd /Users/christian/ForestFormer3D && chmod +x benchmark/setup_old_worktree.sh benchmark/wait_idle.sh && bash -n benchmark/setup_old_worktree.sh && bash -n benchmark/wait_idle.sh && INTERVAL=1 NEEDED=2 FF3D_DOCKER=/usr/bin/false bash benchmark/wait_idle.sh -- echo LAUNCHED`
Expected (the Mac has no nvidia-smi, so gpu=0; `/proc/loadavg` is absent, so load=0):
```
watcher started 2026-09-22T12:40:00: need 2x1s idle (gpu+load+containers)
12:40 idle 1/2 (load 0)
12:40 idle 2/2 (load 0)
2026-09-22T12:40:02 IDLE -- launching: echo LAUNCHED
LAUNCHED
```

- [ ] **Step 4: Dry-run the worktree script on a throwaway clone**

Run:
```bash
cd /private/tmp/claude-501/-Users-christian-ForestFormer3D/26b04714-f74e-43e6-b235-299a244f9313/scratchpad
rm -rf wt-test && git clone -q /Users/christian/ForestFormer3D wt-test
FF3D_ROOT="$PWD/wt-test" bash /Users/christian/ForestFormer3D/benchmark/setup_old_worktree.sh
FF3D_ROOT="$PWD/wt-test" bash /Users/christian/ForestFormer3D/benchmark/setup_old_worktree.sh
git -C wt-test worktree list
```
Expected: first run ends with `old worktree ready: .../wt-test/work_dirs/old-main @ 6a75c37`; second run prints `worktree exists at ... (6a75c37)` then the same ready line; `worktree list` shows two entries, the second `.../work_dirs/old-main  6a75c37 (detached HEAD)`. (The clone has `main` locally because `git clone` copies branches; on carrot the clone of `fix/review-findings` contains 6a75c37 as an ancestor.)

- [ ] **Step 5: Commit**

```bash
cd /Users/christian/ForestFormer3D
git add benchmark/setup_old_worktree.sh benchmark/wait_idle.sh
git commit -m "bench: old-code worktree setup and GPU idle guard

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 4: `benchmark/run_release_eval.sh`

**Files:**
- Create: `benchmark/run_release_eval.sh`

**Interfaces:**
- Consumes: `ff3d_docker`, `ff3d_docker_old`, `ff3d_preprocess`, `ff3d_prepare_checkpoint`, `ff3d_daemonize` from `common.sh`; fixed `tools/test.py <config> <ckpt> --work-dir <dir>`; old `tools/test.py` (same CLI, writes `<work_dir>/<scan>.ply`); fixed `tools/final_eval.py <dir>` appending `<dir>/evaluation_total_test.txt`.
- Produces: `work_dirs/bench-release-fixed/{<scan>.ply, evaluation_total_test.txt, .done-test, .done-eval}`, `work_dirs/bench-release-old/{...}`, `work_dirs/clean_forestformer/epoch_3000_{converted,raw}.pth`, `.layout`, log in `work_dirs/logs/release-eval-<ts>.log`. Task 7's `collect.py` reads the two `evaluation_total_test.txt`.

- [ ] **Step 1: Write `benchmark/run_release_eval.sh`**

```bash
#!/usr/bin/env bash
# Released checkpoint epoch_3000: old vs fixed inference on the ForAINetV2 test split.
#
#   bash benchmark/run_release_eval.sh            # daemonizes itself; prints the log path
#   FF3D_FOREGROUND=1 bash benchmark/run_release_eval.sh   # run in the foreground
#
# Stages (each guarded by a marker so the script can be re-run after an interruption):
#   1. preprocess all splits once (main checkout)
#   2. prepare epoch_3000_{converted,raw}.pth via fix_spconv_checkpoint.py exit code
#   3. fixed: tools/test.py on the CONVERTED file  -> work_dirs/bench-release-fixed
#   4. old:   tools/test.py on the RAW file        -> work_dirs/bench-release-old
#      (old test.py permutes in memory; the old predict() writes <work_dir>/<scan>.ply)
#   5. fixed tools/final_eval.py on both directories
set -euo pipefail
source "$(dirname "$0")/common.sh"
ff3d_daemonize release-eval "$0" "$@"

FIXED_DIR="work_dirs/bench-release-fixed"
OLD_DIR="work_dirs/bench-release-old"
N_TEST="$(grep -c . "$FF3D_DATA/meta_data/test_list.txt")"

run_test_stage() {  # run_test_stage <old|fixed> <ckpt (root-relative)> <out_dir (root-relative)>
  local variant="$1" ckpt="$2" out="$3" runner n
  if [ -f "$FF3D_ROOT/$out/.done-test" ]; then
    ff3d_log "$variant test.py already done ($out)"
    return 0
  fi
  mkdir -p "$FF3D_ROOT/$out"
  rm -f "$FF3D_ROOT/$out"/*.ply
  if [ "$variant" = "old" ]; then runner=ff3d_docker_old; else runner=ff3d_docker; fi
  ff3d_log "$variant test.py: $ckpt -> $out"
  "$runner" python tools/test.py "$FF3D_CONFIG" "$ckpt" --work-dir "$out"
  n="$(find "$FF3D_ROOT/$out" -maxdepth 1 -name '*.ply' | wc -l | tr -d ' ')"
  [ "$n" = "$N_TEST" ] || ff3d_die "$variant produced $n ply files, expected $N_TEST in $out"
  touch "$FF3D_ROOT/$out/.done-test"
  ff3d_log "$variant test.py done: $n ply files"
}

run_eval_stage() {  # run_eval_stage <out_dir (root-relative)>
  local out="$1"
  if [ -f "$FF3D_ROOT/$out/.done-eval" ]; then
    ff3d_log "final_eval already done ($out)"
    return 0
  fi
  rm -f "$FF3D_ROOT/$out/evaluation_total_test.txt"
  ff3d_log "final_eval.py $out"
  ff3d_docker python tools/final_eval.py "$out"
  grep -q '^Instance Segmentation F1 score:' "$FF3D_ROOT/$out/evaluation_total_test.txt" \
    || ff3d_die "no F1 line in $out/evaluation_total_test.txt"
  touch "$FF3D_ROOT/$out/.done-eval"
  grep '^Instance Segmentation F1 score:' "$FF3D_ROOT/$out/evaluation_total_test.txt" | tail -1
}

ff3d_log "release eval start (root $FF3D_ROOT, image $FF3D_IMAGE)"
[ -f "$FF3D_RELEASE_CKPT" ] || ff3d_die "missing $FF3D_RELEASE_CKPT (run benchmark/fetch_zenodo.sh)"
[ -f "$FF3D_OLD/tools/test.py" ] || ff3d_die "missing old worktree (run benchmark/setup_old_worktree.sh)"

ff3d_preprocess

layout="$(ff3d_prepare_checkpoint work_dirs/clean_forestformer/epoch_3000_fix.pth work_dirs/clean_forestformer/epoch_3000)"
ff3d_log "Zenodo checkpoint layout: $layout"

run_test_stage fixed work_dirs/clean_forestformer/epoch_3000_converted.pth "$FIXED_DIR"
run_test_stage old   work_dirs/clean_forestformer/epoch_3000_raw.pth       "$OLD_DIR"

run_eval_stage "$FIXED_DIR"
run_eval_stage "$OLD_DIR"
ff3d_log "release eval finished"
```

- [ ] **Step 2: Syntax-check and exercise the daemonize + guard paths without Docker**

Run:
```bash
cd /Users/christian/ForestFormer3D && chmod +x benchmark/run_release_eval.sh && bash -n benchmark/run_release_eval.sh
R=/private/tmp/claude-501/-Users-christian-ForestFormer3D/26b04714-f74e-43e6-b235-299a244f9313/scratchpad/rel-test
rm -rf "$R" && mkdir -p "$R/data/ForAINetV2/meta_data" && printf 'a_test\nb_test\n' > "$R/data/ForAINetV2/meta_data/test_list.txt"
FF3D_ROOT="$R" FF3D_FOREGROUND=1 bash benchmark/run_release_eval.sh; echo "exit=$?"
FF3D_ROOT="$R" bash benchmark/run_release_eval.sh; sleep 1; cat "$R"/work_dirs/logs/release-eval-*.log
```
Expected: first command prints `... ERROR: missing $R/work_dirs/clean_forestformer/epoch_3000_fix.pth (run benchmark/fetch_zenodo.sh)` and `exit=1`. Second prints `started release-eval pid <N> -- follow with: tail -f $R/work_dirs/logs/release-eval-<ts>.log` and the log contains the same ERROR line (proves the nohup re-exec path works).

- [ ] **Step 3: Commit**

```bash
cd /Users/christian/ForestFormer3D
git add benchmark/run_release_eval.sh
git commit -m "bench: released checkpoint old-vs-fixed inference script

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 5: `benchmark/run_train_200.sh <old|fixed>`

**Files:**
- Create: `benchmark/run_train_200.sh`

**Interfaces:**
- Consumes: `common.sh` helpers; `tools/train.py <config> --work-dir <dir> --cfg-options ...` (same CLI in both code versions); `benchmark/wait_idle.sh`.
- Produces: `work_dirs/bench-<variant>-200/{epoch_200.pth, epoch_200_{converted,raw}.pth, epoch_200.layout, <ts>/<ts>.log, <ts>/vis_data/scalars.json, .done-train, test/{<scan>.ply, evaluation_total_test.txt, .done-test, .done-eval}}`; log `work_dirs/logs/train-<variant>-200-<ts>.log`.

Checkpoint layout rule for training runs (from Global Constraints): `train.py` saves spconv weights in the RAW `(out, k, k, k, in)` layout, so `ff3d_prepare_checkpoint` is expected to print `raw` (fix script exit 0). The script does not rely on that expectation: it always uses `epoch_200_converted.pth` for the fixed `test.py` and `epoch_200_raw.pth` for the old `test.py`, whichever way the exit code went, and logs a WARNING if the layout was not `raw`.

- [ ] **Step 1: Write `benchmark/run_train_200.sh`**

```bash
#!/usr/bin/env bash
# 200-epoch training run, old or fixed code, then test-split scoring of epoch_200.pth.
#
#   bash benchmark/wait_idle.sh -- bash benchmark/run_train_200.sh fixed
#   bash benchmark/wait_idle.sh -- bash benchmark/run_train_200.sh old
#
# The script daemonizes itself (nohup) unless FF3D_FOREGROUND=1; the log path is printed.
# Stages with markers: train (.done-train), prepare checkpoint, test (.done-test), eval (.done-eval).
# Training is resumable: a re-run after a crash passes --resume auto so mmengine continues
# from the last checkpoint in the work dir.
set -euo pipefail
source "$(dirname "$0")/common.sh"

VARIANT="${1:-}"
case "$VARIANT" in
  old|fixed) ;;
  *) echo "usage: $0 <old|fixed>" >&2; exit 2 ;;
esac
ff3d_daemonize "train-${VARIANT}-200" "$0" "$@"

WORK="work_dirs/bench-${VARIANT}-200"
TEST_OUT="$WORK/test"
EPOCHS="${FF3D_EPOCHS:-200}"
VAL_INTERVAL="${FF3D_VAL_INTERVAL:-20}"
CFG_OPTS=(train_cfg.max_epochs="$EPOCHS" train_cfg.val_interval="$VAL_INTERVAL" default_hooks.checkpoint.max_keep_ckpts=2)
N_TEST="$(grep -c . "$FF3D_DATA/meta_data/test_list.txt")"
if [ "$VARIANT" = "old" ]; then RUNNER=ff3d_docker_old; else RUNNER=ff3d_docker; fi

ff3d_log "train-${VARIANT}-200 start (root $FF3D_ROOT, image $FF3D_IMAGE, epochs $EPOCHS)"
[ "$VARIANT" = "fixed" ] || [ -f "$FF3D_OLD/tools/train.py" ] || ff3d_die "missing old worktree (run benchmark/setup_old_worktree.sh)"
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
```

- [ ] **Step 2: Syntax-check and exercise the argument and guard paths without Docker**

Run:
```bash
cd /Users/christian/ForestFormer3D && chmod +x benchmark/run_train_200.sh && bash -n benchmark/run_train_200.sh
bash benchmark/run_train_200.sh; echo "exit=$?"
bash benchmark/run_train_200.sh both; echo "exit=$?"
R=/private/tmp/claude-501/-Users-christian-ForestFormer3D/26b04714-f74e-43e6-b235-299a244f9313/scratchpad/train-test
rm -rf "$R" && mkdir -p "$R/data/ForAINetV2/meta_data" && printf 'a_test\n' > "$R/data/ForAINetV2/meta_data/test_list.txt"
FF3D_ROOT="$R" FF3D_FOREGROUND=1 bash benchmark/run_train_200.sh old; echo "exit=$?"
```
Expected: lines 2 and 3 print `usage: benchmark/run_train_200.sh <old|fixed>` with `exit=2`; the last prints `<timestamp> ERROR: missing old worktree (run benchmark/setup_old_worktree.sh)` and `exit=1`.

- [ ] **Step 3: Commit**

```bash
cd /Users/christian/ForestFormer3D
git add benchmark/run_train_200.sh
git commit -m "bench: 200-epoch old/fixed training run with test-split scoring

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 6: `benchmark/collect.py` with CPU tests

**Files:**
- Create: `benchmark/collect.py`
- Test: `tests/test_benchmark_collect.py`

**Interfaces:**
- Consumes: `work_dirs/bench-release-{fixed,old}/evaluation_total_test.txt`, `work_dirs/bench-{old,fixed}-200/test/evaluation_total_test.txt`, `work_dirs/bench-{old,fixed}-200/*/vis_data/scalars.json`, `work_dirs/bench-{old,fixed}-200/*/*.log`.
- Produces: `docs/benchmarks/<YYYY-MM-DD>-carrot-ff3d.md`. Python API (used by the test):
  - `parse_final_eval(path: Path) -> dict[str, float]` keys `F1, mPrecision, mRecall, mPQ, mSQ, mRQ, mMUCov, mMWCov, mIoU, mIoU_binary, oAcc` (last block wins)
  - `parse_scalars(paths: list[Path]) -> tuple[list[dict], list[dict]]` (train records, val records; val keys stripped of any `prefix/`)
  - `loss_per_epoch(train: list[dict]) -> dict[int, float]`
  - `val_curve(val: list[dict]) -> dict[int, dict[str, float]]`
  - `parse_log_wallclock(paths: list[Path]) -> dict` with `first_ts, last_ts, first_epoch, last_epoch, wall_s, sec_per_epoch`
  - `summarize_training(work_dir: Path) -> dict`
  - `render(date: str, release: dict, training: dict) -> str`
  - `main(argv: list[str] | None = None) -> int` with `--root`, `--date`, `--out`, `--allow-missing`

Exact `evaluation_total_test.txt` lines produced by the fixed `tools/final_eval.py` global block (`log_string(...)` with no file argument, lines 403-406, 430-433 and 497-527 of the current file):
```
Semantic Segmentation oAcc: 0.9
Semantic Segmentation mAcc: 0.88
Semantic Segmentation IoU: [0.0, 0.95, 0.8, 0.7]
Semantic Segmentation mIoU: 0.8166666666666667
Binary Semantic Segmentation mIoU: 0.9
Instance Segmentation for Offset:
Instance Segmentation MUCov: [0.7]
Instance Segmentation mMUCov: 0.7
Instance Segmentation MWCov: [0.72]
Instance Segmentation mMWCov: 0.72
Instance Segmentation Precision: [0.81]
Instance Segmentation mPrecision: 0.81
Instance Segmentation Recall: [0.79]
Instance Segmentation mRecall: 0.79
Instance Segmentation F1 score: 0.7998750000000001
Instance Segmentation RQ: [1 1]
Instance Segmentation meanRQ: 1.0
Instance Segmentation meanSQ: 0.85
Instance Segmentation meanPQ: 0.85
Instance Segmentation mean PQ star: 0.85
Instance Segmentation meanPQ (things): 0.8
```
Labels ending in `(things)` / `(stuff)` and list values are ignored; `meanPQ`/`meanSQ`/`meanRQ` (over stuff+things classes) map to `mPQ`/`mSQ`/`mRQ`.

- [ ] **Step 1: Write the failing test with synthetic fixtures**

```python
# tests/test_benchmark_collect.py
"""CPU-only tests for benchmark/collect.py against synthetic final_eval, scalars.json and
mmengine log fixtures. numpy only, no torch, no mmengine."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("collect", REPO / "benchmark" / "collect.py")
collect = importlib.util.module_from_spec(spec)
spec.loader.exec_module(collect)

FINAL_EVAL_BLOCK = """Semantic Segmentation oAcc: {oacc}
Semantic Segmentation mAcc: 0.88
Semantic Segmentation IoU: [0.0, 0.95, 0.8, 0.7]
Semantic Segmentation mIoU: {miou}
  
Binary Semantic Segmentation oAcc: 0.97
Binary Semantic Segmentation mAcc: 0.95
Binary Semantic Segmentation IoU: [0.0, 0.9, 0.9]
Binary Semantic Segmentation mIoU: 0.9
  
Instance Segmentation for Offset:
Instance Segmentation MUCov: [0.7]
Instance Segmentation mMUCov: 0.7
Instance Segmentation MWCov: [0.72]
Instance Segmentation mMWCov: 0.72
Instance Segmentation Precision: [0.81]
Instance Segmentation mPrecision: 0.81
Instance Segmentation Recall: [0.79]
Instance Segmentation mRecall: 0.79
Instance Segmentation F1 score: {f1}
Instance Segmentation RQ: [1 1]
Instance Segmentation meanRQ: 1.0
Instance Segmentation SQ: [0.8 0.9]
Instance Segmentation meanSQ: 0.85
Instance Segmentation PQ: [0.8 0.9]
Instance Segmentation meanPQ: 0.85
Instance Segmentation PQ star: [0.8 0.9]
Instance Segmentation mean PQ star: 0.85
Instance Segmentation RQ (things): [1]
Instance Segmentation meanRQ (things): 1.0
Instance Segmentation SQ (things): [0.9]
Instance Segmentation meanSQ (things): 0.9
Instance Segmentation PQ (things): [0.9]
Instance Segmentation meanPQ (things): 0.9
Instance Segmentation RQ (stuff): [1]
Instance Segmentation meanRQ (stuff): 1.0
Instance Segmentation SQ (stuff): [0.8]
Instance Segmentation meanSQ (stuff): 0.8
Instance Segmentation PQ (stuff): [0.8]
Instance Segmentation meanPQ (stuff): 0.8
"""


def write_final_eval(d: Path, f1: float, miou: float = 0.81, oacc: float = 0.9, stale_first: bool = False):
    d.mkdir(parents=True, exist_ok=True)
    text = ""
    if stale_first:  # final_eval appends; an earlier run's block must be ignored
        text += FINAL_EVAL_BLOCK.format(f1=0.1, miou=0.1, oacc=0.1)
    text += FINAL_EVAL_BLOCK.format(f1=f1, miou=miou, oacc=oacc)
    (d / "evaluation_total_test.txt").write_text(text)


def write_training(work: Path, epochs: int, loss0: float, f1s: dict, nan_at: int | None = None,
                   ts0="2026/09/23 00:00:00", sec_per_epoch=60):
    """23 iterations per epoch, one train record per epoch (mmengine logs at end of epoch
    when len(dataloader) <= logger interval), val record every 20 epochs with step=epoch."""
    import datetime as dt
    ts_dir = work / "20260923_000000"
    vis = ts_dir / "vis_data"
    vis.mkdir(parents=True)
    t0 = dt.datetime.strptime(ts0, "%Y/%m/%d %H:%M:%S")
    scalars, log = [], []
    for e in range(1, epochs + 1):
        loss = float("nan") if nan_at == e else loss0 / e
        scalars.append({"lr": 1e-4, "data_time": 0.05, "loss": loss, "time": 2.5, "epoch": e,
                        "memory": 20000, "step": 23 * e})
        t = t0 + dt.timedelta(seconds=sec_per_epoch * e)
        loss_s = "nan" if loss != loss else f"{loss:.4f}"
        log.append(f"{t:%Y/%m/%d %H:%M:%S} - mmengine - INFO - Epoch(train)  [{e}][23/23]  "
                   f"lr: 1.0000e-04  eta: 0:00:00  time: 2.5000  data_time: 0.0500  memory: 20000  loss: {loss_s}")
        if e in f1s:
            scalars.append({"mIoU": 0.8, "mIoU_binary": 0.9, "mMWCov": 0.6, "mMUCov": 0.6, "mPrecision": 0.7,
                            "mRecall": 0.7, "F1": f1s[e], "mPQ": 0.5, "mSQ": 0.7, "mRQ": 0.6,
                            "data_time": 0.1, "time": 3.0, "step": e})
            log.append(f"{t:%Y/%m/%d %H:%M:%S} - mmengine - INFO - Epoch(val) [{e}][15/15]    F1: {f1s[e]:.4f}")
    (vis / "scalars.json").write_text("\n".join(json.dumps(s) for s in scalars) + "\n")
    (ts_dir / "20260923_000000.log").write_text("\n".join(log) + "\n")


@pytest.fixture
def root(tmp_path):
    wd = tmp_path / "work_dirs"
    write_final_eval(wd / "bench-release-fixed", f1=0.82, stale_first=True)
    write_final_eval(wd / "bench-release-old", f1=0.80)
    write_training(wd / "bench-old-200", 200, 10.0, {20: 0.30, 40: 0.40, 200: 0.55})
    write_training(wd / "bench-fixed-200", 200, 9.0, {20: 0.35, 40: 0.45, 200: 0.60})
    write_final_eval(wd / "bench-old-200" / "test", f1=0.50)
    write_final_eval(wd / "bench-fixed-200" / "test", f1=0.58)
    return tmp_path


def test_parse_final_eval_last_block_wins(root):
    m = collect.parse_final_eval(root / "work_dirs/bench-release-fixed/evaluation_total_test.txt")
    assert m["F1"] == pytest.approx(0.82)
    assert m["mIoU"] == pytest.approx(0.81)
    assert m["oAcc"] == pytest.approx(0.9)
    assert m["mPQ"] == pytest.approx(0.85)       # 'meanPQ', not 'meanPQ (things)'
    assert m["mIoU_binary"] == pytest.approx(0.9)
    assert m["mPrecision"] == pytest.approx(0.81)
    assert "MUCov" not in m                      # list-valued lines are skipped


def test_parse_scalars_and_curves(root):
    paths = sorted((root / "work_dirs/bench-fixed-200").glob("*/vis_data/scalars.json"))
    train, val = collect.parse_scalars(paths)
    assert len(train) == 200 and len(val) == 3
    losses = collect.loss_per_epoch(train)
    assert losses[1] == pytest.approx(9.0) and losses[200] == pytest.approx(0.045)
    curve = collect.val_curve(val)
    assert curve[20]["F1"] == pytest.approx(0.35) and curve[200]["F1"] == pytest.approx(0.60)


def test_val_prefix_is_stripped(tmp_path):
    p = tmp_path / "scalars.json"
    p.write_text(json.dumps({"ForAINetV2/F1": 0.5, "ForAINetV2/mIoU": 0.7, "step": 20}) + "\n")
    _, val = collect.parse_scalars([p])
    assert val[0]["F1"] == 0.5 and val[0]["mIoU"] == 0.7 and val[0]["step"] == 20


def test_wallclock_from_log(root):
    logs = sorted((root / "work_dirs/bench-old-200").glob("*/*.log"))
    w = collect.parse_log_wallclock(logs)
    assert w["first_epoch"] == 1 and w["last_epoch"] == 200
    assert w["wall_s"] == pytest.approx(199 * 60)
    assert w["sec_per_epoch"] == pytest.approx(60.0)


def test_summarize_training_detects_nan(tmp_path):
    write_training(tmp_path / "w", 30, 5.0, {20: 0.2}, nan_at=25)
    s = collect.summarize_training(tmp_path / "w")
    assert s["nan_epochs"] == [25]
    assert s["epochs_done"] == 30
    assert s["best_val"] == (20, pytest.approx(0.2))


def test_main_writes_report_with_pass_fail(root, capsys):
    out = root / "docs/benchmarks/2026-09-25-carrot-ff3d.md"
    rc = collect.main(["--root", str(root), "--date", "2026-09-25", "--out", str(out)])
    assert rc == 0
    text = out.read_text()
    assert "# ForestFormer3D benchmark on carrot (2026-09-25)" in text
    assert "| fixed | 0.8200 |" in text and "| old | 0.8000 |" in text
    assert "| fixed | 200 | 60.0 |" in text
    assert "Released checkpoint: fixed F1 0.8200 >= old F1 0.8000: PASS" in text
    assert "Fixed 200-epoch run: 200/200 epochs, no NaN loss, 3 val points: PASS" in text
    assert "| 20 | 0.3000 | 0.3500 |" in text


def test_main_missing_dir_fails_unless_allowed(tmp_path):
    out = tmp_path / "r.md"
    rc = collect.main(["--root", str(tmp_path), "--date", "2026-09-25", "--out", str(out)])
    assert rc == 1
    rc = collect.main(["--root", str(tmp_path), "--date", "2026-09-25", "--out", str(out), "--allow-missing"])
    assert rc == 0
    assert "n/a" in out.read_text() and "NOT RUN" in out.read_text()
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cd /Users/christian/ForestFormer3D && python3 -m pytest tests/test_benchmark_collect.py -v`
Expected: collection error `FileNotFoundError` for `benchmark/collect.py` (all tests error).

- [ ] **Step 3: Write `benchmark/collect.py`**

```python
#!/usr/bin/env python3
"""Collect the Phase 2 benchmark results into docs/benchmarks/<date>-carrot-ff3d.md.

Inputs under --root (default $FF3D_ROOT or the current directory):
  work_dirs/bench-release-fixed/evaluation_total_test.txt   fixed tools/test.py on epoch_3000
  work_dirs/bench-release-old/evaluation_total_test.txt     old   tools/test.py on epoch_3000
  work_dirs/bench-{old,fixed}-200/*/vis_data/scalars.json   mmengine scalars (train + val)
  work_dirs/bench-{old,fixed}-200/*/*.log                   mmengine log (timestamps)
  work_dirs/bench-{old,fixed}-200/test/evaluation_total_test.txt  test score of epoch_200

Usage:  python benchmark/collect.py --root /workspace --date 2026-09-25
Pure Python + numpy; no torch or mmengine imports.
"""
import argparse
import datetime as dt
import json
import math
import os
import re
import sys
from pathlib import Path

import numpy as np

# evaluation_total_test.txt label -> report key (only scalar-valued lines of the global block)
FINAL_EVAL_LABELS = {
    'Semantic Segmentation oAcc': 'oAcc',
    'Semantic Segmentation mAcc': 'mAcc',
    'Semantic Segmentation mIoU': 'mIoU',
    'Binary Semantic Segmentation mIoU': 'mIoU_binary',
    'Instance Segmentation mMUCov': 'mMUCov',
    'Instance Segmentation mMWCov': 'mMWCov',
    'Instance Segmentation mPrecision': 'mPrecision',
    'Instance Segmentation mRecall': 'mRecall',
    'Instance Segmentation F1 score': 'F1',
    'Instance Segmentation meanRQ': 'mRQ',
    'Instance Segmentation meanSQ': 'mSQ',
    'Instance Segmentation meanPQ': 'mPQ',
    'Instance Segmentation mean PQ star': 'mPQ_star',
}
RELEASE_COLUMNS = ['F1', 'mPrecision', 'mRecall', 'mPQ', 'mIoU', 'mIoU_binary', 'mMUCov', 'mMWCov']
LOG_TRAIN_RE = re.compile(
    r'^(\d{4}/\d{2}/\d{2} \d{2}:\d{2}:\d{2}) - mmengine - INFO - Epoch\(train\)\s*\[(\d+)\]\[(\d+)/(\d+)\]')
TS_FMT = '%Y/%m/%d %H:%M:%S'


def parse_final_eval(path):
    """Scalar metrics of the LAST global block in evaluation_total_test.txt (the file is
    appended to on every final_eval.py run, so later lines override earlier ones)."""
    out = {}
    for line in Path(path).read_text().splitlines():
        if ': ' not in line:
            continue
        label, value = line.split(': ', 1)
        key = FINAL_EVAL_LABELS.get(label.strip())
        if key is None:
            continue
        try:
            out[key] = float(value.strip())
        except ValueError:
            continue
    return out


def parse_scalars(paths):
    """Return (train_records, val_records). A train record has 'loss' and 'epoch'; a val
    record has 'F1' (any 'prefix/' on val keys is stripped). Files are read in the given
    order so a resumed run's later timestamp directory comes last."""
    train, val = [], []
    for p in paths:
        for line in Path(p).read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if 'loss' in rec and 'epoch' in rec:
                train.append(rec)
                continue
            stripped = {k.split('/', 1)[-1]: v for k, v in rec.items()}
            if 'F1' in stripped:
                val.append(stripped)
    return train, val


def loss_per_epoch(train):
    by_epoch = {}
    for r in train:
        by_epoch.setdefault(int(r['epoch']), []).append(float(r['loss']))
    return {e: float(np.mean(v)) for e, v in sorted(by_epoch.items())}


def val_curve(val):
    curve = {}
    for r in val:
        epoch = int(r['step'])
        curve[epoch] = {k: float(v) for k, v in r.items() if k != 'step' and isinstance(v, (int, float))}
    return dict(sorted(curve.items()))


def parse_log_wallclock(paths):
    """Wall clock between the first and last 'Epoch(train)' log line, divided by the number
    of epochs spanned (validation time is included, which is what a user waits for)."""
    first = last = None
    for p in paths:
        for line in Path(p).read_text().splitlines():
            m = LOG_TRAIN_RE.match(line)
            if not m:
                continue
            ts = dt.datetime.strptime(m.group(1), TS_FMT)
            epoch = int(m.group(2))
            if first is None:
                first = (ts, epoch)
            last = (ts, epoch)
    if first is None:
        return {'first_ts': None, 'last_ts': None, 'first_epoch': None, 'last_epoch': None,
                'wall_s': None, 'sec_per_epoch': None}
    wall = (last[0] - first[0]).total_seconds()
    spanned = last[1] - first[1]
    return {'first_ts': first[0], 'last_ts': last[0], 'first_epoch': first[1], 'last_epoch': last[1],
            'wall_s': wall, 'sec_per_epoch': (wall / spanned) if spanned > 0 else None}


def summarize_training(work_dir):
    work_dir = Path(work_dir)
    scalars = sorted(work_dir.glob('*/vis_data/scalars.json'))
    logs = sorted(work_dir.glob('*/*.log'))
    if not scalars:
        return None
    train, val = parse_scalars(scalars)
    losses = loss_per_epoch(train)
    curve = val_curve(val)
    nan_epochs = sorted(e for e, l in losses.items() if math.isnan(l))
    best = max(((e, m['F1']) for e, m in curve.items()), key=lambda t: t[1], default=None)
    last = (max(curve), curve[max(curve)]['F1']) if curve else None
    test_file = work_dir / 'test' / 'evaluation_total_test.txt'
    return {
        'epochs_done': max(losses) if losses else 0,
        'final_loss': losses[max(losses)] if losses else None,
        'nan_epochs': nan_epochs,
        'losses': losses,
        'val': curve,
        'best_val': best,
        'last_val': last,
        'wallclock': parse_log_wallclock(logs),
        'test': parse_final_eval(test_file) if test_file.exists() else None,
    }


def _f(x, nd=4):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return 'n/a'
    return f'{x:.{nd}f}'


def render(date, release, training, epochs_target=200):
    lines = [f'# ForestFormer3D benchmark on carrot ({date})', '',
             'Hardware: carrot (H100), image `forestformer3d:cu118`. Dataset: ForAINetV2 test split '
             '(27 plots), scored with the fixed `tools/final_eval.py`. Old code = `main` @ 6a75c37 '
             'in a git worktree, same image. Spec: `docs/superpowers/specs/2026-09-22-ff3d-fixes-benchmark-tegel-design.md` section 5.',
             '', '## Table 1: released checkpoint epoch_3000, old vs fixed inference', '',
             '| variant | ' + ' | '.join(RELEASE_COLUMNS) + ' |',
             '|' + '---|' * (len(RELEASE_COLUMNS) + 1)]
    for variant in ('old', 'fixed'):
        m = release.get(variant)
        cells = [_f(m.get(c)) if m else 'n/a' for c in RELEASE_COLUMNS]
        lines.append(f'| {variant} | ' + ' | '.join(cells) + ' |')
    lines += ['', '## Table 2: 200-epoch training, old vs fixed', '',
              '| variant | epochs | sec/epoch | final train loss | best val F1 (epoch) | last val F1 (epoch) | test F1 (epoch_200) | test mIoU | NaN epochs |',
              '|---|---|---|---|---|---|---|---|---|']
    for variant in ('old', 'fixed'):
        s = training.get(variant)
        if not s:
            lines.append(f'| {variant} | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |')
            continue
        best = f"{_f(s['best_val'][1])} ({s['best_val'][0]})" if s['best_val'] else 'n/a'
        last = f"{_f(s['last_val'][1])} ({s['last_val'][0]})" if s['last_val'] else 'n/a'
        test_f1 = _f(s['test']['F1']) if s['test'] else 'n/a'
        test_miou = _f(s['test'].get('mIoU')) if s['test'] else 'n/a'
        nan = ', '.join(map(str, s['nan_epochs'])) if s['nan_epochs'] else 'none'
        lines.append(f"| {variant} | {s['epochs_done']} | {_f(s['wallclock']['sec_per_epoch'], 1)} | "
                     f"{_f(s['final_loss'])} | {best} | {last} | {test_f1} | {test_miou} | {nan} |")
    epochs = sorted({e for s in training.values() if s for e in s['val']})
    if epochs:
        lines += ['', '### Validation curve (val split, every val_interval epochs)', '',
                  '| epoch | old val F1 | fixed val F1 | old train loss | fixed train loss |', '|---|---|---|---|---|']
        for e in epochs:
            row = [str(e)]
            for variant in ('old', 'fixed'):
                s = training.get(variant)
                row.append(_f(s['val'][e]['F1']) if s and e in s['val'] else 'n/a')
            for variant in ('old', 'fixed'):
                s = training.get(variant)
                row.append(_f(s['losses'].get(e)) if s else 'n/a')
            lines.append('| ' + ' | '.join(row) + ' |')
    for variant in ('old', 'fixed'):
        s = training.get(variant)
        w = s['wallclock'] if s else None
        if w and w['first_ts']:
            lines.append('')
            lines.append(f"Wall clock {variant}: {w['first_ts']:%Y-%m-%d %H:%M} to {w['last_ts']:%Y-%m-%d %H:%M} "
                         f"({w['wall_s'] / 3600:.2f} h over epochs {w['first_epoch']}..{w['last_epoch']}, validation included).")
    lines += ['', '## Success criteria', '']
    old, fixed = release.get('old'), release.get('fixed')
    if old and fixed:
        ok = fixed['F1'] >= old['F1']
        lines.append(f"- Released checkpoint: fixed F1 {_f(fixed['F1'])} >= old F1 {_f(old['F1'])}: {'PASS' if ok else 'FAIL'}")
    else:
        lines.append('- Released checkpoint: NOT RUN')
    s = training.get('fixed')
    if s:
        ok = s['epochs_done'] >= epochs_target and not s['nan_epochs'] and len(s['val']) > 0
        lines.append(f"- Fixed 200-epoch run: {s['epochs_done']}/{epochs_target} epochs, "
                     f"{'no NaN loss' if not s['nan_epochs'] else 'NaN at epochs ' + str(s['nan_epochs'])}, "
                     f"{len(s['val'])} val points: {'PASS' if ok else 'FAIL'}")
    else:
        lines.append('- Fixed 200-epoch run: NOT RUN')
    s = training.get('old')
    lines.append(f"- Old 200-epoch run (informational): {s['epochs_done'] if s else 0}/{epochs_target} epochs"
                 + (', NaN at epochs ' + str(s['nan_epochs']) if s and s['nan_epochs'] else '')
                 + ('' if s else ': NOT RUN'))
    lines.append('')
    return '\n'.join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--root', default=os.environ.get('FF3D_ROOT', '.'))
    parser.add_argument('--date', default=dt.date.today().isoformat())
    parser.add_argument('--out', default=None, help='default: <root>/docs/benchmarks/<date>-carrot-ff3d.md')
    parser.add_argument('--allow-missing', action='store_true', help='write n/a instead of failing')
    args = parser.parse_args(argv)
    root = Path(args.root)
    out = Path(args.out) if args.out else root / 'docs' / 'benchmarks' / f'{args.date}-carrot-ff3d.md'

    missing = []
    release = {}
    for variant in ('old', 'fixed'):
        f = root / 'work_dirs' / f'bench-release-{variant}' / 'evaluation_total_test.txt'
        if f.exists():
            release[variant] = parse_final_eval(f)
        else:
            missing.append(str(f))
    training = {}
    for variant in ('old', 'fixed'):
        d = root / 'work_dirs' / f'bench-{variant}-200'
        s = summarize_training(d) if d.exists() else None
        if s is None:
            missing.append(str(d / '*/vis_data/scalars.json'))
        training[variant] = s
    if missing and not args.allow_missing:
        for m in missing:
            print(f'missing: {m}', file=sys.stderr)
        print('use --allow-missing to write a partial report', file=sys.stderr)
        return 1
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render(args.date, release, training))
    print(f'wrote {out}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cd /Users/christian/ForestFormer3D && chmod +x benchmark/collect.py && python3 -m pytest tests/test_benchmark_collect.py -v`
Expected: `7 passed`.

- [ ] **Step 5: Run the whole CPU suite once**

Run: `cd /Users/christian/ForestFormer3D && python3 -m pytest -q`
Expected: all Phase 1 tests plus `test_benchmark_fetch.py` (3) and `test_benchmark_collect.py` (7) pass, `test_benchmark_unfix.py` skipped (1 item) on the Mac; no failures.

- [ ] **Step 6: Commit**

```bash
cd /Users/christian/ForestFormer3D
git add benchmark/collect.py tests/test_benchmark_collect.py
git commit -m "bench: collect release and training results into a markdown report

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 7: Runbook `docs/benchmarks/RUNBOOK-carrot.md`

**Files:**
- Create: `docs/benchmarks/RUNBOOK-carrot.md`

**Interfaces:**
- Consumes: every script above by its final name and CLI.
- Produces: the ordered command list executed in Task 8.

- [ ] **Step 1: Write the runbook**

````markdown
# Runbook: Phase 2 benchmark on carrot

All commands run on carrot inside `/raid/cwinkelmann/ForestFormer3D` unless stated. Scripts
live in `benchmark/`; each long one daemonizes itself with `nohup` and prints its log path.

## 0. Reach carrot

```bash
# on the Mac: VPN up, then
ssh carrot
cd /raid/cwinkelmann/ForestFormer3D
git fetch origin && git checkout fix/review-findings && git pull --ff-only
docker image inspect forestformer3d:cu118 --format '{{.Id}}'   # Phase 0 image must exist
nvidia-smi --query-gpu=name,memory.total --format=csv            # expect H100
```
If `docker` needs sudo on carrot: `export FF3D_DOCKER="sudo docker"` in every shell below
(also inside the `nohup` re-exec, which inherits the environment).

## 1. Data and checkpoint (one-time, idempotent, ~3 GB download)

```bash
bash benchmark/fetch_zenodo.sh --list        # 3 lines: train_val_data.zip, test_data.zip, clean_forestformer.zip
nohup bash benchmark/fetch_zenodo.sh > work_dirs/logs/fetch.log 2>&1 &   # mkdir -p work_dirs/logs first
tail -f work_dirs/logs/fetch.log             # ends with "train_val_data: 61 ply", "test_data: 27 ply", "checkpoints: .../epoch_3000_fix.pth"
```

## 2. Old-code worktree

```bash
bash benchmark/setup_old_worktree.sh         # ends with "old worktree ready: .../work_dirs/old-main @ 6a75c37"
```

## 3. Released checkpoint: old vs fixed inference

```bash
bash benchmark/run_release_eval.sh           # prints: started release-eval pid N -- follow with: tail -f work_dirs/logs/release-eval-<ts>.log
tail -f work_dirs/logs/release-eval-*.log
```
Stages in the log: preprocessing (CPU, 61 scans), `Zenodo checkpoint layout: converted|raw`,
`fixed test.py`, `old test.py`, two `final_eval.py`, two `Instance Segmentation F1 score:` lines.
Measuring duration: after the first scan, `grep -c '\.ply' <(ls work_dirs/bench-release-fixed)`
and the log timestamps give seconds per scan; multiply by 27 for each variant.

## 4. Two 200-epoch training runs, sequentially

```bash
bash benchmark/wait_idle.sh -- bash benchmark/run_train_200.sh old     # waits for GPU idle, then daemonizes
tail -f work_dirs/logs/train-old-200-*.log
# when the log ends with "train-old-200 finished":
bash benchmark/wait_idle.sh -- bash benchmark/run_train_200.sh fixed
tail -f work_dirs/logs/train-fixed-200-*.log
```
Expected duration is unknown before the first run. Measure it after 3 epochs:
```bash
grep 'Epoch(train)' work_dirs/bench-old-200/*/*.log | tail -3      # timestamps of consecutive epochs
```
seconds between two consecutive epoch lines x 200 = training time (add one validation pass
per 20 epochs, visible as `Epoch(val)` lines). Run both variants back to back; do not start
the second while the first trains, `wait_idle.sh` enforces that (it also refuses while a
`forestformer3d` container is running).

## 5. Collect

```bash
DATE=$(date +%F)
docker run --rm --gpus all -v /raid/cwinkelmann/ForestFormer3D:/workspace -w /workspace forestformer3d:cu118 \
  python benchmark/collect.py --root /workspace --date "$DATE"     # --gpus: the entrypoint imports the CUDA extensions
cat docs/benchmarks/$DATE-carrot-ff3d.md
```
(`collect.py` needs only python3 + numpy; the container is used so nothing has to be
installed on the host. Add `--allow-missing` to write a partial report while runs are pending.)

## 6. Bring the report home and commit

```bash
# on the Mac
scp carrot:/raid/cwinkelmann/ForestFormer3D/docs/benchmarks/$(date +%F)-carrot-ff3d.md \
    /Users/christian/ForestFormer3D/docs/benchmarks/
cd /Users/christian/ForestFormer3D && git add docs/benchmarks/*-carrot-ff3d.md
git commit -m "docs: carrot benchmark results

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```
Keep the raw outputs on carrot (`work_dirs/bench-*`, `work_dirs/logs`); they are not committed.

## After an SSH drop

Nothing is lost: every run is a `nohup` process that keeps writing to `work_dirs/logs/<name>-<ts>.log`.
```bash
ssh carrot
cd /raid/cwinkelmann/ForestFormer3D
ls -t work_dirs/logs | head            # newest log first
tail -f work_dirs/logs/<newest>.log
pgrep -af 'run_release_eval|run_train_200|wait_idle'   # still running?
docker ps                              # the container of the current stage
```
If the process died (reboot, OOM), re-run the same command: stage markers (`.done-train`,
`.done-test`, `.done-eval`, `.layout`, `.unpacked-<md5>`) skip finished stages and
`run_train_200.sh` resumes training from `last_checkpoint` with `--resume auto`.

## Files written by root

Docker runs as root, so outputs under `work_dirs/` and `docs/benchmarks/` are root-owned.
If `git add` or `scp` complains: `sudo chown -R $(id -u):$(id -g) work_dirs docs/benchmarks`.
````

- [ ] **Step 2: Check every command in the runbook names an existing script**

Run: `cd /Users/christian/ForestFormer3D && for s in $(grep -o 'benchmark/[a-z_0-9]*\.\(sh\|py\)' docs/benchmarks/RUNBOOK-carrot.md | sort -u); do test -f "$s" && echo "ok $s" || echo "MISSING $s"; done`
Expected: `ok` for `benchmark/collect.py`, `benchmark/fetch_zenodo.sh`, `benchmark/run_release_eval.sh`, `benchmark/run_train_200.sh`, `benchmark/setup_old_worktree.sh`, `benchmark/wait_idle.sh`; no `MISSING`.

- [ ] **Step 3: Commit**

```bash
cd /Users/christian/ForestFormer3D
git add docs/benchmarks/RUNBOOK-carrot.md
git commit -m "docs: carrot benchmark runbook

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 8: Execute on carrot and commit the report

**Files:**
- Create: `docs/benchmarks/<YYYY-MM-DD>-carrot-ff3d.md` (generated by `collect.py`)

**Interfaces:**
- Consumes: `docs/benchmarks/RUNBOOK-carrot.md` sections 0-6.
- Produces: the committed report; raw outputs stay on carrot.

This task needs VPN or the `carrot` SSH alias. If carrot is unreachable, stop here and report; do not fake numbers.

- [ ] **Step 1: Push the branch and prepare carrot (runbook section 0)**

Run on the Mac: `cd /Users/christian/ForestFormer3D && git push origin fix/review-findings`
Then `ssh carrot 'cd /raid/cwinkelmann/ForestFormer3D && git fetch origin && git checkout fix/review-findings && git pull --ff-only && docker image inspect forestformer3d:cu118 --format "{{.Id}}" && nvidia-smi --query-gpu=name --format=csv,noheader'`
Expected: the image id line and `NVIDIA H100 ...`.

- [ ] **Step 2: Run the CPU tests and the container tests once on carrot**

Run: `ssh carrot 'cd /raid/cwinkelmann/ForestFormer3D && docker run --rm --gpus all -v $PWD:/workspace -w /workspace -e PYTHONPATH=/workspace forestformer3d:cu118 python -m pytest tests/test_benchmark_unfix.py tests/test_benchmark_collect.py -q'`
Expected: `9 passed` (2 unfix + 7 collect; the container has torch, so the unfix tests run there. The fetch test needs `unzip` and `curl` on the host and was run on the Mac in Task 2).

- [ ] **Step 3: Fetch data and set up the worktree (runbook sections 1-2)**

Run the section 1 and 2 commands. Expected final lines: `train_val_data: 61 ply`, `test_data: 27 ply`, `checkpoints: /raid/cwinkelmann/ForestFormer3D/work_dirs/clean_forestformer/epoch_3000_fix.pth`, `old worktree ready: ... @ 6a75c37`. If the PLY counts differ, inspect `downloads/` and the zip contents (`unzip -l downloads/test_data.zip | head`) before continuing.

- [ ] **Step 4: Release evaluation (runbook section 3)**

Run `bash benchmark/run_release_eval.sh`, follow the log. Record from the log: the `Zenodo checkpoint layout:` line (expected `converted`, spec 8 allows `raw`), and the two `Instance Segmentation F1 score:` lines. Expected: both directories hold 27 `.ply` files and one `evaluation_total_test.txt`.

- [ ] **Step 5: Training runs (runbook section 4)**

Run old then fixed with the `wait_idle.sh` guard. After 3 epochs of the old run, note seconds per epoch in the session notes and the extrapolated total. Expected end of each log: `Instance Segmentation F1 score: ...` then `train-<variant>-200 finished`; `work_dirs/bench-<variant>-200/epoch_200.pth` exists and `epoch_200.layout` reads `raw`.

- [ ] **Step 6: Collect, copy back, commit (runbook sections 5-6)**

Run the section 5 command, then section 6. Expected: the report contains both tables with numbers, the validation-curve table with rows 20, 40, ..., 200, and a `Success criteria` section where `Released checkpoint: ... PASS` and `Fixed 200-epoch run: 200/200 epochs, no NaN loss, 10 val points: PASS`. If a criterion reads FAIL, commit the report anyway (the spec promises no target number for the 200-epoch runs) and open the discussion in the commit body: which criterion failed and the raw F1 lines.

- [ ] **Step 7: Commit the report on the Mac**

```bash
cd /Users/christian/ForestFormer3D
git add docs/benchmarks/*-carrot-ff3d.md
git commit -m "docs: carrot benchmark results (release old vs fixed, 200-epoch runs)

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

## Self-review against spec section 5

- Layout on carrot, worktree of main at `work_dirs/old-main`, same image: Task 3 + `ff3d_docker_old` in Task 1.
- `fetch_zenodo.sh` idempotent: Task 2 (md5 skip, `.unpacked-<md5>` markers, tested twice in a row).
- `run_release_eval.sh`: preprocess once, `epoch_3000` converted-or-as-is decided by the fix script's exit code, old code fed the raw checkpoint, both scored by the fixed `final_eval.py`: Task 4 via `ff3d_prepare_checkpoint` (Task 1).
- `run_train_200.sh <old|fixed>`: 200 epochs, `val_interval=20`, `max_keep_ckpts=2`, nohup with a log, last-epoch test F1 through the same scoring path: Task 5. The spec names the log `work_dirs/bench-<variant>-200/train.log`; this plan writes the nohup log to `work_dirs/logs/train-<variant>-200-<ts>.log` and mmengine's own log to `work_dirs/bench-<variant>-200/<ts>/<ts>.log`; the runbook says where to look.
- `collect.py` with the two tables, wall clock per epoch, from `evaluation_total_test.txt` and `scalars.json`: Task 6.
- Success criteria (fixed release F1 >= old; fixed run finishes without NaN with a val curve next to the old one): rendered as PASS/FAIL in Task 6, checked in Task 8.
- Spec 8 risk "released checkpoint may already be permuted": handled by the exit-code branch; the `.layout` file records what was found.
- Type consistency: `ff3d_prepare_checkpoint <in> <stem>` prints `raw|converted` and writes `<stem>_converted.pth`, `<stem>_raw.pth`, `<stem>.layout` in Tasks 1, 4, 5; `collect.py` reads `bench-release-{old,fixed}` and `bench-{old,fixed}-200/test` which are exactly the directories Tasks 4 and 5 write.
