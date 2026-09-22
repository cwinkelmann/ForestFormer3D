# Runbook: Phase 2 benchmark on carrot

All commands run on carrot inside `/raid/cwinkelmann/ForestFormer3D` unless stated otherwise.
Scripts live in `benchmark/`; each long-running one daemonizes itself with `nohup`
(`ff3d_daemonize` in `benchmark/common.sh`) and prints its log path, then exits — an SSH drop
never kills the job.

GPU assignment (carrot is an 8x H100 box shared with other users; GPU 0 is already running our
own SegmentAnyTree job, GPU 1 another user's job — never touch those):

| stage | env var |
|---|---|
| release eval | `FF3D_GPU=2` |
| old-200 training | `FF3D_GPU=3` |
| fixed-200 training | `FF3D_GPU=4` |

The two 200-epoch trainings run **in parallel** on GPU 3 and GPU 4, in separate shells.
Docker on carrot is rootless: no `sudo`, no `docker` group issues, files are owned by the
invoking user — `FF3D_DOCKER` never needs to become `sudo docker` here, and there is no
chown step.

## 0. Prerequisites

```bash
# on the Mac: VPN up, then
ssh carrot                                   # alias for cwinkelmann@10.188.1.1
cd /raid/cwinkelmann/ForestFormer3D
docker image inspect forestformer3d:cu118 --format '{{.Id}}'   # already built; must succeed
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv   # confirm GPU 2/3/4 are free
```

## 1. One-time setup (already done on carrot — commands shown for reproducibility)

```bash
git clone https://github.com/cwinkelmann/ForestFormer3D.git /raid/cwinkelmann/ForestFormer3D
cd /raid/cwinkelmann/ForestFormer3D
git checkout fix/review-findings

mkdir -p work_dirs/logs
# Zenodo record 16742708 (ForAINetV2 + epoch_3000_fix.pth), served from the local cache so
# nothing is downloaded:
FF3D_ZENODO_CACHE=/raid/cwinkelmann/zenodo-16742708 bash benchmark/fetch_zenodo.sh
# ends with: "train_val_data: 61 ply", "test_data: 27 ply",
#            "checkpoint: .../work_dirs/clean_forestformer/epoch_3000_fix.pth"

# Preprocessing (batch_load + create_data); this is also the first stage that
# run_release_eval.sh / run_train_200.sh run automatically, idempotent via the test infos
# pkl marker, so it is safe to trigger it standalone too:
bash -c 'source benchmark/common.sh; FF3D_GPU=2 ff3d_preprocess'
# marker: data/ForAINetV2/forainetv2_oneformer3d_infos_test.pkl

# Old-code worktree (main @ 6a75c37), with the entrypoint copy needed for its container:
bash benchmark/setup_old_worktree.sh
# ends with: "old worktree ready: /raid/cwinkelmann/ff3d-old-main @ 6a75c37"
```

Checkpoint layout note (also automatic, inside `run_release_eval.sh`'s
`ff3d_prepare_checkpoint`): the Zenodo `epoch_3000_fix.pth` is already in the **converted**
layout (`tools/fix_spconv_checkpoint.py` exits 2 on it), so
`work_dirs/clean_forestformer/epoch_3000_converted.pth` is a copy of it, and
`epoch_3000_raw.pth` was produced from it by `benchmark/unfix_spconv_checkpoint.py` (for the
old `tools/test.py`, which permutes weights in memory).

## 2. Released checkpoint: old vs fixed inference (GPU 2)

```bash
FF3D_GPU=2 bash benchmark/run_release_eval.sh
# prints: started release-eval pid N -- follow with: tail -f work_dirs/logs/release-eval-<ts>.log
tail -f work_dirs/logs/release-eval-*.log
```
Stages logged in order: preprocessing (skipped if already done), `Zenodo checkpoint layout:
converted|raw`, `fixed test.py` (27 test scans), `old test.py` (27 test scans), two
`final_eval.py` runs, two `Instance Segmentation F1 score:` lines. Stage markers live under
`work_dirs/bench-release/.done-{fixed,old,eval-fixed,eval-old}` — a re-run after an SSH drop
skips whatever is already marked.

Preview without running anything:
```bash
FF3D_GPU=2 FF3D_DRY_RUN=1 FF3D_FOREGROUND=1 bash benchmark/run_release_eval.sh
```

## 3. Two 200-epoch trainings, in parallel (GPU 3 and GPU 4)

In two separate shells (or `tmux`/`screen` panes) on carrot:

```bash
# shell A
FF3D_GPU=3 bash benchmark/run_train_200.sh old
# prints: started train-old-200 pid N -- follow with: tail -f work_dirs/logs/train-old-200-<ts>.log
```
```bash
# shell B
FF3D_GPU=4 bash benchmark/run_train_200.sh fixed
# prints: started train-fixed-200 pid N -- follow with: tail -f work_dirs/logs/train-fixed-200-<ts>.log
```
Each run trains 200 epochs (`FF3D_EPOCHS`, default 200) with validation every 20 epochs
(`FF3D_VAL_INTERVAL`, default 20), then scores `epoch_200.pth` on the test split through the
same `tools/test.py` + `tools/final_eval.py` path as the release eval. Stage markers:
`work_dirs/bench-<variant>-200/.done-train`, `.../test/.done-test`, `.../test/.done-eval`.

`benchmark/wait_idle.sh` (GPU-idle gate) is optional and off by default; only set
`FF3D_WAIT_IDLE=1` if a run must wait for its pinned GPU to go idle first:
```bash
FF3D_GPU=3 FF3D_WAIT_IDLE=1 bash benchmark/run_train_200.sh old
```

Preview without running anything: `FF3D_GPU=3 FF3D_DRY_RUN=1 FF3D_FOREGROUND=1 bash benchmark/run_train_200.sh old`.

### Measuring duration

Duration is unknown ahead of time. After the first few epochs of each run:
```bash
grep 'Epoch(train)' work_dirs/bench-old-200/*/*.log | head -5
grep 'Epoch(train)' work_dirs/bench-fixed-200/*/*.log | head -5
```
Take the timestamps of two consecutive `Epoch(train)[e][.../...]` lines, divide the gap by the
number of iterations between them to get sec/iteration, or compare the first line of epoch N
to the first line of epoch N+1 for sec/epoch directly; multiply by 200 (plus one `Epoch(val)`
pass every 20 epochs) for the total.

## 4. Monitoring

```bash
tail -f work_dirs/logs/release-eval-*.log
tail -f work_dirs/logs/train-old-200-*.log
tail -f work_dirs/logs/train-fixed-200-*.log

nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv   # GPU 2/3/4 busy?
docker ps                                                                # forestformer3d:cu118 containers
pgrep -af 'run_release_eval|run_train_200|wait_idle'                     # our nohup processes
```
mmengine's own per-run logs and scalars sit under
`work_dirs/bench-<variant>-200/<timestamp>/` (`<timestamp>.log`, `vis_data/scalars.json`) —
this is what `benchmark/collect.py` reads for the loss/F1 curves and wall-clock time.

## 5. Collect + report

Run once both the release eval and (at least one of) the trainings have produced results;
`--allow-missing` writes a partial report if some are still pending.

```bash
DATE=$(date +%F)
docker run --rm -e PYTHONPATH=/workspace -w /workspace \
  -v /raid/cwinkelmann/ForestFormer3D:/workspace \
  forestformer3d:cu118 python3 benchmark/collect.py --root /workspace --date "$DATE" --allow-missing
cat docs/benchmarks/$DATE-carrot-ff3d.md
```
`collect.py` is pure Python + numpy (no torch/mmengine import, no GPU needed), so no `--gpus`
flag; the container is used only so nothing extra has to be installed on the host. Drop
`--allow-missing` once all four inputs (2 release `evaluation_total_test.txt`, 2 training
work dirs) are complete, to fail loudly instead of silently on a missing one.

## 6. Bring the report home and commit

```bash
# on the Mac
scp carrot:/raid/cwinkelmann/ForestFormer3D/docs/benchmarks/$(date +%F)-carrot-ff3d.md \
    /Users/christian/ForestFormer3D/docs/benchmarks/
cd /Users/christian/ForestFormer3D
git add docs/benchmarks/$(date +%F)-carrot-ff3d.md
git commit -m "docs: carrot benchmark results"
git push fork fix/review-findings    # push to the fork remote, never origin
```
Keep the raw outputs on carrot (`work_dirs/bench-*`, `work_dirs/logs/`); they are not
committed.

## 7. Recovery after an SSH drop

Nothing is lost: every run is a `nohup` process writing to `work_dirs/logs/<name>-<ts>.log`.
```bash
ssh carrot
cd /raid/cwinkelmann/ForestFormer3D
ls -t work_dirs/logs | head                          # newest logs first
tail -f work_dirs/logs/<newest>.log
pgrep -af 'run_release_eval|run_train_200|wait_idle'  # still running?
docker ps                                             # container of the current stage
```
If the process died (reboot, OOM, preemption), re-run the exact same command with the same
`FF3D_GPU`: stage markers (`.done-fixed`, `.done-old`, `.done-eval-fixed`, `.done-eval-old`,
`.done-train`, `.done-test`, `.done-eval`, `.layout`, `.unpacked-<md5>`) skip finished stages,
and `run_train_200.sh` resumes an interrupted training from `last_checkpoint` with
`--resume auto` automatically (no flag to add — it's conditional on that file existing).

## Interpretation notes

- **Old variant = original code AND original config**, not code alone: `run_train_200.sh old`
  / the old-path stages of `run_release_eval.sh` run the ORIGINAL `main@6a75c37`
  `tools/train.py`/`tools/test.py` together with the old code's *own* original config, via the
  git worktree at `/raid/cwinkelmann/ff3d-old-main`. Only the Docker image (CUDA/Python/library
  versions) is held constant between old and fixed — code and config both differ.
- Both variants are scored with the same, current, fixed `tools/final_eval.py`, so Table 1/2
  metrics are directly comparable even though the old inference path itself differs.
- The release-eval table (Table 1) isolates inference-only regressions on the frozen
  `epoch_3000` checkpoint; the 200-epoch tables (Table 2 + validation curve) isolate
  training-loop regressions (loss curve, non-finite epochs, val/test F1) end to end.
