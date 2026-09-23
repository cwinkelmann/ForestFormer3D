---
name: ff3d-carrot
description: Use when running anything on the carrot GPU server (or the T14 workstation) for ForestFormer3D - ssh/VPN, checkout, Docker image, benchmark/common.sh, the host geo venv, data layout, GPU etiquette and the test suites.
---

# carrot: the GPU server for ForestFormer3D

Every GPU step of this project runs on **carrot**, an 8x H100 80 GB box shared with
other users. Nothing GPU-related runs on the Mac. The Mac runs `ff3d_geo` conversions,
reports, figures and the CPU tests.

## 1. Connect

```bash
# VPN must be up first (the host is only reachable through it).
ssh carrot            # ~/.ssh/config: Host carrot carrot.internal
                      #   HostName 10.188.1.1, User cwinkelmann, ServerAliveInterval 60
```

`ssh carrot 'hostname'` prints `carrot.internal`. If it hangs, the VPN is down — do not
debug further, bring the VPN up.

## 2. The checkout

```bash
cd /raid/cwinkelmann/ForestFormer3D
git remote -v                                  # origin = https://github.com/cwinkelmann/ForestFormer3D.git
git pull --ff-only origin fix/review-findings  # the working branch; NEVER `git pull` blind
git log --oneline -1                           # note the hash for any report you write
```

On carrot the fork is the `origin` remote (on the Mac, the fork is the `fork` remote and
pushes go there, never to `origin`). The old-code benchmark worktree is a **sibling**
directory, `/raid/cwinkelmann/ff3d-old-main` (`benchmark/setup_old_worktree.sh`); it must
never live under `work_dirs/`, which is separately bind-mounted into the container.

## 3. The Docker image

```bash
docker image inspect forestformer3d:cu118 --format '{{.Id}} {{.Created}}'
# sha256:80590db5... 2026-09-22T15:22:56+02:00  -> already built, nothing to do
```

Docker on carrot is **rootless**: no `sudo`, no `docker` group fiddling, files written by a
container are owned by you. Rebuild only if that inspect fails:

```bash
cd /raid/cwinkelmann/ForestFormer3D
docker build -t forestformer3d:cu118 .          # ~19 steps, tens of minutes, resumable
```

The image is `nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04` + torch 2.0.1/torchvision
0.15.2 `+cu118`, MinkowskiEngine, spconv-cu118, torch-scatter/cluster/points-kernels and
the `segmentator` C++ extension for compute 8.0/8.6/8.9/9.0. `docker/entrypoint.sh`
asserts `torch.cuda.is_available()` and patches
`replace_mmdetection_files/transforms_3d.py` into mmdet3d at every container start — so a
CPU-only container run needs `--entrypoint python` to bypass it.

## 4. `benchmark/common.sh` — how containers are launched

Source it, never hand-roll a `docker run`:

```bash
cd /raid/cwinkelmann/ForestFormer3D
source benchmark/common.sh
export FF3D_GPU=5                     # ALWAYS pin one physical GPU
ff3d_docker python tools/test.py --help
```

`ff3d_docker` runs `docker run --rm --gpus "device=$FF3D_GPU" --shm-size=64g -e
PYTHONPATH=/workspace -w /workspace -v $FF3D_ROOT:/workspace forestformer3d:cu118 "$@"`.
Never `--gpus all` with `CUDA_VISIBLE_DEVICES` masking: that still reserves every card.

Overridable variables (all have carrot defaults):

| variable | default | meaning |
|---|---|---|
| `FF3D_ROOT` | `/raid/cwinkelmann/ForestFormer3D` | checkout mounted at `/workspace` |
| `FF3D_IMAGE` | `forestformer3d:cu118` | image |
| `FF3D_GPU` | `0` | the one physical GPU to pin |
| `FF3D_SHM` | `64g` | `--shm-size` |
| `FF3D_LOGS` | `$FF3D_ROOT/work_dirs/logs` | where `ff3d_daemonize` writes |
| `FF3D_DRY_RUN` | `0` | `1` prints every docker command instead of running it |
| `FF3D_OLD_ROOT` | `/raid/cwinkelmann/ff3d-old-main` | old-code worktree for `ff3d_docker_old` |

**Always preview a new script with `FF3D_DRY_RUN=1 FF3D_FOREGROUND=1 bash <script>` first.**
Other helpers: `ff3d_preprocess` (batch_load + create_data, marker
`data/ForAINetV2/forainetv2_oneformer3d_infos_test.pkl`, `FORCE_PREP=1` to redo),
`ff3d_prepare_checkpoint`, `ff3d_daemonize <name> <script>` (re-execs under `nohup`,
prints `started <name> pid N -- follow with: tail -f <log>` and exits, so an SSH drop
never kills a job).

## 5. The host geo venv

`ff3d_geo` is pure Python (no torch) and runs on the **host**, launching only the two GPU
steps in a container. It has its own venv:

```bash
source /raid/cwinkelmann/ff3d-geo-venv/bin/activate   # Python 3.12.3
python -m ff3d_geo --help
```

Made once with:

```bash
python3 -m venv /raid/cwinkelmann/ff3d-geo-venv
/raid/cwinkelmann/ff3d-geo-venv/bin/pip install -r tests/requirements-cpu.txt
/raid/cwinkelmann/ff3d-geo-venv/bin/pip install rasterio      # needed by `ff3d_geo masks`
```

Nothing is ever pip-installed into the Docker image.

## 6. Data layout on carrot

```
/raid/cwinkelmann/ForestFormer3D/
  data/ForAINetV2/train_val_data/        65 labelled plots (Zenodo)
  data/ForAINetV2/test_data/             29 labelled plots (Zenodo) + converted ALS tiles
  data/ForAINetV2/forainetv2_instance_data/   per-scan _vert/_sem_label/_ins_label/_offsets .npy
  work_dirs/clean_forestformer/epoch_3000_fix.pth   the released checkpoint
  work_dirs/bench-*/                     Phase 2 benchmark outputs
  work_dirs/berlin-<tile>/               per-km-tile inference outputs
  work_dirs/tegel-r12, tegel-r13         the two 100 m Phase 3 tiles
  work_dirs/logs/                        every nohup log + the per-GPU queue scripts
  inputs/                                100 m input tiles
  inputs/berlin/3dm_33_<E>_<N>_1_be.las  1 km ALS input tiles
  inputs/berlin/sub/<tile>/              100 m sub-tiles written by `ff3d_geo split`
/raid/cwinkelmann/ff3d-old-main          old-code worktree (main @ 6a75c37)
/raid/cwinkelmann/zenodo-16742708        local Zenodo cache (files.tsv + the zips)
/raid/cwinkelmann/ff3d-geo-venv          host venv
```

Neither `work_dirs/` nor `data/` is in git.

## 7. Dataset and checkpoint (Zenodo record 16742708)

```bash
cd /raid/cwinkelmann/ForestFormer3D
FF3D_ZENODO_CACHE=/raid/cwinkelmann/zenodo-16742708 bash benchmark/fetch_zenodo.sh
# ends with: train_val_data: 65 ply / test_data: 29 ply /
#            checkpoint: .../work_dirs/clean_forestformer/epoch_3000_fix.pth
```

Cache-first: with a complete cache (a `files.tsv` of `key<TAB>url<TAB>md5:<hex><TAB>size`
plus every zip it names) **nothing is downloaded**. `--list` prints the files, `--dry-run`
the actions. Idempotent via `work_dirs/.zenodo/<key>.unpacked-<md5>`.
`epoch_3000_fix.pth` is already in the **converted** spconv layout
(`tools/fix_spconv_checkpoint.py` exits 2 on it).

## 8. GPU etiquette — read before starting anything

```bash
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv
```

- **GPU 1 belongs to another user. Never touch it.** GPU 0 has carried our own
  SegmentAnyTree job. Check the table before claiming a card; pick one at 0 MiB / 0 %.
- Pin exactly one card per job (`FF3D_GPU=<n>` or `--gpu <n>`). One job per card.
- Long runs go through `nohup ... > work_dirs/logs/<name>-$(date +%Y%m%d-%H%M%S).log 2>&1 &`
  (or `ff3d_daemonize`), then poll with `tail -5 <log>` every few minutes. Never a tight
  polling loop, never a blocking `tail -f` you cannot escape.
- **Never `pkill -f python` or any broad kill** — other users' jobs share this box. Kill
  only a PID you started, found with `pgrep -af 'ff3d_geo|run_release_eval|run_train_200'`
  and `docker ps`.
- Inspect, don't disturb: `docker ps`, `ls -t work_dirs/logs | head`, `nvidia-smi`.

## 9. Tests

```bash
# In the container (GPU suite, 25 tests: model construction, tiling, a smoke scenario)
source benchmark/common.sh; FF3D_GPU=5 ff3d_docker pytest -m gpu tests/gpu
FF3D_GPU=5 bash docker/smoke.sh        # one loss step + one full plot on a synthetic plot

# On the Mac (no GPU, no torch)
cd ~/ForestFormer3D
python3 -m venv .venv-cpu && .venv-cpu/bin/pip install -r tests/requirements-cpu.txt  # once
.venv-cpu/bin/python -m pytest -q tests
```

## 10. Lenovo T14 (optional second worker)

```bash
ssh christian@192.168.188.166
cd ~/ForestFormer3D
docker build -t forestformer3d:cu118 .    # build is RESUMABLE; it was paused at step 12/19
```

RTX 4080 SUPER, 16 GB — one GPU, so `FF3D_GPU=0` and one job at a time; lower `radius` or
the `max_points` cap in `_predict_full_plot` if inference OOMs. Repo, checkpoint and a
`~/ff3d-geo-venv` are already in place. **ComfyUI on that machine holds GPU memory**;
stop it (or check `nvidia-smi` for its process) before starting an inference run.

## Related skills

`ff3d-inference-km-tiles` (the production runs), `ff3d-evaluation` (benchmarks),
`ff3d-outputs-and-viewers` (what the runs produce).
