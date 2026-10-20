# Container runtimes: docker, rootless docker, podman

Every GPU step of this project runs in `forestformer3d:cu118`, and since commit `925a382`
the *whole* pipeline can (see `docs/inference-pipeline.md`, "One container for the whole
run"). That makes the container runtime a deployment decision rather than an implementation
detail, because the runtime decides **who owns the output files** and **how the card gets
into the container**. This document is the answer to both, per host.

## The problem: root-owned results

A root docker daemon runs the container as real root, so everything the container writes
through a bind mount is `root:root` on the host:

```
$ ls -l work_dirs/berlin-3dm_33_381_5828_1_be/*.ply | head -1
-rw-r--r-- 1 root root 9142871 Oct 20 18:41 ..._E382200_N5828900_100m.ply
```

Harmless until something has to rewrite those files -- the host half of the pipeline, a
re-run, an `rsync` back to the Mac -- and in one-container mode the result LAS, the tree
GeoPackages and the GeoTIFFs land the same way. `docker run --user $(id -u)` is **not** a
fix here: `docker/entrypoint.sh` writes `replace_mmdetection_files/transforms_3d.py` into
site-packages at every container start, which needs root *inside* the container.

The fix is a rootless runtime. Rootless maps container root to the invoking user through
`/etc/subuid`, so a file the container creates as uid 0 appears on the host as yours, while
the container still believes it is root and can patch its own site-packages.

## Per host

| host | runtime | rootless | notes |
|---|---|---|---|
| carrot (8x H100) | docker | yes | `docker context show` -> `rootless`; `DockerRootDir /raid/cwinkelmann/docker` |
| olive (2x A6000) | docker | yes | as carrot |
| T14 (RTX 4080 SUPER) | **podman** | yes | root docker also present and untouched; see below |
| Mac | none | -- | no GPU, no container; `ff3d_geo` runs in `.venv-cpu` |

## Why podman on the T14, not rootless docker

Both work. They differ in one place that matters: **how the GPU is injected**.

- **Rootless docker** reaches the card through `nvidia-container-runtime`, which for a
  rootless daemon needs `no-cgroups = true` in `/etc/nvidia-container-runtime/config.toml`.
  That file is global, root-owned, and has no per-user override -- NVIDIA's install guide
  only ever edits it as root. Setting it also takes working `--gpus` away from the **root**
  daemon, which on the T14 serves other projects. One knob, two daemons, and they want
  opposite values.
- **podman** reaches the card through a CDI spec: `nvidia-ctk cdi generate` writes
  `/etc/cdi/nvidia.yaml`, which describes the device nodes and driver mounts. Root
  *publishes* a description of the hardware; the unprivileged runtime *reads* it. Nothing
  that already worked changes behaviour, and `nvidia-container-cli` is not in the loop at
  all, so its cgroups setting cannot apply.

Measured on the T14, 2026-10-20: podman 4.9.3, CDI spec in place, **`no-cgroups` left at its
packaged default** (`/etc/nvidia-container-runtime/config.toml` line 13 is still
`#no-cgroups = false`), and the container came up with the GPU:

```
$ podman run --rm --device nvidia.com/gpu=0 forestformer3d:cu118 nvidia-smi -L
[entrypoint] torch 2.0.1+cu118 cuda 11.8 device NVIDIA GeForce RTX 4080 SUPER; extensions OK
GPU 0: NVIDIA GeForce RTX 4080 SUPER (UUID: GPU-065d03a4-...)
```

That line is the whole claim verified end to end: the entrypoint asserts
`torch.cuda.is_available()`, so it could not have printed without a working device. Root
docker's `--gpus` still works on the same host.

And the ownership probe, which is the point of the exercise:

```
$ podman run --rm -v $HOME/ForestFormer3D/work_dirs:/w forestformer3d:cu118 \
    bash -c 'id -u; touch /w/.probe'
0
$ ls -l work_dirs/.probe
-rw-r--r-- 1 christian christian 0 Oct 20 21:54 work_dirs/.probe
```

uid 0 inside, `christian` outside.

## Setup

Four commands as root, once per host:

```bash
sudo apt-get install -y podman                      # pulls in uidmap (newuidmap/newgidmap)
sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml
sudo loginctl enable-linger "$USER"                 # detached runs survive logout
# verify: 644 root:root, and the devices it found
nvidia-ctk cdi list                                 # nvidia.com/gpu=0, =<UUID>, =all
```

Then, as the user, bring the image across from the root daemon's store (no sudo needed with
`docker` group membership; 18 GB, ~7 min, no intermediate file):

```bash
docker save forestformer3d:cu118 | podman load
# Loaded image: docker.io/library/forestformer3d:cu118
```

podman stores it fully qualified, but the short name still resolves against the local store,
so `FF3D_IMAGE` needs no change. Rootless podman keeps its own store under
`~/.local/share/containers/storage` (on the T14: overlay on ZFS, which works -- no
`fuse-overlayfs` needed), entirely separate from the root daemon's images and volumes.

## Using it

`FF3D_DOCKER` is the whole switch:

```bash
source benchmark/common.sh
export FF3D_DOCKER=podman
FF3D_GPU=0 ff3d_docker nvidia-smi -L                               # smoke test
FF3D_GPU=0 ff3d_docker bash benchmark/berlin_run_gpu.sh 0 <tile>   # a whole km tile
```

No queue script changes. Two things inside the repo had to learn about podman:

**The GPU flag.** `ff3d_gpu_args` in `benchmark/common.sh` derives it from `$FF3D_DOCKER`:
`--gpus device=N` for docker, `--device nvidia.com/gpu=N` for podman. podman nominally
accepts `--gpus` for docker compatibility, but 4.9.3 does not even list it in
`podman run --help`, and a container that starts with **no** GPU does not fail at `run` --
it fails a minute later inside torch, which reads like a broken image. So the device is
named explicitly. The value is read at call time, because the queue scripts
`export FF3D_GPU=N` *after* sourcing `common.sh` (`tests/test_benchmark_runtime.py`).

**The "am I already in a container?" marker.** One-container mode has to know not to wrap
its GPU steps in a `docker run` that cannot exist inside the container. `FF3D_IN_CONTAINER`
says so explicitly, but the fallback is a marker file -- and the two runtimes write
different ones: docker touches `/.dockerenv`, podman writes `/run/.containerenv`. Checking
only docker's would make a podman container look like a host; `ff3d_geo/cli.py`'s
`_in_image()` and both Berlin wrappers now check both
(`tests/test_geo_in_container.py`, `tests/test_benchmark_in_container.py`).

## What is still docker-only

- `benchmark/wait_idle.sh` reads `.HostConfig.DeviceRequests` from `docker inspect` to work
  out which card a container holds. podman records CDI devices elsewhere, so this helper
  stays docker-only. It is a carrot queue tool, and carrot is docker.
- `benchmark/sat_run_gpu.sh` calls `docker` directly rather than through `ff3d_docker`
  (SegmentAnyTree runs on carrot only).

## Troubleshooting

| symptom | cause |
|---|---|
| `Error: setting up CDI devices: unresolvable CDI devices nvidia.com/gpu=0` | no `/etc/cdi/nvidia.yaml`, or it predates a driver update -- regenerate it |
| container starts, torch reports no CUDA | `--gpus` was passed to podman and ignored; use `--device nvidia.com/gpu=N` (i.e. an old `common.sh` without `ff3d_gpu_args`) |
| `newuidmap: command not found` | `uidmap` missing (normally a podman dependency) |
| `potentially insufficient UIDs or GIDs` | `/etc/subuid`/`/etc/subgid` has no entry for the user |
| geo steps try to `docker run` inside a container | `/run/.containerenv` not checked -- pre-`5583ac5` code; or set `FF3D_IN_CONTAINER=1` |
| a detached run dies at logout | `loginctl enable-linger` not set (podman has no daemon to outlive the session) |

**After every driver update, regenerate the CDI spec.** The spec pins the device list and
library paths as they were at generation time; a mismatch surfaces as an unresolvable CDI
device at `podman run`, which is at least loud.
