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

**The GPU flag.** `ff3d_gpu_args` in `benchmark/common.sh` picks it per runtime:
`--gpus device=N` for docker, `--device nvidia.com/gpu=N` for podman. podman nominally
accepts `--gpus` for docker compatibility, but 4.9.3 does not even list it in
`podman run --help`, and a container that starts with **no** GPU does not fail at `run` --
it fails a minute later inside torch, which reads like a broken image. So the device is
named explicitly. The value is read at call time, because the queue scripts
`export FF3D_GPU=N` *after* sourcing `common.sh` (`tests/test_benchmark_runtime.py`).

Which runtime it is comes from `$FF3D_DOCKER --version`, not from the name: the
`podman-docker` package installs `/usr/bin/docker` *as* podman, so a host can call podman
"docker" and get the `--gpus` that podman ignores -- the silent no-GPU start again, on a
host where nothing looked unusual. The name is only the fallback for when the binary cannot
be run at all (a dry run on a machine with no runtime installed).

**The "am I already in a container?" marker.** One-container mode has to know not to wrap
its GPU steps in a `docker run` that cannot exist inside the container. `FF3D_IN_CONTAINER`
says so explicitly, but the fallback is a marker file -- and the two runtimes write
different ones: docker touches `/.dockerenv`, podman writes `/run/.containerenv`. Checking
only docker's would make a podman container look like a host. `ff3d_geo/cli.py`'s
`_in_image()` checks both, and the shell side checks them once in
`benchmark/geo_env.sh`'s `ff3d_in_image`, which every script that needs the geo half
sources. The two implementations are held to the same answers, value for value, by
`tests/test_benchmark_geo_env.py` -- including mixed case, where they used to disagree
(`FF3D_IN_CONTAINER=False` meant host to Python and container to the shell).
`ff3d_docker` also passes `FF3D_IN_CONTAINER=1` and `FF3D_GPU` in, so inside a container
started by it nothing depends on a marker, and `ff3d_geo` can refuse a `--gpu` that
disagrees with the card the container was actually given.

## Migrating a host that already ran root docker

Switching the runtime does not fix the files the root daemon already wrote, and the first
podman run on the T14 died on them 20 s in:

```
PermissionError: [Errno 13] Permission denied:
  './data/ForAINetV2/points/3dm_33_382_5828_E382000_N5828000_100m.bin'
RuntimeError: step 'preprocess' failed with exit 1
```

Note what did and did not break. A root-owned *file* at mode 644 is still readable, so
nothing that only reads earlier results is affected -- the stitch can still read a km tile
inferred under root docker. What blocks is a root-owned *directory* at mode 755: a new file
cannot be created in it. `data/ForAINetV2/{points,instance_mask,semantic_mask}` are created
on the first preprocess, so on this host they belonged to root and the podman container
(mapped to `christian`) could not add to them:

```
drwxr-xr-x root:root       data/ForAINetV2/points
drwxrwxr-x christian:...   data/ForAINetV2/forainetv2_instance_data
```

Those three are regenerable outputs of `tools/create_data_forainetv2.py`, and their parent
directory is yours -- and unlinking needs write permission on the *parent*, not on the file
-- so they can be cleared without sudo and the next run recreates them:

```bash
cd data/ForAINetV2
for d in points instance_mask semantic_mask; do mv "$d" "root-owned-$d-$(date +%H%M%S)"; done
```

(910 MB, 25 MB and 210 KB on the T14; `mv` rather than `rm` so a mistake is recoverable --
deleting them later needs sudo, because the files inside are root's.)

For everything at once, including a `work_dirs/berlin-<tile>/` you want to re-run into:

```bash
sudo chown -R "$USER:$USER" ~/ForestFormer3D/{data,work_dirs,inputs}
```

How much is affected is worth checking before you start, since it decides whether you need
sudo at all:

```bash
cd ~/ForestFormer3D
for d in data work_dirs inputs; do echo "$d: $(find $d -user root | wc -l) of $(find $d | wc -l)"; done
# T14 after two km tiles under root docker:
# data: 2114 of 3127 / work_dirs: 862 of 1699 / inputs: 202 of 616
```

Everything podman writes from then on is yours, including the final LAS, tree GeoPackages
and GeoTIFFs.

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
| `--gpu N cannot be honoured: this container was started for GPU M` | right: the container holds M. Start it with `FF3D_GPU=N`, or ask for `--gpu M` |
| a detached run dies at logout | `loginctl enable-linger` not set (podman has no daemon to outlive the session) |
| an entrypoint change has no effect | ENTRYPOINT is the image's own COPY of it; rebuild, or `--entrypoint /src/docker/entrypoint.sh` with the checkout mounted at /src |

**After every driver update, regenerate the CDI spec.** The spec pins the device list and
library paths as they were at generation time; a mismatch surfaces as an unresolvable CDI
device at `podman run`, which is at least loud.
