# ForestFormer3D Phase 0: Environment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A CUDA 11.8 / PyTorch 2.0.1 Docker image that runs ForestFormer3D on the H100 server `carrot` with no manual post-install steps, gated by a container smoke test (one `loss()` step, one full-plot `predict()`) and a `pytest` scaffold that also runs on the Mac.

**Architecture:** The current CUDA 11.6 `Dockerfile` is kept as `Dockerfile.a100-cu116`; a new `Dockerfile` builds every CUDA extension (MinkowskiEngine, spconv, torch-scatter, torch-cluster, torch-points-kernels, segmentator) for compute 8.0/8.6/8.9/9.0 on `pytorch/pytorch:2.0.1-cuda11.8-cudnn8-devel`. `docker/entrypoint.sh` installs the one remaining site-packages patch and verifies the extensions on every container start; `docker/smoke.sh` runs `tests/gpu/test_smoke.py` inside the image. `pyproject.toml` gives the repo a pytest configuration where `pytest` runs CPU tests and `pytest -m gpu tests/gpu` runs container-only tests.

**Tech Stack:** Docker (NVIDIA runtime), PyTorch 2.0.1 + CUDA 11.8, mmengine 0.7.3 / mmdet 3.0.0 / mmdet3d 22aaa47 / mmcv 2.0.1, spconv-cu118 2.3.6, MinkowskiEngine 02fc608, torch-scatter 2.1.1, torch-cluster 1.6.1, torch-points-kernels 0.7.0, pytest.

**Spec:** `docs/superpowers/specs/2026-09-22-ff3d-fixes-benchmark-tegel-design.md` (Phase 0 = section 3; testing summary section 7; risks section 8). Shared interface contract for all four phase plans: the "Cross-phase interfaces" block reproduced under Global Constraints.

## Global Constraints

- Repo: `/Users/christian/ForestFormer3D`, branch `fix/review-findings` (already checked out). All paths below are relative to that root unless absolute.
- The Mac has no `torch`, `mmengine` or `mmdet3d`. CPU tests written in this phase must import nothing from `oneformer3d`; GPU tests live in `tests/gpu/`, are marked `@pytest.mark.gpu`, and run only with `pytest -m gpu tests/gpu` inside the container.
- Python on the Mac is `python3` (Homebrew, 3.13.1, pytest 9.1.1). Inside the image it is `python` (conda, 3.10).
- Base image `pytorch/pytorch:2.0.1-cuda11.8-cudnn8-devel`; mmcv 2.0.1 from `https://download.openmmlab.com/mmcv/dist/cu118/torch2.0/index.html`; mmengine 0.7.3; mmdet 3.0.0; mmdet3d git commit `22aaa47fdb53ce1870ff92cb7e3f96ae38d17f61`; `spconv-cu118==2.3.6`; MinkowskiEngine commit `02fc608bea4c0549b0a7b00ca1bf15dee4a0b228` built with `TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0"`, openblas, `--force_cuda`; torch-scatter 2.1.1, torch-cluster 1.6.1, torch-points-kernels 0.7.0 built from source with `FORCE_CUDA=1` and the same arch list; segmentator Karbo123 commit `76efe46d03dd27afa78df972b17d07f2c6cfb696`; extras `laspy[lazrs]`, `tqdm`, `pytest`, `plyfile`; `numpy==1.24.1` and the other pins carried over from the old Dockerfile where still compatible.
- Dockerfile rules (spec section 3.1): no `--install-option`, `pip uninstall -y` if uninstalling at all, no `apt-key adv`, no `nvidia-utils` packages in the image, no infinite-sleep `CMD`; the image ends with `ENTRYPOINT ["/workspace/docker/entrypoint.sh"]`.
- Container conventions (contract): repo mounted at `/workspace`, `PYTHONPATH=/workspace`, image tag `forestformer3d:cu118`, run pattern `docker run --rm --gpus all --shm-size=64g -v <checkout>:/workspace forestformer3d:cu118 <cmd>`.
- `pyproject.toml` (contract, exact values): `[project] name="forestformer3d" version="0.1.0" requires-python=">=3.10" dependencies=[]`; `[tool.setuptools.packages.find] include=["oneformer3d*","ff3d_geo*"]`; `[tool.pytest.ini_options] testpaths=["tests"] markers=["gpu: needs CUDA and the Docker image"] addopts="-m 'not gpu'"`. `tests/conftest.py` exists after this phase.
- `docker/entrypoint.sh` (contract): copies `replace_mmdetection_files/transforms_3d.py` into the mmdet3d site-packages, verifies imports of `torch_points_kernels.instance_iou`, `spconv`, `MinkowskiEngine`, then `exec "$@"`. `docker/smoke.sh` runs `pytest -m gpu tests/gpu/test_smoke.py` inside the container.
- The model's `loss()` currently reads `kwargs['epoch']` (`oneformer3d/oneformer3d.py:1970`); Phase 1 removes that kwarg. The smoke test passes `epoch=2000` for now and Phase 1 updates it. Likewise `predict()` currently selects full-plot tiling with `'test' in lidar_path` (`oneformer3d/oneformer3d.py:2270`); Phase 1 replaces this with `test_cfg.full_plot`.
- Every commit message ends with a blank line and then `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`.
- Steps that need `carrot` (Task 5) are manual: `ssh carrot` needs the VPN or an SSH alias and is not reachable at planning time.

---

## File map

| Path | Task | Responsibility |
|------|------|----------------|
| `pyproject.toml` | 1 | Package metadata and the pytest configuration (marker `gpu`, default `-m 'not gpu'`) |
| `tests/conftest.py` | 1 | Puts the checkout root on `sys.path` for tests |
| `tests/gpu/conftest.py` | 1 | Skips collection of GPU test modules where `torch` is missing; fails fast without CUDA |
| `tests/test_scaffold.py` | 1 | CPU proof that the pytest configuration is active |
| `.gitignore` | 1, 3 | Adds `.pytest_cache/` and `segmentator/csrc/build` |
| `Dockerfile.a100-cu116` | 2 | The old CUDA 11.6 image, renamed, unchanged |
| `Dockerfile` | 2 | The new CUDA 11.8 image |
| `docker/entrypoint.sh` | 3 | Site-packages patch, segmentator link, extension check, `exec "$@"` |
| `docker/smoke.sh` | 3 | Runs the GPU smoke test in the container |
| `.dockerignore` | 3 | Keeps `.git`, data and work_dirs out of the build context |
| `tests/gpu/test_smoke.py` | 4 | Builds the model from the config, one `loss()` and one `predict()` on a synthetic plot |
| `readme.md` | 5 | New section "Environment (CUDA 11.8 image)" |

---

### Task 1: pytest scaffold

**Files:**
- Create: `pyproject.toml`
- Create: `tests/conftest.py`
- Create: `tests/gpu/conftest.py`
- Create: `tests/test_scaffold.py`
- Modify: `.gitignore` (3 lines today)

**Interfaces:**
- Consumes: nothing.
- Produces: `pyproject.toml` with the contract's `[tool.pytest.ini_options]`; the `gpu` marker; `tests/gpu/` as the home of container-only tests (Task 4 adds `tests/gpu/test_smoke.py`; Phase 1 adds CPU tests under `tests/`).

- [ ] **Step 1: Write the CPU test that proves the configuration is active**

Create `tests/test_scaffold.py`:

```python
"""Proves the pytest configuration in pyproject.toml is picked up.

Imports nothing from the repo: the Mac has no torch/mmengine, so CPU tests in
Phase 0 must not touch `oneformer3d`.
"""
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _run_pytest(*args: str) -> str:
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", *args],
        cwd=REPO_ROOT, capture_output=True, text=True, check=False)
    return proc.stdout + proc.stderr


def test_gpu_marker_is_registered():
    # `pytest --markers` prints "@pytest.mark.gpu: needs CUDA and the Docker image"
    assert "gpu: needs CUDA and the Docker image" in _run_pytest("--markers")


def test_gpu_tests_are_deselected_by_default():
    # --collect-only never executes tests, so this cannot recurse into itself.
    out = _run_pytest("--collect-only", "-q", "tests/test_scaffold.py")
    assert "1 deselected" in out, out


@pytest.mark.gpu
def test_marker_only_selected_with_m_gpu():
    # Collected but deselected by the default addopts; selected by `pytest -m gpu`.
    assert True
```

- [ ] **Step 2: Run it to verify it fails (no configuration yet)**

Run: `cd /Users/christian/ForestFormer3D && python3 -m pytest tests/test_scaffold.py -q`
Expected: `PytestUnknownMarkWarning`/errors and `test_gpu_marker_is_registered FAILED` (the marker line is not in `--markers`), `test_gpu_tests_are_deselected_by_default FAILED` (nothing is deselected without `addopts`).

- [ ] **Step 3: Write `pyproject.toml`**

```toml
[project]
name = "forestformer3d"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = []

[tool.setuptools.packages.find]
include = ["oneformer3d*", "ff3d_geo*"]

[tool.pytest.ini_options]
testpaths = ["tests"]
markers = ["gpu: needs CUDA and the Docker image"]
addopts = "-m 'not gpu'"
```

- [ ] **Step 4: Write `tests/conftest.py` and `tests/gpu/conftest.py`**

`tests/conftest.py`:

```python
"""Shared pytest setup: put the checkout root on sys.path so tests can import repo packages.

Inside the Docker image PYTHONPATH=/workspace already does this; on the Mac it lets
Phase 1's CPU tests import pure-Python modules by file path.
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
```

`tests/gpu/conftest.py`:

```python
"""GPU-only tests: run with `pytest -m gpu tests/gpu` inside the Docker image (docker/smoke.sh).

Pytest imports every test module before applying `-m`, so on a machine without torch
(the Mac) the modules in this directory are not collected at all.
"""
import importlib.util

import pytest

collect_ignore_glob = [] if importlib.util.find_spec("torch") else ["test_*.py"]


@pytest.fixture(scope="session", autouse=True)
def require_cuda():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.fail("tests/gpu needs a CUDA device; run them through docker/smoke.sh")
```

- [ ] **Step 5: Ignore the pytest cache**

Append to `.gitignore` so it reads:

```gitignore
# Ignore Python bytecode
*.pyc
__pycache__/
.pytest_cache/
```

- [ ] **Step 6: Run the CPU tests to verify they pass**

Run: `cd /Users/christian/ForestFormer3D && python3 -m pytest -v`
Expected:

```
collected 3 items / 1 deselected / 2 selected

tests/test_scaffold.py::test_gpu_marker_is_registered PASSED
tests/test_scaffold.py::test_gpu_tests_are_deselected_by_default PASSED

======================= 2 passed, 1 deselected in 0.3s ========================
```

Run: `cd /Users/christian/ForestFormer3D && python3 -m pytest -m gpu tests/gpu -q`
Expected: `no tests ran` (directory exists, `torch` missing, modules not collected). No `ImportError`.

- [ ] **Step 7: Commit**

```bash
cd /Users/christian/ForestFormer3D
git add pyproject.toml tests/conftest.py tests/gpu/conftest.py tests/test_scaffold.py .gitignore
git commit -m "test: add pytest scaffold with gpu marker

pyproject.toml carries the pytest configuration: 'pytest' runs CPU tests,
'pytest -m gpu tests/gpu' runs container-only tests. tests/test_scaffold.py
proves the marker and the default deselection without importing the repo.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 2: CUDA 11.8 Dockerfile

**Amendment (2026-09-22):** The first GPU-server build failed at `FROM`: Docker Hub has no
`pytorch/pytorch:2.0.1-cuda11.8-cudnn8-devel` tag (those 2.0.1 tags stop at CUDA 11.7; the
11.8 tags start at torch 2.1.0). Per the controller ruling, the base image is now
`nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04` (Ubuntu 22.04 Python 3.10, apt `python3
python3-dev python3-pip python3-venv python-is-python3`), with torch 2.0.1 / torchvision
0.15.2 installed from the cu118 wheel index as the first pip layer. Every other pin, the
`FROM` and base-image references below and in Global Constraints are superseded by this
note; see `docs/superpowers/specs/2026-09-22-ff3d-fixes-benchmark-tegel-design.md` section
3.1 and `.superpowers/sdd/2026-09-22-ff3d-00-environment/task-2-fix-report.md` for the
ruling and full rationale.

**Files:**
- Rename: `Dockerfile` -> `Dockerfile.a100-cu116` (unchanged content)
- Create: `Dockerfile`

**Interfaces:**
- Consumes: `docker/entrypoint.sh` (created in Task 3; the `COPY` line below refers to it, and the image is first built in Task 5, so the order of Tasks 2 and 3 does not matter for the build).
- Produces: image `forestformer3d:cu118` with `PYTHONPATH=/workspace`, `WORKDIR /workspace`, compiled extensions importable as `MinkowskiEngine`, `spconv.pytorch`, `torch_scatter`, `torch_cluster`, `torch_points_kernels`, `segmentator` (site-packages symlink to `/opt/segmentator`; the build directory `/opt/segmentator/csrc/build/libsegmentator.so` is what Task 3's entrypoint links into the mounted checkout); build arg `TORCH_CUDA_ARCH_LIST` (default `8.0;8.6;8.9;9.0`, spec section 8 fallback `8.0;8.6;8.9+PTX`).

- [ ] **Step 1: Keep the old image for reference**

```bash
cd /Users/christian/ForestFormer3D && git mv Dockerfile Dockerfile.a100-cu116
```

- [ ] **Step 2: Write the new `Dockerfile`**

Differences from `Dockerfile.a100-cu116`, all deliberate: Ubuntu 20.04 in the new base ships gcc 9, so the toolchain PPA goes; `apt-key adv`, `nvidia-utils-530`, `nvidia-cuda-dev` (a distro CUDA that would shadow the image's 11.8), `debugpy`, the mid-file `CMD` sleep loop and the unpinned `torch-cluster` reinstall go; MinkowskiEngine is built from a clone because pip 23 has no `--install-option`; spconv/cumm move to cu118; `pccm`, `ccimport`, `lark`, `pybind11`, `ninja` are resolved by pip as spconv's dependencies instead of being pinned to cu116-era versions; segmentator is built under `/opt` so the `/workspace` mount cannot hide it.

```dockerfile
# ForestFormer3D runtime image: PyTorch 2.0.1 / CUDA 11.8, extensions compiled for
# compute 8.0 (A100), 8.6 (A10/A40), 8.9 (L4/L40) and 9.0 (H100).
# The previous CUDA 11.6 image is kept unchanged as Dockerfile.a100-cu116.
#
# Build (checkout root):  docker build -t forestformer3d:cu118 .
# Run:  docker run --rm --gpus all --shm-size=64g -v "$PWD":/workspace forestformer3d:cu118 <cmd>
FROM pytorch/pytorch:2.0.1-cuda11.8-cudnn8-devel

# Override with --build-arg TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9+PTX" if MinkowskiEngine
# fails to compile for 9.0 (design spec section 8).
ARG TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0"
ENV TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST} \
    CUDA_HOME=/usr/local/cuda \
    PATH=/usr/local/cuda/bin:$PATH \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH \
    PYTHONPATH=/workspace \
    DEBIAN_FRONTEND=noninteractive

# Compilers and CMake for the CUDA extensions, OpenBLAS for MinkowskiEngine,
# GL/X runtime libraries for the open3d and opencv wheels. Ubuntu 20.04 ships gcc 9,
# which CUDA 11.8 supports, so no toolchain PPA is needed.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential cmake ninja-build git rsync \
        libopenblas-dev \
        libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Pins every later build step must see: numpy 1.24 (numba 0.57 / mmdet3d ceiling) and a
# setuptools that still runs `python setup.py install` for MinkowskiEngine.
RUN pip install --no-cache-dir numpy==1.24.1 "setuptools==67.8.0" wheel

# OpenMMLab stack. --no-deps: the numeric pins are installed explicitly further down.
RUN pip install --no-cache-dir --no-deps \
        mmengine==0.7.3 \
        mmdet==3.0.0 \
        mmsegmentation==1.0.0 \
        git+https://github.com/open-mmlab/mmdetection3d.git@22aaa47fdb53ce1870ff92cb7e3f96ae38d17f61 \
    && pip install --no-cache-dir --no-deps mmcv==2.0.1 \
        -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.0/index.html

# spconv for CUDA 11.8 with its matching cumm build (spconv 2.3.6 needs cumm-cu118 >=0.4.5,<0.5).
RUN pip install --no-cache-dir spconv-cu118==2.3.6 cumm-cu118==0.4.11 \
    && python -c "import spconv.pytorch"

# MinkowskiEngine at the commit the paper used, built from a clone: pip 23 removed
# --install-option, so the flags go to setup.py directly. TORCH_CUDA_ARCH_LIST is read by
# torch's CUDAExtension; MAX_JOBS caps the parallel nvcc invocations.
RUN git clone https://github.com/NVIDIA/MinkowskiEngine.git /opt/MinkowskiEngine \
    && cd /opt/MinkowskiEngine \
    && git checkout 02fc608bea4c0549b0a7b00ca1bf15dee4a0b228 \
    && MAX_JOBS=16 python setup.py install --blas=openblas --force_cuda \
    && cd / && rm -rf /opt/MinkowskiEngine \
    && python -c "import MinkowskiEngine as ME; print('MinkowskiEngine', ME.__version__)"

# torch-scatter 2.1.1 and torch-cluster 1.6.1 from source (tags are unprefixed) for the
# same arch list. --no-build-isolation so setup.py sees the image's torch.
RUN git clone --depth 1 --branch 2.1.1 https://github.com/rusty1s/pytorch_scatter.git /opt/pytorch_scatter \
    && cd /opt/pytorch_scatter \
    && FORCE_CUDA=1 pip install --no-cache-dir --no-deps --no-build-isolation . \
    && cd / && rm -rf /opt/pytorch_scatter \
    && git clone --depth 1 --branch 1.6.1 https://github.com/rusty1s/pytorch_cluster.git /opt/pytorch_cluster \
    && cd /opt/pytorch_cluster \
    && FORCE_CUDA=1 pip install --no-cache-dir --no-deps --no-build-isolation . \
    && cd / && rm -rf /opt/pytorch_cluster \
    && python -c "import torch_scatter, torch_cluster; print('torch_scatter', torch_scatter.__version__, 'torch_cluster', torch_cluster.__version__)"

# Numeric / IO pins carried over from Dockerfile.a100-cu116 (cu116-specific cumm/spconv and
# the packages pip already resolved for spconv are dropped).
RUN pip install --no-cache-dir --no-deps \
        addict==2.4.0 \
        yapf==0.33.0 \
        termcolor==2.3.0 \
        packaging==23.1 \
        rich==13.3.5 \
        opencv-python==4.7.0.72 \
        pycocotools==2.0.6 \
        Shapely==1.8.5 \
        scipy==1.10.1 \
        terminaltables==3.1.10 \
        numba==0.57.0 \
        llvmlite==0.40.0 \
        pyquaternion==0.9.9 \
        lyft-dataset-sdk==0.0.8 \
        nuscenes-devkit==1.1.10 \
        pandas==2.0.1 \
        python-dateutil==2.8.2 \
        matplotlib==3.5.2 \
        pyparsing==3.0.9 \
        cycler==0.11.0 \
        kiwisolver==1.4.4 \
        scikit-learn==1.2.2 \
        joblib==1.2.0 \
        threadpoolctl==3.1.0 \
        cachetools==5.3.0 \
        trimesh==3.21.6 \
        open3d==0.17.0 \
        plotly==5.18.0 \
        dash==2.14.2 \
        plyfile==1.0.2 \
        flask==3.0.0 \
        werkzeug==3.0.1 \
        click==8.1.7 \
        blinker==1.7.0 \
        itsdangerous==2.1.2 \
        importlib_metadata==2.1.2 \
        zipp==3.17.0 \
        tensorboard==2.15.1 \
        tensorboard-data-server==0.7.2 \
        protobuf \
        absl-py \
        future \
        MarkupSafe==2.0.1 \
        markdown \
        grpcio \
        google-auth-oauthlib \
        google-auth \
        requests-oauthlib \
        oauthlib

# torch-points-kernels 0.7.0 (oneformer3d/panoptic_losses.py imports instance_iou). PyPI has
# only an sdist, so this is a source build; FORCE_CUDA makes it compile the CUDA kernels.
RUN FORCE_CUDA=1 pip install --no-cache-dir --no-deps --no-build-isolation torch-points-kernels==0.7.0 \
    && python -c "from torch_points_kernels import instance_iou; print('torch_points_kernels OK')"

# ScanNet superpoint segmentator (data/ForAINetV2/batch_load_ForAINetV2_data.py imports it).
# Built under /opt so the /workspace mount cannot hide it; `make install` only symlinks the
# clone into site-packages, and docker/entrypoint.sh links csrc/build into the mounted
# checkout because the repo's segmentator/main.py imports .csrc.build.libsegmentator.
RUN git clone https://github.com/Karbo123/segmentator.git /opt/segmentator \
    && cd /opt/segmentator \
    && git reset --hard 76efe46d03dd27afa78df972b17d07f2c6cfb696 \
    && mkdir -p csrc/build && cd csrc/build \
    && cmake .. \
        -DCMAKE_PREFIX_PATH="$(python -c 'import torch; print(torch.utils.cmake_prefix_path)')" \
        -DPYTHON_INCLUDE_DIR="$(python -c 'import sysconfig; print(sysconfig.get_paths()["include"])')" \
        -DPYTHON_LIBRARY="$(python -c 'import sysconfig; print(sysconfig.get_config_var("LIBDIR") + "/libpython3.10.so")')" \
        -DCMAKE_INSTALL_PREFIX="$(python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')" \
    && make -j"$(nproc)" \
    && make install \
    && test -f /opt/segmentator/csrc/build/libsegmentator.so \
    && cd / && python -c "import segmentator; print('segmentator OK')"

# Project extras: LAS/LAZ IO for Phase 3, progress bars, tests.
RUN pip install --no-cache-dir laspy==2.5.3 lazrs==0.5.3 tqdm==4.66.1 pytest==7.4.4

# Everything importable together (mmdet3d checks the mmcv/mmdet/mmengine version ranges).
RUN python -c "import mmcv, mmengine, mmdet, mmdet3d, numpy, spconv.pytorch, MinkowskiEngine, torch_scatter, torch_cluster, segmentator, laspy, plyfile, open3d, tqdm; from torch_points_kernels import instance_iou; print('mmcv', mmcv.__version__, 'mmengine', mmengine.__version__, 'mmdet', mmdet.__version__, 'mmdet3d', mmdet3d.__version__, 'numpy', numpy.__version__)"

WORKDIR /workspace
# The checkout is normally mounted over /workspace; the copy only makes the image
# self-contained when run without a mount.
COPY docker/entrypoint.sh /workspace/docker/entrypoint.sh
RUN chmod +x /workspace/docker/entrypoint.sh
ENTRYPOINT ["/workspace/docker/entrypoint.sh"]
```

- [ ] **Step 3: Static checks on the Mac (the build itself is Task 5 on carrot)**

Run:

```bash
cd /Users/christian/ForestFormer3D
git status --short Dockerfile Dockerfile.a100-cu116
grep -c -E 'install-option|apt-key|nvidia-utils|nvidia-cuda-dev|sleep' Dockerfile
grep -n -E '^ENTRYPOINT|^FROM|TORCH_CUDA_ARCH_LIST=|spconv-cu118==2.3.6|mmcv==2.0.1|02fc608|--branch 2.1.1|--branch 1.6.1|torch-points-kernels==0.7.0|76efe46' Dockerfile
diff <(git show HEAD:Dockerfile) Dockerfile.a100-cu116 && echo "old image unchanged"
```

Expected: `R  Dockerfile -> Dockerfile.a100-cu116` and `?? Dockerfile`; the forbidden-pattern count prints `0`; the second grep prints one line per pattern (FROM, ARG/ENV arch list, mmcv, spconv, MinkowskiEngine commit, both branches, torch-points-kernels, segmentator commit, ENTRYPOINT); the diff prints `old image unchanged`.

- [ ] **Step 4: Commit**

```bash
cd /Users/christian/ForestFormer3D
git add Dockerfile Dockerfile.a100-cu116
git commit -m "build: replace CUDA 11.6 image with CUDA 11.8 / PyTorch 2.0.1 Dockerfile

Keep the old file as Dockerfile.a100-cu116. The new image builds MinkowskiEngine,
spconv, torch-scatter, torch-cluster, torch-points-kernels and segmentator for
compute 8.0/8.6/8.9/9.0 so it runs on H100; drops apt-key, nvidia-utils,
--install-option and the sleep CMD; ends with the docker/entrypoint.sh ENTRYPOINT.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 3: entrypoint, smoke runner, .dockerignore

**Files:**
- Create: `docker/entrypoint.sh` (executable)
- Create: `docker/smoke.sh` (executable)
- Create: `.dockerignore`
- Modify: `.gitignore`

**Interfaces:**
- Consumes: the image layout from Task 2 (`/opt/segmentator/csrc/build`, mmdet3d in site-packages), `replace_mmdetection_files/transforms_3d.py` in the checkout.
- Produces: `docker/entrypoint.sh` as the image `ENTRYPOINT` (patch, link, verify, `exec "$@"`, default command `bash`); `docker/smoke.sh` (env `IMAGE`, default `forestformer3d:cu118`) that runs `pytest -m gpu tests/gpu/test_smoke.py` inside the container; Phase 2's benchmark scripts reuse the same `docker run` line.

- [ ] **Step 1: Write `docker/entrypoint.sh`**

```bash
#!/usr/bin/env bash
# Container entrypoint for forestformer3d:cu118.
#   1. Installs the one remaining site-packages patch (mmdet3d transforms_3d.py: makes
#      flip/rotate/scale also transform the per-point vote_label offsets).
#   2. Links the segmentator C++ build from the image into the mounted checkout, because
#      segmentator/main.py imports .csrc.build.libsegmentator relative to the package.
#   3. Verifies the CUDA extensions import.
#   4. Executes the command given to `docker run` (default: bash).
#
# Until Phase 1 lands, tools/train.py additionally needs the two mmengine patches
# (replace_mmdetection_files/loops.py and base_model.py, see readme.md step 4); the
# smoke test calls model.loss() directly and does not need them.
set -euo pipefail

WORKSPACE="${WORKSPACE:-/workspace}"

MMDET3D_DIR="$(python -c 'import mmdet3d, os; print(os.path.dirname(mmdet3d.__file__))')"
PATCH_SRC="${WORKSPACE}/replace_mmdetection_files/transforms_3d.py"
PATCH_DST="${MMDET3D_DIR}/datasets/transforms/transforms_3d.py"
if [ -f "${PATCH_SRC}" ]; then
    cp "${PATCH_SRC}" "${PATCH_DST}"
    echo "[entrypoint] patched ${PATCH_DST}"
else
    echo "[entrypoint] WARNING: ${PATCH_SRC} not found; is the checkout mounted at ${WORKSPACE}?" >&2
fi

if [ -d "${WORKSPACE}/segmentator/csrc" ] && [ ! -e "${WORKSPACE}/segmentator/csrc/build" ]; then
    ln -s /opt/segmentator/csrc/build "${WORKSPACE}/segmentator/csrc/build"
    echo "[entrypoint] linked segmentator build into ${WORKSPACE}/segmentator/csrc/build"
fi

python - <<'EOF'
import torch
assert torch.cuda.is_available(), "CUDA is not available in the container (run with --gpus all)"
from torch_points_kernels import instance_iou  # noqa: F401
import spconv.pytorch  # noqa: F401
import MinkowskiEngine  # noqa: F401
import torch_scatter  # noqa: F401
import torch_cluster  # noqa: F401
import segmentator  # noqa: F401
print(f"[entrypoint] torch {torch.__version__} cuda {torch.version.cuda} "
      f"device {torch.cuda.get_device_name(0)}; extensions OK")
EOF

if [ "$#" -eq 0 ]; then
    set -- bash
fi
exec "$@"
```

- [ ] **Step 2: Write `docker/smoke.sh`**

```bash
#!/usr/bin/env bash
# Phase 0 gate: build the model inside the CUDA image, run one loss() step and one
# full-plot predict() on a synthetic plot (tests/gpu/test_smoke.py).
#
# Usage, on the GPU host, from anywhere inside the checkout:
#   docker/smoke.sh                       # image forestformer3d:cu118
#   IMAGE=forestformer3d:dev docker/smoke.sh
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${IMAGE:-forestformer3d:cu118}"

exec docker run --rm --gpus all --shm-size=64g \
    -v "${REPO}:/workspace" \
    "${IMAGE}" \
    python -m pytest -m gpu tests/gpu/test_smoke.py -v -p no:cacheprovider
```

`-p no:cacheprovider` keeps the container (root) from writing `.pytest_cache` into the mounted checkout.

- [ ] **Step 3: Write `.dockerignore` and extend `.gitignore`**

`.dockerignore`:

```
.git
data/
work_dirs/
.idea
.pytest_cache
**/__pycache__
**/*.pyc
```

`.gitignore` (full content after the edit):

```gitignore
# Ignore Python bytecode
*.pyc
__pycache__/
.pytest_cache/
# Symlink to the image's segmentator build, created by docker/entrypoint.sh
segmentator/csrc/build
```

- [ ] **Step 4: Make the scripts executable and check them**

Run:

```bash
cd /Users/christian/ForestFormer3D
chmod +x docker/entrypoint.sh docker/smoke.sh
bash -n docker/entrypoint.sh && bash -n docker/smoke.sh && echo "syntax OK"
ls -l docker/
grep -n 'exec "\$@"' docker/entrypoint.sh
grep -n 'pytest -m gpu tests/gpu/test_smoke.py' docker/smoke.sh
```

Expected: `syntax OK`; both files listed with mode `-rwxr-xr-x`; one grep hit each.

- [ ] **Step 5: Commit**

```bash
cd /Users/christian/ForestFormer3D
git add docker/entrypoint.sh docker/smoke.sh .dockerignore .gitignore
git commit -m "build: add container entrypoint, smoke runner and .dockerignore

docker/entrypoint.sh installs the mmdet3d transforms_3d.py patch, links the
segmentator build into the mounted checkout, verifies the CUDA extensions and
execs the command. docker/smoke.sh runs the GPU smoke test in the image.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 4: GPU smoke test

**Files:**
- Create: `tests/gpu/test_smoke.py`
- Reads (no edits): `configs/oneformer3d_qs_radius16_qp300_2many.py`, `oneformer3d/oneformer3d.py:1865-2072` (`loss`), `:2260-2611` (`predict`), `oneformer3d/formatting.py` (which keys land in `gt_pts_seg`)

**Interfaces:**
- Consumes: `ForAINetV2OneFormer3D_XAwarequery.loss(batch_inputs_dict, batch_data_samples, epoch=int) -> dict[str, Tensor]` and `.predict(batch_inputs_dict, batch_data_samples) -> list[Det3DDataSample]`; `batch_inputs_dict = {'points': [Tensor[N,3] on cuda]}`. Each `Det3DDataSample` needs, exactly as the training pipeline (`Pack3DDetInputs_`) produces them: metainfo `lidar_path` (a path containing `test` selects full-plot tiling today); `gt_pts_seg` (`PointData`) with `pts_semantic_mask` LongTensor[N] (0 ground, 1 wood, 2 leaf), `pts_instance_mask` LongTensor[N] (-1 ground, else 0..K-1), `instance_mask` numpy bool[N] (`loss()` calls `torch.from_numpy` on it), `ratio_inspoint` dict {instance id: fraction of the instance inside the crop}; `gt_instances_3d` (`InstanceData_`, the length-relaxed subclass) with `labels_3d` LongTensor[K]; `eval_ann_info` dict with numpy `pts_semantic_mask`, `pts_instance_mask` (read by `predict()` for the PLY's GT columns). `predict()` writes `<test_cfg.output_dir>/<lidar_path stem>.ply` with vertex fields `x y z semantic_pred instance_pred score semantic_gt instance_gt`.
- Produces: the Phase 0 gate; Phase 1 edits two lines of this file (drop `epoch=2000`, replace the `test_data` lidar_path trick with `test_cfg.full_plot=True`).

Why two knobs are set on the model in the test: an untrained `BiSemantic` head may predict no "wood" voxel, and `loss()` then divides by zero when it samples query points (`self.query_point_num / 0` at `oneformer3d.py:1996`), so the test biases that head to call every voxel foreground; and an untrained objectness score sits near 0.25, below the config's `score_th = 0.4`, so `predict()` would keep no instance; the test sets `score_th = 0.0`. This is a plumbing gate, not a quality test.

- [ ] **Step 1: Write `tests/gpu/test_smoke.py`**

```python
"""Phase 0 gate: the CUDA image can build the model, run one loss() step and one predict().

Runs only inside the Docker image: `docker/smoke.sh` or, in the container,
`pytest -m gpu tests/gpu/test_smoke.py`.

Phase 1 changes two things here: loss() loses its `epoch` kwarg, and predict()
selects full-plot tiling with test_cfg.full_plot instead of 'test' in lidar_path.
"""
from pathlib import Path

import numpy as np
import pytest
import torch

pytestmark = pytest.mark.gpu

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG = REPO_ROOT / "configs" / "oneformer3d_qs_radius16_qp300_2many.py"
N_POINTS = 5000
# Contains "test" so the current predict() takes the full-plot tiling branch.
LIDAR_PATH = "data/ForAINetV2/test_data/smoke_plot.ply"


def make_synthetic_plot(n_points=N_POINTS, seed=0):
    """A flat ground disc (radius 10 m) with two fake trees.

    Returns points float32[N,3]; semantic int64[N] (0 ground, 1 wood, 2 leaf);
    instance int64[N] (-1 ground, 0 and 1 for the trees).
    """
    rng = np.random.default_rng(seed)
    n_ground = n_points // 2
    n_tree = (n_points - n_ground) // 2

    r = 10.0 * np.sqrt(rng.random(n_ground))
    a = 2.0 * np.pi * rng.random(n_ground)
    ground = np.stack([r * np.cos(a), r * np.sin(a), rng.normal(0.0, 0.05, n_ground)], 1)
    pts = [ground]
    sem = [np.zeros(n_ground, dtype=np.int64)]
    ins = [np.full(n_ground, -1, dtype=np.int64)]

    for inst_id, (cx, cy) in enumerate([(-4.0, -4.0), (4.0, 4.0)]):
        n_stem = n_tree // 4
        n_crown = n_tree - n_stem
        stem = np.stack([cx + rng.normal(0.0, 0.1, n_stem),
                         cy + rng.normal(0.0, 0.1, n_stem),
                         12.0 * rng.random(n_stem)], 1)
        direction = rng.normal(size=(n_crown, 3))
        direction /= np.linalg.norm(direction, axis=1, keepdims=True)
        crown = np.array([cx, cy, 12.0]) + direction * 3.0 * np.cbrt(rng.random((n_crown, 1)))
        pts += [stem, crown]
        sem += [np.ones(n_stem, dtype=np.int64), np.full(n_crown, 2, dtype=np.int64)]
        ins += [np.full(n_stem, inst_id, dtype=np.int64), np.full(n_crown, inst_id, dtype=np.int64)]

    points = np.concatenate(pts).astype(np.float32)
    semantic = np.concatenate(sem)
    instance = np.concatenate(ins)
    assert points.shape == (n_points, 3)
    return points, semantic, instance


def make_sample(semantic, instance):
    """A Det3DDataSample shaped like the output of Pack3DDetInputs_ for one plot."""
    from mmdet3d.structures import Det3DDataSample, PointData
    from oneformer3d.structures import InstanceData_

    sample = Det3DDataSample()
    sample.set_metainfo(dict(lidar_path=LIDAR_PATH))

    gt_pts_seg = PointData()
    gt_pts_seg.pts_semantic_mask = torch.from_numpy(semantic).cuda()
    gt_pts_seg.pts_instance_mask = torch.from_numpy(instance).cuda()
    gt_pts_seg.instance_mask = semantic != 0          # numpy bool: loss() calls torch.from_numpy
    gt_pts_seg.ratio_inspoint = {0: 1.0, 1: 1.0}      # both trees fully inside the "crop"
    sample.gt_pts_seg = gt_pts_seg

    gt_instances_3d = InstanceData_()
    gt_instances_3d.labels_3d = torch.tensor([1, 1]).cuda()  # semantic class per instance
    sample.gt_instances_3d = gt_instances_3d

    sample.eval_ann_info = dict(pts_semantic_mask=semantic, pts_instance_mask=instance)
    return sample


@pytest.fixture(scope="module")
def plot():
    return make_synthetic_plot()


@pytest.fixture(scope="module")
def model():
    import oneformer3d  # noqa: F401  registers the model, decoder, criterion classes
    from mmdet3d.registry import MODELS
    from mmengine.config import Config
    from mmengine.registry import init_default_scope

    cfg = Config.fromfile(str(CONFIG))
    init_default_scope("mmdet3d")
    torch.manual_seed(0)
    np.random.seed(0)
    net = MODELS.build(cfg.model).cuda()

    # Untrained BiSemantic head: make every voxel "wood" so query sampling has candidates.
    head = net.BiSemantic[1]  # Seq(MLP, Linear(num_channels, 2), LogSoftmax)
    with torch.no_grad():
        head.weight.zero_()
        head.bias.copy_(torch.tensor([-1.0, 1.0], device=head.bias.device))
    return net


def test_loss_is_finite_and_backpropagates(model, plot):
    points, semantic, instance = plot
    model.train()
    inputs = dict(points=[torch.from_numpy(points).cuda()])
    samples = [make_sample(semantic, instance)]

    # epoch > prepare_epoch (1000 in the config) exercises the query decoder and the
    # instance criterion, not only the discriminative / binary-semantic losses.
    losses = model.loss(inputs, samples, epoch=2000)

    assert "discriminative_loss" in losses and "semantic_loss_bi" in losses
    tensors = {k: v for k, v in losses.items() if torch.is_tensor(v)}
    assert len(tensors) > 2, f"criterion losses missing: {sorted(losses)}"
    for name, value in tensors.items():
        assert torch.isfinite(value).all(), f"{name} is not finite: {value}"

    total = sum(v.sum() for v in tensors.values())
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    optimizer.zero_grad()
    total.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads, "no parameter received a gradient"
    assert all(torch.isfinite(g).all() for g in grads)
    optimizer.step()


def test_predict_writes_nonempty_instance_map(model, plot, tmp_path):
    from plyfile import PlyData

    points, semantic, instance = plot
    model.eval()
    model.test_cfg["output_dir"] = str(tmp_path)  # what tools/test.py does with --work-dir
    model.score_th = 0.0                          # untrained objectness is ~0.25 < config 0.4

    inputs = dict(points=[torch.from_numpy(points).cuda()])
    samples = [make_sample(semantic, instance)]
    with torch.no_grad():
        out = model.predict(inputs, samples)

    assert len(out) == 1
    ply_path = tmp_path / "smoke_plot.ply"
    assert ply_path.exists(), f"predict() did not write {ply_path}"

    vertex = PlyData.read(str(ply_path))["vertex"]
    assert len(vertex) == N_POINTS
    instance_pred = np.asarray(vertex["instance_pred"])
    semantic_pred = np.asarray(vertex["semantic_pred"])
    assert (instance_pred >= 0).sum() > 0, "no instance survived tiling and merging"
    assert set(np.unique(semantic_pred)).issubset({-1, 0, 1, 2})
    assert np.array_equal(np.asarray(vertex["instance_gt"]), instance)
```

- [ ] **Step 2: Check it on the Mac (collection only; execution is Task 5)**

Run:

```bash
cd /Users/christian/ForestFormer3D
python3 -m py_compile tests/gpu/test_smoke.py && echo "compiles"
python3 -m pytest -q
python3 -m pytest -m gpu tests/gpu -q
```

Expected: `compiles`; `2 passed, 1 deselected`; `no tests ran` (torch missing, module not collected, no ImportError).

- [ ] **Step 3: Commit**

```bash
cd /Users/christian/ForestFormer3D
git add tests/gpu/test_smoke.py
git commit -m "test: add GPU smoke test (one loss step, one full-plot predict)

Builds ForAINetV2OneFormer3D_XAwarequery from the active config, runs loss()
with backward on a synthetic 5,000-point plot with two fake trees, then
predict() in full-plot mode and checks the written PLY has instances.
Passes epoch=2000 and a test_data lidar_path until Phase 1 removes both.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 5: README section and carrot bring-up

**Files:**
- Modify: `readme.md:34-36` (insert a section before `# ForestFormer3D environment setup`)

**Interfaces:**
- Consumes: Tasks 1-4.
- Produces: the documented build/run/smoke commands that Phase 2's `benchmark/*.sh` and Phase 3's `ff3d_geo` CLI docs refer to; the image `forestformer3d:cu118` present on carrot with the smoke test green.

- [ ] **Step 1: Add the README section**

In `readme.md`, replace this exact text (line 34 through 36):

```markdown
This version uses 2 inference iterations by default. If your trees are not extremely densely distributed, you can set the number of iterations to 1 instead.
# ForestFormer3D environment setup
This guide provides step-by-step instructions to build and configure the Docker environment for ForestFormer3D, set up debugging in Visual Studio Code, and resolve common issues.
```

with:

````markdown
This version uses 2 inference iterations by default. If your trees are not extremely densely distributed, you can set the number of iterations to 1 instead.

---

## Environment (CUDA 11.8 image)

`Dockerfile` builds `forestformer3d:cu118`: PyTorch 2.0.1 / CUDA 11.8 with MinkowskiEngine,
spconv, torch-scatter, torch-cluster, torch-points-kernels and the segmentator extension
compiled for compute 8.0, 8.6, 8.9 and 9.0 (A100, A10/A40, L4/L40, H100). The previous
CUDA 11.6 image is kept as `Dockerfile.a100-cu116` for reference; the manual steps 2 to 4
below belong to that old image. With the new image nothing is reinstalled or copied by
hand: `docker/entrypoint.sh` installs the `transforms_3d.py` patch and checks the CUDA
extensions on every container start. (Until the training loop stops needing the `epoch`
kwarg, `tools/train.py` still needs `loops.py` and `base_model.py` from step 4.)

```bash
# Build, from the checkout root (30-60 min the first time; four CUDA architectures)
docker build -t forestformer3d:cu118 .

# If MinkowskiEngine fails to compile for compute 9.0:
docker build --build-arg TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9+PTX" -t forestformer3d:cu118 .

# Run a command with the checkout mounted at /workspace
docker run --rm --gpus all --shm-size=64g -v "$PWD":/workspace forestformer3d:cu118 \
    python tools/test.py configs/oneformer3d_qs_radius16_qp300_2many.py \
    work_dirs/clean_forestformer/epoch_3000_fix.pth --work-dir work_dirs/release_eval

# Interactive shell
docker run --rm -it --gpus all --shm-size=64g -v "$PWD":/workspace forestformer3d:cu118

# Smoke test: one loss step and one full-plot inference on a synthetic plot
docker/smoke.sh
```

Tests: `pytest` at the checkout root runs the CPU tests; `pytest -m gpu tests/gpu` runs
the GPU tests and only works inside the image.

---

# ForestFormer3D environment setup (legacy CUDA 11.6 image)
This guide provides step-by-step instructions to build and configure the Docker environment for ForestFormer3D, set up debugging in Visual Studio Code, and resolve common issues. It describes `Dockerfile.a100-cu116`; see "Environment (CUDA 11.8 image)" above for the current image.
````

- [ ] **Step 2: Check and commit the README**

Run: `cd /Users/christian/ForestFormer3D && grep -n '^## Environment (CUDA 11.8 image)\|^# ForestFormer3D environment setup (legacy\|docker/smoke.sh' readme.md`
Expected: three lines (the new heading, the renamed legacy heading, the smoke.sh mention).

```bash
cd /Users/christian/ForestFormer3D
git add readme.md
git commit -m "docs: document the CUDA 11.8 image and the smoke test

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

- [ ] **Step 3: Publish the branch and bring the image up on carrot (manual)**

On the Mac:

```bash
cd /Users/christian/ForestFormer3D && git push -u origin fix/review-findings
```

Connect the VPN (or make sure the `carrot` SSH alias resolves), then:

```bash
ssh carrot
```

On carrot (the checkout already holds `data/ForAINetV2` and `work_dirs/clean_forestformer` from earlier work; keep them):

```bash
cd /raid/cwinkelmann
if [ -d ForestFormer3D/.git ]; then
    cd ForestFormer3D && git fetch origin && git checkout fix/review-findings && git pull --ff-only
else
    git clone -b fix/review-findings https://github.com/SmartForest-no/ForestFormer3D.git /raid/cwinkelmann/ForestFormer3D
    cd /raid/cwinkelmann/ForestFormer3D
fi
git log --oneline -1
nvidia-smi --query-gpu=name,driver_version --format=csv
docker build -t forestformer3d:cu118 . 2>&1 | tee /raid/cwinkelmann/ff3d-cu118-build.log
```

Expected:
- `git log` shows the Task 5 commit (`docs: document the CUDA 11.8 image and the smoke test`).
- `nvidia-smi` prints `NVIDIA H100 80GB HBM3, <driver>` with a driver of at least 520 (CUDA 11.8 requires it).
- The build's last verification layer prints `mmcv 2.0.1 mmengine 0.7.3 mmdet 3.0.0 mmdet3d 1.1.0 numpy 1.24.1`, preceded by `MinkowskiEngine 0.5.4`, `torch_scatter 2.1.1 torch_cluster 1.6.1`, `torch_points_kernels OK`, `segmentator OK`; the build ends with `Successfully tagged forestformer3d:cu118` (or `naming to docker.io/library/forestformer3d:cu118` under BuildKit).
- If the MinkowskiEngine layer fails with an nvcc error mentioning `sm_90` / `compute_90`, rebuild with `docker build --build-arg TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9+PTX" -t forestformer3d:cu118 .` and record that in `docs/benchmarks/<date>-carrot-ff3d.md` in Phase 2 (spec section 8).

- [ ] **Step 4: Run the smoke test on carrot (manual)**

```bash
cd /raid/cwinkelmann/ForestFormer3D && docker/smoke.sh
```

Expected output (timings vary; first run also compiles numba kernels):

```
[entrypoint] patched /opt/conda/lib/python3.10/site-packages/mmdet3d/datasets/transforms/transforms_3d.py
[entrypoint] linked segmentator build into /workspace/segmentator/csrc/build
[entrypoint] torch 2.0.1 cuda 11.8 device NVIDIA H100 80GB HBM3; extensions OK
============================= test session starts ==============================
collected 2 items

tests/gpu/test_smoke.py::test_loss_is_finite_and_backpropagates PASSED
tests/gpu/test_smoke.py::test_predict_writes_nonempty_instance_map PASSED

============================== 2 passed in 60s ==============================
```

The `Processing regions` tqdm bar from `predict()` (36 cylinders over the 20 m plot at step `radius/4`) appears between the two tests. `git status` on carrot afterwards shows only the ignored `segmentator/csrc/build` link, nothing to commit.

- [ ] **Step 5: Record the gate**

Nothing to commit on carrot. Note on the Mac, in the Phase 2 plan's first task or in the PR description, the image digest and the smoke run time:

```bash
ssh carrot 'docker images --digests forestformer3d:cu118 --format "{{.Repository}}:{{.Tag}} {{.Digest}} {{.Size}}"'
```

Expected: one line `forestformer3d:cu118 <none-or-sha256:...> <size ~15-20 GB>`. Phase 0 is done when Step 4 shows `2 passed`.

---

## Self-review

- Spec section 3.1 (image, pins, rules): Task 2. Section 3.2 (entrypoint: patch, verify, exec): Task 3. Section 3.3 (smoke test, `docker/smoke.sh`, gate on carrot): Tasks 3, 4, 5. Section 7 (`pyproject.toml`, `tests/`, `tests/gpu/`, `-m gpu`): Task 1. Section 8 fallback arch list: Task 2 `ARG` and Task 5 Step 3.
- Contract names: `pyproject.toml` values verbatim; `tests/conftest.py` present; `docker/entrypoint.sh`, `docker/smoke.sh`, `forestformer3d:cu118`, `--shm-size=64g`, `/workspace`, `PYTHONPATH=/workspace` all match.
- Placeholders: none; every code step has full file content, every check has a command and its expected output. The carrot steps are manual by design (spec section 8).
- Type consistency: `make_sample` produces exactly the fields `loss()` reads at `oneformer3d.py:1899-1901,1975-1976,2050-2063` (`pts_instance_mask`, `instance_mask`, `pts_semantic_mask`, `ratio_inspoint`, `gt_instances_3d.labels_3d`) and `predict()` reads at `:2263,2274-2275` (`lidar_path`, `eval_ann_info[...]`).
