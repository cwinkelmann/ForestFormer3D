# ForestFormer3D runtime image: PyTorch 2.0.1 / CUDA 11.8, extensions compiled for
# compute 8.0 (A100), 8.6 (A10/A40), 8.9 (L4/L40) and 9.0 (H100).
# The previous CUDA 11.6 image is kept unchanged as Dockerfile.a100-cu116.
#
# Base image: pytorch/pytorch only ships a 2.0.1-cuda11.8-cudnn8-devel tag starting at
# torch 2.1.0 (the 2.0.1 tags stop at CUDA 11.7). We instead build on the official
# nvidia/cuda 11.8 devel image (Ubuntu 22.04, Python 3.10) and install the torch 2.0.1
# / torchvision 0.15.2 wheels for cu118 directly from the PyTorch wheel index.
#
# Build (checkout root):  docker build -t forestformer3d:cu118 .
# Run:  docker run --rm --gpus all --shm-size=64g -v "$PWD":/workspace forestformer3d:cu118 <cmd>
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

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
# GL/X runtime libraries for the open3d and opencv wheels, and Python 3.10 itself (this
# base has no conda environment). Ubuntu 22.04 ships gcc 11, which CUDA 11.8 supports, so
# no toolchain PPA is needed.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential cmake ninja-build git rsync \
        libopenblas-dev \
        libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 libgomp1 \
        python3 python3-dev python3-pip python3-venv python-is-python3 \
    && rm -rf /var/lib/apt/lists/*

# pip<24.1 so --index-url resolution and the modern torch/torchvision wheels below behave
# the same way the previous conda-based pip did; Ubuntu's python3-pip is otherwise too old.
RUN python3 -m pip install --no-cache-dir --upgrade "pip<24.1"

# torch 2.0.1 / torchvision 0.15.2 for CUDA 11.8, from the official wheel index (no
# pytorch/pytorch:2.0.1-cuda11.8-cudnn8-devel tag exists — see the header comment).
RUN pip install --no-cache-dir torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118

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
# sysconfig's LIBDIR resolves to /usr/lib/x86_64-linux-gnu on Ubuntu 22.04, where
# python3-dev installs libpython3.10.so (there is no conda lib dir on this base image).
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
