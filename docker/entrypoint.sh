#!/usr/bin/env bash
# Container entrypoint for forestformer3d:cu118.
#   1. Installs the one remaining site-packages patch (mmdet3d transforms_3d.py: makes
#      flip/rotate/scale also transform the per-point vote_label offsets).
#   2. Links the segmentator C++ build from the image into the mounted checkout, because
#      segmentator/main.py imports .csrc.build.libsegmentator relative to the package.
#   3. Verifies the CUDA extensions import.
#   4. Executes the command given to `docker run` (default: bash).
set -euo pipefail

# Where the checkout is. ff3d_docker mounts it at /workspace, but one-container mode
# supports any mount point (ff3d_geo._mount_root), and this patch is the one thing that is
# not baked into the image -- so looking only under /workspace meant a checkout mounted at
# -v $PWD:/src ran on UNPATCHED mmdet3d transforms behind a warning. Look where it could
# be: an explicit WORKSPACE, the usual mount, then the working directory ff3d_docker's
# caller chose. Finding nothing at all is still only a warning, because `docker run
# <image> nvidia-smi -L` mounts nothing and must keep working.
PATCH_REL="replace_mmdetection_files/transforms_3d.py"
if [ -z "${WORKSPACE:-}" ]; then
    for candidate in /workspace "$PWD"; do
        if [ -f "${candidate}/${PATCH_REL}" ]; then
            WORKSPACE="$candidate"
            break
        fi
    done
fi
WORKSPACE="${WORKSPACE:-/workspace}"

MMDET3D_DIR="$(python -c 'import mmdet3d, os; print(os.path.dirname(mmdet3d.__file__))')"
PATCH_SRC="${WORKSPACE}/${PATCH_REL}"
PATCH_DST="${MMDET3D_DIR}/datasets/transforms/transforms_3d.py"
if [ -f "${PATCH_SRC}" ]; then
    cp "${PATCH_SRC}" "${PATCH_DST}"
    echo "[entrypoint] patched ${PATCH_DST}"
else
    echo "[entrypoint] WARNING: no ${PATCH_REL} under /workspace or ${PWD}; mmdet3d is "\
         "UNPATCHED. Mount the checkout, or set WORKSPACE to where it is." >&2
fi

if [ -d "${WORKSPACE}/segmentator/csrc" ] && [ ! -e "${WORKSPACE}/segmentator/csrc/build" ]; then
    ln -s /opt/segmentator/csrc/build "${WORKSPACE}/segmentator/csrc/build"
    echo "[entrypoint] linked segmentator build into ${WORKSPACE}/segmentator/csrc/build"
fi

python - <<'EOF'
import torch
assert torch.cuda.is_available(), "CUDA is not available in the container (docker: --gpus device=N, podman: --device nvidia.com/gpu=N)"
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
