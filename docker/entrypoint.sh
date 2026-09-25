#!/usr/bin/env bash
# Container entrypoint for forestformer3d:cu118.
#   1. Installs the one remaining site-packages patch (mmdet3d transforms_3d.py: makes
#      flip/rotate/scale also transform the per-point vote_label offsets).
#   2. Links the segmentator C++ build from the image into the mounted checkout, because
#      segmentator/main.py imports .csrc.build.libsegmentator relative to the package.
#   3. Verifies the CUDA extensions import.
#   4. Executes the command given to `docker run` (default: bash).
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
