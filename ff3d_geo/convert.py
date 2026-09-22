"""LAS <-> ForestFormer3D PLY conversion with a JSON sidecar for georeferencing."""

from __future__ import annotations

import json
from pathlib import Path

import laspy
import numpy as np
from plyfile import PlyData, PlyElement

from ff3d_geo.origin import parse_origin

# Vertex layout for a *labeled* ForestFormer3D PLY (results_to_las output, Task 3):
# what data/ForAINetV2/load_forainetv2_data.py's export() reads when semantic_seg
# and treeID are both present (its ``has_labels`` branch).
PLY_VERTEX_DTYPE = [
    ("x", "f8"),
    ("y", "f8"),
    ("z", "f8"),
    ("semantic_seg", "i4"),
    ("treeID", "i4"),
]

# Vertex layout for the *unlabeled* input PLY that las_to_ply writes: x, y, z only.
# load_forainetv2_data.py's export() finds no semantic_seg/treeID fields and, run
# with --unlabeled, takes its constant-label path (semantic 0 = ground, instance -1)
# instead of raising KeyError.
_PLY_INPUT_DTYPE = [
    ("x", "f8"),
    ("y", "f8"),
    ("z", "f8"),
]

# Value written to the ``semantic`` extra dim for points the model left unlabelled (-1).
SEMANTIC_UNLABELLED = 255


def las_to_ply(
    las_path,
    ply_path,
    sidecar_path,
    origin: tuple[float, float] | None = None,
    epsg: int = 25833,
) -> dict:
    """Write an unlabeled ForestFormer3D input PLY plus a georeferencing sidecar.

    The PLY keeps the LAS coordinates exactly as stored (local tile coordinates);
    the pipeline centers them itself and records the shift in ``<scan>_offsets.npy``.
    Only ``x``, ``y``, ``z`` vertex fields are written (no ``semantic_seg``/``treeID``),
    so ``data/ForAINetV2/load_forainetv2_data.py``'s ``export(..., unlabeled=True)``
    takes its constant-label path (semantic 0 = ground, instance -1 everywhere) rather
    than reading stale/fake labels.

    The ALS ``classification`` array is saved as ``<las stem>_classification.npy``
    next to the sidecar so ``results_to_las`` (Task 3) can restore it.
    """
    las_path = Path(las_path)
    ply_path = Path(ply_path)
    sidecar_path = Path(sidecar_path)
    if origin is None:
        origin = parse_origin(las_path.name)

    las = laspy.read(str(las_path))
    n_points = int(las.header.point_count)

    vertex = np.empty(n_points, dtype=_PLY_INPUT_DTYPE)
    vertex["x"] = np.asarray(las.x, dtype=np.float64)
    vertex["y"] = np.asarray(las.y, dtype=np.float64)
    vertex["z"] = np.asarray(las.z, dtype=np.float64)

    ply_path.parent.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(vertex, "vertex")], text=False, byte_order="<").write(
        str(ply_path)
    )

    sidecar_path.parent.mkdir(parents=True, exist_ok=True)
    classification_npy = sidecar_path.parent / f"{las_path.stem}_classification.npy"
    np.save(classification_npy, np.asarray(las.classification, dtype=np.uint8))

    sidecar = {
        "stem": las_path.stem,
        "origin": [float(origin[0]), float(origin[1])],
        "epsg": int(epsg),
        "source_scale": [float(s) for s in las.header.scales],
        "source_offset": [float(o) for o in las.header.offsets],
        "source_point_format": int(las.header.point_format.id),
        "source_version": str(las.header.version),
        "n_points": n_points,
        "classification_npy": str(classification_npy.resolve()),
        "source_las": str(las_path.resolve()),
    }
    sidecar_path.write_text(json.dumps(sidecar, indent=2))
    return sidecar
