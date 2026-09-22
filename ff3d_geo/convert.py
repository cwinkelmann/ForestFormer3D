"""LAS <-> ForestFormer3D PLY conversion with a JSON sidecar for georeferencing."""

from __future__ import annotations

import json
from pathlib import Path

import laspy
import numpy as np
import pyproj
from plyfile import PlyData, PlyElement

from ff3d_geo.origin import parse_origin

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


def results_to_las(result_ply, sidecar_path, offsets_npy, out_las) -> None:
    """Georeference a ``tools/test.py`` result PLY into a LAS 1.4 / point format 6 file.

    ``result_ply`` has vertex fields ``x y z`` (float32, coordinates centered by
    ``batch_load``), ``semantic_pred`` (int32: 0 ground, 1 wood, 2 leaf, -1 for
    points without any vote), ``instance_pred`` (int32, -1 = none) and ``score``
    (float32, -1.0 without an instance). The centering shift is undone with
    ``offsets_npy`` (``<scan>_offsets.npy``, float64 ``[mean_x, mean_y, min_z]``
    that ``batch_load`` subtracted), and the sidecar's ``origin`` restores the
    tile's UTM position: ``x = ply_x + offsets[0] + origin[0]``,
    ``y = ply_y + offsets[1] + origin[1]``, ``z = ply_z + offsets[2]``.

    The pipeline never reorders points, so ``result_ply`` is expected to have
    exactly ``sidecar["n_points"]`` vertices in the original LAS order; a
    mismatch raises ``ValueError`` rather than silently misaligning the
    restored ALS ``classification``.

    Extra dimensions written: ``treeID`` int32 (``instance_pred``, -1 = none),
    ``semantic`` uint8 (``semantic_pred``, with the nodata sentinel 255 for
    ``semantic_pred == -1`` since -1 cannot be represented as uint8), ``score``
    float32. The ALS ``classification`` is restored from the sidecar's
    ``classification_npy``; the CRS is written as a WKT VLR for the sidecar's
    EPSG code.
    """
    sidecar = json.loads(Path(sidecar_path).read_text())
    offsets = np.load(offsets_npy).astype(np.float64).reshape(-1)
    if offsets.shape != (3,):
        raise ValueError(f"{offsets_npy}: expected 3 offsets, got shape {offsets.shape}")

    vertex = PlyData.read(str(result_ply))["vertex"].data
    n_points = len(vertex)
    if n_points != sidecar["n_points"]:
        raise ValueError(
            f"{result_ply} has {n_points} points but sidecar {sidecar_path} "
            f"records {sidecar['n_points']} points; point order would not match"
        )

    classification = np.load(sidecar["classification_npy"])
    if len(classification) != n_points:
        raise ValueError(
            f"classification has {len(classification)} entries but "
            f"{result_ply} has {n_points} points"
        )

    origin_e, origin_n = sidecar["origin"]
    x = vertex["x"].astype(np.float64) + offsets[0] + origin_e
    y = vertex["y"].astype(np.float64) + offsets[1] + origin_n
    z = vertex["z"].astype(np.float64) + offsets[2]

    header = laspy.LasHeader(point_format=6, version="1.4")
    header.scales = np.array(sidecar["source_scale"], dtype=np.float64)
    header.offsets = np.floor([x.min(), y.min(), z.min()])
    header.add_extra_dim(
        laspy.ExtraBytesParams(
            name="treeID", type=np.int32, description="ForestFormer3D instance, -1 none"
        )
    )
    header.add_extra_dim(
        laspy.ExtraBytesParams(
            name="semantic", type=np.uint8, description="0 ground 1 wood 2 leaf 255 n/a"
        )
    )
    header.add_extra_dim(
        laspy.ExtraBytesParams(name="score", type=np.float32, description="instance score")
    )
    header.add_crs(pyproj.CRS.from_epsg(int(sidecar["epsg"])))

    las = laspy.LasData(header)
    las.x = x
    las.y = y
    las.z = z
    las.classification = classification.astype(np.uint8)
    las.treeID = vertex["instance_pred"].astype(np.int32)
    semantic_pred = vertex["semantic_pred"].astype(np.int64)
    las.semantic = np.where(
        semantic_pred < 0, SEMANTIC_UNLABELLED, semantic_pred
    ).astype(np.uint8)
    las.score = vertex["score"].astype(np.float32)

    out_las = Path(out_las)
    out_las.parent.mkdir(parents=True, exist_ok=True)
    las.write(str(out_las))
