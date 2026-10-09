"""Pre-inference polygon mask: keep roof and water points away from the model.

The Berlin ALS 2021 classification has no building class (roof points sit in the
vegetation bins 3/4/5; measured on 3dm_33_379_5826: 88.7 % of the points inside ALKIS
building footprints are class 5), so the model segments roofs as trees, and jetties and
moored boats over water survive every post-filter. :func:`ff3d_geo.buildings.mask_buildings`
repairs the result afterwards; this module removes such points BEFORE inference instead:
``las_to_ply`` leaves every point inside a (buffered) polygon out of the PLY the model
sees, and ``results_to_las`` puts them back at their original positions with
``semantic = 3`` (masked), ``treeID = -1`` and ``score = -1``, so the result LAS still holds
every source point in source order.

Polygon sources are GeoPackages, given as ``path`` or ``path:layer``; typical Berlin inputs
are the ALKIS building footprints (``benchmark/fetch_berlin_buildings.py``), the water
polygons of ``alkis:tatsaechlichenutzungflaechen`` (``bezeich`` in AX_Fliessgewaesser,
AX_StehendesGewaesser, AX_Hafenbecken) and the ``alkis:bauwerkeflaechen`` structures
(``bezbwf`` Landebrücke / Überdachung / Carport / Brücke), fetched with the same script.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ff3d_geo.origin import parse_origin


def parse_polygon_spec(spec) -> tuple[Path, str | None]:
    """``path`` or ``path:layer`` -> (path, layer). A ``:`` inside the path itself is
    only taken as a layer separator when the remainder holds no path separator."""
    text = str(spec)
    if ":" in text:
        head, tail = text.rsplit(":", 1)
        if tail and "/" not in tail and "\\" not in tail and not tail.lower().endswith((".gpkg", ".shp")):
            return Path(head), tail
    return Path(text), None


def load_mask_polygons(specs, bbox=None, buffer_m: float = 1.0):
    """All polygons of ``specs`` (clipped to ``bbox``, buffered) as one GeoSeries."""
    import geopandas as gpd
    import pandas as pd

    from ff3d_geo.buildings import load_footprints

    parts = []
    for spec in specs:
        path, layer = parse_polygon_spec(spec)
        if not path.is_file():
            raise FileNotFoundError(f"mask polygons {path} do not exist")
        parts.append(load_footprints(path, bbox=bbox, buffer_m=buffer_m, layer=layer))
    if not parts:
        return gpd.GeoSeries([], dtype="geometry")
    return gpd.GeoSeries(pd.concat(parts, ignore_index=True))


def mask_points(x: np.ndarray, y: np.ndarray, geoms) -> np.ndarray:
    """Boolean array: True where the point lies inside any polygon."""
    from ff3d_geo.buildings import points_in_footprints

    if len(geoms) == 0:
        return np.zeros(x.shape[0], dtype=bool)
    return points_in_footprints(x, y, geoms)


def mask_local_points(x, y, origin, specs, buffer_m: float = 1.0):
    """Mask points given in LOCAL tile coordinates against polygons in the data's CRS.

    Every LAS in this pipeline holds local tile coordinates with the origin in its file
    name (``ff3d_geo.origin.parse_origin``), while the ALKIS polygons are EPSG:25833, so
    both the bbox query and the point-in-polygon test have to run on ``origin + local``.
    Getting this wrong is silent -- the mask simply matches nothing -- and it has already
    happened twice, so both callers (:func:`ff3d_geo.convert.las_to_ply` for the
    ForestFormer3D path and :func:`write_premasked_subtiles` for the methods that read
    the split directly) go through this one function.

    Returns ``(mask, n_polygons)``.
    """
    gx = np.asarray(x, dtype=np.float64) + origin[0]
    gy = np.asarray(y, dtype=np.float64) + origin[1]
    bbox = (float(gx.min()), float(gy.min()), float(gx.max()), float(gy.max())) if gx.size else None
    geoms = load_mask_polygons(specs, bbox=bbox, buffer_m=buffer_m)
    return mask_points(gx, gy, geoms), len(geoms)


def write_premasked_subtiles(sub_dir, out_dir, specs, buffer_m: float = 1.0) -> dict:
    """Write a masked copy of a haloed split, for a method that is not ``ff3d_geo run``.

    ``ff3d_geo run`` applies the mask itself (``las_to_ply``), but SegmentAnyTree and
    AMS3D read the sub-tile LAS files directly, so they need the split itself masked.
    Per sub-tile ``<stem>.las`` of ``sub_dir`` this writes

    * ``<out_dir>/<stem>.las`` -- the points OUTSIDE the polygons, in source order, with
      the input's point format, scales and extra dimensions preserved; omitted entirely
      when every point is masked, because the segmenters cannot read an empty cloud,
    * ``<out_dir>/<stem>_premask.npz`` -- ``mask`` over all source points plus the masked
      points' ``x``, ``y``, ``z`` and ``classification``, which
      :func:`restore_premasked_las` puts back once the method has run.

    Returns a summary dict (sub-tiles, points, masked points, fully masked stems).
    """
    import laspy

    sub_dir, out_dir = Path(sub_dir), Path(out_dir)
    lass = sorted(p for p in sub_dir.glob("*.las") if not p.name.endswith("_premask.las"))
    if not lass:
        raise FileNotFoundError(f"no sub-tile LAS under {sub_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    n_points = n_masked = 0
    empty = []
    for las_path in lass:
        las = laspy.read(str(las_path))
        x = np.asarray(las.x, dtype=np.float64)
        y = np.asarray(las.y, dtype=np.float64)
        masked, _ = mask_local_points(x, y, parse_origin(las_path.name), specs, buffer_m)
        np.savez(out_dir / f"{las_path.stem}_premask.npz", mask=masked,
                 x=x[masked], y=y[masked], z=np.asarray(las.z, dtype=np.float64)[masked],
                 classification=np.asarray(las.classification)[masked].astype(np.uint8))
        n_points += x.size
        n_masked += int(masked.sum())
        if masked.all():
            empty.append(las_path.stem)
            continue
        kept = laspy.LasData(las.header, points=las.points[~masked])
        kept.write(str(out_dir / las_path.name))
    return {"sub_tiles": len(lass), "n_points": n_points, "n_masked": n_masked,
            "fully_masked": empty,
            "fraction": (n_masked / n_points) if n_points else 0.0}


def restore_premasked_las(result_las, npz_path, out_las, epsg: int = 25833) -> int:
    """Put the masked points back into one method's per-sub-tile result.

    ``result_las`` is what the method produced for the masked sub-tile (same point count
    and order as the masked input; SegmentAnyTree and AMS3D both preserve order), and
    ``npz_path`` the sidecar :func:`write_premasked_subtiles` wrote. The output carries
    every source point again, in source order, so ``ff3d_geo stitch`` accepts it against
    the ORIGINAL split manifest; the masked points get ``semantic`` 3, ``treeID`` -1 and
    ``score`` -1, exactly as ``results_to_las`` does for ForestFormer3D.

    ``result_las`` may be ``None`` for a sub-tile that was fully masked (no input was
    written for it); the output is then all-masked. Returns the point count written.
    """
    import laspy

    from ff3d_geo.convert import SEMANTIC_MASKED, result_point_header

    pm = np.load(npz_path)
    mask = np.asarray(pm["mask"], dtype=bool)
    n_total = mask.shape[0]
    n_kept = n_total - int(mask.sum())

    if result_las is None:
        if n_kept:
            raise ValueError(
                f"{npz_path} keeps {n_kept} points for the model, so its result LAS is required"
            )
        kept = None
    else:
        kept = laspy.read(str(result_las))
        if kept.header.point_count != n_kept:
            raise ValueError(
                f"{result_las} has {kept.header.point_count} points but {npz_path} keeps "
                f"{n_kept} of {n_total}; the result does not belong to this masked sub-tile"
            )

    def full(values, fill, dtype, masked_values=None):
        out = np.full(n_total, fill, dtype=dtype)
        if values is not None:
            out[~mask] = values
        if masked_values is not None:
            out[mask] = masked_values
        return out

    x = full(np.asarray(kept.x) if kept else None, 0.0, np.float64, pm["x"])
    y = full(np.asarray(kept.y) if kept else None, 0.0, np.float64, pm["y"])
    z = full(np.asarray(kept.z) if kept else None, 0.0, np.float64, pm["z"])
    classification = full(np.asarray(kept.classification) if kept else None, 0, np.uint8,
                          pm["classification"])
    tree_id = full(np.asarray(kept.treeID) if kept else None, -1, np.int32)
    semantic = full(np.asarray(kept.semantic) if kept else None, SEMANTIC_MASKED, np.uint8,
                    np.full(int(mask.sum()), SEMANTIC_MASKED, dtype=np.uint8))
    score = full(np.asarray(kept.score) if kept else None, -1.0, np.float32,
                 np.full(int(mask.sum()), -1.0, dtype=np.float32))

    scales = kept.header.scales if kept is not None else np.array([0.01, 0.01, 0.01])
    header = result_point_header(epsg, scales, np.floor([x.min(), y.min(), z.min()]))
    out = laspy.LasData(header)
    out.x, out.y, out.z = x, y, z
    out.classification = classification
    out.treeID = tree_id
    out.semantic = semantic
    out.score = score
    out_las = Path(out_las)
    out_las.parent.mkdir(parents=True, exist_ok=True)
    out.write(str(out_las))
    return n_total


def restore_premasked_dir(results_dir, premask_dir, out_dir, epsg: int = 25833) -> dict:
    """:func:`restore_premasked_las` over a whole sub-tile directory.

    Every ``*_premask.npz`` of ``premask_dir`` must be matched by a result LAS of the same
    stem in ``results_dir``, except for the stems that were fully masked (no input, hence
    no result). A result without an npz is an error: it would reach the stitch with the
    masked point count.
    """
    results_dir, premask_dir, out_dir = Path(results_dir), Path(premask_dir), Path(out_dir)
    npzs = sorted(premask_dir.glob("*_premask.npz"))
    if not npzs:
        raise FileNotFoundError(f"no *_premask.npz under {premask_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    written = restored = synthesised = 0
    missing = []
    for npz in npzs:
        stem = npz.name[: -len("_premask.npz")]
        result = results_dir / f"{stem}.las"
        if not result.is_file():
            mask = np.load(npz)["mask"]
            if not np.asarray(mask, dtype=bool).all():
                missing.append(stem)
                continue
            restore_premasked_las(None, npz, out_dir / f"{stem}.las", epsg=epsg)
            synthesised += 1
        else:
            restore_premasked_las(result, npz, out_dir / f"{stem}.las", epsg=epsg)
            restored += 1
        written += 1
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} sub-tile(s) kept points but have no result in {results_dir}: "
            + ", ".join(missing[:5]) + ("..." if len(missing) > 5 else "")
        )
    return {"written": written, "restored": restored, "fully_masked": synthesised}
