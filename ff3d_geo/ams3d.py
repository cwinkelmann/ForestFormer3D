"""Adaptive mean shift 3D (AMS3D) tree crown segmentation, a CPU-only benchmark method.

Ported from the throwaway spike ``docs/experiments/ams3d_spike.py`` in the
``GEE_animation`` repository (branch ``als-ams3d-spike``, 2026-09-22), whose
write-up ``docs/experiments/2026-09-22-ams3d-spike.md`` describes the method, the
parameter configs ``default``/``A``/``B``/``C`` and their results on a Spandau (R13)
hectare against the ``ALS segmentation 2017.gpkg`` crown file. The algorithm here is
the spike's, step for step, so that the port reproduces the spike's numbers on its
two patches (``tests/test_geo_ams3d.py`` checks the r12 patch); what changed is the
packaging: parameters live in :class:`Ams3dParams`, the per-point result is an
instance id array like the ForestFormer3D result contract, and
:func:`run_ams3d_tile` writes the same LAS 1.4 / point format 6 file that
``ff3d_geo.convert.results_to_las`` writes, so ``ff3d_geo.merge``, ``trees``,
``raster`` and ``report`` work on it unchanged.

Method (after Ferraz et al. 2012/2016), per vegetation point ``h_min .. h_max``
above ground:

* the kernel is a cylinder of horizontal radius ``h_s = max(s_min, a_s * h)`` and
  vertical half-height ``h_r = max(r_min, a_r * h)``, ``h`` being the height above
  ground of the CURRENT mode position, so the window grows with tree height (the
  "adaptive" part);
* the Epanechnikov product kernel makes the mean shift step the plain centroid of
  the points inside the cylinder;
* every vegetation point is a seed; converged modes closer than ``merge_xy``
  horizontally and ``merge_z`` vertically are one tree; clusters with fewer than
  ``min_points`` points are attached to the nearest bigger cluster.

The spike found that the textbook kernel splits tall leaf-off crowns vertically
into stacked clusters; merging modes generously (1.5 m / 6 m, clusters under 50
points absorbed, config ``C``) fixed that and landed at the reference tree
density, so ``C`` is the default here and ``default``/``A``/``B`` stay reachable via
:data:`CONFIGS`.

AMS3D has no wood/leaf distinction and never labels a point it does not cluster:
``semantic`` is 0 for ALS class 2 (ground), 2 (leaf) for every point that received
an instance and 255 (n/a) for the rest, and ``score`` is -1 throughout.
"""

from __future__ import annotations

import re
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import laspy
import numpy as np
import pyproj
from scipy import ndimage
from scipy.interpolate import RegularGridInterpolator
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from ff3d_geo.origin import parse_origin

#: ALS classes the spike loaded: ground (2) and the three vegetation classes.
GROUND_CLASS = 2
VEGETATION_CLASSES = (3, 4, 5)

#: ``semantic`` values of the result contract (see ``ff3d_geo.convert``).
SEMANTIC_GROUND = 0
SEMANTIC_LEAF = 2
SEMANTIC_UNLABELLED = 255

#: ``score`` written for every point: AMS3D has no instance confidence.
SCORE_NONE = -1.0

DEFAULT_EPSG = 25833

_SIZE_RE = re.compile(r"_(\d+)m(?:\.|$)")


@dataclass(frozen=True)
class Ams3dParams:
    """AMS3D parameters; the defaults are the spike's config ``C``.

    ``ground_cell_m``: cell of the min-z ground grid built from ALS class 2 that
    height-normalises the points. ``h_min``/``h_max``: only vegetation points this
    far above ground are segmented. ``a_s``/``s_min``: horizontal kernel radius
    ``max(s_min, a_s * h)``; ``a_r``/``r_min``: vertical half-height
    ``max(r_min, a_r * h)``. ``max_iter``/``tol``: mean shift stops when the mode
    moves less than ``tol`` metres or after ``max_iter`` iterations.
    ``merge_xy``/``merge_z``: converged modes closer than this are one tree.
    ``min_points``: smaller clusters are attached to the nearest bigger one.
    ``chunk``: seeds per neighbour query (bounds memory, not a result knob).
    """

    ground_cell_m: float = 1.0
    h_min: float = 2.0
    h_max: float = 60.0
    a_s: float = 0.12
    s_min: float = 1.0
    a_r: float = 0.27
    r_min: float = 1.5
    max_iter: int = 40
    tol: float = 0.05
    merge_xy: float = 1.5
    merge_z: float = 6.0
    min_points: int = 50
    chunk: int = 4000

    def as_dict(self) -> dict:
        return asdict(self)


#: The spike's four configurations (``2026-09-22-ams3d-spike.md``, results table).
CONFIGS: dict[str, Ams3dParams] = {
    "default": Ams3dParams(merge_xy=1.0, merge_z=2.0, min_points=20),
    "A": Ams3dParams(a_r=0.5, merge_xy=1.0, merge_z=4.0, min_points=20),
    "B": Ams3dParams(a_s=0.15, a_r=0.5, merge_xy=1.0, merge_z=4.0, min_points=50),
    "C": Ams3dParams(),
}


def normalize_height(xyz: np.ndarray, classification: np.ndarray, cell: float = 1.0) -> np.ndarray:
    """Height above ground from a gridded min-z of class-2 points (holes filled with
    the nearest ground cell), bilinearly interpolated to every point.

    The grid covers the bounding box of ``xyz``; raises ``ValueError`` when there
    is no ground point at all (then no height can be normalised).
    """
    xyz = np.asarray(xyz, dtype=np.float64)
    g = xyz[np.asarray(classification) == GROUND_CLASS]
    if len(g) == 0:
        raise ValueError("normalize_height: no ALS class-2 (ground) points")
    x0, y0 = xyz[:, 0].min(), xyz[:, 1].min()
    nx = int(np.ceil((xyz[:, 0].max() - x0) / cell)) + 1
    ny = int(np.ceil((xyz[:, 1].max() - y0) / cell)) + 1
    ix = ((g[:, 0] - x0) / cell).astype(int)
    iy = ((g[:, 1] - y0) / cell).astype(int)
    dtm = np.full((ny, nx), np.inf)
    np.minimum.at(dtm, (iy, ix), g[:, 2])
    hole = ~np.isfinite(dtm)
    if hole.any():
        near = ndimage.distance_transform_edt(hole, return_distances=False, return_indices=True)
        dtm = dtm[near[0], near[1]]
    interp = RegularGridInterpolator(
        (y0 + np.arange(ny) * cell, x0 + np.arange(nx) * cell), dtm,
        method="linear", bounds_error=False, fill_value=None,
    )
    return xyz[:, 2] - interp(xyz[:, [1, 0]])


def mean_shift(P: np.ndarray, params: Ams3dParams, threads: int = 1) -> tuple[np.ndarray, int]:
    """Run adaptive mean shift from every point of ``P`` (x, y, h).

    Returns the converged mode of each point and the number of iterations used.
    ``threads`` is passed to ``cKDTree.query_ball_point``; it changes the speed,
    never the result.
    """
    P = np.asarray(P, dtype=np.float64)
    k = params.a_s / params.a_r            # scale h so the cylinder has aspect ratio 1
    Ps = P * [1, 1, k]
    tree = cKDTree(Ps)
    modes = P.copy()
    active = np.ones(len(P), bool)
    for it in range(1, params.max_iter + 1):
        idx = np.flatnonzero(active)
        for chunk in np.array_split(idx, max(1, len(idx) // params.chunk)):
            q = modes[chunk]
            hs = np.maximum(params.s_min, params.a_s * q[:, 2])
            hr = np.maximum(params.r_min, params.a_r * q[:, 2])
            # a ball in scaled space that contains the cylinder
            rad = np.sqrt(hs ** 2 + (hr * k) ** 2)
            nb = tree.query_ball_point(q * [1, 1, k], rad, workers=threads, return_sorted=False)
            lens = np.fromiter(map(len, nb), dtype=np.int64, count=len(nb))
            j = np.concatenate(nb).astype(np.int64)
            i = np.repeat(np.arange(len(chunk)), lens)
            d = P[j] - q[i]
            inside = (d[:, 0] ** 2 + d[:, 1] ** 2 <= hs[i] ** 2) & (np.abs(d[:, 2]) <= hr[i])
            i, j = i[inside], j[inside]
            cnt = np.bincount(i, minlength=len(chunk)).astype(float)
            new = np.column_stack(
                [np.bincount(i, weights=P[j, c], minlength=len(chunk)) for c in range(3)]
            ) / np.maximum(cnt, 1)[:, None]
            new[cnt == 0] = q[cnt == 0]
            shift = np.linalg.norm(new - q, axis=1)
            modes[chunk] = new
            active[chunk[shift < params.tol]] = False
        if not active.any():
            return modes, it
    return modes, params.max_iter


def cluster_modes(modes: np.ndarray, params: Ams3dParams) -> np.ndarray:
    """Union modes within ``merge_xy`` horizontally and ``merge_z`` vertically."""
    S = np.asarray(modes, dtype=np.float64) * [1, 1, params.merge_xy / params.merge_z]
    tree = cKDTree(S)
    pairs = tree.query_pairs(params.merge_xy, output_type="ndarray")
    n = len(modes)
    adj = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(n, n))
    _, labels = connected_components(adj, directed=False)
    return labels


def relabel_small(P: np.ndarray, labels: np.ndarray, params: Ams3dParams,
                  threads: int = 1) -> np.ndarray:
    """Attach points of clusters smaller than ``min_points`` to the nearest big cluster."""
    counts = np.bincount(labels)
    small = counts[labels] < params.min_points
    if small.all() or not small.any():
        return labels
    tree = cKDTree(P[~small])
    _, nn = tree.query(P[small], workers=threads)
    labels = labels.copy()
    labels[small] = labels[~small][nn]
    _, labels = np.unique(labels, return_inverse=True)
    return labels


def segment_ams3d(xyz: np.ndarray, classification: np.ndarray, params: Ams3dParams,
                  threads: int = 1) -> np.ndarray:
    """Per-point instance id (int32; -1 for ground, unclassified and unassigned points).

    Only ALS classes 2/3/4/5 take part (the spike loaded nothing else): class 2
    builds the ground grid, classes 3/4/5 between ``h_min`` and ``h_max`` above it
    are segmented; every one of them ends up in a cluster, so the id is -1 exactly
    for points outside that set. Ids are ``0 .. n_trees-1`` in order of first
    appearance. Coordinates may be local or projected: the mean shift runs on
    ``x - min(x)``, ``y - min(y)`` like the spike, so the result does not depend on
    the absolute origin.
    """
    xyz = np.asarray(xyz, dtype=np.float64)
    classification = np.asarray(classification)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or len(xyz) != len(classification):
        raise ValueError("segment_ams3d: xyz must be (N, 3) and match classification")
    ids = np.full(len(xyz), -1, dtype=np.int32)

    keep = np.isin(classification, (GROUND_CLASS,) + VEGETATION_CLASSES)
    if not keep.any() or not (classification[keep] == GROUND_CLASS).any():
        return ids
    sub_xyz = xyz[keep]
    sub_cls = classification[keep]
    h = normalize_height(sub_xyz, sub_cls, params.ground_cell_m)
    veg = (h >= params.h_min) & (h <= params.h_max) & (sub_cls != GROUND_CLASS)
    if not veg.any():
        return ids
    x0, y0 = sub_xyz[:, 0].min(), sub_xyz[:, 1].min()
    P = np.column_stack([sub_xyz[veg, 0] - x0, sub_xyz[veg, 1] - y0, h[veg]])

    modes, _ = mean_shift(P, params, threads=threads)
    labels = relabel_small(P, cluster_modes(modes, params), params, threads=threads)
    _, labels = np.unique(labels, return_inverse=True)

    keep_idx = np.flatnonzero(keep)
    ids[keep_idx[veg]] = labels.astype(np.int32)
    return ids


def _parse_size(name: str) -> int | None:
    match = _SIZE_RE.search(Path(name).name)
    return int(match.group(1)) if match else None


def _write_result_las(out_las, x, y, z, classification, tree_id, semantic, score,
                      epsg: int, scales) -> None:
    """Write the ``results_to_las`` LAS contract (same extra-dim descriptions, so
    ``merge_las`` accepts these files next to ForestFormer3D ones)."""
    header = laspy.LasHeader(point_format=6, version="1.4")
    header.scales = np.asarray(scales, dtype=np.float64)
    header.offsets = np.floor([x.min(), y.min(), z.min()]) if len(x) else np.zeros(3)
    header.add_extra_dim(laspy.ExtraBytesParams(
        name="treeID", type=np.int32, description="ForestFormer3D instance, -1 none"))
    header.add_extra_dim(laspy.ExtraBytesParams(
        name="semantic", type=np.uint8, description="0 ground 1 wood 2 leaf 255 n/a"))
    header.add_extra_dim(laspy.ExtraBytesParams(
        name="score", type=np.float32, description="instance score"))
    header.add_crs(pyproj.CRS.from_epsg(int(epsg)))
    las = laspy.LasData(header)
    las.x, las.y, las.z = x, y, z
    las.classification = np.asarray(classification, dtype=np.uint8)
    las.treeID = np.asarray(tree_id, dtype=np.int32)
    las.semantic = np.asarray(semantic, dtype=np.uint8)
    las.score = np.asarray(score, dtype=np.float32)
    out_las = Path(out_las)
    out_las.parent.mkdir(parents=True, exist_ok=True)
    las.write(str(out_las))


def run_ams3d_tile(las_in, las_out, params: Ams3dParams = Ams3dParams(),
                   buffer_m: float = 10.0, epsg: int = DEFAULT_EPSG,
                   threads: int = 1) -> dict:
    """Segment one LAS tile and write the ForestFormer3D result LAS contract.

    ``las_in`` is either a ``ff3d_geo.split.split_las`` sub-tile (local coordinates,
    origin and size in the name as ``E<x>_N<y>_<size>m``) or any LAS already in
    projected coordinates. Local coordinates are recognised by an origin token whose
    easting exceeds the file's maximum x; they are shifted to UTM on output.

    ``buffer_m > 0`` says the file carries that much context beyond its
    ``0 .. size`` core (``split_las(..., buffer_m=...)``): the buffer points take
    part in the segmentation so edge trees keep their full crowns, but only the
    core points are written, so merged sub-tiles do not duplicate points. It
    needs the ``_<size>m`` token; ``buffer_m = 0`` writes every point.

    Output: LAS 1.4 / point format 6, CRS ``epsg``, ALS ``classification`` kept,
    extra dims ``treeID`` int32 (-1 none), ``semantic`` uint8 (0 ground / 2 leaf /
    255 n/a) and ``score`` float32 (-1). Returns ``{"n_points", "n_trees",
    "n_vegetation", "seconds"}``.
    """
    t0 = time.perf_counter()
    las_in = Path(las_in)
    las = laspy.read(str(las_in))
    x = np.asarray(las.x, dtype=np.float64)
    y = np.asarray(las.y, dtype=np.float64)
    z = np.asarray(las.z, dtype=np.float64)
    classification = np.asarray(las.classification, dtype=np.uint8)
    scales = np.asarray(las.header.scales, dtype=np.float64)
    del las

    try:
        origin = parse_origin(las_in.name)
    except ValueError:
        origin = None
    local = origin is not None and len(x) > 0 and float(x.max()) < origin[0]
    shift = origin if local else (0.0, 0.0)

    if buffer_m > 0:
        size = _parse_size(las_in.name)
        if size is None:
            raise ValueError(
                f"{las_in.name}: buffer_m={buffer_m} needs a _<size>m token in the file "
                "name to know the core extent (pass buffer_m=0 to keep every point)"
            )
        cx0, cy0 = (0.0, 0.0) if local else (origin if origin else (0.0, 0.0))
        core = (x >= cx0) & (x < cx0 + size) & (y >= cy0) & (y < cy0 + size)
    else:
        core = np.ones(len(x), dtype=bool)

    ids = segment_ams3d(np.column_stack([x, y, z]), classification, params, threads=threads)
    semantic = np.full(len(x), SEMANTIC_UNLABELLED, dtype=np.uint8)
    semantic[classification == GROUND_CLASS] = SEMANTIC_GROUND
    semantic[ids >= 0] = SEMANTIC_LEAF

    ids_core = ids[core]
    # Renumber the surviving ids densely so the merged id set stays compact.
    if (ids_core >= 0).any():
        _, dense = np.unique(ids_core[ids_core >= 0], return_inverse=True)
        ids_core = ids_core.copy()
        ids_core[ids_core >= 0] = dense.astype(np.int32)
    _write_result_las(
        las_out, x[core] + shift[0], y[core] + shift[1], z[core], classification[core],
        ids_core, semantic[core], np.full(int(core.sum()), SCORE_NONE, dtype=np.float32),
        epsg, scales,
    )
    return {
        "n_points": int(core.sum()),
        "n_trees": int(np.unique(ids_core[ids_core >= 0]).size),
        "n_vegetation": int((ids >= 0).sum()),
        "seconds": time.perf_counter() - t0,
    }


def params_for(config: str, **overrides) -> Ams3dParams:
    """``CONFIGS[config]`` with field overrides (``params_for("C", min_points=30)``)."""
    if config not in CONFIGS:
        raise ValueError(f"unknown AMS3D config {config!r}; choose from {sorted(CONFIGS)}")
    return replace(CONFIGS[config], **overrides)


# ---- km-tile pipeline ---------------------------------------------------------

def _is_local_tile(las_path: Path) -> bool:
    """True when the file name carries an ``E<x>_N<y>`` origin and its x range lies
    below that easting, i.e. it is a local-coordinate sub-tile (the r12/r13 100 m
    tiles, or a ``split_las`` output) rather than a projected km tile."""
    try:
        origin = parse_origin(las_path.name)
    except ValueError:
        return False
    with laspy.open(str(las_path)) as reader:
        return float(reader.header.maxs[0]) < origin[0]


def _subtile_job(args: tuple) -> dict:
    """Pool worker: segment one sub-tile, write its result LAS and tree GeoPackage."""
    sub_in, sub_out, gpkg_out, params, buffer_m, epsg = args
    from ff3d_geo.trees import trees_to_gpkg

    info = run_ams3d_tile(sub_in, sub_out, params, buffer_m=buffer_m, epsg=epsg)
    trees_to_gpkg(sub_out, gpkg_out)
    info["stem"] = Path(sub_in).stem
    return info


def run_ams3d_pipeline(las_path, out_dir, params: Ams3dParams = Ams3dParams(),
                       buffer_m: float = 10.0, workers: int | None = None,
                       size_m: int = 100, epsg: int = DEFAULT_EPSG,
                       keep_subtiles: bool = False, config_name: str | None = None,
                       log=print) -> dict:
    """``split -> segment sub-tiles in a process pool -> merge -> trees -> masks -> report``.

    A projected km tile is cut into ``size_m`` sub-tiles with ``buffer_m`` of context
    (``split_las``); a local-coordinate tile with an origin token is one sub-tile
    on its own (no buffer available). Each sub-tile result holds only its core
    points, so ``merge_las`` produces exactly the source point set. Writes
    ``<out>/<T>.las``, ``<T>_trees.gpkg``, ``<T>_crowns.gpkg``,
    ``<T>_instance_<cell>.tif``, ``<T>_semantic_<cell>.tif`` and
    ``<T>_report.json/.md`` with ``T`` = the input stem, like the ForestFormer3D
    pipeline. The report JSON carries an extra ``ams3d`` block (config, params,
    workers, per-sub-tile runtimes). Returns the report dict.
    """
    import os
    from multiprocessing import get_context

    from ff3d_geo.merge import merge_las, merge_trees
    from ff3d_geo.raster import las_to_masks
    from ff3d_geo.report import build_report, report_markdown, write_report
    from ff3d_geo.split import split_las

    t_start = time.perf_counter()
    las_path = Path(las_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = las_path.stem
    if workers is None:
        workers = max(1, (os.cpu_count() or 2) // 2)

    sub_in_dir = out_dir / "ams3d_subtiles_in"
    sub_out_dir = out_dir / "ams3d_subtiles_out"
    t0 = time.perf_counter()
    if _is_local_tile(las_path):
        subtiles = [las_path]
        sub_buffer = 0.0
        log(f"{las_path.name}: local-coordinate tile, segmented as one sub-tile (no buffer)")
    else:
        subtiles = split_las(las_path, sub_in_dir, size_m=size_m, buffer_m=buffer_m)
        sub_buffer = buffer_m
        log(f"{las_path.name}: split into {len(subtiles)} sub-tiles of {size_m} m "
            f"(+{buffer_m} m buffer) in {time.perf_counter() - t0:.1f} s")
    if not subtiles:
        raise ValueError(f"{las_path}: split produced no sub-tiles")

    jobs = [
        (sub, sub_out_dir / f"{sub.stem}.las", sub_out_dir / f"{sub.stem}_trees.gpkg",
         params, sub_buffer, epsg)
        for sub in subtiles
    ]
    sub_out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    results: list[dict] = []
    n_workers = min(workers, len(jobs))
    # numpy's and scipy's bundled OpenBLAS each start a thread pool sized to the host
    # (64 threads apiece on carrot's 224 cores) in every spawned worker; 48 workers x
    # 127 pool threads then spend their time in the pool barrier and the km tile
    # crawled at ~13 runnable processes. Spawned children inherit os.environ, and
    # OpenBLAS reads these on import, so pin the pools to one thread per worker.
    for var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(var, "1")
    if n_workers == 1:
        for job in jobs:
            info = _subtile_job(job)
            results.append(info)
            log(f"  {info['stem']}: {info['n_trees']} trees, {info['n_vegetation']} "
                f"vegetation pts, {info['seconds']:.1f} s")
    else:
        with get_context("spawn").Pool(n_workers) as pool:
            for info in pool.imap_unordered(_subtile_job, jobs):
                results.append(info)
                log(f"  {info['stem']}: {info['n_trees']} trees, {info['n_vegetation']} "
                    f"vegetation pts, {info['seconds']:.1f} s")
    seg_seconds = time.perf_counter() - t0
    log(f"segmented {len(jobs)} sub-tiles with {n_workers} workers in {seg_seconds:.1f} s")
    results.sort(key=lambda r: r["stem"])

    sub_las = [job[1] for job in jobs]
    sub_gpkg = [job[2] for job in jobs]
    merged_las = out_dir / f"{stem}.las"
    merged_gpkg = out_dir / f"{stem}_trees.gpkg"
    t0 = time.perf_counter()
    info = merge_las(sub_las, merged_las)
    merge_trees(sub_gpkg, merged_gpkg, info["id_offsets"])
    log(f"merged {info['n_points']} points, {info['n_trees']} trees into {merged_las} "
        f"in {time.perf_counter() - t0:.1f} s")

    t0 = time.perf_counter()
    masks = las_to_masks(merged_las, out_dir)
    log(f"wrote {masks['instance']}, {masks['semantic']}, {masks['crowns']} "
        f"in {time.perf_counter() - t0:.1f} s")

    total = time.perf_counter() - t_start
    report = build_report(merged_las, merged_gpkg, runtime_s=total)
    report["ams3d"] = {
        "config": config_name,
        "params": params.as_dict(),
        "buffer_m": sub_buffer,
        "size_m": size_m,
        "workers": n_workers,
        "n_subtiles": len(jobs),
        "segmentation_s": seg_seconds,
        "subtiles": [
            {k: r[k] for k in ("stem", "n_points", "n_trees", "n_vegetation", "seconds")}
            for r in results
        ],
    }
    write_report(report, out_dir / f"{stem}_report.json", out_dir / f"{stem}_report.md")
    log(report_markdown(report))

    if not keep_subtiles:
        for p in sub_las + sub_gpkg:
            Path(p).unlink(missing_ok=True)
        for p in subtiles:
            if p.parent == sub_in_dir:
                p.unlink(missing_ok=True)
        for d in (sub_in_dir, sub_out_dir):
            if d.is_dir() and not any(d.iterdir()):
                d.rmdir()
    log(f"total wall time {total:.1f} s")
    return report
