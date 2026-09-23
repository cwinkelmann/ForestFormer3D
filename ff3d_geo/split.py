"""Cut a 1 km ALS LAS tile into grid-aligned 100 m sub-tiles in local coordinates.

Each sub-tile is written as LAS 1.4 point format 6 with ``x``/``y`` shifted so the
sub-tile's lower-left corner is the origin (local coordinates run ``0..size_m``);
``z`` is left unchanged. The sub-tile's absolute origin (UTM easting/northing) is
encoded in its file name via the ``E<x>_N<y>`` token that ``ff3d_geo.origin.parse_origin``
reads back.

The source LAS is read in bounded chunks (``laspy.open(...).chunk_iterator``) and
bucketed into per-sub-tile LAS writers kept open in a dict, so a multi-hundred-
million-point 1 km tile never needs to be held in memory all at once.

Alongside each sub-tile a point identity sidecar ``<stem>_ident.npy`` (``IDENT_DTYPE``)
records, per written point and in the sub-tile's point order, which source file the
point came from and its 0-based position in that file. ``<out>/split_manifest.json``
lists the sources and the written sub-tiles. Together they let ``ff3d_geo.stitch``
map a per-sub-tile prediction back onto the source km tiles and recognise that a halo
point of one sub-tile and a core point of its neighbour are the same physical point.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Sequence
from pathlib import Path

import laspy
import numpy as np

# Point dimensions copied verbatim (no scale/offset conversion needed) when present
# in the source point format.
_COPY_DIMS = ("intensity", "return_number", "number_of_returns")

# Chunk size for the source LAS reader; bounds peak memory independent of tile size.
_CHUNK_POINTS = 2_000_000

_TRAILING_1BE_RE = re.compile(r"_1_be$")

#: Identity of a written sub-tile point: ``tile`` indexes the manifest's ``sources``
#: (0 = the primary km tile, ``1 + i`` = ``neighbours[i]``), ``index`` is the point's
#: 0-based position in that source file. Re-exported by ``ff3d_geo.stitch``.
IDENT_DTYPE = np.dtype([("tile", "<u2"), ("index", "<u4")])


def _default_prefix(stem: str) -> str:
    """Strip a trailing ``_1_be`` from a source LAS stem (Berlin ALS naming)."""
    return _TRAILING_1BE_RE.sub("", stem)


def subtile_origins(
    mins: tuple[float, float, float],
    maxs: tuple[float, float, float],
    size_m: int,
) -> list[tuple[int, int]]:
    """Grid-aligned lower-left ``(easting, northing)`` corners covering ``mins``..``maxs``.

    The grid starts at ``floor(min / size_m) * size_m`` on each axis and steps by
    ``size_m`` up to and including the cell containing ``maxs``. Returned in x-major,
    y-minor order (all y values for one x before moving to the next x).
    """
    x0 = int(math.floor(mins[0] / size_m) * size_m)
    x1 = int(math.floor(maxs[0] / size_m) * size_m)
    y0 = int(math.floor(mins[1] / size_m) * size_m)
    y1 = int(math.floor(maxs[1] / size_m) * size_m)

    origins = []
    x = x0
    while x <= x1:
        y = y0
        while y <= y1:
            origins.append((x, y))
            y += size_m
        x += size_m
    return origins


def _subtile_stem(prefix: str, origin: tuple[int, int], size_m: int) -> str:
    return f"{prefix}_E{origin[0]}_N{origin[1]}_{size_m}m"


def _make_writer(out_dir: Path, prefix: str, origin: tuple[int, int], size_m: int) -> tuple[Path, "laspy.LasWriter"]:
    tmp_path = out_dir / f".{_subtile_stem(prefix, origin, size_m)}.las.tmp"
    header = laspy.LasHeader(point_format=6, version="1.4")
    header.scales = np.array([0.001, 0.001, 0.001])
    header.offsets = np.array([0.0, 0.0, 0.0])
    writer = laspy.open(str(tmp_path), mode="w", header=header)
    return tmp_path, writer


def split_las(
    las_path,
    out_dir,
    size_m: int = 100,
    min_points: int = 1000,
    prefix: str | None = None,
    buffer_m: float = 0.0,
    neighbours: Sequence = (),
) -> list[Path]:
    """Split ``las_path`` into ``size_m``-metre local-coordinate sub-tiles under ``out_dir``.

    Returns the sorted list of written sub-tile paths; sub-tiles with fewer than
    ``min_points`` CORE points are not written. Reads the source LAS in bounded chunks
    so the whole tile is never loaded into memory at once.

    With ``buffer_m > 0`` every sub-tile also receives the points that lie within
    ``buffer_m`` outside its ``size_m`` core, so the local coordinates run
    ``-buffer_m .. size_m + buffer_m`` and a point near a grid line is written to each
    sub-tile whose buffered extent contains it (up to 4, 9 at a grid corner). Callers
    that segment the sub-tiles independently use the buffer as context for the edge
    trees: ``ff3d_geo.ams3d`` crops its result back to the ``0 .. size_m`` core before
    merging, ``ff3d_geo.stitch`` uses the halo as the overlap that lets it recognise
    one tree across two sub-tiles. Either way no point is counted twice; the
    conservation check below is on the core assignment only, which is still exactly
    one sub-tile per point.

    ``neighbours`` are further km tiles (the 8 neighbours of ``las_path``) whose points
    fill the halo of the sub-tiles on the km-tile border, so those borders behave like
    internal ones. A neighbour is read in the same bounded chunks and selected by the
    same expanded-box test, so a neighbour that does not overlap the primary's grid
    contributes nothing when ``buffer_m == 0``.

    Every source point of the PRIMARY tile must land in exactly one sub-tile core: the
    grid is derived from the header's ``mins``/``maxs``, which laspy does NOT re-derive
    from the points, so a stale bound would leave points outside the grid failing every
    cell mask and vanishing silently. The per-sub-tile core counts (including the sparse
    ones dropped by ``min_points``) are therefore summed and compared with
    ``header.point_count``, and a mismatch raises ``ValueError`` rather than writing a
    quietly incomplete split.

    Writes per kept sub-tile ``<out_dir>/<stem>_ident.npy`` (``IDENT_DTYPE``, aligned
    with the sub-tile's point order) and, last of all, ``<out_dir>/split_manifest.json``;
    the manifest's presence means the split ran to completion.
    """
    las_path = Path(las_path)
    out_dir = Path(out_dir)
    neighbour_paths = [Path(p) for p in neighbours]
    out_dir.mkdir(parents=True, exist_ok=True)

    buffer_m = float(buffer_m)
    if buffer_m < 0:
        raise ValueError(f"split_las: buffer_m must be >= 0, got {buffer_m}")

    if prefix is None:
        prefix = _default_prefix(las_path.stem)

    tmp_paths: dict[tuple[int, int], Path] = {}
    ident_tmp_paths: dict[tuple[int, int], Path] = {}
    written: list[Path] = []
    finished = False
    try:
        with laspy.open(str(las_path)) as reader:
            mins = reader.header.mins
            maxs = reader.header.maxs
            source_count = int(reader.header.point_count)
            origins = subtile_origins(mins, maxs, size_m)
            x0_grid = int(math.floor(mins[0] / size_m) * size_m)
            y0_grid = int(math.floor(mins[1] / size_m) * size_m)
            # Outer bounds of the grid, used to skip whole neighbour chunks.
            x1_grid = max(o[0] for o in origins) + size_m
            y1_grid = max(o[1] for o in origins) + size_m

            writers: dict[tuple[int, int], "laspy.LasWriter"] = {}
            ident_files: dict[tuple[int, int], "object"] = {}
            counts_core: dict[tuple[int, int], int] = {origin: 0 for origin in origins}
            counts_total: dict[tuple[int, int], int] = {origin: 0 for origin in origins}

            def _feed(points, chunk_start: int, tile_index: int) -> None:
                """Bucket one chunk of ``tile_index``'s points into the open writers."""
                x = np.asarray(points.x)
                y = np.asarray(points.y)
                if tile_index == 0:
                    # Exact-cell membership: this is the assignment the conservation
                    # check below verifies, and what ``min_points`` counts.
                    ix = np.floor((x - x0_grid) / size_m).astype(np.int64)
                    iy = np.floor((y - y0_grid) / size_m).astype(np.int64)
                    ex = x0_grid + ix * size_m
                    ny = y0_grid + iy * size_m
                elif (x.size == 0
                      or x.max() < x0_grid - buffer_m or x.min() >= x1_grid + buffer_m
                      or y.max() < y0_grid - buffer_m or y.min() >= y1_grid + buffer_m):
                    # Nothing in this neighbour chunk can reach any sub-tile's halo.
                    return

                for origin in origins:
                    if tile_index == 0:
                        core = (ex == origin[0]) & (ny == origin[1])
                        counts_core[origin] += int(core.sum())
                    else:
                        core = None
                    if buffer_m > 0 or core is None:
                        mask = (
                            (x >= origin[0] - buffer_m) & (x < origin[0] + size_m + buffer_m)
                            & (y >= origin[1] - buffer_m) & (y < origin[1] + size_m + buffer_m)
                        )
                    else:
                        mask = core
                    sel = np.flatnonzero(mask)
                    n_sel = sel.size
                    if n_sel == 0:
                        continue
                    counts_total[origin] += n_sel
                    sub = points[mask]
                    writer = writers[origin]

                    new_rec = laspy.ScaleAwarePointRecord.zeros(
                        n_sel,
                        point_format=writer.header.point_format,
                        scales=writer.header.scales,
                        offsets=writer.header.offsets,
                    )
                    new_rec.x = np.asarray(sub.x) - origin[0]
                    new_rec.y = np.asarray(sub.y) - origin[1]
                    new_rec.z = np.asarray(sub.z)
                    new_rec.classification = np.asarray(sub.classification)
                    for dim in copy_dims:
                        setattr(new_rec, dim, np.asarray(getattr(sub, dim)))

                    writer.write_points(new_rec)

                    # The sidecar is appended raw and loaded back once per sub-tile at
                    # the end, so peak memory stays independent of the tile's size.
                    ident = np.empty(n_sel, IDENT_DTYPE)
                    ident["tile"] = tile_index
                    ident["index"] = chunk_start + sel
                    ident_files[origin].write(ident.tobytes())

            try:
                for origin in origins:
                    tmp_path, writer = _make_writer(out_dir, prefix, origin, size_m)
                    tmp_paths[origin] = tmp_path
                    writers[origin] = writer
                    ident_tmp = out_dir / f".{_subtile_stem(prefix, origin, size_m)}_ident.tmp"
                    ident_tmp_paths[origin] = ident_tmp
                    ident_files[origin] = ident_tmp.open("wb")

                source_dims = {d.name for d in reader.header.point_format.dimensions}
                copy_dims = [d for d in _COPY_DIMS if d in source_dims]
                chunk_start = 0
                for points in reader.chunk_iterator(_CHUNK_POINTS):
                    _feed(points, chunk_start, 0)
                    chunk_start += len(points)

                for i, neighbour_path in enumerate(neighbour_paths):
                    with laspy.open(str(neighbour_path)) as nb_reader:
                        nb_dims = {d.name for d in nb_reader.header.point_format.dimensions}
                        copy_dims = [d for d in _COPY_DIMS if d in nb_dims]
                        chunk_start = 0
                        for points in nb_reader.chunk_iterator(_CHUNK_POINTS):
                            _feed(points, chunk_start, 1 + i)
                            chunk_start += len(points)
            finally:
                for writer in writers.values():
                    writer.close()
                for handle in ident_files.values():
                    handle.close()

            # Before min_points drops anything: the grid comes from the header's
            # mins/maxs, so a stale bound would silently lose the points outside it.
            written_points = sum(counts_core.values())
            if written_points != source_count:
                raise ValueError(
                    f"split_las: {written_points} of {las_path}'s {source_count} points "
                    f"landed in a sub-tile; the missing points lie outside the grid "
                    f"derived from the header bounds {tuple(mins)}..{tuple(maxs)} "
                    "(a stale LAS header does not get re-derived on read)"
                )

        subtiles = []
        for origin in origins:
            tmp_path = tmp_paths[origin]
            ident_tmp = ident_tmp_paths[origin]
            stem = _subtile_stem(prefix, origin, size_m)
            if counts_core[origin] >= min_points:
                tmp_path.rename(out_dir / f"{stem}.las")
                ident = np.fromfile(ident_tmp, dtype=IDENT_DTYPE)
                np.save(out_dir / f"{stem}_ident.npy", ident)
                ident_tmp.unlink(missing_ok=True)
                written.append(out_dir / f"{stem}.las")
                subtiles.append({
                    "stem": stem,
                    "origin": [int(origin[0]), int(origin[1])],
                    "source": 0,
                    "n_points": int(counts_total[origin]),
                    "n_core": int(counts_core[origin]),
                })
            else:
                tmp_path.unlink(missing_ok=True)
                ident_tmp.unlink(missing_ok=True)

        written = sorted(written)
        order = {s["stem"]: s for s in subtiles}
        manifest = {
            "size_m": int(size_m),
            "buffer_m": buffer_m,
            "prefix": prefix,
            "sources": _source_records([las_path, *neighbour_paths]),
            "subtiles": [order[p.stem] for p in written],
        }
        (out_dir / "split_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        finished = True
    finally:
        if not finished:
            # Leave no .<stem>.las.tmp / .<stem>_ident.tmp behind on any failure path.
            for tmp_path in (*tmp_paths.values(), *ident_tmp_paths.values()):
                tmp_path.unlink(missing_ok=True)

    return written


def _source_records(paths: Sequence[Path]) -> list[dict]:
    """``{"key", "path", "n_points"}`` per source LAS, in ``tile`` index order."""
    records = []
    for path in paths:
        with laspy.open(str(path)) as reader:
            n_points = int(reader.header.point_count)
        records.append({
            "key": _default_prefix(Path(path).stem),
            "path": str(Path(path).resolve()),
            "n_points": n_points,
        })
    return records
