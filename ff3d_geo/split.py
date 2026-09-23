"""Cut a 1 km ALS LAS tile into grid-aligned 100 m sub-tiles in local coordinates.

Each sub-tile is written as LAS 1.4 point format 6 with ``x``/``y`` shifted so the
sub-tile's lower-left corner is the origin (local coordinates run ``0..size_m``);
``z`` is left unchanged. The sub-tile's absolute origin (UTM easting/northing) is
encoded in its file name via the ``E<x>_N<y>`` token that ``ff3d_geo.origin.parse_origin``
reads back.

The source LAS is read in bounded chunks (``laspy.open(...).chunk_iterator``) and
bucketed into per-sub-tile LAS writers kept open in a dict, so a multi-hundred-
million-point 1 km tile never needs to be held in memory all at once.
"""

from __future__ import annotations

import math
import re
from pathlib import Path

import laspy
import numpy as np

# Point dimensions copied verbatim (no scale/offset conversion needed) when present
# in the source point format.
_COPY_DIMS = ("intensity", "return_number", "number_of_returns")

# Chunk size for the source LAS reader; bounds peak memory independent of tile size.
_CHUNK_POINTS = 2_000_000

_TRAILING_1BE_RE = re.compile(r"_1_be$")


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


def _make_writer(out_dir: Path, prefix: str, origin: tuple[int, int], size_m: int) -> tuple[Path, "laspy.LasWriter"]:
    tmp_path = out_dir / f".{prefix}_E{origin[0]}_N{origin[1]}_{size_m}m.las.tmp"
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
) -> list[Path]:
    """Split ``las_path`` into ``size_m``-metre local-coordinate sub-tiles under ``out_dir``.

    Returns the sorted list of written sub-tile paths; sub-tiles with fewer than
    ``min_points`` points are not written. Reads the source LAS in bounded chunks
    so the whole tile is never loaded into memory at once.

    Every source point must land in exactly one sub-tile: the grid is derived from
    the header's ``mins``/``maxs``, which laspy does NOT re-derive from the points,
    so a stale bound would leave points outside the grid failing every cell mask and
    vanishing silently. The per-sub-tile counts (including the sparse ones dropped by
    ``min_points``) are therefore summed and compared with ``header.point_count``, and
    a mismatch raises ``ValueError`` rather than writing a quietly incomplete split.
    """
    las_path = Path(las_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if prefix is None:
        prefix = _default_prefix(las_path.stem)

    tmp_paths: dict[tuple[int, int], Path] = {}
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

            source_dims = {d.name for d in reader.header.point_format.dimensions}
            copy_dims = [d for d in _COPY_DIMS if d in source_dims]

            writers: dict[tuple[int, int], "laspy.LasWriter"] = {}
            counts: dict[tuple[int, int], int] = {origin: 0 for origin in origins}

            try:
                for origin in origins:
                    tmp_path, writer = _make_writer(out_dir, prefix, origin, size_m)
                    tmp_paths[origin] = tmp_path
                    writers[origin] = writer

                for points in reader.chunk_iterator(_CHUNK_POINTS):
                    x = np.asarray(points.x)
                    y = np.asarray(points.y)
                    ix = np.floor((x - x0_grid) / size_m).astype(np.int64)
                    iy = np.floor((y - y0_grid) / size_m).astype(np.int64)
                    ex = x0_grid + ix * size_m
                    ny = y0_grid + iy * size_m

                    for origin in origins:
                        mask = (ex == origin[0]) & (ny == origin[1])
                        n_sel = int(mask.sum())
                        if n_sel == 0:
                            continue
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
                        counts[origin] += n_sel
            finally:
                for writer in writers.values():
                    writer.close()

            # Before min_points drops anything: the grid comes from the header's
            # mins/maxs, so a stale bound would silently lose the points outside it.
            written_points = sum(counts.values())
            if written_points != source_count:
                raise ValueError(
                    f"split_las: {written_points} of {las_path}'s {source_count} points "
                    f"landed in a sub-tile; the missing points lie outside the grid "
                    f"derived from the header bounds {tuple(mins)}..{tuple(maxs)} "
                    "(a stale LAS header does not get re-derived on read)"
                )

        for origin in origins:
            tmp_path = tmp_paths[origin]
            final_path = out_dir / f"{prefix}_E{origin[0]}_N{origin[1]}_{size_m}m.las"
            if counts[origin] >= min_points:
                tmp_path.rename(final_path)
                written.append(final_path)
            else:
                tmp_path.unlink(missing_ok=True)
        finished = True
    finally:
        if not finished:
            # Leave no .<stem>.las.tmp behind on any failure path.
            for tmp_path in tmp_paths.values():
                tmp_path.unlink(missing_ok=True)

    return sorted(written)
