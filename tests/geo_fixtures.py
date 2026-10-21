"""Synthetic two-cone forest tile shared by geo tests (grid.py/trees.py, baseline.py, ...).

Plain functions only (no ``@pytest.fixture`` here): this module is imported directly
by test files (``from geo_fixtures import ...``) after their own ``pytest.importorskip``
calls for laspy/pyproj/etc., rather than being registered as a pytest plugin, so the
system-python test run (no geo libs installed) never has to import it.
"""

import numpy as np

# Two cone-shaped trees on flat ground (z = 0) in a 40 m x 40 m tile of local
# coordinates. Apex (x, y, height, crown radius); apexes sit at cell centers of
# a 1 m grid so CHM local maxima land exactly on them.
TWO_CONES = [(10.5, 10.5, 10.0, 3.0), (28.5, 26.5, 15.0, 4.0)]
TILE_ORIGIN = (381300.0, 5828300.0)  # UTM 33N lower-left corner used for the result LAS


def two_cone_points():
    """Return x, y, z, classification, tree_id, semantic arrays for the synthetic tile."""
    g = np.arange(0.0, 40.0 + 1e-9, 0.5)
    gx, gy = np.meshgrid(g, g)
    xs, ys, zs = [gx.ravel()], [gy.ravel()], [np.zeros(gx.size)]
    cls, tid, sem = [np.full(gx.size, 2, np.uint8)], [np.full(gx.size, -1, np.int32)], [np.zeros(gx.size, np.uint8)]
    for i, (ax, ay, h, r) in enumerate(TWO_CONES, start=1):
        c = np.arange(-r, r + 1e-9, 0.25)
        cx, cy = np.meshgrid(c, c)
        dist = np.hypot(cx, cy).ravel()
        keep = dist <= r
        crown_x, crown_y = (ax + cx.ravel())[keep], (ay + cy.ravel())[keep]
        crown_z = h * (1.0 - dist[keep] / r)
        stem_z = np.arange(0.25, h, 0.25)
        xs += [crown_x, np.full(stem_z.size, ax)]
        ys += [crown_y, np.full(stem_z.size, ay)]
        zs += [crown_z, stem_z]
        cls += [np.full(crown_z.size, 5, np.uint8), np.full(stem_z.size, 4, np.uint8)]
        tid += [np.full(crown_z.size, i, np.int32), np.full(stem_z.size, i, np.int32)]
        sem += [np.full(crown_z.size, 2, np.uint8), np.full(stem_z.size, 1, np.uint8)]
    return (
        np.concatenate(xs), np.concatenate(ys), np.concatenate(zs),
        np.concatenate(cls), np.concatenate(tid), np.concatenate(sem),
    )


def write_two_cone_las(path, with_predictions=False):
    """Write the synthetic tile. Without predictions: LAS 1.2 / pf1, local coordinates,
    like the Tegel input. With predictions: LAS 1.4 / pf6 in EPSG:25833 with the
    treeID / semantic / score extra dims that results_to_las writes."""
    import laspy
    import pyproj

    x, y, z, cls, tid, sem = two_cone_points()
    if with_predictions:
        header = laspy.LasHeader(point_format=6, version="1.4")
        header.add_extra_dim(laspy.ExtraBytesParams(name="treeID", type=np.int32))
        header.add_extra_dim(laspy.ExtraBytesParams(name="semantic", type=np.uint8))
        header.add_extra_dim(laspy.ExtraBytesParams(name="score", type=np.float32))
        header.add_crs(pyproj.CRS.from_epsg(25833))
        x = x + TILE_ORIGIN[0]
        y = y + TILE_ORIGIN[1]
    else:
        header = laspy.LasHeader(point_format=1, version="1.2")
    header.scales = np.array([0.01, 0.01, 0.01])
    header.offsets = np.floor([x.min(), y.min(), z.min()])
    las = laspy.LasData(header)
    las.x, las.y, las.z = x, y, z
    las.classification = cls
    if with_predictions:
        las.treeID = tid
        las.semantic = sem
        las.score = np.where(tid >= 0, 0.9, 0.0).astype(np.float32)
    las.write(str(path))
    return path


def write_grid_las(path, xyz, classification):
    """Write ``xyz`` (N, 3) + ``classification`` as LAS 1.4 / point format 6.

    Used by the split tests and the CLI's ``split`` round trip: the source of a
    ``split_las`` call is a plain ALS tile in absolute (UTM) coordinates, so the
    header offsets are placed near the data -- LAS X/Y/Z are int32, and scale
    0.001 with a zero offset overflows for UTM-scale (~5.8e6) northings.

    ``xyz`` is quantized IN PLACE to what LAS will actually store, so a caller can
    compare against the same values ``split_las`` sees. The quantization truncates
    rather than rounds to nearest, so a point already inside a 100 m grid cell
    cannot get nudged onto the cell boundary and land in a neighboring (possibly
    sparse, ``min_points``-filtered) sub-tile instead.
    """
    import laspy

    header = laspy.LasHeader(point_format=6, version="1.4")
    scales = np.array([0.001, 0.001, 0.001])
    offsets = np.floor(xyz.min(axis=0))
    header.scales = scales
    header.offsets = offsets
    xyz[:] = offsets + np.floor((xyz - offsets) / scales) * scales
    las = laspy.LasData(header)
    las.x, las.y, las.z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    las.classification = classification
    las.write(str(path))
    return path


def write_result_las(path, x, y, z, tree_id, semantic, score=None, offsets=None):
    """Write a *result* LAS exactly as ``ff3d_geo.convert.results_to_las`` does.

    LAS 1.4 / point format 6 in EPSG:25833 with the three extra dims ``treeID``
    (int32, -1 = no tree), ``semantic`` (uint8, 0 ground / 1 wood / 2 leaf / 255
    nodata) and ``score`` (float32) -- descriptions included verbatim, because
    laspy compares point formats dimension by dimension (description included)
    when appending points, so a fixture without them would not catch code that
    rebuilds the extra dims from names only.

    ``score`` defaults to 0.9 where ``tree_id >= 0`` and 0.0 elsewhere; ``offsets``
    defaults to the floor of the per-axis minimum (LAS X/Y/Z are int32, so a zero
    offset overflows at UTM scale).
    """
    import laspy
    import pyproj

    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    header = laspy.LasHeader(point_format=6, version="1.4")
    header.scales = np.array([0.001, 0.001, 0.001])
    header.offsets = (np.floor([x.min(), y.min(), z.min()]) if offsets is None
                      else np.asarray(offsets, dtype=np.float64))
    header.add_extra_dim(laspy.ExtraBytesParams(
        name="treeID", type=np.int32, description="ForestFormer3D instance, -1 none"))
    header.add_extra_dim(laspy.ExtraBytesParams(
        name="semantic", type=np.uint8, description="0 ground 1 wood 2 leaf 255 n/a"))
    header.add_extra_dim(laspy.ExtraBytesParams(
        name="score", type=np.float32, description="instance score"))
    header.add_crs(pyproj.CRS.from_epsg(25833))
    las = laspy.LasData(header)
    las.x, las.y, las.z = x, y, z
    las.treeID = np.asarray(tree_id, dtype=np.int32)
    las.semantic = np.asarray(semantic, dtype=np.uint8)
    las.score = (np.where(np.asarray(tree_id) >= 0, 0.9, 0.0).astype(np.float32)
                 if score is None else np.asarray(score, dtype=np.float32))
    las.write(str(path))
    return path
