"""Tests for ff3d_geo.baseline.chm_local_maxima.

``two_cone_las``/``two_cone_spec`` are defined locally here rather than in
``tests/conftest.py`` because, at the time this module was written,
Task 4 (ff3d_geo/grid.py + ff3d_geo/trees.py) had not yet appended the
shared synthetic-tile fixtures to conftest.py. The geometry mirrors
task-4-brief.md's ``TWO_CONES``/``two_cone_points`` exactly (two cone-shaped
trees on flat ground in a 40 m x 40 m tile, apexes at 1 m grid-cell centers)
so Task 4 can drop these in favor of the shared conftest fixtures once it
lands, without changing any assertions here.
"""

import math

import pytest

laspy = pytest.importorskip("laspy")
np = pytest.importorskip("numpy")

from ff3d_geo.baseline import chm_local_maxima  # noqa: E402

# Apex (x, y, height, crown radius); apexes sit at cell centers of a 1 m grid
# so CHM local maxima land exactly on them.
TWO_CONES = [(10.5, 10.5, 10.0, 3.0), (28.5, 26.5, 15.0, 4.0)]


def _two_cone_points():
    """x, y, z, classification arrays for the synthetic tile (ground + two cones)."""
    g = np.arange(0.0, 40.0 + 1e-9, 0.5)
    gx, gy = np.meshgrid(g, g)
    xs, ys, zs = [gx.ravel()], [gy.ravel()], [np.zeros(gx.size)]
    cls = [np.full(gx.size, 2, np.uint8)]
    for ax, ay, h, r in TWO_CONES:
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
    return np.concatenate(xs), np.concatenate(ys), np.concatenate(zs), np.concatenate(cls)


def _write_las(path, x, y, z, classification):
    header = laspy.LasHeader(point_format=1, version="1.2")
    header.scales = np.array([0.01, 0.01, 0.01])
    header.offsets = np.floor([x.min(), y.min(), z.min()])
    las = laspy.LasData(header)
    las.x, las.y, las.z = x, y, z
    las.classification = classification
    las.write(str(path))
    return path


@pytest.fixture
def two_cone_spec():
    """Apexes (x, y, height, crown radius) in local coordinates."""
    return {"cones": TWO_CONES}


@pytest.fixture
def two_cone_las(tmp_path):
    x, y, z, cls = _two_cone_points()
    return _write_las(tmp_path / "cones_E381300_N5828300.las", x, y, z, cls)


@pytest.fixture
def ground_only_las(tmp_path):
    """Flat ground tile with no vegetation classes at all."""
    g = np.arange(0.0, 20.0 + 1e-9, 0.5)
    gx, gy = np.meshgrid(g, g)
    x, y, z = gx.ravel(), gy.ravel(), np.zeros(gx.size)
    cls = np.full(gx.size, 2, np.uint8)
    return _write_las(tmp_path / "ground_only.las", x, y, z, cls)


def test_two_cone_tile_gives_two_maxima_at_the_apexes(two_cone_las, two_cone_spec):
    maxima = chm_local_maxima(two_cone_las)

    assert len(maxima) == 2
    for (x, y, height), (ax, ay, h, _r) in zip(sorted(maxima), two_cone_spec["cones"]):
        assert math.hypot(x - ax, y - ay) <= 1.0
        assert abs(height - h) <= 0.5


def test_min_height_filters_short_maxima(two_cone_las):
    assert len(chm_local_maxima(two_cone_las, min_height_m=12.0)) == 1


def test_no_vegetation_gives_no_maxima(ground_only_las):
    assert chm_local_maxima(ground_only_las) == []
