"""Tests for ff3d_geo.baseline.chm_local_maxima.

Needs laspy and numpy, so it starts with ``pytest.importorskip`` for each
(see tests/test_geo_origin.py's module docstring) so the system-python test
run (no geo libs installed) skips it instead of failing.

The two-cone synthetic-forest tile builder lives in tests/geo_fixtures.py
(plain functions, not pytest fixtures registered in conftest.py) and is
imported here after the importorskip calls above.
"""

import math

import pytest

laspy = pytest.importorskip("laspy")
np = pytest.importorskip("numpy")

from ff3d_geo.baseline import chm_local_maxima  # noqa: E402
from geo_fixtures import TWO_CONES, write_two_cone_las  # noqa: E402


@pytest.fixture
def two_cone_spec():
    """Apexes (x, y, height, crown radius) in local coordinates."""
    return {"cones": TWO_CONES}


@pytest.fixture
def two_cone_las(tmp_path):
    return write_two_cone_las(tmp_path / "cones_E381300_N5828300.las", with_predictions=False)


@pytest.fixture
def ground_only_las(tmp_path):
    """Flat ground tile with no vegetation classes at all."""
    g = np.arange(0.0, 20.0 + 1e-9, 0.5)
    gx, gy = np.meshgrid(g, g)
    x, y, z = gx.ravel(), gy.ravel(), np.zeros(gx.size)
    cls = np.full(gx.size, 2, np.uint8)

    header = laspy.LasHeader(point_format=1, version="1.2")
    header.scales = np.array([0.01, 0.01, 0.01])
    header.offsets = np.floor([x.min(), y.min(), z.min()])
    las = laspy.LasData(header)
    las.x, las.y, las.z = x, y, z
    las.classification = cls
    path = tmp_path / "ground_only.las"
    las.write(str(path))
    return path


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
