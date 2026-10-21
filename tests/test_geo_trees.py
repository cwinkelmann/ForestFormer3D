"""Tests for ff3d_geo.grid and ff3d_geo.trees.trees_to_gpkg.

Needs laspy, shapely, pyproj and geopandas, so it starts with
``pytest.importorskip`` for each (see tests/test_geo_origin.py's module
docstring) so the system-python test run (no geo libs installed) skips it
instead of failing.

The two-cone synthetic-forest tile builder lives in tests/geo_fixtures.py
(plain functions, not pytest fixtures registered in conftest.py) and is
imported here after the importorskip calls above.
"""

import math

import pytest

laspy = pytest.importorskip("laspy")
pytest.importorskip("shapely")
pytest.importorskip("pyproj")
gpd = pytest.importorskip("geopandas")

import numpy as np  # noqa: E402

from ff3d_geo.grid import GridExtent, grid_extent, grid_reduce, nearest_fill  # noqa: E402
from ff3d_geo.trees import trees_to_gpkg  # noqa: E402
from geo_fixtures import TILE_ORIGIN, TWO_CONES, write_two_cone_las  # noqa: E402


@pytest.fixture
def two_cone_spec():
    """Apexes (x, y, height, crown radius) in local coordinates and the UTM origin."""
    return {"cones": TWO_CONES, "origin": TILE_ORIGIN}


@pytest.fixture
def two_cone_result_las(tmp_path):
    return write_two_cone_las(tmp_path / "cones_result.las", with_predictions=True)


def test_grid_reduce_min_and_nearest_fill():
    x = np.array([0.2, 0.7, 2.5])
    y = np.array([0.1, 0.9, 0.5])
    z = np.array([5.0, 3.0, 9.0])
    extent = grid_extent(x, y, 1.0)
    assert extent == GridExtent(0.0, 0.0, 1.0, 3, 1)
    grid = grid_reduce(extent, x, y, z, "min")
    assert grid[0, 0] == 3.0 and np.isnan(grid[0, 1]) and grid[0, 2] == 9.0
    filled = nearest_fill(grid)
    assert filled[0, 1] == 6.0  # mean of the two touching cells


def test_two_cones_give_two_rows_with_heights_and_crown_areas(two_cone_result_las, two_cone_spec, tmp_path):
    gpkg = tmp_path / "trees.gpkg"
    origin = two_cone_spec["origin"]

    n = trees_to_gpkg(two_cone_result_las, gpkg)

    assert n == 2
    gdf = gpd.read_file(gpkg, layer="trees").sort_values("tree_id")
    assert list(gdf.columns) == ["tree_id", "x", "y", "top_z", "height", "crown_area_m2",
                                 "n_points", "mean_score", "geometry"]
    assert gdf.crs.to_epsg() == 25833
    for row, (ax, ay, h, r) in zip(gdf.itertuples(), two_cone_spec["cones"]):
        assert abs(row.height - h) <= 0.5
        assert abs(row.x - (ax + origin[0])) <= 0.5
        assert abs(row.y - (ay + origin[1])) <= 0.5
        assert abs(row.crown_area_m2 - math.pi * r * r) <= 0.2 * math.pi * r * r
        assert abs(row.top_z - h) <= 0.01
        assert abs(row.mean_score - 0.9) <= 1e-6
        assert row.geometry.x == row.x and row.geometry.y == row.y
    assert list(gdf.tree_id) == [1, 2]


def test_no_trees_writes_empty_layer(two_cone_result_las, tmp_path):
    las = laspy.read(str(two_cone_result_las))
    las.treeID[:] = -1
    empty = tmp_path / "empty.las"
    las.write(str(empty))
    assert trees_to_gpkg(empty, tmp_path / "empty.gpkg") == 0
    assert len(gpd.read_file(tmp_path / "empty.gpkg", layer="trees")) == 0
