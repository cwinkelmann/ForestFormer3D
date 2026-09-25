"""Raster masks + crown polygons from a georeferenced result LAS (ff3d_geo.raster)."""

import numpy as np
import pytest

laspy = pytest.importorskip("laspy")
pytest.importorskip("shapely")
pytest.importorskip("pyproj")
gpd = pytest.importorskip("geopandas")
pytest.importorskip("pyogrio")
rasterio = pytest.importorskip("rasterio")

from ff3d_geo.cli import main
from ff3d_geo.raster import _cell_tag, las_to_masks

from geo_fixtures import write_result_las

# The synthetic tile: a 10 m x 10 m square of UTM 33N whose header mins/maxs snap to
# exactly 20 x 20 cells of 0.5 m. Row 0 is the NORTHERNMOST row, so
# row = floor((Y1 - y) / 0.5) with Y1 = 5828010, col = floor((x - 381000) / 0.5).
X0, Y0 = 381000.0, 5828000.0
X1, Y1 = 381010.0, 5828010.0
CELL = 0.5
ROWS = COLS = 20

GROUND, WOOD, LEAF, NODATA = 0, 1, 2, 255


def _cell_xy(row, col, dx=0.1, dy=0.1):
    """A point inside cell (row, col), offset from its NW corner."""
    return X0 + col * CELL + dx, Y1 - row * CELL - dy


def _synthetic_result_las(path):
    """Two ground quadrants, four trees and hand-placed semantic-vote cells.

    Trees: 0 (3 points, one cell, row 7 col 4), 1 (a SINGLE point -> hull_is_point,
    row 13 col 10), 2 (row 2 col 15 plus a cell of its own at row 4 col 12) and 3
    (row 2 col 15, higher than tree 2 there, so it wins that cell).
    """
    xs, ys, zs, tids, sems = [], [], [], [], []

    def add(x, y, z, tid, sem):
        xs.append(x); ys.append(y); zs.append(z); tids.append(tid); sems.append(sem)

    # Ground: a 0.25 m lattice over the SW quadrant only (cols 0-9, rows 10-19), so the
    # rest of the tile has no voted points at all unless a point is placed there.
    g = np.arange(0.0, 5.0 - 1e-9, 0.25)
    for gx in g:
        for gy in g:
            add(X0 + gx, Y0 + gy, 0.0, -1, GROUND)

    # NE anchor so the header maxs snap outward to exactly (X1, Y1). semantic 255 =
    # no vote, so cell (0, 19) must come out as 255 in the semantic raster.
    add(X1 - 0.1, Y1 - 0.1, 1.0, -1, NODATA)

    # Semantic vote cells (all with treeID -1, so they never touch the instance raster):
    for x, y, sem in [
        (*_cell_xy(1, 1), WOOD), (*_cell_xy(1, 1, 0.2, 0.2), LEAF),      # tie -> LEAF
        (*_cell_xy(1, 3), GROUND), (*_cell_xy(1, 3, 0.2, 0.2), WOOD),    # tie -> WOOD
        (*_cell_xy(1, 5), GROUND), (*_cell_xy(1, 5, 0.2, 0.2), GROUND),
        (*_cell_xy(1, 5, 0.3, 0.3), LEAF),                               # majority GROUND
    ]:
        add(x, y, 3.0, -1, sem)

    # Tree 0: three non-collinear points in cell (7, 4).
    for (dx, dy), z in zip([(0.1, 0.1), (0.2, 0.3), (0.3, 0.1)], [10.0, 11.0, 12.0]):
        add(*_cell_xy(7, 4, dx, dy), z, 0, LEAF)
    # Tree 1: one single point -> degenerate hull.
    add(*_cell_xy(13, 10), 8.0, 1, WOOD)
    # Trees 2 and 3 share cell (2, 15); tree 3's points are higher, so 3 wins it.
    for (dx, dy), z in zip([(0.10, 0.10), (0.20, 0.20), (0.10, 0.20)], [5.0, 5.0, 4.0]):
        add(*_cell_xy(2, 15, dx, dy), z, 2, LEAF)
    for (dx, dy), z in zip([(0.05, 0.05), (0.15, 0.15), (0.25, 0.12)], [20.0, 20.0, 19.0]):
        add(*_cell_xy(2, 15, dx, dy), z, 3, LEAF)
    # ... and tree 2 also owns cell (4, 12) alone, so every tree reaches the raster.
    add(*_cell_xy(4, 12), 6.0, 2, LEAF)

    return write_result_las(
        path,
        np.array(xs), np.array(ys), np.array(zs),
        np.array(tids, np.int32), np.array(sems, np.uint8),
    )


@pytest.fixture
def result_las(tmp_path):
    return _synthetic_result_las(tmp_path / "tile.las")


@pytest.mark.parametrize(
    "cell, tag", [(0.5, "50cm"), (0.25, "25cm"), (1.0, "1m"), (1.5, "1.5m"), (2.0, "2m")]
)
def test_cell_tag(cell, tag):
    assert _cell_tag(cell) == tag


def test_las_to_masks_writes_three_files(result_las, tmp_path):
    out = tmp_path / "masks"
    info = las_to_masks(result_las, out, cell_m=CELL)
    assert info["shape"] == (ROWS, COLS)
    assert info["n_trees"] == 4
    assert info["cell_m"] == CELL
    assert info["instance"].name == "tile_instance_50cm.tif"
    assert info["semantic"].name == "tile_semantic_50cm.tif"
    assert info["crowns"].name == "tile_crowns.gpkg"
    for key in ("instance", "semantic", "crowns"):
        assert info[key].is_file()


def test_instance_raster_values(result_las, tmp_path):
    info = las_to_masks(result_las, tmp_path, cell_m=CELL)
    with rasterio.open(info["instance"]) as src:
        assert src.dtypes[0] == "int32"
        assert src.nodata == -1
        data = src.read(1)
    assert data.shape == (ROWS, COLS)
    assert data[7, 4] == 0  # id 0 is a real id, not nodata
    assert data[13, 10] == 1
    assert data[4, 12] == 2
    assert data[2, 15] == 3  # the higher tree wins a shared cell
    assert data[19, 0] == -1  # ground-only cell
    assert data[0, 0] == -1  # empty cell
    assert sorted(np.unique(data[data >= 0]).tolist()) == [0, 1, 2, 3]
    # exactly four cells hold a tree: each tree's points sit in one cell, and the
    # cell trees 2 and 3 share counts once (for tree 3).
    assert (data >= 0).sum() == 4


def test_semantic_raster_majority_ties_and_nodata(result_las, tmp_path):
    info = las_to_masks(result_las, tmp_path, cell_m=CELL)
    with rasterio.open(info["semantic"]) as src:
        assert src.dtypes[0] == "uint8"
        assert src.nodata == 255
        data = src.read(1)
    assert data[19, 0] == GROUND  # ground lattice
    assert data[1, 1] == LEAF  # 1 wood vs 1 leaf -> higher class index
    assert data[1, 3] == WOOD  # 1 ground vs 1 wood -> higher class index
    assert data[1, 5] == GROUND  # 2 ground vs 1 leaf -> plain majority
    assert data[7, 4] == LEAF  # tree 0
    assert data[0, 0] == NODATA  # no points at all
    assert data[0, 19] == NODATA  # only a semantic==255 point


def test_georeferencing(result_las, tmp_path):
    info = las_to_masks(result_las, tmp_path, cell_m=CELL)
    for key in ("instance", "semantic"):
        with rasterio.open(info[key]) as src:
            assert src.crs == rasterio.crs.CRS.from_epsg(25833)
            assert src.transform * (0, 0) == pytest.approx((X0, Y1))
            assert src.transform * (COLS, ROWS) == pytest.approx((X1, Y0))
            # the cell holding tree 0 maps back to its known UTM corner
            assert src.transform * (4, 7) == pytest.approx((381002.0, 5828006.5))
            assert src.res == (CELL, CELL)


def test_crowns_layer(result_las, tmp_path):
    info = las_to_masks(result_las, tmp_path, cell_m=CELL)
    crowns = gpd.read_file(info["crowns"], layer="crowns", engine="pyogrio")
    assert list(crowns.columns) == [
        "tree_id", "n_points", "top_z", "crown_area_m2", "hull_is_point", "geometry"
    ]
    assert crowns.crs.to_epsg() == 25833
    with rasterio.open(info["instance"]) as src:
        raster_ids = set(np.unique(src.read(1)[src.read(1) >= 0]).tolist())
    assert set(crowns["tree_id"].tolist()) == raster_ids

    by_id = crowns.set_index("tree_id")
    assert by_id.loc[0, "n_points"] == 3
    assert by_id.loc[0, "top_z"] == pytest.approx(12.0)
    assert not bool(by_id.loc[0, "hull_is_point"])
    assert by_id.loc[0, "geometry"].geom_type == "Polygon"

    assert by_id.loc[1, "n_points"] == 1
    assert bool(by_id.loc[1, "hull_is_point"])
    assert by_id.loc[1, "crown_area_m2"] == pytest.approx(CELL * CELL)
    assert by_id.loc[1, "geometry"].geom_type == "Polygon"

    assert by_id.loc[2, "n_points"] == 4
    assert by_id.loc[3, "top_z"] == pytest.approx(20.0)
    assert all(g.geom_type == "Polygon" for g in crowns.geometry)


def test_prefix_and_cell_override(result_las, tmp_path):
    info = las_to_masks(result_las, tmp_path, cell_m=1.0, prefix="berlin")
    assert info["instance"].name == "berlin_instance_1m.tif"
    assert info["semantic"].name == "berlin_semantic_1m.tif"
    assert info["crowns"].name == "berlin_crowns.gpkg"
    assert info["shape"] == (10, 10)
    with rasterio.open(info["instance"]) as src:
        assert src.res == (1.0, 1.0)
        assert src.read(1)[3, 2] == 0  # tree 0's 0.5 m cell (7, 4) -> 1 m cell (3, 2)


def test_tree_free_las(tmp_path):
    """A LAS with no treeID >= 0 (open field, water, a --score-th that kept nothing)."""
    las = write_result_las(
        tmp_path / "bare.las",
        np.array([X0 + 0.1, X0 + 3.1]),
        np.array([Y0 + 0.1, Y0 + 3.1]),
        np.array([0.0, 1.0]),
        np.array([-1, -1], np.int32),
        np.array([GROUND, WOOD], np.uint8),
    )
    info = las_to_masks(las, tmp_path / "out", cell_m=CELL)
    assert info["n_trees"] == 0

    with rasterio.open(info["instance"]) as src:
        instance = src.read(1)
    assert (instance == -1).all()

    with rasterio.open(info["semantic"]) as src:
        semantic = src.read(1)
    assert set(np.unique(semantic).tolist()) == {GROUND, WOOD, NODATA}

    crowns = gpd.read_file(info["crowns"], layer="crowns", engine="pyogrio")
    assert len(crowns) == 0
    assert list(crowns.columns) == [
        "tree_id", "n_points", "top_z", "crown_area_m2", "hull_is_point", "geometry"
    ]


def test_collinear_tree_falls_back_to_a_square(tmp_path):
    """3+ distinct but collinear points: the convex hull is a LineString, not a Polygon."""
    n = 4
    las = write_result_las(
        tmp_path / "line.las",
        np.full(n, X0 + 2.0) + np.arange(n) * 0.0,  # one x ...
        Y0 + 2.0 + np.arange(n) * 0.5,              # ... walking north: a vertical line
        np.arange(n, dtype=float),
        np.zeros(n, np.int32),
        np.full(n, LEAF, np.uint8),
    )
    crowns = gpd.read_file(
        las_to_masks(las, tmp_path / "out", cell_m=CELL)["crowns"],
        layer="crowns", engine="pyogrio",
    )
    assert len(crowns) == 1
    row = crowns.iloc[0]
    assert bool(row["hull_is_point"])
    assert row["geometry"].geom_type == "Polygon"
    assert row["crown_area_m2"] == pytest.approx(CELL * CELL)
    assert row["n_points"] == n


def test_points_on_the_max_edge_land_in_the_last_cell(result_las, tmp_path):
    """The clamp in _cell_index: a point at exactly max x / max y is still in range.

    Without it, ``(x - x0) / cell`` at the snapped maximum indexes one past the last
    column. Every Berlin tile's southern and eastern edges hit this.
    """
    corners = [(X0, Y0), (X1, Y0), (X0, Y1), (X1, Y1)]
    n = len(corners)
    las = write_result_las(
        tmp_path / "corners.las",
        np.array([c[0] for c in corners]),
        np.array([c[1] for c in corners]),
        np.arange(n, dtype=float),
        np.arange(n, dtype=np.int32),
        np.full(n, LEAF, np.uint8),
    )
    info = las_to_masks(las, tmp_path / "out", cell_m=CELL)
    assert info["shape"] == (ROWS, COLS)
    with rasterio.open(info["instance"]) as src:
        data = src.read(1)
    assert data[ROWS - 1, 0] == 0  # (min x, min y) -> SW corner cell
    assert data[ROWS - 1, COLS - 1] == 1  # (max x, min y), both clamped
    assert data[0, 0] == 2  # (min x, max y)
    assert data[0, COLS - 1] == 3  # (max x, max y), both clamped
    assert (data >= 0).sum() == 4


def test_cli_masks(result_las, tmp_path, capsys):
    out = tmp_path / "cli"
    assert main(["masks", "--las", str(result_las), "--out", str(out), "--cell", "0.5"]) == 0
    printed = capsys.readouterr().out
    assert str(out / "tile_instance_50cm.tif") in printed
    assert str(out / "tile_semantic_50cm.tif") in printed
    assert str(out / "tile_crowns.gpkg") in printed
    assert "4 trees" in printed
    assert (out / "tile_crowns.gpkg").is_file()
