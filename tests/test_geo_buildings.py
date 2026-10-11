"""Tests for ff3d_geo.buildings: masking ALKIS building footprints out of a result LAS.

The Berlin ALS 2021 tiles have no building class, so ForestFormer3D predicts tree
instances on roofs. ``mask_buildings`` removes them using the official footprints.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("laspy")
pytest.importorskip("geopandas")
pytest.importorskip("shapely")
pytest.importorskip("pyproj")

sys.path.insert(0, str(Path(__file__).resolve().parent))

from geo_fixtures import write_result_las  # noqa: E402

ORIGIN = (381000.0, 5829000.0)
# One 20 m square footprint with its lower-left corner 10 m into the tile.
ROOF = (ORIGIN[0] + 10.0, ORIGIN[1] + 10.0, ORIGIN[0] + 30.0, ORIGIN[1] + 30.0)


def _grid(x0, y0, nx=8, ny=8, step=2.0):
    gx, gy = np.meshgrid(x0 + np.arange(nx) * step, y0 + np.arange(ny) * step)
    return gx.ravel(), gy.ravel()


def _write_footprints(path):
    import geopandas as gpd
    from shapely.geometry import box

    gdf = gpd.GeoDataFrame(
        {"uuid": ["DEBE00TEST0001"], "bezgfk": ["Wohnhaus"]},
        geometry=[box(*ROOF)],
        crs="EPSG:25833",
    )
    gdf.to_file(str(path), driver="GPKG", layer="buildings")
    return path


def _write_scene(path):
    """Three instances: 1 entirely on the roof, 2 half on it, 3 well off it.

    Instance 2 straddles the footprint's eastern edge: 32 of its 64 points are
    inside, which is exactly 50 %, so a ``min_roof_fraction`` above 0.5 keeps it.
    """
    # instance 1: 8 x 8 points fully inside the footprint (12..26 m)
    x1, y1 = _grid(ROOF[0] + 2.0, ROOF[1] + 2.0)
    # instance 2: 8 x 8 points, left half inside (22..28), right half outside (32..38)
    x2a, y2a = _grid(ROOF[2] - 8.0, ROOF[1] + 2.0, nx=4, ny=8)
    x2b, y2b = _grid(ROOF[2] + 2.0, ROOF[1] + 2.0, nx=4, ny=8)
    x2, y2 = np.concatenate([x2a, x2b]), np.concatenate([y2a, y2b])
    # instance 3: far away, no overlap at all
    x3, y3 = _grid(ORIGIN[0] + 60.0, ORIGIN[1] + 60.0)
    # ground points, none of them in an instance
    xg, yg = _grid(ORIGIN[0] + 50.0, ORIGIN[1] + 5.0, nx=10, ny=10, step=1.0)

    x = np.concatenate([x1, x2, x3, xg])
    y = np.concatenate([y1, y2, y3, yg])
    z = np.concatenate([
        np.full(x1.size, 8.0), np.full(x2.size, 12.0),
        np.full(x3.size, 15.0), np.zeros(xg.size),
    ])
    tree_id = np.concatenate([
        np.full(x1.size, 1), np.full(x2.size, 2),
        np.full(x3.size, 3), np.full(xg.size, -1),
    ])
    semantic = np.concatenate([
        np.full(x1.size, 2), np.full(x2.size, 2),
        np.full(x3.size, 2), np.zeros(xg.size),
    ])
    write_result_las(path, x, y, z, tree_id, semantic)
    return path, {"inside_1": x1.size, "inside_2": x2a.size, "off": x3.size}


def test_mask_buildings_removes_roof_instance_and_keeps_ids(tmp_path):
    import laspy

    from ff3d_geo.buildings import SEMANTIC_BUILDING, mask_buildings

    las_in = tmp_path / "3dm_33_381_5829_1_be.las"
    _write_scene(las_in)
    gpkg = _write_footprints(tmp_path / "alkis.gpkg")
    out = tmp_path / "masked" / las_in.name

    # buffer 0 so the expected counts are exactly the geometric ones.
    info = mask_buildings(las_in, gpkg, out, buffer_m=0.0, min_roof_fraction=0.6)

    assert info["n_footprints"] == 1
    assert info["instances_before"] == 3
    assert info["instances_removed"] == 1  # instance 1, fully on the roof
    assert info["instances_partially_masked"] == 1  # instance 2, half on it
    assert info["instances_after"] == 2
    assert info["points_masked"] == 64 + 32

    las = laspy.read(str(out))
    tree_id = np.asarray(las.treeID)
    semantic = np.asarray(las.semantic)
    # Ids are not renumbered: the survivors keep 2 and 3.
    assert sorted(np.unique(tree_id[tree_id >= 0]).tolist()) == [2, 3]
    # Instance 3 is untouched.
    assert int(np.count_nonzero(tree_id == 3)) == 64
    # Instance 2 keeps only its points outside the footprint.
    assert int(np.count_nonzero(tree_id == 2)) == 32
    # Every masked point is semantic 3 and has no tree.
    building = semantic == SEMANTIC_BUILDING
    assert int(building.sum()) == 96
    assert not np.any(tree_id[building] >= 0)
    # The ALS classification is preserved point for point.
    source = laspy.read(str(las_in))
    assert np.array_equal(np.asarray(las.classification), np.asarray(source.classification))


def test_mask_buildings_min_roof_fraction_drops_the_half_instance(tmp_path):
    from ff3d_geo.buildings import mask_buildings

    las_in = tmp_path / "tile.las"
    _write_scene(las_in)
    gpkg = _write_footprints(tmp_path / "alkis.gpkg")

    info = mask_buildings(las_in, gpkg, tmp_path / "out.las",
                          buffer_m=0.0, min_roof_fraction=0.5)
    assert info["instances_removed"] == 2  # 50 % counts as "at least half"
    assert info["instances_partially_masked"] == 0
    assert info["instances_after"] == 1


def test_mask_buildings_without_footprints_changes_nothing(tmp_path):
    import geopandas as gpd

    from ff3d_geo.buildings import mask_buildings

    las_in = tmp_path / "tile.las"
    _write_scene(las_in)
    empty = tmp_path / "empty.gpkg"
    gpd.GeoDataFrame({"uuid": []}, geometry=[], crs="EPSG:25833").to_file(
        str(empty), driver="GPKG", layer="buildings")

    info = mask_buildings(las_in, empty, tmp_path / "out.las")
    assert info["points_masked"] == 0
    assert info["instances_removed"] == 0
    assert info["instances_after"] == info["instances_before"] == 3


def test_report_has_a_buildings_row(tmp_path):
    from ff3d_geo.buildings import mask_buildings
    from ff3d_geo.report import build_report, report_markdown
    from ff3d_geo.trees import trees_to_gpkg

    las_in = tmp_path / "tile.las"
    _write_scene(las_in)
    gpkg = _write_footprints(tmp_path / "alkis.gpkg")
    out = tmp_path / "masked.las"
    info = mask_buildings(las_in, gpkg, out, buffer_m=0.0, min_roof_fraction=0.6)

    trees = tmp_path / "trees.gpkg"
    trees_to_gpkg(out, trees)
    report = build_report(out, trees, buildings=info)
    assert report["buildings"]["instances_removed"] == 1
    assert report["buildings"]["points_masked"] == 96

    md = report_markdown(report)
    assert "| Buildings (ALKIS footprints) | 1 instances removed, 96 points masked" in md
    assert "3 building (ALKIS footprint" in md
    assert "no vegetation and no building classes" in md


def test_cli_buildings_writes_the_whole_set(tmp_path):
    pytest.importorskip("rasterio")

    from ff3d_geo.cli import main

    las_in = tmp_path / "3dm_33_381_5829_1_be.las"
    _write_scene(las_in)
    gpkg = _write_footprints(tmp_path / "alkis.gpkg")
    out_dir = tmp_path / "masked"

    rc = main(["buildings", "--las", str(las_in), "--buildings", str(gpkg),
               "--out", str(out_dir), "--buffer", "0", "--min-roof-fraction", "0.6"])
    assert rc == 0
    stem = las_in.stem
    for name in (f"{stem}.las", f"{stem}_trees.gpkg", f"{stem}_crowns.gpkg",
                 f"{stem}_report.json", f"{stem}_report.md",
                 f"{stem}_instance_50cm.tif", f"{stem}_semantic_50cm.tif"):
        assert (out_dir / name).is_file(), name
    # The original is untouched.
    assert las_in.is_file()
    assert "Buildings (ALKIS footprints)" in (out_dir / f"{stem}_report.md").read_text()
