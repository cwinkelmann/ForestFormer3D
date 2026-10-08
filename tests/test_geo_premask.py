"""Pre-inference polygon mask (ff3d_geo.premask): points inside the polygons never reach
the model and come back as semantic 3 without an instance, in the original order."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("laspy")
pytest.importorskip("plyfile")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import laspy  # noqa: E402
from plyfile import PlyData  # noqa: E402

from ff3d_geo.convert import SEMANTIC_MASKED, las_to_ply, results_to_las  # noqa: E402
from ff3d_geo.premask import parse_polygon_spec  # noqa: E402
from test_geo_convert import CLASSIFICATION, XYZ, write_fake_result_ply, write_three_point_las  # noqa: E402,F401


def test_parse_polygon_spec_splits_an_optional_layer():
    assert parse_polygon_spec("/a/b/water.gpkg") == (Path("/a/b/water.gpkg"), None)
    assert parse_polygon_spec("/a/b/alkis.gpkg:water") == (Path("/a/b/alkis.gpkg"), "water")
    assert parse_polygon_spec("C:/x/y.gpkg") == (Path("C:/x/y.gpkg"), None)      # a drive colon is not a layer


def _write_result(input_ply, offsets_npy, result_ply, semantic, instance, score):
    """A tools/test.py-like result for the points of ``input_ply`` (centred float32 + labels)."""
    from plyfile import PlyElement

    src = PlyData.read(str(input_ply))["vertex"].data
    xyz = np.column_stack([src["x"], src["y"], src["z"]]).astype(np.float64)
    offsets = np.array([xyz[:, 0].mean(), xyz[:, 1].mean(), xyz[:, 2].min()])
    np.save(offsets_npy, offsets)
    v = np.empty(len(src), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"), ("semantic_pred", "i4"),
                                  ("instance_pred", "i4"), ("score", "f4")])
    v["x"], v["y"], v["z"] = (xyz - offsets).T.astype(np.float32)
    v["semantic_pred"], v["instance_pred"], v["score"] = semantic, instance, score
    PlyData([PlyElement.describe(v, "vertex")], text=False, byte_order="<").write(str(result_ply))


ORIGIN = np.array([381300.0, 5828300.0])      # the E<int>_N<int> token of the test LAS names


def _polygon_around(path, x, y, half=0.2, layer="water"):
    gpd = pytest.importorskip("geopandas")
    from shapely.geometry import box

    g = gpd.GeoDataFrame({"name": ["pond"]}, geometry=[box(x - half, y - half, x + half, y + half)], crs="EPSG:25833")
    g.to_file(path, driver="GPKG", layer=layer)
    return f"{path}:{layer}"


def test_premask_round_trip_restores_every_point_in_order(tmp_path):
    pytest.importorskip("geopandas")
    las_path = write_three_point_las(tmp_path / "tile_E381300_N5828300.las")
    # the polygon sits at the point's GEOREFERENCED position: the LAS holds local
    # coordinates and the origin comes from its name, so masking in local space finds nothing
    spec = _polygon_around(tmp_path / "mask.gpkg", XYZ[1, 0] + ORIGIN[0], XYZ[1, 1] + ORIGIN[1])
    ply_path = tmp_path / "tile.ply"; sidecar_path = tmp_path / "tile.sidecar.json"

    info = las_to_ply(las_path, ply_path, sidecar_path, mask_polygons=[spec], mask_buffer=0.0)
    assert info["n_points"] == 3 and info["n_points_inference"] == 2
    assert info["premask"]["n_masked"] == 1 and info["premask"]["n_polygons"] == 1
    assert len(PlyData.read(str(ply_path))["vertex"].data) == 2                  # the model never sees point 1
    pm = np.load(info["premask"]["npz"])
    assert pm["mask"].tolist() == [False, True, False]

    offsets_npy = tmp_path / "tile_offsets.npy"; result_ply = tmp_path / "result.ply"
    _write_result(ply_path, offsets_npy, result_ply, semantic=[0, 2], instance=[-1, 4], score=[-1.0, 0.8])

    out_las = tmp_path / "out.las"
    results_to_las(result_ply, sidecar_path, offsets_npy, out_las)
    out = laspy.read(str(out_las))
    assert out.header.point_count == 3
    got = np.column_stack([out.x, out.y, out.z])
    expected = XYZ + np.array([381300.0, 5828300.0, 0.0])
    assert np.abs(got - expected).max() <= 0.01                                   # same order, masked point exact
    np.testing.assert_array_equal(out.classification, CLASSIFICATION)
    np.testing.assert_array_equal(out.treeID, [-1, -1, 4])
    np.testing.assert_array_equal(out.semantic, [0, SEMANTIC_MASKED, 2])
    np.testing.assert_allclose(out.score, [-1.0, -1.0, 0.8], atol=1e-6)


def test_results_to_las_rejects_a_result_that_ignores_the_mask(tmp_path):
    pytest.importorskip("geopandas")
    las_path = write_three_point_las(tmp_path / "tile_E381300_N5828300.las")
    spec = _polygon_around(tmp_path / "mask.gpkg", XYZ[1, 0] + ORIGIN[0], XYZ[1, 1] + ORIGIN[1])
    ply_path = tmp_path / "tile.ply"; sidecar_path = tmp_path / "tile.sidecar.json"
    las_to_ply(las_path, ply_path, sidecar_path, mask_polygons=[spec], mask_buffer=0.0)
    # a result with all three points (as if the mask had not been applied) must not be georeferenced
    las_to_ply(las_path, tmp_path / "full.ply", tmp_path / "full.sidecar.json")
    offsets_npy = tmp_path / "o.npy"; result_ply = tmp_path / "r.ply"
    write_fake_result_ply(tmp_path / "full.ply", offsets_npy, result_ply)
    with pytest.raises(ValueError, match="after the pre-inference mask"):
        results_to_las(result_ply, sidecar_path, offsets_npy, tmp_path / "out.las")


def test_no_polygon_hits_leaves_the_ply_complete_and_records_zero(tmp_path):
    pytest.importorskip("geopandas")
    las_path = write_three_point_las(tmp_path / "tile_E381300_N5828300.las")
    spec = _polygon_around(tmp_path / "mask.gpkg", 999.0, 999.0)        # nowhere near the tile
    info = las_to_ply(las_path, tmp_path / "t.ply", tmp_path / "t.sidecar.json", mask_polygons=[spec])
    assert info["premask"]["n_masked"] == 0 and info["n_points_inference"] == 3
    assert json.loads((tmp_path / "t.sidecar.json").read_text())["premask"]["buffer_m"] == 1.0


def test_report_carries_the_premask_block(tmp_path):
    gpd = pytest.importorskip("geopandas")
    from geo_fixtures import write_two_cone_las
    from ff3d_geo.report import build_report, report_markdown
    from ff3d_geo.trees import trees_to_gpkg

    las = tmp_path / "two.las"; write_two_cone_las(las, with_predictions=True)
    gpkg = tmp_path / "two_trees.gpkg"; trees_to_gpkg(las, gpkg)
    rep = build_report(las, gpkg, premask={"polygons": ["/x/alkis_water.gpkg:water"], "n_polygons": 3,
                                           "buffer_m": 1.0, "n_masked": 0, "npz": "/x/two_premask.npz"})
    assert rep["premask"]["n_polygons"] == 3 and "npz" not in rep["premask"]
    assert "buildings" not in rep                     # no post-hoc building block invented from semantic 3
    md = report_markdown(rep)
    assert "Pre-inference mask" in md and "alkis_water.gpkg" in md


def test_a_polygon_at_the_local_coordinates_masks_nothing(tmp_path):
    """The regression this file was written for: a polygon placed at the LAS's local
    coordinates (not at origin + local) must find no points. The first run of the 57-tile
    mask reported `n_polygons: 0, n_masked: 0` on the most built-up tile of the mosaic
    because the bbox query and the point test used the sub-tile's local metres."""
    pytest.importorskip("geopandas")
    las_path = write_three_point_las(tmp_path / "tile_E381300_N5828300.las")
    spec = _polygon_around(tmp_path / "local.gpkg", XYZ[1, 0], XYZ[1, 1])
    info = las_to_ply(las_path, tmp_path / "t.ply", tmp_path / "t.sidecar.json",
                      mask_polygons=[spec], mask_buffer=0.0)
    assert info["premask"]["n_polygons"] == 0 and info["premask"]["n_masked"] == 0
