"""benchmark/sat_to_ff3d.py: SegmentAnyTree output -> the ForestFormer3D result contract.

Needs laspy and pyproj (skipped otherwise); the product test additionally needs
geopandas/shapely (tree table, report) and rasterio (masks).
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

laspy = pytest.importorskip("laspy")
pytest.importorskip("pyproj")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmark"))

from sat_to_ff3d import (  # noqa: E402
    NO_SCORE,
    SEMANTIC_UNKNOWN,
    main,
    sat_semantic,
    sat_to_las,
    sat_tree_ids,
)


def write_sat_las(path, x, y, z, classification, pred_semantic, pred_instance):
    """A SegmentAnyTree ``final_results/<name>_out.las``: LAS 1.4 / pf6, extra dims
    PredSemantic (uint8) and PredInstance (uint32), no CRS (the sub-tile inputs have none)."""
    header = laspy.LasHeader(point_format=6, version="1.4")
    header.scales = np.array([0.001, 0.001, 0.001])
    header.offsets = np.floor([np.min(x), np.min(y), np.min(z)])
    header.add_extra_dim(laspy.ExtraBytesParams(name="PredSemantic", type=np.uint8))
    header.add_extra_dim(laspy.ExtraBytesParams(name="PredInstance", type=np.uint32))
    las = laspy.LasData(header)
    las.x, las.y, las.z = x, y, z
    las.classification = np.asarray(classification, dtype=np.uint8)
    las.PredSemantic = np.asarray(pred_semantic, dtype=np.uint8)
    las.PredInstance = np.asarray(pred_instance, dtype=np.uint32)
    las.write(str(path))
    return path


def test_tree_ids_shift_by_one_and_keep_none():
    ids = sat_tree_ids(np.array([0, 1, 2, 5, 0], dtype=np.uint32))
    assert ids.dtype == np.int32
    assert ids.tolist() == [-1, 0, 1, 4, -1]


def test_semantic_mapping_non_tree_ground_tree_leaf_else_unknown():
    sem = sat_semantic(np.array([0, 1, 1, 7], dtype=np.uint8))
    assert sem.dtype == np.uint8
    assert sem.tolist() == [0, 2, 2, SEMANTIC_UNKNOWN]


def _subtile(path, origin_token, pred_instance, pred_semantic, classification):
    n = len(pred_instance)
    rng = np.random.default_rng(n)
    x = rng.uniform(0, 100, n)
    y = rng.uniform(0, 100, n)
    z = rng.uniform(30, 60, n)
    return write_sat_las(path, x, y, z, classification, pred_semantic, pred_instance), (x, y, z)


def test_subtiles_merge_with_unique_ids_and_utm_origin(tmp_path):
    a, (ax, ay, az) = _subtile(tmp_path / "t_E381000_N5828000_100m_out.las", None,
                               [0, 1, 2, 2, 5], [0, 1, 1, 1, 1], [2, 5, 5, 4, 5])
    b, (bx, by, bz) = _subtile(tmp_path / "t_E381100_N5828000_100m_out.las", None,
                               [0, 1, 1, 3], [0, 1, 1, 1], [2, 5, 5, 3])
    out = tmp_path / "tile.las"
    # passed in reverse order on purpose: the converter sorts by name
    info = sat_to_las([b, a], out)
    assert info["n_points"] == 9 and info["n_files"] == 2
    assert info["id_offsets"] == {str(a): 0, str(b): 5}   # a's max id is 4 -> b starts at 5
    assert info["n_trees"] == 5

    m = laspy.read(str(out))
    assert str(m.header.version) == "1.4" and m.header.point_format.id == 6
    assert m.header.parse_crs().to_epsg() == 25833
    dims = {d.name: d for d in m.point_format.extra_dimensions}
    assert set(dims) == {"treeID", "semantic", "score"}
    assert dims["treeID"].dtype == np.int32 and dims["semantic"].dtype == np.uint8
    assert dims["score"].dtype == np.float32
    assert all(len(d.description) < 32 for d in dims.values())

    ids = np.asarray(m.treeID)
    assert ids[:5].tolist() == [-1, 0, 1, 1, 4]
    assert ids[5:].tolist() == [-1, 5, 5, 7]
    assert np.asarray(m.semantic).tolist() == [0, 2, 2, 2, 2, 0, 2, 2, 2]
    assert np.asarray(m.classification).tolist() == [2, 5, 5, 4, 5, 2, 5, 5, 3]
    assert np.all(np.asarray(m.score) == np.float32(NO_SCORE))
    np.testing.assert_allclose(np.asarray(m.x)[:5], ax + 381000, atol=0.002)
    np.testing.assert_allclose(np.asarray(m.y)[5:], by + 5828000, atol=0.002)
    np.testing.assert_allclose(np.asarray(m.z), np.concatenate([az, bz]), atol=0.002)


def test_absolute_inputs_keep_their_coordinates(tmp_path):
    x = np.array([381300.5, 381350.25]); y = np.array([5828300.0, 5828390.0]); z = np.array([40.0, 55.0])
    p = write_sat_las(tmp_path / "plot_out.las", x, y, z, [2, 5], [0, 1], [0, 1])
    with pytest.raises(ValueError):
        sat_to_las([p], tmp_path / "fail.las")            # no E<x>_N<y> token, not --absolute
    sat_to_las([p], tmp_path / "ok.las", absolute=True)
    m = laspy.read(str(tmp_path / "ok.las"))
    np.testing.assert_allclose(np.asarray(m.x), x, atol=0.002)
    assert np.asarray(m.treeID).tolist() == [-1, 0]


def test_rejects_a_file_without_the_sat_dims(tmp_path):
    header = laspy.LasHeader(point_format=6, version="1.4")
    las = laspy.LasData(header)
    las.x, las.y, las.z = [1.0], [2.0], [3.0]
    p = tmp_path / "x_E1_N2_out.las"
    las.write(str(p))
    with pytest.raises(ValueError, match="PredInstance"):
        sat_to_las([p], tmp_path / "out.las")


def test_main_writes_the_ff3d_products(tmp_path):
    pytest.importorskip("geopandas")
    pytest.importorskip("shapely")
    pytest.importorskip("rasterio")
    from geo_fixtures import two_cone_points

    x, y, z, cls, tid, _sem = two_cone_points()
    p = write_sat_las(tmp_path / "cones_E381300_N5828300_100m_out.las", x, y, z, cls,
                      (tid >= 1).astype(np.uint8), np.where(tid >= 1, tid, 0))
    out = tmp_path / "sat-cones"
    assert main(["--sat-las", str(p), "--tile", "cones", "--out", str(out), "--runtime-s", "12"]) == 0

    rep = json.loads((out / "cones_report.json").read_text())
    assert rep["n_trees"] == 2 and rep["n_points"] == len(x)
    assert rep["method"] == "SegmentAnyTree" and rep["runtime_s"] == 12
    assert rep["chm_baseline_count"] == 2
    import geopandas as gpd

    trees = gpd.read_file(out / "cones_trees.gpkg", layer="trees")
    assert len(trees) == 2 and sorted(trees["tree_id"].tolist()) == [0, 1]
    assert np.allclose(trees["mean_score"], NO_SCORE)
    assert (out / "cones_instance_50cm.tif").exists()
    assert (out / "cones_semantic_50cm.tif").exists()
    assert len(gpd.read_file(out / "cones_crowns.gpkg", layer="crowns")) == 2
    assert "Trees (model) | 2" in (out / "cones_report.md").read_text()
