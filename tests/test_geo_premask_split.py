"""The pre-inference mask for the methods that read the split directly (SegmentAnyTree,
AMS3D): mask their input sub-tiles, run, then restore the masked points so the results
still match the ORIGINAL split manifest that ff3d_geo stitch validates against."""

import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("laspy")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import laspy  # noqa: E402

from ff3d_geo.convert import SEMANTIC_MASKED, result_point_header  # noqa: E402
from ff3d_geo.premask import (  # noqa: E402
    restore_premasked_dir,
    restore_premasked_las,
    write_premasked_subtiles,
)

# four points of a sub-tile, in local tile coordinates; the origin is in the file name
SUB_XYZ = np.array([[1.0, 1.0, 10.0], [2.0, 2.0, 11.0], [50.0, 50.0, 3.0], [51.0, 50.0, 4.0]])
SUB_CLASS = np.array([5, 5, 2, 2], dtype=np.uint8)
ORIGIN = (381300.0, 5828300.0)


def _write_subtile(path, xyz=SUB_XYZ, classification=SUB_CLASS):
    """A split sub-tile as ff3d_geo split writes it: local coordinates, ALS classification."""
    header = laspy.LasHeader(point_format=1, version="1.2")
    header.scales = np.array([0.01, 0.01, 0.01])
    header.offsets = np.zeros(3)
    las = laspy.LasData(header)
    las.x, las.y, las.z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    las.classification = classification
    las.write(str(path))
    return path


def _roof_polygon(path, half=5.0, layer="buildings"):
    """A footprint over the first two points, in georeferenced coordinates."""
    gpd = pytest.importorskip("geopandas")
    from shapely.geometry import box

    cx, cy = ORIGIN[0] + 1.5, ORIGIN[1] + 1.5
    g = gpd.GeoDataFrame({"n": ["hall"]},
                         geometry=[box(cx - half, cy - half, cx + half, cy + half)],
                         crs="EPSG:25833")
    g.to_file(path, driver="GPKG", layer=layer)
    return f"{path}:{layer}"


def _method_result(path, xyz, classification, tree_id, semantic, score):
    """What SegmentAnyTree / AMS3D hand back: the contract LAS for the points they saw."""
    header = result_point_header(25833, [0.01] * 3, np.floor(xyz.min(axis=0)))
    las = laspy.LasData(header)
    las.x, las.y, las.z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    las.classification = classification
    las.treeID = np.asarray(tree_id, dtype=np.int32)
    las.semantic = np.asarray(semantic, dtype=np.uint8)
    las.score = np.asarray(score, dtype=np.float32)
    las.write(str(path))
    return path


def test_premask_split_writes_the_kept_points_and_an_npz(tmp_path):
    pytest.importorskip("geopandas")
    sub = tmp_path / "sub"; sub.mkdir()
    _write_subtile(sub / "t_E381300_N5828300_100m.las")
    spec = _roof_polygon(tmp_path / "alkis.gpkg")

    out = tmp_path / "sub-pm"
    info = write_premasked_subtiles(sub, out, [spec], buffer_m=0.0)
    assert info == {"sub_tiles": 1, "n_points": 4, "n_masked": 2, "fully_masked": [],
                    "fraction": 0.5}

    kept = laspy.read(str(out / "t_E381300_N5828300_100m.las"))
    assert kept.header.point_count == 2
    assert np.allclose(np.asarray(kept.x), SUB_XYZ[2:, 0])          # source order preserved
    np.testing.assert_array_equal(kept.classification, SUB_CLASS[2:])

    pm = np.load(out / "t_E381300_N5828300_100m_premask.npz")
    assert pm["mask"].tolist() == [True, True, False, False]
    np.testing.assert_allclose(pm["x"], SUB_XYZ[:2, 0])
    np.testing.assert_array_equal(pm["classification"], SUB_CLASS[:2])


def test_a_fully_masked_subtile_gets_no_input_las(tmp_path):
    """The segmenters cannot read an empty cloud, so the sub-tile is left out and the
    restore synthesises its all-masked result instead."""
    pytest.importorskip("geopandas")
    sub = tmp_path / "sub"; sub.mkdir()
    _write_subtile(sub / "lake_E381300_N5828300_100m.las", xyz=SUB_XYZ[:2], classification=SUB_CLASS[:2])
    spec = _roof_polygon(tmp_path / "alkis.gpkg")

    out = tmp_path / "sub-pm"
    info = write_premasked_subtiles(sub, out, [spec], buffer_m=0.0)
    assert info["fully_masked"] == ["lake_E381300_N5828300_100m"] and info["n_masked"] == 2
    assert not (out / "lake_E381300_N5828300_100m.las").exists()
    assert (out / "lake_E381300_N5828300_100m_premask.npz").exists()

    res = restore_premasked_dir(tmp_path / "results", out, tmp_path / "restored")
    assert res == {"written": 1, "restored": 0, "fully_masked": 1}
    back = laspy.read(str(tmp_path / "restored" / "lake_E381300_N5828300_100m.las"))
    assert back.header.point_count == 2
    np.testing.assert_array_equal(back.semantic, [SEMANTIC_MASKED] * 2)
    np.testing.assert_array_equal(back.treeID, [-1, -1])


def test_restore_puts_the_masked_points_back_in_source_order(tmp_path):
    pytest.importorskip("geopandas")
    sub = tmp_path / "sub"; sub.mkdir()
    _write_subtile(sub / "t_E381300_N5828300_100m.las")
    spec = _roof_polygon(tmp_path / "alkis.gpkg")
    out = tmp_path / "sub-pm"
    write_premasked_subtiles(sub, out, [spec], buffer_m=0.0)

    # the method saw only the two unmasked points and labelled them one tree
    results = tmp_path / "results"; results.mkdir()
    _method_result(results / "t_E381300_N5828300_100m.las", SUB_XYZ[2:], SUB_CLASS[2:],
                   tree_id=[7, 7], semantic=[2, 1], score=[0.8, 0.8])

    n = restore_premasked_las(results / "t_E381300_N5828300_100m.las",
                              out / "t_E381300_N5828300_100m_premask.npz",
                              tmp_path / "restored.las")
    assert n == 4
    back = laspy.read(str(tmp_path / "restored.las"))
    assert back.header.point_count == 4
    assert np.abs(np.column_stack([back.x, back.y, back.z]) - SUB_XYZ).max() <= 0.01
    np.testing.assert_array_equal(back.classification, SUB_CLASS)
    np.testing.assert_array_equal(back.treeID, [-1, -1, 7, 7])
    np.testing.assert_array_equal(back.semantic, [SEMANTIC_MASKED, SEMANTIC_MASKED, 2, 1])
    np.testing.assert_allclose(back.score, [-1.0, -1.0, 0.8, 0.8], atol=1e-6)


def test_restore_rejects_a_result_with_the_wrong_point_count(tmp_path):
    """The guard that saved the ForestFormer3D mosaic: a result from another run must not
    reach the stitch."""
    pytest.importorskip("geopandas")
    sub = tmp_path / "sub"; sub.mkdir()
    _write_subtile(sub / "t_E381300_N5828300_100m.las")
    spec = _roof_polygon(tmp_path / "alkis.gpkg")
    out = tmp_path / "sub-pm"
    write_premasked_subtiles(sub, out, [spec], buffer_m=0.0)

    results = tmp_path / "results"; results.mkdir()
    _method_result(results / "t_E381300_N5828300_100m.las", SUB_XYZ, SUB_CLASS,
                   tree_id=[1, 1, 2, 2], semantic=[2, 2, 1, 1], score=[0.5] * 4)
    with pytest.raises(ValueError, match="does not belong to this masked sub-tile"):
        restore_premasked_las(results / "t_E381300_N5828300_100m.las",
                              out / "t_E381300_N5828300_100m_premask.npz",
                              tmp_path / "x.las")


def test_restore_dir_refuses_a_missing_result_for_a_kept_subtile(tmp_path):
    pytest.importorskip("geopandas")
    sub = tmp_path / "sub"; sub.mkdir()
    _write_subtile(sub / "t_E381300_N5828300_100m.las")
    spec = _roof_polygon(tmp_path / "alkis.gpkg")
    out = tmp_path / "sub-pm"
    write_premasked_subtiles(sub, out, [spec], buffer_m=0.0)
    with pytest.raises(FileNotFoundError, match="kept points but have no result"):
        restore_premasked_dir(tmp_path / "empty", out, tmp_path / "restored")
