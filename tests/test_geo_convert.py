"""Tests for ff3d_geo.convert.las_to_ply and results_to_las.

Needs laspy, plyfile and pyproj, so it starts with ``pytest.importorskip`` for
each (see tests/test_geo_origin.py's module docstring) so the system-python
test run (no geo libs installed) skips it instead of failing.
"""

import json

import numpy as np
import pytest

laspy = pytest.importorskip("laspy")
plyfile = pytest.importorskip("plyfile")
pyproj = pytest.importorskip("pyproj")

from plyfile import PlyData, PlyElement  # noqa: E402

from ff3d_geo.convert import las_to_ply, results_to_las  # noqa: E402

XYZ = np.array([[0.5, 0.25, 1.0], [99.99, 50.0, 20.5], [10.0, 10.0, 0.0]])
CLASSIFICATION = np.array([2, 5, 2], dtype=np.uint8)


def write_three_point_las(path):
    header = laspy.LasHeader(point_format=1, version="1.2")
    header.scales = np.array([0.01, 0.01, 0.01])
    header.offsets = np.array([0.0, 0.0, 0.0])
    las = laspy.LasData(header)
    las.x, las.y, las.z = XYZ[:, 0], XYZ[:, 1], XYZ[:, 2]
    las.classification = CLASSIFICATION
    las.write(str(path))
    return path


def test_las_to_ply_writes_local_coordinates_xyz_only_and_sidecar(tmp_path):
    las_path = write_three_point_las(tmp_path / "tile_E381300_N5828300.las")
    ply_path = tmp_path / "test_data" / "tile_E381300_N5828300.ply"
    sidecar_path = tmp_path / "out" / "tile_E381300_N5828300.sidecar.json"

    sidecar = las_to_ply(las_path, ply_path, sidecar_path)

    vertex = PlyData.read(str(ply_path))["vertex"]
    assert vertex.data.dtype.names == ("x", "y", "z")
    assert vertex.data["x"].dtype == np.float64
    np.testing.assert_allclose(np.column_stack([vertex["x"], vertex["y"], vertex["z"]]), XYZ)

    assert sidecar == json.loads(sidecar_path.read_text())
    assert sidecar["stem"] == "tile_E381300_N5828300"
    assert sidecar["origin"] == [381300.0, 5828300.0]
    assert sidecar["epsg"] == 25833
    assert sidecar["source_scale"] == [0.01, 0.01, 0.01]
    assert sidecar["source_offset"] == [0.0, 0.0, 0.0]
    assert sidecar["source_point_format"] == 1
    assert sidecar["source_version"] == "1.2"
    assert sidecar["n_points"] == 3
    np.testing.assert_array_equal(np.load(sidecar["classification_npy"]), CLASSIFICATION)
    assert sidecar["classification_npy"].endswith("tile_E381300_N5828300_classification.npy")


def test_las_to_ply_explicit_origin_overrides_name(tmp_path):
    las_path = write_three_point_las(tmp_path / "noname.las")
    sidecar = las_to_ply(
        las_path, tmp_path / "a.ply", tmp_path / "a.json", origin=(1.0, 2.0), epsg=3857
    )
    assert sidecar["origin"] == [1.0, 2.0] and sidecar["epsg"] == 3857


def write_fake_result_ply(input_ply, offsets_npy, result_ply):
    """Mimic batch_load + tools/test.py: center the coordinates (float32),
    add semantic_pred/instance_pred/score fields, including unlabelled votes."""
    src = PlyData.read(str(input_ply))["vertex"].data
    xyz = np.column_stack([src["x"], src["y"], src["z"]]).astype(np.float64)
    offsets = np.array([xyz[:, 0].mean(), xyz[:, 1].mean(), xyz[:, 2].min()], dtype=np.float64)
    np.save(offsets_npy, offsets)
    result = np.empty(
        len(src),
        dtype=[
            ("x", "f4"),
            ("y", "f4"),
            ("z", "f4"),
            ("semantic_pred", "i4"),
            ("instance_pred", "i4"),
            ("score", "f4"),
        ],
    )
    result["x"], result["y"], result["z"] = (xyz - offsets).T.astype(np.float32)
    # point 0: no vote at all (-1/-1); point 1: wood with a tree; point 2: ground.
    result["semantic_pred"] = [-1, 1, 0]
    result["instance_pred"] = [-1, 7, -1]
    result["score"] = [-1.0, 0.9, -1.0]
    PlyData([PlyElement.describe(result, "vertex")], text=False, byte_order="<").write(
        str(result_ply)
    )
    return offsets


def test_round_trip_restores_utm_coordinates_crs_and_fields(tmp_path):
    las_path = write_three_point_las(tmp_path / "tile_E381300_N5828300.las")
    ply_path = tmp_path / "tile_E381300_N5828300.ply"
    sidecar_path = tmp_path / "tile_E381300_N5828300.sidecar.json"
    las_to_ply(las_path, ply_path, sidecar_path)
    offsets_npy = tmp_path / "tile_E381300_N5828300_offsets.npy"
    result_ply = tmp_path / "result.ply"
    write_fake_result_ply(ply_path, offsets_npy, result_ply)
    out_las = tmp_path / "out" / "tile.las"

    results_to_las(result_ply, sidecar_path, offsets_npy, out_las)

    out = laspy.read(str(out_las))
    assert str(out.header.version) == "1.4" and out.header.point_format.id == 6
    assert out.header.parse_crs().to_epsg() == 25833
    expected = XYZ + np.array([381300.0, 5828300.0, 0.0])
    got = np.column_stack([out.x, out.y, out.z])
    assert np.abs(got - expected).max() <= 0.01
    np.testing.assert_array_equal(out.classification, CLASSIFICATION)
    np.testing.assert_array_equal(out.treeID, [-1, 7, -1])
    # point 0 had semantic_pred == -1 (no vote) -> nodata sentinel 255, not a wrapped uint8.
    np.testing.assert_array_equal(out.semantic, [255, 1, 0])
    np.testing.assert_allclose(out.score, [-1.0, 0.9, -1.0], atol=1e-6)
    assert out.treeID.dtype == np.int32
    assert out.semantic.dtype == np.uint8
    assert out.score.dtype == np.float32
    assert list(out.header.scales) == [0.01, 0.01, 0.01]


def test_results_to_las_rejects_classification_length_mismatch(tmp_path):
    las_path = write_three_point_las(tmp_path / "tile_E381300_N5828300.las")
    ply_path = tmp_path / "in.ply"
    sidecar_path = tmp_path / "in.sidecar.json"
    sidecar = las_to_ply(las_path, ply_path, sidecar_path)
    np.save(sidecar["classification_npy"], np.array([2, 2], dtype=np.uint8))  # wrong length
    offsets_npy = tmp_path / "offsets.npy"
    result_ply = tmp_path / "result.ply"
    write_fake_result_ply(ply_path, offsets_npy, result_ply)
    with pytest.raises(ValueError, match="classification has 2 entries"):
        results_to_las(result_ply, sidecar_path, offsets_npy, tmp_path / "out.las")


def test_results_to_las_rejects_point_count_mismatch(tmp_path):
    las_path = write_three_point_las(tmp_path / "tile_E381300_N5828300.las")
    ply_path = tmp_path / "in.ply"
    sidecar_path = tmp_path / "in.sidecar.json"
    las_to_ply(las_path, ply_path, sidecar_path)
    offsets_npy = tmp_path / "offsets.npy"
    result_ply = tmp_path / "result.ply"
    write_fake_result_ply(ply_path, offsets_npy, result_ply)

    # Truncate the result PLY to simulate a mismatch against sidecar n_points
    # (which would silently misalign point order if not caught).
    result = PlyData.read(str(result_ply))["vertex"].data[:2]
    PlyData([PlyElement.describe(result, "vertex")], text=False, byte_order="<").write(
        str(result_ply)
    )

    with pytest.raises(ValueError, match="3 points"):
        results_to_las(result_ply, sidecar_path, offsets_npy, tmp_path / "out.las")
