"""Tests for ff3d_geo.convert.las_to_ply.

Needs laspy and plyfile, so it starts with ``pytest.importorskip`` for each
(see tests/test_geo_origin.py's module docstring) so the system-python test
run (no geo libs installed) skips it instead of failing.
"""

import json

import numpy as np
import pytest

laspy = pytest.importorskip("laspy")
plyfile = pytest.importorskip("plyfile")

from plyfile import PlyData  # noqa: E402

from ff3d_geo.convert import las_to_ply  # noqa: E402

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
    # Only x, y, z: no semantic_seg/treeID, so load_forainetv2_data.py's
    # export(..., unlabeled=True) takes its constant-label path (sem 0, ins -1)
    # instead of reading these as real (and wrong) labels.
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
    sidecar = las_to_ply(las_path, tmp_path / "a.ply", tmp_path / "a.json", origin=(1.0, 2.0), epsg=3857)
    assert sidecar["origin"] == [1.0, 2.0] and sidecar["epsg"] == 3857
