"""benchmark/ptf_to_ff3d.py: PointTreeFormer LAZ -> ff3d contract, buffer cropped, order matched."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("laspy")
pytest.importorskip("geopandas")
pytest.importorskip("rasterio")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmark"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from geo_fixtures import write_result_las  # noqa: E402

import laspy  # noqa: E402
from ptf_to_ff3d import convert, core_mask, order_like, tile_from_name  # noqa: E402

T = "3dm_33_380_5828_1_be"
E, N = 380000.0, 5828000.0


def _ptf_file(path: Path, rng):
    """A tiny 'clipped' PointTreeFormer tile: 400 core points plus 60 buffer points west of it."""
    n = 400
    # the real source ALS is on a 1 mm grid and PointTreeFormer re-quantised it to 1 cm; the
    # fixture does the same so a 1 mm reference written from these values matches exactly
    snap = lambda v, off: np.round((v - off) / 0.001).astype(np.int64) * 0.001 + off      # what laspy hands back  # noqa: E731
    x = snap(E + rng.uniform(0, 1000, n), E); y = snap(N + rng.uniform(0, 1000, n), N); z = snap(rng.uniform(30, 60, n), 0.0)
    xb = E - rng.uniform(0.1, 20, 60); yb = N + rng.uniform(0, 1000, 60); zb = rng.uniform(30, 60, 60)
    inst = np.where(x < E + 500, 0, 1); inst[:20] = -1
    cls = np.where(inst < 0, 0, 2); cls[20:40] = 1
    hdr = laspy.LasHeader(version="1.4", point_format=6)
    hdr.scales = [0.01, 0.01, 0.01]; hdr.offsets = [E, N, 0.0]
    for name, dt in (("classification_prediction", np.int64), ("classification_binary_prediction", np.int64), ("instance_id_prediction", np.int64)):
        hdr.add_extra_dim(laspy.ExtraBytesParams(name=name, type=dt))
    las = laspy.LasData(hdr)
    las.x = np.concatenate([x, xb]); las.y = np.concatenate([y, yb]); las.z = np.concatenate([z, zb])
    las.classification = np.where(np.concatenate([cls, np.zeros(60, int)]) == 0, 2, 5).astype(np.uint8)
    las.classification_prediction = np.concatenate([cls, np.zeros(60, int)])
    las.classification_binary_prediction = (las.classification_prediction > 0).astype(np.int64)
    las.instance_id_prediction = np.concatenate([inst, np.full(60, -1)])
    las.write(str(path))
    return x, y, z, inst, cls


def test_tile_and_core_mask():
    tile, b = tile_from_name(Path("clipped_3dm_33_380_5828_1_be.laz"))
    assert tile == T and b == (E, N, E + 1000, N + 1000)
    assert core_mask(np.array([E - 1, E, E + 999.99, E + 1000]), np.array([N + 5] * 4), b).tolist() == [False, True, True, False]


def test_convert_crops_the_buffer_and_maps_the_dimensions(tmp_path):
    rng = np.random.default_rng(0)
    x, y, z, inst, cls = _ptf_file(tmp_path / f"clipped_{T}.laz", rng)
    info = convert(tmp_path / f"clipped_{T}.laz", tmp_path / "out", masks=False, log=lambda m: None)
    assert info["n_points"] == 400 and info["n_buffer_points"] == 60 and info["n_instances"] == 2
    las = laspy.read(str(tmp_path / "out" / f"{T}.las"))
    assert las.header.point_count == 400 and las.header.parse_crs().to_epsg() == 25833
    assert np.array_equal(np.asarray(las.treeID), inst) and np.array_equal(np.asarray(las.semantic), cls)
    assert (np.asarray(las.score) == -1).all()
    rep = json.loads((tmp_path / "out" / f"{T}_report.json").read_text())
    assert rep["source"]["method"] == "PointTreeFormer" and rep["n_trees"] == 2


def test_order_like_reorders_to_the_reference_and_refuses_strangers(tmp_path):
    rng = np.random.default_rng(1)
    x, y, z, inst, cls = _ptf_file(tmp_path / f"clipped_{T}.laz", rng)
    perm = rng.permutation(400)                      # the reference lists the same points differently
    ref = tmp_path / "ref.las"
    write_result_las(ref, x[perm], y[perm], z[perm], np.zeros(400, int), np.zeros(400, int), offsets=(E, N, 0.0))
    info = convert(tmp_path / f"clipped_{T}.laz", tmp_path / "out", order_like_las=ref, masks=False, log=lambda m: None)
    assert info["reordered"]
    las = laspy.read(str(tmp_path / "out" / f"{T}.las"))
    r = laspy.read(str(ref))
    assert np.array_equal(np.asarray(las.X), np.asarray(r.X)) and np.array_equal(np.asarray(las.Y), np.asarray(r.Y))
    assert np.array_equal(np.asarray(las.treeID), inst[perm])     # labels travelled with their points
    a = np.column_stack([np.arange(5), np.arange(5), np.arange(5)])
    with pytest.raises(ValueError):
        order_like(a, a + 1)
    with pytest.raises(ValueError):
        order_like(a, a[:3])
