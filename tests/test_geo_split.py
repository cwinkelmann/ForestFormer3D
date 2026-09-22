import numpy as np
import pytest

laspy = pytest.importorskip("laspy")

from ff3d_geo.origin import parse_origin
from ff3d_geo.split import split_las, subtile_origins
from geo_fixtures import write_grid_las as _write_las


def test_subtile_origins_are_grid_aligned():
    origins = subtile_origins((380999.5, 5828000.2, 0), (381250.0, 5828150.0, 30), 100)
    assert origins == [(380900, 5828000), (380900, 5828100), (381000, 5828000), (381000, 5828100),
                       (381100, 5828000), (381100, 5828100), (381200, 5828000), (381200, 5828100)]


def test_split_writes_local_coordinates_and_keeps_classes(tmp_path):
    rng = np.random.default_rng(0)
    n = 6000
    xyz = np.column_stack([
        381000 + rng.uniform(0, 200, n), 5829000 + rng.uniform(0, 100, n), rng.uniform(30, 60, n)])
    cls = rng.choice([2, 3, 4, 5], n).astype(np.uint8)
    src = tmp_path / "3dm_33_381_5829_1_be.las"
    _write_las(src, xyz, cls)
    out = tmp_path / "sub"
    written = split_las(src, out, size_m=100, min_points=10)
    assert [p.name for p in written] == [
        "3dm_33_381_5829_E381000_N5829000_100m.las", "3dm_33_381_5829_E381100_N5829000_100m.las"]
    total = 0
    for p in written:
        e, nn = parse_origin(p.name)
        las = laspy.read(p)
        assert las.header.point_format.id == 6
        assert las.x.min() >= 0 and las.x.max() <= 100 and las.y.min() >= 0 and las.y.max() <= 100
        sel = (xyz[:, 0] >= e) & (xyz[:, 0] < e + 100) & (xyz[:, 1] >= nn) & (xyz[:, 1] < nn + 100)
        assert len(las.points) == sel.sum()
        np.testing.assert_allclose(np.sort(las.z), np.sort(xyz[sel, 2]), atol=2e-3)
        assert set(np.unique(las.classification)) <= {2, 3, 4, 5}
        total += len(las.points)
    assert total == n


def test_split_skips_sparse_subtiles(tmp_path):
    xyz = np.array([[10.0, 10.0, 1.0]] * 50 + [[150.0, 10.0, 1.0]] * 5)
    src = tmp_path / "tile_1_be.las"
    _write_las(src, xyz, np.full(len(xyz), 2, np.uint8))
    written = split_las(src, tmp_path / "sub", size_m=100, min_points=20)
    assert [p.name for p in written] == ["tile_E0_N0_100m.las"]


def test_split_prefix_never_ends_in_digits(tmp_path):
    xyz = np.array([[1.0, 1.0, 1.0]] * 30)
    src = tmp_path / "plot_7.las"
    _write_las(src, xyz, np.full(30, 2, np.uint8))
    written = split_las(src, tmp_path / "sub", size_m=100, min_points=1)
    assert written[0].name == "plot_7_E0_N0_100m.las"
    assert not written[0].stem.endswith(tuple("0123456789")) or written[0].stem.endswith("_100m")
