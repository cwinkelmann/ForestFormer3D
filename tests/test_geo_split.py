import json
import struct

import numpy as np
import pytest

laspy = pytest.importorskip("laspy")

from ff3d_geo.origin import parse_origin
from ff3d_geo.split import IDENT_DTYPE, split_las, subtile_origins
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


def test_split_counts_every_source_point_including_sparse_subtiles(tmp_path):
    """The conservation check passes on a normal tile even when min_points drops one
    sub-tile: the skipped sub-tile's points still count as accounted for."""
    xyz = np.array([[10.0, 10.0, 1.0]] * 50 + [[150.0, 10.0, 1.0]] * 5)
    src = tmp_path / "tile_1_be.las"
    _write_las(src, xyz, np.full(len(xyz), 2, np.uint8))
    written = split_las(src, tmp_path / "sub", size_m=100, min_points=20)
    assert [p.name for p in written] == ["tile_E0_N0_100m.las"]
    assert len(laspy.read(written[0]).points) == 50
    assert not list((tmp_path / "sub").glob(".*.tmp"))


def test_split_rejects_a_las_whose_header_bounds_are_stale(tmp_path):
    """A LAS header whose mins/maxs do not cover the points (laspy does not re-derive
    them on read) would drop the points outside the grid silently; it must raise."""
    xyz = np.column_stack([
        np.concatenate([np.full(200, 50.0), np.full(200, 250.0)]),
        np.full(400, 50.0),
        np.full(400, 1.0)])
    src = tmp_path / "stale_1_be.las"
    _write_las(src, xyz, np.full(len(xyz), 2, np.uint8))

    # Claim the tile ends at x = 100 although half the points sit at x = 250. The
    # bounds have to be patched in the written header: laspy's writer re-derives
    # mins/maxs from the points it writes, and only a reader takes them as given.
    # LAS public header: max/min X, Y, Z are six doubles from byte 179.
    raw = bytearray(src.read_bytes())
    struct.pack_into("<6d", raw, 179, 100.0, 0.0, 100.0, 0.0, 1.0, 1.0)
    src.write_bytes(bytes(raw))
    with laspy.open(str(src)) as reader:
        assert list(reader.header.maxs) == [100.0, 100.0, 1.0]
        assert int(reader.header.point_count) == 400

    out = tmp_path / "sub"
    with pytest.raises(ValueError, match=r"200 of .* 400 points"):
        split_las(src, out, size_m=100, min_points=1)
    # nothing half-written is left behind
    assert not list(out.glob("*.las"))
    assert not list(out.glob(".*.tmp"))


def test_split_prefix_never_ends_in_digits(tmp_path):
    xyz = np.array([[1.0, 1.0, 1.0]] * 30)
    src = tmp_path / "plot_7.las"
    _write_las(src, xyz, np.full(30, 2, np.uint8))
    written = split_las(src, tmp_path / "sub", size_m=100, min_points=1)
    assert written[0].name == "plot_7_E0_N0_100m.las"
    assert not written[0].stem.endswith(tuple("0123456789")) or written[0].stem.endswith("_100m")


def _tile(tmp_path, name, x0, n, seed):
    rng = np.random.default_rng(seed)
    xyz = np.column_stack([
        x0 + rng.uniform(0, 200, n), 5829000 + rng.uniform(0, 100, n), rng.uniform(30, 60, n)])
    cls = rng.choice([2, 3, 4, 5], n).astype(np.uint8)
    return _write_las(tmp_path / name, xyz, cls), xyz


def test_split_with_halo_adds_neighbour_strip_and_writes_ident(tmp_path):
    src, xyz = _tile(tmp_path, "3dm_33_381_5829_1_be.las", 381000, 8000, 1)
    out = tmp_path / "sub"
    written = split_las(src, out, size_m=100, min_points=10, buffer_m=20)
    assert [p.name for p in written] == [
        "3dm_33_381_5829_E381000_N5829000_100m.las", "3dm_33_381_5829_E381100_N5829000_100m.las"]
    left = laspy.read(written[0])
    expect_left = int((xyz[:, 0] < 381120).sum())          # core 0..100 plus 20 m of the east neighbour
    assert len(left.points) == expect_left
    assert left.x.min() >= 0 and 100 < left.x.max() <= 120  # nothing west of the km tile exists
    ident = np.load(out / "3dm_33_381_5829_E381000_N5829000_100m_ident.npy")
    assert ident.dtype == IDENT_DTYPE and len(ident) == expect_left
    assert set(ident["tile"].tolist()) == {0}
    np.testing.assert_allclose(np.asarray(left.x) + 381000, xyz[ident["index"], 0], atol=2e-3)
    np.testing.assert_allclose(np.asarray(left.y) + 5829000, xyz[ident["index"], 1], atol=2e-3)
    right = laspy.read(written[1])
    assert -20 <= right.x.min() < 0 and right.x.max() <= 100
    manifest = json.loads((out / "split_manifest.json").read_text())
    assert manifest["size_m"] == 100 and manifest["buffer_m"] == 20 and manifest["prefix"] == "3dm_33_381_5829"
    assert manifest["sources"] == [{"key": "3dm_33_381_5829", "path": str(src.resolve()), "n_points": 8000}]
    subs = {s["stem"]: s for s in manifest["subtiles"]}
    assert subs["3dm_33_381_5829_E381000_N5829000_100m"] == {
        "stem": "3dm_33_381_5829_E381000_N5829000_100m", "origin": [381000, 5829000], "source": 0,
        "n_points": expect_left, "n_core": int((xyz[:, 0] < 381100).sum())}


def test_split_halo_takes_points_from_neighbour_km_tiles(tmp_path):
    rng = np.random.default_rng(2)
    n = 4000
    main_xyz = np.column_stack([381000 + rng.uniform(0, 100, n), 5829000 + rng.uniform(0, 100, n), rng.uniform(30, 60, n)])
    east_xyz = np.column_stack([381100 + rng.uniform(0, 100, n), 5829000 + rng.uniform(0, 100, n), rng.uniform(30, 60, n)])
    cls = np.full(n, 5, np.uint8)
    src = _write_las(tmp_path / "3dm_33_381_5829_1_be.las", main_xyz, cls)
    east = _write_las(tmp_path / "3dm_33_382_5829_1_be.las", east_xyz, cls)
    out = tmp_path / "sub"
    written = split_las(src, out, size_m=100, min_points=10, buffer_m=20, neighbours=[east])
    assert [p.name for p in written] == ["3dm_33_381_5829_E381000_N5829000_100m.las"]
    las = laspy.read(written[0])
    ident = np.load(out / "3dm_33_381_5829_E381000_N5829000_100m_ident.npy")
    from_east = ident["tile"] == 1
    assert 0 < from_east.sum() == int((east_xyz[:, 0] < 381120).sum())
    assert (~from_east).sum() == n
    np.testing.assert_allclose(np.asarray(las.x)[from_east] + 381000, east_xyz[ident["index"][from_east], 0], atol=2e-3)
    manifest = json.loads((out / "split_manifest.json").read_text())
    assert [s["key"] for s in manifest["sources"]] == ["3dm_33_381_5829", "3dm_33_382_5829"]
    assert manifest["subtiles"][0]["n_core"] == n and manifest["subtiles"][0]["n_points"] == n + int(from_east.sum())


def test_split_without_halo_is_unchanged_and_still_writes_ident(tmp_path):
    src, xyz = _tile(tmp_path, "3dm_33_381_5829_1_be.las", 381000, 6000, 0)
    out = tmp_path / "sub"
    written = split_las(src, out, size_m=100, min_points=10)
    left = laspy.read(written[0])
    assert left.x.min() >= 0 and left.x.max() <= 100
    ident = np.load(out / "3dm_33_381_5829_E381000_N5829000_100m_ident.npy")
    assert len(ident) == len(left.points) and (ident["tile"] == 0).all()
    manifest = json.loads((out / "split_manifest.json").read_text())
    assert manifest["buffer_m"] == 0
    assert all(s["n_core"] == s["n_points"] for s in manifest["subtiles"])


def test_split_min_points_counts_core_points_only(tmp_path):
    # 5 core points in the east cell, 50 in the west; the west cell's halo would push the
    # east cell over min_points if halo points counted, but they must not.
    xyz = np.array([[95.0, 10.0, 1.0]] * 50 + [[150.0, 10.0, 1.0]] * 5)
    src = _write_las(tmp_path / "tile_1_be.las", xyz, np.full(len(xyz), 2, np.uint8))
    written = split_las(src, tmp_path / "sub", size_m=100, min_points=20, buffer_m=20)
    assert [p.name for p in written] == ["tile_E0_N0_100m.las"]
    assert not (tmp_path / "sub" / "tile_E100_N0_100m_ident.npy").exists()


def test_split_removes_a_stale_manifest_before_rewriting(tmp_path, monkeypatch):
    """The manifest is written last so its presence means "complete". A rerun that fails
    partway must therefore not leave the PREVIOUS run's manifest describing the new,
    half-written sub-tiles -- stitch would key points by an ident sidecar that no longer
    matches the LAS beside it."""
    import ff3d_geo.split as split_mod

    src, _ = _tile(tmp_path, "3dm_33_381_5829_1_be.las", 381000, 8000, 3)
    out = tmp_path / "sub"
    split_las(src, out, size_m=100, min_points=10)
    assert (out / "split_manifest.json").exists()

    real_save = split_mod.np.save
    calls = []

    def flaky_save(path, arr):
        calls.append(path)
        if len(calls) == 2:
            raise OSError("no space left on device")
        return real_save(path, arr)

    monkeypatch.setattr(split_mod.np, "save", flaky_save)
    with pytest.raises(OSError, match="no space left"):
        split_las(src, out, size_m=100, min_points=10, buffer_m=20)
    assert not (out / "split_manifest.json").exists()
    assert not list(out.glob(".*.tmp"))


def test_split_rejects_a_source_repeated_among_the_neighbours(tmp_path):
    src, _ = _tile(tmp_path, "3dm_33_381_5829_1_be.las", 381000, 2000, 4)
    east, _ = _tile(tmp_path, "3dm_33_382_5829_1_be.las", 381200, 2000, 5)
    with pytest.raises(ValueError, match="distinct"):
        split_las(src, tmp_path / "a", size_m=100, min_points=10, buffer_m=20, neighbours=[src])
    with pytest.raises(ValueError, match="distinct"):
        split_las(src, tmp_path / "b", size_m=100, min_points=10, buffer_m=20,
                  neighbours=[east, east])
    assert not list(tmp_path.glob("*/*.las"))


def test_split_without_halo_never_opens_the_neighbours(tmp_path):
    """buffer_m == 0 leaves no halo for a neighbour to fill, so none is read -- a
    neighbour path that does not even exist must not make the split fail."""
    src, xyz = _tile(tmp_path, "3dm_33_381_5829_1_be.las", 381000, 4000, 6)
    out = tmp_path / "sub"
    written = split_las(src, out, size_m=100, min_points=10,
                        neighbours=[tmp_path / "no_such_tile_1_be.las"])
    assert [p.name for p in written] == [
        "3dm_33_381_5829_E381000_N5829000_100m.las", "3dm_33_381_5829_E381100_N5829000_100m.las"]
    assert laspy.read(written[0]).x.max() <= 100
    manifest = json.loads((out / "split_manifest.json").read_text())
    assert [s["key"] for s in manifest["sources"]] == ["3dm_33_381_5829"]
