"""Tests for ff3d_geo.stitch: overlap matching, union-find, global ids, km-tile LAS.

Needs laspy/shapely/pyproj/geopandas/pyogrio, so it starts with ``pytest.importorskip``
for each (see tests/test_geo_origin.py's module docstring) so the system-python test
run (no geo libs installed) skips it instead of failing.
"""

import json

import numpy as np
import pytest

laspy = pytest.importorskip("laspy")
pytest.importorskip("shapely")
pytest.importorskip("pyproj")
gpd = pytest.importorskip("geopandas")
pytest.importorskip("pyogrio")

# IDENT_DTYPE is taken from stitch, not split: this test builds the ``<stem>_ident.npy``
# sidecars itself (synthetic fixtures of the split contract) so it does not depend on
# ff3d_geo.split having grown the halo/ident support yet. The re-export is asserted to
# be identical to split's below, once split has one.
from ff3d_geo.stitch import (IDENT_DTYPE, UnionFind, adjacent_pairs, ident_keys,
                             load_mosaic, match_instances, stitch)
from geo_fixtures import write_grid_las, write_result_las

E0, N0 = 381000, 5829000


def _disc(cx, cy, r, n, rng):
    a, d = rng.uniform(0, 2 * np.pi, n), np.sqrt(rng.uniform(0, 1, n)) * r
    return np.column_stack([cx + d * np.cos(a), cy + d * np.sin(a), rng.uniform(5, 20, n)])


def _mosaic(tmp_path, b_labels_for_shared=None):
    """One 200 m x 100 m 'km tile' split into two 100 m cores (A west, B east) with a
    20 m halo. Tree T1 straddles x = 100 (label 3 in A, 7 in B), T2 sits in A's core only
    (label 1), T3 in B's core only (label 2); everything else is ground (-1)."""
    rng = np.random.default_rng(0)
    ground = np.column_stack([rng.uniform(0, 200, 3000), rng.uniform(0, 100, 3000), np.zeros(3000)])
    t1, t2, t3 = _disc(100, 50, 6, 400, rng), _disc(50, 50, 5, 300, rng), _disc(150, 50, 5, 300, rng)
    xyz = np.vstack([ground, t1, t2, t3])
    xyz[:, 0] += E0
    xyz[:, 1] += N0
    tree = np.concatenate([np.full(3000, 0), np.full(400, 1), np.full(300, 2), np.full(300, 3)])
    cls = np.where(tree == 0, 2, 5).astype(np.uint8)
    src = write_grid_las(tmp_path / "3dm_33_381_5829_1_be.las", xyz, cls)
    sub = tmp_path / "sub"
    sub.mkdir()
    res = tmp_path / "res"
    res.mkdir()
    subtiles = []
    for stem, x0, la in (("t_E381000_N5829000_100m", E0, {1: 3, 2: 1}),
                         ("t_E381100_N5829000_100m", E0 + 100, {1: 7, 3: 2})):
        inside = (xyz[:, 0] >= x0 - 20) & (xyz[:, 0] < x0 + 120)
        idx = np.flatnonzero(inside)
        ident = np.empty(len(idx), IDENT_DTYPE)
        ident["tile"], ident["index"] = 0, idx
        np.save(sub / f"{stem}_ident.npy", ident)
        labels = np.array([la.get(t, -1) for t in tree[idx]], np.int32)
        if b_labels_for_shared is not None and stem.startswith("t_E381100"):
            labels = b_labels_for_shared(xyz[idx], tree[idx], labels)
        sem = np.where(labels >= 0, 2, 0).astype(np.uint8)
        write_result_las(res / f"{stem}.las", xyz[idx, 0], xyz[idx, 1], xyz[idx, 2], labels, sem)
        core = (xyz[idx, 0] >= x0) & (xyz[idx, 0] < x0 + 100)
        subtiles.append({"stem": stem, "origin": [x0, N0], "source": 0,
                         "n_points": int(len(idx)), "n_core": int(core.sum())})
    manifest = sub / "split_manifest.json"
    manifest.write_text(json.dumps({
        "size_m": 100, "buffer_m": 20, "prefix": "t",
        "sources": [{"key": "3dm_33_381_5829", "path": str(src), "n_points": int(len(xyz))}],
        "subtiles": subtiles}))
    return manifest, res, xyz, tree


def test_match_instances_iou_over_shared_points_only():
    keys_a = np.arange(0, 100, dtype=np.int64)
    labels_a = np.where(keys_a < 60, 3, -1).astype(np.int32)      # label 3 = keys 0..59
    keys_b = np.arange(40, 140, dtype=np.int64)
    labels_b = np.where(keys_b < 70, 7, np.where(keys_b < 130, 8, -1)).astype(np.int32)
    # shared keys 40..99: label 3 covers 40..59 (20), label 7 covers 40..69 (30), overlap 20
    pairs = match_instances(keys_a, labels_a, keys_b, labels_b, iou_threshold=0.5, min_shared=5)
    assert pairs == [(3, 7, pytest.approx(20 / 30), 20)]
    assert match_instances(keys_a, labels_a, keys_b, labels_b, iou_threshold=0.7, min_shared=5) == []
    assert match_instances(keys_a, labels_a, keys_b, labels_b, iou_threshold=0.5, min_shared=21) == []


def test_stitch_ident_dtype_matches_split():
    """The re-export must stay byte-identical to the one ``split`` writes the sidecars with.

    Guarded so this file passes both before and after ``ff3d_geo.split`` grows the
    halo/ident support (the sub-tile sidecars here are synthetic).
    """
    split = pytest.importorskip("ff3d_geo.split")
    split_dtype = getattr(split, "IDENT_DTYPE", None)
    if split_dtype is None:
        pytest.skip("ff3d_geo.split has no IDENT_DTYPE yet")
    assert split_dtype == IDENT_DTYPE


def test_ident_keys_pack_global_tile_and_index():
    ident = np.array([(0, 5), (1, 7)], IDENT_DTYPE)
    keys = ident_keys(ident, np.array([4, 2]))
    assert keys.tolist() == [(4 << 32) | 5, (2 << 32) | 7]


def test_union_find_is_deterministic():
    uf = UnionFind()
    assert uf.union(("b", 1), ("a", 3)) and not uf.union(("a", 3), ("b", 1))
    assert uf.find(("b", 1)) == ("a", 3)


def test_load_mosaic_and_adjacent_pairs(tmp_path):
    manifest, _, _, _ = _mosaic(tmp_path)
    mosaic = load_mosaic([manifest])
    assert mosaic.size_m == 100 and mosaic.buffer_m == 20
    assert [s["key"] for s in mosaic.sources] == ["3dm_33_381_5829"]
    assert adjacent_pairs(mosaic) == [("t_E381000_N5829000_100m", "t_E381100_N5829000_100m")]


def test_stitch_unifies_the_straddling_tree_and_writes_one_km_las(tmp_path):
    manifest, res, xyz, tree = _mosaic(tmp_path)
    out = tmp_path / "out"
    info = stitch([manifest], [res], out)
    assert info["n_pairs_tested"] == 1 and info["n_unified"] == 1 and info["n_cross_km"] == 0
    assert info["n_trees"] == 3
    las = laspy.read(out / "3dm_33_381_5829_1_be.las")
    assert len(las.points) == len(xyz)
    ids = np.asarray(las.treeID)
    np.testing.assert_allclose(np.asarray(las.x), xyz[:, 0], atol=2e-3)       # source order kept
    assert sorted(np.unique(ids[ids >= 0]).tolist()) == [0, 1, 2]                # dense global ids
    assert len(np.unique(ids[tree == 1])) == 1 and (ids[tree == 1] >= 0).all()  # T1 is ONE id on both sides
    assert len({int(ids[tree == 1][0]), int(ids[tree == 2][0]), int(ids[tree == 3][0])}) == 3
    assert (ids[tree == 0] == -1).all() and (np.asarray(las.semantic)[tree == 0] == 0).all()
    assert las.header.parse_crs().to_epsg() == 25833
    assert {"treeID", "semantic", "score"} <= set(las.point_format.extra_dimension_names)
    t = gpd.read_file(out / "3dm_33_381_5829_1_be_trees.gpkg", layer="trees")
    assert len(t) == 3 and (out / "3dm_33_381_5829_1_be_report.json").is_file()
    st = json.loads((out / "stitch.json").read_text())
    assert st["n_unified"] == 1 and st["iou_threshold"] == 0.5


def test_stitch_keeps_fragments_apart_below_the_iou_threshold(tmp_path):
    def split_b(pts, tree_of, labels):        # B sees T1 as two halves: 7 north, 9 south
        south = (tree_of == 1) & (pts[:, 1] < N0 + 50)
        return np.where(south, 9, labels).astype(np.int32)
    manifest, res, xyz, tree = _mosaic(tmp_path, b_labels_for_shared=split_b)
    info = stitch([manifest], [res], tmp_path / "out", iou_threshold=0.6)
    assert info["n_unified"] == 0 and info["n_trees"] == 5
    info = stitch([manifest], [res], tmp_path / "out2", iou_threshold=0.3)
    assert info["n_unified"] == 1 and info["n_trees"] == 4       # 3 joins the bigger half only


def test_stitch_names_a_missing_subtile_result(tmp_path):
    manifest, res, _, _ = _mosaic(tmp_path)
    (res / "t_E381100_N5829000_100m.las").unlink()
    with pytest.raises(FileNotFoundError, match="t_E381100_N5829000_100m"):
        stitch([manifest], [res], tmp_path / "out")
