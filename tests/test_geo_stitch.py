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


def test_stitch_skips_a_neighbour_only_source(tmp_path):
    """A km tile that only lent halo points to somebody else's split owns no core.

    ``split_las(..., neighbours=[...])`` lists those tiles in the manifest's ``sources``
    (their point indices appear in the ident sidecars), but no sub-tile core covers them.
    Writing one would produce a full-size km LAS of pure nodata plus a report claiming
    zero trees, so it is skipped and named in ``stitch.json`` instead.
    """
    manifest, res, _, _ = _mosaic(tmp_path)
    content = json.loads(manifest.read_text())
    content["sources"].append({"key": "3dm_33_382_5829",
                               "path": str(tmp_path / "3dm_33_382_5829_1_be.las"),
                               "n_points": 999})
    manifest.write_text(json.dumps(content))

    out = tmp_path / "out"
    info = stitch([manifest], [res], out)
    assert set(info["tiles"]) == {"3dm_33_381_5829_1_be"}
    assert not (out / "3dm_33_382_5829_1_be.las").exists()
    assert not (out / "3dm_33_382_5829_1_be_trees.gpkg").exists()
    st = json.loads((out / "stitch.json").read_text())
    assert st["neighbour_only_sources"] == ["3dm_33_382_5829"]
    # The neighbour still shifts the global tile order, so ownership must survive it.
    assert info["n_trees"] == 3


def test_stitch_rejects_a_core_that_no_longer_round_trips(tmp_path):
    manifest, res, _, _ = _mosaic(tmp_path)
    content = json.loads(manifest.read_text())
    content["subtiles"][1]["n_core"] += 1
    manifest.write_text(json.dumps(content))
    with pytest.raises(ValueError, match="t_E381100_N5829000_100m owns"):
        stitch([manifest], [res], tmp_path / "out")


def test_stitch_records_runner_up_candidates(tmp_path):
    """The 1:N rate of the greedy rule is counted, not silently discarded."""
    def split_b(pts, tree_of, labels):
        south = (tree_of == 1) & (pts[:, 1] < N0 + 50)
        return np.where(south, 9, labels).astype(np.int32)
    manifest, res, _, _ = _mosaic(tmp_path, b_labels_for_shared=split_b)
    out = tmp_path / "out"
    stitch([manifest], [res], out, iou_threshold=0.3)
    st = json.loads((out / "stitch.json").read_text())
    # One match (3<->7) with one runner up (3<->9, IoU 0.4775, above RUNNER_UP_IOU).
    assert st["runner_up_histogram"] == {"1": 1} and st["runner_up_iou"] == 0.2


# --- one end-to-end mosaic through the real split_las(neighbours=...) contract -------

def _two_km_tiles(tmp_path):
    """Two 200 m x 200 m 'km tiles' side by side, with eight ground-truth trees.

    Covers every case the halo is meant to fix: a tree fully inside one core, one
    straddling an internal sub-tile line (with points exactly ON the line), one on the
    four-way corner of four sub-tiles, one straddling the KM border, and one that only
    the neighbouring sub-tile's halo predicted. Point order in each source file is
    permuted so nothing can rely on it being spatial.
    """
    rng = np.random.default_rng(1)
    parts, gts = [], []
    ground = np.column_stack([rng.uniform(E0, E0 + 400, 8000),
                              rng.uniform(N0, N0 + 200, 8000), np.zeros(8000)])
    parts.append(ground)
    gts.append(np.zeros(len(ground), np.int64))
    for xline in (E0 + 100, E0 + 200, E0 + 300):      # points exactly on grid lines
        parts.append(np.column_stack([np.full(40, float(xline)),
                                      rng.uniform(N0, N0 + 200, 40), np.zeros(40)]))
        gts.append(np.zeros(40, np.int64))
    parts.append(np.array([[E0 + 100, N0 + 100, 0.0], [E0 + 200, N0 + 100, 0.0],
                           [E0 + 300, N0 + 100, 0.0]]))
    gts.append(np.zeros(3, np.int64))
    trees = {1: (E0 + 50, N0 + 50, 5, 300),        # inside one core
             2: (E0 + 100, N0 + 40, 6, 400),       # straddles the internal line x=E0+100
             3: (E0 + 100, N0 + 100, 6, 500),      # four-way corner
             4: (E0 + 110, N0 + 150, 5, 300),      # only the WEST sub-tile's halo sees it
             5: (E0 + 200, N0 + 50, 6, 400),       # straddles the KM border x=E0+200
             6: (E0 + 350, N0 + 150, 5, 300)}      # inside one core of the east km tile
    for gid, (cx, cy, r, n) in trees.items():
        pts = _disc(cx, cy, r, n, rng)
        if gid == 2:
            pts[:10, 0] = E0 + 100                 # exactly on the line
        parts.append(pts)
        gts.append(np.full(n, gid, np.int64))
    xyz = np.vstack(parts)
    gt = np.concatenate(gts)
    xyz[:, :2] = np.round(xyz[:, :2], 3)           # ALS data is mm quantised
    cls = np.where(gt == 0, 2, 5).astype(np.uint8)

    sources = {}
    for key, mask in (("3dm_33_381_5829", xyz[:, 0] < E0 + 200),
                      ("3dm_33_382_5829", xyz[:, 0] >= E0 + 200)):
        idx = rng.permutation(np.flatnonzero(mask))
        sources[key] = {"xyz": xyz[idx].copy(), "gt": gt[idx], "cls": cls[idx]}
        sources[key]["path"] = write_grid_las(tmp_path / f"{key}_1_be.las",
                                              sources[key]["xyz"], sources[key]["cls"])
    return sources, trees


def _fabricate_results(sub_dir, res_dir, own, neighbour, rng):
    """Write a result LAS per sub-tile: ground truth relabelled per sub-tile.

    Local labels are a fresh random permutation in every sub-tile (that is what the
    model does), and the coordinates go through the float32 centering round trip of
    ``batch_load`` + ``results_to_las`` so the core box is re-derived from coordinates
    that really have been through the pipeline. Tree 4 is predicted ONLY by the sub-tile
    that holds it in its halo, never by the core owner.
    """
    res_dir.mkdir(exist_ok=True)
    manifest = json.loads((sub_dir / "split_manifest.json").read_text())
    per_tile = [own, neighbour]
    for sub in manifest["subtiles"]:
        stem = sub["stem"]
        las = laspy.read(str(sub_dir / f"{stem}.las"))
        ident = np.load(sub_dir / f"{stem}_ident.npy")
        ox, oy = sub["origin"]
        lx, ly, lz = np.asarray(las.x), np.asarray(las.y), np.asarray(las.z)
        mean = np.array([lx.mean(), ly.mean(), lz.min()])
        x = (lx - mean[0]).astype(np.float32).astype(np.float64) + mean[0] + ox
        y = (ly - mean[1]).astype(np.float32).astype(np.float64) + mean[1] + oy
        truth = np.empty(len(ident), np.int64)
        for tile in (0, 1):
            m = ident["tile"] == tile
            truth[m] = per_tile[tile]["gt"][ident["index"][m]]
        perm = rng.permutation(np.arange(10, 30))
        labels = np.where(truth > 0, perm[truth], -1).astype(np.int32)
        if (ox, oy) != (E0, N0 + 100):            # only that sub-tile predicts tree 4
            labels[truth == 4] = -1
        sem = np.where(truth > 0, 2, 0).astype(np.uint8)
        write_result_las(res_dir / f"{stem}.las", x, y, lz, labels, sem)


def test_stitch_over_a_real_two_km_tile_neighbour_split(tmp_path):
    split_las = pytest.importorskip("ff3d_geo.split").split_las
    rng = np.random.default_rng(5)
    sources, trees = _two_km_tiles(tmp_path)
    west, east = sources["3dm_33_381_5829"], sources["3dm_33_382_5829"]
    for name, own, other in (("sub1", west, east), ("sub2", east, west)):
        split_las(own["path"], tmp_path / name, size_m=100, buffer_m=20,
                  neighbours=[other["path"]], min_points=1)
        _fabricate_results(tmp_path / name, tmp_path / name.replace("sub", "res"),
                           own, other, rng)
    manifests = [tmp_path / "sub1" / "split_manifest.json",
                 tmp_path / "sub2" / "split_manifest.json"]
    results = [tmp_path / "res1", tmp_path / "res2"]

    # (c) ownership: every source point claimed by exactly one sub-tile core.
    mosaic = load_mosaic(manifests)
    assert len(mosaic.subtiles) == 8 and len(mosaic.sources) == 2
    import ff3d_geo.stitch as stitch_module
    cache = stitch_module._SubtileCache(
        mosaic, stitch_module._find_results(mosaic, results))
    claimed = {s["key"]: [] for s in mosaic.sources}
    for sub in mosaic.subtiles:
        data = cache(sub["stem"])
        tile = np.int64(mosaic.source_index(sub["source"]))
        own = data.core & ((data.keys >> np.int64(32)) == tile)
        assert int(own.sum()) == sub["n_core"], sub["stem"]
        claimed[sub["source"]].append(data.keys[own] & np.int64(0xFFFFFFFF))
    for source in mosaic.sources:
        got = np.sort(np.concatenate(claimed[source["key"]]))
        assert got.tolist() == list(range(source["n_points"])), source["key"]

    out = tmp_path / "out"
    info = stitch(manifests, results, out)
    assert info["n_cross_km"] == 1                                       # (a) tree 5
    ids, sem = {}, {}
    for key, source in sources.items():
        las = laspy.read(out / f"{key}_1_be.las")
        assert len(las.points) == len(source["xyz"])
        np.testing.assert_allclose(np.asarray(las.x), source["xyz"][:, 0], atol=1e-6)
        np.testing.assert_array_equal(np.asarray(las.classification), source["cls"])
        ids[key] = np.asarray(las.treeID)
        sem[key] = np.asarray(las.semantic)
    all_ids = np.concatenate([ids[k] for k in sources])
    all_gt = np.concatenate([sources[k]["gt"] for k in sources])
    all_sem = np.concatenate([sem[k] for k in sources])

    assert (all_ids[all_gt == 0] == -1).all() and (all_sem[all_gt == 0] == 0).all()
    per_tree = {gid: set(np.unique(all_ids[all_gt == gid]).tolist()) for gid in trees}
    for gid in (1, 2, 3, 5, 6):              # (a) km border, (b) corner: one id each
        assert len(per_tree[gid]) == 1 and per_tree[gid] != {-1}, (gid, per_tree[gid])
    assert per_tree[4] == {-1}               # halo-only prediction owns nothing
    assigned = [next(iter(per_tree[gid])) for gid in (1, 2, 3, 5, 6)]
    assert len(set(assigned)) == 5           # no id shared between two trees
    assert sorted(np.unique(all_ids[all_ids >= 0]).tolist()) == list(range(info["n_trees"]))

    # (e) each tree's row is written ONCE, in the tile holding most of its points, and
    # measured over all of its points -- not once per tile as a fragment.
    tables = {key: gpd.read_file(out / f"{key}_1_be_trees.gpkg", layer="trees")
              for key in sources}
    assert sum(len(t) for t in tables.values()) == info["n_trees"]
    assert json.loads((out / "stitch.json").read_text())["n_cross_km_trees"] == 1
    border_id = next(iter(per_tree[5]))
    holders = [key for key, t in tables.items() if border_id in set(t["tree_id"])]
    assert len(holders) == 1                        # exactly one gpkg lists the straddler
    row = tables[holders[0]].set_index("tree_id").loc[border_id]
    assert row["n_points"] == int((all_gt == 5).sum())   # ... with its FULL point count
    west_points = int((sources["3dm_33_381_5829"]["gt"] == 5).sum())
    assert 0 < west_points < row["n_points"]        # the tree really does span both tiles
    assert row["top_z"] == pytest.approx(
        max(float(sources[k]["xyz"][sources[k]["gt"] == 5, 2].max()) for k in sources))
    for key, table in tables.items():
        assert table["tree_id"].is_unique
        assert (key == holders[0]) or border_id not in set(table["tree_id"])
    # the reports count the owner-assigned rows, so the mosaic total is n_trees
    assert sum(json.loads((out / f"{key}_1_be_report.json").read_text())["n_trees"]
               for key in sources) == info["n_trees"]

    # (d) determinism: reversed manifest order and reversed pair order give the same ids.
    info_rev = stitch(manifests[::-1], results[::-1], tmp_path / "out_rev")
    for key in sources:
        np.testing.assert_array_equal(
            np.asarray(laspy.read(tmp_path / "out_rev" / f"{key}_1_be.las").treeID), ids[key])
    assert info_rev["n_trees"] == info["n_trees"]
    np.testing.assert_array_equal(np.load(out / "stitch_ids.npy"),
                                  np.load(tmp_path / "out_rev" / "stitch_ids.npy"))

    original = stitch_module.adjacent_pairs
    stitch_module.adjacent_pairs = lambda m: list(reversed(original(m)))
    try:
        stitch(manifests, results, tmp_path / "out_pairs")
    finally:
        stitch_module.adjacent_pairs = original
    for key in sources:
        np.testing.assert_array_equal(
            np.asarray(laspy.read(tmp_path / "out_pairs" / f"{key}_1_be.las").treeID), ids[key])
