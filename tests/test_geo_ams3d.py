"""Tests for ff3d_geo.ams3d (adaptive mean shift crown segmentation, CPU only).

Needs laspy/scipy/shapely (and geopandas/rasterio for the pipeline test), so the
module starts with ``pytest.importorskip`` like the other geo test modules.

The regression test at the end reruns the port on the spike's r12 patch and is
marked ``slow``; it skips when neither the Berlin km tile nor the local 100 m
tile is on this machine.
"""

import json
from pathlib import Path

import numpy as np
import pytest

laspy = pytest.importorskip("laspy")
pytest.importorskip("scipy")
pytest.importorskip("shapely")

from ff3d_geo.ams3d import (  # noqa: E402
    CONFIGS,
    SCORE_NONE,
    SEMANTIC_GROUND,
    SEMANTIC_LEAF,
    SEMANTIC_UNLABELLED,
    Ams3dParams,
    cluster_modes,
    params_for,
    run_ams3d_tile,
    segment_ams3d,
)
from ff3d_geo.split import split_las  # noqa: E402
from geo_fixtures import write_grid_las  # noqa: E402

# (x, y, top height, crown radius) of the Gaussian-blob crowns; far enough apart
# that no kernel ever spans two of them.
BLOBS = [(12.0, 12.0, 12.0, 2.0), (40.0, 12.0, 18.0, 2.5), (15.0, 45.0, 25.0, 3.0)]
# One tall tree whose crown is two blobs 5 m apart vertically at the same xy: the
# textbook kernel (config ``default``, merge_z 2 m) leaves them as two stacked
# clusters, config C's 6 m vertical merge joins them (the spike's main finding).
STACKED_XY = (45.0, 45.0)
STACKED_Z = (10.0, 15.0)


def blob_scene(seed: int = 0, n_per_blob: int = 600):
    """Flat 1 m ground grid (class 2) at z = 0, three Gaussian-blob crowns and the
    stacked tree (class 5); returns xyz, classification and one boolean mask per tree."""
    rng = np.random.default_rng(seed)
    g = np.arange(0.0, 60.0, 1.0)
    gx, gy = np.meshgrid(g, g)
    xs, ys, zs = [gx.ravel()], [gy.ravel()], [np.zeros(gx.size)]
    cls = [np.full(gx.size, 2, np.uint8)]
    for cx, cy, h, r in BLOBS:
        xs.append(rng.normal(cx, r / 2, n_per_blob))
        ys.append(rng.normal(cy, r / 2, n_per_blob))
        zs.append(np.clip(rng.normal(0.7 * h, 0.12 * h, n_per_blob), 2.5, h))
        cls.append(np.full(n_per_blob, 5, np.uint8))
    for zc in STACKED_Z:
        xs.append(rng.normal(STACKED_XY[0], 1.0, n_per_blob))
        ys.append(rng.normal(STACKED_XY[1], 1.0, n_per_blob))
        zs.append(rng.normal(zc, 0.8, n_per_blob))
        cls.append(np.full(n_per_blob, 5, np.uint8))
    xyz = np.column_stack([np.concatenate(xs), np.concatenate(ys), np.concatenate(zs)])
    classification = np.concatenate(cls)
    masks = []
    for cx, cy, _, r in BLOBS + [(STACKED_XY[0], STACKED_XY[1], 0.0, 2.0)]:
        masks.append((np.hypot(xyz[:, 0] - cx, xyz[:, 1] - cy) < 2.5 * r) & (classification == 5))
    return xyz, classification, masks


def test_params_default_is_config_c_and_configs_match_the_spike_table():
    c = Ams3dParams()
    assert (c.ground_cell_m, c.h_min, c.h_max) == (1.0, 2.0, 60.0)
    assert (c.a_s, c.s_min, c.a_r, c.r_min) == (0.12, 1.0, 0.27, 1.5)
    assert (c.merge_xy, c.merge_z, c.min_points) == (1.5, 6.0, 50)
    assert CONFIGS["C"] == c
    d = CONFIGS["default"]
    assert (d.merge_xy, d.merge_z, d.min_points) == (1.0, 2.0, 20) and d.a_r == 0.27
    assert CONFIGS["A"].a_r == 0.5 and CONFIGS["A"].merge_z == 4.0 and CONFIGS["A"].min_points == 20
    assert CONFIGS["B"].a_s == 0.15 and CONFIGS["B"].a_r == 0.5 and CONFIGS["B"].min_points == 50
    assert params_for("C", min_points=30).min_points == 30
    with pytest.raises(ValueError):
        params_for("Z")


def test_synthetic_scene_config_c_finds_every_crown_once():
    xyz, cls, masks = blob_scene()
    # a few points of classes AMS3D ignores (unclassified, building) must come back -1
    xyz = np.vstack([xyz, [[30.0, 30.0, 8.0], [31.0, 30.0, 9.0]]])
    cls = np.concatenate([cls, [1, 6]]).astype(np.uint8)
    masks = [np.concatenate([m, [False, False]]) for m in masks]

    ids = segment_ams3d(xyz, cls, CONFIGS["C"])

    assert ids.dtype == np.int32 and ids.shape == (len(xyz),)
    assert (ids[cls == 2] == -1).all()
    assert (ids[-2:] == -1).all()
    assert (ids[cls == 5] >= 0).all()          # every vegetation point >= 2 m is assigned
    assert np.unique(ids[ids >= 0]).size == len(BLOBS) + 1
    seen = set()
    for m in masks:
        labels = np.unique(ids[m])
        assert labels.size == 1, "points of one blob must share one id"
        seen.add(int(labels[0]))
    assert len(seen) == len(BLOBS) + 1


def test_merge_step_joins_vertically_stacked_modes():
    modes = np.array([[5.0, 5.0, 10.0], [5.0, 5.0, 14.0]])
    assert cluster_modes(modes, CONFIGS["default"]).tolist() == [0, 1]
    assert cluster_modes(modes, CONFIGS["C"]).tolist() == [0, 0]

    xyz, cls, masks = blob_scene()
    stacked = masks[-1]
    split = segment_ams3d(xyz, cls, CONFIGS["default"])
    joined = segment_ams3d(xyz, cls, CONFIGS["C"])
    assert np.unique(split[stacked]).size == 2
    assert np.unique(joined[stacked]).size == 1


def test_segmentation_is_translation_invariant():
    xyz, cls, _ = blob_scene(n_per_blob=300)
    a = segment_ams3d(xyz, cls, CONFIGS["C"])
    b = segment_ams3d(xyz + [381300.0, 5828300.0, 40.0], cls, CONFIGS["C"])
    # same partition up to relabelling
    pairs = {(int(i), int(j)) for i, j in zip(a, b)}
    assert len(pairs) == np.unique(a).size == np.unique(b).size


def _write_local_subtile(path, xyz, cls):
    """A split_las-style sub-tile: LAS 1.4 / pf6, local coordinates, offsets 0."""
    header = laspy.LasHeader(point_format=6, version="1.4")
    header.scales = np.array([0.001, 0.001, 0.001])
    header.offsets = np.zeros(3)
    las = laspy.LasData(header)
    las.x, las.y, las.z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    las.classification = cls
    las.write(str(path))
    return path


def test_run_ams3d_tile_writes_the_result_contract_and_crops_the_buffer(tmp_path):
    xyz, cls, _ = blob_scene(n_per_blob=300)
    # pretend this 60 m scene is a 40 m core with a 10 m buffer: local coords -10..50
    xyz = xyz - [10.0, 10.0, 0.0]
    src = _write_local_subtile(tmp_path / "syn_E381300_N5828300_40m.las", xyz, cls)
    out = tmp_path / "syn_E381300_N5828300_40m_result.las"

    info = run_ams3d_tile(src, out, CONFIGS["C"], buffer_m=10.0)

    core = (xyz[:, 0] >= 0) & (xyz[:, 0] < 40) & (xyz[:, 1] >= 0) & (xyz[:, 1] < 40)
    assert info["n_points"] == int(core.sum()) < len(xyz)
    las = laspy.read(str(out))
    assert las.header.point_format.id == 6 and str(las.header.version) == "1.4"
    assert las.header.parse_crs().to_epsg() == 25833
    assert [d.name for d in las.header.point_format.extra_dimensions] == ["treeID", "semantic", "score"]
    assert len(las.points) == int(core.sum())
    assert las.x.min() >= 381300.0 and las.x.max() < 381340.0
    assert las.y.min() >= 5828300.0 and las.y.max() < 5828340.0
    tree_id = np.asarray(las.treeID)
    semantic = np.asarray(las.semantic)
    classification = np.asarray(las.classification)
    assert tree_id.dtype == np.int32 and semantic.dtype == np.uint8
    assert np.asarray(las.score).dtype == np.float32 and (np.asarray(las.score) == SCORE_NONE).all()
    assert set(np.unique(classification)) == {2, 5}
    assert (tree_id[classification == 2] == -1).all()
    assert (semantic[classification == 2] == SEMANTIC_GROUND).all()
    assert (semantic[tree_id >= 0] == SEMANTIC_LEAF).all()
    assert (semantic[(tree_id < 0) & (classification != 2)] == SEMANTIC_UNLABELLED).all()
    # ids are dense 0..n-1 after the crop
    ids = np.unique(tree_id[tree_id >= 0])
    assert ids.tolist() == list(range(info["n_trees"]))


def test_run_ams3d_tile_with_buffer_needs_a_size_token(tmp_path):
    xyz, cls, _ = blob_scene(n_per_blob=100)
    src = _write_local_subtile(tmp_path / "syn_E381300_N5828300.las", xyz, cls)
    with pytest.raises(ValueError, match="_<size>m"):
        run_ams3d_tile(src, tmp_path / "out.las", CONFIGS["C"], buffer_m=10.0)
    # buffer 0 keeps every point and still shifts local -> UTM
    info = run_ams3d_tile(src, tmp_path / "out.las", CONFIGS["C"], buffer_m=0.0)
    assert info["n_points"] == len(xyz)
    assert laspy.read(str(tmp_path / "out.las")).x.min() >= 381300.0


def test_split_las_buffer_duplicates_edge_points_into_neighbours(tmp_path):
    rng = np.random.default_rng(1)
    n = 8000
    xyz = np.column_stack([
        381000 + rng.uniform(0, 200, n), 5829000 + rng.uniform(0, 100, n), rng.uniform(30, 60, n)])
    cls = rng.choice([2, 5], n).astype(np.uint8)
    src = tmp_path / "3dm_33_381_5829_1_be.las"
    write_grid_las(src, xyz, cls)

    written = split_las(src, tmp_path / "sub", size_m=100, min_points=10, buffer_m=10.0)
    assert [p.name for p in written] == [
        "3dm_33_381_5829_E381000_N5829000_100m.las", "3dm_33_381_5829_E381100_N5829000_100m.las"]
    left = laspy.read(str(written[0]))
    right = laspy.read(str(written[1]))
    # the left sub-tile carries the right one's first 10 m and vice versa
    assert left.x.min() >= 0 and left.x.max() < 110 and left.x.max() > 100
    assert right.x.min() < 0 and right.x.min() >= -10 and right.x.max() < 100
    n_left = int(((xyz[:, 0] < 381110)).sum())
    n_right = int(((xyz[:, 0] >= 381090)).sum())
    assert len(left.points) == n_left and len(right.points) == n_right
    # core points are still conserved exactly once
    core_left = int((xyz[:, 0] < 381100).sum())
    assert int((left.x < 100).sum()) == core_left
    assert int((right.x >= 0).sum()) == n - core_left


def test_pipeline_on_a_small_projected_tile(tmp_path):
    pytest.importorskip("geopandas")
    pytest.importorskip("rasterio")
    from ff3d_geo.ams3d import run_ams3d_pipeline

    # 2 x 1 sub-tiles of 40 m, one crown right on the shared border so the buffer
    # matters, two more away from it.
    rng = np.random.default_rng(2)
    g = np.arange(0.0, 80.0, 1.0)
    gx, gy = np.meshgrid(g, np.arange(0.0, 40.0, 1.0))
    xs, ys, zs = [gx.ravel()], [gy.ravel()], [np.zeros(gx.size)]
    cls = [np.full(gx.size, 2, np.uint8)]
    for cx, cy, h in [(40.0, 20.0, 18.0), (12.0, 12.0, 14.0), (66.0, 28.0, 22.0)]:
        xs.append(rng.normal(cx, 1.2, 400))
        ys.append(rng.normal(cy, 1.2, 400))
        zs.append(np.clip(rng.normal(0.7 * h, 0.12 * h, 400), 2.5, h))
        cls.append(np.full(400, 5, np.uint8))
    xyz = np.column_stack([np.concatenate(xs) + 381000.0, np.concatenate(ys) + 5829000.0,
                           np.concatenate(zs) + 40.0])
    classification = np.concatenate(cls)
    src = tmp_path / "3dm_33_381_5829_1_be.las"
    write_grid_las(src, xyz, classification)
    out = tmp_path / "out"

    logs: list[str] = []
    report = run_ams3d_pipeline(src, out, CONFIGS["C"], buffer_m=10.0, workers=2,
                                size_m=40, config_name="C", log=logs.append)

    stem = "3dm_33_381_5829_1_be"
    for suffix in (".las", "_trees.gpkg", "_crowns.gpkg", "_instance_50cm.tif",
                   "_semantic_50cm.tif", "_report.json", "_report.md"):
        assert (out / f"{stem}{suffix}").is_file(), suffix
    assert not (out / "ams3d_subtiles_in").exists() and not (out / "ams3d_subtiles_out").exists()
    assert report["n_points"] == len(xyz)
    # Each sub-tile segments its buffered extent and writes only its core points, so
    # the crown sitting on the seam at x = 40 gets one id per side (the same border
    # effect the ForestFormer3D split/merge pipeline has); the other two are whole.
    assert report["n_trees"] == 4
    assert report["ams3d"]["config"] == "C" and report["ams3d"]["workers"] == 2
    assert report["ams3d"]["n_subtiles"] == 2 and report["ams3d"]["buffer_m"] == 10.0
    assert len(report["ams3d"]["subtiles"]) == 2
    on_disk = json.loads((out / f"{stem}_report.json").read_text())
    assert on_disk["ams3d"]["params"] == CONFIGS["C"].as_dict()
    merged = laspy.read(str(out / f"{stem}.las"))
    assert len(merged.points) == len(xyz)
    ids = np.asarray(merged.treeID)
    assert np.unique(ids[ids >= 0]).size == 4
    mx = np.asarray(merged.x) - 381000.0
    my = np.asarray(merged.y) - 5829000.0
    for cx, cy in [(12.0, 12.0), (66.0, 28.0)]:
        inside = (np.hypot(mx - cx, my - cy) < 3.0) & (ids >= 0)
        assert np.unique(ids[inside]).size == 1
    seam = (np.hypot(mx - 40.0, my - 20.0) < 3.0) & (ids >= 0)
    assert np.unique(ids[seam]).size == 2
    assert set(np.unique(ids[seam & (mx < 40)])) != set(np.unique(ids[seam & (mx >= 40)]))
    assert any("workers" in line for line in logs)


# --- regression against the spike ------------------------------------------------

SPIKE_STATS = {
    # docs/experiments/ams3d_spike_r12_stats.json of the GEE_animation spike
    # (config ``default``, r12 patch E 381300-381400 / N 5828300-5828400 with a
    # 10 m buffer): trees whose apex lies in the patch and their mean convex-hull
    # crown diameter.
    "trees_in_patch": 658,
    "diam_mean": 5.398971439417245,
    "height_mean": 18.65521032826744,
}
KM_TILE = Path("/Volumes/2TB/winmol/ALS_Data/berlin_als_2021/3dm_33_381_5828_1_be.las")
LOCAL_TILE = Path.home() / "work/hnee/ForestFormer3D_runs/berlin_in/r12_tegel_E381300_N5828300_100m.las"
R12 = (381300.0, 5828300.0, 100.0)


def _apex_crowns(xyz, cls, ids, xmin, ymin, size):
    """(height, crown diameter) of every cluster whose apex lies inside the patch,
    computed exactly like the spike's ``crowns_from_labels``."""
    from shapely.geometry import MultiPoint

    from ff3d_geo.ams3d import normalize_height

    keep = np.isin(cls, (2, 3, 4, 5))
    h = np.full(len(cls), np.nan)
    h[keep] = normalize_height(xyz[keep], cls[keep])
    rows = []
    for lab in np.unique(ids[ids >= 0]):
        m = ids == lab
        pts = xyz[m]
        top = np.argmax(h[m])
        ax, ay = pts[top, 0], pts[top, 1]
        if not (xmin <= ax < xmin + size and ymin <= ay < ymin + size):
            continue
        hull = MultiPoint(pts[:, :2]).convex_hull
        rows.append((float(h[m].max()), float(2 * np.sqrt(hull.area / np.pi))))
    return np.array(rows)


@pytest.mark.slow
def test_regression_r12_patch_matches_the_spike():
    """Rerun the port on the spike's r12 patch with the spike's ``default`` config.

    With the km tile present the patch is cut with the spike's 10 m buffer, so the
    numbers must agree to within 1 % (only floating-point ordering differs). With
    just the local 100 m tile (no buffer) edge crowns are cut, so a looser 8 % on
    the count and 5 % on the mean diameter is asked.
    """
    xmin, ymin, size = R12
    if KM_TILE.is_file():
        las = laspy.read(str(KM_TILE))
        x, y, z = np.asarray(las.x), np.asarray(las.y), np.asarray(las.z)
        cls = np.asarray(las.classification)
        m = (x >= xmin - 10) & (x < xmin + size + 10) & (y >= ymin - 10) & (y < ymin + size + 10)
        xyz = np.column_stack([x[m], y[m], z[m]])
        cls = cls[m]
        count_tol, diam_tol = 0.01, 0.01
    elif LOCAL_TILE.is_file():
        las = laspy.read(str(LOCAL_TILE))
        xyz = np.column_stack([np.asarray(las.x) + xmin, np.asarray(las.y) + ymin, np.asarray(las.z)])
        cls = np.asarray(las.classification)
        count_tol, diam_tol = 0.08, 0.05
    else:
        pytest.skip(f"neither {KM_TILE} nor {LOCAL_TILE} is present")

    ids = segment_ams3d(xyz, cls, CONFIGS["default"], threads=-1)
    crowns = _apex_crowns(xyz, cls, ids, xmin, ymin, size)
    n_trees, diam_mean = len(crowns), float(crowns[:, 1].mean())
    assert abs(n_trees - SPIKE_STATS["trees_in_patch"]) <= count_tol * SPIKE_STATS["trees_in_patch"], (
        n_trees, SPIKE_STATS["trees_in_patch"])
    assert abs(diam_mean - SPIKE_STATS["diam_mean"]) <= diam_tol * SPIKE_STATS["diam_mean"], (
        diam_mean, SPIKE_STATS["diam_mean"])
