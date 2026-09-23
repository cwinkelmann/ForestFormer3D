# Phase 5: Seamless tree ids across sub-tiles (core + halo inference, overlap stitching) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove the 100 m sub-tile seams from the Berlin km-tile results: no under-segmented strip along the grid lines, no tree cut into two ids by a line, one id namespace over the whole mosaic (km-tile borders included), with the Potree viewer showing hashed colours.

**Architecture:** `split` writes core-plus-halo sub-tiles (100 m core, 20 m halo from the km tile and its neighbours) and records every point's `(source tile, point index)` in a sidecar. `run` is untouched. A new `stitch` replaces `merge`: each point takes its label from the sub-tile whose core contains it, instances of adjacent sub-tiles are unified with a union-find over their IoU on shared halo points, and every component gets a dense global id. `border-check` measures the seam metrics before and after. Production scripts run split+run per GPU and one mosaic-wide stitch at the end.

**Tech Stack:** numpy, laspy[lazrs], plyfile, shapely, pyproj, geopandas, pyogrio (all in `tests/requirements-cpu.txt`); pytest. GPU work runs on carrot through the existing `ff3d_geo run` docker steps.

**Spec:** `docs/superpowers/specs/2026-09-23-seamless-tree-ids-design.md` (measured numbers in §1, option B in §3).

## Global Constraints

- Branch `fix/review-findings`; one commit per task; commit messages carry NO trailers of any kind (no `Co-Authored-By`, no `Claude-Session`); commit with explicit pathspecs, never `git add -A` (untracked `segment_any_tree_gpu.docker`, `benchmark/sat_to_ff3d.py` and `.playwright-mcp/` stay untracked).
- CPU tests run on the Mac with `.venv-cpu/bin/python -m pytest -q tests`; optional packages via `pytest.importorskip` at module level; nothing under `tests/` may import torch or mmengine. Baseline before this plan: `tests/test_geo_merge.py tests/test_geo_split.py tests/test_geo_cli.py` = 54 passed.
- `ff3d_geo` never imports torch; the model side (`oneformer3d/`, `tools/test.py`, the config) is NOT modified by this plan. No GPU job is run from the Mac; carrot commands follow `.claude/skills/ff3d-carrot/SKILL.md` and `.claude/skills/ff3d-inference-km-tiles/SKILL.md`.
- Sub-tile naming stays `<prefix>_E<x>_N<y>_<size>m.las` with the CORE origin in the token (`ff3d_geo.origin.parse_origin`); stems must not end in `_<digits>` (`cli._UNSAFE_STEM = re.compile(r"_\d+$")`). Local coordinates are relative to the core origin, so halo points have `x` or `y` in `[-halo, 0)` or `[size, size + halo)`; LAS scale 0.001 / offset 0 (int32) stores negatives fine and `batch_load` centres by the mean anyway.
- Result LAS contract (`ff3d_geo.convert.results_to_las`): LAS 1.4 pf6, EPSG:25833 WKT, extra dims `treeID` int32 "ForestFormer3D instance, -1 none", `semantic` uint8 "0 ground 1 wood 2 leaf 255 n/a", `score` float32 "instance score" (descriptions included; laspy compares them). Points stay in input order and `results_to_las` verifies the count against the sidecar.
- `trees_to_gpkg(las, gpkg)`, `build_report(las, gpkg, runtime_s)`, `write_report`, `report_markdown`, `las_to_masks(las, out_dir, cell_m, prefix)` are reused unchanged on the stitched km-tile LAS.
- Existing behaviour must hold: `split_las(..., buffer_m=0)` writes byte-identical sub-tiles to today (the existing split tests stay green), `merge` keeps working on halo-0 outputs, and the single-tile `run` (r12) is untouched.
- Test data on the Mac: `/Volumes/2TB/winmol/ALS_Data/berlin_als_2021/<T>.las` (inputs), `/Volumes/2TB/winmol/ALS_Data/berlin_als_2021_ff3d/<T>/` (today's outputs), Potree site `/Volumes/2TB/winmol/ALS_Data/berlin_potree` (`benchmark/serve_potree.py --root ... --port 8080`; an instance may already be listening).
- **Concurrent work in the same checkout (2026-09-23).** Another session has UNCOMMITTED changes in `ff3d_geo/split.py` (a `buffer_m` parameter: the primary tile's points within `buffer_m` outside each core are added to the sub-tile, local coordinates `-buffer_m..size_m+buffer_m`, `min_points` on core points only), `ff3d_geo/cli.py` (`split --buffer`, new `ams3d` and `buildings` subcommands), `ff3d_geo/report.py`, `benchmark/build_potree_site.py`, `benchmark/potree_index.html` (a method switcher) and untracked `ff3d_geo/ams3d.py`, `ff3d_geo/buildings.py`, `tests/test_geo_ams3d.py`. This plan BUILDS ON the `buffer_m` halo (that is the halo; do not add a second parameter) and must not commit, revert or reformat that session's work: run `git status` before every commit and stage only the files each task names. If `buffer_m` has been committed by then, fine; if it is still uncommitted when Task 1 starts, ask before committing `ff3d_geo/split.py` (the commit would carry the other session's hunk).

---

### Task 0: Viewer colours (done)

Commit `afac64e` ("fix: Potree tree-id colours are hashed, not positional") replaced the positional slot hash in `benchmark/potree_index.html` with a lowbias32 mixer, made tree markers use their tree's slot colour, and synced the site copy. Nothing to do here except keep the two copies identical whenever the viewer is touched again (`diff benchmark/potree_index.html /Volumes/2TB/winmol/ALS_Data/berlin_potree/index.html`).

---

### Task 1: `ff3d_geo/split.py` — halo, point identity sidecar, manifest

**Files:**
- Modify: `ff3d_geo/split.py`
- Test: `tests/test_geo_split.py` (extend)

**Interfaces:**
- `IDENT_DTYPE = np.dtype([("tile", "<u2"), ("index", "<u4")])` (module constant, also re-exported by `ff3d_geo.stitch`).
- `split_las(las_path, out_dir, size_m: int = 100, min_points: int = 1000, prefix: str | None = None, buffer_m: float = 0.0, neighbours: Sequence[Path] = ()) -> list[Path]`.
  - Grid and core membership exactly as today (`subtile_origins` from the primary's header bounds; every primary point lands in exactly one core; the conservation check stays).
  - With `buffer_m > 0` a point also goes to every sub-tile whose expanded box `[x0 - halo, x0 + size + halo) x [y0 - halo, y0 + size + halo)` contains it (up to 4 sub-tiles, 9 at corners). Halo points of the primary and of each neighbour are written the same way; a neighbour file is read in chunks and only its points within `buffer_m` of the primary grid's outer bounds are considered.
  - `min_points` counts CORE points only.
  - Per written sub-tile: `<out>/<stem>_ident.npy`, an `IDENT_DTYPE` array aligned with the sub-tile's point order; `tile` is 0 for the primary and `1 + i` for `neighbours[i]`, `index` the 0-based position in that source file's point order (chunk iteration order).
  - `<out>/split_manifest.json`: `{"size_m", "buffer_m", "prefix", "sources": [{"key", "path", "n_points"}], "subtiles": [{"stem", "origin": [x0, y0], "source": 0, "n_points", "n_core"}]}` where `key` is the source stem with a trailing `_1_be` stripped (`_default_prefix`) and `path` absolute. Written with `buffer_m == 0` too (then `n_core == n_points`).
- `subtile_origins` unchanged.

- [ ] **Step 1: Write the failing tests** (append to `tests/test_geo_split.py`; `_write_las` there is `geo_fixtures.write_grid_las`, which truncates `xyz` in place to LAS precision, so comparisons against `xyz` are exact to 1e-3)

```python
import json

from ff3d_geo.split import IDENT_DTYPE


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
```

- [ ] **Step 2: Run them to verify they fail**

Run: `.venv-cpu/bin/python -m pytest -q tests/test_geo_split.py`
Expected: the four new tests FAIL (`ImportError: cannot import name 'IDENT_DTYPE'`), the six existing ones pass.

- [ ] **Step 3: Implement**

In `split_las`: keep the existing chunk loop for the primary but replace the per-origin exact-cell mask by a box mask on the expanded box when `buffer_m > 0` (`(x >= x0 - halo) & (x < x0 + size + halo) & (y >= ...)`); count `core` with the existing exact-cell mask (`ex == origin[0]`) and `n_points` per sub-tile separately; collect `ident` pieces per origin as `np.empty(n_sel, IDENT_DTYPE)` with `tile = 0`, `index = chunk_start + np.flatnonzero(mask)` (track `chunk_start` across `chunk_iterator`). Then loop over `neighbours` with the same chunked reader, `tile = 1 + i`, skipping chunks with no point inside the primary grid's expanded outer bounds. The conservation check compares the primary's core counts with `header.point_count`, unchanged. After the writers close: rename the sub-tiles with `counts_core >= min_points`, `np.save(<stem>_ident.npy, np.concatenate(pieces))` for those, unlink the temp files and drop the pieces of the others; write the manifest last (the manifest's presence means the split is complete). Keep the `.tmp` cleanup on failure paths and add the ident files of an aborted split to it.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv-cpu/bin/python -m pytest -q tests/test_geo_split.py tests/test_geo_cli.py`
Expected: 10 + 42 passed (the CLI's split round trip must be unaffected by the extra files).

- [ ] **Step 5: Commit**

```bash
git add ff3d_geo/split.py tests/test_geo_split.py
git commit -m "feat: split writes core-plus-halo sub-tiles with a point identity sidecar and manifest"
```

---

### Task 2: `ff3d_geo/stitch.py` — overlap matching, union-find, global ids, km-tile LAS

**Files:**
- Create: `ff3d_geo/stitch.py`
- Modify: `ff3d_geo/convert.py` (factor `result_point_header`)
- Test: `tests/test_geo_stitch.py`, `tests/test_geo_convert.py` (one assertion)

**Interfaces:**
- `convert.result_point_header(epsg: int, scales, offsets) -> laspy.LasHeader`: LAS 1.4 pf6 with the three extra dims and descriptions exactly as `results_to_las` writes them plus the CRS; `results_to_las` calls it (refactor, no behaviour change).
- `stitch.ident_keys(ident: np.ndarray, tile_map: np.ndarray) -> np.ndarray`: int64 `global_tile << 32 | index` where `tile_map[local tile] = global tile index` (the manifest's source order mapped onto the mosaic's sorted source keys).
- `stitch.match_instances(keys_a, labels_a, keys_b, labels_b, iou_threshold: float = 0.5, min_shared: int = 20) -> list[tuple[int, int, float, int]]`: over the keys present in both arrays (`np.intersect1d(..., return_indices=True)`), for every pair `(la, lb)` of non-negative labels co-occurring on shared points, `iou = n_ab / (n_a + n_b - n_ab)` with `n_a`, `n_b` the label sizes ON THE SHARED POINTS; return `(la, lb, iou, n_ab)` for pairs with `n_ab >= min_shared` and `iou >= iou_threshold`, sorted by `-iou`.
- `stitch.UnionFind` with `find(node) -> node`, `union(a, b) -> bool` (path compression; the root is the smaller node tuple, so results are deterministic).
- `stitch.Mosaic` (dataclass): `size_m`, `buffer_m`, `sources: list[dict]` (key, path, n_points; sorted by key, unique), `subtiles: list[dict]` (stem, origin, source key, n_points, n_core, manifest dir, tile_map). `stitch.load_mosaic(manifest_paths) -> Mosaic` merges manifests, requiring equal `size_m`/`buffer_m` and rejecting a stem listed twice or a source key with two different point counts.
- `stitch.adjacent_pairs(mosaic) -> list[tuple[str, str]]`: unordered stem pairs whose core origins differ by exactly one `size_m` step in x and/or y (8-neighbourhood), each once, sorted.
- `stitch.stitch(manifests, results_dirs, out_dir, iou_threshold=0.5, min_shared=20, runtime_s=None) -> dict`:
  1. `load_mosaic`; find each sub-tile's result as `<dir>/<stem>.las` in `results_dirs` (first hit; a missing one raises `FileNotFoundError` naming the stem).
  2. Load per sub-tile (cached, LRU of 32 by stem): `keys` (int64), `labels` (int32 `treeID`), `semantic`, `score`, `core` (bool: `x, y` inside the core box, computed from UTM coordinates and the core origin), checking `len(las) == n_points` from the manifest and `== len(ident)`.
  3. For every adjacent pair, `match_instances` on the full key arrays; union `(stem_a, la)` with `(stem_b, lb)` for every returned pair. Count `n_pairs_tested`, `n_unified`, `n_cross_km` (pairs whose sub-tiles have different source keys).
  4. Global ids: nodes = every `(stem, label >= 0)` seen in any sub-tile's CORE points; components by `find`; component ordering by `mix64(hash of root)` (`(root_index * 0x9E3779B97F4A7C15) & (2**63 - 1)` over the root's rank in sorted order is fine), dense ids `0..N-1` in that order. Write `<out>/stitch_ids.npy` (structured: `gid u4, stem U64, local i4` for each node) and `<out>/stitch.json` with the counts, `n_trees = N`, thresholds and the manifests used.
  5. Per source km tile `T` (`source["path"]`, stem `S` e.g. `3dm_33_381_5829_1_be`): arrays `treeID = -1 (int32)`, `semantic = 255 (uint8)`, `score = -1.0 (float32)` of `n_points`; for each sub-tile of that source, core points write their (globally mapped) values at `index`. Then write `<out>/<S>.las` from the source file in chunks (`chunk_iterator(2_000_000)`, copying `x y z classification` and `intensity/return_number/number_of_returns` when present) with the header from `result_point_header(25833, [0.001]*3, floor(mins))`, through a `.tmp` rename. Then `trees_to_gpkg(<S>.las, <S>_trees.gpkg)` and `build_report(...)`/`write_report(<S>_report.json, <S>_report.md)` with `runtime_s`.
  6. Return `{"n_trees", "n_pairs_tested", "n_unified", "n_cross_km", "tiles": {S: {"las", "gpkg", "n_points", "n_trees_in_tile"}}}`.
- Any core point whose sub-tile was dropped by `min_points` keeps `-1 / 255 / -1.0` (documented; today such points are absent from the merged LAS altogether).

- [ ] **Step 1: Write the failing tests**

`tests/test_geo_stitch.py`:

```python
import json

import numpy as np
import pytest

laspy = pytest.importorskip("laspy")
pytest.importorskip("shapely")
pytest.importorskip("pyproj")
gpd = pytest.importorskip("geopandas")
pytest.importorskip("pyogrio")

from ff3d_geo.split import IDENT_DTYPE
from ff3d_geo.stitch import (UnionFind, adjacent_pairs, ident_keys, load_mosaic,
                             match_instances, stitch)
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
```

Add to `tests/test_geo_convert.py`:

```python
def test_result_point_header_matches_results_to_las():
    from ff3d_geo.convert import result_point_header
    h = result_point_header(25833, [0.001] * 3, [0.0, 0.0, 0.0])
    assert h.point_format.id == 6 and str(h.version) == "1.4"
    assert [(d.name, d.description) for d in h.point_format.extra_dimensions] == [
        ("treeID", "ForestFormer3D instance, -1 none"),
        ("semantic", "0 ground 1 wood 2 leaf 255 n/a"),
        ("score", "instance score")]
    assert h.parse_crs().to_epsg() == 25833
```

- [ ] **Step 2: Run them to verify they fail**

Run: `.venv-cpu/bin/python -m pytest -q tests/test_geo_stitch.py tests/test_geo_convert.py`
Expected: `ModuleNotFoundError: No module named 'ff3d_geo.stitch'`; the convert test fails on the import.

- [ ] **Step 3: Implement** `ff3d_geo/stitch.py` per the interfaces (pure numpy for the matching: `np.unique(np.stack([la, lb]), axis=1, return_counts=True)` over shared points with both labels `>= 0`; label sizes over the shared set with `np.unique(..., return_counts=True)`), and the `result_point_header` refactor in `convert.py`. The km-tile write goes through `laspy.open(tmp, mode="w", header=...)` and `ScaleAwarePointRecord.zeros` per chunk like `split.py` does; `treeID` etc. are assigned per chunk from the full arrays by slice.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv-cpu/bin/python -m pytest -q tests/test_geo_stitch.py tests/test_geo_convert.py tests/test_geo_merge.py`
Expected: all passed (7 new in stitch, 1 in convert, merge unchanged).

- [ ] **Step 5: Commit**

```bash
git add ff3d_geo/stitch.py ff3d_geo/convert.py tests/test_geo_stitch.py tests/test_geo_convert.py
git commit -m "feat: ff3d_geo.stitch unifies sub-tile instances over halo overlaps into dense mosaic-wide ids"
```

---

### Task 3: `ff3d_geo/border.py` + `border-check` — the seam metrics

**Files:**
- Create: `ff3d_geo/border.py`
- Modify: `ff3d_geo/cli.py` (subcommand `border-check`)
- Test: `tests/test_geo_border.py`, `tests/test_geo_cli.py` (parser test)

**Interfaces:**
- `border.no_instance_profile(x, y, tree_id, vegetation: np.ndarray, size_m: float = 100.0, bin_m: float = 0.5, max_m: float = 10.0) -> dict`: distance of each point to the nearest multiple of `size_m` in x or y; returns `{"bins": [lo...], "frac": [...], "n": [...], "interior_frac": float, "strip_excess_pp": float}` where `interior_frac` is the `tree_id == -1` fraction of vegetation points `>= max_m` from any line and `strip_excess_pp` is `100 * (frac(< max_m) - interior_frac)`.
- `border.split_pairs(x, y, tree_id, size_m=100.0, tol_m=1.0) -> dict`: per-instance extents (`np.minimum.at`/`np.maximum.at`), `n_trees`, `n_touching` (an extent edge within `tol_m` of a line), `n_crossing` (extent spans a line), `n_pairs` (greedy pairing of instances ending at a line with instances starting across the same line whose perpendicular extents overlap, best overlap first, each instance used once), `touching_frac`.
- `border.border_check(las_path, size_m=100.0) -> dict`: reads the LAS once (`x, y, treeID, classification, semantic`), `vegetation = isin(classification, [3,4,5]) & (semantic != 0) & (semantic != 255)`, returns both dicts merged plus `n_points`.
- CLI: `python -m ff3d_geo border-check --las <T>.las [--size 100] [--json <path>]` prints the profile table and the counts, writes JSON when asked.

- [ ] **Step 1: Write the failing tests**

`tests/test_geo_border.py`:

```python
import numpy as np
import pytest

laspy = pytest.importorskip("laspy")
pytest.importorskip("pyproj")

from ff3d_geo.border import border_check, no_instance_profile, split_pairs
from geo_fixtures import write_result_las


def test_no_instance_profile_measures_the_strip_excess():
    rng = np.random.default_rng(0)
    x, y = rng.uniform(0, 300, 60000), rng.uniform(0, 300, 60000)
    d = np.minimum(np.minimum(x % 100, 100 - x % 100), np.minimum(y % 100, 100 - y % 100))
    tid = np.where(d < 2.0, -1, 5).astype(np.int32)              # unlabelled only inside 2 m of a line
    prof = no_instance_profile(x, y, tid, np.ones_like(tid, bool), size_m=100, bin_m=0.5, max_m=10)
    assert prof["bins"][:3] == [0.0, 0.5, 1.0] and len(prof["bins"]) == 20
    assert prof["frac"][0] == 1.0 and prof["frac"][3] == 1.0 and prof["frac"][4] == 0.0
    assert prof["interior_frac"] == 0.0 and 15 < prof["strip_excess_pp"] < 25


def test_split_pairs_counts_crossing_touching_and_paired_fragments():
    rng = np.random.default_rng(1)
    pts, ids = [], []
    for tid, cx, cy in ((0, 50, 50), (1, 97, 50), (2, 103, 50), (3, 50, 98), (4, 50, 103), (5, 150, 20)):
        p = np.column_stack([cx + rng.uniform(-3, 3, 200), cy + rng.uniform(-3, 3, 200)])
        pts.append(p)
        ids.append(np.full(200, tid))
    p, ids = np.vstack(pts), np.concatenate(ids)
    ids[(ids == 1) & (p[:, 0] >= 100)] = 2      # ids 1/2 are one crown cut at x = 100
    ids[(ids == 2) & (p[:, 0] < 100)] = 1
    ids[(ids == 3) & (p[:, 1] >= 100)] = 4
    ids[(ids == 4) & (p[:, 1] < 100)] = 3
    r = split_pairs(p[:, 0], p[:, 1], ids.astype(np.int32), size_m=100, tol_m=1.0)
    assert r["n_trees"] == 6 and r["n_crossing"] == 0
    assert r["n_touching"] == 4 and r["n_pairs"] == 2


def test_border_check_reads_a_result_las(tmp_path):
    rng = np.random.default_rng(2)
    n = 5000
    x, y = 381000 + rng.uniform(0, 200, n), 5829000 + rng.uniform(0, 100, n)
    tid = np.where(np.abs(x - 381100) < 1.5, -1, (x > 381100).astype(np.int32))
    las = write_result_las(tmp_path / "t.las", x, y, rng.uniform(1, 20, n), tid,
                           np.full(n, 2, np.uint8))
    las_data = laspy.read(las)
    las_data.classification = np.full(n, 5, np.uint8)
    las_data.write(las)
    r = border_check(las, size_m=100)
    assert r["n_points"] == n and r["n_trees"] == 2 and r["n_crossing"] == 0
    assert r["frac"][0] == 1.0 and r["interior_frac"] == 0.0
```

Add to `tests/test_geo_cli.py`:

```python
def test_parser_exposes_border_check():
    args = build_parser().parse_args(["border-check", "--las", "t.las"])
    assert args.size == 100 and args.json is None
```

- [ ] **Step 2: Run to verify they fail** — `.venv-cpu/bin/python -m pytest -q tests/test_geo_border.py tests/test_geo_cli.py -k "border"`; expected `ModuleNotFoundError` / `SystemExit` on the unknown subcommand.

- [ ] **Step 3: Implement** `ff3d_geo/border.py` (numpy only, laspy for `border_check`) and the lazy-import subcommand in `cli.main`.

- [ ] **Step 4: Run** `.venv-cpu/bin/python -m pytest -q tests/test_geo_border.py tests/test_geo_cli.py` — all passed. Then reproduce today's numbers as the baseline (this is the check that the metric is right):

```bash
.venv-cpu/bin/python -m ff3d_geo border-check \
  --las /Volumes/2TB/winmol/ALS_Data/berlin_als_2021_ff3d/3dm_33_381_5829_1_be/3dm_33_381_5829_1_be.las
```

Expected (from the spec §1b, tolerance ±0.005 / ±20 pairs): `frac[0] = 0.374`, `interior_frac = 0.124`, `strip_excess_pp ≈ 6.3`, `n_trees = 31385`, `n_crossing = 0`, `n_touching = 5150`, `n_pairs ≈ 1843`.

- [ ] **Step 5: Commit**

```bash
git add ff3d_geo/border.py ff3d_geo/cli.py tests/test_geo_border.py tests/test_geo_cli.py
git commit -m "feat: ff3d_geo border-check measures the sub-tile seam metrics"
```

---

### Task 4: CLI — `split --buffer/--neighbours`, `stitch` subcommand

**Files:**
- Modify: `ff3d_geo/cli.py` (`build_parser`, `main`)
- Test: `tests/test_geo_cli.py` (extend)

**Interfaces:**
- `split` gains `--buffer M` (float, default 0) and `--neighbours <las>...` (default none); prints the written sub-tile paths as before.
- `stitch --manifest <split_manifest.json>... --results <dir>... --out <dir> [--iou 0.5] [--min-shared 20] [--runtime-s S]` runs `stitch.stitch` and prints one line per km tile (`wrote <S>.las (<n_points> points, <n> trees in tile)`) and one summary line (`unified <n_unified> of <n_pairs_tested> tested pairs (<n_cross_km> across km tiles); <n_trees> trees in the mosaic`), then the report markdown for each tile.
- `merge` is unchanged.

- [ ] **Step 1: Write the failing tests** (append to `tests/test_geo_cli.py`; the stitch fixture is importable as `from test_geo_stitch import _mosaic`)

```python
def test_parser_exposes_split_halo_and_stitch():
    parser = build_parser()
    args = parser.parse_args(["split", "--las", "a.las", "--out", "sub", "--buffer", "20",
                              "--neighbours", "b.las", "c.las"])
    assert args.buffer == 20.0 and [p.name for p in args.neighbours] == ["b.las", "c.las"]
    args = parser.parse_args(["stitch", "--manifest", "m.json", "--results", "r1", "r2", "--out", "o"])
    assert [p.name for p in args.manifest] == ["m.json"] and len(args.results) == 2
    assert args.iou == 0.5 and args.min_shared == 20 and args.runtime_s is None


def test_stitch_subcommand_writes_km_tiles_and_prints_the_summary(tmp_path, capsys):
    from test_geo_stitch import _mosaic

    manifest, res, xyz, _ = _mosaic(tmp_path)
    out = tmp_path / "out"
    rc = main(["stitch", "--manifest", str(manifest), "--results", str(res), "--out", str(out)])
    assert rc == 0
    printed = capsys.readouterr().out
    assert f"wrote {out / '3dm_33_381_5829_1_be.las'} ({len(xyz)} points, 3 trees in tile)" in printed
    assert "unified 1 of 1 tested pairs (0 across km tiles); 3 trees in the mosaic" in printed
    assert (out / "3dm_33_381_5829_1_be_trees.gpkg").is_file()


def test_split_subcommand_passes_halo_and_neighbours_through(tmp_path, capsys):
    from geo_fixtures import write_grid_las

    rng = np.random.default_rng(3)
    a = np.column_stack([381000 + rng.uniform(0, 100, 2000), 5829000 + rng.uniform(0, 100, 2000), rng.uniform(1, 9, 2000)])
    b = np.column_stack([381100 + rng.uniform(0, 100, 2000), 5829000 + rng.uniform(0, 100, 2000), rng.uniform(1, 9, 2000)])
    src = write_grid_las(tmp_path / "3dm_33_381_5829_1_be.las", a, np.full(2000, 5, np.uint8))
    nb = write_grid_las(tmp_path / "3dm_33_382_5829_1_be.las", b, np.full(2000, 5, np.uint8))
    out = tmp_path / "sub"
    assert main(["split", "--las", str(src), "--out", str(out), "--buffer", "20",
                 "--neighbours", str(nb), "--min-points", "10"]) == 0
    ident = np.load(out / "3dm_33_381_5829_E381000_N5829000_100m_ident.npy")
    assert (ident["tile"] == 1).sum() == int((b[:, 0] < 381120).sum())
```

- [ ] **Step 2: Run to verify they fail** — `.venv-cpu/bin/python -m pytest -q tests/test_geo_cli.py -k "halo or stitch"`.

- [ ] **Step 3: Implement** (lazy import of `ff3d_geo.stitch` in `main`, like `merge`).

- [ ] **Step 4: Run** `.venv-cpu/bin/python -m pytest -q tests` — everything passes (system-python `python3 -m pytest tests` still skips the geo modules cleanly).

- [ ] **Step 5: Commit**

```bash
git add ff3d_geo/cli.py tests/test_geo_cli.py
git commit -m "feat: split --buffer/--neighbours and the stitch subcommand"
```

---

### Task 5: Production scripts, skills and pipeline doc

**Files:**
- Modify: `benchmark/berlin_run_gpu.sh` (split with halo and neighbours; no merge/masks), `docs/inference-pipeline.md` ("km tiles" section), `.claude/skills/ff3d-inference-km-tiles/SKILL.md` (sections 2, 5, 6), `.claude/skills/ff3d-outputs-and-viewers/SKILL.md` (section 1: ids are mosaic-wide and dense; `border-check`)
- Create: `benchmark/berlin_stitch.sh`
- Test: `tests/test_inference_script.py` already checks tracked scripts parse with `bash -n`; extend it to cover the two scripts (`subprocess.run(["bash", "-n", path])`).

**Interfaces:**
- `benchmark/berlin_run_gpu.sh <gpu> <tile stem>...`: per tile `T`, neighbours are the existing files among `inputs/berlin/3dm_33_<E±1>_<N±1>_1_be.las` (a bash loop over the eight offsets; `T` parsed with `IFS=_`), then
  `python -m ff3d_geo split --las inputs/berlin/$T.las --out inputs/berlin/sub/$T --buffer 20 --neighbours $NB > work_dirs/logs/split-$T.txt`, then the unchanged `run`. The `merge` and `masks` lines are removed; the log milestone `tile done (run ${R}s)` stays and the run time is also appended to `work_dirs/logs/runtime-$T.txt` (one integer) for the stitch script.
- `benchmark/berlin_stitch.sh <tile stem>...`: one `python -m ff3d_geo stitch --manifest inputs/berlin/sub/$T/split_manifest.json ... --results work_dirs/berlin-$T ... --out work_dirs/berlin-mosaic --runtime-s <sum of runtime-$T.txt>` over ALL given tiles, then `python -m ff3d_geo masks --las work_dirs/berlin-mosaic/$T.las --out work_dirs/berlin-mosaic --prefix $T` and `python -m ff3d_geo border-check --las work_dirs/berlin-mosaic/$T.las --json work_dirs/berlin-mosaic/${T}_border.json` per tile.
- Docs: replace the "Trees cut by a sub-tile border stay split" paragraph with the halo/stitch description and the id scheme; the skill's section 5 (recovery) gets "stitch needs every sub-tile result of every manifest; a missing one is named"; section 6 (residue) unchanged; section 7 (copy home) adds `work_dirs/berlin-mosaic/`.

- [ ] **Step 1: Test** — in `tests/test_inference_script.py` add a parametrised `bash -n` check over `benchmark/berlin_run_gpu.sh` and `benchmark/berlin_stitch.sh`, and a grep-style assertion that `berlin_run_gpu.sh` no longer contains `ff3d_geo merge` and does contain `--buffer 20`.
- [ ] **Step 2: Implement** the scripts and doc edits.
- [ ] **Step 3: Run** `.venv-cpu/bin/python -m pytest -q tests/test_inference_script.py` and `bash -n` both scripts by hand.
- [ ] **Step 4: Commit**

```bash
git add benchmark/berlin_run_gpu.sh benchmark/berlin_stitch.sh docs/inference-pipeline.md \
        .claude/skills/ff3d-inference-km-tiles/SKILL.md .claude/skills/ff3d-outputs-and-viewers/SKILL.md \
        tests/test_inference_script.py
git commit -m "tools: per-GPU queue splits with a 20 m halo; mosaic-wide stitch script; docs"
```

---

### Task 6: Validation on carrot — one km tile, then the mosaic, report, viewer rebuild

**Files:**
- Create: `docs/benchmarks/2026-09-24-seamless-ids.md`
- Modify: `docs/benchmarks/2026-09-22-tegel-berlin-2021.md` (one pointer paragraph in §3), `docs/superpowers/plans/2026-09-23-ff3d-05-seamless-ids.md` (tick the boxes)

- [ ] **Step 1: One km tile.** On carrot (`ssh carrot`, `cd /raid/cwinkelmann/ForestFormer3D && git pull --ff-only origin fix/review-findings`), with the seven available neighbours of `3dm_33_381_5829_1_be` present in `inputs/berlin/` (`380_5828, 380_5829, 381_5828, 381_5830, 382_5828, 382_5829` — `380_5830` and `382_5830` do not exist), on one idle GPU (never GPU 1):

```bash
nohup bash benchmark/berlin_run_gpu.sh 5 3dm_33_381_5829_1_be > work_dirs/logs/berlin-halo-gpu5-$(date +%Y%m%d-%H%M%S).log 2>&1 &
# expect ~1.96x today's per-sub-tile time (sub-tiles hold 140 m x 140 m); poll every few minutes
```

Then `bash benchmark/berlin_stitch.sh 3dm_33_381_5829_1_be` (only this tile's manifest: the km borders get halo context from the neighbours' raw points, cross-km unification is exercised in Step 3). Record from `work_dirs/berlin-mosaic/`: `stitch.json`, `3dm_33_381_5829_1_be_border.json`, the report.

Acceptance (spec §3): `strip_excess_pp` from 6.3 to below 1.0; `frac[0]` within 0.03 of `interior_frac`; `n_pairs` from 1843 to below 100; `n_touching / n_trees` from 16.4 % to 6-10 %; `n_crossing > 0` (crowns now cross lines); `n_trees` within ±5 % of 31,385 minus the ~1,843 duplicates (i.e. ~29,500). If `strip_excess_pp` stays above 1.0, print the profile: an excess confined to the outer bins means the halo is too small (raise `--buffer` to 24 or 32 and rerun); an excess flat over 10 m means something else (stop and report).

- [ ] **Step 2: r12 regression.** `python -m ff3d_geo run --las inputs/r12_tegel_E381300_N5828300_100m.las --checkpoint work_dirs/clean_forestformer/epoch_3000_fix.pth --out work_dirs/tegel-r12-check --gpu 5`; the report must say 397 CHM maxima and a tree count within run-to-run noise of 402 (`docs/benchmarks/2026-09-22-tegel-als.md`; the noise floor is in `docs/benchmarks/2026-09-23-inference-profile.md`). No halo is involved, so any difference is nondeterminism, not this plan.

- [ ] **Step 3: The mosaic.** Split+run all eleven tiles (queues per GPU as in the skill, 1-2 tiles per GPU), then ONE `berlin_stitch.sh` over all eleven. Check `stitch.json`: `n_cross_km > 0`; for one cross-km pair from `stitch_ids.npy`, confirm the id appears in both km-tile LAS files. Copy `work_dirs/berlin-mosaic/` home (`rsync` as in the skill, section 7) to `/Volumes/2TB/winmol/ALS_Data/berlin_als_2021_ff3d_v2/<T>/` (one directory per tile: `<T>.las`, `_trees.gpkg`, `_crowns.gpkg`, the GeoTIFFs, `_report.*`, `_border.json`).

- [ ] **Step 4: Viewer.** Rebuild the octrees with PotreeConverter 2.1.1 from the new LAS files into a new site directory (`berlin_potree_v2`; keep the old one), run `benchmark/build_potree_site.py --site ... --ff3d-dir .../berlin_als_2021_ff3d_v2 ...`, serve it (`benchmark/serve_potree.py --root ... --port 8081`), and take a headless screenshot of a tile boundary in `tree id` mode: no colour bands (Task 0) and crowns continuing across the 100 m lines.

- [ ] **Step 5: Report** `docs/benchmarks/2026-09-24-seamless-ids.md`: the before/after table of the four metrics for `3dm_33_381_5829_1_be` (today's numbers from spec §1b), the mosaic totals (trees before: sum of the eleven tables, 300,452; after: `n_trees`; `n_unified`, `n_cross_km`), runtime per sub-tile with and without halo, and the screenshot pair under `docs/benchmarks/assets/seamless/`. Add one paragraph to `2026-09-22-tegel-berlin-2021.md` §3 pointing to it. Commit:

```bash
git add docs/benchmarks/2026-09-24-seamless-ids.md docs/benchmarks/assets/seamless \
        docs/benchmarks/2026-09-22-tegel-berlin-2021.md docs/superpowers/plans/2026-09-23-ff3d-05-seamless-ids.md
git commit -m "docs: seamless tree ids validated on the Berlin mosaic (border strip, split crowns, cross-km ids)"
```

---

## Notes for the executor

- **Cost.** A 20 m halo on a 100 m core is `140^2 / 100^2 = 1.96x` the points per sub-tile and, since `_predict_full_plot` cost is proportional to the cylinder count (`(140/4)^2` vs `(100/4)^2` lattice), about 2x the inference time per km tile (today's skill figure: "well under 20 s per sub-tile" uncontended, 92-99 min per km tile before the vectorisation commits). Stitch is host-only and minutes.
- **200 m cores** (`split --size 200 --buffer 20`, `1.44x`) are supported by the same code; the file token becomes `_200m`. Try after the 100 m validation passes; the risk is `_predict_full_plot`'s per-tile memory (votes and mask index lists scale with the tile's point count, ~4.6 M points at 200 m; the cylinder work does not).
- **Halo size.** The measured seam profile reaches the interior level at ~10 m from a line; the cylinder lattice's one-sided coverage extends 16 m. 20 m is the default; the halo should be a multiple of the lattice step (4 m at `region_step_factor 0.25`, 8 m at 0.5 — use 24 m there) so the core sees the same lattice as a stand-alone tile would.
- **8 m step (`region_step_factor=0.5`).** Independent of this plan: it is a per-tile model setting passed with `--cfg-options`, gives 3.4x per sample and costs ~0.05 F1 on the benchmark (`docs/benchmarks/2026-09-23-inference-profile.md`). With the halo, factor 0.5 lands at ~1.7x faster than today's seamed runs. The stitch does not depend on it.
- **Ids.** Dense `0..N-1` over the mosaic in hash order; re-stitching after adding tiles renumbers everything (the mapping is in `stitch_ids.npy`). A tree on a km border has one id in both km-tile LAS files; its tree-table row in each file covers only that file's points (documented in the skill).
- **Viewer.** With dense mosaic-wide ids each km tile's `[min, max]` spans most of `0..N`, so the 8192-slot LUT holds ~37 ids per slot for eleven tiles; ids sharing a slot are spatially random, which is what the hashed LUT was verified on. Do not switch to a `tile * 10^6 + local` scheme without also changing the viewer (see spec §3).
