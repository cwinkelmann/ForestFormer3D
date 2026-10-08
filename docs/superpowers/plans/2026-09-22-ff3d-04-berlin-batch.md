# Phase 4: Berlin ALS 2021 km tiles in batch (split, batched run, merge) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run the released ForestFormer3D checkpoint over the three Berlin ALS 2021 km tiles around Revier 12 Tegelsee (`3dm_33_380_5828_1_be`, `3dm_33_381_5828_1_be`, `3dm_33_381_5829_1_be`) with the `ff3d_geo` pipeline, producing one georeferenced LAS, one tree GeoPackage and one report per km tile.

**Architecture:** Three small additions to the pure-Python `ff3d_geo` package. `split.py` cuts a km tile into 100 m sub-tiles in local coordinates whose file names carry the origin token the existing CLI parses. `cli.run` accepts several `--las` files and runs ONE preprocess and ONE inference for all of them (one container, one model load, N result PLYs), then georeferences each. `merge.py` stitches the sub-tile LAS files and tree tables of one km tile back together with globally unique tree ids, and the existing report runs once per km tile. Trees on sub-tile borders are cut; that is documented, not solved.

**Tech Stack:** numpy, laspy[lazrs], plyfile, shapely, pyproj, geopandas, pyogrio (all in `tests/requirements-cpu.txt`); pytest. GPU work runs on carrot through the existing `ff3d_geo` docker steps.

**Spec:** `docs/superpowers/specs/2026-09-22-ff3d-fixes-benchmark-tegel-design.md` §6 (ff3d_geo) as amended by the Phase 3 rulings; this plan extends it. Design approved in chat on 2026-09-22 (user chose the three km tiles around Revier 12 and 100 m sub-tiles).

## Global Constraints

- Branch `fix/review-findings`; one commit per task; commit messages carry NO trailers of any kind (no `Co-Authored-By`, no `Claude-Session`); commit with explicit pathspecs (an untracked `segment_any_tree_gpu.docker` must stay untracked).
- CPU tests run on the Mac with `.venv-cpu/bin/python -m pytest -q`; optional packages via `pytest.importorskip` at module level; nothing under `tests/` may import torch or mmengine.
- Input tiles: LAS 1.4 point format 6, no CRS in the header, EPSG:25833, 1 km square, file name `3dm_33_<E km>_<N km>_1_be.las`; classes 2 ground, 3/4/5 vegetation.
- Sub-tile naming for the existing CLI: `<prefix>_E<easting>_N<northing>_100m.las` with integer metre origins (`ff3d_geo.origin.parse_origin`); the stem must not end in `_<digits>` (`cli._UNSAFE_STEM`), which `_100m` satisfies.
- `las_to_ply` expects LOCAL coordinates (0..100 m) and the sidecar `origin` restores UTM; `results_to_las(result_ply, sidecar, offsets_npy, out_las)` writes LAS 1.4 pf6 with extra dims `treeID` int32 (-1 none), `semantic` uint8 (255 nodata), `score` float32, EPSG:25833.
- `trees_to_gpkg(las, gpkg)` writes layer `trees` with columns `TREE_COLUMNS = ["tree_id", "x", "y", "top_z", "height", "crown_area_m2", "n_points", "mean_score"]` (point geometry, CRS from the LAS).
- `build_report(las, gpkg, runtime_s=None)` / `write_report(report, json, md)` / `report_markdown(report)` are unchanged.
- The CLI never touches `data/ForAINetV2/meta_data/test_list.txt`; the test info pkl goes to `<out>` (`create_data ... --test-list --splits test --out-dir`), and `tools/test.py` gets `--cfg-options test_dataloader.dataset.ann_file=<abs pkl>`.
- Carrot: checkout `/raid/cwinkelmann/ForestFormer3D`, host venv `/raid/cwinkelmann/ff3d-geo-venv`, image `forestformer3d:cu118`, GPU 5 only (GPUs 2-4 run Phase 2 jobs), inputs at `inputs/berlin/`.

---

### Task 1: `ff3d_geo/split.py` — km tile to 100 m sub-tiles

**Files:**
- Create: `ff3d_geo/split.py`
- Test: `tests/test_geo_split.py`

**Interfaces:**
- Produces: `split_las(las_path, out_dir, size_m: int = 100, min_points: int = 1000, prefix: str | None = None) -> list[Path]` and `subtile_origins(mins, maxs, size_m) -> list[tuple[int, int]]`.
- Sub-tile file: `<prefix>_E<x>_N<y>_<size_m>m.las` where `prefix` defaults to the input stem with a trailing `_1_be` removed and any other `_<digits>` suffix kept safe (e.g. `3dm_33_381_5829_1_be.las` -> `3dm_33_381_5829` -> `3dm_33_381_5829_E381000_N5829000_100m.las`). The stem must never end in `_<digits>`.
- Sub-tile contents: LAS 1.4 point format 6, `x - x0`, `y - y0` (local, 0..size_m), `z` unchanged, `classification` copied, `intensity`/return fields copied when present, scales 0.001, offsets 0; header CRS unset (the sidecar carries EPSG); origins are the integer-metre lower-left corners on a grid aligned to multiples of `size_m` starting at `floor(min_x / size_m) * size_m`.
- Sub-tiles with fewer than `min_points` points are skipped (not written); the function returns the written paths sorted.
- Memory: read the LAS in chunks (`laspy.open(...).chunk_iterator(2_000_000)`), bucket points by sub-tile index per chunk, and append to per-sub-tile writers (`laspy.open(path, mode="w", header=...)` kept open in a dict); never load the whole 23 M point tile into memory at once.

- [ ] **Step 1: Write the failing tests**

`tests/test_geo_split.py`:

```python
import numpy as np
import pytest

laspy = pytest.importorskip("laspy")

from ff3d_geo.origin import parse_origin
from ff3d_geo.split import split_las, subtile_origins


def _write_las(path, xyz, classification):
    header = laspy.LasHeader(point_format=6, version="1.4")
    header.scales = np.array([0.001, 0.001, 0.001])
    header.offsets = np.array([0.0, 0.0, 0.0])
    las = laspy.LasData(header)
    las.x, las.y, las.z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    las.classification = classification
    las.write(path)


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
```

- [ ] **Step 2: Run them to verify they fail**

Run: `.venv-cpu/bin/python -m pytest -q tests/test_geo_split.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'ff3d_geo.split'`

- [ ] **Step 3: Implement `ff3d_geo/split.py`**

Chunked reader, per-sub-tile writers, `min_points` filter applied at the end (write to a temporary name, then delete files below the threshold; or buffer per sub-tile up to the threshold before opening a writer — either is fine, the test only checks the outcome). Grid: `x0 = floor(min_x / size_m) * size_m`, `y0` likewise; index `ix = floor((x - x0) / size_m)`. The sub-tile header copies `point_format` 6 and `version` 1.4, sets `scales = [0.001]*3`, `offsets = [0, 0, 0]`. Copy `classification`, and `intensity`, `return_number`, `number_of_returns` when present in the source point format.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv-cpu/bin/python -m pytest -q tests/test_geo_split.py`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add ff3d_geo/split.py tests/test_geo_split.py
git commit -m "feat: ff3d_geo.split cuts a km tile into local-coordinate sub-tiles"
```

---

### Task 2: `ff3d_geo/merge.py` — sub-tile results back into one km tile

**Files:**
- Create: `ff3d_geo/merge.py`
- Test: `tests/test_geo_merge.py` (fixtures from `tests/geo_fixtures.py` where useful)

**Interfaces:**
- Produces: `merge_las(las_paths, out_las) -> dict` and `merge_trees(gpkg_paths, out_gpkg, id_offsets) -> int`.
- `merge_las`: concatenates the sub-tile result LAS files (each LAS 1.4 pf6 with extra dims `treeID` int32, `semantic` uint8, `score` float32, EPSG:25833 WKT) into ONE LAS with the same point format, extra dims and CRS; `treeID >= 0` values are re-numbered so they are unique across sub-tiles: sub-tile k's ids get `offset_k = sum(max_id_j + 1 for j < k)` added (ids of -1 stay -1); returns `{"n_points": int, "n_trees": int, "id_offsets": {path_str: offset_k}}`. Header scales 0.001, offsets = floor of the global minimum x/y/z. Read and write per file (do not hold all points twice).
- `merge_trees`: reads layer `trees` from each GeoPackage, adds `id_offsets[path_str]` to `tree_id`, concatenates in the same order, writes layer `trees` to `out_gpkg` (same columns/CRS); returns the row count. The `tree_id` set of the merged GeoPackage must equal the `treeID >= 0` set of the merged LAS when the inputs came from the same sub-tiles.

- [ ] **Step 1: Write the failing tests**

`tests/test_geo_merge.py`:

```python
import numpy as np
import pytest

laspy = pytest.importorskip("laspy")
pytest.importorskip("shapely")
pytest.importorskip("pyproj")
gpd = pytest.importorskip("geopandas")
pytest.importorskip("pyogrio")

from ff3d_geo.merge import merge_las, merge_trees
from ff3d_geo.trees import trees_to_gpkg


def _result_las(path, origin, tree_ids, semantic, n=200):
    rng = np.random.default_rng(int(origin[0]) % 997)
    header = laspy.LasHeader(point_format=6, version="1.4")
    header.scales = np.array([0.001] * 3)
    header.offsets = np.array([origin[0], origin[1], 0.0])
    header.add_extra_dim(laspy.ExtraBytesParams(name="treeID", type=np.int32))
    header.add_extra_dim(laspy.ExtraBytesParams(name="semantic", type=np.uint8))
    header.add_extra_dim(laspy.ExtraBytesParams(name="score", type=np.float32))
    header.add_crs(__import__("pyproj").CRS.from_epsg(25833))
    las = laspy.LasData(header)
    las.x = origin[0] + rng.uniform(0, 100, n)
    las.y = origin[1] + rng.uniform(0, 100, n)
    las.z = rng.uniform(0, 30, n)
    las.treeID = np.resize(np.asarray(tree_ids, np.int32), n)
    las.semantic = np.resize(np.asarray(semantic, np.uint8), n)
    las.score = np.full(n, 0.5, np.float32)
    las.write(path)
    return path


def test_merge_las_renumbers_tree_ids_across_subtiles(tmp_path):
    a = _result_las(tmp_path / "a.las", (381000, 5829000), [-1, 0, 1, 2], [0, 1, 2, 2])
    b = _result_las(tmp_path / "b.las", (381100, 5829000), [0, 0, 1, -1], [2, 2, 1, 0])
    out = tmp_path / "merged.las"
    info = merge_las([a, b], out)
    m = laspy.read(out)
    assert len(m.points) == 400 and info["n_points"] == 400
    assert info["id_offsets"] == {str(a): 0, str(b): 3}
    ids = np.asarray(m.treeID)
    assert set(ids[ids >= 0].tolist()) == {0, 1, 2, 3, 4} and info["n_trees"] == 5
    assert (ids == -1).sum() == 100
    assert m.header.parse_crs().to_epsg() == 25833
    assert {"treeID", "semantic", "score"} <= set(m.point_format.extra_dimension_names)
    assert m.x.min() >= 381000 and m.x.max() <= 381200


def test_merge_trees_matches_merged_las_ids(tmp_path):
    a = _result_las(tmp_path / "a.las", (381000, 5829000), [-1, 0, 1, 2], [0, 1, 2, 2])
    b = _result_las(tmp_path / "b.las", (381100, 5829000), [0, 0, 1, -1], [2, 2, 1, 0])
    ga, gb = tmp_path / "a.gpkg", tmp_path / "b.gpkg"
    trees_to_gpkg(a, ga)
    trees_to_gpkg(b, gb)
    info = merge_las([a, b], tmp_path / "merged.las")
    n = merge_trees([ga, gb], tmp_path / "merged.gpkg", info["id_offsets"])
    t = gpd.read_file(tmp_path / "merged.gpkg", layer="trees")
    assert n == 5 and sorted(t["tree_id"].tolist()) == [0, 1, 2, 3, 4]
    assert t.crs.to_epsg() == 25833
```

- [ ] **Step 2: Run them to verify they fail**

Run: `.venv-cpu/bin/python -m pytest -q tests/test_geo_merge.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'ff3d_geo.merge'`

- [ ] **Step 3: Implement `ff3d_geo/merge.py`**

`merge_las`: first pass reads each header (`laspy.open`) for point counts, mins and maxs and each file's `treeID` max (read the file once with `laspy.read`, compute `max_id`, keep the LasData only for the write pass if memory allows; the sub-tiles are ~1 M points so reading each once is fine). Build the output header from the first file (`laspy.LasHeader(point_format=6, version="1.4")`, copy the three extra dims via `add_extra_dim`, `add_crs(CRS.from_epsg(25833))` or copy the first file's CRS), scales 0.001, offsets = floor of global mins. Write with `laspy.open(out, mode="w", header=header)` and `writer.write_points(points)` per file after adding the id offset (`ids[ids >= 0] += offset`). `merge_trees`: `gpd.read_file(p, layer="trees", engine="pyogrio")` per file, `tree_id += offset`, `pd.concat`, `to_file(out, layer="trees", driver="GPKG", engine="pyogrio")`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv-cpu/bin/python -m pytest -q tests/test_geo_merge.py`
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add ff3d_geo/merge.py tests/test_geo_merge.py
git commit -m "feat: ff3d_geo.merge stitches sub-tile LAS and tree tables with unique ids"
```

---

### Task 3: batched `run` — several `--las` in one preprocess and one inference

**Files:**
- Modify: `ff3d_geo/cli.py` (`plan_run`, `build_parser`, `main`)
- Test: `tests/test_geo_cli.py` (extend)

**Interfaces:**
- `plan_run(las, ...)` accepts `las` as a single path OR a list of paths (`las_paths = [Path(p) for p in ([las] if isinstance(las, (str, Path)) else las)]`); `--las` becomes `nargs="+"`. Every existing single-file call and test keeps working unchanged.
- With N files: stems `stem_i`; each must pass `_UNSAFE_STEM` and parse an origin (`--origin` is only allowed with exactly one file; otherwise raise `ValueError`). `<out>` defaults to `work_dirs/tegel-<first stem>` for one file and MUST be given (`--out`) for several. Steps: `las_to_ply` (all N, one host step), `prepare_inputs` (`scan_list.txt` with N lines, stale `<stem>_*.npy` and `<out>/<stem>.ply` removed for each), ONE `preprocess` docker step (same command, the scan list now has N lines), `check_preprocess` (checks all N `_vert.npy`/`_offsets.npy` and the pkl), ONE `inference` docker step (unchanged command), `results_to_las` (all N), `trees_to_gpkg` (all N), `report` (all N; each `<stem>_report.json/.md`, runtime attributed as `timings["inference"] / N`). Per-file outputs keep today's names in `<out>`.
- Report step prints one line per file (`<stem>: <n_trees> trees, first pass usable: yes|no`) instead of the full markdown when N > 1.

- [ ] **Step 1: Write the failing tests** (add to `tests/test_geo_cli.py`; reuse its existing fixtures/helpers)

```python
def test_run_accepts_several_las_files_in_one_batch(tmp_path, monkeypatch):
    repo = _fake_repo(tmp_path)  # use the module's existing helper for a fake repo layout
    a = tmp_path / "t_E381000_N5829000_100m.las"; a.write_bytes(b"")
    b = tmp_path / "t_E381100_N5829000_100m.las"; b.write_bytes(b"")
    steps = plan_run([a, b], out=repo / "work_dirs" / "batch", repo=repo, gpu="5")
    names = [s.name for s in steps]
    assert names == ["las_to_ply", "prepare_inputs", "preprocess", "check_preprocess",
                     "inference", "results_to_las", "trees_to_gpkg", "report"]
    rendered = "\n".join(s.render() for s in steps)
    assert rendered.count("ff3d_docker") == 2          # one preprocess, one inference
    assert "t_E381000_N5829000_100m" in rendered and "t_E381100_N5829000_100m" in rendered


def test_run_batch_requires_out_and_rejects_origin(tmp_path):
    repo = _fake_repo(tmp_path)
    a = tmp_path / "t_E381000_N5829000_100m.las"; b = tmp_path / "t_E381100_N5829000_100m.las"
    with pytest.raises(ValueError, match="--out"):
        plan_run([a, b], repo=repo)
    with pytest.raises(ValueError, match="--origin"):
        plan_run([a, b], out=repo / "work_dirs" / "x", origin=(1.0, 2.0), repo=repo)


def test_run_batch_prepare_inputs_writes_all_stems(tmp_path):
    repo = _fake_repo(tmp_path)
    a = tmp_path / "t_E381000_N5829000_100m.las"; b = tmp_path / "t_E381100_N5829000_100m.las"
    out = repo / "work_dirs" / "batch"
    steps = plan_run([a, b], out=out, repo=repo)
    next(s for s in steps if s.name == "prepare_inputs").func()
    assert (out / "scan_list.txt").read_text().splitlines() == [a.stem, b.stem]
```

Adapt helper names to what `tests/test_geo_cli.py` already provides (read the file first); keep the assertions.

- [ ] **Step 2: Run to verify they fail**

Run: `.venv-cpu/bin/python -m pytest -q tests/test_geo_cli.py -k batch`
Expected: FAIL (`plan_run` treats the list as a path / TypeError)

- [ ] **Step 3: Implement** in `ff3d_geo/cli.py`; the single-file path must produce byte-identical rendered plans to before (run the whole existing `tests/test_geo_cli.py`).

- [ ] **Step 4: Run the tests**

Run: `.venv-cpu/bin/python -m pytest -q tests/test_geo_cli.py` then `.venv-cpu/bin/python -m pytest -q`
Expected: all passed (existing count + 3)

- [ ] **Step 5: Commit**

```bash
git add ff3d_geo/cli.py tests/test_geo_cli.py
git commit -m "feat: ff3d_geo run batches several sub-tiles into one preprocess and one inference"
```

---

### Task 4: `split` / `merge` subcommands, runbook, Berlin run, report

**Files:**
- Modify: `ff3d_geo/cli.py` (two subcommands), `tests/test_geo_cli.py`, `docs/benchmarks/RUNBOOK-tegel.md` (new section "8. Berlin ALS 2021 km tiles")
- Create: `docs/benchmarks/2026-09-22-tegel-berlin-2021.md`

**Interfaces:**
- `python -m ff3d_geo split --las <km tile> --out <dir> [--size 100] [--min-points 1000]` prints the written sub-tile paths, one per line.
- `python -m ff3d_geo merge --las <subtile result las>... --gpkg <subtile gpkg>... --out-las <las> --out-gpkg <gpkg> [--report-json <json> --report-md <md>]` runs `merge_las`, `merge_trees` and, when `--report-json` is given, `build_report`/`write_report` on the merged files and prints the report markdown.
- Order of `--las` and `--gpkg` must correspond (same sub-tile order); the command checks `len` equality and that stems match pairwise (`<stem>.las` vs `<stem>_trees.gpkg`).

- [ ] **Step 1: Tests** — in `tests/test_geo_cli.py`: `split --help` and `merge --help` parse; `merge` with mismatched lists raises a clear `SystemExit`/`ValueError`; a `split` round trip on a tiny synthetic LAS through `main([...])` writes the expected file names (reuse `tests/test_geo_split.py`'s writer via a shared helper in `tests/geo_fixtures.py`).

- [ ] **Step 2: Implement the subcommands** (lazy imports of `ff3d_geo.split` / `ff3d_geo.merge` inside `main`, like `trees`).

- [ ] **Step 3: Runbook section 8** — the carrot loop, host venv, GPU 5:

```bash
cd /raid/cwinkelmann/ForestFormer3D && source /raid/cwinkelmann/ff3d-geo-venv/bin/activate
for T in 3dm_33_380_5828_1_be 3dm_33_381_5828_1_be 3dm_33_381_5829_1_be; do
  python -m ff3d_geo split --las inputs/berlin/$T.las --out inputs/berlin/sub/$T > work_dirs/logs/split-$T.txt
  python -m ff3d_geo run --las inputs/berlin/sub/$T/*.las --checkpoint work_dirs/clean_forestformer/epoch_3000_fix.pth \
      --out work_dirs/berlin-$T --gpu 5 2>&1 | tee work_dirs/logs/berlin-$T-$(date +%Y%m%d-%H%M%S).log
  python -m ff3d_geo merge --las work_dirs/berlin-$T/*_100m.las --gpkg work_dirs/berlin-$T/*_100m_trees.gpkg \
      --out-las work_dirs/berlin-$T/$T.las --out-gpkg work_dirs/berlin-$T/${T}_trees.gpkg \
      --report-json work_dirs/berlin-$T/${T}_report.json --report-md work_dirs/berlin-$T/${T}_report.md
done
```

(`--las` with a glob relies on the shell; the sub-tile LAS files in `<out>` are the georeferenced results named `<stem>.las`, distinct from the inputs under `inputs/berlin/sub/`.) Copy back: `rsync -av --exclude '*.ply' --exclude '*_100m.las' --exclude '*_100m_trees.gpkg' carrot:/raid/cwinkelmann/ForestFormer3D/work_dirs/berlin-*/ ~/work/hnee/ForestFormer3D_runs/berlin_out/berlin-2021/`.

- [ ] **Step 4: Commit code + runbook**, then run the loop on carrot (each km tile is ~100 sub-tiles; expect ~1 h per km tile; run the three tiles sequentially in one nohup'd script under `work_dirs/logs/`, poll every few minutes).

- [ ] **Step 5: Report** `docs/benchmarks/2026-09-22-tegel-berlin-2021.md`: input table (tile, points, density, sub-tiles run/skipped), per-km-tile results (the three `_report.md` blocks), a short section on sub-tile border effects (count of trees whose crown hull touches a sub-tile border, computed from the merged GeoPackage with a 1 m tolerance), runtime (split, inference wall time per tile), and the recommendation per the existing decision rule; QGIS screenshot pending (user). Commit: `docs: Berlin ALS 2021 inference over the three km tiles around Revier 12 Tegelsee`.
