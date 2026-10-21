"""ff3d_geo.filter: instances below a minimum height are unassigned, nothing is renumbered."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("laspy")
pytest.importorskip("geopandas")
pytest.importorskip("shapely")
pytest.importorskip("pyproj")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from geo_fixtures import write_result_las  # noqa: E402

from ff3d_geo.filter import drop_instances, filter_short_instances, short_instance_ids  # noqa: E402

ORIGIN = (381000.0, 5829000.0)


def _grid(x0, y0, nx=8, ny=8, step=1.0):
    xs, ys = np.meshgrid(x0 + np.arange(nx) * step, y0 + np.arange(ny) * step)
    return xs.ravel(), ys.ravel()


def _scene(path):
    """Three instances over a flat ground at z = 0: a 15 m tree (id 1), a 1.2 m grass blob
    with a wide 8 x 8 m hull (id 2, the meadow artefact) and a 3 m shrub (id 3)."""
    x1, y1 = _grid(ORIGIN[0] + 10, ORIGIN[1] + 10, 4, 4)
    x2, y2 = _grid(ORIGIN[0] + 30, ORIGIN[1] + 10)
    x3, y3 = _grid(ORIGIN[0] + 50, ORIGIN[1] + 10, 3, 3)
    xg, yg = _grid(ORIGIN[0], ORIGIN[1], 70, 30, 1.0)
    x = np.concatenate([x1, x2, x3, xg]); y = np.concatenate([y1, y2, y3, yg])
    z = np.concatenate([np.linspace(1, 15, x1.size), np.linspace(0.2, 1.2, x2.size),
                        np.linspace(0.5, 3.0, x3.size), np.zeros(xg.size)])
    tree_id = np.concatenate([np.full(x1.size, 1), np.full(x2.size, 2), np.full(x3.size, 3), np.full(xg.size, -1)])
    semantic = np.concatenate([np.full(x1.size, 2), np.full(x2.size, 2), np.full(x3.size, 2), np.zeros(xg.size)])
    write_result_las(path, x, y, z, tree_id, semantic)
    return {"n_blob": x2.size}


def test_short_instance_ids_uses_the_table_height():
    rows = [{"tree_id": 1, "height": 15.0}, {"tree_id": 2, "height": 1.2}, {"tree_id": 3, "height": 3.0}]
    assert short_instance_ids(rows, 2.0) == [2]
    assert short_instance_ids(rows, 3.5) == [2, 3]
    assert short_instance_ids(rows, 0.0) == []


def test_filter_drops_the_grass_blob_and_keeps_the_other_ids(tmp_path):
    import laspy

    src = tmp_path / "t.las"
    n = _scene(src)
    out = tmp_path / "out" / "t.las"
    info = filter_short_instances(src, out, min_height=2.0)
    assert info["instances_before"] == 3 and info["instances_removed"] == 1 and info["instances_after"] == 2
    assert info["points_unassigned"] == n["n_blob"] and info["min_height"] == 2.0
    las = laspy.read(str(out))
    ids = np.asarray(las.treeID)
    assert set(np.unique(ids[ids >= 0]).tolist()) == {1, 3}          # not renumbered
    assert (np.asarray(las.score)[ids < 0] == 0).all()
    assert laspy.read(str(src)).header.point_count == las.header.point_count


def test_drop_instances_rewrites_in_place_atomically(tmp_path):
    import laspy

    src = tmp_path / "t.las"
    _scene(src)
    info = drop_instances(src, src, [3])
    assert info["instances_removed"] == 1 and not (tmp_path / "t.las.tmp").exists()
    ids = np.asarray(laspy.read(str(src)).treeID)
    assert set(np.unique(ids[ids >= 0]).tolist()) == {1, 2}
    assert drop_instances(src, src, [])["instances_removed"] == 0            # idempotent no-op


def test_cli_filter_writes_the_set_and_the_report_block(tmp_path):
    src = tmp_path / "t.las"
    _scene(src)
    out = tmp_path / "filtered"
    r = subprocess.run([sys.executable, "-m", "ff3d_geo", "filter", "--las", str(src), "--out", str(out),
                        "--min-height", "2"], capture_output=True, text=True,
                       cwd=str(Path(__file__).resolve().parents[1]))
    assert r.returncode == 0, r.stderr
    assert (out / "t.las").exists() and (out / "t_trees.gpkg").exists() and (out / "t_crowns.gpkg").exists()
    rep = json.loads((out / "t_report.json").read_text())
    assert rep["n_trees"] == 2 and rep["height_filter"]["instances_removed"] == 1
    assert "Height filter" in (out / "t_report.md").read_text()


def test_stitch_min_height_drops_short_instances_mosaic_wide(tmp_path):
    from test_geo_stitch import _mosaic
    from ff3d_geo.stitch import stitch

    manifest, res, _xyz, _tree = _mosaic(tmp_path)
    plain = stitch([manifest], [res], tmp_path / "plain")
    tall = stitch([manifest], [res], tmp_path / "tall", min_height=1000.0)   # nothing is that tall
    assert plain["n_short_removed"] == 0 and plain["min_height"] == 0.0
    assert tall["n_short_removed"] == plain["n_trees"] and tall["n_trees"] == 0
    for tile in tall["tiles"].values():
        assert tile["n_trees_in_tile"] == 0
        rep = json.loads(Path(tile["las"]).with_name(Path(tile["las"]).stem + "_report.json").read_text())
        assert rep["height_filter"]["min_height"] == 1000.0 and rep["n_trees"] == 0
