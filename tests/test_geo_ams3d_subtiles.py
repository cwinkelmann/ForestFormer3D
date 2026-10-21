"""``ff3d_geo ams3d --subtiles``: AMS3D on an existing haloed split, for ``stitch``.

The point of the mode is the contract ``ff3d_geo.stitch`` enforces on a results dir --
one LAS per sub-tile, EVERY point of the split sub-tile in the SAME order, halo included,
shifted to projected coordinates, with treeID/semantic/score -- so that is what is
tested, ending with a real stitch over the output.
"""

import json
from pathlib import Path

import numpy as np
import pytest

laspy = pytest.importorskip("laspy")
pytest.importorskip("scipy")
pytest.importorskip("shapely")

from ff3d_geo.ams3d import CONFIGS, SCORE_NONE, run_ams3d_subtiles  # noqa: E402
from ff3d_geo.cli import main as cli_main  # noqa: E402
from ff3d_geo.split import split_las  # noqa: E402
from geo_fixtures import write_grid_las  # noqa: E402


def _two_subtile_split(tmp_path: Path, buffer_m: float = 20.0):
    """A 200 x 100 m projected tile split into two haloed 100 m sub-tiles."""
    rng = np.random.default_rng(3)
    n = 6000
    xyz = np.column_stack([
        381000 + rng.uniform(0, 200, n), 5829000 + rng.uniform(0, 100, n), rng.uniform(30, 60, n)])
    cls = rng.choice([2, 5], n).astype(np.uint8)
    src = tmp_path / "3dm_33_381_5829_1_be.las"
    write_grid_las(src, xyz, cls)
    sub_dir = tmp_path / "sub"
    written = split_las(src, sub_dir, size_m=100, min_points=10, buffer_m=buffer_m)
    assert len(written) == 2 and (sub_dir / "split_manifest.json").is_file()
    return src, sub_dir, written


def test_subtiles_mode_keeps_every_point_in_split_order(tmp_path):
    _, sub_dir, written = _two_subtile_split(tmp_path)
    out = tmp_path / "res"

    summary = run_ams3d_subtiles(sub_dir, out, CONFIGS["C"], workers=1, config_name="C")

    assert summary["n_subtiles"] == 2 and [s["stem"] for s in summary["subtiles"]] == sorted(p.stem for p in written)
    assert json.loads((out / "ams3d_subtiles.json").read_text())["config"] == "C"
    for sub in written:
        split = laspy.read(str(sub))
        res = laspy.read(str(out / f"{sub.stem}.las"))
        # the halo is NOT cropped: stitch needs labels on the shared halo points
        assert len(res.points) == len(split.points)
        # same order, shifted by the origin in the name (what stitch's order check compares)
        import re
        e, n_ = (float(v) for v in re.search(r"_E(\d+)_N(\d+)_", sub.name).groups())
        assert np.allclose(np.asarray(res.x), np.asarray(split.x) + e, atol=1e-3)
        assert np.allclose(np.asarray(res.y), np.asarray(split.y) + n_, atol=1e-3)
        assert [d.name for d in res.header.point_format.extra_dimensions] == ["treeID", "semantic", "score"]
        ids = np.asarray(res.treeID)
        assert ids.dtype == np.int32 and ids.min() >= -1
        assert (np.asarray(res.score) == SCORE_NONE).all()
        assert (ids[np.asarray(res.classification) == 2] == -1).all()


def test_subtiles_output_stitches_into_one_seamless_tile(tmp_path):
    pytest.importorskip("geopandas")
    from ff3d_geo.stitch import stitch

    src, sub_dir, _ = _two_subtile_split(tmp_path)
    out = tmp_path / "res"
    run_ams3d_subtiles(sub_dir, out, CONFIGS["C"], workers=1)

    mosaic_dir = tmp_path / "mosaic"
    stitch([sub_dir / "split_manifest.json"], [out], mosaic_dir)

    merged = laspy.read(str(mosaic_dir / f"{src.stem}.las"))
    assert len(merged.points) == len(laspy.read(str(src)).points)   # core ownership: no duplicates
    ids = np.asarray(merged.treeID)
    assert (mosaic_dir / "stitch.json").is_file() and ids.max() >= 0
    assert np.unique(ids[ids >= 0]).tolist() == list(range(int(ids.max()) + 1))  # dense mosaic ids


def test_subtiles_mode_refuses_a_bare_directory_and_the_cli_wants_exactly_one_input(tmp_path):
    with pytest.raises(ValueError, match="split directory"):
        run_ams3d_subtiles(tmp_path, tmp_path / "x", CONFIGS["C"], workers=1)
    with pytest.raises(ValueError, match="exactly one"):
        cli_main(["ams3d", "--out", str(tmp_path / "o")])
    with pytest.raises(ValueError, match="exactly one"):
        cli_main(["ams3d", "--las", str(tmp_path / "a.las"), "--subtiles", str(tmp_path), "--out", str(tmp_path / "o")])
