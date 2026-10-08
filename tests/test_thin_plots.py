# tests/test_thin_plots.py
"""CPU-only tests for benchmark/thin_plots.py on a synthetic labelled PLY.

Covers the density arithmetic (target = density * hull area), that every field of
the input survives the thinning, that canopy mode keeps the highest points of each
xy cell, and that both modes are deterministic for a given seed.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

pytest.importorskip("plyfile")
pytest.importorskip("shapely")

import numpy as np  # noqa: E402  (after the importorskip guards)
from plyfile import PlyData, PlyElement  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("thin_plots",
                                              REPO / "benchmark" / "thin_plots.py")
thin_plots = importlib.util.module_from_spec(spec)
spec.loader.exec_module(thin_plots)

PLY_DTYPE = [("x", "f4"), ("y", "f4"), ("z", "f4"),
             ("semantic_seg", "i4"), ("treeID", "i4"), ("extra", "u1")]


def make_plot(path: Path, side: float = 20.0, per_cell: int = 40,
              cell: float = 0.5, seed: int = 7) -> np.ndarray:
    """A square plot of ``side`` x ``side`` m with ``per_cell`` points per 0.5 m cell.

    Inside a cell the z values are distinct and increase with the point index, so
    "the highest points of the cell" is unambiguous.
    """
    rng = np.random.default_rng(seed)
    n_cells = int(side / cell)
    cx, cy = np.meshgrid(np.arange(n_cells), np.arange(n_cells), indexing="ij")
    cx = np.repeat(cx.ravel(), per_cell)
    cy = np.repeat(cy.ravel(), per_cell)
    within = np.tile(np.arange(per_cell), n_cells * n_cells)
    n = cx.size
    vertex = np.empty(n, dtype=PLY_DTYPE)
    vertex["x"] = cx * cell + 0.25
    vertex["y"] = cy * cell + 0.25
    vertex["z"] = within * 0.5          # 0, 0.5, ... per cell, strictly increasing
    vertex["semantic_seg"] = rng.integers(1, 4, size=n)
    vertex["treeID"] = rng.integers(0, 12, size=n)
    vertex["extra"] = rng.integers(0, 255, size=n)
    path.parent.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(vertex, "vertex")], text=False,
            byte_order="<").write(str(path))
    return vertex


def test_name_and_tag():
    assert thin_plots.thinned_name("plot_1_test", 25, "uniform") == "plot_1_test_thin25u"
    assert thin_plots.thinned_name("plot_1_test", 25, "canopy") == "plot_1_test_thin25c"
    # no dot in the stem: the data tools split scan names on "_<digits>" and a dot
    # would survive into the npy file names
    assert "." not in thin_plots.thinned_name("p", 12.5, "canopy")


def test_hull_area_of_a_square(tmp_path):
    x = np.array([0.0, 10.0, 10.0, 0.0, 5.0])
    y = np.array([0.0, 0.0, 10.0, 10.0, 5.0])
    assert thin_plots.hull_area(x, y) == pytest.approx(100.0)


@pytest.mark.parametrize("mode", ["uniform", "canopy"])
def test_density_arithmetic_and_all_fields_preserved(tmp_path, mode):
    src = tmp_path / "src" / "plot_a_test.ply"
    vertex = make_plot(src)                       # 20 x 20 m, 40 pts per 0.5 m cell
    density = 25.0
    out = tmp_path / "dst" / "out.ply"
    row = thin_plots.thin_one(src, out, density, mode, seed=0)

    # hull area of the point centres: (20 - 0.5) m square
    assert row["area"] == pytest.approx(19.5 * 19.5, rel=1e-3)
    assert row["target"] == int(round(density * row["area"]))
    assert row["kept"] == row["target"]
    assert row["density_out"] == pytest.approx(density, rel=0.01)
    assert row["kept"] < row["points"]

    kept = PlyData.read(str(out)).elements[0].data
    assert kept.dtype.names == vertex.dtype.names   # every field survived
    assert len(kept) == row["kept"]
    # the kept rows are rows of the input, unchanged
    for name in vertex.dtype.names:
        assert kept[name].dtype == vertex[name].dtype
    assert np.all(np.isin(kept["treeID"], vertex["treeID"]))


def test_canopy_keeps_the_highest_points_per_cell(tmp_path):
    src = tmp_path / "plot_b_test.ply"
    vertex = make_plot(src, side=10.0, per_cell=20)
    out = tmp_path / "canopy.ply"
    row = thin_plots.thin_one(src, out, density=25.0, mode="canopy", seed=0)
    kept = PlyData.read(str(out)).elements[0].data

    cell = 0.5
    key = (np.floor(kept["x"] / cell).astype(np.int64) * 10_000
           + np.floor(kept["y"] / cell).astype(np.int64))
    all_key = (np.floor(vertex["x"] / cell).astype(np.int64) * 10_000
               + np.floor(vertex["y"] / cell).astype(np.int64))
    # every occupied cell contributes at least one point ...
    assert set(np.unique(key)) == set(np.unique(all_key))
    # ... and what it contributes is the TOP of that cell, never a lower point
    for k in np.unique(key)[:20]:
        cell_all = np.sort(vertex["z"][all_key == k])[::-1]
        cell_kept = np.sort(kept["z"][key == k])[::-1]
        assert np.allclose(cell_kept, cell_all[:cell_kept.size])
    assert row["kept"] == row["target"]
    # a uniform subsample of the same plot keeps low points; canopy must not
    uni = tmp_path / "uniform.ply"
    thin_plots.thin_one(src, uni, density=25.0, mode="uniform", seed=0)
    uni_kept = PlyData.read(str(uni)).elements[0].data
    assert kept["z"].mean() > uni_kept["z"].mean()


@pytest.mark.parametrize("mode", ["uniform", "canopy"])
def test_determinism(tmp_path, mode):
    src = tmp_path / "plot_c_test.ply"
    make_plot(src, side=10.0, per_cell=20)
    a, b, c = (tmp_path / f"{n}.ply" for n in "abc")
    thin_plots.thin_one(src, a, 25.0, mode, seed=0)
    thin_plots.thin_one(src, b, 25.0, mode, seed=0)
    thin_plots.thin_one(src, c, 25.0, mode, seed=1)
    ka = PlyData.read(str(a)).elements[0].data
    kb = PlyData.read(str(b)).elements[0].data
    kc = PlyData.read(str(c)).elements[0].data
    assert np.array_equal(ka["x"], kb["x"]) and np.array_equal(ka["z"], kb["z"])
    if mode == "uniform":
        # a different seed must give a different subsample
        assert not np.array_equal(ka["x"], kc["x"])


def test_target_above_the_input_keeps_everything(tmp_path):
    src = tmp_path / "plot_d_test.ply"
    vertex = make_plot(src, side=5.0, per_cell=4)
    out = tmp_path / "all.ply"
    row = thin_plots.thin_one(src, out, density=10_000.0, mode="uniform", seed=0)
    assert row["kept"] == row["points"] == vertex.size


def test_cli_writes_files_and_scan_list(tmp_path, capsys):
    src_dir = tmp_path / "src"
    for stem in ("plot_a_test", "plot_b_test"):
        make_plot(src_dir / f"{stem}.ply", side=10.0, per_cell=20)
    scan_list = tmp_path / "list.txt"
    scan_list.write_text("plot_a_test\nplot_b_test\n")
    out_list = tmp_path / "thin_list.txt"
    rc = thin_plots.main([
        "--src", str(src_dir), "--dst", str(tmp_path / "dst"),
        "--density", "25", "--mode", "canopy", "--seed", "0",
        "--list", str(scan_list), "--out-list", str(out_list),
    ])
    assert rc == 0
    assert (tmp_path / "dst" / "plot_a_test_thin25c.ply").is_file()
    assert (tmp_path / "dst" / "plot_b_test_thin25c.ply").is_file()
    assert out_list.read_text().split() == ["plot_a_test_thin25c", "plot_b_test_thin25c"]


def test_hull_area_needs_three_points():
    with pytest.raises(ValueError, match="at least 3 points"):
        thin_plots.hull_area(np.array([0.0, 1.0]), np.array([0.0, 1.0]))


def test_canopy_warns_when_the_budget_is_below_the_cell_count(tmp_path, capsys):
    """Below ~4 pts/m2 with 0.5 m cells canopy mode stops being canopy-biased."""
    src = tmp_path / "plot_e_test.ply"
    make_plot(src, side=10.0, per_cell=20)          # 400 cells over ~90 m2
    out = tmp_path / "sparse.ply"
    row = thin_plots.thin_one(src, out, density=1.0, mode="canopy", seed=0)
    assert row["kept"] == row["target"] < 400
    assert "canopy mode degenerated" in capsys.readouterr().err
