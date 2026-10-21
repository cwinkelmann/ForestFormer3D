# tests/test_instance_diagnostics.py
"""CPU-only tests for benchmark/instance_diagnostics.py on hand-built result PLYs.

Covers the fragment/merge counting, the detected-tree conditioning, the
matched-pair height delta, and the zero-prediction plot that used to crash
``--per-plot``.
"""
from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import pytest

pytest.importorskip("plyfile")
pytest.importorskip("scipy")

import numpy as np  # noqa: E402
from plyfile import PlyData, PlyElement  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "instance_diagnostics", REPO / "benchmark" / "instance_diagnostics.py")
diag = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diag)

DTYPE = [("x", "f4"), ("y", "f4"), ("z", "f4"),
         ("semantic_pred", "i4"), ("instance_pred", "i4"), ("score", "f4"),
         ("semantic_gt", "i4"), ("instance_gt", "i4")]


def write_ply(path: Path, z, sem_gt, ins_gt, ins_pred, sem_pred=None):
    n = len(z)
    v = np.empty(n, dtype=DTYPE)
    v["x"] = np.arange(n) % 10
    v["y"] = np.arange(n) // 10
    v["z"] = z
    v["score"] = 1.0
    v["semantic_gt"] = sem_gt
    v["instance_gt"] = ins_gt
    v["instance_pred"] = ins_pred
    v["semantic_pred"] = sem_gt if sem_pred is None else sem_pred
    path.parent.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(v, "vertex")], text=False,
            byte_order="<").write(str(path))


def test_zero_prediction_plot_does_not_crash(tmp_path, capsys):
    """Regression: --per-plot used to raise TypeError on a None sentinel."""
    d = tmp_path / "run"
    # 20 ground points (raw convention: semantic 0, treeID 0) + one GT tree,
    # and not a single predicted instance.
    z = np.concatenate([np.zeros(20), np.linspace(1, 9, 30)])
    sem_gt = np.concatenate([np.zeros(20, int), np.full(30, 2)])
    ins_gt = np.concatenate([np.zeros(20, int), np.full(30, 7)])
    write_ply(d / "empty_plot_test.ply", z, sem_gt, ins_gt,
              np.full(50, -1))
    out_json = tmp_path / "d.json"
    rc = diag.main(["--run", f"x={d}", "--per-plot", "--json", str(out_json)])
    assert rc == 0
    text = capsys.readouterr().out
    assert "empty_plot_test" in text            # the per-plot line was printed
    payload = json.loads(out_json.read_text())  # and the json really got written
    pooled = payload["runs"]["x"]["pooled"]
    assert pooled["n_gt"] == 1 and pooled["n_pred"] == 0
    assert pooled["plots_no_pred"] == 1
    assert pooled["n_detected"] == 0
    assert math.isnan(pooled["split_frac_detected"])
    assert math.isnan(pooled["merged_frac"])
    assert math.isnan(pooled["dz_median"])


def make_reference_plot(path: Path):
    """4 GT trees: A found exactly, B split 2-ways, C+D merged into one prediction.

    Heights are unambiguous: the top z of every GT tree and of every prediction
    is a distinct known value.
    """
    # 10 ground points, then 4 trees of 20 points each
    z = [0.0] * 10
    sem_gt = [0] * 10
    ins_gt = [0] * 10
    ins_pred = [-1] * 10
    tops = {1: 11.0, 2: 21.0, 3: 16.0, 4: 6.0}
    for tree, top in tops.items():
        z += list(np.linspace(top - 5, top, 20))
        sem_gt += [2] * 20
        ins_gt += [tree] * 20
    ins_pred += [0] * 20              # A: prediction 0 == tree 1 exactly
    ins_pred += [1] * 14 + [2] * 6    # B: tree 2 split into predictions 1 and 2
    ins_pred += [3] * 20              # C+D: prediction 3 covers trees 3 and 4
    ins_pred += [3] * 20
    write_ply(path, np.array(z), np.array(sem_gt), np.array(ins_gt),
              np.array(ins_pred))
    return tops


def test_fragment_merge_and_detected_conditioning(tmp_path):
    tops = make_reference_plot(tmp_path / "ref_plot_test.ply")
    row = diag.diagnose_plot(tmp_path / "ref_plot_test.ply")
    assert row["n_gt"] == 4 and row["n_pred"] == 4
    # tree 1 -> 1 fragment, tree 2 -> 2, trees 3 and 4 -> 1 each
    assert list(row["frags"]) == [1, 2, 1, 1]
    # prediction 3 covers 100 % of tree 3 and of tree 4
    assert list(row["covers"]) == [1, 1, 1, 2]
    s = diag.summarize([row])
    assert s["frag_per_tree"] == pytest.approx(1.25)
    assert s["split_frac"] == pytest.approx(0.25)
    assert s["n_detected"] == 4
    assert s["frag_per_detected_tree"] == pytest.approx(1.25)
    assert s["split_frac_detected"] == pytest.approx(0.25)
    assert s["merged_frac"] == pytest.approx(0.25)
    assert tops  # heights used by the next test


def test_detected_conditioning_ignores_missed_trees(tmp_path):
    """Misses must not make the fragmentation rate look better."""
    # 4 GT trees, only tree 1 predicted (perfectly): the all-GT numbers collapse,
    # the detected-tree numbers do not.
    z, sem_gt, ins_gt, ins_pred = [], [], [], []
    for tree, top in {1: 11.0, 2: 21.0, 3: 16.0, 4: 6.0}.items():
        z += list(np.linspace(top - 5, top, 20))
        sem_gt += [2] * 20
        ins_gt += [tree] * 20
        ins_pred += ([0] * 10 + [1] * 10) if tree == 1 else [-1] * 20
    p = tmp_path / "miss_plot_test.ply"
    write_ply(p, np.array(z), np.array(sem_gt), np.array(ins_gt),
              np.array(ins_pred))
    s = diag.summarize([diag.diagnose_plot(p)])
    assert s["n_gt"] == 4 and s["n_detected"] == 1
    assert s["frag_per_tree"] == pytest.approx(0.5)      # 2/4: dragged down by misses
    assert s["split_frac"] == pytest.approx(0.25)
    assert s["frag_per_detected_tree"] == pytest.approx(2.0)   # the real rate
    assert s["split_frac_detected"] == pytest.approx(1.0)


def test_matched_pair_height_delta(tmp_path):
    """dz is per matched pair; h_pred/h_gt over different populations are biased."""
    # 3 GT trees with tops 30, 20, 10 above the plot minimum (z_min = 0).
    # Only the tallest is predicted, and its prediction is 2 m short of the top.
    z = [0.0]
    sem_gt = [0]
    ins_gt = [0]
    ins_pred = [-1]
    for tree, top in {1: 30.0, 2: 20.0, 3: 10.0}.items():
        z += list(np.linspace(top - 5, top, 20))
        sem_gt += [2] * 20
        ins_gt += [tree] * 20
        # prediction 0 takes the first 18 of tree 1's 20 points (IoU 0.9 -> matched),
        # so its top is the 18th of 20 values from top-5 to top: top - 5*2/19
        ins_pred += ([0] * 18 + [-1] * 2) if tree == 1 else [-1] * 20
    p = tmp_path / "tall_plot_test.ply"
    write_ply(p, np.array(z), np.array(sem_gt), np.array(ins_gt), np.array(ins_pred))
    row = diag.diagnose_plot(p)
    s = diag.summarize([row])
    assert s["matched"] == 1
    expected_dz = (30.0 - 5 * 2 / 19) - 30.0
    assert s["dz_median"] == pytest.approx(expected_dz, abs=1e-6)
    assert s["dz_mean"] == pytest.approx(expected_dz, abs=1e-6)
    # the unconditioned comparison claims the prediction is ~9 m TALLER than a
    # GT tree, purely because the two short trees were missed
    assert s["pred_h_median"] - s["gt_h_median"] > 8.0


def test_gt_id_zero_is_a_tree_when_the_file_is_normalized(tmp_path):
    """ground = -1 convention: instance id 0 is a real tree, not 'unannotated'."""
    z = np.concatenate([np.zeros(10), np.linspace(1, 5, 20)])
    sem_gt = np.concatenate([np.zeros(10, int), np.full(20, 2)])
    ins_gt = np.concatenate([np.full(10, -1), np.zeros(20, int)])
    p = tmp_path / "norm_plot_test.ply"
    write_ply(p, z, sem_gt, ins_gt, np.concatenate([np.full(10, -1), np.zeros(20, int)]))
    row = diag.diagnose_plot(p)
    assert row["n_gt"] == 1 and row["matched"] == 1


def test_ply_without_gt_is_skipped(tmp_path, capsys):
    from plyfile import PlyData as _P  # noqa: F401

    d = tmp_path / "nogt"
    d.mkdir()
    v = np.empty(5, dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"),
                           ("semantic_pred", "i4"), ("instance_pred", "i4"),
                           ("score", "f4")])
    for name in v.dtype.names:
        v[name] = 0
    PlyData([PlyElement.describe(v, "vertex")], text=False,
            byte_order="<").write(str(d / "a_plot_test.ply"))
    assert diag.diagnose_plot(d / "a_plot_test.ply") is None
