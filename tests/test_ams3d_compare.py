"""benchmark/ams3d_compare.py: Hungarian IoU agreement between two result LAS files."""

import numpy as np
import pytest

pytest.importorskip("laspy")
pytest.importorskip("scipy")
pytest.importorskip("pyproj")

from geo_fixtures import write_result_las  # noqa: E402

import importlib.util  # noqa: E402
from pathlib import Path  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "ams3d_compare", Path(__file__).resolve().parents[1] / "benchmark" / "ams3d_compare.py")
ams3d_compare = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ams3d_compare)


def test_iou_agreement_counts_hungarian_matches_at_half_iou(tmp_path):
    n = 400
    rng = np.random.default_rng(0)
    x = 381300 + rng.uniform(0, 50, n)
    y = 5828300 + rng.uniform(0, 50, n)
    z = rng.uniform(40, 60, n)
    # A: four trees of 100 points; B: tree 0 identical, tree 1 split in two halves
    # (IoU 0.5 each -> exactly one of them can match), trees 2+3 merged into one
    # (IoU 0.5 with each -> one match), plus 20 unassigned points on both sides.
    a = np.repeat(np.arange(4), 100)
    b = a.copy()
    b[100:150] = 1
    b[150:200] = 4
    b[200:400] = 2
    a[-20:] = -1
    b[-20:] = -1
    sem = np.where(a >= 0, 2, 255)
    la = write_result_las(tmp_path / "a.las", x, y, z, a, sem)
    # B in a different row order to exercise the alignment
    perm = rng.permutation(n)
    lb = write_result_las(tmp_path / "b.las", x[perm], y[perm], z[perm], b[perm], sem[perm])

    out = ams3d_compare.iou_agreement(la, lb)
    assert out["trees_a"] == 4 and out["trees_b"] == 4
    # tree 0 exact (IoU 1), tree 1 vs one of its halves (IoU 0.5, the other half
    # cannot also match), the merged 180-point tree vs tree 2 (IoU 100/180 = 0.56;
    # tree 3's 80 points give 0.44 and lose the assignment) -> 3 matches
    assert out["matched_iou50"] == 3
    assert abs(out["mean_iou_of_matches"] - (1.0 + 0.5 + 100 / 180) / 3) < 1e-9
    assert out["points_both_assigned"] == 380
