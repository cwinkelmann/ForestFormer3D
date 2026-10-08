"""benchmark/instance_agreement.py: one-to-one agreement between two instance labelings."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmark"))

from instance_agreement import agreement, format_table  # noqa: E402


def test_identical_labelings_match_completely():
    ids = np.array([-1, -1, 0, 0, 0, 1, 1, 5, 5, 5, 5])
    r = agreement(ids, ids)
    assert r["n_a"] == r["n_b"] == 3 and r["matched"] == 3
    assert r["matched_frac_a"] == r["matched_frac_b"] == 1.0 and r["iou_median"] == 1.0
    assert r["frag_b_per_a"] == 1.0 and r["split_a_frac"] == 0.0
    assert r["pts_a"] == r["pts_b"] == r["pts_both"] == 9


def test_split_and_missed_instances_are_counted():
    #            A: tree 0 (6 pts)        tree 1 (4 pts)   none
    a = np.array([0, 0, 0, 0, 0, 0,        1, 1, 1, 1,      -1, -1])
    #            B cuts A's tree 0 in two halves, ignores tree 1, adds a tree on A's "none"
    b = np.array([7, 7, 7, 8, 8, 8,        -1, -1, -1, -1,  9, 9])
    r = agreement(a, b, iou_thr=0.5)
    assert r["n_a"] == 2 and r["n_b"] == 3
    assert r["matched"] == 1                       # 3/6 = 0.5 reaches the threshold once
    assert r["matched_frac_a"] == 0.5 and r["matched_frac_b"] == pytest.approx(1 / 3)
    assert r["frag_b_per_a"] == 1.0                # tree 0 -> 2 pieces, tree 1 -> 0
    assert r["split_a_frac"] == 0.5
    assert r["frag_a_per_b"] == pytest.approx(2 / 3)   # 7 and 8 lie in A's tree 0; 9 in none
    assert r["split_b_frac"] == 0.0
    assert r["pts_both"] == 6
    table = format_table(r, "A", "B")
    assert "| matched pairs (IoU >= 0.5) | 1 |" in table


def test_empty_side_is_handled():
    a = np.array([0, 0, 1, -1])
    r = agreement(a, np.full(4, -1))
    assert r["n_b"] == 0 and r["matched"] == 0 and r["iou_median"] is None


def test_length_mismatch_is_an_error():
    with pytest.raises(ValueError):
        agreement(np.array([0, 1]), np.array([0]))


def test_threshold_below_half_is_refused():
    with pytest.raises(ValueError):
        agreement(np.array([0, 1]), np.array([0, 1]), iou_thr=0.3)


def test_mutual_best_picks_the_larger_overlap():
    #            A: tree 0 (5 pts)      tree 1 (3 pts)
    a = np.array([0, 0, 0, 0, 0,        1, 1, 1])
    #            B: 4 covers 4 of A0 and all of A1's 3 -> IoU 4/8 = 0.5 with A0, 3/7 with A1;
    #               5 covers the last point of A0 -> IoU 1/5
    b = np.array([4, 4, 4, 4, 5,        4, 4, 4])
    r = agreement(a, b, iou_thr=0.5)
    assert r["n_a"] == 2 and r["n_b"] == 2 and r["matched"] == 1
    assert r["iou_median"] == 0.5
    assert r["frag_a_per_b"] == 1.5          # B4 holds 4/5 of A0 and 3/3 of A1 (2 pieces); B5 holds 1/5 of A0 (1)
    assert r["split_b_frac"] == 0.5
