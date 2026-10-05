"""The pure helpers of benchmark/berlin_analytics.py (the data-bound parts need the 2TB)."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmark"))

from berlin_analytics import (  # noqa: E402
    FLAT_AREA,
    FLAT_H,
    flat_blob_mask,
    fmt,
    md_table,
    pct,
    pooled_agreement,
    quantiles,
    real,
    tile_key,
)


def test_tile_key_strips_the_prefix_and_suffix():
    assert tile_key("3dm_33_380_5828_1_be") == "380_5828"


def test_md_table_is_pipe_markdown_with_right_aligned_numbers():
    t = md_table(["a", "b"], [["x", 1], ["y", 2]])
    assert t.splitlines() == ["| a | b |", "| :--- | ---: |", "| x | 1 |", "| y | 2 |"]


def test_fmt_and_pct_handle_ints_floats_and_missing():
    assert fmt(1234567) == "1,234,567"
    assert fmt(3.14159, 2) == "3.14"
    assert fmt(float("nan")) == "–" and fmt(None) == "–"
    assert pct(0.5375) == "53.8 %" and pct(float("nan")) == "–"


def test_quantiles_ignore_non_finite_values():
    assert quantiles([1, 2, 3, np.nan, np.inf, 4, 5]) == [1.4, 3.0, 4.6]
    assert all(np.isnan(v) for v in quantiles([]))


def test_flat_blob_flags_only_flat_and_wide_instances():
    h = np.array([FLAT_H - 0.5, FLAT_H - 0.5, FLAT_H + 5, 0.1])
    a = np.array([FLAT_AREA + 1, FLAT_AREA - 1, FLAT_AREA + 100, 1.0])
    assert flat_blob_mask(h, a).tolist() == [True, False, False, False]


def test_pooled_agreement_pools_by_counts_not_by_averaging_fractions():
    big = {"n_a": 900, "n_b": 1000, "matched": 450, "n_points": 9000, "iou_median": 0.7,
           "split_a_frac": 0.3, "split_b_frac": 0.1}
    small = {"n_a": 100, "n_b": 100, "matched": 100, "n_points": 1000, "iou_median": 0.9,
             "split_a_frac": 0.0, "split_b_frac": 0.0}
    p = pooled_agreement([big, small])
    assert p["tiles"] == 2 and p["n_points"] == 10000 and p["matched"] == 550
    assert p["matched_frac_a"] == pytest.approx(550 / 1000)       # not (0.5 + 1.0) / 2
    assert p["matched_frac_b"] == pytest.approx(550 / 1100)
    assert p["split_a_frac"] == pytest.approx(0.27)                # weighted by n_a
    assert p["iou_median"] == pytest.approx(0.8)


def test_real_drops_macos_resource_forks(tmp_path):
    (tmp_path / "a.json").write_text("{}")
    (tmp_path / "._a.json").write_bytes(b"\x00\x05\x16\x07")
    assert [p.name for p in real(sorted(tmp_path.glob("*.json")))] == ["a.json"]
