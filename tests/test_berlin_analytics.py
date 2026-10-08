"""The pure helpers of benchmark/berlin_analytics.py (the data-bound parts need the 2TB)."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmark"))

from berlin_analytics import (  # noqa: E402
    FLAT_AREA,
    agreement_pairs,
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


def test_load_trees_dedupes_only_mosaic_wide_ids(tmp_path):
    """Stitched methods (stitch.json present) list a km-border tree in two tiles' regenerated
    tables and must count it once; a method with per-tile ids (PointTreeFormer) reuses id 0..
    in every tile and must be counted per (tile, id)."""
    gpd = pytest.importorskip("geopandas")
    from shapely.geometry import Point
    import berlin_analytics as ba

    def table(root, tile, ids, npts):
        d = root / tile; d.mkdir(parents=True)
        g = gpd.GeoDataFrame({"tree_id": ids, "x": [0.0] * len(ids), "y": [0.0] * len(ids), "top_z": [10.0] * len(ids),
                              "height": [10.0] * len(ids), "crown_area_m2": [20.0] * len(ids), "n_points": npts,
                              "mean_score": [0.5] * len(ids)}, geometry=[Point(0, 0)] * len(ids), crs="EPSG:25833")
        g.to_file(d / f"{tile}_trees.gpkg", driver="GPKG", layer="trees")

    als = tmp_path
    for key, with_stitch in (("ff3d", True), ("ptf", False)):
        root = als / ba.METHODS[key][1]
        table(root, "3dm_33_380_5828_1_be", [0, 1], [100, 50])
        table(root, "3dm_33_381_5828_1_be", [1, 2], [30, 70])          # id 1 appears in both tiles
        if with_stitch:
            (root / "stitch.json").write_text("{}")
    ff3d = ba.load_trees(als, "ff3d", log=lambda m: None)
    ptf = ba.load_trees(als, "ptf", log=lambda m: None)
    assert len(ff3d) == 3 and ff3d[ff3d.tree_id == 1].tile.tolist() == ["3dm_33_380_5828_1_be"]   # kept where it has most points
    assert len(ptf) == 4


def test_agreement_pairs_prefers_the_current_tag_and_keeps_the_fallback_pair(tmp_path, monkeypatch):
    """The report's mosaic tag wins per pair; a method that ran on a subset of the tiles
    (PointTreeFormer) keeps the tag of the mosaic its agreement was computed on."""
    import berlin_analytics as ba

    for name in ("ff3d_vs_sat_44", "ff3d_vs_sat_57", "sat_vs_ams3d_57", "ff3d_vs_ptf_44",
                 "ff3d_vs_sat_33", "ams3d_3tiles"):
        (tmp_path / name).mkdir()
    monkeypatch.setattr(ba, "AGREEMENT_TAG", "57")
    monkeypatch.setattr(ba, "AGREEMENT_TAG_FALLBACK", {"ff3d_vs_ptf": "44"})
    assert {k: v.name for k, v in ba.agreement_pairs(tmp_path).items()} == {
        ("ff3d", "sat"): "ff3d_vs_sat_57",
        ("sat", "ams3d"): "sat_vs_ams3d_57",
        ("ff3d", "ptf"): "ff3d_vs_ptf_44",
    }
