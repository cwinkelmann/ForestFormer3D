"""The mosaic arithmetic behind benchmark/terrain_mosaic.py.

The GDAL calls (gdalbuildvrt, gdal_translate, gdaldem) are not exercised here --
they are one subprocess each and the CLI is not installed on every machine that
runs this suite. What IS tested is everything that decides WHAT gets mosaicked
and whether the result is sound: the union grid, the lattice check that makes a
paste-merge legal, and the seam statistics that reveal tiles filled in isolation
disagreeing along a km border.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmark"))

from terrain_mosaic import (  # noqa: E402
    TileRaster,
    adjacent_pairs,
    mosaic_plan,
    seam_profile,
    seam_report,
)


def tile(name, x0, y0, nx=1000, ny=1000, cell=1.0):
    return TileRaster(name=name, path=Path(f"/nowhere/{name}.tif"),
                      x0=x0, y0=y0, nx=nx, ny=ny, cell=cell)


# ----------------------------------------------------------------- mosaic_plan
def test_mosaic_plan_unions_the_tile_grid():
    p = mosaic_plan([tile("a", 374000, 5825000), tile("b", 375000, 5826000)])
    assert (p["x0"], p["y0"]) == (374000.0, 5825000.0)
    assert (p["nx"], p["ny"]) == (2000, 2000)
    assert p["cell"] == 1.0 and p["n_tiles"] == 2


def test_mosaic_plan_counts_the_gaps_in_a_sparse_mosaic():
    """The Berlin set is 57 tiles inside an 11 x 7 km bounding box, so a third of
    the merged raster is legitimately nodata and the report must say so."""
    tiles = [tile("a", 374000, 5825000), tile("b", 376000, 5825000)]
    p = mosaic_plan(tiles)
    assert (p["nx"], p["ny"]) == (3000, 1000)
    assert p["covered_cells"] == 2_000_000
    assert p["gap_fraction"] == pytest.approx(1 / 3)


def test_mosaic_plan_accepts_a_short_edge_tile():
    """One Berlin tile is 941 cells wide (the ALS stops before the km line)."""
    p = mosaic_plan([tile("a", 374059, 5825000, nx=941), tile("b", 375000, 5825000)])
    assert (p["nx"], p["ny"]) == (1941, 1000)
    assert p["covered_cells"] == 941_000 + 1_000_000


def test_mosaic_plan_rejects_mixed_cell_sizes():
    with pytest.raises(ValueError, match="cell size"):
        mosaic_plan([tile("a", 374000, 5825000), tile("b", 375000, 5825000, cell=0.5)])


def test_mosaic_plan_rejects_a_tile_off_the_common_lattice():
    """A paste-merge is only legal when every tile's origin is a whole number of
    cells from every other's; otherwise the merge needs resampling."""
    with pytest.raises(ValueError, match="lattice"):
        mosaic_plan([tile("a", 374000, 5825000), tile("b", 375000.5, 5825000)])


def test_mosaic_plan_needs_a_tile():
    with pytest.raises(ValueError, match="no tiles"):
        mosaic_plan([])


# --------------------------------------------------------------- adjacent_pairs
def test_adjacent_pairs_finds_east_and_north_neighbours():
    a = tile("a", 374000, 5825000)
    east = tile("e", 375000, 5825000)
    north = tile("n", 374000, 5826000)
    pairs = adjacent_pairs([a, east, north])
    assert {(p[0].name, p[1].name, p[2]) for p in pairs} == {("a", "e", "x"), ("a", "n", "y")}


def test_adjacent_pairs_ignores_diagonal_and_detached_tiles():
    a = tile("a", 374000, 5825000)
    diagonal = tile("d", 375000, 5826000)
    far = tile("f", 380000, 5825000)
    assert adjacent_pairs([a, diagonal, far]) == []


def test_adjacent_pairs_requires_an_overlapping_edge():
    """Touching at x but offset in y past the tile height shares no cell edge."""
    a = tile("a", 374000, 5825000, ny=100)
    b = tile("b", 375000, 5825500, ny=100)
    assert adjacent_pairs([a, b]) == []


# ----------------------------------------------------------------- seam_profile
def test_seam_profile_measures_a_known_step():
    left = np.array([10.0, 10.0, 10.0, 10.0])
    right = np.array([10.5, 10.5, 11.0, 10.5])
    s = seam_profile(left, right)
    assert s["n"] == 4
    assert s["mean_abs"] == pytest.approx(0.625)
    assert s["max"] == pytest.approx(1.0)


def test_seam_profile_ignores_pairs_where_either_side_is_nodata():
    left = np.array([10.0, np.nan, 10.0])
    right = np.array([10.2, 10.0, np.nan])
    s = seam_profile(left, right)
    assert s["n"] == 1 and s["mean_abs"] == pytest.approx(0.2)


def test_seam_profile_of_an_empty_overlap_is_not_an_error():
    s = seam_profile(np.array([np.nan]), np.array([np.nan]))
    assert s["n"] == 0 and s["mean_abs"] is None and s["max"] is None


def test_seam_profile_rejects_mismatched_strips():
    with pytest.raises(ValueError, match="differ in length"):
        seam_profile(np.array([1.0, 2.0]), np.array([1.0]))


# ------------------------------------------------------------------ seam_report
def test_seam_report_flags_the_seams_over_the_threshold():
    stats = [
        {"pair": "a|e", "axis": "x", "n": 4, "mean_abs": 0.05, "p99": 0.1, "max": 0.1},
        {"pair": "a|n", "axis": "y", "n": 4, "mean_abs": 0.40, "p99": 0.9, "max": 0.9},
    ]
    r = seam_report(stats, threshold=0.25)
    assert r["n_seams"] == 2 and r["n_over_threshold"] == 1
    assert r["worst"]["pair"] == "a|n"
    assert r["mean_abs_over_all_seams"] == pytest.approx(0.225)


def test_seam_report_skips_seams_that_measured_nothing():
    stats = [{"pair": "a|e", "axis": "x", "n": 0, "mean_abs": None, "p99": None, "max": None}]
    r = seam_report(stats, threshold=0.25)
    assert r["n_seams"] == 1 and r["n_measured"] == 0
    assert r["n_over_threshold"] == 0 and r["worst"] is None
    assert r["mean_abs_over_all_seams"] is None
