import numpy as np
import pytest

laspy = pytest.importorskip("laspy")
pytest.importorskip("pyproj")

from ff3d_geo.border import border_check, no_instance_profile, split_pairs
from geo_fixtures import write_result_las


def test_no_instance_profile_measures_the_strip_excess():
    rng = np.random.default_rng(0)
    x, y = rng.uniform(0, 300, 60000), rng.uniform(0, 300, 60000)
    d = np.minimum(np.minimum(x % 100, 100 - x % 100), np.minimum(y % 100, 100 - y % 100))
    tid = np.where(d < 2.0, -1, 5).astype(np.int32)              # unlabelled only inside 2 m of a line
    prof = no_instance_profile(x, y, tid, np.ones_like(tid, bool), size_m=100, bin_m=0.5, max_m=10)
    assert prof["bins"][:3] == [0.0, 0.5, 1.0] and len(prof["bins"]) == 20
    assert prof["frac"][0] == 1.0 and prof["frac"][3] == 1.0 and prof["frac"][4] == 0.0
    assert prof["interior_frac"] == 0.0 and 15 < prof["strip_excess_pp"] < 25


def test_split_pairs_counts_crossing_touching_and_paired_fragments():
    rng = np.random.default_rng(1)
    pts, ids = [], []
    for tid, cx, cy in ((0, 50, 50), (1, 97, 50), (2, 103, 50), (3, 50, 98), (4, 50, 103), (5, 150, 20)):
        p = np.column_stack([cx + rng.uniform(-3, 3, 200), cy + rng.uniform(-3, 3, 200)])
        pts.append(p)
        ids.append(np.full(200, tid))
    p, ids = np.vstack(pts), np.concatenate(ids)
    ids[(ids == 1) & (p[:, 0] >= 100)] = 2      # ids 1/2 are one crown cut at x = 100
    ids[(ids == 2) & (p[:, 0] < 100)] = 1
    ids[(ids == 3) & (p[:, 1] >= 100)] = 4
    ids[(ids == 4) & (p[:, 1] < 100)] = 3
    r = split_pairs(p[:, 0], p[:, 1], ids.astype(np.int32), size_m=100, tol_m=1.0)
    assert r["n_trees"] == 6 and r["n_crossing"] == 0
    assert r["n_touching"] == 4 and r["n_pairs"] == 2


def test_border_check_reads_a_result_las(tmp_path):
    # x/y stay within one cell of the x = 381100 line on each side (50 m and 20 m
    # margins) rather than spanning a full 100 m period: a domain that lands exactly
    # on grid-line multiples on all four sides (as e.g. 381000..381200 does) makes
    # its OWN bounding-box edges coincide with genuine "nearest size_m multiple"
    # lines that no_instance_profile has no way to tell apart from the real
    # embedded split at x = 381100 -- so points near those edges would count as
    # near-a-line too, even though this fixture's tid never marks them unlabelled.
    rng = np.random.default_rng(2)
    n = 5000
    x, y = 381000 + rng.uniform(50, 150, n), 5829000 + rng.uniform(20, 80, n)
    tid = np.where(np.abs(x - 381100) < 1.5, -1, (x > 381100).astype(np.int32))
    las = write_result_las(tmp_path / "t.las", x, y, rng.uniform(1, 20, n), tid,
                           np.full(n, 2, np.uint8))
    las_data = laspy.read(las)
    las_data.classification = np.full(n, 5, np.uint8)
    las_data.write(las)
    r = border_check(las, size_m=100)
    assert r["n_points"] == n and r["n_trees"] == 2 and r["n_crossing"] == 0
    assert r["frac"][0] == 1.0 and r["interior_frac"] == 0.0
