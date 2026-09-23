"""Tests for ff3d_geo.report: plausibility report builder and markdown renderer.

Needs laspy, shapely, pyproj and geopandas, so it starts with
``pytest.importorskip`` for each (see tests/test_geo_origin.py's module
docstring) so the system-python test run (no geo libs installed) skips it
instead of failing.

The two-cone synthetic-forest tile builder lives in tests/geo_fixtures.py
(plain functions, not pytest fixtures registered in conftest.py) and is
imported here after the importorskip calls above.
"""

import json

import pytest

pytest.importorskip("laspy")
pytest.importorskip("shapely")
pytest.importorskip("pyproj")
pytest.importorskip("geopandas")

import laspy  # noqa: E402
import numpy as np  # noqa: E402
import pyproj  # noqa: E402

from ff3d_geo.report import build_report, recommend, report_markdown, write_report  # noqa: E402
from ff3d_geo.trees import trees_to_gpkg  # noqa: E402
from geo_fixtures import TILE_ORIGIN, two_cone_points, write_two_cone_las  # noqa: E402


@pytest.fixture
def two_cone_result_las(tmp_path):
    return write_two_cone_las(tmp_path / "cones_result.las", with_predictions=True)


def test_build_report_on_two_cone_tile(two_cone_result_las, tmp_path):
    gpkg = tmp_path / "trees.gpkg"
    trees_to_gpkg(two_cone_result_las, gpkg)

    report = build_report(two_cone_result_las, gpkg, runtime_s=12.5)

    assert report["tile"] == "cones_result"
    assert report["n_points"] > 6000
    assert report["n_trees"] == 2
    assert report["chm_baseline_count"] == 2
    assert abs(report["height_stats"]["min"] - 10.0) <= 0.5
    assert abs(report["height_stats"]["median"] - 12.5) <= 0.5
    assert abs(report["height_stats"]["max"] - 15.0) <= 0.5
    assert report["ground_vs_vegetation_agreement"] == 1.0
    c = report["confusion"]
    assert c["model_ground_als_other"] == 0 and c["model_other_als_ground"] == 0
    assert c["model_ground_als_ground"] + c["model_other_als_other"] == report["n_points"]
    assert set(report["per_class_counts"]) == {"semantic_0", "semantic_1", "semantic_2"}
    assert report["per_class_counts"]["semantic_2"]["class_5"] > 0
    assert report["runtime_s"] == 12.5
    assert report["nodata_fraction"] == 0.0
    assert report["n_voted"] == report["n_points"]
    assert report["recommendation"]["first_pass_usable"] is True

    json_path, md_path = tmp_path / "r.json", tmp_path / "r.md"
    write_report(report, json_path, md_path)
    assert json.loads(json_path.read_text())["n_trees"] == 2
    md = md_path.read_text()
    assert md == report_markdown(report)
    assert "| Trees (model) | 2 |" in md
    assert "| Trees (CHM local maxima baseline) | 2 |" in md
    assert "| Ground vs non-ground agreement | 100.0 % |" in md
    assert "| Nodata fraction (semantic 255, excluded above) | 0.0 % |" in md
    assert "| Inference runtime | 12 s |" in md


def _write_two_cone_las_with_nodata_ground(path, nodata_fraction_of_ground=0.1):
    """Same tile as ``write_two_cone_las(with_predictions=True)`` but with a slice of the
    ALS-ground points (classification == 2) marked ``semantic == 255`` (no model vote), to
    exercise the "exclude nodata from agreement" handling. Returns (path, injected fraction
    of *all* points, number of voted points)."""
    x, y, z, cls, tid, sem = two_cone_points()
    sem = sem.copy()

    ground_idx = np.flatnonzero(cls == 2)
    n_nodata = int(round(nodata_fraction_of_ground * ground_idx.size))
    nodata_idx = ground_idx[:n_nodata]
    sem[nodata_idx] = 255

    header = laspy.LasHeader(point_format=6, version="1.4")
    header.add_extra_dim(laspy.ExtraBytesParams(name="treeID", type=np.int32))
    header.add_extra_dim(laspy.ExtraBytesParams(name="semantic", type=np.uint8))
    header.add_extra_dim(laspy.ExtraBytesParams(name="score", type=np.float32))
    header.add_crs(pyproj.CRS.from_epsg(25833))
    x = x + TILE_ORIGIN[0]
    y = y + TILE_ORIGIN[1]
    header.scales = np.array([0.01, 0.01, 0.01])
    header.offsets = np.floor([x.min(), y.min(), z.min()])
    las = laspy.LasData(header)
    las.x, las.y, las.z = x, y, z
    las.classification = cls
    las.treeID = tid
    las.semantic = sem
    las.score = np.where(tid >= 0, 0.9, 0.0).astype(np.float32)
    las.write(str(path))

    total = sem.size
    return path, nodata_idx.size / total, total - nodata_idx.size


def test_nodata_points_excluded_from_agreement_and_reported(tmp_path):
    path, expected_nodata_fraction, expected_n_voted = _write_two_cone_las_with_nodata_ground(
        tmp_path / "cones_nodata.las"
    )
    gpkg = tmp_path / "trees.gpkg"
    trees_to_gpkg(path, gpkg)

    report = build_report(path, gpkg)

    # The nodata points are a subset of ALS-ground points whose semantic value was
    # overwritten to 255; every remaining (voted) point still has
    # (semantic == 0) == (classification == 2), so agreement must stay 1.0.
    assert report["ground_vs_vegetation_agreement"] == 1.0
    assert abs(report["nodata_fraction"] - expected_nodata_fraction) < 1e-9
    assert report["n_voted"] == expected_n_voted
    c = report["confusion"]
    assert c["model_ground_als_other"] == 0 and c["model_other_als_ground"] == 0
    assert c["model_ground_als_ground"] + c["model_other_als_other"] == expected_n_voted


def test_no_voted_points_gives_null_agreement(tmp_path):
    # Every point (ground and vegetation) has semantic == 255: no model votes at all.
    x, y, z, cls, tid, sem = two_cone_points()
    sem = np.full_like(sem, 255)

    header = laspy.LasHeader(point_format=6, version="1.4")
    header.add_extra_dim(laspy.ExtraBytesParams(name="treeID", type=np.int32))
    header.add_extra_dim(laspy.ExtraBytesParams(name="semantic", type=np.uint8))
    header.add_extra_dim(laspy.ExtraBytesParams(name="score", type=np.float32))
    header.add_crs(pyproj.CRS.from_epsg(25833))
    x, y = x + TILE_ORIGIN[0], y + TILE_ORIGIN[1]
    header.scales = np.array([0.01, 0.01, 0.01])
    header.offsets = np.floor([x.min(), y.min(), z.min()])
    las = laspy.LasData(header)
    las.x, las.y, las.z = x, y, z
    las.classification = cls
    las.treeID = tid
    las.semantic = sem
    las.score = np.where(tid >= 0, 0.9, 0.0).astype(np.float32)
    path = tmp_path / "cones_all_nodata.las"
    las.write(str(path))

    gpkg = tmp_path / "trees.gpkg"
    trees_to_gpkg(path, gpkg)

    report = build_report(path, gpkg)

    assert report["ground_vs_vegetation_agreement"] is None
    assert report["n_voted"] == 0
    assert report["nodata_fraction"] == 1.0


def test_recommend_decision_rule():
    base = {"chm_baseline_count": 100, "n_trees": 149,
            "height_stats": {"median": 20.0}, "chm_height_stats": {"median": 17.5}}
    assert recommend(base)["first_pass_usable"] is True
    assert recommend({**base, "n_trees": 151})["first_pass_usable"] is False
    assert recommend({**base, "height_stats": {"median": 21.0}})["fine_tuning_recommended"] is True
    assert recommend({**base, "height_stats": None})["first_pass_usable"] is False


def test_recommend_flags_mismatched_baseline():
    mismatched = {"chm_baseline_count": 10, "n_trees": 2,
                  "height_stats": {"median": 12.0}, "chm_height_stats": {"median": 11.0}}
    rec = recommend(mismatched)
    assert rec["first_pass_usable"] is False
    assert rec["fine_tuning_recommended"] is True


def test_recommend_treats_zero_baseline_and_zero_trees_as_matching():
    zero = {"chm_baseline_count": 0, "n_trees": 0,
            "height_stats": None, "chm_height_stats": None}
    rec = recommend(zero)
    # No trees found by either the model or the CHM baseline: the counts agree
    # (both zero), even though 0 > 0.5 * 0 would otherwise read as "outside +-50 %".
    assert "both zero" in rec["reasons"][0]
    # Still not first-pass-usable overall, because there are no heights to compare.
    assert rec["first_pass_usable"] is False
    assert rec["reasons"][1] == "no heights to compare"
