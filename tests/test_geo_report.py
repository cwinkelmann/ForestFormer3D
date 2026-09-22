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

from ff3d_geo.report import build_report, recommend, report_markdown, write_report  # noqa: E402
from ff3d_geo.trees import trees_to_gpkg  # noqa: E402
from geo_fixtures import write_two_cone_las  # noqa: E402


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
    assert report["recommendation"]["first_pass_usable"] is True

    json_path, md_path = tmp_path / "r.json", tmp_path / "r.md"
    write_report(report, json_path, md_path)
    assert json.loads(json_path.read_text())["n_trees"] == 2
    md = md_path.read_text()
    assert md == report_markdown(report)
    assert "| Trees (model) | 2 |" in md
    assert "| Trees (CHM local maxima baseline) | 2 |" in md
    assert "| Ground vs vegetation agreement | 100.0 % |" in md
    assert "| Inference runtime | 12 s |" in md


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
