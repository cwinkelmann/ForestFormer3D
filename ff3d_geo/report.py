"""Plausibility report for one tile: model output vs ALS classification and CHM."""

from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import laspy
import numpy as np

from ff3d_geo.baseline import chm_local_maxima

# Decision rule for the "Recommendation on ALS fine-tuning" section of the report.
COUNT_TOLERANCE = 0.5  # tree count within +-50 % of the CHM baseline
HEIGHT_TOLERANCE_M = 3.0  # median tree height within 3 m of the CHM maxima median


def _stats(values: np.ndarray) -> dict | None:
    if values.size == 0:
        return None
    return {
        "min": float(np.min(values)),
        "median": float(np.median(values)),
        "max": float(np.max(values)),
    }


def build_report(las_path, gpkg_path, runtime_s: float | None = None) -> dict:
    """Build the ``<stem>_report.json`` content from a result LAS and its GeoPackage."""
    las = laspy.read(str(las_path))
    tree_id = np.asarray(las.treeID, dtype=np.int64)
    semantic = np.asarray(las.semantic, dtype=np.int64)
    classification = np.asarray(las.classification, dtype=np.int64)

    maxima = chm_local_maxima(las_path)
    chm_heights = np.array([h for _, _, h in maxima], dtype=np.float64)
    heights = gpd.read_file(str(gpkg_path), layer="trees")["height"].to_numpy(dtype=np.float64)

    model_ground = semantic == 0
    als_ground = classification == 2
    confusion = {
        "model_ground_als_ground": int(np.sum(model_ground & als_ground)),
        "model_ground_als_other": int(np.sum(model_ground & ~als_ground)),
        "model_other_als_ground": int(np.sum(~model_ground & als_ground)),
        "model_other_als_other": int(np.sum(~model_ground & ~als_ground)),
    }
    per_class = {
        f"semantic_{s}": {
            f"class_{c}": int(np.sum((semantic == s) & (classification == c)))
            for c in np.unique(classification)
        }
        for s in np.unique(semantic)
    }

    report = {
        "tile": Path(las_path).stem,
        "n_points": int(tree_id.size),
        "n_trees": int(np.unique(tree_id[tree_id >= 0]).size),
        "chm_baseline_count": len(maxima),
        "height_stats": _stats(heights),
        "chm_height_stats": _stats(chm_heights),
        "ground_vs_vegetation_agreement": float(np.mean(model_ground == als_ground)),
        "confusion": confusion,
        "per_class_counts": per_class,
        "runtime_s": runtime_s,
    }
    report["recommendation"] = recommend(report)
    return report


def recommend(report: dict) -> dict:
    """Apply the decision rule: usable as a first pass when the tree count is within
    50 % of the CHM baseline and the median height within 3 m of the CHM median."""
    baseline = report["chm_baseline_count"]
    n_trees = report["n_trees"]
    count_ok = baseline > 0 and abs(n_trees - baseline) <= COUNT_TOLERANCE * baseline
    hs, cs = report["height_stats"], report["chm_height_stats"]
    height_ok = hs is not None and cs is not None and abs(hs["median"] - cs["median"]) <= HEIGHT_TOLERANCE_M
    usable = bool(count_ok and height_ok)
    reasons = [
        f"tree count {n_trees} vs CHM baseline {baseline} ({'ok' if count_ok else 'outside +-50 %'})",
        (
            f"median height {hs['median']:.1f} m vs CHM median {cs['median']:.1f} m "
            f"({'ok' if height_ok else 'outside 3 m'})"
            if hs is not None and cs is not None
            else "no heights to compare"
        ),
    ]
    return {
        "first_pass_usable": usable,
        "fine_tuning_recommended": not usable,
        "reasons": reasons,
    }


def report_markdown(report: dict) -> str:
    """Render the report as the per-tile markdown block used in docs/benchmarks/<date>-tegel-als.md."""
    hs = report["height_stats"] or {"min": float("nan"), "median": float("nan"), "max": float("nan")}
    cs = report["chm_height_stats"] or {"min": float("nan"), "median": float("nan"), "max": float("nan")}
    runtime = "n/a" if report["runtime_s"] is None else f"{report['runtime_s']:.0f} s"
    c = report["confusion"]
    lines = [
        f"### {report['tile']}",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Points | {report['n_points']} |",
        f"| Inference runtime | {runtime} |",
        f"| Trees (model) | {report['n_trees']} |",
        f"| Trees (CHM local maxima baseline) | {report['chm_baseline_count']} |",
        f"| Height min / median / max (model, m) | {hs['min']:.1f} / {hs['median']:.1f} / {hs['max']:.1f} |",
        f"| Height min / median / max (CHM, m) | {cs['min']:.1f} / {cs['median']:.1f} / {cs['max']:.1f} |",
        f"| Ground vs vegetation agreement | {100 * report['ground_vs_vegetation_agreement']:.1f} % |",
        "",
        "| Model \\ ALS | class 2 (ground) | other |",
        "|---|---|---|",
        f"| semantic 0 (ground) | {c['model_ground_als_ground']} | {c['model_ground_als_other']} |",
        f"| semantic 1/2 (wood/leaf) | {c['model_other_als_ground']} | {c['model_other_als_other']} |",
        "",
        f"First pass usable: **{'yes' if report['recommendation']['first_pass_usable'] else 'no'}** "
        f"({'; '.join(report['recommendation']['reasons'])})",
        "",
    ]
    return "\n".join(lines)


def write_report(report: dict, json_path, md_path=None) -> None:
    Path(json_path).write_text(json.dumps(report, indent=2))
    if md_path is not None:
        Path(md_path).write_text(report_markdown(report))
