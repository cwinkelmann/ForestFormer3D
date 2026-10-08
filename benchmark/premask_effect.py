#!/usr/bin/env python3
"""How many tree instances sit on buildings, water or jetties -- per mosaic, per polygon layer.

This is the measurement behind the pre-inference mask (``ff3d_geo run --mask-polygons``,
``ff3d_geo/premask.py``): run it on a mosaic produced WITHOUT the mask to count the
artefacts, and on one produced WITH it to show they are gone. It reads nothing but the
result LAS files and the ALKIS GeoPackages, so it works for any method's products
(ForestFormer3D, SegmentAnyTree, AMS3D) as long as they carry ``treeID``.

    python benchmark/premask_effect.py \\
        --las work_dirs/berlin-mosaic-57/3dm_33_379_5826_1_be.las \\
        --polygons inputs/berlin/alkis_buildings_57tiles.gpkg \\
                   inputs/berlin/alkis_water_57tiles.gpkg:water \\
                   inputs/berlin/alkis_structures_57tiles.gpkg:structures \\
        --json work_dirs/logs/premask/effect-57.json

Per tile and polygon layer it reports the points inside the buffered polygons, how many of
them carry an instance, and how many instances have at least ``--min-fraction`` of their
points inside -- the ones a post-hoc mask would drop whole (``ff3d_geo buildings
--min-roof-fraction``) and the ones the pre-inference mask never creates. An instance is
counted for a layer when that layer holds the majority of its masked points, so the three
layers do not double-count the same tree.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ff3d_geo.premask import load_mask_polygons, mask_points  # noqa: E402


def instance_stats(tree_id: np.ndarray, inside: np.ndarray, min_fraction: float = 0.5) -> dict:
    """Instance counts for one boolean point mask.

    ``tree_id`` is the per-point instance id (-1 = unassigned), ``inside`` the per-point
    polygon membership. Returned: the number of instances touching the polygons at all,
    the number whose share of points inside reaches ``min_fraction`` ("on" the polygons),
    and the point counts behind them.
    """
    if tree_id.shape != inside.shape:
        raise ValueError(f"tree_id and inside differ in length: {tree_id.shape} vs {inside.shape}")
    assigned = tree_id >= 0
    ids = tree_id[assigned]
    n_instances = int(np.unique(ids).size) if ids.size else 0
    hit = assigned & inside
    if not hit.any():
        return {"instances": n_instances, "instances_touching": 0, "instances_on": 0,
                "points_inside": int(inside.sum()), "points_inside_assigned": 0,
                "fraction_on": 0.0}
    # bincount over a dense remap of the ids present, so sparse ids cost nothing
    uniq, dense = np.unique(ids, return_inverse=True)
    total = np.bincount(dense, minlength=uniq.size)
    inside_per_id = np.bincount(dense, weights=inside[assigned].astype(np.float64), minlength=uniq.size)
    touching = int((inside_per_id > 0).sum())
    on = int((inside_per_id >= min_fraction * total).sum())
    return {"instances": n_instances, "instances_touching": touching, "instances_on": on,
            "points_inside": int(inside.sum()), "points_inside_assigned": int(hit.sum()),
            "fraction_on": on / n_instances if n_instances else 0.0}


def _assign_exclusive(masks: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Give a point that several layers claim to the first layer claiming it.

    The ALKIS layers overlap (a jetty over water is in both ``water`` and ``structures``),
    and a tree would otherwise be counted once per layer. The order of ``masks`` decides,
    which is the order the layers were given on the command line.
    """
    taken = None
    out = {}
    for name, m in masks.items():
        if taken is None:
            taken = np.zeros_like(m)
        out[name] = m & ~taken
        taken = taken | m
    return out


def tile_effect(las_path, specs, buffer_m: float = 1.0, min_fraction: float = 0.5) -> dict:
    """Per-layer instance statistics for one result LAS."""
    import laspy

    las = laspy.read(str(las_path))
    x = np.asarray(las.x, dtype=np.float64)
    y = np.asarray(las.y, dtype=np.float64)
    tree_id = np.asarray(las.treeID).astype(np.int64)
    bbox = (float(x.min()), float(y.min()), float(x.max()), float(y.max())) if x.size else None

    masks, n_polygons = {}, {}
    for spec in specs:
        geoms = load_mask_polygons([spec], bbox=bbox, buffer_m=buffer_m)
        n_polygons[spec] = len(geoms)
        masks[spec] = mask_points(x, y, geoms) if geoms else np.zeros(x.size, dtype=bool)

    rec = {"las": str(las_path), "n_points": int(x.size), "buffer_m": buffer_m,
           "min_fraction": min_fraction, "layers": {}}
    for spec, m in _assign_exclusive(masks).items():
        s = instance_stats(tree_id, m, min_fraction)
        s["n_polygons"] = n_polygons[spec]
        rec["layers"][spec] = s
    rec["instances"] = int(np.unique(tree_id[tree_id >= 0]).size)
    rec["semantic_masked_points"] = int((np.asarray(las.semantic) == 3).sum()) if "semantic" in las.point_format.dimension_names else None
    return rec


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--las", nargs="+", required=True, type=Path, help="result LAS files (one mosaic)")
    ap.add_argument("--polygons", nargs="+", required=True,
                    help="GeoPackage specs <path>[:<layer>], in priority order for overlaps")
    ap.add_argument("--buffer", type=float, default=1.0, help="polygon buffer in metres (default 1)")
    ap.add_argument("--min-fraction", type=float, default=0.5,
                    help="share of an instance's points inside the polygons to count it as sitting on them")
    ap.add_argument("--json", type=Path, default=None, help="write the per-tile records here")
    a = ap.parse_args(argv)

    recs = []
    for las in a.las:
        rec = tile_effect(las, a.polygons, a.buffer, a.min_fraction)
        recs.append(rec)
        parts = " | ".join(
            f"{Path(spec).stem.replace('alkis_', '').replace('_57tiles', '')}: "
            f"{s['instances_on']} on ({s['points_inside_assigned']:,} pts)"
            for spec, s in rec["layers"].items())
        print(f"{las.stem:34} {rec['instances']:>8,} instances | {parts}", flush=True)

    if recs:
        print("-" * 110)
        tot_inst = sum(r["instances"] for r in recs)
        print(f"{'TOTAL over ' + str(len(recs)) + ' tiles':34} {tot_inst:>8,} instances", end="")
        for spec in a.polygons:
            on = sum(r["layers"][spec]["instances_on"] for r in recs)
            pts = sum(r["layers"][spec]["points_inside_assigned"] for r in recs)
            name = Path(spec).stem.replace("alkis_", "").replace("_57tiles", "")
            print(f" | {name}: {on:,} on ({on / tot_inst * 100:.2f} %, {pts:,} pts)", end="")
        print()
    if a.json:
        a.json.parent.mkdir(parents=True, exist_ok=True)
        a.json.write_text(json.dumps(recs, indent=1))
        print(f"wrote {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
