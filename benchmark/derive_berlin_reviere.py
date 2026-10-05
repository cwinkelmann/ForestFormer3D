#!/usr/bin/env python3
"""Derive Berlin forest district (Revier) polygons and a km-tile coverage table.

No source publishes Revier boundaries: the WINMOL field GeoPackages hold crowns and AOIs
only, and the Forstbetriebskarte's ``b_forstverwalt`` layer is cartographic labels. But the
stand id of ``c_hauptbaumarten`` encodes the district -- ``best_dist`` looks like
``00101301-0062-a-H010`` and characters 4:6 (``13``) are the Revier number (12 = Tegelsee,
13 = Spandau, the two WINMOL sample areas). Dissolving the stands per Revier gives the
district outline as far as it is forest.

The second output answers "are tiles missing?": every km tile with more than ``--min-ha``
of Revier forest, with how much forest it holds and whether the mosaic processed it
(a result directory exists), the LAS is merely downloaded, or it was never fetched.

    python benchmark/derive_berlin_reviere.py \\
        --als-data /Volumes/2TB/winmol/ALS_Data \\
        --results berlin_als_2021_ff3d_v2 --las-dir berlin_als_2021

Writes ``<als-data>/berlin_forest/reviere.gpkg`` with layers ``reviere`` (one MultiPolygon
per Revier: ``revier``, ``name``, ``n_stands``, ``area_ha``) and ``coverage`` (one square
per km tile and Revier: ``tile``, ``revier``, ``forest_ha``, ``status`` in
{processed, downloaded, missing}).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REVIER_NAMES = {"11": "Revier 11", "12": "Revier 12 Tegelsee", "13": "Revier 13 Spandau", "15": "Revier 15"}
EPSG = 25833


def revier_of(best_dist: str) -> str:
    """``00101301-0062-a-H010`` -> ``13``: the Revier number inside the stand id."""
    return str(best_dist)[4:6]


def dissolve_reviere(stands):
    """Dissolve the stand polygons (made valid first) into one geometry per Revier."""
    from shapely import make_valid

    s = stands.copy()
    s["revier"] = s["best_dist"].map(revier_of)
    s["geometry"] = s.geometry.apply(make_valid)
    rev = s.dissolve(by="revier").reset_index()[["revier", "geometry"]]
    rev["name"] = rev["revier"].map(lambda r: REVIER_NAMES.get(r, f"Revier {r}"))
    rev["n_stands"] = rev["revier"].map(s.groupby("revier").size())
    rev["area_ha"] = (rev.geometry.area / 1e4).round(1)
    return rev[["revier", "name", "n_stands", "area_ha", "geometry"]]


def tile_status(tile: str, processed: set[str], downloaded: set[str]) -> str:
    return "processed" if tile in processed else "downloaded" if tile in downloaded else "missing"


def coverage(reviere, processed: set[str], downloaded: set[str], min_ha: float = 0.5):
    """One km square per (tile, Revier) holding more than ``min_ha`` of that Revier's forest."""
    import geopandas as gpd
    from shapely.geometry import box

    rows = []
    for _, r in reviere.iterrows():
        minx, miny, maxx, maxy = r.geometry.bounds
        for e in range(int(minx // 1000), int(maxx // 1000) + 1):
            for n in range(int(miny // 1000), int(maxy // 1000) + 1):
                sq = box(e * 1000, n * 1000, (e + 1) * 1000, (n + 1) * 1000)
                ha = r.geometry.intersection(sq).area / 1e4
                if ha > min_ha:
                    tile = f"3dm_33_{e}_{n}_1_be"
                    rows.append({"tile": tile, "revier": r["revier"], "name": r["name"], "forest_ha": round(ha, 1),
                                 "status": tile_status(tile, processed, downloaded), "geometry": sq})
    return gpd.GeoDataFrame(rows, geometry="geometry", crs=f"EPSG:{EPSG}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--als-data", type=Path, default=Path("/Volumes/2TB/winmol/ALS_Data"))
    ap.add_argument("--stands", type=Path, default=None, help="default <als-data>/berlin_forest/forstbetriebskarte_2014.gpkg")
    ap.add_argument("--results", default="berlin_als_2021_ff3d_v2", help="result dir (per-tile subdirs) under --als-data")
    ap.add_argument("--las-dir", default="berlin_als_2021", help="downloaded km-tile LAS dir under --als-data")
    ap.add_argument("--out", type=Path, default=None, help="default <als-data>/berlin_forest/reviere.gpkg")
    ap.add_argument("--min-ha", type=float, default=0.5)
    a = ap.parse_args(argv)
    import geopandas as gpd

    stands_path = a.stands or a.als_data / "berlin_forest" / "forstbetriebskarte_2014.gpkg"
    out = a.out or a.als_data / "berlin_forest" / "reviere.gpkg"
    if not stands_path.exists():
        print(f"!!! {stands_path} missing: run benchmark/fetch_berlin_forest_stands.py first", file=sys.stderr)
        return 2
    stands = gpd.read_file(stands_path, layer="hauptbaumarten", columns=["best_dist", "geometry"])
    rev = dissolve_reviere(stands)
    processed = {p.name for p in (a.als_data / a.results).glob("3dm_33_*_1_be") if p.is_dir()}
    downloaded = {p.stem for p in (a.als_data / a.las_dir).glob("3dm_33_*_1_be.las")}
    cov = coverage(rev, processed, downloaded, a.min_ha)

    out.parent.mkdir(parents=True, exist_ok=True)
    out.unlink(missing_ok=True)
    rev.to_file(out, layer="reviere", driver="GPKG")
    cov.to_file(out, layer="coverage", driver="GPKG")
    print(f"wrote {out}: {len(rev)} Reviere, {len(cov)} tile/Revier squares "
          f"({len(processed)} tiles processed, {len(downloaded)} downloaded)")
    for _, r in rev.iterrows():
        c = cov[cov.revier == r.revier]
        gap = c[c.status != "processed"].sort_values("forest_ha", ascending=False)
        print(f"  {r['name']}: {r.n_stands} stands, {r.area_ha} ha, {len(c)} km tiles > {a.min_ha} ha; "
              f"not processed: {len(gap)} ({gap.forest_ha.sum():.0f} ha)")
        for _, g in gap.iterrows():
            print(f"      {g.tile[7:15]}  {g.forest_ha:6.1f} ha  {g.status}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
