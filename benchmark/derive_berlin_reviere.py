#!/usr/bin/env python3
"""Berlin forest district (Revier) polygons, the WINMOL survey footprints, and a km-tile
coverage table that says which tiles each area needs and whether the mosaic has them.

Two kinds of district outline go into one GeoPackage:

* ``footprints`` -- the WINMOL 2025 survey footprints of Revier 12 Tegelsee and Revier 13
  Spandau (``WINDWURF_Tegel/Revier_12/ortho/R12_footprint.gpkg``,
  ``Revier_13/Ortho/R13_footprint.gpkg``, EPSG:32633), reprojected to EPSG:25833. These are
  the areas the drone campaign flew and the Probekreise sit in: the reference for "is the
  Revier covered".
* ``reviere`` -- the forest administration's district as far as it is forest, dissolved
  from the Forstbetriebskarte stands: the stand id ``best_dist`` looks like
  ``00101301-0062-a-H010`` and characters 4:6 (``13``) are the Revier number. Wider than
  the footprints (the whole district, not the flown part) and covering Reviere 11 and 15 too.

The ``coverage`` layer answers "are tiles missing?": every km tile with more than
``--min-ha`` of an area, how much of it, and whether the mosaic processed the tile (a
result directory exists), the LAS is merely downloaded, or it was never fetched.

    python benchmark/derive_berlin_reviere.py \\
        --als-data /Volumes/2TB/winmol/ALS_Data \\
        --results berlin_als_2021_ff3d_v2 --las-dir berlin_als_2021

Writes ``<als-data>/berlin_forest/reviere.gpkg`` with layers ``footprints`` (``key``,
``name``, ``area_ha``), ``reviere`` (``revier``, ``name``, ``n_stands``, ``area_ha``) and
``coverage`` (one square per km tile and area: ``tile``, ``key``, ``name``, ``forest_ha``,
``status`` in {processed, downloaded, missing}).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REVIER_NAMES = {"11": "Revier 11", "12": "Revier 12 Tegelsee", "13": "Revier 13 Spandau", "15": "Revier 15"}
EPSG = 25833
WINDWURF = Path("/Volumes/2TB/winmol/training_data/WINDWURF_Tegel")
FOOTPRINTS = {"R12": ("Revier 12 Tegelsee survey footprint", WINDWURF / "Revier_12" / "ortho" / "R12_footprint.gpkg"),
              "R13": ("Revier 13 Spandau survey footprint", WINDWURF / "Revier_13" / "Ortho" / "R13_footprint.gpkg")}


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


def read_footprints(footprints: dict = FOOTPRINTS, warn=print):
    """One row per survey footprint that exists on disk, reprojected to EPSG:25833."""
    import geopandas as gpd
    from shapely import make_valid

    rows = []
    for key, (name, path) in footprints.items():
        if not path.exists():
            warn(f"  footprint {key}: {path} missing, skipped")
            continue
        g = gpd.read_file(path).to_crs(EPSG)
        geom = make_valid(g.geometry.union_all())
        rows.append({"key": key, "name": name, "area_ha": round(geom.area / 1e4, 1), "geometry": geom})
    return gpd.GeoDataFrame(rows, geometry="geometry", crs=f"EPSG:{EPSG}")


def tile_status(tile: str, processed: set[str], downloaded: set[str]) -> str:
    return "processed" if tile in processed else "downloaded" if tile in downloaded else "missing"


def coverage(areas, processed: set[str], downloaded: set[str], min_ha: float = 0.5):
    """One km square per (tile, area) holding more than ``min_ha`` of that area.

    ``areas`` is an iterable of (key, name, geometry); a footprint and a dissolved Revier
    are both areas, the ``key`` tells them apart (``R13`` vs ``13``)."""
    import geopandas as gpd
    from shapely.geometry import box

    rows = []
    for key, name, geom in areas:
        minx, miny, maxx, maxy = geom.bounds
        for e in range(int(minx // 1000), int(maxx // 1000) + 1):
            for n in range(int(miny // 1000), int(maxy // 1000) + 1):
                sq = box(e * 1000, n * 1000, (e + 1) * 1000, (n + 1) * 1000)
                ha = geom.intersection(sq).area / 1e4
                if ha > min_ha:
                    tile = f"3dm_33_{e}_{n}_1_be"
                    rows.append({"tile": tile, "key": key, "name": name, "forest_ha": round(ha, 1),
                                 "status": tile_status(tile, processed, downloaded), "geometry": sq})
    return gpd.GeoDataFrame(rows, geometry="geometry", crs=f"EPSG:{EPSG}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--als-data", type=Path, default=Path("/Volumes/2TB/winmol/ALS_Data"))
    ap.add_argument("--stands", type=Path, default=None, help="default <als-data>/berlin_forest/forstbetriebskarte_2014.gpkg")
    ap.add_argument("--results", default="berlin_als_2021_ff3d_v3", help="result dir (per-tile subdirs) under --als-data")
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
    fp = read_footprints()
    areas = [(r.key, r["name"], r.geometry) for _, r in fp.iterrows()] + \
            [(r.revier, r["name"] + " (forest stands)", r.geometry) for _, r in rev.iterrows()]
    cov = coverage(areas, processed, downloaded, a.min_ha)

    out.parent.mkdir(parents=True, exist_ok=True)
    out.unlink(missing_ok=True)
    if len(fp):
        fp.to_file(out, layer="footprints", driver="GPKG")
    rev.to_file(out, layer="reviere", driver="GPKG")
    cov.to_file(out, layer="coverage", driver="GPKG")
    print(f"wrote {out}: {len(fp)} footprints, {len(rev)} Reviere, {len(cov)} tile/area squares "
          f"({len(processed)} tiles processed, {len(downloaded)} downloaded)")
    for key, name, geom in areas:
        c = cov[cov.key == key]
        gap = c[c.status != "processed"].sort_values("forest_ha", ascending=False)
        print(f"  {name} [{key}]: {geom.area / 1e4:.0f} ha, {len(c)} km tiles > {a.min_ha} ha; "
              f"not processed: {len(gap)} tiles, {gap.forest_ha.sum():.0f} ha "
              f"({100 * gap.forest_ha.sum() / max(c.forest_ha.sum(), 1e-9):.0f} %)")
        for _, g in gap.iterrows():
            print(f"      {g.tile[7:15]}  {g.forest_ha:6.1f} ha  {g.status}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
