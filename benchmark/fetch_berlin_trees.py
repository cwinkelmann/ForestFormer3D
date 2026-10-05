#!/usr/bin/env python3
"""Fetch Berlin's tree cadastre (Baumbestand Berlin) for a bbox into one GeoPackage.

Source (official, open; capabilities checked 2026-10-05)::

    WFS 2.0     https://gdi.berlin.de/services/wfs/baumbestand
    typenames   baumbestand:strassenbaeume   street trees
                baumbestand:anlagenbaeume    trees in green spaces ("einen Teil der Baeume
                                             in Gruenanlagen" -- a part, not all)
    licence     Datenlizenz Deutschland - Zero - 2.0 (dl-de/zero-2-0)
    provider    Senatsverwaltung fuer Stadtentwicklung, Bauen und Wohnen Berlin

It is the managed trees -- streets and parks -- with species (``art_dtsch``, ``art_bot``,
``gattung``), planting year (``pflanzjahr``), height (``baumhoehe``), crown diameter
(``kronedurch``) and trunk girth (``stammumfg``). It does NOT cover the forest, which Berlin
inventories by stand; inside Tegel and Spandau forest it has nothing.

Why a file and not a live WFS layer in QGIS: the project generator first referenced both
feature types as WFS layers and QGIS 3.44 reported them unavailable on load although every
request it makes (GetCapabilities, DescribeFeatureType, GML 3.2 GetFeature) succeeds from
this machine. The service declares its namespace as ``xmlns:baumbestand="baumbestand"``,
a bare word rather than a URL, which is a known trigger for QGIS WFS typename resolution
failing; it could not be verified headless here (PyQGIS does not load on this Mac). A
GeoPackage of ~21k points is tiny, loads through the OGR provider like everything else in
the project, works offline, and is what the comparison code would read anyway.

    python benchmark/fetch_berlin_trees.py \\
        --bbox 374000 5826000 384000 5831000 \\
        --out /Volumes/2TB/winmol/ALS_Data/berlin_trees/baumbestand_berlin.gpkg

Output: layers ``strassenbaeume`` and ``anlagenbaeume`` (EPSG:25833, one row per tree, all
attributes the service returns) plus ``fetch`` (bbox, counts, date, service, typename).
Paged with ``COUNT``/``STARTINDEX`` so no single request is large.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import urllib.parse
import urllib.request
from pathlib import Path

WFS = "https://gdi.berlin.de/services/wfs/baumbestand"
TYPENAMES = {"strassenbaeume": "baumbestand:strassenbaeume",
             "anlagenbaeume": "baumbestand:anlagenbaeume"}
EPSG = 25833


def get_json(params: dict, timeout: float = 120.0) -> dict:
    url = WFS + "?" + urllib.parse.urlencode(params)
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.load(r)


def fetch_typename(typename: str, bbox, page: int = 5000, log=print) -> list[dict]:
    base = {"SERVICE": "WFS", "VERSION": "2.0.0", "REQUEST": "GetFeature", "TYPENAMES": typename,
            "SRSNAME": f"EPSG:{EPSG}", "OUTPUTFORMAT": "application/json",
            "BBOX": ",".join(str(v) for v in bbox) + f",EPSG:{EPSG}"}
    features, start = [], 0
    while True:
        chunk = get_json({**base, "COUNT": page, "STARTINDEX": start})
        got = chunk.get("features", [])
        features.extend(got)
        total = chunk.get("numberMatched")
        log(f"  {typename}: {len(features)}{'/' + str(total) if total is not None else ''}")
        if len(got) < page:
            break
        start += page
    return features


def to_gdf(features: list[dict]):
    import geopandas as gpd
    from shapely.geometry import shape

    rows = [dict(f["properties"], geometry=shape(f["geometry"])) for f in features if f.get("geometry")]
    return gpd.GeoDataFrame(rows, geometry="geometry", crs=f"EPSG:{EPSG}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bbox", nargs=4, type=float, required=True, metavar=("MINX", "MINY", "MAXX", "MAXY"),
                    help="EPSG:25833 metres")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--page-size", type=int, default=5000)
    a = ap.parse_args(argv)
    import geopandas as gpd

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.unlink(missing_ok=True)
    meta = []
    for layer, typename in TYPENAMES.items():
        feats = fetch_typename(typename, a.bbox, a.page_size)
        gdf = to_gdf(feats)
        gdf.to_file(a.out, layer=layer, driver="GPKG")
        print(f"wrote {layer}: {len(gdf)} trees, {len(gdf.columns) - 1} attributes")
        meta.append({"layer": layer, "typename": typename, "n": len(gdf), "service": WFS,
                     "bbox": list(a.bbox), "epsg": EPSG, "fetched": dt.date.today().isoformat(),
                     "licence": "dl-de/zero-2-0"})
    gpd.GeoDataFrame(meta, geometry=[None] * len(meta), crs=f"EPSG:{EPSG}").to_file(a.out, layer="fetch", driver="GPKG")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
