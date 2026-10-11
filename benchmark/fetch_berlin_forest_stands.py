#!/usr/bin/env python3
"""Fetch Berlin's forest stand map (Forstbetriebskarte 2014, Umweltatlas) for a bbox.

The per-stand species inventory of the Berliner Forsten -- the only source for what grows
INSIDE the forest, where the tree cadastre (streets, parks) has nothing. Source (official,
open; capabilities checked 2026-10-05)::

    WFS 2.0     https://gdi.berlin.de/services/wfs/ua_forstbetriebskarte_2014
    typenames   ua_forstbetriebskarte_2014:c_hauptbaumarten   stands: main species by layer
                ua_forstbetriebskarte_2014:a_mischbaumarten   mixed-species detail
                ua_forstbetriebskarte_2014:b_forstverwalt     forest administration units
    licence     Datenlizenz Deutschland - Zero - 2.0 (dl-de/zero-2-0)
    catalogue   gdi.berlin.de record f15f6603-4640-3d64-9dfb-45575347a901
                "Alters- und Bestandesstruktur der Waelder - Forstbetriebskarte 2014"

Per stand (``c_hauptbaumarten``): ``best_dist`` (stand id), ``invgname`` (forest district,
e.g. Tegel), ``betrkl`` (e.g. Hochwald), ``gis_area`` (m2), ``grpalter`` (age class) and, for
each canopy layer ``s1`` (main), ``s2``, ``s3`` and ``ue`` (standards), up to five species
``<layer>_<k>_ba`` (code), ``_deuts`` (German name), ``_misch`` (share, %), ``_bhd`` (DBH,
cm), ``_hoehe`` (height, m). A stand is a polygon of typically 1-20 ha; the map is from the
2014 management inventory, so it predates the 2021 ALS by seven years.

    python benchmark/fetch_berlin_forest_stands.py \\
        --bbox 374000 5826000 384000 5831000 \\
        --out /Volumes/2TB/winmol/ALS_Data/berlin_forest/forstbetriebskarte_2014.gpkg

Output layers: ``hauptbaumarten`` (the stands), ``mischbaumarten``, ``forstverwalt``, and
``fetch`` (bbox, counts, date, service). Polygons are written as MultiPolygon.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import urllib.parse
import urllib.request
from pathlib import Path

WFS = "https://gdi.berlin.de/services/wfs/ua_forstbetriebskarte_2014"
TYPENAMES = {"hauptbaumarten": "ua_forstbetriebskarte_2014:c_hauptbaumarten",
             "mischbaumarten": "ua_forstbetriebskarte_2014:a_mischbaumarten",
             "forstverwalt": "ua_forstbetriebskarte_2014:b_forstverwalt"}
EPSG = 25833


def get_json(params: dict, timeout: float = 180.0) -> dict:
    with urllib.request.urlopen(WFS + "?" + urllib.parse.urlencode(params), timeout=timeout) as r:
        return json.load(r)


def fetch_typename(typename: str, bbox, page: int = 1000, log=print) -> list[dict]:
    base = {"SERVICE": "WFS", "VERSION": "2.0.0", "REQUEST": "GetFeature", "TYPENAMES": typename,
            "SRSNAME": f"EPSG:{EPSG}", "OUTPUTFORMAT": "application/json",
            "BBOX": ",".join(str(v) for v in bbox) + f",EPSG:{EPSG}"}
    feats, start = [], 0
    while True:
        chunk = get_json({**base, "COUNT": page, "STARTINDEX": start})
        got = chunk.get("features", [])
        feats.extend(got)
        log(f"  {typename}: {len(feats)}/{chunk.get('numberMatched', '?')}")
        if len(got) < page:
            return feats
        start += page


def to_gdf(features: list[dict]):
    import geopandas as gpd
    from shapely.geometry import shape
    from shapely.geometry import MultiPolygon, Polygon

    rows = []
    for f in features:
        if not f.get("geometry"):
            continue
        g = shape(f["geometry"])
        if isinstance(g, Polygon):
            g = MultiPolygon([g])
        rows.append(dict(f["properties"], geometry=g))
    return gpd.GeoDataFrame(rows, geometry="geometry", crs=f"EPSG:{EPSG}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bbox", nargs=4, type=float, required=True, metavar=("MINX", "MINY", "MAXX", "MAXY"))
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    import geopandas as gpd

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.unlink(missing_ok=True)
    meta = []
    for layer, typename in TYPENAMES.items():
        gdf = to_gdf(fetch_typename(typename, a.bbox))
        if len(gdf):
            gdf.to_file(a.out, layer=layer, driver="GPKG")
        print(f"wrote {layer}: {len(gdf)} features, {len(gdf.columns) - 1} attributes")
        meta.append({"layer": layer, "typename": typename, "n": len(gdf), "service": WFS, "bbox": list(a.bbox),
                     "epsg": EPSG, "fetched": dt.date.today().isoformat(), "licence": "dl-de/zero-2-0",
                     "inventory_year": 2014})
    gpd.GeoDataFrame(meta, geometry=[None] * len(meta), crs=f"EPSG:{EPSG}").to_file(a.out, layer="fetch", driver="GPKG")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
