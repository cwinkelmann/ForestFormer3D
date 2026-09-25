#!/usr/bin/env python3
"""Fetch Berlin ALKIS building footprints (Gebäude) from the Berlin GDI WFS.

Why this exists: the Berlin ALS 2021 point clouds carry **no building class**.
Measured on three km tiles, the classes present are 2 (ground), 3, 4, 5, 7 and 32 --
class 6 (building) is absent and roof points sit in the vegetation classes 3/4/5.
ForestFormer3D therefore happily predicts tree instances on roofs.  The official
building footprints are the fix: ``ff3d_geo.buildings`` masks every predicted point
that falls inside one out of the tree instances.

Service (checked 2026-09-23, official source, not OSM)::

    https://gdi.berlin.de/services/wfs/alkis_gebaeude
    typename      alkis_gebaeude:gebaeude   (Title "Gebäude")
    CRS           EPSG:25833 (DefaultCRS), 25832, 4258, 4326, 3857
    outputFormat  application/json (GeoJSON), plus GML 2/3.1.1/3.2
    CountDefault  1000000  -- but this script pages anyway (``--page-size``)

Attributes kept (see DescribeFeatureType): ``uuid`` (ALKIS identifier), ``gfk`` /
``bezgfk`` (Gebäudefunktion, e.g. 1010 "Wohnhaus"), ``bat`` / ``bezbat`` (Bauart),
``baw`` / ``bezbaw`` (Bauweise), ``nam`` and ``shape_area``.

One GeoPackage per run holds everything: layer ``buildings`` (the footprints,
deduplicated by ``uuid``, EPSG:25833) and layer ``tiles`` (one row per km tile
already fetched, so a re-run skips it).  Delete the GeoPackage to force a refetch.

Examples
--------
    python benchmark/fetch_berlin_buildings.py --tiles all \
        --out /Volumes/2TB/winmol/ALS_Data/berlin_buildings/alkis_buildings.gpkg

    python benchmark/fetch_berlin_buildings.py --bbox 381000 5829000 382000 5830000 \
        --out /tmp/buildings.gpkg
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

WFS_ROOT = "https://gdi.berlin.de/services/wfs"
DEFAULT_SERVICE = "alkis_gebaeude"
DEFAULT_TYPENAME = "alkis_gebaeude:gebaeude"
USER_AGENT = "ForestFormer3D-buildings-fetch/1.0 (+benchmark/fetch_berlin_buildings.py)"
EPSG = 25833
CRS_URN = f"urn:ogc:def:crs:EPSG::{EPSG}"

BUILDINGS_LAYER = "buildings"
TILES_LAYER = "tiles"

# ALKIS attributes worth keeping: the identifier plus the function/type fields.
KEEP_FIELDS = [
    "uuid",  # ALKIS object identifier
    "gfk", "bezgfk",  # Gebäudefunktion (code / text)
    "bat", "bezbat",  # Bauart
    "baw", "bezbaw",  # Bauweise
    "nam",  # Name
    "shape_area",
]

# The eleven Tegel km tiles of the Berlin ALS 2021 benchmark subset ...
TEGEL_TILES = [
    "379_5828", "379_5829",
    "380_5828", "380_5829",
    "381_5828", "381_5829", "381_5830",
    "382_5828", "382_5829",
    "383_5828", "383_5829",
]
# ... and the eight Revier 13 (Spandau) tiles.
R13_TILES = [f"{e}_{n}" for e in ("374", "375", "376", "377") for n in ("5827", "5828")]
TILE_GROUPS = {
    "tegel": TEGEL_TILES,
    "r13": R13_TILES,
    "all": TEGEL_TILES + R13_TILES,
}


def tile_bounds(tile: str) -> tuple[float, float, float, float]:
    """Lower-left corner from the tile key ``<E km>_<N km>``; 1 km square."""
    e_km, n_km = tile.split("_")
    minx = int(e_km) * 1000.0
    miny = int(n_km) * 1000.0
    return minx, miny, minx + 1000.0, miny + 1000.0


def expand_tiles(values: list[str]) -> list[str]:
    """``--tiles`` accepts km-tile keys and the group names in ``TILE_GROUPS``."""
    out: list[str] = []
    for value in values:
        for tile in TILE_GROUPS.get(value, [value]):
            if tile not in out:
                out.append(tile)
    return out


def http_get(url: str, retries: int = 3, timeout: float = 180.0) -> bytes:
    delay = 2.0
    last: Exception | None = None
    for attempt in range(retries + 1):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.read()
        except (urllib.error.URLError, TimeoutError, OSError) as exc:  # pragma: no cover - network
            last = exc
            if attempt == retries:
                break
            time.sleep(delay)
            delay *= 2
    raise RuntimeError(f"WFS request failed after {retries + 1} attempts: {url}\n  {last}")


def get_feature_url(root: str, service: str, typename: str, bbox, page_size: int,
                    start_index: int) -> str:
    params = {
        "service": "WFS",
        "version": "2.0.0",
        "request": "GetFeature",
        "typenames": typename,
        "srsName": CRS_URN,
        "outputFormat": "application/json",
        "count": str(page_size),
        "startindex": str(start_index),
        # The bbox CRS is spelled out so the axis order cannot be guessed wrong.
        "bbox": "%.3f,%.3f,%.3f,%.3f,%s" % (bbox[0], bbox[1], bbox[2], bbox[3], CRS_URN),
    }
    return f"{root}/{service}?" + urllib.parse.urlencode(params)


def fetch_bbox(bbox, root: str = WFS_ROOT, service: str = DEFAULT_SERVICE,
               typename: str = DEFAULT_TYPENAME, page_size: int = 5000,
               fetch=http_get, log=print) -> list[dict]:
    """All GeoJSON features intersecting ``bbox``, paged with count/startindex.

    WFS 2.0 paging: a page that returns fewer features than requested is the last
    one; ``numberMatched`` is only used for the log line because some servers report
    ``unknown``.
    """
    features: list[dict] = []
    start_index = 0
    while True:
        url = get_feature_url(root, service, typename, bbox, page_size, start_index)
        payload = json.loads(fetch(url).decode("utf-8"))
        page = payload.get("features", [])
        features.extend(page)
        matched = payload.get("numberMatched")
        if len(page) < page_size:
            break
        start_index += len(page)
        log(f"      page at startindex {start_index} (matched {matched})")
    return features


def feature_rows(features: list[dict], tile: str | None) -> tuple[list[dict], list]:
    """GeoJSON features -> (attribute rows, shapely geometries), dropping empties."""
    from shapely.geometry import shape

    rows, geoms = [], []
    for feature in features:
        geometry = feature.get("geometry")
        if not geometry:
            continue
        geom = shape(geometry)
        if geom.is_empty:
            continue
        props = feature.get("properties", {}) or {}
        row = {field: props.get(field) for field in KEEP_FIELDS}
        if row.get("uuid") in (None, ""):
            row["uuid"] = str(feature.get("id", ""))
        row["tile"] = tile or ""
        rows.append(row)
        geoms.append(geom)
    return rows, geoms


def _read_layer(gpkg: Path, layer: str):
    import geopandas as gpd

    if not gpkg.is_file():
        return None
    try:
        return gpd.read_file(str(gpkg), layer=layer)
    except Exception:
        return None


def fetch_tiles(tiles: list[str], out_gpkg: Path, root: str = WFS_ROOT,
                service: str = DEFAULT_SERVICE, typename: str = DEFAULT_TYPENAME,
                page_size: int = 5000, force: bool = False, bounds=None,
                fetch=http_get, log=print) -> dict:
    """Fetch every tile's footprints into one GeoPackage; skip tiles already in it.

    Returns ``{"fetched": {tile: n}, "skipped": [...], "n_buildings": int}``; the
    counts are the features returned for that tile *before* the cross-tile ``uuid``
    deduplication (a building on a tile border is returned for both tiles).

    ``bounds`` overrides ``tile_bounds`` for the given keys (used by ``--bbox``).
    """
    import geopandas as gpd
    import pandas as pd
    from shapely.geometry import box

    bounds = dict(bounds or {})

    out_gpkg = Path(out_gpkg)
    out_gpkg.parent.mkdir(parents=True, exist_ok=True)

    existing = _read_layer(out_gpkg, BUILDINGS_LAYER)
    done = _read_layer(out_gpkg, TILES_LAYER)
    covered = set() if force or done is None else set(done["tile"].astype(str))

    new_frames = []
    tile_rows = []
    result = {"fetched": {}, "skipped": [], "n_buildings": 0}
    for tile in tiles:
        if tile in covered:
            log(f"  {tile}: already in {out_gpkg.name}, skipped")
            result["skipped"].append(tile)
            continue
        bbox = bounds.get(tile) or tile_bounds(tile)
        log(f"  {tile}: bbox {bbox[0]:.0f} {bbox[1]:.0f} {bbox[2]:.0f} {bbox[3]:.0f}")
        features = fetch_bbox(bbox, root=root, service=service, typename=typename,
                              page_size=page_size, fetch=fetch, log=log)
        rows, geoms = feature_rows(features, tile)
        log(f"  {tile}: {len(rows)} footprints")
        result["fetched"][tile] = len(rows)
        if rows:
            new_frames.append(gpd.GeoDataFrame(
                pd.DataFrame(rows), geometry=geoms, crs=f"EPSG:{EPSG}"))
        tile_rows.append({
            "tile": tile,
            "minx": bbox[0], "miny": bbox[1], "maxx": bbox[2], "maxy": bbox[3],
            "n_features": len(rows),
            "service": f"{root}/{service}",
            "typename": typename,
            "fetched_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        })

    if not new_frames and not tile_rows:
        result["n_buildings"] = 0 if existing is None else len(existing)
        return result

    frames = [f for f in ([existing] if existing is not None and not force else []) + new_frames
              if f is not None and len(f)]
    if frames:
        merged = gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), crs=f"EPSG:{EPSG}")
        # A footprint straddling a tile border comes back for both tiles.
        merged = merged.drop_duplicates(subset="uuid", keep="first").reset_index(drop=True)
        merged.to_file(str(out_gpkg), driver="GPKG", layer=BUILDINGS_LAYER)
        result["n_buildings"] = len(merged)

    # The coverage layer carries the tile square as its geometry, so "which tiles
    # are in this GeoPackage" is answerable in QGIS as well as by this script.
    tile_frame = gpd.GeoDataFrame(
        pd.DataFrame(tile_rows),
        geometry=[box(r["minx"], r["miny"], r["maxx"], r["maxy"]) for r in tile_rows],
        crs=f"EPSG:{EPSG}",
    )
    if done is not None and not force and len(done):
        tile_frame = gpd.GeoDataFrame(
            pd.concat([done, tile_frame], ignore_index=True), crs=f"EPSG:{EPSG}"
        ).drop_duplicates(subset="tile", keep="last").reset_index(drop=True)
    if len(tile_frame):
        tile_frame.to_file(str(out_gpkg), driver="GPKG", layer=TILES_LAYER)
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Fetch Berlin ALKIS building footprints (WFS) into one GeoPackage.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--tiles", nargs="+", metavar="KEY",
                        help="km-tile keys like 381_5829, or a group: "
                             + ", ".join(sorted(TILE_GROUPS)))
    source.add_argument("--bbox", nargs=4, type=float, metavar=("MINX", "MINY", "MAXX", "MAXY"),
                        help=f"one EPSG:{EPSG} bounding box instead of tile keys")
    parser.add_argument("--out", required=True, type=Path, help="output GeoPackage")
    parser.add_argument("--service", default=DEFAULT_SERVICE,
                        help="WFS service name under %s (default: %%(default)s)" % WFS_ROOT)
    parser.add_argument("--typename", default=DEFAULT_TYPENAME,
                        help="WFS feature type (default: %(default)s)")
    parser.add_argument("--root", default=WFS_ROOT, help=argparse.SUPPRESS)
    parser.add_argument("--page-size", type=int, default=5000,
                        help="features per GetFeature request (default: %(default)s)")
    parser.add_argument("--force", action="store_true",
                        help="refetch tiles already recorded in the GeoPackage")
    args = parser.parse_args(argv)

    if args.bbox is not None:
        key = "bbox_%.0f_%.0f_%.0f_%.0f" % tuple(args.bbox)
        tiles, bounds = [key], {key: tuple(args.bbox)}
    else:
        tiles, bounds = expand_tiles(args.tiles), None

    print(f"{len(tiles)} tile(s) -> {args.out}")
    info = fetch_tiles(tiles, args.out, root=args.root, service=args.service,
                       typename=args.typename, page_size=args.page_size,
                       force=args.force, bounds=bounds)
    total = sum(info["fetched"].values())
    print(f"fetched {total} footprints for {len(info['fetched'])} tile(s), "
          f"{len(info['skipped'])} skipped; {info['n_buildings']} unique in {args.out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
