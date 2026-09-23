#!/usr/bin/env python3
"""Fetch Berlin digital orthophotos (DOP20, 20 cm) from the Berlin GDI WMS.

The 2021 orthophotos live on the Berlin Geodateninfrastruktur WMS at
``https://gdi.berlin.de/services/wms/dop_2021`` (the older FIS-Broker
``fbinter.stadt-berlin.de/fb/wms/senstadt/<service>`` endpoints are gone -- they
return HTTP 404).  The service publishes two renderable layers:

    dop_2021_rgb   Digitale farbige Orthophotos 2021 (DOP20RGB)
    dop_2021_cir   Digitale Color-Infrarot-Orthophotos 2021 (DOP20CIR)

The underlying product is 4-band RGBI, but a WMS only ever serves rendered
3-band images, so a true RGBI GeoTIFF is assembled here by fetching both layers
and stacking R,G,B from the RGB layer with the NIR channel (band 1) of the CIR
rendering -- use ``--rgbi``.

One 1 km tile at 20 cm is 5000 x 5000 px; it is requested as ``--chunk`` sized
quadrants (2500 px -> four GetMap calls) and assembled into a single tiled,
LZW-compressed uint8 GeoTIFF in EPSG:25833.

Examples
--------
    python benchmark/fetch_berlin_dop.py --tiles 381_5829 \
        --out-dir /Volumes/2TB/winmol/ALS_Data/berlin_dop_2021/dop_2021_rgb

    python benchmark/fetch_berlin_dop.py --tiles all --rgbi \
        --out-dir /Volumes/2TB/winmol/ALS_Data/berlin_dop_2021/dop_2021_rgbi
"""

from __future__ import annotations

import argparse
import io
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

import numpy as np
import rasterio
from rasterio.transform import from_origin

WMS_ROOT = "https://gdi.berlin.de/services/wms"
USER_AGENT = "ForestFormer3D-dop-fetch/1.0 (+benchmark/fetch_berlin_dop.py)"
CRS = "EPSG:25833"

# The eleven 1 km tiles of the Berlin ALS 2021 benchmark subset.
DEFAULT_TILES = [
    "379_5828", "379_5829",
    "380_5828", "380_5829",
    "381_5828", "381_5829", "381_5830",
    "382_5828", "382_5829",
    "383_5828", "383_5829",
]


def tile_name(tile: str) -> str:
    return f"3dm_33_{tile}_1_be"


def tile_bounds(tile: str) -> tuple[float, float, float, float]:
    """Lower-left corner from the tile key ``<E km>_<N km>``; 1 km square."""
    e_km, n_km = tile.split("_")
    minx = int(e_km) * 1000.0
    miny = int(n_km) * 1000.0
    return minx, miny, minx + 1000.0, miny + 1000.0


def get_map(service: str, layer: str, bbox, width: int, height: int,
            fmt: str, retries: int, timeout: float) -> bytes:
    params = {
        "service": "WMS",
        "version": "1.3.0",
        "request": "GetMap",
        "layers": layer,
        "styles": "",
        "crs": CRS,
        "bbox": "%.3f,%.3f,%.3f,%.3f" % bbox,
        "width": str(width),
        "height": str(height),
        "format": fmt,
        "transparent": "FALSE",
    }
    url = f"{WMS_ROOT}/{service}?" + urllib.parse.urlencode(params)
    delay = 2.0
    last = None
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                ctype = resp.headers.get("Content-Type", "")
                data = resp.read()
            if not ctype.startswith("image/"):
                raise RuntimeError(
                    f"server returned {ctype!r}: {data[:300]!r}")
            return data
        except Exception as exc:  # noqa: BLE001 - retry anything transient
            last = exc
            if attempt == retries:
                break
            print(f"      retry {attempt + 1}/{retries} after {type(exc).__name__}: "
                  f"{str(exc)[:120]}", file=sys.stderr)
            time.sleep(delay)
            delay *= 2
    raise RuntimeError(f"GetMap failed after {retries} retries: {last}")


def decode(data: bytes) -> np.ndarray:
    """Decode a GetMap response into a (bands, rows, cols) uint8 array."""
    with rasterio.open(io.BytesIO(data)) as src:
        arr = src.read()
    if arr.dtype != np.uint8:
        arr = arr.astype(np.uint8)
    return arr


def fetch_tile_rgb(service: str, layer: str, tile: str, px: int, chunk: int,
                   fmt: str, retries: int, timeout: float,
                   sleep: float) -> np.ndarray:
    """Fetch one km tile as a (3, px, px) uint8 array, quadrant by quadrant."""
    minx, miny, maxx, maxy = tile_bounds(tile)
    res = (maxx - minx) / px
    out = None
    n = (px + chunk - 1) // chunk
    for row in range(n):
        for col in range(n):
            c0, r0 = col * chunk, row * chunk
            w = min(chunk, px - c0)
            h = min(chunk, px - r0)
            # row 0 is the top of the image -> highest northing
            sub = (minx + c0 * res, maxy - (r0 + h) * res,
                   minx + (c0 + w) * res, maxy - r0 * res)
            print(f"      quadrant r{row}c{col} {w}x{h} bbox={sub}")
            arr = decode(get_map(service, layer, sub, w, h, fmt, retries, timeout))
            if out is None:
                out = np.zeros((arr.shape[0], px, px), dtype=np.uint8)
            out[:, r0:r0 + h, c0:c0 + w] = arr[:, :h, :w]
            time.sleep(sleep)
    return out


def write_geotiff(path: str, arr: np.ndarray, tile: str) -> None:
    minx, miny, maxx, maxy = tile_bounds(tile)
    px = arr.shape[1]
    res = (maxx - minx) / px
    transform = from_origin(minx, maxy, res, res)
    tmp = path + ".part"
    with rasterio.open(
        tmp, "w", driver="GTiff", height=px, width=px, count=arr.shape[0],
        dtype="uint8", crs=CRS, transform=transform,
        compress="LZW", tiled=True, blockxsize=512, blockysize=512,
        predictor=2, BIGTIFF="IF_SAFER",
    ) as dst:
        dst.write(arr)
        names = ("red", "green", "blue", "nir")[:arr.shape[0]]
        dst.descriptions = names
    os.replace(tmp, path)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--year", default="2021",
                    help="orthophoto year; picks the default service (default: 2021)")
    ap.add_argument("--service", default=None,
                    help="WMS service id under gdi.berlin.de/services/wms "
                         "(default: dop_<year>)")
    ap.add_argument("--layer", default=None,
                    help="WMS layer name (default: <service>_rgb, or _cir with --cir)")
    ap.add_argument("--cir", action="store_true",
                    help="fetch the colour-infrared rendering instead of RGB")
    ap.add_argument("--rgbi", action="store_true",
                    help="fetch RGB + CIR and write a 4-band R,G,B,NIR GeoTIFF")
    ap.add_argument("--tiles", nargs="+", default=["all"],
                    help="tile keys like 381_5829, or 'all' for the benchmark set")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--px-per-tile", type=int, default=5000,
                    help="pixels per 1 km tile edge (5000 = 20 cm)")
    ap.add_argument("--chunk", type=int, default=2500,
                    help="max pixels per GetMap request edge")
    ap.add_argument("--format", default="image/geotiff",
                    help="GetMap format (image/geotiff, image/tiff, image/png)")
    ap.add_argument("--retries", type=int, default=4)
    ap.add_argument("--timeout", type=float, default=300.0)
    ap.add_argument("--sleep", type=float, default=1.0,
                    help="seconds between GetMap calls (be polite)")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args(argv)

    service = args.service or f"dop_{args.year}"
    rgb_layer = args.layer or f"{service}_rgb"
    cir_layer = f"{service}_cir"
    if args.cir and not args.layer:
        rgb_layer = cir_layer

    tiles = DEFAULT_TILES if args.tiles == ["all"] else args.tiles
    os.makedirs(args.out_dir, exist_ok=True)

    ok, failed, skipped = [], [], []
    for tile in tiles:
        name = tile_name(tile)
        out = os.path.join(args.out_dir, name + ".tif")
        if os.path.exists(out) and not args.overwrite:
            print(f"[skip] {out} exists")
            skipped.append(name)
            continue
        print(f"[fetch] {name} from {service}/{rgb_layer}")
        try:
            arr = fetch_tile_rgb(service, rgb_layer, tile, args.px_per_tile,
                                 args.chunk, args.format, args.retries,
                                 args.timeout, args.sleep)
            if args.rgbi:
                print(f"[fetch] {name} NIR from {service}/{cir_layer}")
                cir = fetch_tile_rgb(service, cir_layer, tile, args.px_per_tile,
                                     args.chunk, args.format, args.retries,
                                     args.timeout, args.sleep)
                arr = np.concatenate([arr[:3], cir[:1]], axis=0)
            write_geotiff(out, arr, tile)
            mb = os.path.getsize(out) / 1e6
            print(f"[done] {out}  {arr.shape}  {mb:.1f} MB")
            ok.append((name, mb))
        except Exception as exc:  # noqa: BLE001
            print(f"[FAIL] {name}: {exc}", file=sys.stderr)
            failed.append((name, str(exc)))

    print(f"\n{len(ok)} written, {len(skipped)} skipped, {len(failed)} failed")
    for name, mb in ok:
        print(f"  {name}.tif  {mb:.1f} MB")
    for name, err in failed:
        print(f"  FAILED {name}: {err}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
