#!/usr/bin/env python3
"""Fetch Berlin ALS 2021 point-cloud km tiles (``3dm_33_<E>_<N>_1_be.las``).

Source: Senatsverwaltung fuer Stadtentwicklung Berlin, "Airborne Laserscanning
(ALS) Primaere 3D Laserscan-Daten", published as an INSPIRE ATOM download
service.  The service feed is

    https://gdi.berlin.de/data/a_als/atom

and its single dataset feed (``.../atom/0.atom``) links nine **regional ZIP
packages** -- ``Mitte.zip``, ``Nord.zip``, ``Nordost.zip``, ``Nordwest.zip``,
``Ost.zip``, ``Sued.zip``, ``Suedost.zip``, ``Suedwest.zip``, ``West.zip`` --
each several GB and each holding the 1 km LAS tiles of that city region as
*stored* (uncompressed) zip members named ``3dm_33_<E>_<N>_1_be.las``.

There is no per-tile URL.  This script therefore opens each package over HTTP
**range requests** (``RemoteZipFile`` below), reads only its central directory,
and copies out the members that were asked for -- no multi-GB package is ever
downloaded in full.  The member index is cached in ``<out-dir>/.als_index.json``
so later runs cost one HTTP request per tile.

Licence of the data: Datenlizenz Deutschland - Zero - Version 2.0
(https://www.govdata.de/dl-de/zero-2-0).
Metadata record:
https://gdi.berlin.de/geonetwork/srv/api/records/85a97801-36bb-4627-8790-f5e53b1e38b3

Examples
--------
    # what the feed offers and where each requested tile lives
    python benchmark/fetch_berlin_als.py --list --tiles 381_5829 380_5828

    # two named tiles
    python benchmark/fetch_berlin_als.py --tiles 381_5829 381_5830 \
        --out-dir /Volumes/2TB/winmol/ALS_Data/berlin_als_2021

    # everything intersecting an EPSG:25833 bounding box (metres)
    python benchmark/fetch_berlin_als.py --bbox 379000 5828000 383000 5830000 \
        --out-dir /Volumes/2TB/winmol/ALS_Data/berlin_als_2021
"""

from __future__ import annotations

import argparse
import io
import json
import os
import re
import struct
import sys
import time
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
import zipfile

SERVICE_FEED = "https://gdi.berlin.de/data/a_als/atom"
USER_AGENT = "ForestFormer3D-als-fetch/1.0 (+benchmark/fetch_berlin_als.py)"
ATOM_NS = "{http://www.w3.org/2005/Atom}"
TILE_RE = re.compile(r"^3dm_33_(\d{3})_(\d{4})_1_be\.las$")


def tile_key(name: str) -> str | None:
    """``3dm_33_381_5829_1_be.las`` -> ``381_5829`` (None if not a tile name)."""
    m = TILE_RE.match(os.path.basename(name))
    return f"{m.group(1)}_{m.group(2)}" if m else None


def tile_name(key: str) -> str:
    return f"3dm_33_{key}_1_be.las"


def keys_from_bbox(bbox) -> list[str]:
    """Tile keys of every 1 km tile intersecting an EPSG:25833 bbox in metres."""
    minx, miny, maxx, maxy = (float(v) for v in bbox)
    if maxx <= minx or maxy <= miny:
        raise SystemExit("--bbox needs minE minN maxE maxN with maxE>minE, maxN>minN")
    out = []
    for e in range(int(minx // 1000), int((maxx - 1e-9) // 1000) + 1):
        for n in range(int(miny // 1000), int((maxy - 1e-9) // 1000) + 1):
            out.append(f"{e}_{n}")
    return sorted(out)


def http_get(url: str, timeout: float = 120.0, headers=None) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, **(headers or {})})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def package_urls(feed: str = SERVICE_FEED, timeout: float = 120.0) -> list[str]:
    """Service feed -> dataset feed -> the regional ZIP links, in feed order."""
    root = ET.fromstring(http_get(feed, timeout))
    dataset_feeds = [
        link.get("href")
        for entry in root.findall(f"{ATOM_NS}entry")
        for link in entry.findall(f"{ATOM_NS}link")
        if link.get("href", "").endswith(".atom")
    ]
    if not dataset_feeds:
        raise SystemExit(f"no dataset feed linked from {feed}")
    urls: list[str] = []
    for df in dataset_feeds:
        sub = ET.fromstring(http_get(df, timeout))
        for link in sub.iter(f"{ATOM_NS}link"):
            href = link.get("href", "")
            if href.lower().endswith(".zip") and href not in urls:
                urls.append(href)
    if not urls:
        raise SystemExit(f"no .zip packages linked from {dataset_feeds}")
    return urls


class RemoteFile(io.RawIOBase):
    """Seekable read-only file over HTTP range requests (for ``zipfile``)."""

    def __init__(self, url: str, chunk: int = 8 << 20, retries: int = 6,
                 timeout: float = 300.0):
        self.url, self.chunk, self.retries, self.timeout = url, chunk, retries, timeout
        self.pos = 0
        head = urllib.request.Request(url, method="HEAD",
                                      headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(head, timeout=timeout) as r:
            self.size = int(r.headers["Content-Length"])
            if (r.headers.get("Accept-Ranges") or "").lower() != "bytes":
                raise SystemExit(f"{url} does not advertise byte ranges")
        self._cache_start, self._cache = -1, b""

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=0):
        self.pos = {0: off, 1: self.pos + off, 2: self.size + off}[whence]
        return self.pos

    def _fetch(self, start: int, n: int) -> bytes:
        delay = 3.0
        for attempt in range(self.retries + 1):
            try:
                req = urllib.request.Request(
                    self.url,
                    headers={"User-Agent": USER_AGENT,
                             "Range": f"bytes={start}-{start + n - 1}"})
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    data = r.read()
                if len(data) == n:
                    return data
                raise OSError(f"short range read: {len(data)} of {n}")
            except (OSError, urllib.error.URLError) as exc:
                if attempt == self.retries:
                    raise
                print(f"    range retry {attempt + 1}: {str(exc)[:90]}", flush=True)
                time.sleep(delay)
                delay = min(60.0, delay * 2)
        raise AssertionError("unreachable")

    def read(self, n: int = -1) -> bytes:
        if n < 0:
            n = self.size - self.pos
        n = min(n, self.size - self.pos)
        if n <= 0:
            return b""
        if not (self._cache_start <= self.pos
                and self.pos + n <= self._cache_start + len(self._cache)):
            want = min(max(n, self.chunk), self.size - self.pos)
            self._cache_start, self._cache = self.pos, self._fetch(self.pos, want)
        off = self.pos - self._cache_start
        out = self._cache[off:off + n]
        self.pos += len(out)
        return out


def build_index(urls, cache_path: str | None, refresh: bool = False) -> dict:
    """{tile key: {"zip": url, "member": name, "size": bytes}} over all packages."""
    if cache_path and not refresh and os.path.exists(cache_path):
        try:
            with open(cache_path) as fh:
                cached = json.load(fh)
            if cached.get("packages") == list(urls):
                return cached["tiles"]
        except (OSError, ValueError, KeyError):
            pass
    tiles: dict[str, dict] = {}
    for url in urls:
        print(f"[index] {url}", flush=True)
        zf = zipfile.ZipFile(RemoteFile(url))
        n = 0
        for info in zf.infolist():
            key = tile_key(info.filename)
            if key:
                tiles[key] = {"zip": url, "member": info.filename,
                              "size": info.file_size}
                n += 1
        print(f"[index]   {n} tiles", flush=True)
    if cache_path:
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        tmp = cache_path + ".tmp"
        with open(tmp, "w") as fh:
            json.dump({"packages": list(urls), "tiles": tiles}, fh, indent=1)
        os.replace(tmp, cache_path)
    return tiles


def las_header(path: str) -> tuple[str, int]:
    """(version, point count) of a LAS file; raises on a bad magic."""
    with open(path, "rb") as fh:
        hdr = fh.read(375)
    if hdr[:4] != b"LASF":
        raise ValueError(f"{path}: not a LAS file (magic {hdr[:4]!r})")
    ver = f"{hdr[24]}.{hdr[25]}"
    if hdr[25] >= 4:
        n = struct.unpack("<Q", hdr[247:255])[0]
    else:
        n = struct.unpack("<I", hdr[107:111])[0]
    return ver, n


def fetch_tile(entry: dict, out_path: str, zips: dict) -> tuple[bool, str]:
    """Copy one member out of its remote package. Returns (downloaded?, note)."""
    want = entry["size"]
    if os.path.exists(out_path):
        have = os.path.getsize(out_path)
        if have == want:
            return False, f"have ({have / 1e6:.0f} MB)"
        print(f"    size mismatch ({have} != {want}), refetching", flush=True)
    url = entry["zip"]
    if url not in zips:
        zips[url] = zipfile.ZipFile(RemoteFile(url))
    t0 = time.time()
    tmp = out_path + ".part"
    with zips[url].open(entry["member"]) as src, open(tmp, "wb") as dst:
        while True:
            chunk = src.read(16 << 20)
            if not chunk:
                break
            dst.write(chunk)
    got = os.path.getsize(tmp)
    if got != want:
        os.remove(tmp)
        raise OSError(f"{entry['member']}: wrote {got} bytes, expected {want}")
    os.replace(tmp, out_path)
    ver, npts = las_header(out_path)
    return True, (f"{got / 1e6:.0f} MB in {time.time() - t0:.0f} s, "
                  f"LAS {ver}, {npts:,} points")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tiles", nargs="+", metavar="E_N",
                    help="tile keys like 381_5829 (km easting _ km northing)")
    ap.add_argument("--bbox", nargs=4, metavar=("minE", "minN", "maxE", "maxN"),
                    help="EPSG:25833 bounding box in metres; every intersecting "
                         "1 km tile is selected")
    ap.add_argument("--out-dir", help="destination directory (required unless --list)")
    ap.add_argument("--list", action="store_true",
                    help="print the ATOM packages and the resolved tiles, download nothing")
    ap.add_argument("--feed", default=SERVICE_FEED, help=f"ATOM service feed (default: {SERVICE_FEED})")
    ap.add_argument("--index-cache", default=None,
                    help="member-index JSON (default: <out-dir>/.als_index.json, "
                         "or ./.als_index.json with --list and no --out-dir)")
    ap.add_argument("--refresh-index", action="store_true",
                    help="re-read every package's central directory")
    args = ap.parse_args(argv)

    if not args.list and not args.out_dir:
        ap.error("--out-dir is required unless --list is given")
    if args.tiles and args.bbox:
        ap.error("--tiles and --bbox are mutually exclusive")

    keys = None
    if args.bbox:
        keys = keys_from_bbox(args.bbox)
    elif args.tiles:
        keys = sorted({k.strip() for k in args.tiles})
        bad = [k for k in keys if not re.fullmatch(r"\d{3}_\d{4}", k)]
        if bad:
            ap.error(f"tile keys must look like 381_5829: {bad}")

    urls = package_urls(args.feed)
    cache = args.index_cache or os.path.join(args.out_dir or ".", ".als_index.json")
    index = build_index(urls, cache, refresh=args.refresh_index)

    if args.list:
        print(f"\n{len(urls)} ATOM packages, {len(index)} tiles total:")
        for u in urls:
            n = sum(1 for e in index.values() if e["zip"] == u)
            print(f"  {os.path.basename(u):<14} {n:5d} tiles  {u}")
        if keys is None:
            keys = sorted(index)
        print(f"\n{len(keys)} requested tile(s):")
        for k in keys:
            e = index.get(k)
            print(f"  {tile_name(k):<26} "
                  + (f"{e['size'] / 1e6:8.0f} MB  {os.path.basename(e['zip'])}"
                     if e else "   NOT IN ANY PACKAGE"))
        return 0

    if keys is None:
        ap.error("give --tiles or --bbox (or use --list to see what exists)")

    os.makedirs(args.out_dir, exist_ok=True)
    zips: dict = {}
    got, skipped, missing, failed = [], [], [], []
    for k in keys:
        entry = index.get(k)
        name = tile_name(k)
        if entry is None:
            print(f"[miss] {name}: in no package (outside Berlin?)", file=sys.stderr)
            missing.append(k)
            continue
        out_path = os.path.join(args.out_dir, name)
        print(f"[fetch] {name} from {os.path.basename(entry['zip'])}", flush=True)
        try:
            downloaded, note = fetch_tile(entry, out_path, zips)
        except Exception as exc:  # noqa: BLE001
            print(f"[FAIL] {name}: {exc}", file=sys.stderr)
            failed.append((k, str(exc)))
            continue
        print(f"  {'[done]' if downloaded else '[skip]'} {name}: {note}", flush=True)
        (got if downloaded else skipped).append(k)

    print(f"\n{len(got)} downloaded, {len(skipped)} already present, "
          f"{len(missing)} not in any package, {len(failed)} failed")
    for k, err in failed:
        print(f"  FAILED {tile_name(k)}: {err}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
