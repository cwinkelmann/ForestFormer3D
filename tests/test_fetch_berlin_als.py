# tests/test_fetch_berlin_als.py
"""CPU-only, network-free tests for benchmark/fetch_berlin_als.py.

Covers the tile-name <-> key mapping, the bbox -> tile-key arithmetic, the ATOM
feed parsing (service feed -> dataset feed -> the regional ZIP links), the LAS
header reader and the member-index cache.  Nothing here opens a socket: the two
HTTP helpers are monkeypatched.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "fetch_berlin_als", REPO / "benchmark" / "fetch_berlin_als.py")
als = importlib.util.module_from_spec(spec)
spec.loader.exec_module(als)

SERVICE_FEED = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <link href="https://example.invalid/a_als/atom/0.atom" rel="alternate"
          type="application/atom+xml"/>
  </entry>
</feed>
"""

DATASET_FEED = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry><link href="https://example.invalid/a_als/atom/Blattschnitt.gif"/></entry>
  <entry><link href="https://example.invalid/a_als/atom/Nordwest.zip"/></entry>
  <entry><link href="https://example.invalid/a_als/atom/West.zip"/></entry>
  <entry><link href="https://example.invalid/a_als/atom/West.zip"/></entry>
</feed>
"""


def test_tile_key_roundtrip():
    assert als.tile_key("3dm_33_381_5829_1_be.las") == "381_5829"
    assert als.tile_key("some/dir/3dm_33_380_5828_1_be.las") == "380_5828"
    assert als.tile_name("381_5829") == "3dm_33_381_5829_1_be.las"
    for junk in ("readme.txt", "3dm_33_381_5829_1_be.laz", "als_33381-5829.laz"):
        assert als.tile_key(junk) is None


def test_keys_from_bbox_covers_every_intersecting_tile():
    # A box strictly inside one km square selects exactly that tile.
    assert als.keys_from_bbox([381100, 5829100, 381900, 5829900]) == ["381_5829"]
    # A 2x2 km box on the lattice selects four tiles, not nine: the upper bounds
    # are exclusive, so 383000/5831000 do not pull in the next row/column.
    assert als.keys_from_bbox([381000, 5829000, 383000, 5831000]) == [
        "381_5829", "381_5830", "382_5829", "382_5830"]
    with pytest.raises(SystemExit):
        als.keys_from_bbox([383000, 5829000, 381000, 5831000])


def test_package_urls_follows_the_dataset_feed(monkeypatch):
    pages = {
        "https://example.invalid/a_als/atom": SERVICE_FEED,
        "https://example.invalid/a_als/atom/0.atom": DATASET_FEED,
    }
    monkeypatch.setattr(als, "http_get",
                        lambda url, timeout=0.0: pages[url].encode())
    urls = als.package_urls("https://example.invalid/a_als/atom")
    assert urls == ["https://example.invalid/a_als/atom/Nordwest.zip",
                    "https://example.invalid/a_als/atom/West.zip"]


def test_las_header_reads_version_and_count(tmp_path):
    hdr = bytearray(375)
    hdr[0:4] = b"LASF"
    hdr[24], hdr[25] = 1, 4
    hdr[247:255] = (4_616_907).to_bytes(8, "little")
    path = tmp_path / "t.las"
    path.write_bytes(bytes(hdr))
    assert als.las_header(str(path)) == ("1.4", 4_616_907)

    bad = tmp_path / "bad.las"
    bad.write_bytes(b"NOPE" + bytes(371))
    with pytest.raises(ValueError):
        als.las_header(str(bad))


def test_build_index_uses_and_invalidates_the_cache(tmp_path, monkeypatch):
    cache = tmp_path / ".als_index.json"
    cache.write_text(json.dumps({
        "packages": ["https://example.invalid/Nordwest.zip"],
        "tiles": {"381_5829": {"zip": "https://example.invalid/Nordwest.zip",
                               "member": "3dm_33_381_5829_1_be.las",
                               "size": 883373519}},
    }))

    def boom(url):  # any network access is a test failure
        raise AssertionError(f"should not have opened {url}")

    monkeypatch.setattr(als, "RemoteFile", boom)
    index = als.build_index(["https://example.invalid/Nordwest.zip"], str(cache))
    assert index["381_5829"]["size"] == 883373519

    # A different package list must not be served from that cache.
    with pytest.raises(AssertionError):
        als.build_index(["https://example.invalid/West.zip"], str(cache))
