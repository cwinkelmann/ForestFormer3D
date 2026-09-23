# tests/test_fetch_berlin_buildings.py
"""CPU-only, network-free tests for benchmark/fetch_berlin_buildings.py.

Covers the tile-key arithmetic, the GetFeature URL, WFS 2.0 count/startindex paging,
and the "tile already in the GeoPackage" skip.  Nothing here opens a socket: the
HTTP helper is injected.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pytest

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "fetch_berlin_buildings", REPO / "benchmark" / "fetch_berlin_buildings.py")
bld = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bld)


def _feature(uuid, x, y):
    return {
        "type": "Feature",
        "id": f"gebaeude.{uuid}",
        "properties": {"uuid": uuid, "gfk": 1010, "bezgfk": "Wohnhaus", "shape_area": 100.0},
        "geometry": {"type": "Polygon", "coordinates": [[
            [x, y], [x + 10, y], [x + 10, y + 10], [x, y + 10], [x, y]]]},
    }


def test_tile_bounds_and_groups():
    assert bld.tile_bounds("381_5829") == (381000.0, 5829000.0, 382000.0, 5830000.0)
    assert len(bld.TILE_GROUPS["tegel"]) == 11
    assert len(bld.TILE_GROUPS["r13"]) == 8
    assert bld.expand_tiles(["tegel", "381_5829"]) == bld.TEGEL_TILES  # no duplicate
    assert bld.expand_tiles(["r13", "999_9999"])[-1] == "999_9999"


def test_get_feature_url_carries_crs_count_and_bbox():
    url = bld.get_feature_url(bld.WFS_ROOT, bld.DEFAULT_SERVICE, bld.DEFAULT_TYPENAME,
                              (381000.0, 5829000.0, 382000.0, 5830000.0), 5000, 100)
    q = parse_qs(urlparse(url).query)
    assert q["typenames"] == ["alkis_gebaeude:gebaeude"]
    assert q["outputFormat"] == ["application/json"]
    assert q["srsName"] == ["urn:ogc:def:crs:EPSG::25833"]
    assert q["count"] == ["5000"] and q["startindex"] == ["100"]
    # The bbox spells out its CRS so the axis order cannot be guessed wrong.
    assert q["bbox"] == ["381000.000,5829000.000,382000.000,5830000.000,"
                         "urn:ogc:def:crs:EPSG::25833"]


def test_fetch_bbox_pages_until_a_short_page():
    calls = []

    def fake(url):
        start = int(parse_qs(urlparse(url).query)["startindex"][0])
        calls.append(start)
        n = 2 if start < 4 else 1  # page size 2 -> 2, 2, 1 and stop
        feats = [_feature(f"u{start + i}", 381000.0 + start + i, 5829000.0) for i in range(n)]
        return json.dumps({"features": feats, "numberMatched": 5}).encode()

    feats = bld.fetch_bbox((381000, 5829000, 382000, 5830000), page_size=2,
                           fetch=fake, log=lambda *a: None)
    assert calls == [0, 2, 4]
    assert len(feats) == 5


def test_fetch_tiles_writes_both_layers_and_skips_on_a_rerun(tmp_path):
    gpd = pytest.importorskip("geopandas")
    pytest.importorskip("shapely")

    def fake(url):
        q = parse_qs(urlparse(url).query)
        minx = float(q["bbox"][0].split(",")[0])
        # one distinct footprint per tile, plus one shared uuid on every tile
        feats = [_feature(f"u{minx:.0f}", minx + 10, 5829010.0),
                 _feature("shared", minx + 20, 5829020.0)]
        return json.dumps({"features": feats, "numberMatched": 2}).encode()

    out = tmp_path / "alkis.gpkg"
    info = bld.fetch_tiles(["381_5829", "382_5829"], out, page_size=5000,
                           fetch=fake, log=lambda *a: None)
    assert info["fetched"] == {"381_5829": 2, "382_5829": 2}
    # the "shared" uuid is deduplicated across the two tiles
    assert info["n_buildings"] == 3

    buildings = gpd.read_file(str(out), layer=bld.BUILDINGS_LAYER)
    assert buildings.crs.to_epsg() == 25833
    assert set(buildings["uuid"]) == {"u381000", "u382000", "shared"}
    assert "bezgfk" in buildings.columns

    tiles = gpd.read_file(str(out), layer=bld.TILES_LAYER)
    assert set(tiles["tile"]) == {"381_5829", "382_5829"}

    # A re-run adds only the new tile and leaves the old rows alone.
    again = bld.fetch_tiles(["381_5829", "383_5829"], out, page_size=5000,
                            fetch=fake, log=lambda *a: None)
    assert again["skipped"] == ["381_5829"]
    assert list(again["fetched"]) == ["383_5829"]
    tiles = gpd.read_file(str(out), layer=bld.TILES_LAYER)
    assert set(tiles["tile"]) == {"381_5829", "382_5829", "383_5829"}
    assert len(gpd.read_file(str(out), layer=bld.BUILDINGS_LAYER)) == 4
