"""The QGIS project generator (benchmark/make_qgis_project.py): the pure parts.

The derived phase shells out to gdalbuildvrt/ogr2ogr and the project phase needs the
2TB data, so neither runs here; what is tested is everything in between -- the km-tile
geometry the extents come from, the layer ids, and that the XML the generator emits is
well-formed and references only layers it defined. A `.qgz` that QGIS cannot parse is a
blank window with no error, so well-formedness is the one check worth having.
"""

import sys
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmark"))

from make_qgis_project import (  # noqa: E402
    CYCLIC,
    SEMANTIC,
    Tree,
    esc,
    km_bounds,
    layer_id,
    raster_layer_xml,
    renderer_multiband,
    renderer_paletted,
    renderer_pseudocolor,
    symbol_fill,
    symbol_marker,
    union_bounds,
    vector_layer_xml,
    write_qgz,
)

FAKE_WKT = 'PROJCRS["x",BASEGEOGCRS["y"]]'


def test_km_bounds_come_from_the_tile_name():
    assert km_bounds("3dm_33_380_5828_1_be") == (380000.0, 5828000.0, 381000.0, 5829000.0)
    with pytest.raises(ValueError):
        km_bounds("r12_tegel_E381300_N5828300_100m")


def test_union_bounds_spans_the_block():
    b = union_bounds(["3dm_33_374_5827_1_be", "3dm_33_375_5828_1_be", "3dm_33_383_5830_1_be"])
    assert b == (374000.0, 5827000.0, 384000.0, 5831000.0)


def test_layer_ids_are_stable_and_xml_safe():
    a, b = layer_id("ff3d crowns"), layer_id("ff3d crowns")
    assert a == b and " " not in a
    assert layer_id("ff3d crowns") != layer_id("sat crowns")


def test_esc_handles_every_xml_special():
    assert esc('a<b>&"c"') == "a&lt;b&gt;&amp;&quot;c&quot;"


def _parse(fragment: str) -> ET.Element:
    return ET.fromstring(fragment)


def test_renderers_are_well_formed():
    assert _parse(renderer_pseudocolor(0, 1000, CYCLIC)).get("type") == "singlebandpseudocolor"
    items = _parse(renderer_pseudocolor(0, 1000, CYCLIC)).findall(".//item")
    assert len(items) == len(CYCLIC)
    assert float(items[0].get("value")) == 0.0 and float(items[-1].get("value")) == 1000.0
    pal = _parse(renderer_paletted(SEMANTIC)).findall(".//paletteEntry")
    assert [e.get("value") for e in pal] == ["0", "1", "2", "3"]
    mb = _parse(renderer_multiband(4, 1, 2))
    assert (mb.get("redBand"), mb.get("greenBand"), mb.get("blueBand")) == ("4", "1", "2")


def test_vector_symbols_carry_the_data_defined_size():
    m = _parse(symbol_marker("255,255,255,255", '"height" / 2'))
    expr = [o for o in m.iter("Option") if o.get("name") == "expression"]
    assert expr and expr[0].get("value") == '"height" / 2'
    f = _parse(symbol_fill("255,212,121,255"))
    assert [o.get("value") for o in f.iter("Option") if o.get("name") == "style"] == ["no"]


def test_layers_parse_and_carry_source_and_extent():
    b = km_bounds("3dm_33_380_5828_1_be")
    ll = (13.2, 52.6, 13.3, 52.7)
    r = _parse(raster_layer_xml("r1", "inst", "./derived/x.vrt", renderer_multiband(), b, ll, FAKE_WKT))
    assert r.findtext("datasource") == "./derived/x.vrt" and r.findtext("provider") == "gdal"
    assert float(r.find("extent/xmin").text) == 380000.0
    v = _parse(vector_layer_xml("v1", "crowns", "./derived/c.gpkg|layername=crowns", "Polygon",
                                symbol_fill("0,0,0,255"), b, ll, FAKE_WKT))
    assert v.get("geometry") == "Polygon" and v.findtext("provider") == "ogr"
    # a live WFS layer (the Berlin tree cadastre) carries its request string as the datasource
    src = "restrictToRequestBBOX='1' srsname='EPSG:25833' typename='baumbestand:strassenbaeume' url='https://gdi.berlin.de/services/wfs/baumbestand' version='2.0.0'"
    w = _parse(vector_layer_xml("w1", "street trees", src, "Point", symbol_marker("0,0,0,255"), b, ll, FAKE_WKT,
                                provider="WFS"))
    assert w.findtext("provider") == "WFS" and w.findtext("datasource") == src


def test_project_references_only_layers_it_defines(tmp_path):
    t = Tree("test project", FAKE_WKT)
    b = km_bounds("3dm_33_380_5828_1_be")
    ll = (13.2, 52.6, 13.3, 52.7)

    def body():
        t.layer(raster_layer_xml("lid_a", "a", "./a.vrt", renderer_multiband(), b, ll, FAKE_WKT),
                "lid_a", "a", "./a.vrt", "gdal", True)
        t.layer(vector_layer_xml("lid_b", "b", "./b.gpkg|layername=b", "Point",
                                 symbol_marker("0,0,0,255"), b, ll, FAKE_WKT),
                "lid_b", "b", "./b.gpkg|layername=b", "ogr", False)

    t.group("G", True, True, body)
    xml = t.render(b, ll)
    root = ET.fromstring(xml.split("\n", 1)[1])          # drop the DOCTYPE line
    defined = {m.findtext("id") for m in root.iter("maplayer")}
    referenced = {n.get("id") for n in root.iter("layer-tree-layer")}
    ordered = {n.get("id") for n in root.find("layerorder")}
    assert defined == referenced == ordered == {"lid_a", "lid_b"}
    assert root.find("properties/Paths/Absolute").text == "false"
    assert root.find("projectCrs/spatialrefsys/authid").text == "EPSG:25833"

    out = tmp_path / "p.qgz"
    write_qgz(out, xml)
    with zipfile.ZipFile(out) as z:
        assert z.namelist() == ["p.qgs"]
        assert z.read("p.qgs").decode() == xml
