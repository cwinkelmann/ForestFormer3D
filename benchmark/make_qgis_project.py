#!/usr/bin/env python3
"""Build a QGIS project over the Berlin ALS 2021 results, without PyQGIS.

Two phases, both plain Python plus the GDAL command-line tools:

  derived   mosaics every per-tile product of a method into ONE layer: a GDAL VRT per
            raster family (instance ids, semantic classes, orthophotos, terrain) and one
            GeoPackage per vector family (crowns, trees) with a ``tile`` column added.
            33 tiles x 4 products x 4 methods would otherwise be ~500 layers.
  project   writes ``berlin_als_2021.qgz`` (styled, grouped by method, EPSG:25833,
            relative paths) and ``berlin_als_2021_pointclouds.qgz`` (the LAS files as
            point-cloud layers, kept apart because QGIS indexes each one on first load).

Why not PyQGIS: on this Mac the QGIS 3.44 bundle's python refuses to load its own numpy
("mapping process and mapped file have different Team IDs" -- a code-signing check), and
the app's ``--code`` route never runs under an offscreen platform. The project XML is a
stable, documented format, so it is written directly; QGIS tolerates a style it cannot
parse by falling back to its default renderer, so a styling slip degrades, never breaks.

Methods are discovered from what is on disk. A method is included with however many
tiles have all four products; a tile missing any of them is skipped WITH A WARNING, so
re-run the script once a copy completes and the layer picks the tile up.

    python benchmark/make_qgis_project.py --phase all
    python benchmark/make_qgis_project.py --phase project      # XML only, derived data exists

Instance ids are coloured on a continuous cyclic ramp over 0..max id. That only works
because ``ff3d_geo stitch`` hands out dense Weyl-ordered ids, so neighbouring trees have
unrelated id values and get unrelated colours; a paletted renderer with 800k classes
would not open.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

TILE_RE = re.compile(r"^3dm_33_(\d{3})_(\d{4})_1_be$")
EPSG = 25833
CRS_DESC = "ETRS89 / UTM zone 33N"
PROJ4 = "+proj=utm +zone=33 +ellps=GRS80 +towgs84=0,0,0,0,0,0,0 +units=m +no_defs"

# method key -> (display name, result dir relative to ALS_Data, stitch.json for max id)
METHODS = {
    "ff3d": ("ForestFormer3D", "berlin_als_2021_ff3d_v2", "berlin_als_2021_ff3d_v2/stitch.json"),
    "ff3d_masked": ("ForestFormer3D, buildings masked", "berlin_als_2021_ff3d_v2/masked",
                    "berlin_als_2021_ff3d_v2/stitch.json"),
    "sat": ("SegmentAnyTree", "berlin_als_2021_sat", "berlin_als_2021_sat/stitch.json"),
    "ams3d": ("AMS3D", "berlin_als_2021_ams3d", None),
}
PRODUCTS = ("_instance_50cm.tif", "_semantic_50cm.tif", "_crowns.gpkg", "_trees.gpkg")

# A cyclic palette for instance ids: turbo out and back, so the ramp never has a flat
# region and ids anywhere in the range spread over the whole hue circle.
TURBO = ["#30123b", "#4662d7", "#36aaf9", "#1ae4b6", "#72fe5e", "#c8ef34",
         "#faba39", "#f66b19", "#ca2a04", "#7a0403"]
CYCLIC = TURBO + TURBO[-2:0:-1]
VIRIDIS = ["#440154", "#414487", "#2a788e", "#22a884", "#7ad151", "#fde725"]
SEMANTIC = [(0, "#9e9e9e", "ground"), (1, "#8d5a2b", "wood"), (2, "#2e8b57", "leaf"),
            (3, "#d63a3a", "building (masked)")]


# ----------------------------------------------------------------------------- geometry
def km_bounds(tile: str) -> tuple[float, float, float, float]:
    """(minx, miny, maxx, maxy) of a Berlin km tile from its name."""
    m = TILE_RE.match(tile)
    if not m:
        raise ValueError(f"not a Berlin km-tile name: {tile}")
    e, n = int(m.group(1)) * 1000.0, int(m.group(2)) * 1000.0
    return e, n, e + 1000.0, n + 1000.0


def union_bounds(tiles) -> tuple[float, float, float, float]:
    bs = [km_bounds(t) for t in tiles]
    return (min(b[0] for b in bs), min(b[1] for b in bs),
            max(b[2] for b in bs), max(b[3] for b in bs))


def wgs84(bounds) -> tuple[float, float, float, float]:
    """The same box in lon/lat, which QGIS stores next to the projected extent."""
    from pyproj import Transformer

    tr = Transformer.from_crs(EPSG, 4326, always_xy=True)
    (x0, y0), (x1, y1) = tr.transform(bounds[0], bounds[1]), tr.transform(bounds[2], bounds[3])
    return min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)


# ------------------------------------------------------------------------------- disk
def tiles_of(root: Path, products=PRODUCTS, warn=print) -> list[str]:
    """Tiles under ``root`` that carry every product; the rest are reported and skipped."""
    out = []
    for d in sorted(p for p in root.iterdir() if p.is_dir() and TILE_RE.match(p.name)):
        missing = [s for s in products if not (d / f"{d.name}{s}").exists()]
        if missing:
            warn(f"  skip {root.name}/{d.name}: missing {', '.join(missing)}")
            continue
        out.append(d.name)
    return out


def run(cmd: list[str]) -> str:
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"{cmd[0]} failed ({r.returncode}): {r.stderr.strip()[:400]}")
    return r.stdout


def gpkg_first_layer(gdal_bin: Path, gpkg: Path) -> str:
    out = run([str(gdal_bin / "ogrinfo"), "-q", "-ro", str(gpkg)])
    m = re.search(r"^\d+:\s+(\S+)", out, re.M)
    if not m:
        raise RuntimeError(f"no layer found in {gpkg}")
    return m.group(1)


def max_tree_id(gdal_bin: Path, gpkg: Path, layer: str) -> int:
    out = run([str(gdal_bin / "ogrinfo"), "-q", "-ro", str(gpkg),
               "-sql", f"SELECT MAX(tree_id) AS m FROM {layer}"])
    m = re.search(r"m \(\w+\) = (\d+)", out)
    return int(m.group(1)) if m else 0


# --------------------------------------------------------------------------- derived
def build_vrt(gdal_bin: Path, out: Path, files: list[Path], nodata=None) -> None:
    cmd = [str(gdal_bin / "gdalbuildvrt"), "-overwrite", "-q"]
    if nodata is not None:
        cmd += ["-srcnodata", str(nodata), "-vrtnodata", str(nodata)]
    run(cmd + [str(out)] + [str(f) for f in files])


def merge_gpkg(gdal_bin: Path, out: Path, inputs: list[tuple[str, Path]], layer: str) -> None:
    """Append every tile's layer into one GeoPackage layer, tagging rows with ``tile``."""
    out.unlink(missing_ok=True)
    for tile, src in inputs:
        src_layer = gpkg_first_layer(gdal_bin, src)
        run([str(gdal_bin / "ogr2ogr"), "-q", "-f", "GPKG", "-append", "-nln", layer,
             "-nlt", "PROMOTE_TO_MULTI", str(out), str(src),
             "-sql", f"SELECT *, '{tile}' AS tile FROM \"{src_layer}\""])


def phase_derived(als: Path, out: Path, gdal_bin: Path) -> dict:
    """Mosaic everything; returns the manifest the project phase reads."""
    derived = out / "derived"
    derived.mkdir(parents=True, exist_ok=True)
    manifest: dict = {"methods": {}, "ortho": {}, "terrain": {}, "buildings": None}

    for key, (name, rel, stitch) in METHODS.items():
        root = als / rel
        if not root.is_dir():
            print(f"  {name}: no directory {root}, skipped")
            continue
        tiles = tiles_of(root)
        if not tiles:
            print(f"  {name}: no complete tile, skipped")
            continue
        print(f"  {name}: {len(tiles)} tiles")
        inst = derived / f"{key}_instance_50cm.vrt"
        sem = derived / f"{key}_semantic_50cm.vrt"
        build_vrt(gdal_bin, inst, [root / t / f"{t}_instance_50cm.tif" for t in tiles], nodata=-1)
        build_vrt(gdal_bin, sem, [root / t / f"{t}_semantic_50cm.tif" for t in tiles], nodata=255)
        crowns = derived / f"{key}_crowns.gpkg"
        trees = derived / f"{key}_trees.gpkg"
        merge_gpkg(gdal_bin, crowns, [(t, root / t / f"{t}_crowns.gpkg") for t in tiles], "crowns")
        merge_gpkg(gdal_bin, trees, [(t, root / t / f"{t}_trees.gpkg") for t in tiles], "trees")
        max_id = 0
        if stitch and (als / stitch).exists():
            max_id = int(json.load(open(als / stitch)).get("n_trees", 0))
        if not max_id:
            max_id = max_tree_id(gdal_bin, trees, "trees")
        manifest["methods"][key] = {
            "name": name, "tiles": tiles, "max_id": max_id,
            "instance": inst.name, "semantic": sem.name, "crowns": crowns.name, "trees": trees.name,
            "las": [str((root / t / f"{t}.las").relative_to(als)) for t in tiles
                    if (root / t / f"{t}.las").exists()],
        }

    for key, rel, label in (("dop2021_rgb", "berlin_dop_2021/dop_2021_rgb", "DOP 2021 leaf-off (RGB)"),
                            ("dop2021_rgbi", "berlin_dop_2021/dop_2021_rgbi", "DOP 2021 leaf-off (RGBI)"),
                            ("dop2025", "berlin_dop_2025_sommer", "DOP 2025 leaf-on (RGB)")):
        d = als / rel
        files = sorted(p for p in d.glob("3dm_33_*.tif") if not p.name.startswith("._")) if d.is_dir() else []
        if files:
            v = derived / f"{key}.vrt"
            build_vrt(gdal_bin, v, files)
            # Read the band count rather than trusting the directory name: the 2025 summer
            # set is delivered as RGBI although nothing in its name says so.
            bands = len(re.findall(r"^Band \d+ ", run([str(gdal_bin / "gdalinfo"), str(v)]), re.M))
            manifest["ortho"][key] = {"name": label.replace("(RGB)", f"({'RGBI' if bands == 4 else 'RGB'})"),
                                      "vrt": v.name, "tiles": [p.stem for p in files], "bands": bands}
            print(f"  {label}: {len(files)} tiles, {bands} bands")

    terrain = als / "berlin_terrain"
    if terrain.is_dir():
        for key, suffix, label, nodata in (("dtm", "_dtm_1m.tif", "DTM 1 m", -9999),
                                            ("dsm", "_dsm_50cm.tif", "DSM 0.5 m", None),
                                            ("chm", "_chm_50cm.tif", "CHM 0.5 m", None)):
            files = sorted(terrain / t.name / f"{t.name}{suffix}" for t in terrain.iterdir()
                           if t.is_dir() and TILE_RE.match(t.name) and (t / f"{t.name}{suffix}").exists())
            if files:
                v = derived / f"terrain_{key}.vrt"
                build_vrt(gdal_bin, v, files, nodata=nodata)
                manifest["terrain"][key] = {"name": label, "vrt": v.name, "tiles": [f.parent.name for f in files]}
                print(f"  terrain {label}: {len(files)} tiles")

    b = als / "berlin_buildings" / "alkis_buildings.gpkg"
    if b.exists():
        manifest["buildings"] = {"path": str(b.relative_to(als)),
                                 "layer": gpkg_first_layer(gdal_bin, b)}

    (out / "derived_manifest.json").write_text(json.dumps(manifest, indent=1))
    return manifest


# ----------------------------------------------------------------------------- XML
def esc(s: str) -> str:
    return (s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
             .replace('"', "&quot;"))


def layer_id(name: str) -> str:
    return f"{re.sub(r'[^A-Za-z0-9_]', '_', name)}_{hashlib.md5(name.encode()).hexdigest()[:12]}"


def srs_xml(wkt: str) -> str:
    return (f"<spatialrefsys nativeFormat=\"Wkt\"><wkt>{esc(wkt)}</wkt><proj4>{esc(PROJ4)}</proj4>"
            f"<srsid>2104</srsid><srid>{EPSG}</srid><authid>EPSG:{EPSG}</authid>"
            f"<description>{CRS_DESC}</description><projectionacronym>utm</projectionacronym>"
            f"<ellipsoidacronym>EPSG:7019</ellipsoidacronym><geographicflag>false</geographicflag>"
            f"</spatialrefsys>")


def extent_xml(bounds, ll) -> str:
    x0, y0, x1, y1 = bounds
    lx0, ly0, lx1, ly1 = ll
    return (f"<extent><xmin>{x0}</xmin><ymin>{y0}</ymin><xmax>{x1}</xmax><ymax>{y1}</ymax></extent>"
            f"<wgs84extent><xmin>{lx0}</xmin><ymin>{ly0}</ymin><xmax>{lx1}</xmax><ymax>{ly1}</ymax></wgs84extent>")


def minmax_xml(limits="None") -> str:
    return (f"<minMaxOrigin><limits>{limits}</limits><extent>WholeRaster</extent>"
            "<statAccuracy>Estimated</statAccuracy><cumulativeCutLower>0.02</cumulativeCutLower>"
            "<cumulativeCutUpper>0.98</cumulativeCutUpper><stdDevFactor>2</stdDevFactor></minMaxOrigin>")


def hex_rgb(h: str) -> str:
    h = h.lstrip("#")
    return ",".join(str(int(h[i:i + 2], 16)) for i in (0, 2, 4)) + ",255"


def renderer_pseudocolor(vmin: float, vmax: float, colors: list[str], label_fmt="{:.0f}") -> str:
    n = len(colors)
    items = "".join(
        f"<item value=\"{vmin + (vmax - vmin) * i / (n - 1)}\" color=\"{c}\" alpha=\"255\" "
        f"label=\"{label_fmt.format(vmin + (vmax - vmin) * i / (n - 1))}\"/>"
        for i, c in enumerate(colors))
    return (f"<rasterrenderer type=\"singlebandpseudocolor\" band=\"1\" opacity=\"1\" alphaBand=\"-1\" "
            f"nodataColor=\"\" classificationMin=\"{vmin}\" classificationMax=\"{vmax}\">"
            f"<rasterTransparency/>{minmax_xml()}<rastershader>"
            f"<colorrampshader colorRampType=\"INTERPOLATED\" classificationMode=\"1\" clip=\"0\" "
            f"minimumValue=\"{vmin}\" maximumValue=\"{vmax}\" labelPrecision=\"0\">{items}"
            f"</colorrampshader></rastershader></rasterrenderer>")


def renderer_paletted(entries) -> str:
    pal = "".join(f"<paletteEntry value=\"{v}\" color=\"{c}\" alpha=\"255\" label=\"{esc(l)}\"/>"
                  for v, c, l in entries)
    return (f"<rasterrenderer type=\"paletted\" band=\"1\" opacity=\"1\" alphaBand=\"-1\" nodataColor=\"\">"
            f"<rasterTransparency/>{minmax_xml()}<colorPalette>{pal}</colorPalette></rasterrenderer>")


def renderer_multiband(r=1, g=2, b=3) -> str:
    return (f"<rasterrenderer type=\"multibandcolor\" opacity=\"1\" alphaBand=\"-1\" nodataColor=\"\" "
            f"redBand=\"{r}\" greenBand=\"{g}\" blueBand=\"{b}\"><rasterTransparency/>"
            f"{minmax_xml('None')}</rasterrenderer>")


def raster_layer_xml(lid: str, name: str, source: str, renderer: str, bounds, ll, wkt: str) -> str:
    return (f"<maplayer type=\"raster\" autoRefreshTime=\"0\" autoRefreshMode=\"Disabled\" "
            f"hasScaleBasedVisibilityFlag=\"0\" maxScale=\"0\" minScale=\"1e+08\" styleCategories=\"AllStyleCategories\" "
            f"legendPlaceholderImage=\"\" refreshOnNotifyEnabled=\"0\" refreshOnNotifyMessage=\"\">"
            f"{extent_xml(bounds, ll)}<id>{lid}</id><datasource>{esc(source)}</datasource>"
            f"<layername>{esc(name)}</layername><srs>{srs_xml(wkt)}</srs>"
            f"<provider>gdal</provider><noData><noDataList bandNo=\"1\" useSrcNoData=\"1\"/></noData>"
            f"<map-layer-style-manager current=\"default\"><map-layer-style name=\"default\"/></map-layer-style-manager>"
            f"<pipe><provider><resampling enabled=\"false\" zoomedInResamplingMethod=\"nearestNeighbour\" "
            f"zoomedOutResamplingMethod=\"nearestNeighbour\" maxOversampling=\"2\"/></provider>"
            f"{renderer}<brightnesscontrast brightness=\"0\" contrast=\"0\" gamma=\"1\"/>"
            f"<huesaturation colorizeOn=\"0\" colorizeRed=\"255\" colorizeGreen=\"128\" colorizeBlue=\"128\" "
            f"colorizeStrength=\"100\" saturation=\"0\" grayscaleMode=\"0\" invertColors=\"0\"/>"
            f"<rasterresampler maxOversampling=\"2\"/><resamplingStage>resamplingFilter</resamplingStage></pipe>"
            f"<blendMode>0</blendMode></maplayer>")


def renderer_categorized(attr: str, categories: list[tuple], outline="60,60,60,120", width="0.15") -> str:
    """``<renderer-v2 type="categorizedSymbol">`` over ``attr``: one (value, fill colour) or
    (value, fill colour, outline colour) per category plus a final grey catch-all (an empty
    value matches everything else)."""
    cats, syms = [], []
    for i, cat in enumerate(list(categories) + [("", "158,158,158,90")]):
        value, colour = cat[0], cat[1]
        label = value or "other"
        cats.append(f"<category value=\"{esc(value)}\" symbol=\"{i}\" label=\"{esc(label)}\" render=\"true\" type=\"string\"/>")
        syms.append(symbol_fill(cat[2] if len(cat) > 2 else outline, width, colour, "solid", name=str(i)))
    return (f"<renderer-v2 type=\"categorizedSymbol\" attr=\"{esc(attr)}\" forceraster=\"0\" symbollevels=\"0\" "
            f"enableorderby=\"0\" referencescale=\"-1\"><categories>{''.join(cats)}</categories>"
            f"<symbols>{''.join(syms)}</symbols><rotation/><sizescale/></renderer-v2>")


def symbol_fill(outline: str, width="0.3", fill="0,0,0,0", style="no", name="0") -> str:
    return (f"<symbol type=\"fill\" name=\"{name}\" alpha=\"1\" clip_to_extent=\"1\" force_rhr=\"0\" is_animated=\"0\" frame_rate=\"10\">"
            f"<layer class=\"SimpleFill\" enabled=\"1\" locked=\"0\" pass=\"0\"><Option type=\"Map\">"
            f"<Option name=\"color\" type=\"QString\" value=\"{fill}\"/>"
            f"<Option name=\"outline_color\" type=\"QString\" value=\"{outline}\"/>"
            f"<Option name=\"outline_style\" type=\"QString\" value=\"solid\"/>"
            f"<Option name=\"outline_width\" type=\"QString\" value=\"{width}\"/>"
            f"<Option name=\"outline_width_unit\" type=\"QString\" value=\"MM\"/>"
            f"<Option name=\"style\" type=\"QString\" value=\"{style}\"/>"
            f"</Option></layer></symbol>")


def symbol_marker(color: str, size_expr: str | None = None) -> str:
    dd = ""
    if size_expr:
        dd = ("<data_defined_properties><Option type=\"Map\"><Option name=\"name\" type=\"QString\" value=\"\"/>"
              "<Option name=\"properties\" type=\"Map\"><Option name=\"size\" type=\"Map\">"
              "<Option name=\"active\" type=\"bool\" value=\"true\"/>"
              f"<Option name=\"expression\" type=\"QString\" value=\"{esc(size_expr)}\"/>"
              "<Option name=\"type\" type=\"int\" value=\"3\"/></Option></Option>"
              "<Option name=\"type\" type=\"QString\" value=\"collection\"/></Option></data_defined_properties>")
    return (f"<symbol type=\"marker\" name=\"0\" alpha=\"0.9\" clip_to_extent=\"1\" force_rhr=\"0\" is_animated=\"0\" frame_rate=\"10\">"
            f"<layer class=\"SimpleMarker\" enabled=\"1\" locked=\"0\" pass=\"0\"><Option type=\"Map\">"
            f"<Option name=\"name\" type=\"QString\" value=\"circle\"/>"
            f"<Option name=\"color\" type=\"QString\" value=\"{color}\"/>"
            f"<Option name=\"outline_color\" type=\"QString\" value=\"35,35,35,255\"/>"
            f"<Option name=\"outline_width\" type=\"QString\" value=\"0.2\"/>"
            f"<Option name=\"size\" type=\"QString\" value=\"2\"/>"
            f"<Option name=\"size_unit\" type=\"QString\" value=\"MM\"/>"
            f"</Option>{dd}</layer></symbol>")


def vector_layer_xml(lid: str, name: str, source: str, geometry: str, symbol: str, bounds, ll, wkt: str,
                     provider: str = "ogr", renderer: str | None = None) -> str:
    """``renderer`` (a whole ``<renderer-v2>``) replaces the default single-symbol one built
    from ``symbol``; pass it for categorized layers."""
    renderer_xml = renderer or (
        f"<renderer-v2 type=\"singleSymbol\" forceraster=\"0\" symbollevels=\"0\" enableorderby=\"0\" referencescale=\"-1\">"
        f"<symbols>{symbol}</symbols><rotation/><sizescale/></renderer-v2>")
    return (f"<maplayer type=\"vector\" geometry=\"{geometry}\" autoRefreshTime=\"0\" autoRefreshMode=\"Disabled\" "
            f"hasScaleBasedVisibilityFlag=\"0\" maxScale=\"0\" minScale=\"1e+08\" simplifyDrawingHints=\"1\" "
            f"simplifyDrawingTol=\"1\" simplifyMaxScale=\"1\" simplifyLocal=\"1\" simplifyAlgorithm=\"0\" "
            f"styleCategories=\"AllStyleCategories\" readOnly=\"0\" labelsEnabled=\"0\" symbologyReferenceScale=\"-1\" "
            f"refreshOnNotifyEnabled=\"0\" refreshOnNotifyMessage=\"\" legendPlaceholderImage=\"\">"
            f"{extent_xml(bounds, ll)}<id>{lid}</id><datasource>{esc(source)}</datasource>"
            f"<layername>{esc(name)}</layername><srs>{srs_xml(wkt)}</srs>"
            f"<provider encoding=\"UTF-8\">{provider}</provider>"
            f"<map-layer-style-manager current=\"default\"><map-layer-style name=\"default\"/></map-layer-style-manager>"
            f"{renderer_xml}"
            f"<blendMode>0</blendMode><featureBlendMode>0</featureBlendMode><layerOpacity>1</layerOpacity></maplayer>")


def pointcloud_layer_xml(lid: str, name: str, source: str, bounds, ll, wkt: str) -> str:
    return (f"<maplayer type=\"point-cloud\" autoRefreshTime=\"0\" autoRefreshMode=\"Disabled\" "
            f"hasScaleBasedVisibilityFlag=\"0\" maxScale=\"0\" minScale=\"1e+08\" styleCategories=\"AllStyleCategories\" "
            f"refreshOnNotifyEnabled=\"0\" refreshOnNotifyMessage=\"\" legendPlaceholderImage=\"\">"
            f"{extent_xml(bounds, ll)}<id>{lid}</id><datasource>{esc(source)}</datasource>"
            f"<layername>{esc(name)}</layername><srs>{srs_xml(wkt)}</srs><provider>pdal</provider>"
            f"<map-layer-style-manager current=\"default\"><map-layer-style name=\"default\"/></map-layer-style-manager>"
            f"<blendMode>0</blendMode></maplayer>")


class Tree:
    """Collects layers and the layer-tree groups they sit in, then renders the project."""

    def __init__(self, title: str, wkt: str):
        self.title, self.wkt = title, wkt
        self.layers: list[str] = []          # maplayer XML, in tree order
        self.tree: list[str] = []            # layer-tree XML, in tree order
        self.order: list[str] = []           # layer ids, in tree order

    def group(self, name: str, checked: bool, expanded: bool, body_fn) -> None:
        self.tree.append(f"<layer-tree-group name=\"{esc(name)}\" checked=\"{'Qt::Checked' if checked else 'Qt::Unchecked'}\" "
                         f"expanded=\"{int(expanded)}\"><customproperties><Option/></customproperties>")
        body_fn()
        self.tree.append("</layer-tree-group>")

    def layer(self, xml: str, lid: str, name: str, source: str, provider: str, checked: bool) -> None:
        self.layers.append(xml)
        self.order.append(lid)
        self.tree.append(f"<layer-tree-layer id=\"{lid}\" name=\"{esc(name)}\" source=\"{esc(source)}\" "
                         f"providerKey=\"{provider}\" checked=\"{'Qt::Checked' if checked else 'Qt::Unchecked'}\" "
                         f"expanded=\"0\" legend_exp=\"\" patch_size=\"-1,-1\" legend_split_behavior=\"0\">"
                         f"<customproperties><Option/></customproperties></layer-tree-layer>")

    def render(self, bounds, ll) -> str:
        order = "".join(f"<layer id=\"{i}\"/>" for i in self.order)
        return ("<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>\n"
                f"<qgis version=\"3.44.10\" projectname=\"{esc(self.title)}\" saveUser=\"\" saveUserFull=\"\">"
                f"<homePath path=\"\"/><title>{esc(self.title)}</title>"
                f"<projectCrs>{srs_xml(self.wkt)}</projectCrs>"
                f"<layer-tree-group><customproperties><Option/></customproperties>"
                f"{''.join(self.tree)}<custom-order enabled=\"0\"/></layer-tree-group>"
                f"<mapcanvas name=\"theMapCanvas\" annotationsVisible=\"1\"><units>meters</units>"
                f"<extent><xmin>{bounds[0]}</xmin><ymin>{bounds[1]}</ymin><xmax>{bounds[2]}</xmax><ymax>{bounds[3]}</ymax></extent>"
                f"<rotation>0</rotation><destinationsrs>{srs_xml(self.wkt)}</destinationsrs>"
                f"<rendermaptile>0</rendermaptile></mapcanvas>"
                f"<projectlayers>{''.join(self.layers)}</projectlayers>"
                f"<layerorder>{order}</layerorder>"
                f"<properties><Paths><Absolute type=\"bool\">false</Absolute></Paths>"
                f"<Gui><CanvasColorBluePart type=\"int\">255</CanvasColorBluePart><CanvasColorGreenPart type=\"int\">255</CanvasColorGreenPart>"
                f"<CanvasColorRedPart type=\"int\">255</CanvasColorRedPart></Gui>"
                f"<Measurement><Ellipsoid type=\"QString\">EPSG:7019</Ellipsoid></Measurement>"
                f"<PositionPrecision><Automatic type=\"bool\">true</Automatic><DecimalPlaces type=\"int\">2</DecimalPlaces></PositionPrecision>"
                f"<SpatialRefSys><ProjectionsEnabled type=\"int\">1</ProjectionsEnabled></SpatialRefSys>"
                f"</properties></qgis>\n")


FILE_PROVIDERS = {"ogr", "gdal", "pdal", "delimitedtext", "mdal", "copc", "ept"}


def read_qgs(project: Path) -> ET.Element:
    """The ``<qgis>`` root of a ``.qgs`` file or of the ``.qgs`` inside a ``.qgz``."""
    if project.suffix.lower() == ".qgz":
        with zipfile.ZipFile(project) as z:
            qgs = next(n for n in z.namelist() if n.lower().endswith(".qgs"))
            return ET.fromstring(z.read(qgs))
    return ET.fromstring(project.read_bytes())


def absolute_source(source: str, provider: str, base: Path) -> tuple[str, Path | None]:
    """Resolve a datasource string against the directory of the project it came from.

    File providers store ``<path>|layername=...`` and a path may be relative to that project
    (QGIS writes relative paths when ``Paths/Absolute`` is false); the embedded copy lives
    elsewhere, so it must carry the absolute path. Returns the rewritten source and the file
    it names (None for a URL-style source such as a WMS/XYZ layer)."""
    if provider not in FILE_PROVIDERS or "://" in source.split("|", 1)[0]:
        return source, None
    path, sep, rest = source.partition("|")
    p = Path(path)
    if not p.is_absolute():
        p = (base / p).resolve()
    return f"{p}{sep}{rest}", p


def import_project(T: "Tree", project: Path, group: str, warn=print) -> dict:
    """Embed every layer of another QGIS project verbatim under one top-level group.

    The ``<maplayer>`` elements are copied as they are (styling, labels, joins and the
    layer's own CRS travel with them), the other project's layer tree is copied beneath a
    new group so its grouping survives, and the layers are appended to the draw order in
    the other project's ``<layerorder>``. Relative datasources are resolved against the
    other project's directory; a layer whose file is gone is still embedded (QGIS marks it
    unavailable on load) but counted. An id already present in ``T`` gets a ``_imp`` suffix
    everywhere it occurs. Returns counts: layers, groups, missing, renamed."""
    root = read_qgs(project)
    base = project.resolve().parent
    taken = set(T.order)
    renamed, missing, sources = {}, [], {}
    for ml in root.iter("maplayer"):
        lid = ml.findtext("id") or ""
        if lid in taken:
            renamed[lid] = lid + "_imp"
        taken.add(renamed.get(lid, lid))
        ds = ml.find("datasource")
        if ds is not None and ds.text:
            ds.text, file = absolute_source(ds.text, ml.findtext("provider") or "", base)
            sources[lid] = ds.text
            if file is not None and not file.exists():
                missing.append(ds.text)
    for ml in root.iter("maplayer"):
        idel = ml.find("id")
        if idel is not None and idel.text in renamed:
            idel.text = renamed[idel.text]
    tree = root.find("layer-tree-group")
    for tl in tree.iter("layer-tree-layer"):
        lid = tl.get("id", "")
        if lid in sources:
            tl.set("source", sources[lid])
        if lid in renamed:
            tl.set("id", renamed[lid])
    for ml in root.iter("maplayer"):
        T.layers.append(ET.tostring(ml, encoding="unicode"))
    ordered = [renamed.get(l.get("id"), l.get("id")) for l in root.find("layerorder")] if root.find("layerorder") is not None else []
    ids = [ml.findtext("id") for ml in root.iter("maplayer")]
    T.order.extend([i for i in ordered if i in ids] + [i for i in ids if i not in ordered])
    n_groups = sum(1 for _ in tree.iter("layer-tree-group")) - 1   # iter() yields the root too

    def body():
        for child in tree:
            if child.tag in ("layer-tree-group", "layer-tree-layer"):
                T.tree.append(ET.tostring(child, encoding="unicode"))

    T.group(group, False, False, body)
    for m in missing:
        warn(f"  import {project.name}: file missing for {m}")
    return {"layers": len(ids), "groups": n_groups, "missing": len(missing), "renamed": len(renamed)}


def write_qgz(path: Path, xml: str) -> None:
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr(path.with_suffix(".qgs").name, xml)


# ---------------------------------------------------------------------------- project
def phase_project(als: Path, out: Path, manifest: dict, live_wfs: bool = False,
                  imports: list[Path] = ()) -> list[Path]:
    from pyproj import CRS

    wkt = CRS.from_epsg(EPSG).to_wkt()
    rel = lambda p: "./" + str(Path(p).relative_to(out)) if str(p).startswith(str(out)) else str(p)  # noqa: E731
    derived = out / "derived"
    all_tiles = sorted({t for m in manifest["methods"].values() for t in m["tiles"]}
                       | {t for o in manifest["ortho"].values() for t in o["tiles"]})
    bounds = union_bounds(all_tiles)
    ll = wgs84(bounds)
    T = Tree("Berlin ALS 2021 - tree segmentation (ForestFormer3D, SegmentAnyTree, AMS3D)", wkt)

    def basemap():
        lid = layer_id("osm_basemap")
        src = "type=xyz&url=https://tile.openstreetmap.org/%7Bz%7D/%7Bx%7D/%7By%7D.png&zmax=19&zmin=0"
        xml = (f"<maplayer type=\"raster\" autoRefreshTime=\"0\" autoRefreshMode=\"Disabled\" hasScaleBasedVisibilityFlag=\"0\" "
               f"maxScale=\"0\" minScale=\"1e+08\" styleCategories=\"AllStyleCategories\" legendPlaceholderImage=\"\" "
               f"refreshOnNotifyEnabled=\"0\" refreshOnNotifyMessage=\"\"><id>{lid}</id><datasource>{esc(src)}</datasource>"
               f"<layername>OpenStreetMap</layername><srs>{srs_xml(CRS.from_epsg(3857).to_wkt()).replace(f'EPSG:{EPSG}', 'EPSG:3857').replace(str(EPSG), '3857').replace(CRS_DESC, 'WGS 84 / Pseudo-Mercator')}</srs>"
               f"<provider>wms</provider><map-layer-style-manager current=\"default\"><map-layer-style name=\"default\"/></map-layer-style-manager>"
               f"<pipe><provider><resampling enabled=\"false\" zoomedInResamplingMethod=\"nearestNeighbour\" zoomedOutResamplingMethod=\"nearestNeighbour\" maxOversampling=\"2\"/></provider>"
               f"{renderer_multiband()}<brightnesscontrast brightness=\"0\" contrast=\"0\" gamma=\"1\"/>"
               f"<huesaturation colorizeOn=\"0\" colorizeRed=\"255\" colorizeGreen=\"128\" colorizeBlue=\"128\" colorizeStrength=\"100\" saturation=\"0\" grayscaleMode=\"0\" invertColors=\"0\"/>"
               f"<rasterresampler maxOversampling=\"2\"/><resamplingStage>resamplingFilter</resamplingStage></pipe><blendMode>0</blendMode></maplayer>")
        T.layer(xml, lid, "OpenStreetMap", src, "wms", False)

    def method_group(key: str, m: dict, checked: bool):
        name = m["name"]
        b = union_bounds(m["tiles"]); l = wgs84(b)
        tag = f"{name} ({len(m['tiles'])} tiles)"

        def body():
            crowns_src = f"{rel(derived / m['crowns'])}|layername=crowns"
            lid = layer_id(f"{key}_crowns")
            T.layer(vector_layer_xml(lid, "crowns", crowns_src, "Polygon",
                                     symbol_fill("255,212,121,255"), b, l, wkt),
                    lid, "crowns", crowns_src, "ogr", checked)
            trees_src = f"{rel(derived / m['trees'])}|layername=trees"
            lid = layer_id(f"{key}_trees")
            T.layer(vector_layer_xml(lid, "tree tops (size = height)", trees_src, "Point",
                                     symbol_marker("255,255,255,255", '"height" / 2'), b, l, wkt),
                    lid, "tree tops (size = height)", trees_src, "ogr", False)
            lid = layer_id(f"{key}_instance")
            T.layer(raster_layer_xml(lid, "instance ids (0.5 m)", rel(derived / m["instance"]),
                                     renderer_pseudocolor(0, max(m["max_id"], 1), CYCLIC), b, l, wkt),
                    lid, "instance ids (0.5 m)", rel(derived / m["instance"]), "gdal", checked)
            lid = layer_id(f"{key}_semantic")
            T.layer(raster_layer_xml(lid, "semantic classes (0.5 m)", rel(derived / m["semantic"]),
                                     renderer_paletted(SEMANTIC), b, l, wkt),
                    lid, "semantic classes (0.5 m)", rel(derived / m["semantic"]), "gdal", False)

        T.group(tag, checked, checked, body)

    def ortho_group():
        def body():
            for key, o in manifest["ortho"].items():
                b = union_bounds(o["tiles"]); l = wgs84(b)
                lid = layer_id(key)
                # Only the set fetched AS the NIR product is drawn as false colour; a 4-band
                # delivery that merely happens to carry NIR (the 2025 summer DOP) stays RGB.
                nir = key.endswith("rgbi") and o["bands"] == 4
                r = renderer_multiband(4, 1, 2) if nir else renderer_multiband()
                nm = o["name"] + (" false colour NIR,R,G" if nir else "")
                T.layer(raster_layer_xml(lid, nm, rel(derived / o["vrt"]), r, b, l, wkt),
                        lid, nm, rel(derived / o["vrt"]), "gdal", key == "dop2025")
        T.group("Orthophotos", True, False, body)

    def terrain_group():
        def body():
            for key, t in manifest["terrain"].items():
                b = union_bounds(t["tiles"]); l = wgs84(b)
                lid = layer_id(f"terrain_{key}")
                vmin, vmax = (0, 40) if key == "chm" else (30, 70)
                T.layer(raster_layer_xml(lid, f"{t['name']} ({len(t['tiles'])} tiles)", rel(derived / t["vrt"]),
                                         renderer_pseudocolor(vmin, vmax, VIRIDIS, "{:.0f} m"), b, l, wkt),
                        lid, f"{t['name']} ({len(t['tiles'])} tiles)", rel(derived / t["vrt"]), "gdal", False)
        if manifest["terrain"]:
            T.group("Terrain (from the ALS, where exported)", False, False, body)

    def buildings_layer():
        bl = manifest.get("buildings")
        if not bl:
            return
        src = f"{rel(als / bl['path'])}|layername={bl['layer']}"
        lid = layer_id("alkis_buildings")
        T.layer(vector_layer_xml(lid, "ALKIS building footprints", src, "Polygon",
                                 symbol_fill("214,58,58,255", "0.4", "214,58,58,60", "solid"), bounds, ll, wkt),
                lid, "ALKIS building footprints", src, "ogr", True)

    def tree_cadastre_group(live_wfs: bool):
        """Berlin's tree cadastre: street trees and (part of) the park trees with species,
        planting year and height -- a species-labelled reference outside the forest
        (9,103 + 12,349 trees inside this mosaic, checked 2026-10-05).

        Read from the GeoPackage benchmark/fetch_berlin_trees.py writes, through the OGR
        provider like every other layer. QGIS 3.44 reported the same feature types as
        LIVE WFS layers "unavailable" on project load although the service answers every
        request QGIS makes; its namespace is the bare word ``baumbestand`` rather than a
        URL, a known trigger for the WFS provider's typename resolution, and that could
        not be verified headless on this Mac. ``--live-wfs`` keeps the live form for a
        QGIS build where it works."""
        rows = (("cadastre_street", "strassenbaeume", "baumbestand:strassenbaeume",
                 "street trees (Strassenbaeume)", "255,170,0,230"),
                ("cadastre_park", "anlagenbaeume", "baumbestand:anlagenbaeume",
                 "park trees (Anlagenbaeume)", "60,200,90,230"))
        gpkg = als / "berlin_trees" / "baumbestand_berlin.gpkg"

        def body():
            for key, layer, typename, label, colour in rows:
                nm = f"{label}, size = height"
                sym = symbol_marker(colour, 'coalesce("baumhoehe", 10) / 5')
                lid = layer_id(key)
                if live_wfs:
                    src = (f"restrictToRequestBBOX='1' srsname='EPSG:{EPSG}' typename='{typename}' "
                           f"url='https://gdi.berlin.de/services/wfs/baumbestand' version='auto'")
                    T.layer(vector_layer_xml(lid, nm, src, "Point", sym, bounds, ll, wkt, provider="WFS"),
                            lid, nm, src, "WFS", False)
                else:
                    src = f"{rel(gpkg)}|layername={layer}"
                    T.layer(vector_layer_xml(lid, nm, src, "Point", sym, bounds, ll, wkt),
                            lid, nm, src, "ogr", True)

        if live_wfs:
            T.group("Berlin tree cadastre (live WFS, dl-de/zero-2-0)", False, False, body)
        elif gpkg.exists():
            T.group("Berlin tree cadastre (Baumbestand, dl-de/zero-2-0)", True, False, body)
        else:
            print(f"  tree cadastre: {gpkg} missing (benchmark/fetch_berlin_trees.py), group skipped")

    def forest_stands_group():
        """Berlin's forest stand map (Forstbetriebskarte 2014, Umweltatlas): the only source
        for what grows INSIDE the forest. Each stand carries up to five species per canopy
        layer with mixing share; the layer is coloured by the dominant species of the main
        layer (s1_1_deuts), categories ordered by area-weighted share so the legend reads
        like the composition. Fetched by benchmark/fetch_berlin_forest_stands.py."""
        gpkg = als / "berlin_forest" / "forstbetriebskarte_2014.gpkg"
        if not gpkg.exists():
            print(f"  forest stands: {gpkg} missing (benchmark/fetch_berlin_forest_stands.py), group skipped")
            return
        import geopandas as gpd

        g = gpd.read_file(gpkg, layer="hauptbaumarten", columns=["s1_1_deuts", "gis_area"])
        share = g.groupby("s1_1_deuts")["gis_area"].sum().sort_values(ascending=False)
        palette = ["217,164,65,160", "122,74,29,160", "46,139,87,160", "200,239,52,160", "235,235,225,170",
                   "160,82,45,160", "181,101,29,160", "31,95,63,160", "120,160,200,160", "200,120,160,160"]
        cats = [(str(sp), palette[i]) for i, sp in enumerate(share.index[:len(palette)])]
        src = f"{rel(gpkg)}|layername=hauptbaumarten"
        lid = layer_id("forest_stands_2014")
        nm = "stands by dominant species of the main canopy layer (2014 inventory)"

        reviere = als / "berlin_forest" / "reviere.gpkg"   # benchmark/derive_berlin_reviere.py

        def body():
            if reviere.exists():
                # km tiles the Reviere reach into, by whether the mosaic processed them: the
                # answer to "are tiles missing" at a glance
                csrc = f"{rel(reviere)}|layername=coverage"
                clid = layer_id("reviere_coverage")
                cnm = "km tiles with Revier forest: processed / downloaded only / not downloaded"
                T.layer(vector_layer_xml(clid, cnm, csrc, "Polygon", "", bounds, ll, wkt,
                                         renderer=renderer_categorized("status", [
                                             ("processed", "46,139,87,50"), ("downloaded", "255,165,0,110"),
                                             ("missing", "220,20,60,110")], outline="90,90,90,160")),
                        clid, cnm, csrc, "ogr", True)
                rsrc = f"{rel(reviere)}|layername=reviere"
                rlid = layer_id("reviere")
                rnm = "forest districts (Reviere, dissolved from the stand ids)"
                T.layer(vector_layer_xml(rlid, rnm, rsrc, "Polygon", "", bounds, ll, wkt,
                                         renderer=renderer_categorized("name", [
                                             ("Revier 12 Tegelsee", "0,0,0,0", "0,70,180,255"),
                                             ("Revier 13 Spandau", "0,0,0,0", "170,0,120,255"),
                                             ("Revier 11", "0,0,0,0", "20,20,20,255"),
                                             ("Revier 15", "0,0,0,0", "20,20,20,255")], width="0.9")),
                        rlid, rnm, rsrc, "ogr", True)
            else:
                print(f"  reviere: {reviere} missing (benchmark/derive_berlin_reviere.py), layers skipped")
            T.layer(vector_layer_xml(lid, nm, src, "Polygon", "", bounds, ll, wkt,
                                     renderer=renderer_categorized("s1_1_deuts", cats)),
                    lid, nm, src, "ogr", True)

        T.group("Berlin forest stand map (Forstbetriebskarte 2014, dl-de/zero-2-0)", True, True, body)

    # tree order = draw order top-down: vectors over rasters over orthophotos over basemap;
    # other projects' layers come first (unchecked), so they draw over everything when turned on
    for proj in imports:
        n = import_project(T, proj, f"{proj.stem} (imported project, {proj})")
        print(f"  imported {proj}: {n['layers']} layers in {n['groups']} groups, "
              f"{n['missing']} files missing, {n['renamed']} ids renamed")
    tree_cadastre_group(live_wfs)
    forest_stands_group()
    buildings_layer()
    for key in ("ff3d", "sat", "ams3d", "ff3d_masked"):
        if key in manifest["methods"]:
            method_group(key, manifest["methods"][key], checked=(key == "ff3d"))
    terrain_group()
    ortho_group()
    basemap()

    main = out / "berlin_als_2021.qgz"
    write_qgz(main, T.render(bounds, ll))
    written = [main]

    # point clouds in their own project: QGIS builds a COPC index per LAS on first load
    P = Tree("Berlin ALS 2021 - result point clouds (LAS, treeID/semantic/score)", wkt)
    for key in ("ff3d", "sat", "ff3d_masked"):
        m = manifest["methods"].get(key)
        if not m or not m["las"]:
            continue

        def body(m=m, key=key):
            for las in m["las"]:
                tile = Path(las).stem
                b = km_bounds(tile); l = wgs84(b)
                lid = layer_id(f"{key}_las_{tile}")
                P.layer(pointcloud_layer_xml(lid, tile, rel(als / las), b, l, wkt), lid, tile, rel(als / las), "pdal", False)
        P.group(f"{m['name']} point clouds ({len(m['las'])} LAS, load on demand)", False, False, body)
    pc = out / "berlin_als_2021_pointclouds.qgz"
    write_qgz(pc, P.render(bounds, ll))
    written.append(pc)
    return written


# ------------------------------------------------------------------------------- main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--als-data", type=Path, default=Path("/Volumes/2TB/winmol/ALS_Data"))
    ap.add_argument("--out", type=Path, default=None, help="default <als-data>/berlin_qgis")
    ap.add_argument("--gdal-bin", type=Path, default=Path("/opt/local/bin"))
    ap.add_argument("--phase", choices=["derived", "project", "all"], default="all")
    ap.add_argument("--live-wfs", action="store_true",
                    help="reference the Berlin tree cadastre as live WFS layers instead of the "
                         "GeoPackage from benchmark/fetch_berlin_trees.py (QGIS 3.44 on this Mac "
                         "reports the WFS form unavailable on load)")
    ap.add_argument("--import-project", type=Path, action="append", default=[], metavar="QGZ",
                    help="embed every layer of this QGIS project (.qgz/.qgs) verbatim as one unchecked "
                         "top-level group; repeatable. Relative datasources are resolved against "
                         "that project's directory")
    a = ap.parse_args(argv)
    for proj in a.import_project:
        if not proj.exists():
            print(f"!!! --import-project {proj} does not exist", file=sys.stderr)
            return 2
    out = a.out or a.als_data / "berlin_qgis"
    if not a.als_data.is_dir():
        print(f"!!! {a.als_data} is not a directory (is the 2TB mounted?)", file=sys.stderr)
        return 2
    for tool in ("gdalbuildvrt", "ogr2ogr", "ogrinfo"):
        if not (a.gdal_bin / tool).exists() and not shutil.which(tool):
            print(f"!!! {tool} not found under {a.gdal_bin}", file=sys.stderr)
            return 2
    out.mkdir(parents=True, exist_ok=True)
    if a.phase in ("derived", "all"):
        print("== derived data")
        manifest = phase_derived(a.als_data, out, a.gdal_bin)
    else:
        mp = out / "derived_manifest.json"
        if not mp.exists():
            print(f"!!! {mp} missing: run --phase derived first", file=sys.stderr)
            return 2
        manifest = json.loads(mp.read_text())
    if a.phase in ("project", "all"):
        print("== project")
        for p in phase_project(a.als_data, out, manifest, live_wfs=a.live_wfs, imports=a.import_project):
            print(f"  wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
