"""Overlay builder and LAS patcher behind the Potree site (benchmark/build_potree_site.py,
benchmark/potree_convert_tile.py)."""

import json
import struct

import numpy as np
import pytest

laspy = pytest.importorskip("laspy")
pytest.importorskip("pyproj")
pytest.importorskip("PIL")
pytest.importorskip("matplotlib")

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmark"))

from build_potree_site import (  # noqa: E402
    _round_coords,
    build_chm,
    tile_bounds_from_name,
    write_chm_png,
)
from potree_convert_tile import patched_copy  # noqa: E402

from geo_fixtures import TILE_ORIGIN, TWO_CONES, write_two_cone_las  # noqa: E402


class _Header:
    def __init__(self, mins):
        self.mins = mins


def test_tile_bounds_snap_to_the_km_grid():
    assert tile_bounds_from_name(_Header([381300.0, 5828300.0, 12.0])) == (
        381000.0, 5828000.0, 382000.0, 5829000.0)
    # a header that already sits on the grid keeps its corner
    assert tile_bounds_from_name(_Header([379000.0, 5828000.0, -3.0])) == (
        379000.0, 5828000.0, 380000.0, 5829000.0)


def test_round_coords_walks_nested_rings():
    rings = [[[381000.123, 5828000.456], [381001.987, 5828002.001]]]
    assert _round_coords(rings, 1) == [[[381000.1, 5828000.5], [381002.0, 5828002.0]]]


def test_chm_reproduces_the_synthetic_cone_heights(tmp_path):
    las = tmp_path / "tile.las"
    write_two_cone_las(las, with_predictions=True)
    bounds = tile_bounds_from_name(laspy.open(str(las)).header)
    chm, zmin, zmax, ground_z = build_chm(las, bounds, cell=0.5, dtm_cell=2.0)

    assert chm.shape == (2000, 2000)
    assert ground_z == pytest.approx(0.0, abs=0.01)
    assert zmin == pytest.approx(0.0, abs=0.01)
    assert chm.min() == 0.0

    # row 0 of chm is the SOUTHERN edge (grid rows count up with y)
    for ax, ay, h, _r in TWO_CONES:
        col = int((TILE_ORIGIN[0] + ax - bounds[0]) / 0.5)
        row = int((TILE_ORIGIN[1] + ay - bounds[1]) / 0.5)
        assert chm[row, col] == pytest.approx(h, abs=0.3)
    assert chm.max() == pytest.approx(max(c[2] for c in TWO_CONES), abs=0.3)


def test_chm_png_is_north_up_and_carries_its_extent(tmp_path):
    las = tmp_path / "tile.las"
    write_two_cone_las(las, with_predictions=True)
    bounds = tile_bounds_from_name(laspy.open(str(las)).header)
    chm, _, _, ground_z = build_chm(las, bounds, cell=0.5, dtm_cell=2.0)

    png, meta = tmp_path / "t_chm.png", tmp_path / "t_chm.json"
    write_chm_png(chm, png, meta, bounds, 20.0, ground_z)

    from PIL import Image

    img = np.array(Image.open(png))
    assert img.shape == (2000, 2000, 4)
    # the taller cone is further north, so it must sit in a SMALLER png row
    rows = []
    for ax, ay, _h, _r in TWO_CONES:
        col = int((TILE_ORIGIN[0] + ax - bounds[0]) / 0.5)
        row_grid = int((TILE_ORIGIN[1] + ay - bounds[1]) / 0.5)
        rows.append((2000 - 1 - row_grid, col))
    assert rows[1][0] < rows[0][0]
    for r, c in rows:
        assert img[r, c, 3] > 0            # canopy is opaque
    assert img[0, 0, 3] == 0               # empty corner is transparent

    doc = json.loads(meta.read_text())
    assert doc["crs"] == "EPSG:25833"
    assert doc["bounds"] == [float(v) for v in bounds]
    assert doc["vmax"] == 20.0


def _extra_bytes_description_offsets(path):
    """(offset, length) of every extra-bytes description field in a LAS file."""
    with open(path, "rb") as f:
        head = f.read(1 << 16)
    hsize = struct.unpack_from("<H", head, 94)[0]
    nvlr = struct.unpack_from("<I", head, 100)[0]
    out, o = [], hsize
    for _ in range(nvlr):
        uid = head[o + 2:o + 18].split(b"\x00")[0]
        rid = struct.unpack_from("<H", head, o + 18)[0]
        length = struct.unpack_from("<H", head, o + 20)[0]
        if uid == b"LASF_Spec" and rid == 4:
            out += [o + 54 + k * 192 + 160 for k in range(length // 192)]
        o += 54 + length
    return out


def test_patched_copy_terminates_a_full_length_description(tmp_path):
    src = tmp_path / "src.las"
    write_two_cone_las(src, with_predictions=True)
    offsets = _extra_bytes_description_offsets(src)
    assert len(offsets) == 3

    # what ForestFormer3D writes for treeID: exactly 32 bytes, no terminator
    full = b"ForestFormer3D instance, -1 none"
    assert len(full) == 32
    with open(src, "r+b") as f:
        f.seek(offsets[0])
        f.write(full)

    dst = tmp_path / "dst.las"
    fixed = patched_copy(src, dst)
    assert fixed == ["treeID"]

    raw_src, raw_dst = src.read_bytes(), dst.read_bytes()
    assert len(raw_src) == len(raw_dst)
    desc = raw_dst[offsets[0]:offsets[0] + 32]
    assert b"\x00" in desc
    assert desc.split(b"\x00")[0] == full[:31]
    # nothing but that one field changed
    assert raw_src[:offsets[0]] == raw_dst[:offsets[0]]
    assert raw_src[offsets[0] + 32:] == raw_dst[offsets[0] + 32:]

    # and the points still read back
    a, b = laspy.read(str(src)), laspy.read(str(dst))
    assert a.header.point_count == b.header.point_count
    assert np.array_equal(np.asarray(a.treeID), np.asarray(b.treeID))


def test_patched_copy_leaves_terminated_descriptions_alone(tmp_path):
    src = tmp_path / "src.las"
    write_two_cone_las(src, with_predictions=True)
    dst = tmp_path / "dst.las"
    assert patched_copy(src, dst) == []
    assert src.read_bytes() == dst.read_bytes()


def test_attach_variant_keeps_the_base_record_and_refuses_unknown_tiles():
    from build_potree_site import attach_variant

    recs = [{"tile": "a", "pointcloud": "pointclouds/a/metadata.json", "trees": {"count": 3},
             "layers": {"chm": {"png": "data/a_chm.png"}}}]
    sub = {"pointcloud": "pointclouds_sat/a/metadata.json", "trees": {"count": 5},
           "layers": {"instance": {"png": "data/a_sat_instance.png"}}}
    rec = attach_variant(recs, "a", "sat", sub)
    assert rec is recs[0]
    assert rec["pointcloud"] == "pointclouds/a/metadata.json" and rec["trees"]["count"] == 3
    assert rec["variants"]["sat"] is sub
    assert rec["layers"] == {"chm": {"png": "data/a_chm.png"}}   # shared layers untouched
    attach_variant(recs, "a", "sat", {"trees": {"count": 6}})     # a rebuild replaces it
    assert rec["variants"]["sat"]["trees"]["count"] == 6
    with pytest.raises(KeyError):
        attach_variant(recs, "b", "sat", sub)
