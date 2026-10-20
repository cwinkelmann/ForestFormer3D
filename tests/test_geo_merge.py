import numpy as np
import pytest

laspy = pytest.importorskip("laspy")
pytest.importorskip("shapely")
pytest.importorskip("pyproj")
gpd = pytest.importorskip("geopandas")
pytest.importorskip("pyogrio")

from ff3d_geo.merge import merge_las, merge_trees
from ff3d_geo.trees import trees_to_gpkg

from geo_fixtures import write_result_las


def _result_las(path, origin, tree_ids, semantic, n=200):
    rng = np.random.default_rng(int(origin[0]) % 997)
    return write_result_las(
        path,
        origin[0] + rng.uniform(0, 100, n),
        origin[1] + rng.uniform(0, 100, n),
        rng.uniform(0, 30, n),
        np.resize(np.asarray(tree_ids, np.int32), n),
        np.resize(np.asarray(semantic, np.uint8), n),
        score=np.full(n, 0.5, np.float32),
        offsets=np.array([origin[0], origin[1], 0.0]),
    )


def test_merge_las_renumbers_tree_ids_across_subtiles(tmp_path):
    a = _result_las(tmp_path / "a.las", (381000, 5829000), [-1, 0, 1, 2], [0, 1, 2, 2])
    b = _result_las(tmp_path / "b.las", (381100, 5829000), [0, 0, 1, -1], [2, 2, 1, 0])
    out = tmp_path / "merged.las"
    info = merge_las([a, b], out)
    m = laspy.read(out)
    assert len(m.points) == 400 and info["n_points"] == 400
    assert info["id_offsets"] == {str(a): 0, str(b): 3}
    ids = np.asarray(m.treeID)
    assert set(ids[ids >= 0].tolist()) == {0, 1, 2, 3, 4} and info["n_trees"] == 5
    assert (ids == -1).sum() == 100
    assert m.header.parse_crs().to_epsg() == 25833
    assert {"treeID", "semantic", "score"} <= set(m.point_format.extra_dimension_names)
    assert m.x.min() >= 381000 and m.x.max() <= 381200


def test_merge_trees_matches_merged_las_ids(tmp_path):
    a = _result_las(tmp_path / "a.las", (381000, 5829000), [-1, 0, 1, 2], [0, 1, 2, 2])
    b = _result_las(tmp_path / "b.las", (381100, 5829000), [0, 0, 1, -1], [2, 2, 1, 0])
    ga, gb = tmp_path / "a.gpkg", tmp_path / "b.gpkg"
    trees_to_gpkg(a, ga)
    trees_to_gpkg(b, gb)
    info = merge_las([a, b], tmp_path / "merged.las")
    n = merge_trees([ga, gb], tmp_path / "merged.gpkg", info["id_offsets"])
    t = gpd.read_file(tmp_path / "merged.gpkg", layer="trees")
    assert n == 5 and sorted(t["tree_id"].tolist()) == [0, 1, 2, 3, 4]
    assert t.crs.to_epsg() == 25833


def test_merge_las_keeps_the_extra_dimension_descriptions(tmp_path):
    """The merged LAS must carry results_to_las's extra-dim descriptions through.

    Regression: merge_las used to rebuild treeID/semantic/score from their names
    alone, which drops the descriptions; laspy then rejected every write with
    "Incompatible point formats" (it compares point formats dimension by
    dimension, and DimensionInfo carries the description).
    """
    a = _result_las(tmp_path / "a.las", (381000, 5829000), [-1, 0, 1, 2], [0, 1, 2, 2])
    b = _result_las(tmp_path / "b.las", (381100, 5829000), [0, 0, 1, -1], [2, 2, 1, 0])
    out = tmp_path / "merged.las"
    merge_las([a, b], out)
    merged = laspy.read(out)
    assert merged.header.point_format == laspy.read(a).header.point_format
    assert {d.name: d.description for d in merged.point_format.extra_dimensions} == {
        "treeID": "ForestFormer3D instance, -1 none",
        "semantic": "0 ground 1 wood 2 leaf 255 n/a",
        "score": "instance score",
    }


def test_merge_las_names_a_sub_tile_with_a_different_point_format(tmp_path):
    a = _result_las(tmp_path / "a.las", (381000, 5829000), [-1, 0, 1], [0, 1, 2])
    b = tmp_path / "b.las"
    header = laspy.LasHeader(point_format=6, version="1.4")
    header.scales = np.array([0.001] * 3)
    header.offsets = np.array([381100.0, 5829000.0, 0.0])
    header.add_extra_dim(laspy.ExtraBytesParams(name="treeID", type=np.int32))
    las = laspy.LasData(header)
    las.x, las.y, las.z = [381100.0], [5829000.0], [1.0]
    las.treeID = np.array([0], np.int32)
    las.write(str(b))
    with pytest.raises(ValueError, match="same results_to_las version"):
        merge_las([a, b], tmp_path / "merged.las")


def test_merge_las_error_spells_out_a_description_only_mismatch(tmp_path):
    """Names alone would print two identical lists for the 11d850b failure mode."""
    a = _result_las(tmp_path / "a.las", (381000, 5829000), [-1, 0, 1], [0, 1, 2])
    b = tmp_path / "b.las"
    header = laspy.LasHeader(point_format=6, version="1.4")
    header.scales = np.array([0.001] * 3)
    header.offsets = np.array([381100.0, 5829000.0, 0.0])
    for name, dtype in (("treeID", np.int32), ("semantic", np.uint8), ("score", np.float32)):
        header.add_extra_dim(laspy.ExtraBytesParams(name=name, type=dtype))  # no description
    las = laspy.LasData(header)
    las.x, las.y, las.z = [381100.0], [5829000.0], [1.0]
    las.write(str(b))
    with pytest.raises(ValueError) as excinfo:
        merge_las([a, b], tmp_path / "merged.las")
    message = str(excinfo.value)
    assert "ForestFormer3D instance, -1 none" in message
    assert "description must match" in message


def test_merge_las_rejects_the_same_sub_tile_twice(tmp_path):
    """A repeated path would collapse in id_offsets and collide the two copies' ids."""
    a = _result_las(tmp_path / "a.las", (381000, 5829000), [-1, 0, 1, 2], [0, 1, 2, 2])
    out = tmp_path / "merged.las"
    with pytest.raises(ValueError, match=r"a\.las more than once"):
        merge_las([a, a], out)
    assert not out.exists()


def test_merge_trees_rejects_the_same_gpkg_twice(tmp_path):
    a = _result_las(tmp_path / "a.las", (381000, 5829000), [-1, 0, 1, 2], [0, 1, 2, 2])
    ga = tmp_path / "a.gpkg"
    trees_to_gpkg(a, ga)
    out = tmp_path / "merged.gpkg"
    with pytest.raises(ValueError, match=r"a\.gpkg more than once"):
        merge_trees([ga, ga], out, {str(a): 0, "other": 3})
    assert not out.exists()


def test_merge_las_keeps_the_coordinates_when_the_sub_tiles_disagree_on_the_offset(tmp_path):
    """Sub-tiles each carry their own header offset; the merged file has one of its own.

    ``results_to_las`` and ``sat_file_to_las`` both set a sub-tile's header offset to the
    floor of that sub-tile's own minimum, so no two sub-tiles agree on it, and the merged
    header uses the floor of the global minimum. A LAS point record stores X/Y/Z as
    integers relative to its own header, so copying records between the two without
    rescaling displaces every sub-tile by the difference of the offsets -- tens of metres,
    silently, in a file that still looks well-formed.

    laspy 2.6 began rescaling on write and hid this; the image pins laspy 2.5.3, where a
    merged km tile came out with its sub-tiles shifted against each other (measured
    2026-10-20 in forestformer3d:cu118). The fix must not depend on the laspy version, so
    this test states the invariant: what goes in comes out, whatever the offsets were.
    """
    rng = np.random.default_rng(7)
    parts = []
    for i, (e, n) in enumerate(((381000.0, 5828000.0), (381100.0, 5828000.0))):
        # The 13 m of slack is what makes the offsets differ between the sub-tiles.
        x = e + rng.uniform(13.0 * i, 100.0, 40)
        y = n + rng.uniform(13.0 * (1 - i), 100.0, 40)
        z = rng.uniform(5.0, 30.0, 40)
        path = write_result_las(
            tmp_path / f"part{i}.las", x, y, z,
            np.resize(np.asarray([-1, 0, 1], np.int32), 40),
            np.resize(np.asarray([0, 1, 2], np.uint8), 40),
            score=np.full(40, 0.5, np.float32),
            offsets=np.floor([x.min(), y.min(), z.min()]),
        )
        parts.append((path, x, y, z))
    offsets = [laspy.read(p).header.offsets for p, *_ in parts]
    assert offsets[0][1] != offsets[1][1], "the sub-tiles must disagree, or nothing is tested"

    out = tmp_path / "merged.las"
    merge_las([p for p, *_ in parts], out)
    m = laspy.read(out)
    for axis, want in (("x", 1), ("y", 2), ("z", 3)):
        np.testing.assert_allclose(
            np.asarray(getattr(m, axis)),
            np.concatenate([part[want] for part in parts]),
            atol=0.002, err_msg=f"{axis} moved during the merge")
