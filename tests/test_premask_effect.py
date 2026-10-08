"""The counting behind benchmark/premask_effect.py: which instances sit on a polygon layer."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmark"))

from premask_effect import _assign_exclusive, instance_stats  # noqa: E402


def test_instance_stats_separates_touching_from_sitting_on_the_polygons():
    #            instance:  0   0   0   1   1   -1
    tree_id = np.array([0, 0, 0, 1, 1, -1])
    inside = np.array([True, False, False, True, True, True])
    s = instance_stats(tree_id, inside, min_fraction=0.5)
    assert s["instances"] == 2
    assert s["instances_touching"] == 2          # both have a point inside
    assert s["instances_on"] == 1                # only instance 1 is mostly inside
    assert s["points_inside"] == 4               # includes the unassigned point
    assert s["points_inside_assigned"] == 3      # the unassigned one does not belong to a tree
    assert s["fraction_on"] == pytest.approx(0.5)


def test_instance_stats_is_empty_when_nothing_is_inside():
    s = instance_stats(np.array([0, 0, 1]), np.zeros(3, dtype=bool))
    assert s["instances"] == 2 and s["instances_touching"] == 0 and s["instances_on"] == 0
    assert s["points_inside"] == 0 and s["fraction_on"] == 0.0


def test_instance_stats_handles_sparse_ids_and_an_all_unassigned_tile():
    """Mosaic-wide ids are sparse per tile (a tile holds ids out of a million), and a tile
    can be all ground after the mask."""
    s = instance_stats(np.array([900_000, 900_000, 5]), np.array([True, True, False]))
    assert s["instances"] == 2 and s["instances_on"] == 1
    none = instance_stats(np.array([-1, -1]), np.array([True, False]))
    assert none["instances"] == 0 and none["fraction_on"] == 0.0 and none["instances_on"] == 0


def test_instance_stats_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="differ in length"):
        instance_stats(np.array([0, 1]), np.array([True]))


def test_assign_exclusive_gives_an_overlapping_point_to_the_first_layer():
    """A jetty lies in both the water and the structures layer; counting it in both would
    report the same tree twice."""
    water = np.array([True, True, False])
    structures = np.array([True, False, True])
    out = _assign_exclusive({"water": water, "structures": structures})
    assert out["water"].tolist() == [True, True, False]
    assert out["structures"].tolist() == [False, False, True]
    assert (out["water"] & out["structures"]).sum() == 0
