"""Tests for oneformer3d/labels.py (pure numpy, no torch needed)."""
import numpy as np

from oneformer3d.labels import (compact_instance_ids, compact_instance_ids_with_ratio,
                                normalize_instance_gt)


def test_normalize_raw_scene_marks_ground_and_unannotated():
    # point0: ground (sem 0, raw id 0) -> unannotated
    # point1: unannotated vegetation (sem 2, raw id 0) -> unannotated
    # point2: tree, raw id 7 -> first real instance seen -> compacted to 0
    # point3: tree, raw id 3 -> second real instance seen -> compacted to 1
    semantic = np.array([0, 2, 1, 1])
    instance = np.array([0, 0, 7, 3])
    out = normalize_instance_gt(semantic, instance, raw=True)
    assert out.dtype == np.int64
    assert out.tolist() == [-1, -1, 0, 1]


def test_normalize_raw_negative_instance_is_unannotated():
    semantic = np.array([1, 1])
    instance = np.array([-1, 5])
    out = normalize_instance_gt(semantic, instance, raw=True)
    assert out.tolist() == [-1, 0]


def test_normalize_is_idempotent_with_raw_false():
    semantic = np.array([0, 2, 1, 1, 1])
    instance = np.array([0, 0, 7, 3, 7])
    first_pass = normalize_instance_gt(semantic, instance, raw=True)
    second_pass = normalize_instance_gt(semantic, first_pass, raw=False)
    assert np.array_equal(first_pass, second_pass)


def test_normalize_non_raw_treats_zero_as_valid_instance():
    # Downstream of the crop pipeline, id 0 is a real instance, not "unannotated".
    semantic = np.array([1, 1])
    instance = np.array([0, 0])
    out = normalize_instance_gt(semantic, instance, raw=False)
    assert out.tolist() == [0, 0]


def test_compact_instance_ids_keeps_minus_one_and_compacts_rest():
    out = compact_instance_ids(np.array([5, 5, 9, -1]))
    assert out.tolist() == [0, 0, 1, -1]


def test_compact_instance_ids_with_ratio_remaps_the_dict():
    mask = np.array([-1, 3, 3, 7, 9])
    new_mask, ratio = compact_instance_ids_with_ratio(
        mask, {3: 0.5, 5: 0.9, 7: 0.25, 9: 1.0})
    assert new_mask.tolist() == [-1, 0, 0, 1, 2]
    assert ratio == {0: 0.5, 1: 0.25, 2: 1.0}   # 5 vanished, keys follow the new ids


def test_compact_instance_ids_with_ratio_follows_appearance_order():
    # 9 appears before 3, so 9 -> 0 and 3 -> 1; the ratios must follow.
    new_mask, ratio = compact_instance_ids_with_ratio(
        np.array([9, 3, 9, -1]), {3: 0.25, 9: 1.0})
    assert new_mask.tolist() == [0, 1, 0, -1]
    assert ratio == {0: 1.0, 1: 0.25}


def test_compact_instance_ids_with_ratio_returns_none_without_a_dict():
    new_mask, ratio = compact_instance_ids_with_ratio(np.array([4, 4, -1]))
    assert new_mask.tolist() == [0, 0, -1]
    assert ratio is None


def test_compact_instance_ids_with_ratio_handles_an_empty_scene():
    new_mask, ratio = compact_instance_ids_with_ratio(np.array([-1, -1]), {0: 1.0})
    assert new_mask.tolist() == [-1, -1]
    assert ratio == {}
