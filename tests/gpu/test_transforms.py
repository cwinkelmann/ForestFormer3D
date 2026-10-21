"""Transforms keep instance ids compact and ``ratio_inspoint`` keyed by them.

The helper itself (``compact_instance_ids_with_ratio``) is pure numpy and is
covered by ``tests/test_labels.py``, which runs without torch.
"""
import numpy as np
import pytest
import torch
from mmdet3d.structures import DepthPoints

from oneformer3d.transforms_3d import (CylinderCrop, GridSample,
                                       PointInstClassMapping_, PointSample_)

pytestmark = pytest.mark.gpu


def test_point_sample_is_without_replacement():
    pts = DepthPoints(torch.rand(1000, 3), points_dim=3)
    _, choices = PointSample_(num_points=500)._points_random_sampling(pts, 500)
    assert len(choices) == 500
    assert len(np.unique(choices)) == 500


def voxel_drop_scene():
    """Tree 0 is a single point sharing a 0.2 m voxel with a point of tree 1.

    The semantic mask is abused as a tracer (10 + original id) so the test can
    tell which tree a surviving point came from whichever point GridSample keeps.
    """
    xyz = torch.tensor([
        [0.10, 0.10, 0.10],          # tree 0 (only point)
        [0.10, 0.10, 0.15],          # tree 1, same voxel as tree 0
        [3.0, 0.0, 0.0], [3.0, 0.0, 1.0], [3.0, 0.0, 2.0], [3.0, 0.0, 3.0],   # tree 1
        [6.0, 0.0, 0.0], [6.0, 0.0, 1.0], [6.0, 0.0, 2.0],                     # tree 2
        [9.0, 0.0, 0.0],                                                       # ground
    ])
    inst = np.array([0, 1, 1, 1, 1, 1, 2, 2, 2, -1])
    tracer = np.where(inst >= 0, inst + 10, 0)
    ratio = {0: 0.5, 1: 0.25, 2: 1.0}
    return dict(points=DepthPoints(xyz, points_dim=3), pts_instance_mask=inst,
                pts_semantic_mask=tracer, ratio_inspoint=dict(ratio),
                instance_mask=inst >= 0, vote_label=torch.zeros(10, 3)), ratio


def assert_ratio_follows_ids(out, original_ratio):
    inst, tracer, ratio = out['pts_instance_mask'], out['pts_semantic_mask'], out['ratio_inspoint']
    ids = np.unique(inst[inst >= 0])
    assert ids.tolist() == list(range(len(ids)))
    assert set(ratio) == set(ids.tolist())
    for new_id in ids:
        original = int(tracer[inst == new_id][0]) - 10
        assert ratio[new_id] == original_ratio[original]


def test_grid_sample_remaps_ratio_after_voxel_drop():
    scene, ratio = voxel_drop_scene()
    out = GridSample(grid_size=0.2).transform(scene)
    assert len(out['points']) == 9                       # one of the two colliding points dropped
    assert_ratio_follows_ids(out, ratio)


def test_point_sample_remaps_ratio_after_drop():
    scene, ratio = voxel_drop_scene()
    out = PointSample_(num_points=8).transform(scene)
    assert len(out['points']) == 8
    assert_ratio_follows_ids(out, ratio)


def test_point_inst_class_mapping_keeps_ratio_keys():
    scene, ratio = voxel_drop_scene()
    scene['pts_semantic_mask'] = np.where(scene['pts_instance_mask'] >= 0, 1, 0)
    out = PointInstClassMapping_(num_classes=3).transform(scene)
    assert out['ratio_inspoint'] == ratio
    assert out['gt_labels_3d'].tolist() == [1, 1, 1]


def test_cylinder_crop_ignores_vegetation_without_tree_id():
    xyz = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0],        # ground (raw id 0)
                        [0.0, 1.0, 5.0], [0.0, 1.5, 6.0],        # vegetation, treeID 0 -> ignore
                        [2.0, 2.0, 5.0], [2.0, 2.0, 7.0],        # tree 5
                        [4.0, 4.0, 5.0]])                        # tree 9
    scene = dict(points=DepthPoints(xyz, points_dim=3),
                 pts_instance_mask=np.array([0, 0, 0, 0, 5, 5, 9]),
                 pts_semantic_mask=np.array([0, 0, 2, 2, 1, 1, 1]))
    out = CylinderCrop(radius=100).transform(scene)       # radius covers every point
    assert out['pts_instance_mask'].tolist() == [-1, -1, -1, -1, 0, 0, 1]
    assert np.asarray(out['instance_mask']).tolist() == [False, False, False, False,
                                                         True, True, True]
    assert out['ratio_inspoint'] == {0: 1.0, 1: 1.0}
    assert out['pts_semantic_mask'].tolist() == [0, 0, 2, 2, 1, 1, 1]
    assert np.isnan(out['vote_label'][2]).all() and not np.isnan(out['vote_label'][4]).any()


def test_cylinder_crop_accepts_minus_one_ground():
    """The loader now writes -1 for ground; the crop must not re-map it."""
    xyz = torch.tensor([[0.0, 0.0, 0.0], [2.0, 2.0, 5.0], [2.0, 2.0, 7.0]])
    scene = dict(points=DepthPoints(xyz, points_dim=3),
                 pts_instance_mask=np.array([-1, 4, 4]),
                 pts_semantic_mask=np.array([0, 1, 1]))
    out = CylinderCrop(radius=100).transform(scene)
    assert out['pts_instance_mask'].tolist() == [-1, 0, 0]
    assert out['ratio_inspoint'] == {0: 1.0}
