# Modified from
# https://github.com/facebookresearch/votenet/blob/master/scannet/load_scannet_data.py
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Load FOR-instance dataset with ground truth labels for semantic and
instance segmentations."""
import argparse
import inspect
import json
import os
import sys

import numpy as np

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
from tools.plyutils import read_ply  # noqa: E402

currentdir = os.path.dirname(
    os.path.abspath(inspect.getfile(inspect.currentframe())))


def read_aggregation(filename):
    assert os.path.isfile(filename)
    object_id_to_segs = {}
    label_to_segs = {}
    with open(filename) as f:
        data = json.load(f)
        num_objects = len(data['segGroups'])
        for i in range(num_objects):
            object_id = data['segGroups'][i][
                'objectId'] + 1  # instance ids should be 1-indexed
            label = data['segGroups'][i]['label']
            segs = data['segGroups'][i]['segments']
            object_id_to_segs[object_id] = segs
            if label in label_to_segs:
                label_to_segs[label].extend(segs)
            else:
                label_to_segs[label] = segs
    return object_id_to_segs, label_to_segs


def read_segmentation(filename):
    assert os.path.isfile(filename)
    seg_to_verts = {}
    with open(filename) as f:
        data = json.load(f)
        num_verts = len(data['segIndices'])
        for i in range(num_verts):
            seg_id = data['segIndices'][i]
            if seg_id in seg_to_verts:
                seg_to_verts[seg_id].append(i)
            else:
                seg_to_verts[seg_id] = [i]
    return seg_to_verts, num_verts


def extract_bbox(mesh_vertices, label_ids, instance_ids, bg_sem=np.array([0])):
    # Filter out background points
    valid_mask = ~np.isin(label_ids, bg_sem)
    mesh_vertices = mesh_vertices[valid_mask]
    instance_ids = instance_ids[valid_mask]
    label_ids = label_ids[valid_mask]

    # Get the number of unique instances
    unique_instance_ids = np.unique(instance_ids)
    num_instances = len(unique_instance_ids)

    # Initialize instance_bboxes
    instance_bboxes = np.zeros((num_instances, 7))

    for i, instance_id in enumerate(unique_instance_ids):
        # Select points corresponding to the current instance
        mask = instance_ids == instance_id
        pts = mesh_vertices[mask, :3]

        if pts.shape[0] == 0:
            continue

        # Calculate min_pts, max_pts, locations, and dimensions
        min_pts = pts.min(axis=0)
        max_pts = pts.max(axis=0)
        locations = (min_pts + max_pts) / 2
        dimensions = max_pts - min_pts

        # Store the results in instance_bboxes
        instance_bboxes[i, :3] = locations
        instance_bboxes[i, 3:6] = dimensions
        instance_bboxes[i, 6] = 1

    return instance_bboxes


def export(ply_file,
           output_file=None,
           test_mode=False,
           unlabeled=False):
    """Export one PLY to vert, ins_label, sem_label and bbox arrays.

    Args:
        ply_file (str): Path of the ply file (binary PLY with x, y, z and,
            unless ``unlabeled``, ``semantic_seg`` (1 ground, 2 wood, 3 leaf)
            and ``treeID`` (0 = no tree)).
        output_file (str): Output prefix; None returns arrays only.
        test_mode (bool): Skip bounding boxes.
        unlabeled (bool): Accept files without label fields and write constant
            labels (semantic 0 = ground, instance -1).

    Returns:
        points (float32 Nx3, centred: mean x, mean y, min z subtracted),
        label_ids (int64: 0 ground, 1 wood, 2 leaf),
        instance_ids (int64: for labeled scans, the original convention —
            raw treeID for every non-ground point (0 for vegetation with no
            tree id), 0 for ground; for unlabeled scans, -1 everywhere),
        unaligned_bboxes, aligned_bboxes (Kx7), axis_align_matrix (4x4),
        offsets (float64 [mean_x, mean_y, min_z]).
    """
    pcd = read_ply(ply_file)
    points = np.vstack((pcd['x'], pcd['y'], pcd['z'])).astype(np.float64).T

    is_blue = 'bluepoints' in os.path.basename(ply_file)
    if is_blue:
        offsets = np.zeros(3, dtype=np.float64)
    else:
        mean_x = np.mean(points[:, 0])
        mean_y = np.mean(points[:, 1])
        min_z = np.min(points[:, 2])
        offsets = np.array([mean_x, mean_y, min_z], dtype=np.float64)
        points[:, 0] -= mean_x
        points[:, 1] -= mean_y
        points[:, 2] -= min_z
    points = points.astype(np.float32)

    fields = set(pcd.dtype.names)
    has_labels = {'semantic_seg', 'treeID'} <= fields
    if not has_labels and not unlabeled:
        raise KeyError(f'{ply_file} has no semantic_seg/treeID fields; '
                       f'run batch_load_ForAINetV2_data.py --unlabeled to write constant labels')

    bg_sem = np.array([0])
    if has_labels:
        # Original on-disk convention (byte-for-byte, do not change: Phase 2
        # benchmarks the old model code on data this loader already produced):
        # ground -> -1 temporarily, then remapped to 0; every other point
        # (including vegetation with treeID == 0) keeps its raw treeID.
        label_ids = pcd['semantic_seg'].astype(np.int64) - 1        # 0 ground, 1 wood, 2 leaf
        instance_ids = pcd['treeID'].astype(np.int64)
        instance_ids[np.isin(label_ids, bg_sem)] = -1                # ground -> -1
        valid_mask = instance_ids != -1
        new_instance_ids = np.zeros_like(instance_ids)
        new_instance_ids[valid_mask] = instance_ids[valid_mask]      # raw treeID (incl. 0) elsewhere
        new_instance_ids[instance_ids == -1] = 0                     # ground -> 0
        instance_ids = new_instance_ids
    else:
        label_ids = np.zeros(points.shape[0], dtype=np.int64)
        instance_ids = np.full(points.shape[0], -1, dtype=np.int64)

    axis_align_matrix = np.eye(4)
    if not test_mode:
        # Original code ran points through an (always-identity) alignment
        # matmul before computing aligned_bboxes; because that promotes the
        # float32 points to float64, the aligned/unaligned boxes differ by a
        # few ULPs even though the transform is the identity. Reproduced
        # exactly so the on-disk bytes match what this loader always wrote.
        pts_h = np.ones((points.shape[0], 4))
        pts_h[:, 0:3] = points[:, 0:3]
        aligned_mesh_vertices = np.dot(pts_h, axis_align_matrix.transpose())[:, 0:3]
        unaligned_bboxes = extract_bbox(points, label_ids, instance_ids, bg_sem)
        aligned_bboxes = extract_bbox(aligned_mesh_vertices, label_ids, instance_ids, bg_sem)
    else:
        unaligned_bboxes = None
        aligned_bboxes = None

    if output_file is not None:
        np.save(output_file + '_vert.npy', points)
        np.save(output_file + '_offsets.npy', offsets)
        np.save(output_file + '_sem_label.npy', label_ids)
        np.save(output_file + '_ins_label.npy', instance_ids)
        if not test_mode:
            np.save(output_file + '_unaligned_bbox.npy', unaligned_bboxes)
            np.save(output_file + '_aligned_bbox.npy', aligned_bboxes)
            np.save(output_file + '_axis_align_matrix.npy', axis_align_matrix)

    return points, label_ids, instance_ids, unaligned_bboxes, \
        aligned_bboxes, axis_align_matrix, offsets
