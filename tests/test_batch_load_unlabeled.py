import subprocess
import sys
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / 'data' / 'ForAINetV2' / 'batch_load_ForAINetV2_data.py'


def write_ply(path, xyz, semantic=None, tree_id=None):
    fields = [('x', 'f8'), ('y', 'f8'), ('z', 'f8')]
    if semantic is not None:
        fields += [('semantic_seg', 'i4'), ('treeID', 'i4')]
    vertex = np.zeros(len(xyz), dtype=fields)
    vertex['x'], vertex['y'], vertex['z'] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    if semantic is not None:
        vertex['semantic_seg'], vertex['treeID'] = semantic, tree_id
    PlyData([PlyElement.describe(vertex, 'vertex')], text=False).write(str(path))


def make_layout(tmp_path, scans):
    (tmp_path / 'meta_data').mkdir()
    (tmp_path / 'test_data').mkdir()
    (tmp_path / 'meta_data' / 'test_list.txt').write_text('\n'.join(scans) + '\n')
    # no train_val_data, no train/val lists: only test data is present
    return tmp_path


def run(cwd, *extra):
    return subprocess.run([sys.executable, str(SCRIPT), *extra], cwd=cwd,
                          capture_output=True, text=True)


def test_unlabeled_ply_produces_constant_labels(tmp_path):
    xyz = np.array([[10.0, 20.0, 5.0], [12.0, 22.0, 6.0], [14.0, 24.0, 7.0], [16.0, 26.0, 8.0]])
    make_layout(tmp_path, ['tile_a'])
    write_ply(tmp_path / 'test_data' / 'tile_a.ply', xyz)
    proc = run(tmp_path, '--unlabeled')
    assert proc.returncode == 0, proc.stdout + proc.stderr
    out = tmp_path / 'forainetv2_instance_data'
    for suffix in ['vert', 'offsets', 'sem_label', 'ins_label', 'unaligned_bbox',
                   'aligned_bbox', 'axis_align_matrix']:
        assert (out / f'tile_a_{suffix}.npy').exists(), suffix
    vert = np.load(out / 'tile_a_vert.npy')
    assert vert.dtype == np.float32 and vert.shape == (4, 3)
    np.testing.assert_allclose(np.load(out / 'tile_a_offsets.npy'), [13.0, 23.0, 5.0])
    np.testing.assert_allclose(vert.mean(0)[:2], [0.0, 0.0], atol=1e-6)
    assert vert[:, 2].min() == 0.0
    assert np.load(out / 'tile_a_sem_label.npy').tolist() == [0, 0, 0, 0]
    assert np.load(out / 'tile_a_ins_label.npy').tolist() == [-1, -1, -1, -1]
    assert np.load(out / 'tile_a_unaligned_bbox.npy').shape == (0, 7)
    assert 'skipping' in proc.stdout.lower()          # train/val splits absent


def test_unlabeled_ply_without_flag_fails_loudly(tmp_path):
    make_layout(tmp_path, ['tile_a'])
    write_ply(tmp_path / 'test_data' / 'tile_a.ply', np.zeros((3, 3)))
    proc = run(tmp_path)
    assert proc.returncode == 1
    assert '--unlabeled' in proc.stdout + proc.stderr


def test_labeled_ply_keeps_ground_and_treeid_zero_vegetation_at_zero(tmp_path):
    # Original on-disk convention (R3, unchanged): ground -> 0, vegetation with
    # treeID == 0 -> 0 too (both end up indistinguishable at 0 on disk); every
    # other point keeps its raw treeID. Verified against the un-refactored
    # `export()` at commit 6a75c37 in test_labeled_ply_matches_original_export_output
    # below.
    xyz = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 5.0], [2.0, 2.0, 5.0], [2.0, 2.0, 7.0]])
    make_layout(tmp_path, ['plot_1'])
    write_ply(tmp_path / 'test_data' / 'plot_1.ply', xyz,
              semantic=[1, 1, 3, 2, 2], tree_id=[0, 0, 0, 7, 7])
    proc = run(tmp_path, '--unlabeled')            # flag must not disturb labeled scans
    assert proc.returncode == 0, proc.stdout + proc.stderr
    out = tmp_path / 'forainetv2_instance_data'
    assert np.load(out / 'plot_1_sem_label.npy').tolist() == [0, 0, 2, 1, 1]
    assert np.load(out / 'plot_1_ins_label.npy').tolist() == [0, 0, 0, 7, 7]
    # ground points (label 0) are excluded from bboxes, but the single
    # treeID-0 vegetation point is NOT (extract_bbox only filters by semantic
    # label, not instance id) -- this is the original code's behavior and is
    # reproduced on purpose: 1 degenerate box for the treeID-0 point, 1 for
    # the tree.
    assert np.load(out / 'plot_1_unaligned_bbox.npy').shape == (2, 7)


def test_labeled_ply_matches_original_export_output(tmp_path):
    """Regression test for R3: the labeled path must be byte-identical to the
    original (pre-refactor) `export()`.

    The original code (commit 6a75c37, unchanged since the very first commit
    of this file) contained a triple-quoted block that *looked* like it
    compacted tree ids to 1..K by first appearance, but that block was dead
    code (a no-op string literal) -- it was never executed. The code that
    actually ran kept the raw treeID for every non-ground point (including 0
    for vegetation with no tree id) and only remapped ground to 0. Verified
    empirically by extracting `git show 6a75c37:data/ForAINetV2/load_forainetv2_data.py`
    into a standalone module and running its `export()` on this exact fixture
    side by side with the current `export()`; every returned array
    (points, label_ids, instance_ids, unaligned_bboxes, aligned_bboxes,
    axis_align_matrix, offsets) was `np.array_equal`. This test pins the
    written-to-disk arrays that comparison produced -- do not "fix" the
    tree ids to be 1, 2 here, that would reintroduce the on-disk
    incompatibility R3 forbids.
    """
    xyz = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 5.0],   # vegetation, treeID 0
        [2.0, 2.0, 5.0],   # tree 7
        [2.1, 2.1, 5.1],   # tree 7
        [5.0, 5.0, 6.0],   # tree 3
        [5.1, 5.1, 6.1],   # tree 3
    ])
    make_layout(tmp_path, ['plot_2'])
    write_ply(tmp_path / 'test_data' / 'plot_2.ply', xyz,
              semantic=[1, 1, 3, 2, 2, 2, 2], tree_id=[0, 0, 0, 7, 7, 3, 3])
    proc = run(tmp_path, '--unlabeled')            # flag must not disturb labeled scans
    assert proc.returncode == 0, proc.stdout + proc.stderr
    out = tmp_path / 'forainetv2_instance_data'
    assert np.load(out / 'plot_2_sem_label.npy').tolist() == [0, 0, 2, 1, 1, 1, 1]
    # Ground -> 0, treeID-0 vegetation -> 0, trees keep their RAW treeID
    # (7 and 3, not compacted to 1 and 2) -- this is what the original code
    # actually produced.
    assert np.load(out / 'plot_2_ins_label.npy').tolist() == [0, 0, 0, 7, 7, 3, 3]
    # 3 instances after filtering ground: the treeID-0 point, tree 3, tree 7.
    assert np.load(out / 'plot_2_unaligned_bbox.npy').shape == (3, 7)
    assert np.load(out / 'plot_2_aligned_bbox.npy').shape == (3, 7)


def test_script_has_no_gpu_imports():
    src = SCRIPT.read_text()
    for name in ['import torch', 'import segmentator', 'import open3d', 'Delaunay', "'fast'"]:
        assert name not in src, name
