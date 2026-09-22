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


def test_labeled_ply_uses_minus_one_for_ground_and_treeid_zero_vegetation(tmp_path):
    xyz = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 5.0], [2.0, 2.0, 5.0], [2.0, 2.0, 7.0]])
    make_layout(tmp_path, ['plot_1'])
    write_ply(tmp_path / 'test_data' / 'plot_1.ply', xyz,
              semantic=[1, 1, 3, 2, 2], tree_id=[0, 0, 0, 7, 7])
    proc = run(tmp_path, '--unlabeled')            # flag must not disturb labeled scans
    assert proc.returncode == 0, proc.stdout + proc.stderr
    out = tmp_path / 'forainetv2_instance_data'
    assert np.load(out / 'plot_1_sem_label.npy').tolist() == [0, 0, 2, 1, 1]
    assert np.load(out / 'plot_1_ins_label.npy').tolist() == [-1, -1, -1, 7, 7]
    assert np.load(out / 'plot_1_unaligned_bbox.npy').shape == (1, 7)


def test_script_has_no_gpu_imports():
    src = SCRIPT.read_text()
    for name in ['import torch', 'import segmentator', 'import open3d', 'Delaunay', "'fast'"]:
        assert name not in src, name
