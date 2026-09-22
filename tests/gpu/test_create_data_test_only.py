"""create_data_forainetv2.py accepts a run with only test data preprocessed."""
import os
import subprocess
import sys
from pathlib import Path

import mmengine
import numpy as np
import pytest

pytestmark = pytest.mark.gpu
REPO = Path(__file__).resolve().parents[2]


def test_test_only_run_writes_only_the_test_pkl(tmp_path):
    root = tmp_path / 'ForAINetV2'
    (root / 'meta_data').mkdir(parents=True)
    (root / 'forainetv2_instance_data').mkdir()
    for split, scans in [('train', ['tr_1']), ('val', ['va_1']), ('test', ['te_1'])]:
        (root / 'meta_data' / f'{split}_list.txt').write_text('\n'.join(scans) + '\n')
    inst = root / 'forainetv2_instance_data'
    np.save(inst / 'te_1_vert.npy', np.zeros((5, 3), np.float32))
    np.save(inst / 'te_1_offsets.npy', np.zeros(3))
    np.save(inst / 'te_1_sem_label.npy', np.zeros(5, np.int64))
    np.save(inst / 'te_1_ins_label.npy', np.full(5, -1, np.int64))
    np.save(inst / 'te_1_unaligned_bbox.npy', np.zeros((0, 7)))
    np.save(inst / 'te_1_aligned_bbox.npy', np.zeros((0, 7)))
    np.save(inst / 'te_1_axis_align_matrix.npy', np.eye(4))

    proc = subprocess.run(
        [sys.executable, 'tools/create_data_forainetv2.py', 'forainetv2',
         '--root-path', str(root), '--out-dir', str(root)],
        cwd=REPO, capture_output=True, text=True, env={**os.environ, 'PYTHONPATH': str(REPO)})
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert (root / 'forainetv2_oneformer3d_infos_test.pkl').exists()
    assert not (root / 'forainetv2_oneformer3d_infos_train.pkl').exists()
    assert not (root / 'forainetv2_oneformer3d_infos_val.pkl').exists()
    infos = mmengine.load(root / 'forainetv2_oneformer3d_infos_test.pkl')
    assert infos['metainfo']['classes'] == ('tree',)
    assert len(infos['data_list']) == 1


def _write_scan(inst, name):
    np.save(inst / f'{name}_vert.npy', np.zeros((5, 3), np.float32))
    np.save(inst / f'{name}_offsets.npy', np.zeros(3))
    np.save(inst / f'{name}_sem_label.npy', np.zeros(5, np.int64))
    np.save(inst / f'{name}_ins_label.npy', np.full(5, -1, np.int64))
    np.save(inst / f'{name}_unaligned_bbox.npy', np.zeros((0, 7)))
    np.save(inst / f'{name}_aligned_bbox.npy', np.zeros((0, 7)))
    np.save(inst / f'{name}_axis_align_matrix.npy', np.eye(4))


def test_private_test_list_and_out_dir_leave_the_tracked_pkls_alone(tmp_path):
    """``--test-list <file> --splits test --out-dir <other>`` (ff3d_geo's run pipeline).

    Only the test pkl is written, into the other directory, built from the private
    list -- not from meta_data/test_list.txt -- and nothing in the data root changes.
    """
    root = tmp_path / 'ForAINetV2'
    (root / 'meta_data').mkdir(parents=True)
    inst = root / 'forainetv2_instance_data'
    inst.mkdir()
    # The tracked lists name `tracked_1`; the private list names `private_1`. Both are
    # preprocessed, so picking the wrong list still produces a pkl -- with the wrong scan.
    for split in ('train', 'val', 'test'):
        (root / 'meta_data' / f'{split}_list.txt').write_text('tracked_1\n')
    _write_scan(inst, 'tracked_1')
    _write_scan(inst, 'private_1')

    private_list = tmp_path / 'scan_list.txt'
    private_list.write_text('private_1\n')
    out = tmp_path / 'tegel-out'
    out.mkdir()
    # A sentinel train pkl in the out dir must survive `--splits test` untouched.
    sentinel = out / 'forainetv2_oneformer3d_infos_train.pkl'
    sentinel.write_bytes(b'sentinel')

    proc = subprocess.run(
        [sys.executable, 'tools/create_data_forainetv2.py', 'forainetv2',
         '--root-path', str(root), '--out-dir', str(out),
         '--test-list', str(private_list), '--splits', 'test'],
        cwd=REPO, capture_output=True, text=True, env={**os.environ, 'PYTHONPATH': str(REPO)})
    assert proc.returncode == 0, proc.stdout + proc.stderr

    test_pkl = out / 'forainetv2_oneformer3d_infos_test.pkl'
    assert test_pkl.exists()
    assert not (out / 'forainetv2_oneformer3d_infos_val.pkl').exists()
    assert sentinel.read_bytes() == b'sentinel'
    # nothing was written into the data root itself
    for split in ('train', 'val', 'test'):
        assert not (root / f'forainetv2_oneformer3d_infos_{split}.pkl').exists()

    infos = mmengine.load(test_pkl)
    assert len(infos['data_list']) == 1
    assert infos['data_list'][0]['lidar_points']['lidar_path'] == 'private_1.bin'
