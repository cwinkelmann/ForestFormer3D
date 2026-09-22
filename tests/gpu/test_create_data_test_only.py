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
