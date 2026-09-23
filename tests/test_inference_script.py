import os
import subprocess

import pytest

from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / 'tools' / 'inference_bluepoint.sh'
BERLIN_RUN_GPU_SCRIPT = REPO / 'benchmark' / 'berlin_run_gpu.sh'
BERLIN_STITCH_SCRIPT = REPO / 'benchmark' / 'berlin_stitch.sh'


def test_bash_syntax_and_strict_mode():
    assert subprocess.run(['bash', '-n', str(SCRIPT)]).returncode == 0
    head = SCRIPT.read_text().splitlines()[:5]
    assert head[0] == '#!/usr/bin/env bash'
    assert any('set -euo pipefail' in line for line in head)
    assert 'sed -i' not in SCRIPT.read_text()


@pytest.mark.parametrize('script', [BERLIN_RUN_GPU_SCRIPT, BERLIN_STITCH_SCRIPT])
def test_berlin_scripts_bash_syntax(script):
    assert subprocess.run(['bash', '-n', str(script)]).returncode == 0


def test_berlin_run_gpu_splits_with_a_halo_and_does_not_merge():
    text = BERLIN_RUN_GPU_SCRIPT.read_text()
    assert 'ff3d_geo merge' not in text
    assert '--buffer 20' in text


def test_dry_run_echoes_commands_without_touching_files(tmp_path):
    work = tmp_path / 'repo'
    (work / 'data' / 'ForAINetV2' / 'meta_data').mkdir(parents=True)
    (work / 'data' / 'ForAINetV2' / 'test_data').mkdir()
    (work / 'data' / 'ForAINetV2' / 'meta_data' / 'test_list_initial.txt').write_text('plot_7\n')
    tracked_list = work / 'data' / 'ForAINetV2' / 'meta_data' / 'test_list.txt'
    tracked_list.write_text('sentinel-untouched\n')
    tracked_before = tracked_list.read_bytes()

    tmp_dir = Path(os.environ.get('TMPDIR', '/tmp'))
    temp_lists_before = set(tmp_dir.glob('ff3d_test_list.*'))

    env = {**os.environ, 'DRY_RUN': '1', 'WORK_DIR': str(work), 'ITERATIONS': '1',
           'SCORE_TH': '0.35', 'MODEL_PATH': '/ckpt/epoch_3000_fix.pth'}
    proc = subprocess.run(['bash', str(SCRIPT)], env=env, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    out = proc.stdout

    batch_load_lines = [line for line in out.splitlines()
                         if 'batch_load_ForAINetV2_data.py' in line]
    assert batch_load_lines, out
    for line in batch_load_lines:
        assert line.startswith('DRY: python batch_load_ForAINetV2_data.py --test_scan_names_file ')
        list_path = line.rsplit(' ', 1)[-1]
        assert 'meta_data' not in list_path, line
        assert list_path.startswith('/'), line

    create_data_lines = [line for line in out.splitlines()
                         if 'tools/create_data_forainetv2.py' in line]
    assert create_data_lines, out
    for line in create_data_lines:
        # The pkl must be built from the same private list batch_load exported, not
        # from the tracked meta_data/test_list.txt, and only for the test split.
        assert line.startswith('DRY: python tools/create_data_forainetv2.py forainetv2 '
                               '--test-list '), line
        assert line.endswith(' --splits test'), line
        list_path = line.split(' --test-list ', 1)[1].split(' --splits ', 1)[0]
        assert 'meta_data' not in list_path, line
        assert list_path.startswith('/'), line
    assert ('DRY: python tools/test.py' in out and '/ckpt/epoch_3000_fix.pth' in out
            and '--cfg-options model.test_cfg.score_th=0.35' in out
            and f'model.test_cfg.output_dir={work}/work_dirs/bluepoints' in out)
    assert 'DRY: python tools/final_eval.py' in out

    # The tracked meta_data/test_list.txt must never be touched by this script.
    assert tracked_list.read_bytes() == tracked_before

    # The default temp scan list must be cleaned up on exit: no new
    # ff3d_test_list.* files left behind beyond whatever pre-existed.
    temp_lists_after = set(tmp_dir.glob('ff3d_test_list.*'))
    assert temp_lists_after - temp_lists_before == set()


def test_dry_run_succeeds_without_instance_data_dir(tmp_path):
    # Regression for the find -delete guard: under set -euo pipefail, `find`
    # on a missing directory exits 1 and would abort the whole script before
    # it processes any scan. A fresh checkout (or any DATA_ROOT override that
    # has not been preprocessed yet) has no forainetv2_instance_data/ at all,
    # so this layout must not create it.
    work = tmp_path / 'repo'
    (work / 'data' / 'ForAINetV2' / 'meta_data').mkdir(parents=True)
    (work / 'data' / 'ForAINetV2' / 'test_data').mkdir()
    (work / 'data' / 'ForAINetV2' / 'meta_data' / 'test_list_initial.txt').write_text('plot_7\n')
    instance_data_dir = work / 'data' / 'ForAINetV2' / 'forainetv2_instance_data'
    assert not instance_data_dir.exists()

    env = {**os.environ, 'DRY_RUN': '1', 'WORK_DIR': str(work), 'ITERATIONS': '1',
           'SCORE_TH': '0.35', 'MODEL_PATH': '/ckpt/epoch_3000_fix.pth'}
    proc = subprocess.run(['bash', str(SCRIPT)], env=env, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    assert not instance_data_dir.exists()


def test_caller_supplied_test_list_is_never_deleted(tmp_path):
    work = tmp_path / 'repo'
    (work / 'data' / 'ForAINetV2' / 'meta_data').mkdir(parents=True)
    (work / 'data' / 'ForAINetV2' / 'test_data').mkdir()
    (work / 'data' / 'ForAINetV2' / 'meta_data' / 'test_list_initial.txt').write_text('plot_7\n')

    caller_list = tmp_path / 'caller_owned_list.txt'
    caller_list.write_text('caller-content\n')

    env = {**os.environ, 'DRY_RUN': '1', 'WORK_DIR': str(work), 'ITERATIONS': '1',
           'SCORE_TH': '0.35', 'MODEL_PATH': '/ckpt/epoch_3000_fix.pth',
           'TEST_LIST': str(caller_list)}
    proc = subprocess.run(['bash', str(SCRIPT)], env=env, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr

    # The script must never delete a caller-supplied TEST_LIST, even though
    # it deletes its own default temp file on exit.
    assert caller_list.exists()
    assert f'--test_scan_names_file {caller_list}' in proc.stdout
