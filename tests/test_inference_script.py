import os
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / 'tools' / 'inference_bluepoint.sh'


def test_bash_syntax_and_strict_mode():
    assert subprocess.run(['bash', '-n', str(SCRIPT)]).returncode == 0
    head = SCRIPT.read_text().splitlines()[:5]
    assert head[0] == '#!/usr/bin/env bash'
    assert any('set -euo pipefail' in line for line in head)
    assert 'sed -i' not in SCRIPT.read_text()


def test_dry_run_echoes_commands_without_touching_files(tmp_path):
    work = tmp_path / 'repo'
    (work / 'data' / 'ForAINetV2' / 'meta_data').mkdir(parents=True)
    (work / 'data' / 'ForAINetV2' / 'test_data').mkdir()
    (work / 'data' / 'ForAINetV2' / 'meta_data' / 'test_list_initial.txt').write_text('plot_7\n')
    env = {**os.environ, 'DRY_RUN': '1', 'WORK_DIR': str(work), 'ITERATIONS': '1',
           'SCORE_TH': '0.35', 'MODEL_PATH': '/ckpt/epoch_3000_fix.pth'}
    proc = subprocess.run(['bash', str(SCRIPT)], env=env, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    out = proc.stdout
    assert 'DRY: python batch_load_ForAINetV2_data.py --test_scan_names_file meta_data/test_list.txt' in out
    assert 'DRY: python tools/create_data_forainetv2.py forainetv2' in out
    assert ('DRY: python tools/test.py' in out and '/ckpt/epoch_3000_fix.pth' in out
            and '--cfg-options model.test_cfg.score_th=0.35' in out
            and f'model.test_cfg.output_dir={work}/work_dirs/bluepoints' in out)
    assert 'DRY: python tools/final_eval.py' in out
    assert not (work / 'data' / 'ForAINetV2' / 'meta_data' / 'test_list.txt').exists()
