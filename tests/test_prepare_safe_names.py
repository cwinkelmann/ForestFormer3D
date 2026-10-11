import os
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / 'tools' / 'prepare_safe_testfile_names.sh'


def layout(tmp_path):
    root = tmp_path / 'data' / 'ForAINetV2'
    (root / 'meta_data').mkdir(parents=True)
    (root / 'test_data').mkdir()
    (root / 'meta_data' / 'test_list_initial.txt').write_text(
        'plot_1\nabc_fixednamefixedname\nkeep_me\n')
    for name in ['plot_1.ply', 'abc_fixednamefixedname.ply', 'keep_me.ply', 'plot_1_bluepoints_2.ply']:
        (root / 'test_data' / name).write_bytes(b'ply\n')
    (root / 'test_data' / 'plot_1').mkdir()          # a directory named like a scan
    return root


def test_files_are_renamed_together_with_the_list(tmp_path):
    root = layout(tmp_path)
    proc = subprocess.run(['bash', str(SCRIPT)], cwd=tmp_path, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    assert (root / 'meta_data' / 'test_list_initial.txt').read_text().split() == \
        ['plot_1fixedname', 'abc_fixedname', 'keep_me']
    assert (root / 'meta_data' / 'test_list_initial_original.txt').read_text().split() == \
        ['plot_1', 'abc_fixednamefixedname', 'keep_me']
    assert sorted(os.listdir(root / 'test_data')) == \
        ['abc_fixedname.ply', 'keep_me.ply', 'plot_1_bluepoints_2fixedname.ply',
         'plot_1fixedname', 'plot_1fixedname.ply']


def test_bash_syntax():
    assert subprocess.run(['bash', '-n', str(SCRIPT)]).returncode == 0
