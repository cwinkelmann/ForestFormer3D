import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.runner_options import resolve_output_dir  # noqa: E402


def test_defaults_to_work_dir_when_unset():
    assert resolve_output_dir({'output_dir': None}, 'work_dirs/run1') == 'work_dirs/run1'
    assert resolve_output_dir({}, 'work_dirs/run1') == 'work_dirs/run1'


def test_cfg_options_value_wins():
    assert resolve_output_dir({'output_dir': '/data/out'}, 'work_dirs/run1') == '/data/out'


def test_test_py_has_no_in_memory_permutation():
    src = (Path(__file__).resolve().parents[1] / 'tools' / 'test.py').read_text()
    assert 'permute(' not in src
    assert 'torch.load' not in src
    assert 'resolve_output_dir' in src
