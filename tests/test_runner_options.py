import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

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


def test_hygiene_files_are_gone():
    tracked = subprocess.run(['git', 'ls-files'], cwd=REPO, capture_output=True, text=True).stdout.split()
    for path in ['oneformer3d/oneformer3d_speedup_v1.py', 'oneformer3d/oneformer3d_withoutspeedup.py',
                 'oneformer3d/mink_unet.py', 'tools/merge_prediction_slow.py', 'tools/copy_predictions.py',
                 'data/ForAINetV2/second_inference.py', 'data/ForAINetV2/compare_outputs.py',
                 'data/ForAINetV2/run_and_compare.sh', 'data/ForAINetV2/plyutils.py',
                 'segmentator/test_equivariance.py']:
        assert path not in tracked, path
    assert not [p for p in tracked if p.endswith('.pyc')]
    assert 'CLAUDE.md' in tracked
    assert 'docs/known-issues.md' in tracked
    assert 'segment_any_tree_gpu.docker' not in tracked
    assert 'mink_unet' not in (REPO / 'oneformer3d' / '__init__.py').read_text()
