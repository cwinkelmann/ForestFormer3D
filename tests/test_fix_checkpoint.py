import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / 'tools' / 'fix_spconv_checkpoint.py'
torch_missing = importlib.util.find_spec('torch') is None
pytestmark = pytest.mark.skipif(torch_missing, reason='torch not installed')

if not torch_missing:
    import torch
    sys.path.insert(0, str(REPO))
    from tools.fix_spconv_checkpoint import checkpoint_layout, convert_state_dict, weight_layout


def raw_state_dict():
    return {
        'input_conv.0.weight': torch.zeros(32, 3, 3, 3, 3),
        'unet.blocks.block0.conv_branch.2.weight': torch.zeros(32, 3, 3, 3, 32),
        'unet.conv.2.weight': torch.zeros(64, 2, 2, 2, 32),
        'unet.u.blocks.block0.i_branch.0.weight': torch.zeros(96, 1, 1, 1, 64),
        'unet.blocks.block0.conv_branch.0.weight': torch.zeros(32),     # BatchNorm, untouched
        'decoder.query_proj.0.weight': torch.zeros(256, 32),
    }


def test_weight_layout_rule():
    assert weight_layout((32, 3, 3, 3, 32)) == 'raw'
    assert weight_layout((3, 3, 3, 32, 32)) == 'converted'
    assert weight_layout((64, 2, 2, 2, 32)) == 'raw'
    assert weight_layout((2, 2, 2, 32, 64)) == 'converted'
    assert weight_layout((3, 3, 3, 3, 32)) == 'ambiguous'     # input_conv, converted


def test_convert_permutes_only_spconv_weights():
    out = convert_state_dict(raw_state_dict())
    assert tuple(out['unet.conv.2.weight'].shape) == (2, 2, 2, 32, 64)
    assert tuple(out['input_conv.0.weight'].shape) == (3, 3, 3, 3, 32)
    assert tuple(out['unet.blocks.block0.conv_branch.0.weight'].shape) == (32,)
    assert tuple(out['decoder.query_proj.0.weight'].shape) == (256, 32)
    assert checkpoint_layout(out) == 'converted'
    assert checkpoint_layout(raw_state_dict()) == 'raw'


def run(in_path, out_path):
    return subprocess.run([sys.executable, str(SCRIPT), '--in-path', str(in_path),
                          '--out-path', str(out_path)], capture_output=True, text=True)


def test_cli_converts_raw_and_refuses_double_conversion(tmp_path):
    raw, fixed, twice = tmp_path / 'raw.pth', tmp_path / 'fixed.pth', tmp_path / 'twice.pth'
    torch.save({'state_dict': raw_state_dict(), 'meta': {'epoch': 3}}, raw)
    first = run(raw, fixed)
    assert first.returncode == 0, first.stderr
    saved = torch.load(fixed)
    assert tuple(saved['state_dict']['unet.conv.2.weight'].shape) == (2, 2, 2, 32, 64)
    assert saved['meta'] == {'epoch': 3}

    second = run(fixed, twice)
    assert second.returncode == 2
    assert 'already converted' in second.stdout
    assert not twice.exists()
