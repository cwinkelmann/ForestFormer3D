"""fix_spconv_checkpoint.py followed by unfix_spconv_checkpoint.py is the identity.

Needs torch (the checkpoints are torch.save files). On the Mac without torch the
test is skipped; inside the container it runs as a plain CPU test.
"""
import subprocess
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

REPO = Path(__file__).resolve().parents[1]
FIX = REPO / "tools" / "fix_spconv_checkpoint.py"
UNFIX = REPO / "benchmark" / "unfix_spconv_checkpoint.py"


def _raw_state_dict():
    # raw spconv 2.x layout: (out, kD, kH, kW, in); channel counts > 3 like the real model
    # (input_conv in=3 out=32, unet 32..256); plus a non-5D and a non-unet key
    return {
        "unet.blocks.0.conv.weight": torch.arange(8 * 3 * 3 * 3 * 4, dtype=torch.float32).reshape(8, 3, 3, 3, 4),
        "input_conv.0.weight": torch.arange(32 * 3 * 3 * 3 * 3, dtype=torch.float32).reshape(32, 3, 3, 3, 3),
        "unet.blocks.0.i_branch.0.weight": torch.arange(8 * 1 * 1 * 1 * 4, dtype=torch.float32).reshape(8, 1, 1, 1, 4),
        "unet.blocks.0.bn.weight": torch.ones(8),
        "decoder.linear.weight": torch.ones(7, 7),
    }


def test_fix_then_unfix_is_identity(tmp_path):
    raw = tmp_path / "raw.pth"
    conv = tmp_path / "conv.pth"
    back = tmp_path / "back.pth"
    torch.save({"state_dict": _raw_state_dict(), "meta": {"epoch": 1}}, raw)

    r = subprocess.run([sys.executable, str(FIX), "--in-path", str(raw), "--out-path", str(conv)],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    sd = torch.load(conv, map_location="cpu")["state_dict"]
    assert tuple(sd["unet.blocks.0.conv.weight"].shape) == (3, 3, 3, 4, 8)
    assert tuple(sd["input_conv.0.weight"].shape) == (3, 3, 3, 3, 32)

    r = subprocess.run([sys.executable, str(UNFIX), "--in-path", str(conv), "--out-path", str(back)],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    got = torch.load(back, map_location="cpu")
    assert got["meta"] == {"epoch": 1}
    for k, v in _raw_state_dict().items():
        assert torch.equal(got["state_dict"][k], v), k


def test_unfix_refuses_raw_layout(tmp_path):
    raw = tmp_path / "raw.pth"
    out = tmp_path / "out.pth"
    torch.save({"state_dict": _raw_state_dict()}, raw)
    r = subprocess.run([sys.executable, str(UNFIX), "--in-path", str(raw), "--out-path", str(out)],
                       capture_output=True, text=True)
    assert r.returncode == 2
    assert "already raw" in r.stdout + r.stderr
    assert not out.exists()
