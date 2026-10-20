"""docker/entrypoint.sh must find the checkout wherever it is mounted.

The entrypoint copies `replace_mmdetection_files/transforms_3d.py` into site-packages at
every start -- the one patch that is not baked into the image -- and looked for it under a
hardcoded /workspace. ff3d_geo stopped assuming that mount point when one-container mode
arrived (`_mount_root`, tests/test_geo_in_container.py), so `-v $PWD:/src -w /src` is now a
supported way to run: there the patch was skipped, with a WARNING on stderr, and inference
went ahead on UNPATCHED mmdet3d transforms. A warning is the wrong instrument for that --
it scrolls past in a log nobody reads until the numbers look odd.

Run against a stub `python`, so no torch, no mmdet3d and no image are needed.
"""
import os
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
ENTRYPOINT = REPO / "docker" / "entrypoint.sh"

STUB_PYTHON = """#!/usr/bin/env bash
# `python -c '...'` is the mmdet3d location probe; anything else (the heredoc check) is a
# no-op success.
if [ "${1-}" = "-c" ]; then printf '%s\\n' "$STUB_MMDET3D"; fi
exit 0
"""


@pytest.fixture
def box(tmp_path):
    """A stub python, a fake mmdet3d tree, and a checkout holding the patch source."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    stub = bin_dir / "python"
    stub.write_text(STUB_PYTHON)
    stub.chmod(0o755)

    mmdet3d = tmp_path / "site-packages" / "mmdet3d"
    (mmdet3d / "datasets" / "transforms").mkdir(parents=True)
    (mmdet3d / "datasets" / "transforms" / "transforms_3d.py").write_text("# stock\n")

    checkout = tmp_path / "src"
    (checkout / "replace_mmdetection_files").mkdir(parents=True)
    (checkout / "replace_mmdetection_files" / "transforms_3d.py").write_text("# patched\n")
    return bin_dir, mmdet3d, checkout


def _run(box, cwd, env_extra=None):
    bin_dir, mmdet3d, _ = box
    env = dict(os.environ)
    env.update({"PATH": f"{bin_dir}:{env['PATH']}", "STUB_MMDET3D": str(mmdet3d)})
    env.pop("WORKSPACE", None)
    env.update(env_extra or {})
    return subprocess.run(["bash", str(ENTRYPOINT), "true"],
                          capture_output=True, text=True, env=env, cwd=cwd)


def test_a_checkout_mounted_anywhere_is_still_patched(box):
    _, mmdet3d, checkout = box
    proc = _run(box, cwd=checkout)
    assert proc.returncode == 0, proc.stderr
    assert "WARNING" not in proc.stderr, proc.stderr
    assert (mmdet3d / "datasets" / "transforms" / "transforms_3d.py").read_text() \
        == "# patched\n"


def test_an_explicit_workspace_wins(box, tmp_path):
    _, mmdet3d, checkout = box
    proc = _run(box, cwd=tmp_path, env_extra={"WORKSPACE": str(checkout)})
    assert proc.returncode == 0, proc.stderr
    assert (mmdet3d / "datasets" / "transforms" / "transforms_3d.py").read_text() \
        == "# patched\n"


def test_no_checkout_anywhere_still_only_warns(box, tmp_path):
    """`docker run <image> nvidia-smi -L` mounts nothing, and must keep working: the
    image carries only the entrypoint, never the patch source."""
    empty = tmp_path / "empty"
    empty.mkdir()
    proc = _run(box, cwd=empty)
    assert proc.returncode == 0, proc.stderr
    assert "WARNING" in proc.stderr, proc.stderr
