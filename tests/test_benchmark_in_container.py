"""The Berlin wrappers must also run INSIDE the image, where there is no host venv.

Both scripts used to activate ``$GEO_VENV`` unconditionally. The unified image carries the
geo stack, so the whole queue can run in one container (``ff3d_docker bash
benchmark/berlin_run_gpu.sh 0 <tile>``) -- but only if the script skips an activation that
would fail there, and tells ff3d_geo (via ``FF3D_IN_CONTAINER``) to run its two GPU steps in
place instead of shelling out to a docker the container does not have.

Run for real against a stub ``python`` on PATH, with ``GEO_VENV`` pointing at a directory
that does not exist: activating it would print an error, which is what the host-mode test
asserts and the in-image test asserts the absence of.
"""
import os
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]

STUB_PYTHON = """#!/usr/bin/env bash
# stub python: log the ff3d_geo subcommand and succeed
printf '%s\\n' "$*" >> "$STUB_LOG"
exit 0
"""


@pytest.fixture
def checkout(tmp_path):
    """A checkout-shaped directory: the two scripts plus the inputs they look for."""
    root = tmp_path / "ForestFormer3D"
    (root / "benchmark").mkdir(parents=True)
    (root / "work_dirs" / "logs").mkdir(parents=True)
    (root / "inputs" / "berlin").mkdir(parents=True)
    # geo_env.sh too: the scripts source it for FF3D_ROOT, the venv default and the
    # in-image check, so a checkout without it is not a checkout.
    for name in ("berlin_run_gpu.sh", "berlin_stitch.sh", "geo_env.sh"):
        (root / "benchmark" / name).write_bytes((REPO / "benchmark" / name).read_bytes())
    (root / "inputs" / "berlin" / "3dm_33_381_5828_1_be.las").write_bytes(b"fake")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    stub = bin_dir / "python"
    stub.write_text(STUB_PYTHON)
    stub.chmod(0o755)
    return root, bin_dir


def _run(script, root, bin_dir, tmp_path, env_extra):
    log = tmp_path / "stub.log"
    env = dict(os.environ)
    env.update({
        "PATH": f"{bin_dir}:{env['PATH']}",
        "STUB_LOG": str(log),
        "GEO_VENV": str(tmp_path / "no-such-venv"),
        **env_extra,
    })
    proc = subprocess.run(["bash", str(root / "benchmark" / script), "0",
                           "3dm_33_381_5828_1_be"],
                          capture_output=True, text=True, env=env, cwd=root)
    calls = log.read_text().splitlines() if log.exists() else []
    return proc, calls


def test_run_gpu_inside_the_image_skips_the_venv_and_still_runs_the_steps(checkout, tmp_path):
    root, bin_dir = checkout
    proc, calls = _run("berlin_run_gpu.sh", root, bin_dir, tmp_path,
                       {"FF3D_IN_CONTAINER": "1"})
    both = proc.stdout + proc.stderr
    assert "no-such-venv" not in both, both
    assert "inside the image" in proc.stdout, proc.stdout
    assert any("ff3d_geo split" in c for c in calls), calls
    assert any("ff3d_geo run" in c for c in calls), calls


def test_stitch_inside_the_image_skips_the_venv(checkout, tmp_path):
    root, bin_dir = checkout
    proc, _ = _run("berlin_stitch.sh", root, bin_dir, tmp_path, {"FF3D_IN_CONTAINER": "1"})
    assert "no-such-venv" not in proc.stdout + proc.stderr
    assert "inside the image" in proc.stdout


def test_an_explicit_zero_keeps_the_host_behaviour(checkout, tmp_path):
    """FF3D_IN_CONTAINER=0 must mean the host, not "the variable is set, so container"."""
    root, bin_dir = checkout
    proc, _ = _run("berlin_run_gpu.sh", root, bin_dir, tmp_path,
                   {"FF3D_IN_CONTAINER": "0"})
    assert "inside the image" not in proc.stdout
    assert "no-such-venv" in proc.stdout + proc.stderr   # it did try to activate it


def test_both_scripts_get_their_detection_from_the_shared_helper():
    """The check lives in benchmark/geo_env.sh, once, for all eight scripts that need it.

    It used to be an inline `case` statement copied into these two, which is how the two
    copies came to disagree with ff3d_geo about mixed case. That the helper tests both
    runtimes' markers is pinned in tests/test_benchmark_geo_env.py; what matters here is
    that these scripts do not decide it for themselves any more.
    """
    for name in ("berlin_run_gpu.sh", "berlin_stitch.sh"):
        text = (REPO / "benchmark" / name).read_text()
        code = [l for l in text.splitlines() if not l.strip().startswith("#")]
        assert any("geo_env.sh" in l for l in code), name
        assert any("ff3d_activate_geo" in l for l in code), name
        assert not any("dockerenv" in l or "containerenv" in l for l in code), \
            f"{name} still decides for itself whether it is in a container"
