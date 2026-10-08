"""Proves the pytest configuration in pyproject.toml is picked up.

Imports nothing from the repo: the Mac has no torch/mmengine, so CPU tests in
Phase 0 must not touch `oneformer3d`.
"""
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _run_pytest(*args: str) -> str:
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", *args],
        cwd=REPO_ROOT, capture_output=True, text=True, check=False)
    return proc.stdout + proc.stderr


def test_gpu_marker_is_registered():
    # `pytest --markers` prints "@pytest.mark.gpu: needs CUDA and the Docker image"
    assert "gpu: needs CUDA and the Docker image" in _run_pytest("--markers")


def test_gpu_tests_are_deselected_by_default():
    # --collect-only never executes tests, so this cannot recurse into itself.
    out = _run_pytest("--collect-only", "-q", "tests/test_scaffold.py")
    assert "1 deselected" in out, out


@pytest.mark.gpu
def test_marker_only_selected_with_m_gpu():
    # Collected but deselected by the default addopts; selected by `pytest -m gpu`.
    assert True
