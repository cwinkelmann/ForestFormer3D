"""GPU-only tests: run with `pytest -m gpu tests/gpu` inside the Docker image (docker/smoke.sh).

Pytest imports every test module before applying `-m`, so on a machine without torch
(the Mac) the modules in this directory are not collected at all.
"""
import importlib.util

import pytest

collect_ignore_glob = [] if importlib.util.find_spec("torch") else ["test_*.py"]


@pytest.fixture(scope="session", autouse=True)
def require_cuda():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.fail("tests/gpu needs a CUDA device; run them through docker/smoke.sh")
