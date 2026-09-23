"""Shared pytest setup: put the checkout root on sys.path so tests can import repo packages.

Inside the Docker image PYTHONPATH=/workspace already does this; on the Mac it lets
Phase 1's CPU tests import pure-Python modules by file path.
"""
import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _load_module_without_package_init(module_name: str, relative_path: str,
                                       requires_torch: bool = False,
                                       requires: tuple = ()) -> None:
    """oneformer3d/__init__.py needs spconv+mmdet3d; these submodules do not.

    Loading them by file path (instead of `import oneformer3d.<name>`) bypasses
    the package initialiser so they can be unit-tested without spconv/mmdet3d
    (and, for the torch-only modules, without torch either). `requires` names
    further third-party modules the submodule imports; when one is missing the
    module is simply not loaded and the tests using it skip.
    """
    if module_name in sys.modules:
        return
    if requires_torch and importlib.util.find_spec('torch') is None:
        return
    for dependency in requires:
        if importlib.util.find_spec(dependency) is None:
            return
    spec = importlib.util.spec_from_file_location(module_name, REPO_ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)


_load_module_without_package_init(
    'oneformer3d.tiling', 'oneformer3d/tiling.py', requires_torch=True)
_load_module_without_package_init(
    'oneformer3d.labels', 'oneformer3d/labels.py', requires_torch=False)
_load_module_without_package_init(
    'oneformer3d.ply_io', 'oneformer3d/ply_io.py', requires=('plyfile', 'numpy'))
