"""`oneformer3d` must not pull optional heavy dependencies at import time.

A module-level ``import open3d`` costs ten packages in the image, not one: open3d 0.17
requires ``pyquaternion`` and ``configargparse``, and
``open3d/visualization/draw_plotly.py`` imports ``plotly`` and ``dash`` eagerly, which
drags in flask, werkzeug, itsdangerous and blinker. ``nuscenes-devkit`` -- a declared
dependency of mmdet3d that nothing here imports -- is what pinned ``Shapely<=1.8.5`` and
so blocked the geopandas stack from living in the same environment.

The repo's one open3d call writes a point cloud in ``ScanNetOneFormer3D.predict``, a class
no config builds (CLAUDE.md: the ScanNet/S3DIS classes are inherited and unused), and that
function imports open3d locally anyway. Parsed with ``ast`` rather than imported, so this
runs on a machine with no torch.
"""

import ast
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# Optional at best, and each one expensive: either unused here, or used in one dead
# branch that does its own local import.
HEAVY = {"open3d", "plotly", "dash", "flask", "trimesh", "nuscenes", "lyft_dataset_sdk",
         "pyquaternion"}


def _module_level_imports(path: Path) -> set[str]:
    """Top-level package names imported at module scope (function-local ones are fine)."""
    names: set[str] = set()
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.Import):
            names.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names.add(node.module.split(".")[0])
    return names


def test_the_model_module_imports_nothing_heavy_at_module_level():
    found = sorted(_module_level_imports(REPO / "oneformer3d" / "oneformer3d.py") & HEAVY)
    assert found == [], f"module-level import of {found} in oneformer3d/oneformer3d.py"


def test_no_module_of_the_package_imports_something_heavy_at_module_level():
    offenders = {}
    for path in sorted((REPO / "oneformer3d").glob("*.py")):
        found = sorted(_module_level_imports(path) & HEAVY)
        if found:
            offenders[path.name] = found
    assert offenders == {}
