"""The image's own verification layers may only import what the image installs.

The last layer of the `Dockerfile` imports everything the pipeline needs in one
interpreter, which is the right gate: it catches a geo wheel built against numpy 2 at
build time instead of hours later inside a tile. But it is a hand-maintained list beside
another hand-maintained list -- the `pip install` lines -- and nothing kept the two in
step. Dropping `open3d` from the install lines (commit `bc83218`) left it in the check, so
a build would die with `ModuleNotFoundError: No module named 'open3d'` in the LAST layer,
after every CUDA extension had compiled. No test saw it, and no host had rebuilt since.

This is that test. It reads the text, so it needs neither docker nor the 18 GB image and
runs in the CPU suite on a laptop. `Dockerfile.a100-cu116` is deliberately out of scope:
the README keeps it frozen as the legacy CUDA 11.6 image and it is not built any more.
"""
import ast
import re
import sys
from pathlib import Path

DOCKERFILE = Path(__file__).resolve().parents[1] / "Dockerfile"

#: Import name -> distribution name, where pip does not spell them the same.
ALIASES = {
    "yaml": "pyyaml",
    "skimage": "scikit-image",
    "spconv": "spconv-cu118",
    "cumm": "cumm-cu118",
    "torch_points_kernels": "torch-points-kernels",
    "mmdet3d": "mmdetection3d",          # installed from the open-mmlab git URL
    "sklearn": "scikit-learn",
    "PIL": "pillow",
    "cv2": "opencv-python",
}

#: Import name -> the git checkout it is built from, for the extensions pip never names.
FROM_GIT = {
    "MinkowskiEngine": "minkowskiengine",
    "torch_scatter": "pytorch_scatter",
    "torch_cluster": "pytorch_cluster",
    "segmentator": "segmentator",
}

_REQUIREMENT = re.compile(r"^[A-Za-z][A-Za-z0-9._-]*(\[[^\]]+\])?(==[^\s]+)?$")
_GIT_CLONE = re.compile(r"git clone (?:(?:--\S+)(?:\s+\S+)?\s+)*(\S+?\.git)")
_PYTHON_C = re.compile(r'python3? -c "(.*)"')


def _logical_lines(text: str) -> list[str]:
    """The file with backslash continuations joined, so one RUN is one line."""
    return text.replace("\\\n", " ").splitlines()


def _normalise(name: str) -> str:
    name = name.strip().strip('"').strip("'")
    return name.split("[")[0].split("==")[0].lower().replace("_", "-")


def _installed(text: str) -> set[str]:
    """Distribution names this Dockerfile asks for: pip requirements and git checkouts."""
    names: set[str] = set()
    for line in _logical_lines(text):
        for part in line.split("&&"):
            words = part.split()
            if "pip" not in words or "install" not in words:
                continue
            for token in words[words.index("install") + 1:]:
                bare = token.strip('"').strip("'")
                if bare.startswith("git+"):
                    names.add(_normalise(Path(bare.split("@")[0]).stem))
                elif bare.startswith("-") or "/" in bare or bare == ".":
                    continue                      # flag, flag value, URL, local path
                elif _REQUIREMENT.match(bare):
                    names.add(_normalise(bare))
        for url in _GIT_CLONE.findall(line):
            names.add(_normalise(Path(url).stem))
    return names


def _checked_imports(text: str) -> set[str]:
    """Top-level modules every `python -c` verification layer imports."""
    modules: set[str] = set()
    for line in _logical_lines(text):
        found = _PYTHON_C.search(line)
        if not found:
            continue
        code = found.group(1).replace('\\"', '"')
        for node in ast.walk(ast.parse(code)):
            if isinstance(node, ast.Import):
                modules.update(a.name.split(".")[0] for a in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                modules.add(node.module.split(".")[0])
    return modules - set(sys.stdlib_module_names)


def _provided_by(module: str, installed: set[str]) -> bool:
    candidates = {_normalise(module), _normalise(ALIASES.get(module, module))}
    if module in FROM_GIT:
        candidates.add(_normalise(FROM_GIT[module]))
    return bool(candidates & installed)


def test_the_image_checks_only_what_it_installs():
    text = DOCKERFILE.read_text()
    gone = sorted(m for m in _checked_imports(text) if not _provided_by(m, _installed(text)))
    assert gone == [], (
        "the Dockerfile's verification layers import packages it never installs, so the "
        f"build fails there -- after everything else has compiled: {gone}"
    )


def test_the_parser_sees_what_the_image_really_has():
    """Guard the guard: a parser that silently found nothing would pass anything."""
    installed = _installed(DOCKERFILE.read_text())
    for expected in ("numpy", "mmcv", "mmdetection3d", "geopandas", "shapely", "laspy",
                     "torch", "minkowskiengine", "pytorch-scatter", "segmentator"):
        assert expected in installed, (expected, sorted(installed))
    checked = _checked_imports(DOCKERFILE.read_text())
    for expected in ("mmdet3d", "geopandas", "MinkowskiEngine", "torch_points_kernels"):
        assert expected in checked, (expected, sorted(checked))
