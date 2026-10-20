"""The benchmark scripts must run on any host, not only on carrot.

carrot's layout was the only one that existed when these scripts were written, so its
absolute paths became their defaults: `FF3D_ROOT=/raid/cwinkelmann/ForestFormer3D` and
`source /raid/cwinkelmann/ff3d-geo-venv/bin/activate`. On a second GPU worker (the T14,
olive) that fails far from the cause -- `ff3d_docker` bind-mounts the missing directory
and the container dies with `stat /workspace/docker/entrypoint.sh: no such file or
directory`, which reads like a broken image rather than a wrong mount.

Both paths can be derived instead, and derive to the SAME values on carrot: the repo root
is the parent of the directory holding the script, and on every host so far the geo venv is
`ff3d-geo-venv` beside the checkout (carrot `/raid/cwinkelmann/`, the T14 and olive
`/home/christian/`). Comments may still name carrot's paths as examples; executable lines
may not.
"""
import os
import shutil
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
BENCH = REPO / "benchmark"
CARROT = "/raid/cwinkelmann"


def _code_lines(path: Path):
    """(lineno, text) for executable lines -- comments and blanks dropped."""
    out = []
    for i, line in enumerate(path.read_text().splitlines(), 1):
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            out.append((i, line))
    return out


def test_no_benchmark_script_hardcodes_carrots_paths_in_code():
    offenders = [
        f"{path.name}:{lineno}: {text.strip()}"
        for path in sorted(BENCH.glob("*.sh"))
        for lineno, text in _code_lines(path)
        if CARROT in text
    ]
    assert offenders == [], "carrot-only paths in executable lines:\n" + "\n".join(offenders)


def _source_common(tmp_path) -> dict:
    """Copy common.sh into a fresh checkout-shaped dir, source it, read the paths back."""
    root = tmp_path / "ForestFormer3D"
    (root / "benchmark").mkdir(parents=True)
    shutil.copy(BENCH / "common.sh", root / "benchmark" / "common.sh")
    env = {k: v for k, v in os.environ.items() if not k.startswith("FF3D_")}
    proc = subprocess.run(
        ["bash", "-c",
         f'source "{root}/benchmark/common.sh"; '
         'echo "FF3D_ROOT=$FF3D_ROOT"; echo "FF3D_GEO_VENV=$FF3D_GEO_VENV"'],
        capture_output=True, text=True, env=env, cwd=tmp_path)
    assert proc.returncode == 0, proc.stderr
    return dict(line.split("=", 1) for line in proc.stdout.strip().splitlines())


def test_common_sh_derives_the_repo_root_from_its_own_location(tmp_path):
    assert _source_common(tmp_path)["FF3D_ROOT"] == str(tmp_path / "ForestFormer3D")


def test_common_sh_defaults_the_geo_venv_beside_the_checkout(tmp_path):
    assert _source_common(tmp_path)["FF3D_GEO_VENV"] == str(tmp_path / "ff3d-geo-venv")
