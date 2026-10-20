"""One place decides where the geo venv is and whether we are inside the image.

Nine scripts used to re-derive `ff3d-geo-venv` themselves and honour only `$GEO_VENV`,
while `common.sh` exported an `$FF3D_GEO_VENV` that nothing read -- so the variable the
file documented had no effect anywhere. And the in-image guard lived in two of the eight
scripts that need the geo half, so the other six could not run inside the image at all.

`benchmark/geo_env.sh` holds both decisions now. The important test here is the last one:
the shell helper and `ff3d_geo.cli._in_image` must answer identically, because a
disagreement splits a run down the middle -- the wrapper skipping the venv while ff3d_geo
shells out to a docker that is not there, or the reverse.
"""
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
BENCH = REPO / "benchmark"
GEO_ENV = BENCH / "geo_env.sh"

#: Every benchmark script that needs the pure-Python half on the host.
NEEDS_GEO = [
    "ams3d_run_cpu.sh", "ams3d_stitch.sh", "berlin_extend_mosaic.sh",
    "berlin_run_gpu.sh", "berlin_stitch.sh", "potree_rebuild_mosaic.sh",
    "sat_run_gpu.sh", "speed_benchmark.sh",
]

#: The values a human or a wrapper plausibly puts in FF3D_IN_CONTAINER.
FLAG_VALUES = ["1", "0", "", "yes", "no", "true", "false", "True", "False", "NO", "TRUE",
               " 1 ", "2", "maybe", "void"]


@pytest.mark.parametrize("name", NEEDS_GEO)
def test_no_script_derives_the_geo_venv_itself(name):
    offenders = [line.strip() for line in (BENCH / name).read_text().splitlines()
                 if "ff3d-geo-venv" in line and not line.strip().startswith("#")]
    assert offenders == [], f"{name} re-derives the venv instead of sourcing geo_env.sh"


@pytest.mark.parametrize("name", NEEDS_GEO)
def test_every_script_activates_through_the_shared_helper(name):
    text = (BENCH / name).read_text()
    assert "geo_env.sh" in text, f"{name} does not source benchmark/geo_env.sh"
    assert "ff3d_activate_geo" in text, f"{name} does not call ff3d_activate_geo"
    bare = [line.strip() for line in text.splitlines()
            if "bin/activate" in line and not line.strip().startswith("#")]
    assert bare == [], f"{name} activates a venv directly: {bare}"


def test_the_helper_checks_both_runtimes_markers():
    """podman writes /run/.containerenv, not /.dockerenv; one place, both names."""
    code = [line for line in GEO_ENV.read_text().splitlines()
            if "dockerenv" in line and not line.strip().startswith("#")]
    assert len(code) == 1, code
    assert "/.dockerenv" in code[0] and "/run/.containerenv" in code[0], code[0]


def _shell_in_image(value) -> bool:
    env = {k: v for k, v in os.environ.items() if not k.startswith("FF3D_")}
    if value is not None:
        env["FF3D_IN_CONTAINER"] = value
    proc = subprocess.run(
        ["bash", "-c", f'source "{GEO_ENV}"; if ff3d_in_image; then echo Y; else echo N; fi'],
        capture_output=True, text=True, env=env)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() in ("Y", "N"), proc.stdout
    return proc.stdout.strip() == "Y"


def _python_in_image(value) -> bool:
    """Ask ff3d_geo in a subprocess, so no import of it is needed on a bare machine."""
    env = {k: v for k, v in os.environ.items() if not k.startswith("FF3D_")}
    if value is not None:
        env["FF3D_IN_CONTAINER"] = value
    env["PYTHONPATH"] = str(REPO)
    proc = subprocess.run(
        [sys.executable, "-c",
         "from ff3d_geo.cli import _in_image; print('Y' if _in_image() else 'N')"],
        capture_output=True, text=True, env=env, cwd=REPO)
    if proc.returncode != 0:
        pytest.skip(f"ff3d_geo not importable here: {proc.stderr.strip()[-120:]}")
    return proc.stdout.strip() == "Y"


@pytest.mark.parametrize("value", FLAG_VALUES)
def test_the_shell_and_ff3d_geo_read_the_flag_the_same_way(value):
    assert _shell_in_image(value) == _python_in_image(value), value


def test_a_script_honours_the_documented_variable(tmp_path):
    """FF3D_GEO_VENV is the name common.sh documents; it must actually steer a script."""
    env = {k: v for k, v in os.environ.items() if not k.startswith(("FF3D_", "GEO_"))}
    env.update(FF3D_GEO_VENV=str(tmp_path / "chosen-venv"), FF3D_IN_CONTAINER="0")
    proc = subprocess.run(["bash", str(BENCH / "berlin_stitch.sh"), "3dm_33_381_5828_1_be"],
                          capture_output=True, text=True, env=env, cwd=REPO)
    assert "chosen-venv" in proc.stdout + proc.stderr, (proc.stdout, proc.stderr)
    assert proc.returncode != 0, "a missing geo venv must stop the script, not be ignored"


def test_the_legacy_spelling_still_wins_when_it_is_the_one_set(tmp_path):
    env = {k: v for k, v in os.environ.items() if not k.startswith(("FF3D_", "GEO_"))}
    env["GEO_VENV"] = str(tmp_path / "old-spelling")
    proc = subprocess.run(
        ["bash", "-c", f'source "{GEO_ENV}"; echo "$FF3D_GEO_VENV"'],
        capture_output=True, text=True, env=env)
    assert proc.stdout.strip() == str(tmp_path / "old-spelling"), proc.stdout
