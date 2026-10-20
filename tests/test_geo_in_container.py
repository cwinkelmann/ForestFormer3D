"""The whole pipeline must be able to run INSIDE the image, with no host venv.

``ff3d_geo`` was written for a split deployment: pure Python on the host, two steps
shelled out to ``forestformer3d:cu118`` through ``benchmark/common.sh``'s ``ff3d_docker``.
Since the image carries the geo stack too (shapely, pyproj, geopandas, rasterio), the same
pipeline can run entirely in one container -- but only if those two steps stop wrapping
themselves in ``docker run``, which inside a container has no docker to call.

So a process that is already inside the image runs the inner commands in place. Detection
is explicit via ``FF3D_IN_CONTAINER`` and otherwise falls back to ``/.dockerenv``, which is
why ``_DOCKERENV`` is a module constant a test can point elsewhere. The host behaviour is
unchanged, and ``tests/test_geo_cli.py`` pins it.
"""
import os
from pathlib import Path

import pytest

pytest.importorskip("laspy")
pytest.importorskip("shapely")
pytest.importorskip("geopandas")

from ff3d_geo.cli import (  # noqa: E402
    DEFAULT_CHECKPOINT,
    DEFAULT_CONFIG,
    _in_image,
    plan_run,
)
import ff3d_geo.cli as cli  # noqa: E402

STEM = "r12_tegel_E381300_N5828300_100m"


@pytest.fixture
def repo(tmp_path):
    # NOT named "workspace": one test asserts that the literal /workspace mount point
    # does not leak into the commands, and a repo path containing it would mask that.
    d = tmp_path / "checkout"
    d.mkdir()
    return d


def _docker_steps(steps):
    return [s for s in steps if s.argv is not None]


def test_in_image_honours_the_explicit_flag(monkeypatch):
    for value, expected in [("1", True), ("yes", True), ("0", False), ("", False)]:
        monkeypatch.setenv("FF3D_IN_CONTAINER", value)
        assert _in_image() is expected, value


def test_in_image_falls_back_to_podmans_marker_too(monkeypatch, tmp_path):
    """podman writes /run/.containerenv instead of docker's /.dockerenv.

    With only docker's marker checked, one container in a podman run would decide it was on
    the host and wrap its GPU steps in a `docker run` that cannot exist there.
    """
    monkeypatch.delenv("FF3D_IN_CONTAINER", raising=False)
    monkeypatch.setattr(cli, "_DOCKERENV", tmp_path / "no-dockerenv")
    marker = tmp_path / "containerenv"
    monkeypatch.setattr(cli, "_CONTAINERENV", marker)
    assert _in_image() is False
    marker.touch()
    assert _in_image() is True


def test_in_image_falls_back_to_the_dockerenv_marker(monkeypatch, tmp_path):
    """Both markers have to be redirected: this test also runs INSIDE the image, where
    podman's /run/.containerenv really exists and would answer True on its own."""
    monkeypatch.delenv("FF3D_IN_CONTAINER", raising=False)
    monkeypatch.setattr(cli, "_CONTAINERENV", tmp_path / "no-containerenv")
    marker = tmp_path / "dockerenv"
    monkeypatch.setattr(cli, "_DOCKERENV", marker)
    assert _in_image() is False
    marker.write_text("")
    assert _in_image() is True


def test_the_two_gpu_steps_run_in_place_inside_the_image(monkeypatch, repo, tmp_path):
    monkeypatch.setenv("FF3D_IN_CONTAINER", "1")
    steps = plan_run(tmp_path / f"{STEM}.las", DEFAULT_CHECKPOINT,
                     repo / "work_dirs/out", repo=repo, gpu="0")

    # The step sequence is the same eight; only HOW the two GPU steps run changes.
    assert [s.name for s in steps] == [
        "las_to_ply", "prepare_inputs", "preprocess", "check_preprocess",
        "inference", "results_to_las", "trees_to_gpkg", "report",
    ]
    pre, inf = _docker_steps(steps)
    assert [s.name for s in (pre, inf)] == ["preprocess", "inference"]
    for step in (pre, inf):
        assert "ff3d_docker" not in step.argv[2], step.argv[2]
        assert "common.sh" not in step.argv[2], step.argv[2]
        assert step.cwd == repo
        # No FF3D_GPU/FF3D_ROOT: those steer ff3d_docker, which is not involved. The
        # container's own CUDA_VISIBLE_DEVICES already pins the card.
        assert step.env == {"PYTHONPATH": str(repo)}
    assert inf.argv[2].startswith("python tools/test.py ")


def test_inside_the_image_paths_are_the_real_ones_not_workspace(monkeypatch, repo, tmp_path):
    """A hardcoded /workspace would be wrong whenever the mount point is not /workspace."""
    monkeypatch.setenv("FF3D_IN_CONTAINER", "1")
    steps = plan_run(tmp_path / f"{STEM}.las", DEFAULT_CHECKPOINT,
                     repo / "work_dirs/out", repo=repo, gpu="0")
    script = _docker_steps(steps)[0].argv[2]
    assert f"{repo}/work_dirs/out/scan_list.txt" in script
    assert "/workspace" not in script


def test_on_the_host_the_two_steps_still_go_through_ff3d_docker(monkeypatch, repo, tmp_path):
    monkeypatch.setenv("FF3D_IN_CONTAINER", "0")
    steps = plan_run(tmp_path / f"{STEM}.las", DEFAULT_CHECKPOINT,
                     repo / "work_dirs/out", repo=repo, gpu="3")
    for step in _docker_steps(steps):
        assert step.argv[2].startswith("source benchmark/common.sh; ff3d_docker ")
        assert step.env["FF3D_GPU"] == "3"
        assert "/workspace/work_dirs/out" in step.argv[2]


def test_a_card_the_container_does_not_have_is_refused(monkeypatch, repo, tmp_path):
    """`--gpu` cannot be honoured inside the container, so a mismatch must not be silent.

    ff3d_docker starts the container for ONE card and renumbers it to device 0 inside
    (measured on olive: --gpus device=1 -> /dev/nvidia0, nvidia-smi index 0,
    NVIDIA_VISIBLE_DEVICES=void). So `ff3d_docker bash benchmark/berlin_run_gpu.sh 5 ...`
    without FF3D_GPU=5 used to run on card 0 while every log line said 5 -- and four
    workers launched that way for cards 4..7 all shared card 0. common.sh now carries
    FF3D_GPU in, which is the only thing here that can be compared against.
    """
    monkeypatch.setenv("FF3D_IN_CONTAINER", "1")
    monkeypatch.setenv("FF3D_GPU", "0")
    with pytest.raises(ValueError) as exc:
        plan_run(tmp_path / f"{STEM}.las", DEFAULT_CHECKPOINT, repo / "work_dirs/out",
                 repo=repo, gpu="5")
    assert "5" in str(exc.value) and "0" in str(exc.value)
    assert "FF3D_GPU" in str(exc.value)


def test_the_card_the_container_does_have_is_accepted(monkeypatch, repo, tmp_path):
    monkeypatch.setenv("FF3D_IN_CONTAINER", "1")
    monkeypatch.setenv("FF3D_GPU", "5")
    steps = plan_run(tmp_path / f"{STEM}.las", DEFAULT_CHECKPOINT, repo / "work_dirs/out",
                     repo=repo, gpu="5")
    assert [s.name for s in _docker_steps(steps)] == ["preprocess", "inference"]


def test_without_ff3d_gpu_there_is_nothing_to_check(monkeypatch, repo, tmp_path):
    """A hand-rolled `docker run` tells the container nothing; that must still work."""
    monkeypatch.setenv("FF3D_IN_CONTAINER", "1")
    monkeypatch.delenv("FF3D_GPU", raising=False)
    steps = plan_run(tmp_path / f"{STEM}.las", DEFAULT_CHECKPOINT, repo / "work_dirs/out",
                     repo=repo, gpu="5")
    assert [s.name for s in _docker_steps(steps)] == ["preprocess", "inference"]
