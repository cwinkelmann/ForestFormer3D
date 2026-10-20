"""`ff3d_docker` must name the GPU the way the chosen container runtime understands.

Docker and podman disagree about one argument. Docker hands the request to
nvidia-container-runtime and spells it ``--gpus device=N``. Podman reads a CDI spec
(``/etc/cdi/nvidia.yaml``) and spells the same card ``--device nvidia.com/gpu=N``; it does
accept ``--gpus`` for compatibility, but 4.x builds are known to ignore it, and a container
that starts with no GPU does not fail at ``docker run`` -- it fails minutes later inside
torch, which reads like a broken image.

podman matters here because it is rootless without a daemon: on the T14 the root docker
daemon writes every result LAS, GeoPackage and GeoTIFF as root, and a rootless runtime maps
container root to the invoking user instead. So the runtime has to be swappable through
``FF3D_DOCKER`` alone, with no edit to the queue scripts.
"""
import os
import shutil
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
COMMON = REPO / "benchmark" / "common.sh"


def _dry_run_argv(tmp_path, docker: str, gpu: str = "0") -> list[str]:
    """What `ff3d_docker python x.py` would run, as printed by FF3D_DRY_RUN=1."""
    root = tmp_path / "ForestFormer3D"
    (root / "benchmark").mkdir(parents=True, exist_ok=True)
    shutil.copy(COMMON, root / "benchmark" / "common.sh")
    shutil.copy(REPO / "benchmark" / "geo_env.sh", root / "benchmark" / "geo_env.sh")
    env = {k: v for k, v in os.environ.items() if not k.startswith("FF3D_")}
    env.update(FF3D_DRY_RUN="1", FF3D_DOCKER=docker, FF3D_GPU=gpu)
    proc = subprocess.run(
        ["bash", "-c", f'source "{root}/benchmark/common.sh"; ff3d_docker python x.py'],
        capture_output=True, text=True, env=env, cwd=tmp_path)
    assert proc.returncode == 0, proc.stderr
    line = proc.stdout.strip()
    assert line.startswith("DRY:"), proc.stdout
    return line[len("DRY:"):].split()


def test_docker_gets_the_gpus_flag(tmp_path):
    argv = _dry_run_argv(tmp_path, "docker", gpu="3")
    assert argv[:2] == ["docker", "run"]
    assert "--gpus" in argv and "device=3" in argv
    assert not any("nvidia.com/gpu" in a for a in argv)


def test_podman_gets_a_cdi_device(tmp_path):
    argv = _dry_run_argv(tmp_path, "podman", gpu="3")
    assert argv[:2] == ["podman", "run"]
    assert "--device" in argv and "nvidia.com/gpu=3" in argv
    assert "--gpus" not in argv, "podman 4.x may silently ignore --gpus"


def test_the_gpu_is_read_at_call_time_not_at_source_time(tmp_path):
    """The queue scripts `export FF3D_GPU=N` AFTER sourcing common.sh."""
    root = tmp_path / "ForestFormer3D"
    (root / "benchmark").mkdir(parents=True, exist_ok=True)
    shutil.copy(COMMON, root / "benchmark" / "common.sh")
    shutil.copy(REPO / "benchmark" / "geo_env.sh", root / "benchmark" / "geo_env.sh")
    env = {k: v for k, v in os.environ.items() if not k.startswith("FF3D_")}
    env.update(FF3D_DRY_RUN="1", FF3D_DOCKER="podman")
    proc = subprocess.run(
        ["bash", "-c", f'source "{root}/benchmark/common.sh"; '
                       'export FF3D_GPU=2; ff3d_docker python x.py'],
        capture_output=True, text=True, env=env, cwd=tmp_path)
    assert proc.returncode == 0, proc.stderr
    assert "nvidia.com/gpu=2" in proc.stdout, proc.stdout


def test_a_wrapped_runtime_still_resolves(tmp_path):
    """`FF3D_DOCKER="sudo podman"` is still podman; the whole string is matched."""
    argv = _dry_run_argv(tmp_path, "sudo podman")
    assert argv[:3] == ["sudo", "podman", "run"]
    assert "nvidia.com/gpu=0" in argv


def _dry_run_with_path(tmp_path, bin_dir, docker: str, gpu: str = "0") -> list[str]:
    """As _dry_run_argv, but with `bin_dir` first on PATH so a stub runtime is found."""
    root = tmp_path / "ForestFormer3D"
    (root / "benchmark").mkdir(parents=True, exist_ok=True)
    shutil.copy(COMMON, root / "benchmark" / "common.sh")
    shutil.copy(REPO / "benchmark" / "geo_env.sh", root / "benchmark" / "geo_env.sh")
    env = {k: v for k, v in os.environ.items() if not k.startswith("FF3D_")}
    env.update(FF3D_DRY_RUN="1", FF3D_DOCKER=docker, FF3D_GPU=gpu,
               PATH=f"{bin_dir}:{env['PATH']}")
    proc = subprocess.run(
        ["bash", "-c", f'source "{root}/benchmark/common.sh"; ff3d_docker python x.py'],
        capture_output=True, text=True, env=env, cwd=tmp_path)
    assert proc.returncode == 0, proc.stderr
    line = [l for l in proc.stdout.splitlines() if l.startswith("DRY:")][0]
    return line[len("DRY:"):].split()


def test_a_docker_that_is_really_podman_still_gets_a_cdi_device(tmp_path):
    """On a host with the podman-docker shim, /usr/bin/docker IS podman.

    The name then says "docker" while the binary only understands podman's flags, so a
    switch made on the name alone hands it `--gpus`, which podman 4.x may ignore -- the
    container comes up with no GPU and fails minutes later inside torch. Ask the binary.
    """
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    shim = bin_dir / "docker"
    shim.write_text("#!/usr/bin/env bash\necho 'podman version 4.9.3'\n")
    shim.chmod(0o755)
    argv = _dry_run_with_path(tmp_path, bin_dir, "docker", gpu="2")
    assert "nvidia.com/gpu=2" in argv, argv
    assert "--gpus" not in argv, argv


def test_a_real_docker_is_not_mistaken_for_podman(tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    real = bin_dir / "docker"
    real.write_text("#!/usr/bin/env bash\necho 'Docker version 27.1.1, build 6312585'\n")
    real.chmod(0o755)
    argv = _dry_run_with_path(tmp_path, bin_dir, "docker", gpu="2")
    assert "--gpus" in argv and "device=2" in argv, argv
    assert not any("nvidia.com/gpu" in a for a in argv), argv


def test_the_container_is_told_that_it_is_a_container_and_which_card_it_got(tmp_path):
    """One-container mode needs both facts inside, and neither can be derived there.

    `--gpus device=5` renumbers the card to 0 in the container (measured on olive:
    /dev/nvidia0, nvidia-smi index 0, NVIDIA_VISIBLE_DEVICES=void), so a script running
    inside cannot tell which physical GPU it holds, and `ff3d_geo run --gpu 5` would be
    quietly meaningless. Carrying FF3D_GPU in is what lets it be checked instead.
    """
    argv = _dry_run_argv(tmp_path, "docker", gpu="5")
    joined = " ".join(argv)
    assert "-e FF3D_IN_CONTAINER=1" in joined, argv
    assert "-e FF3D_GPU=5" in joined, argv
