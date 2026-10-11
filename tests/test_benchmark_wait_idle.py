"""benchmark/wait_idle.sh and benchmark/setup_old_worktree.sh, stdlib-only (subprocess/git).

wait_idle.sh: exercised with fake nvidia-smi and docker scripts via FF3D_NVIDIA_SMI /
FF3D_DOCKER so the tests run the same on the Mac (no real nvidia-smi, no docker daemon) and
on the GPU host. setup_old_worktree.sh: exercised against a throwaway clone of this repo, so
it never touches the real /raid worktree.
"""
import os
import shlex
import shutil
import stat
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
WAIT_IDLE = REPO / "benchmark" / "wait_idle.sh"
SETUP_OLD = REPO / "benchmark" / "setup_old_worktree.sh"
BASH = shutil.which("bash") or "/bin/bash"


def _script(tmp_path, name, body):
    p = tmp_path / name
    p.write_text("#!/usr/bin/env bash\n" + body + "\n")
    p.chmod(p.stat().st_mode | stat.S_IEXEC)
    return p


def _fake_docker_empty(tmp_path):
    # `docker ps --format ...` with no running containers, and any `docker inspect` succeeds
    # trivially (unused when ps lists nothing).
    return _script(tmp_path, "docker-empty", 'if [ "$1" = "ps" ]; then exit 0; fi\nexit 0\n')


def _fake_docker_fail(tmp_path):
    return _script(tmp_path, "docker-fail", 'echo "cannot connect to the docker daemon" >&2\nexit 1\n')


def _fake_docker(tmp_path, name, ps_out=None, ps_rc=0, inspect_out=None, inspect_rc=0):
    # A fake `docker` covering both subcommands wait_idle.sh's check_container_busy() uses:
    # `docker ps --format ...` (one "id|image" row per running container) and
    # `docker inspect -f ... <id>` (prints the comma-joined DeviceIDs wait_idle.sh parses).
    lines = ['case "$1" in', "  ps)"]
    if ps_out is not None:
        lines.append("    printf '%s\\n' " + shlex.quote(ps_out))
    lines.append(f"    exit {ps_rc}")
    lines.append("    ;;")
    lines.append("  inspect)")
    if inspect_out is not None:
        lines.append("    printf '%s\\n' " + shlex.quote(inspect_out))
    lines.append(f"    exit {inspect_rc}")
    lines.append("    ;;")
    lines.append("esac")
    return _script(tmp_path, name, "\n".join(lines))


def _run(env_overrides, extra_args, tmp_path):
    env = dict(os.environ)
    env.update(env_overrides)
    return subprocess.run(
        [BASH, str(WAIT_IDLE), *extra_args],
        capture_output=True, text=True, env=env, cwd=str(tmp_path),
    )


def test_bash_syntax_ok():
    for script in (WAIT_IDLE, SETUP_OLD):
        r = subprocess.run([BASH, "-n", str(script)], capture_output=True, text=True)
        assert r.returncode == 0, script.name + ": " + r.stderr


def test_once_idle_when_no_compute_apps(tmp_path):
    nvidia_smi = _script(tmp_path, "nvidia-smi-idle", "exit 0\n")  # no rows printed
    docker = _fake_docker_empty(tmp_path)
    r = _run({"FF3D_NVIDIA_SMI": str(nvidia_smi), "FF3D_DOCKER": str(docker)}, ["--once"], tmp_path)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "idle" in r.stdout


def test_once_busy_when_compute_app_present(tmp_path):
    nvidia_smi = _script(tmp_path, "nvidia-smi-busy", 'echo "12345"\nexit 0\n')
    docker = _fake_docker_empty(tmp_path)
    r = _run({"FF3D_NVIDIA_SMI": str(nvidia_smi), "FF3D_DOCKER": str(docker)}, ["--once"], tmp_path)
    assert r.returncode == 1, r.stdout + r.stderr
    assert "busy" in r.stdout


def test_once_errors_when_nvidia_smi_fails(tmp_path):
    nvidia_smi = _script(tmp_path, "nvidia-smi-fail", 'echo "NVML: driver mismatch" >&2\nexit 1\n')
    docker = _fake_docker_empty(tmp_path)
    r = _run({"FF3D_NVIDIA_SMI": str(nvidia_smi), "FF3D_DOCKER": str(docker)}, ["--once"], tmp_path)
    assert r.returncode == 2, r.stdout + r.stderr
    assert "ERROR" in r.stderr
    # a failed query must never be reported as idle
    assert "idle" not in r.stdout.lower()


def test_once_errors_when_docker_ps_fails(tmp_path):
    nvidia_smi = _script(tmp_path, "nvidia-smi-idle", "exit 0\n")
    docker = _fake_docker_fail(tmp_path)
    r = _run({"FF3D_NVIDIA_SMI": str(nvidia_smi), "FF3D_DOCKER": str(docker)}, ["--once"], tmp_path)
    assert r.returncode == 2, r.stdout + r.stderr
    assert "ERROR" in r.stderr
    assert "idle" not in r.stdout.lower()


# --- check_container_busy() coverage: a running forestformer3d* container is present, and
# `docker inspect`'s DeviceRequests parsing decides whether it counts against FF3D_GPU. ---

def test_once_busy_when_container_on_same_gpu(tmp_path):
    nvidia_smi = _script(tmp_path, "nvidia-smi-idle", "exit 0\n")
    # DeviceIDs "0," matches FF3D_GPU (default 0) -> the container counts as using our GPU.
    docker = _fake_docker(tmp_path, "docker-samegpu",
                           ps_out="cid1|forestformer3d:cu118", inspect_out="0,")
    r = _run({"FF3D_NVIDIA_SMI": str(nvidia_smi), "FF3D_DOCKER": str(docker), "FF3D_GPU": "0"},
             ["--once"], tmp_path)
    assert r.returncode == 1, r.stdout + r.stderr
    assert "busy" in r.stdout
    assert "container busy=1" in r.stdout


def test_once_idle_when_container_on_different_gpu(tmp_path):
    nvidia_smi = _script(tmp_path, "nvidia-smi-idle", "exit 0\n")
    # DeviceIDs "1," explicitly excludes FF3D_GPU (0) -> the container is ruled out, not busy.
    docker = _fake_docker(tmp_path, "docker-othergpu",
                           ps_out="cid1|forestformer3d:cu118", inspect_out="1,")
    r = _run({"FF3D_NVIDIA_SMI": str(nvidia_smi), "FF3D_DOCKER": str(docker), "FF3D_GPU": "0"},
             ["--once"], tmp_path)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "idle" in r.stdout
    # the idle branch reports "no matching container" (the busy=N phrasing is only in the
    # busy branch's message) -- this confirms the same-GPU container was excluded.
    assert "no matching container" in r.stdout


def test_once_busy_when_inspect_fails(tmp_path):
    nvidia_smi = _script(tmp_path, "nvidia-smi-idle", "exit 0\n")
    # `docker inspect` itself erroring must not be treated as "not using our GPU": an
    # undeterminable container counts as busy (never idle), per check_container_busy()'s
    # documented fail-toward-busy semantics. This is a per-container detection failure, not
    # a `docker ps`/nvidia-smi query failure, so it does NOT abort the whole script (exit 2)
    # -- it only makes this sample busy (exit 1).
    docker = _fake_docker(tmp_path, "docker-inspectfail",
                           ps_out="cid1|forestformer3d:cu118", inspect_rc=1)
    r = _run({"FF3D_NVIDIA_SMI": str(nvidia_smi), "FF3D_DOCKER": str(docker), "FF3D_GPU": "0"},
             ["--once"], tmp_path)
    assert r.returncode == 1, r.stdout + r.stderr
    assert r.returncode != 0
    assert "container busy=1" in r.stdout


def test_once_busy_when_inspect_deviceids_empty(tmp_path):
    nvidia_smi = _script(tmp_path, "nvidia-smi-idle", "exit 0\n")
    # `docker inspect` succeeds but reports no DeviceIDs (e.g. the container was started with
    # `--gpus all`, which sets Count=-1 and an empty DeviceIDs list) -- again undeterminable,
    # so it must count as busy rather than being read as "uses no GPU".
    docker = _fake_docker(tmp_path, "docker-emptyids",
                           ps_out="cid1|forestformer3d:cu118", inspect_out="")
    r = _run({"FF3D_NVIDIA_SMI": str(nvidia_smi), "FF3D_DOCKER": str(docker), "FF3D_GPU": "0"},
             ["--once"], tmp_path)
    assert r.returncode == 1, r.stdout + r.stderr
    assert r.returncode != 0
    assert "container busy=1" in r.stdout


def test_loop_mode_launches_command_once_idle(tmp_path):
    nvidia_smi = _script(tmp_path, "nvidia-smi-idle", "exit 0\n")
    docker = _fake_docker_empty(tmp_path)
    env = {
        "FF3D_NVIDIA_SMI": str(nvidia_smi),
        "FF3D_DOCKER": str(docker),
        "FF3D_IDLE_SAMPLES": "2",
        "FF3D_IDLE_INTERVAL": "1",
    }
    r = _run(env, ["--", "echo", "LAUNCHED"], tmp_path)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "LAUNCHED" in r.stdout
    assert "IDLE -- launching" in r.stdout


def test_usage_error_without_command(tmp_path):
    nvidia_smi = _script(tmp_path, "nvidia-smi-idle", "exit 0\n")
    docker = _fake_docker_empty(tmp_path)
    r = _run({"FF3D_NVIDIA_SMI": str(nvidia_smi), "FF3D_DOCKER": str(docker)}, [], tmp_path)
    assert r.returncode == 2
    assert "usage" in r.stderr.lower()


# --- setup_old_worktree.sh -------------------------------------------------------------

def _have_git():
    return shutil.which("git") is not None


@pytest.mark.skipif(not _have_git(), reason="git not available")
def test_setup_old_worktree_idempotent_and_copies_entrypoint(tmp_path):
    clone = tmp_path / "clone"
    old = tmp_path / "clone-old"
    r = subprocess.run(["git", "clone", "-q", str(REPO), str(clone)], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr

    env = dict(os.environ)
    env["FF3D_ROOT"] = str(clone)
    env["FF3D_OLD_ROOT"] = str(old)

    try:
        r1 = subprocess.run([BASH, str(SETUP_OLD)], capture_output=True, text=True, env=env)
        assert r1.returncode == 0, r1.stdout + r1.stderr
        assert (old / ".git").is_file()
        head = subprocess.run(
            ["git", "-C", str(old), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True,
        ).stdout.strip()
        assert head == "6a75c37"

        # docker/entrypoint.sh must have been materialized into the old worktree, and be
        # executable, so ff3d_docker_old's container ENTRYPOINT resolves (the old worktree
        # is bind-mounted over /workspace and has no docker/ dir at 6a75c37).
        old_entrypoint = old / "docker" / "entrypoint.sh"
        assert old_entrypoint.is_file()
        assert os.access(old_entrypoint, os.X_OK)

        # tools/test.py at 6a75c37 calls torch.load()/torch.save() but never imports torch
        # (a real bug found on the GPU host: NameError at run time). The throwaway clone at
        # 6a75c37 lacks the import before setup runs; the script must insert it, once, right
        # after 'import argparse', without disturbing the rest of the file.
        old_test_py = old / "tools" / "test.py"
        test_py_text = old_test_py.read_text()
        lines = test_py_text.splitlines()
        assert lines.count("import torch") == 1
        assert lines[lines.index("import argparse") + 1] == "import torch"

        # second run: no-op / idempotent, and the entrypoint copy + import are still there.
        r2 = subprocess.run([BASH, str(SETUP_OLD)], capture_output=True, text=True, env=env)
        assert r2.returncode == 0, r2.stdout + r2.stderr
        assert "worktree exists" in (r2.stdout + r2.stderr)
        assert "already has 'import torch'" in (r2.stdout + r2.stderr)
        assert old_entrypoint.is_file()
        assert os.access(old_entrypoint, os.X_OK)
        assert old_test_py.read_text().splitlines().count("import torch") == 1
        assert old_test_py.read_text() == test_py_text  # byte-for-byte unchanged, not re-touched

        wt_list = subprocess.run(
            ["git", "-C", str(clone), "worktree", "list"], capture_output=True, text=True
        ).stdout
        assert str(old) in wt_list
        assert "6a75c37" in wt_list
    finally:
        subprocess.run(
            ["git", "-C", str(clone), "worktree", "remove", str(old), "--force"],
            capture_output=True, text=True,
        )


@pytest.mark.skipif(not _have_git(), reason="git not available")
def test_setup_old_worktree_refuses_non_worktree_dir(tmp_path):
    clone = tmp_path / "clone2"
    bogus = tmp_path / "clone2-old"
    subprocess.run(["git", "clone", "-q", str(REPO), str(clone)], capture_output=True, text=True)
    bogus.mkdir()
    (bogus / "somefile").write_text("not a worktree")

    env = dict(os.environ)
    env["FF3D_ROOT"] = str(clone)
    env["FF3D_OLD_ROOT"] = str(bogus)
    r = subprocess.run([BASH, str(SETUP_OLD)], capture_output=True, text=True, env=env)
    assert r.returncode == 1
    assert "not a git worktree" in (r.stdout + r.stderr)
