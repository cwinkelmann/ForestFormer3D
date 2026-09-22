"""benchmark/run_train_200.sh, stdlib-only (subprocess).

Exercised entirely through FF3D_DRY_RUN=1 (common.sh's ff3d_run prints the docker command
line instead of running it) plus FF3D_FOREGROUND=1 (skips the nohup self-daemonize). No
docker, no GPU, no torch: this file only ever inspects the shell commands the script would
have run.

Because a dry run never actually produces epoch_200.pth / *.ply / evaluation_total_test.txt,
each real (non-dry) file-existence check in the script would abort a bare single dry run
partway through. So each test seeds just enough stub files to get the run past the checks
that matter for that stage, and asserts on everything printed before the run stops.
"""
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "benchmark" / "run_train_200.sh"
BASH = "bash"


def _run(env_overrides, args, cwd=None):
    import os
    env = dict(os.environ)
    env.update(env_overrides)
    return subprocess.run(
        [BASH, str(SCRIPT), *args],
        capture_output=True, text=True, env=env, cwd=str(cwd) if cwd else str(REPO),
    )


def _seed_common(root: Path):
    """Data layout common.sh's ff3d_preprocess/ff3d_prepare_checkpoint expect."""
    meta = root / "data" / "ForAINetV2" / "meta_data"
    meta.mkdir(parents=True, exist_ok=True)
    (meta / "test_list.txt").write_text("a_test\n")
    # preprocessing marker present -> ff3d_preprocess is a no-op
    (root / "data" / "ForAINetV2" / "forainetv2_oneformer3d_infos_test.pkl").write_text("")


def _base_env(root: Path, variant: str, old_root: Path = None):
    env = {
        "FF3D_ROOT": str(root),
        "FF3D_DRY_RUN": "1",
        "FF3D_FOREGROUND": "1",
    }
    if variant == "old":
        assert old_root is not None
        env["FF3D_OLD_ROOT"] = str(old_root)
    return env


def _seed_old_worktree(old_root: Path):
    (old_root / "tools").mkdir(parents=True, exist_ok=True)
    (old_root / "tools" / "test.py").write_text("# stub\n")


def test_bash_syntax_ok():
    r = subprocess.run([BASH, "-n", str(SCRIPT)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_usage_error_without_variant():
    r = _run({}, [])
    assert r.returncode == 2, r.stdout + r.stderr
    assert "usage: " in r.stdout + r.stderr
    assert "run_train_200.sh <old|fixed>" in r.stdout + r.stderr


def test_usage_error_with_bad_variant():
    r = _run({}, ["both"])
    assert r.returncode == 2, r.stdout + r.stderr
    assert "usage:" in r.stdout + r.stderr


def test_old_variant_dies_without_old_worktree(tmp_path):
    _seed_common(tmp_path)
    r = _run({"FF3D_ROOT": str(tmp_path), "FF3D_FOREGROUND": "1"}, ["old"])
    assert r.returncode == 1, r.stdout + r.stderr
    assert "ERROR: missing old worktree (run benchmark/setup_old_worktree.sh)" in r.stdout + r.stderr


@pytest.mark.parametrize("variant,ckpt_suffix", [("fixed", "converted"), ("old", "raw")])
def test_dry_run_train_and_test_commands(tmp_path, variant, ckpt_suffix):
    """Fresh run: preprocess-skip, train.py (correct cfg-options + work-dir), checkpoint
    prep, test.py with the checkpoint file matching the variant. Dies at the ply-count
    check (dry test.py never produces .ply files) -- that's expected and is itself proof
    the check ran with N_TEST=1.
    """
    root = tmp_path / "root"
    root.mkdir()
    _seed_common(root)
    work = root / "work_dirs" / f"bench-{variant}-200"
    work.mkdir(parents=True)
    (work / "epoch_200.pth").write_text("")  # stub so the post-train existence check passes

    old_root = tmp_path / "old" if variant == "old" else None
    if old_root:
        _seed_old_worktree(old_root)

    r = _run(_base_env(root, variant, old_root), [variant], cwd=root)
    out = r.stdout + r.stderr

    assert r.returncode == 1, out
    assert "preprocessing present, skipping" in out

    # train.py: correct config, work-dir and the three cfg-options
    assert "python tools/train.py configs/oneformer3d_qs_radius16_qp300_2many.py" in out
    assert f"--work-dir work_dirs/bench-{variant}-200" in out
    assert "--cfg-options train_cfg.max_epochs=200 train_cfg.val_interval=20 default_hooks.checkpoint.max_keep_ckpts=2" in out

    # checkpoint prep on epoch_200.pth
    assert f"tools/fix_spconv_checkpoint.py --in-path work_dirs/bench-{variant}-200/epoch_200.pth" in out
    assert f"--out-path work_dirs/bench-{variant}-200/epoch_200_converted.pth" in out
    assert "epoch_200.pth layout: raw" in out

    # test.py uses the correct checkpoint file for this variant
    assert (f"python tools/test.py configs/oneformer3d_qs_radius16_qp300_2many.py "
            f"work_dirs/bench-{variant}-200/epoch_200_{ckpt_suffix}.pth "
            f"--work-dir work_dirs/bench-{variant}-200/test") in out

    assert f"ERROR: {variant} produced 0 ply files, expected 1 in work_dirs/bench-{variant}-200/test" in out

    if variant == "old":
        # the old worktree wrapper: old_prelude.sh + the old worktree mounted at /workspace,
        # for BOTH the train.py and test.py invocations
        assert out.count(f"-v {old_root}:/workspace") == 2
        assert out.count("bash /old_prelude.sh") == 2
    else:
        assert f"-v {root}:/workspace" in out
        assert "old_prelude.sh" not in out

    assert (work / ".done-train").exists()


@pytest.mark.parametrize("variant,ckpt_suffix", [("fixed", "converted"), ("old", "raw")])
def test_dry_run_final_eval_command(tmp_path, variant, ckpt_suffix):
    """Train/checkpoint/test stages already marked done -> the run should skip straight to
    printing the final_eval.py command, then die on the missing F1 line (dry run never
    writes evaluation_total_test.txt).
    """
    root = tmp_path / "root"
    root.mkdir()
    _seed_common(root)
    work = root / "work_dirs" / f"bench-{variant}-200"
    test_out = work / "test"
    test_out.mkdir(parents=True)
    (work / "epoch_200.pth").write_text("")
    (work / "epoch_200_converted.pth").write_text("")
    (work / "epoch_200_raw.pth").write_text("")
    (work / "epoch_200.layout").write_text("raw\n")
    (work / ".done-train").write_text("")
    (test_out / ".done-test").write_text("")

    old_root = tmp_path / "old" if variant == "old" else None
    if old_root:
        _seed_old_worktree(old_root)

    r = _run(_base_env(root, variant, old_root), [variant], cwd=root)
    out = r.stdout + r.stderr

    assert r.returncode == 1, out
    assert "training already done" in out
    assert "test.py already done" in out
    assert f"python tools/final_eval.py work_dirs/bench-{variant}-200/test" in out
    # final_eval always runs through the FIXED image/checkout, even for the old variant
    assert f"-v {root}:/workspace" in out
    assert f"ERROR: no F1 line in work_dirs/bench-{variant}-200/test/evaluation_total_test.txt" in out
