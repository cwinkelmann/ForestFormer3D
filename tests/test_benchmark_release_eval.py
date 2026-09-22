"""Tests for benchmark/run_release_eval.sh.

FF3D_DRY_RUN=1 FF3D_FOREGROUND=1 exercises the whole stage sequence without invoking docker
or python: common.sh's ff3d_run prints "DRY: <cmd...>" instead of executing it. The fixture
pre-creates the preprocessing marker pkl (forainetv2_oneformer3d_infos_test.pkl) so
ff3d_preprocess takes its already-done branch under dry-run -- its own postcondition check
(the real create_data call would produce that pkl, but under dry-run nothing really runs, so
asserting on it would always fail; pre-seeding it is the documented workaround, see the task
report). ff3d_prepare_checkpoint's docker call already resolves deterministically under
dry-run (ff3d_run always returns 0, so it takes the "raw" branch) without needing torch/python.

No docker, no python subprocess side effects -- stdlib only.
"""
import os
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "benchmark" / "run_release_eval.sh"


def _make_root(tmp_path):
    root = tmp_path / "root"
    meta = root / "data" / "ForAINetV2" / "meta_data"
    meta.mkdir(parents=True)
    (meta / "test_list.txt").write_text("a_test\nb_test\n")
    # pre-seed the preprocessing marker so ff3d_preprocess's already-done branch is taken
    # under dry-run (see module docstring)
    (root / "data" / "ForAINetV2" / "forainetv2_oneformer3d_infos_test.pkl").write_bytes(b"x")
    ckpt_dir = root / "work_dirs" / "clean_forestformer"
    ckpt_dir.mkdir(parents=True)
    (ckpt_dir / "epoch_3000_fix.pth").write_bytes(b"fake-checkpoint")
    return root


def _make_old(tmp_path):
    old = tmp_path / "old"
    (old / "tools").mkdir(parents=True)
    (old / "tools" / "test.py").write_text("# fake old tools/test.py\n")
    (old / "docker").mkdir(parents=True)
    entry = old / "docker" / "entrypoint.sh"
    entry.write_text("#!/bin/sh\nexec \"$@\"\n")
    entry.chmod(0o755)
    return old


def _run(root, old, env_extra=None):
    env = dict(
        os.environ,
        FF3D_ROOT=str(root),
        FF3D_OLD_ROOT=str(old),
        FF3D_DRY_RUN="1",
        FF3D_FOREGROUND="1",
    )
    if env_extra:
        env.update(env_extra)
    # merge stderr into stdout so the two streams interleave in real chronological order
    # (ff3d_prepare_checkpoint's docker call and its own ff3d_log lines are written to
    # stderr with `>&2`, while the rest of the script logs to stdout)
    return subprocess.run(
        ["bash", str(SCRIPT)], env=env, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, text=True
    )


def test_bash_syntax():
    r = subprocess.run(["bash", "-n", str(SCRIPT)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_dry_run_prints_full_stage_sequence(tmp_path):
    root = _make_root(tmp_path)
    old = _make_old(tmp_path)
    r = _run(root, old)
    out = r.stdout
    assert r.returncode == 0, out

    # 1. preprocessing marker already present -> skip branch, no docker call
    assert "preprocessing present, skipping" in out

    # 2. ff3d_prepare_checkpoint: one DRY docker call to fix_spconv_checkpoint.py
    assert "DRY:" in out
    assert "tools/fix_spconv_checkpoint.py" in out
    assert "--in-path work_dirs/clean_forestformer/epoch_3000_fix.pth" in out
    assert "--out-path work_dirs/clean_forestformer/epoch_3000_converted.pth" in out

    # 3. fixed tools/test.py on the CONVERTED checkpoint, plain ff3d_docker (no old_prelude)
    fixed_test_line = [
        l for l in out.splitlines()
        if "tools/test.py" in l and "epoch_3000_converted.pth" in l
    ]
    assert len(fixed_test_line) == 1, out
    assert "--work-dir work_dirs/bench-release-fixed" in fixed_test_line[0]
    assert "old_prelude.sh" not in fixed_test_line[0]

    # 4. old tools/test.py on the RAW checkpoint, via ff3d_docker_old + old_prelude.sh,
    #    with the old worktree mounted at /workspace plus data/ and work_dirs/ from root
    old_test_line = [
        l for l in out.splitlines()
        if "tools/test.py" in l and "epoch_3000_raw.pth" in l
    ]
    assert len(old_test_line) == 1, out
    assert "--work-dir work_dirs/bench-release-old" in old_test_line[0]
    assert "old_prelude.sh" in old_test_line[0]
    assert f"-v {old}:/workspace" in old_test_line[0]
    assert f"-v {root}/data:/workspace/data" in old_test_line[0]
    assert f"-v {root}/work_dirs:/workspace/work_dirs" in old_test_line[0]

    # 5. two final_eval.py calls, one per output dir
    eval_lines = [l for l in out.splitlines() if "tools/final_eval.py" in l]
    assert len(eval_lines) == 2, out
    assert any(l.endswith("work_dirs/bench-release-fixed") for l in eval_lines)
    assert any(l.endswith("work_dirs/bench-release-old") for l in eval_lines)

    # ordering: preprocess-skip, prepare-checkpoint, fixed test, old test, then the two evals
    idx_prep = out.index("preprocessing present, skipping")
    idx_ckpt = out.index("fix_spconv_checkpoint.py")
    idx_fixed = out.index(fixed_test_line[0])
    idx_old = out.index(old_test_line[0])
    idx_eval1 = out.index(eval_lines[0])
    assert idx_prep < idx_ckpt < idx_fixed < idx_old < idx_eval1

    assert "release eval finished" in out


def test_missing_release_checkpoint_fails_before_anything_else(tmp_path):
    root = tmp_path / "root"
    meta = root / "data" / "ForAINetV2" / "meta_data"
    meta.mkdir(parents=True)
    (meta / "test_list.txt").write_text("a_test\nb_test\n")
    old = _make_old(tmp_path)
    r = _run(root, old)
    assert r.returncode == 1
    assert "missing" in (r.stdout)
    assert "epoch_3000_fix.pth" in (r.stdout)
    assert "DRY:" not in (r.stdout)


def test_missing_old_worktree_fails(tmp_path):
    root = _make_root(tmp_path)
    old = tmp_path / "no-such-old-worktree"
    r = _run(root, old)
    assert r.returncode == 1
    assert "missing old worktree" in (r.stdout)


def test_done_fixed_marker_skips_fixed_stage_only(tmp_path):
    root = _make_root(tmp_path)
    old = _make_old(tmp_path)
    bench_dir = root / "work_dirs" / "bench-release"
    bench_dir.mkdir(parents=True)
    (bench_dir / ".done-fixed").write_bytes(b"")

    r = _run(root, old)
    out = r.stdout
    assert r.returncode == 0, out

    assert "fixed test.py already done" in out
    # the fixed test.py docker command itself must NOT be printed
    assert not any(
        "tools/test.py" in l and "epoch_3000_converted.pth" in l for l in out.splitlines()
    )
    # the old stage and both eval stages still run normally
    assert any(
        "tools/test.py" in l and "epoch_3000_raw.pth" in l for l in out.splitlines()
    )
    eval_lines = [l for l in out.splitlines() if "tools/final_eval.py" in l]
    assert len(eval_lines) == 2, out


def test_foreground_default_daemonizes(tmp_path):
    """Without FF3D_FOREGROUND=1, the script forks under nohup and returns immediately,
    printing the log path (mirrors the manual check in the task brief)."""
    root = _make_root(tmp_path)
    old = _make_old(tmp_path)
    env = dict(
        os.environ,
        FF3D_ROOT=str(root),
        FF3D_OLD_ROOT=str(old),
        FF3D_DRY_RUN="1",
    )
    r = subprocess.run(["bash", str(SCRIPT)], env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout
    assert "started release-eval pid" in r.stdout
    logs = list((root / "work_dirs" / "logs").glob("release-eval-*.log"))
    assert len(logs) == 1
