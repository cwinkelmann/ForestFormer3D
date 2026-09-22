"""benchmark/run_train_200.sh, stdlib-only (subprocess).

Exercised entirely through FF3D_DRY_RUN=1 (common.sh's ff3d_run prints the docker/mkdir/
touch/rm command line instead of running it) plus FF3D_FOREGROUND=1 (skips the nohup
self-daemonize). No docker, no GPU, no torch: this file only ever inspects the shell
commands the script would have run.

Contract (common.sh commits 6738a5f, b7000f8, 5c723ec; run_train_200.sh "postconditions are
dry-run aware" and "dry run creates no files"): under FF3D_DRY_RUN=1 NOTHING is written --
not epoch_200.pth, not the .ply files, not evaluation_total_test.txt, and not even the
.done-train/.done-test/.done-eval marker files or the work_dirs/... directories themselves
(mkdir/touch/rm all go through ff3d_run). Every file-existence postcondition check on an
artefact the script's own preceding command was supposed to just produce is skipped with a
"DRY: (postcondition skipped) <path>" line instead of ff3d_die. That means a single fresh
dry run -- with NOTHING pre-seeded beyond the data/preprocessing precondition -- runs the
complete sequence through final_eval and exits 0, while leaving the temp FF3D_ROOT
byte-for-byte as it started (verified with a directory snapshot before/after).

Preconditions on things the user must actually supply (data present, old worktree present)
are untouched by any of this and still fail for real -- covered separately below.

common.sh's ff3d_prepare_checkpoint has its own matching dry-run branch: it can't determine
a real layout without running the container, so it never prints "raw"/"converted" under a
dry run -- run_train_200.sh's `layout` variable becomes that explanation text instead,
which routes to its (harmless, informational) WARNING log line every dry run.
"""
import os
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "benchmark" / "run_train_200.sh"
BASH = "bash"


def _run(env_overrides, args, cwd=None):
    env = dict(os.environ)
    env.update(env_overrides)
    return subprocess.run(
        [BASH, str(SCRIPT), *args],
        capture_output=True, text=True, env=env, cwd=str(cwd) if cwd else str(REPO),
    )


def _seed_common(root: Path):
    """Data layout common.sh's ff3d_preprocess/ff3d_prepare_checkpoint expect. This is
    legitimate PRE-EXISTING input (the Zenodo data + its one-time preprocessing), not a
    post-training artefact, so it stays seeded in every test.
    """
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


def _snapshot(root: Path):
    """All paths (files and dirs) under root, relative -- for before/after dry-run diffs."""
    return {p.relative_to(root) for p in root.rglob("*")}


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
    # precondition on user input (the old worktree must exist) -- NOT dry-run-gated, still
    # a real check even under FF3D_DRY_RUN=1.
    _seed_common(tmp_path)
    r = _run({"FF3D_ROOT": str(tmp_path), "FF3D_FOREGROUND": "1"}, ["old"])
    assert r.returncode == 1, r.stdout + r.stderr
    assert "ERROR: missing old worktree (run benchmark/setup_old_worktree.sh)" in r.stdout + r.stderr


@pytest.mark.parametrize("variant,ckpt_suffix", [("fixed", "converted"), ("old", "raw")])
def test_dry_run_full_sequence_creates_no_files(tmp_path, variant, ckpt_suffix):
    """Fresh run, NOTHING pre-seeded beyond the data/preprocessing precondition: the dry
    postconditions let the script run the complete sequence -- preprocess-skip, train.py,
    checkpoint prep, test.py, final_eval.py -- and exit 0, while writing nothing at all
    under FF3D_ROOT (mkdir/touch/rm are all ff3d_run-gated too).
    """
    root = tmp_path / "root"
    root.mkdir()
    _seed_common(root)
    work_rel = f"work_dirs/bench-{variant}-200"

    old_root = tmp_path / "old" if variant == "old" else None
    if old_root:
        _seed_old_worktree(old_root)

    before = _snapshot(root)
    r = _run(_base_env(root, variant, old_root), [variant], cwd=root)
    after = _snapshot(root)
    out = r.stdout + r.stderr

    assert r.returncode == 0, out
    assert before == after, f"dry run wrote/removed paths under {root}: {after - before} / removed: {before - after}"

    assert "preprocessing present, skipping" in out

    # 1. train.py: correct config, work-dir and the three cfg-options; work-dir mkdir and
    # the epoch_200.pth postcondition are both printed, not executed.
    assert f"DRY: mkdir -p {root}/{work_rel}" in out
    assert "python tools/train.py configs/oneformer3d_qs_radius16_qp300_2many.py" in out
    assert f"--work-dir {work_rel}" in out
    assert "--cfg-options train_cfg.max_epochs=200 train_cfg.val_interval=20 default_hooks.checkpoint.max_keep_ckpts=2" in out
    assert f"DRY: (postcondition skipped) {root}/{work_rel}/epoch_200.pth" in out
    assert f"DRY: touch {root}/{work_rel}/.done-train" in out
    assert "training done" in out

    # 2. checkpoint prep on epoch_200.pth: common.sh's own dry-run branch (never determines
    # a real layout), which routes to the WARNING log line here.
    assert f"DRY: (input not present yet) {work_rel}/epoch_200.pth" in out
    assert f"tools/fix_spconv_checkpoint.py --in-path {work_rel}/epoch_200.pth" in out
    assert f"--out-path {work_rel}/epoch_200_converted.pth" in out
    assert "WARNING: freshly trained checkpoint reported as" in out

    # 3. test.py uses the correct checkpoint file for this variant. No files match the glob
    # yet, so bash leaves it unexpanded and ff3d_run's `printf ' %q'` shell-quotes the literal
    # "*" (-> "\*.ply") when it prints the command.
    assert f"DRY: rm -f {root}/{work_rel}/test/\\*.ply" in out
    assert (f"python tools/test.py configs/oneformer3d_qs_radius16_qp300_2many.py "
            f"{work_rel}/epoch_200_{ckpt_suffix}.pth "
            f"--work-dir {work_rel}/test") in out
    assert f"DRY: (postcondition skipped) {root}/{work_rel}/test/*.ply (expected 1)" in out
    assert f"DRY: touch {root}/{work_rel}/test/.done-test" in out

    # 4. final_eval.py, always through the FIXED image/checkout even for the old variant
    assert f"python tools/final_eval.py {work_rel}/test" in out
    assert f"DRY: (postcondition skipped) {root}/{work_rel}/test/evaluation_total_test.txt" in out
    assert f"DRY: touch {root}/{work_rel}/test/.done-eval" in out

    assert f"train-{variant}-200 finished" in out

    if variant == "old":
        # the old worktree wrapper: old_prelude.sh + the old worktree mounted at /workspace,
        # for BOTH the train.py and test.py invocations, but NOT for final_eval.py
        assert out.count(f"-v {old_root}:/workspace") == 2
        assert out.count("bash /old_prelude.sh") == 2
        assert f"-v {root}:/workspace" in out  # the final_eval.py call, on the main checkout
    else:
        assert f"-v {root}:/workspace" in out
        assert "old_prelude.sh" not in out


@pytest.mark.parametrize("variant,ckpt_suffix", [("fixed", "converted"), ("old", "raw")])
def test_dry_run_skips_completed_stages_and_creates_no_files(tmp_path, variant, ckpt_suffix):
    """Train/checkpoint/test stages already marked done from a prior REAL run (files placed
    directly by the test, not by this dry run) -> re-running skips straight to final_eval.py
    without re-invoking train.py/test.py, and still writes nothing new (the final .done-eval
    touch is ff3d_run-gated too).
    """
    root = tmp_path / "root"
    root.mkdir()
    _seed_common(root)
    work = root / "work_dirs" / f"bench-{variant}-200"
    test_out = work / "test"
    test_out.mkdir(parents=True)
    (work / "epoch_200.pth").write_text("")
    (work / f"epoch_200_{ckpt_suffix}.pth").write_text("")
    (work / "epoch_200_converted.pth").write_text("")
    (work / "epoch_200_raw.pth").write_text("")
    (work / "epoch_200.layout").write_text("raw\n")
    (work / ".done-train").write_text("")
    (test_out / ".done-test").write_text("")

    old_root = tmp_path / "old" if variant == "old" else None
    if old_root:
        _seed_old_worktree(old_root)

    before = _snapshot(root)
    r = _run(_base_env(root, variant, old_root), [variant], cwd=root)
    after = _snapshot(root)
    out = r.stdout + r.stderr

    assert r.returncode == 0, out
    assert before == after, f"dry run wrote/removed paths under {root}: {after - before} / removed: {before - after}"

    assert "training already done" in out
    assert "test.py already done" in out
    assert "python tools/train.py" not in out
    assert "python tools/test.py" not in out
    assert f"python tools/final_eval.py work_dirs/bench-{variant}-200/test" in out
    assert f"DRY: touch {test_out}/.done-eval" in out
    assert not (test_out / ".done-eval").exists()
