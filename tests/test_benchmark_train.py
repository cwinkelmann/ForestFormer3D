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

model.prepare_epoch: the config's default (1000, out of a 3000-epoch paper run) means a
200-epoch benchmark run never leaves the warm-up phase and the instance decoder is never
trained (val/test F1 comes back 0.0 for both variants -- what the first real runs hit).
--cfg-options always includes model.prepare_epoch=<FF3D_PREPARE_EPOCH, default 60> (the
same ~30% warm-up ratio as the paper's 1000/3000), and FF3D_EXTRA_CFG_OPTIONS (a
space-separated list of key=value tokens) is appended after it for future overrides.
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
    assert ("--cfg-options train_cfg.max_epochs=200 train_cfg.val_interval=20 "
            "default_hooks.checkpoint.max_keep_ckpts=2 model.prepare_epoch=60") in out
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


@pytest.mark.parametrize("variant", ["fixed", "old"])
def test_ff3d_prepare_epoch_overrides_default(tmp_path, variant):
    root = tmp_path / "root"
    root.mkdir()
    _seed_common(root)
    old_root = tmp_path / "old" if variant == "old" else None
    if old_root:
        _seed_old_worktree(old_root)

    env = _base_env(root, variant, old_root)
    env["FF3D_PREPARE_EPOCH"] = "120"
    r = _run(env, [variant], cwd=root)
    out = r.stdout + r.stderr

    assert r.returncode == 0, out
    assert "model.prepare_epoch=120" in out
    assert "model.prepare_epoch=60" not in out


@pytest.mark.parametrize("variant", ["fixed", "old"])
def test_ff3d_extra_cfg_options_appended(tmp_path, variant):
    root = tmp_path / "root"
    root.mkdir()
    _seed_common(root)
    old_root = tmp_path / "old" if variant == "old" else None
    if old_root:
        _seed_old_worktree(old_root)

    env = _base_env(root, variant, old_root)
    env["FF3D_EXTRA_CFG_OPTIONS"] = "a.b=1 c.d=2"
    r = _run(env, [variant], cwd=root)
    out = r.stdout + r.stderr

    assert r.returncode == 0, out
    # model.prepare_epoch=60 (the default) stays, plus both extra tokens appended after it
    assert ("--cfg-options train_cfg.max_epochs=200 train_cfg.val_interval=20 "
            "default_hooks.checkpoint.max_keep_ckpts=2 model.prepare_epoch=60 "
            "a.b=1 c.d=2") in out


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


# --------------------------------------------------------------------------------------
# Preconditions around the test list, and the REAL-run (non-dry-run) behaviour of the test
# stage, driven by tests/benchmark_fakes.py's fake docker binary (no docker, no GPU, no
# torch): marker writes, adopting an already-complete output dir, FF3D_FORCE, and the
# tolerated non-zero exit of the old tools/test.py.
# --------------------------------------------------------------------------------------
from benchmark_fakes import write_fake_docker  # noqa: E402


def test_missing_test_list_dies_with_a_clear_message(tmp_path):
    """The test list used to be read (grep -c) BEFORE any precondition: a missing file died
    with a raw `grep: ...: No such file or directory` and an empty one made grep exit 1,
    which set -e turned into a silent death. Both are explicit ff3d_die messages now."""
    root = tmp_path / "root"
    (root / "data" / "ForAINetV2").mkdir(parents=True)
    r = _run({"FF3D_ROOT": str(root), "FF3D_FOREGROUND": "1", "FF3D_DRY_RUN": "1"}, ["fixed"])
    out = r.stdout + r.stderr
    assert r.returncode == 1, out
    assert "ERROR: missing" in out and "test_list.txt" in out
    assert "grep:" not in out


def test_empty_test_list_dies_with_a_clear_message(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    _seed_common(root)
    (root / "data" / "ForAINetV2" / "meta_data" / "test_list.txt").write_text("\n\n")
    r = _run({"FF3D_ROOT": str(root), "FF3D_FOREGROUND": "1", "FF3D_DRY_RUN": "1"}, ["fixed"])
    out = r.stdout + r.stderr
    assert r.returncode == 1, out
    assert "ERROR: empty" in out and "test_list.txt" in out


def _run_real(root: Path, variant: str, tmp_path: Path, old_root: Path = None, env_extra=None):
    fake = write_fake_docker(tmp_path)
    env = {
        "FF3D_ROOT": str(root),
        "FF3D_FOREGROUND": "1",
        "FF3D_DOCKER": str(fake),
        "FAKE_LOG": str(tmp_path / "fake-docker.log"),
        "FAKE_N_PLY": "1",       # == the single scan in _seed_common's test_list.txt
    }
    if old_root is not None:
        env["FF3D_OLD_ROOT"] = str(old_root)
    if env_extra:
        env.update(env_extra)
    env_full = dict(os.environ)
    env_full.pop("FF3D_DRY_RUN", None)
    env_full.update(env)
    r = subprocess.run([BASH, str(SCRIPT), variant], capture_output=True, text=True,
                       env=env_full, cwd=str(root))
    log = tmp_path / "fake-docker.log"
    return r, (log.read_text() if log.exists() else "")


def _seed_old_worktree_runnable(old_root: Path):
    _seed_old_worktree(old_root)
    (old_root / "docker").mkdir(parents=True, exist_ok=True)
    entry = old_root / "docker" / "entrypoint.sh"
    entry.write_text('#!/bin/sh\nexec "$@"\n')
    entry.chmod(0o755)


def test_real_run_writes_every_marker_inside_its_output_dir(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    _seed_common(root)
    r, dockerlog = _run_real(root, "fixed", tmp_path)
    assert r.returncode == 0, r.stdout + r.stderr
    work = root / "work_dirs" / "bench-fixed-200"
    assert (work / ".done-train").exists()
    assert (work / "test" / ".done-test").exists()
    assert (work / "test" / ".done-eval").exists()
    assert (work / "epoch_200.pth").exists()
    assert dockerlog.count("tools/test.py") == 1


def test_real_run_adopts_a_complete_test_dir_instead_of_re_inferring(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    _seed_common(root)
    work = root / "work_dirs" / "bench-fixed-200"
    (work / "test").mkdir(parents=True)
    (work / ".done-train").write_text("")
    (work / "epoch_200.pth").write_text("")
    (work / "test" / "scan_1.ply").write_text("original-1\n")

    r, dockerlog = _run_real(root, "fixed", tmp_path)
    out = r.stdout + r.stderr
    assert r.returncode == 0, out
    assert "adopting 1 existing PLYs" in out, out
    assert "tools/test.py" not in dockerlog, dockerlog
    assert (work / "test" / "scan_1.ply").read_text() == "original-1\n"
    assert (work / "test" / ".done-test").exists()


def test_real_run_ff3d_force_re_runs_the_test_stage(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    _seed_common(root)
    work = root / "work_dirs" / "bench-fixed-200"
    (work / "test").mkdir(parents=True)
    (work / ".done-train").write_text("")
    (work / "epoch_200.pth").write_text("")
    (work / "test" / "scan_1.ply").write_text("original-1\n")

    r, dockerlog = _run_real(root, "fixed", tmp_path, env_extra={"FF3D_FORCE": "1"})
    out = r.stdout + r.stderr
    assert r.returncode == 0, out
    assert "adopting" not in out
    assert dockerlog.count("tools/test.py") == 1, dockerlog
    assert (work / "test" / "scan_1.ply").read_text() == "fresh-ply-1\n"


def test_real_run_old_test_py_nonzero_exit_is_tolerated(tmp_path):
    """Same rule as run_release_eval.sh: the old evaluator crashes after writing every ply,
    so a non-zero exit is a WARNING and the ply count decides."""
    root = tmp_path / "root"
    root.mkdir()
    _seed_common(root)
    old_root = tmp_path / "old"
    _seed_old_worktree_runnable(old_root)

    r, _ = _run_real(root, "old", tmp_path, old_root=old_root,
                     env_extra={"FAKE_OLD_TEST_RC": "1"})
    out = r.stdout + r.stderr
    assert r.returncode == 0, out
    assert "WARNING: old tools/test.py exited 1" in out, out
    assert (root / "work_dirs" / "bench-old-200" / "test" / ".done-test").exists()


def test_real_run_old_test_py_nonzero_exit_still_fails_on_a_short_ply_count(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    _seed_common(root)
    old_root = tmp_path / "old"
    _seed_old_worktree_runnable(old_root)

    r, _ = _run_real(root, "old", tmp_path, old_root=old_root,
                     env_extra={"FAKE_OLD_TEST_RC": "1", "FAKE_N_PLY": "0"})
    out = r.stdout + r.stderr
    assert r.returncode == 1, out
    assert "produced 0 ply files, expected 1" in out
    assert not (root / "work_dirs" / "bench-old-200" / "test" / ".done-test").exists()


def test_real_run_fixed_test_py_nonzero_exit_is_fatal(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    _seed_common(root)
    r, _ = _run_real(root, "fixed", tmp_path, env_extra={"FAKE_TEST_RC": "1"})
    out = r.stdout + r.stderr
    assert r.returncode == 1, out
    assert "fixed tools/test.py failed with exit 1" in out
    assert not (root / "work_dirs" / "bench-fixed-200" / "test" / ".done-test").exists()


def test_dry_run_reports_the_adopt_decision_and_writes_nothing(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    _seed_common(root)
    work = root / "work_dirs" / "bench-fixed-200"
    (work / "test").mkdir(parents=True)
    (work / "test" / "scan_1.ply").write_text("original-1\n")
    before = _snapshot(root)

    r = _run(_base_env(root, "fixed"), ["fixed"], cwd=root)
    out = r.stdout + r.stderr
    assert r.returncode == 0, out
    assert "DRY: (would adopt) 1 existing ply files" in out, out
    assert "python tools/test.py" not in out
    assert _snapshot(root) == before
