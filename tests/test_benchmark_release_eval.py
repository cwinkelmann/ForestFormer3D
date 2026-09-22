"""Tests for benchmark/run_release_eval.sh.

FF3D_DRY_RUN=1 FF3D_FOREGROUND=1 exercises the whole stage sequence without invoking docker
or python: common.sh's ff3d_run prints "DRY: <cmd...>" instead of executing it. The fixture
pre-creates the preprocessing marker pkl (forainetv2_oneformer3d_infos_test.pkl) so
ff3d_preprocess takes its already-done branch under dry-run -- its own postcondition check
(the real create_data call would produce that pkl, but under dry-run nothing really runs, so
asserting on it would always fail; pre-seeding it is the documented workaround, see the task
report).

Per common.sh (ff3d_prepare_checkpoint, ff3d_daemonize -- see their comments): under
FF3D_DRY_RUN=1 common.sh writes NOTHING to disk. ff3d_prepare_checkpoint prints the two
commands it would have run and the three output paths it would have written, but does not
determine or report a "raw"/"converted" layout (that can only be known by actually running
the container). ff3d_daemonize, when FF3D_FOREGROUND is not already set, prints a
"DRY: (daemonize skipped, continuing in foreground)" line and returns instead of forking --
no log directory, no background process.

No docker, no python subprocess side effects -- stdlib only.
"""
import os
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "benchmark" / "run_release_eval.sh"


def _snapshot_files(d: Path) -> set:
    """All paths (files AND directories) under d, relative -- for before/after dry-run diffs.
    Must include directories: a dry run must not even mkdir the output dirs."""
    if not d.exists():
        return set()
    return {str(p.relative_to(d)) for p in d.rglob("*")}


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
    before = _snapshot_files(root / "work_dirs")
    r = _run(root, old)
    out = r.stdout
    assert r.returncode == 0, out

    # a dry run writes nothing at all under work_dirs (no .layout/_converted/_raw files, no
    # bench-release markers, no logs/ dir) -- per common.sh's ff3d_prepare_checkpoint and
    # ff3d_daemonize dry-run contracts
    after = _snapshot_files(root / "work_dirs")
    assert after == before, f"dry run created/removed files under work_dirs: {after ^ before}"

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


def test_done_test_marker_lives_in_the_stage_output_dir_and_skips_that_stage_only(tmp_path):
    """Markers live INSIDE the stage's own output dir (work_dirs/bench-release-fixed/.done-test),
    matching run_train_200.sh -- not in a third work_dirs/bench-release/ directory, which is
    where pre-2026-09-22 runs put them (historical; nothing reads those any more)."""
    root = _make_root(tmp_path)
    old = _make_old(tmp_path)
    fixed_dir = root / "work_dirs" / "bench-release-fixed"
    fixed_dir.mkdir(parents=True)
    (fixed_dir / ".done-test").write_bytes(b"")

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


def test_dry_run_never_writes_markers(tmp_path):
    """A dry run must never create a stage's .done-* marker: nothing was actually verified
    (the docker calls were only printed), so a marker left behind by a dry run would make a
    SUBSEQUENT REAL run silently skip every stage. Also: a second dry run on the same (still
    marker-less) layout must print every stage again, not skip any."""
    root = _make_root(tmp_path)
    old = _make_old(tmp_path)
    stage_dirs = [root / "work_dirs" / "bench-release-fixed", root / "work_dirs" / "bench-release-old"]
    before = _snapshot_files(root / "work_dirs")

    def _no_markers():
        return all(not d.exists() or list(d.glob(".done-*")) == [] for d in stage_dirs)

    r1 = _run(root, old)
    assert r1.returncode == 0, r1.stdout
    assert _no_markers()
    assert _snapshot_files(root / "work_dirs") == before

    r2 = _run(root, old)
    out2 = r2.stdout
    assert r2.returncode == 0, out2
    assert _no_markers()
    assert "already done" not in out2
    assert _snapshot_files(root / "work_dirs") == before

    fixed_test_lines = [
        l for l in out2.splitlines()
        if "tools/test.py" in l and "epoch_3000_converted.pth" in l
    ]
    old_test_lines = [
        l for l in out2.splitlines()
        if "tools/test.py" in l and "epoch_3000_raw.pth" in l
    ]
    eval_lines = [l for l in out2.splitlines() if "tools/final_eval.py" in l]
    assert len(fixed_test_lines) == 1, out2
    assert len(old_test_lines) == 1, out2
    assert len(eval_lines) == 2, out2


def test_dry_run_does_not_touch_preexisting_output_files(tmp_path):
    """A dry run must not create the output dirs or delete result files left by an
    interrupted REAL run: pre-seed work_dirs/bench-release-fixed/{a,b}.ply and
    evaluation_total_test.txt with NO .done-* markers present (as if a real run died between
    test.py and final_eval.py, or the marker write just hadn't happened yet), then assert
    they all still exist, byte-for-byte, and the whole work_dirs snapshot (files and dirs) is
    unchanged -- the ff3d_run-gated mkdir/rm in run_test_stage/run_eval_stage must never
    execute for real under FF3D_DRY_RUN=1."""
    root = _make_root(tmp_path)
    old = _make_old(tmp_path)
    fixed_dir = root / "work_dirs" / "bench-release-fixed"
    fixed_dir.mkdir(parents=True)
    seeded = {
        fixed_dir / "a.ply": b"ply-a",
        fixed_dir / "b.ply": b"ply-b",
        fixed_dir / "evaluation_total_test.txt": b"stale eval\n",
    }
    for p, content in seeded.items():
        p.write_bytes(content)

    before = _snapshot_files(root / "work_dirs")

    r = _run(root, old)
    assert r.returncode == 0, r.stdout

    for p, content in seeded.items():
        assert p.exists(), f"{p} was deleted by a dry run"
        assert p.read_bytes() == content, f"{p} was modified by a dry run"

    after = _snapshot_files(root / "work_dirs")
    assert after == before, f"dry run created/removed paths under work_dirs: {after ^ before}"


def test_dry_run_without_foreground_skips_fork(tmp_path):
    """Per common.sh's ff3d_daemonize dry-run contract: with FF3D_DRY_RUN=1 and no
    FF3D_FOREGROUND, the fork is skipped entirely (no log dir, no background process) and
    the script continues in the foreground, printing its own dry-run output directly."""
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
    assert "DRY: (daemonize skipped, continuing in foreground)" in r.stdout
    assert "started release-eval pid" not in r.stdout
    assert "release eval finished" in r.stdout
    assert not (root / "work_dirs" / "logs").exists()


def test_real_run_without_foreground_daemonizes(tmp_path):
    """Without FF3D_DRY_RUN and without FF3D_FOREGROUND=1, the script forks under nohup and
    returns immediately, printing the log path; the log then contains the same output a
    foreground run would print (mirrors the manual check in the task brief). No checkpoint is
    staged, so the forked run dies at the first precondition -- this only exercises the
    daemonize/nohup re-exec path, not real inference."""
    root = tmp_path / "root"
    meta = root / "data" / "ForAINetV2" / "meta_data"
    meta.mkdir(parents=True)
    (meta / "test_list.txt").write_text("a_test\nb_test\n")
    old = _make_old(tmp_path)
    env = dict(os.environ, FF3D_ROOT=str(root), FF3D_OLD_ROOT=str(old))
    r = subprocess.run(["bash", str(SCRIPT)], env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout
    assert "started release-eval pid" in r.stdout

    import time
    log_dir = root / "work_dirs" / "logs"
    logs = []
    for _ in range(50):
        logs = list(log_dir.glob("release-eval-*.log"))
        if logs and logs[0].read_text():
            break
        time.sleep(0.1)
    assert len(logs) == 1
    text = logs[0].read_text()
    assert "ERROR: missing" in text
    assert "epoch_3000_fix.pth" in text


# --------------------------------------------------------------------------------------
# Real-run (non-dry-run) behaviour, with FF3D_DOCKER pointed at tests/benchmark_fakes.py's
# fake docker binary: the ply-count postcondition, the marker writes, adopting an
# already-complete output dir, FF3D_FORCE, and the tolerated crash of the old tools/test.py.
# --------------------------------------------------------------------------------------
from benchmark_fakes import write_fake_docker  # noqa: E402


def _run_real(root, old, tmp_path, env_extra=None):
    fake = write_fake_docker(tmp_path)
    env = dict(
        os.environ,
        FF3D_ROOT=str(root),
        FF3D_OLD_ROOT=str(old),
        FF3D_FOREGROUND="1",
        FF3D_DOCKER=str(fake),
        FAKE_LOG=str(tmp_path / "fake-docker.log"),
        FAKE_N_PLY="2",          # == the 2 scans in the fixture's test_list.txt
    )
    env.pop("FF3D_DRY_RUN", None)
    if env_extra:
        env.update(env_extra)
    r = subprocess.run(["bash", str(SCRIPT)], env=env, stdout=subprocess.PIPE,
                       stderr=subprocess.STDOUT, text=True)
    log = tmp_path / "fake-docker.log"
    return r, (log.read_text() if log.exists() else "")


def _seed_plys(d: Path, n: int, tag: str = "original"):
    d.mkdir(parents=True, exist_ok=True)
    for i in range(1, n + 1):
        (d / f"scan_{i}.ply").write_text(f"{tag}-{i}\n")


def test_real_run_writes_markers_into_the_output_dirs(tmp_path):
    root = _make_root(tmp_path)
    old = _make_old(tmp_path)
    r, _ = _run_real(root, old, tmp_path)
    assert r.returncode == 0, r.stdout
    for name in ("bench-release-fixed", "bench-release-old"):
        d = root / "work_dirs" / name
        assert (d / ".done-test").exists(), r.stdout
        assert (d / ".done-eval").exists(), r.stdout
    # and nothing in the old third directory
    assert not (root / "work_dirs" / "bench-release").exists()


def test_adopts_a_complete_output_dir_instead_of_re_inferring(tmp_path):
    """The exact mishap of 2026-09-22: a stage that produced all N_TEST PLYs but never got
    its marker (the old evaluator crashed / SSH dropped) must be ADOPTED, not wiped and
    re-run for another 1-2.5 h."""
    root = _make_root(tmp_path)
    old = _make_old(tmp_path)
    fixed_dir = root / "work_dirs" / "bench-release-fixed"
    old_dir = root / "work_dirs" / "bench-release-old"
    _seed_plys(fixed_dir, 2)
    _seed_plys(old_dir, 2)

    r, dockerlog = _run_real(root, old, tmp_path)
    assert r.returncode == 0, r.stdout
    assert r.stdout.count("adopting 2 existing PLYs") == 2, r.stdout
    # test.py never ran; only the two final_eval.py calls did
    assert "tools/test.py" not in dockerlog, dockerlog
    assert dockerlog.count("tools/final_eval.py") == 2, dockerlog
    # the pre-existing PLYs are byte-for-byte untouched, and both markers were written
    for d in (fixed_dir, old_dir):
        assert sorted(p.name for p in d.glob("*.ply")) == ["scan_1.ply", "scan_2.ply"]
        assert (d / "scan_1.ply").read_text() == "original-1\n"
        assert (d / ".done-test").exists()


def test_ff3d_force_re_runs_a_complete_output_dir(tmp_path):
    root = _make_root(tmp_path)
    old = _make_old(tmp_path)
    fixed_dir = root / "work_dirs" / "bench-release-fixed"
    _seed_plys(fixed_dir, 2)
    _seed_plys(root / "work_dirs" / "bench-release-old", 2)

    r, dockerlog = _run_real(root, old, tmp_path, {"FF3D_FORCE": "1"})
    assert r.returncode == 0, r.stdout
    assert "adopting" not in r.stdout, r.stdout
    assert dockerlog.count("tools/test.py") == 2, dockerlog
    # the seeded PLYs were deleted and rewritten by the (fake) inference
    assert (fixed_dir / "scan_1.ply").read_text() == "fresh-ply-1\n"
    assert (fixed_dir / ".done-test").exists()


def test_existing_marker_still_wins_over_ff3d_force(tmp_path):
    """FF3D_FORCE only overrides the adopt branch; a stage whose marker exists stays skipped
    (delete the marker to redo it) -- as documented in RUNBOOK-carrot.md section 7."""
    root = _make_root(tmp_path)
    old = _make_old(tmp_path)
    fixed_dir = root / "work_dirs" / "bench-release-fixed"
    _seed_plys(fixed_dir, 2)
    (fixed_dir / ".done-test").write_text("")

    r, dockerlog = _run_real(root, old, tmp_path, {"FF3D_FORCE": "1"})
    assert r.returncode == 0, r.stdout
    assert "fixed test.py already done" in r.stdout
    assert (fixed_dir / "scan_1.ply").read_text() == "original-1\n"
    # only the OLD stage re-ran test.py
    assert dockerlog.count("tools/test.py") == 1, dockerlog


def test_old_test_py_nonzero_exit_is_tolerated_when_all_plys_are_there(tmp_path):
    """The old tools/test.py at 6a75c37 always crashes in its own evaluator AFTER writing
    every result .ply. That must not abort the script: the N_TEST ply count is the real
    postcondition, and the stage still gets its marker."""
    root = _make_root(tmp_path)
    old = _make_old(tmp_path)
    r, dockerlog = _run_real(root, old, tmp_path, {"FAKE_OLD_TEST_RC": "1"})
    assert r.returncode == 0, r.stdout
    assert "WARNING: old tools/test.py exited 1" in r.stdout, r.stdout
    assert (root / "work_dirs" / "bench-release-old" / ".done-test").exists()
    assert (root / "work_dirs" / "bench-release-old" / ".done-eval").exists()
    assert dockerlog.count("tools/test.py") == 2


def test_old_test_py_nonzero_exit_still_fails_on_a_short_ply_count(tmp_path):
    """Tolerating the crash must not tolerate missing results: with fewer PLYs than N_TEST
    the ply-count postcondition still kills the run and writes no marker."""
    root = _make_root(tmp_path)
    old = _make_old(tmp_path)
    r, _ = _run_real(root, old, tmp_path, {"FAKE_OLD_TEST_RC": "1", "FAKE_N_PLY": "1"})
    assert r.returncode == 1, r.stdout
    assert "produced 1 ply files, expected 2" in r.stdout
    assert not (root / "work_dirs" / "bench-release-old" / ".done-test").exists()


def test_fixed_test_py_nonzero_exit_is_fatal(tmp_path):
    """Only the OLD variant's crash is excused; a failing fixed tools/test.py aborts."""
    root = _make_root(tmp_path)
    old = _make_old(tmp_path)
    r, _ = _run_real(root, old, tmp_path, {"FAKE_TEST_RC": "1"})
    assert r.returncode == 1, r.stdout
    assert "fixed tools/test.py failed with exit 1" in r.stdout
    assert not (root / "work_dirs" / "bench-release-fixed" / ".done-test").exists()


def test_dry_run_reports_the_adopt_decision_and_writes_nothing(tmp_path):
    """Dry-run contract extended to the adopt branch: it announces the decision it WOULD
    take, runs no test.py, and creates no marker."""
    root = _make_root(tmp_path)
    old = _make_old(tmp_path)
    _seed_plys(root / "work_dirs" / "bench-release-fixed", 2)
    before = _snapshot_files(root / "work_dirs")

    r = _run(root, old)
    out = r.stdout
    assert r.returncode == 0, out
    assert "DRY: (would adopt) 2 existing ply files in work_dirs/bench-release-fixed" in out
    assert not any(
        "tools/test.py" in l and "epoch_3000_converted.pth" in l for l in out.splitlines()
    )
    # the old stage has no PLYs, so it is still previewed in full
    assert any("tools/test.py" in l and "epoch_3000_raw.pth" in l for l in out.splitlines())
    assert _snapshot_files(root / "work_dirs") == before


def test_empty_test_list_dies_with_a_clear_message(tmp_path):
    root = _make_root(tmp_path)
    (root / "data" / "ForAINetV2" / "meta_data" / "test_list.txt").write_text("\n\n")
    old = _make_old(tmp_path)
    r = _run(root, old)
    assert r.returncode == 1, r.stdout
    assert "empty" in r.stdout and "test_list.txt" in r.stdout
