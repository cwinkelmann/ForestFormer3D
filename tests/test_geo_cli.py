"""Tests for ff3d_geo.cli (``python -m ff3d_geo run/convert/georef/report``).

Needs laspy/plyfile (and shapely/pyproj/geopandas for the end-to-end path), so the
module starts with ``pytest.importorskip`` like the other geo test modules; the
system-python run (no geo libs installed) skips it instead of failing.

The planning tests build a throw-away "repo" under ``tmp_path``: ``--dry-run``
executes nothing, so the fake repo only has to exist as a directory (no
``benchmark/``, no ``tools/``).
"""

import json
import os
import re
import subprocess
import sys

import pytest

pytest.importorskip("laspy")
pytest.importorskip("plyfile")

import numpy as np  # noqa: E402
from plyfile import PlyData, PlyElement  # noqa: E402

from ff3d_geo.cli import (  # noqa: E402
    DEFAULT_CHECKPOINT,
    DEFAULT_CONFIG,
    DEFAULT_GPU,
    REPO_ROOT,
    Step,
    build_parser,
    execute,
    main,
    plan_run,
)
from geo_fixtures import two_cone_points, write_two_cone_las  # noqa: E402

STEM = "r12_tegel_E381300_N5828300_100m"


@pytest.fixture
def fake_repo(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    return repo


def test_plan_run_builds_the_expected_host_and_docker_steps(fake_repo, tmp_path):
    las = tmp_path / f"{STEM}.las"
    out = fake_repo / "work_dirs" / "tegel-r12"
    steps = plan_run(las, DEFAULT_CHECKPOINT, out, repo=fake_repo, gpu="5")

    assert [s.name for s in steps] == [
        "las_to_ply",
        "prepare_inputs",
        "preprocess",
        "check_preprocess",
        "inference",
        "results_to_las",
        "trees_to_gpkg",
        "report",
    ]
    commands = [s for s in steps if s.argv is not None]
    assert [s.name for s in commands] == ["preprocess", "inference"]

    data_ply = fake_repo / "data/ForAINetV2/test_data" / f"{STEM}.ply"
    assert str(data_ply) in steps[0].name_detail

    pre, inf = commands
    assert pre.cwd == fake_repo and inf.cwd == fake_repo
    assert pre.env == {"FF3D_ROOT": str(fake_repo), "FF3D_GPU": "5"}
    assert inf.env == {"FF3D_ROOT": str(fake_repo), "FF3D_GPU": "5"}

    assert pre.argv[:2] == ["bash", "-c"]
    assert pre.argv[2] == (
        "source benchmark/common.sh; ff3d_docker bash -c "
        '"cd data/ForAINetV2 && python batch_load_ForAINetV2_data.py --unlabeled'
        " --test_scan_names_file /workspace/work_dirs/tegel-r12/scan_list.txt"
        " --train_scan_names_file /workspace/work_dirs/tegel-r12/empty_list.txt"
        " --val_scan_names_file /workspace/work_dirs/tegel-r12/empty_list.txt"
        " && cd /workspace && python tools/create_data_forainetv2.py forainetv2"
        " --test-list /workspace/work_dirs/tegel-r12/scan_list.txt"
        " --splits test"
        ' --out-dir /workspace/work_dirs/tegel-r12"'
    )
    assert inf.argv[:2] == ["bash", "-c"]
    assert inf.argv[2] == (
        "source benchmark/common.sh; ff3d_docker python tools/test.py "
        f"{DEFAULT_CONFIG} {DEFAULT_CHECKPOINT} --work-dir work_dirs/tegel-r12"
        " --cfg-options test_dataloader.dataset.ann_file="
        "/workspace/work_dirs/tegel-r12/forainetv2_oneformer3d_infos_test.pkl"
    )


def test_plan_run_uses_absolute_container_paths_under_workspace(fake_repo, tmp_path):
    steps = plan_run(tmp_path / f"{STEM}.las", DEFAULT_CHECKPOINT,
                     fake_repo / "work_dirs/tegel-r12", repo=fake_repo)
    script = steps[2].argv[2]
    # The scan list must be addressed inside the container, never by its host path.
    assert str(fake_repo) not in script
    assert "/workspace/work_dirs/tegel-r12/scan_list.txt" in script


def test_plan_run_defaults_out_to_work_dirs_tegel_stem(fake_repo, tmp_path):
    steps = plan_run(tmp_path / f"{STEM}.las", DEFAULT_CHECKPOINT, None, repo=fake_repo)
    assert f"--work-dir work_dirs/tegel-{STEM}" in steps[4].argv[2]


def test_plan_run_rejects_stems_ending_in_digits(fake_repo, tmp_path):
    with pytest.raises(ValueError, match=r"_<digits>"):
        plan_run(tmp_path / "plot_E1_N2_07.las", DEFAULT_CHECKPOINT,
                 fake_repo / "o", repo=fake_repo)


def test_plan_run_accepts_a_100m_suffix(fake_repo, tmp_path):
    assert plan_run(tmp_path / f"{STEM}.las", DEFAULT_CHECKPOINT,
                    fake_repo / "o", repo=fake_repo)


def test_plan_run_rejects_an_out_dir_outside_the_repo(fake_repo, tmp_path):
    with pytest.raises(ValueError, match="inside the repo"):
        plan_run(tmp_path / f"{STEM}.las", DEFAULT_CHECKPOINT,
                 tmp_path / "elsewhere", repo=fake_repo)


def test_step_render_quotes_arguments_and_names_python_steps():
    step = Step(name="x", argv=["bash", "-c", "a b"], cwd="/w", env={"FF3D_GPU": "5"})
    assert step.render() == "$ cd /w && FF3D_GPU=5 bash -c 'a b'"
    assert Step(name="report", name_detail="-> r.json").render() == "# python: report -> r.json"
    assert Step(name="report").render() == "# python: report"


def test_run_dry_run_prints_every_command_and_touches_nothing(fake_repo, tmp_path, capsys):
    las = tmp_path / f"{STEM}.las"
    out = fake_repo / "work_dirs" / "tegel-r12"
    rc = main(["run", "--las", str(las), "--out", str(out), "--repo", str(fake_repo),
               "--gpu", "5", "--dry-run"])
    assert rc == 0
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 8
    assert lines[0].startswith("# python: las_to_ply")
    assert lines[2].startswith(f"$ cd {fake_repo} && FF3D_ROOT={fake_repo} FF3D_GPU=5 bash -c ")
    assert "batch_load_ForAINetV2_data.py --unlabeled" in lines[2]
    assert "tools/test.py" in lines[4] and "--work-dir work_dirs/tegel-r12" in lines[4]
    # nothing written: no out dir, no PLY in the repo, no scan list
    assert not out.exists()
    assert list(fake_repo.iterdir()) == []


def test_the_private_scan_list_reaches_both_batch_load_and_create_data(fake_repo, tmp_path):
    """G2 + R-P3-9: the tracked meta_data/test_list.txt is never an input or output of
    any step, and the private scan list reaches create_data (which would otherwise read
    the tracked list and build a pkl that does not contain this tile)."""
    steps = plan_run(tmp_path / f"{STEM}.las", DEFAULT_CHECKPOINT,
                     fake_repo / "work_dirs/o", repo=fake_repo)
    rendered = "\n".join(s.render() for s in steps) + "\n".join(
        s.name_detail or "" for s in steps)
    assert "meta_data/test_list.txt" not in rendered
    assert "meta_data" not in rendered

    script = steps[2].argv[2]
    scan_list = "/workspace/work_dirs/o/scan_list.txt"
    assert f"--test_scan_names_file {scan_list}" in script
    assert f"--test-list {scan_list} --splits test" in script
    # the pkl is written into <out>, so the benchmark pkls in data/ForAINetV2 stay put
    assert "--out-dir /workspace/work_dirs/o" in script
    assert ("--cfg-options test_dataloader.dataset.ann_file="
            "/workspace/work_dirs/o/forainetv2_oneformer3d_infos_test.pkl"
            in steps[4].argv[2])


def test_execute_reports_the_failing_step_by_name(fake_repo, tmp_path, monkeypatch, capsys):
    def boom(argv, **kwargs):
        raise subprocess.CalledProcessError(returncode=3, cmd=argv)

    monkeypatch.setattr(subprocess, "run", boom)
    steps = [Step(name="inference", argv=["bash", "-c", "true"], cwd=fake_repo)]
    with pytest.raises(RuntimeError, match="step 'inference' failed with exit 3"):
        execute(steps, dry_run=False)


def _run_with_fake_preprocess(tmp_path, monkeypatch, produce):
    """Run the plan up to check_preprocess with the docker steps replaced by ``produce``."""
    repo = tmp_path / "repo"
    repo.mkdir()
    las = tmp_path / f"{STEM}.las"
    out = repo / "work_dirs" / "o"
    steps = plan_run(las, DEFAULT_CHECKPOINT, out, repo=repo)
    inst = repo / "data/ForAINetV2/forainetv2_instance_data"

    def fake_run(argv, **kwargs):
        produce(inst, out)
        return subprocess.CompletedProcess(argv, 0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    # las_to_ply needs a real LAS; skip it and write the scan list ourselves.
    out.mkdir(parents=True)
    (out / "scan_list.txt").write_text(f"{STEM}\n")
    execute([s for s in steps if s.name in ("preprocess", "check_preprocess")], dry_run=False)


def test_check_preprocess_passes_when_every_artefact_is_there(tmp_path, monkeypatch, capsys):
    def produce(inst, out):
        inst.mkdir(parents=True, exist_ok=True)
        np.save(inst / f"{STEM}_vert.npy", np.zeros((1, 3)))
        np.save(inst / f"{STEM}_offsets.npy", np.zeros(3))
        (out / "forainetv2_oneformer3d_infos_test.pkl").write_bytes(b"fake pkl")

    _run_with_fake_preprocess(tmp_path, monkeypatch, produce)  # must not raise


@pytest.mark.parametrize("omit", ["_vert.npy", "_offsets.npy", "pkl"])
def test_check_preprocess_names_the_missing_artefact(tmp_path, monkeypatch, capsys, omit):
    def produce(inst, out):
        inst.mkdir(parents=True, exist_ok=True)
        if omit != "_vert.npy":
            np.save(inst / f"{STEM}_vert.npy", np.zeros((1, 3)))
        if omit != "_offsets.npy":
            np.save(inst / f"{STEM}_offsets.npy", np.zeros(3))
        if omit != "pkl":
            (out / "forainetv2_oneformer3d_infos_test.pkl").write_bytes(b"fake pkl")

    expected = "forainetv2_oneformer3d_infos_test.pkl" if omit == "pkl" else f"{STEM}{omit}"
    with pytest.raises(RuntimeError, match=re.escape(expected)):
        _run_with_fake_preprocess(tmp_path, monkeypatch, produce)


def test_check_preprocess_rejects_a_pkl_older_than_the_scan_list(tmp_path, monkeypatch):
    def produce(inst, out):
        inst.mkdir(parents=True, exist_ok=True)
        np.save(inst / f"{STEM}_vert.npy", np.zeros((1, 3)))
        np.save(inst / f"{STEM}_offsets.npy", np.zeros(3))
        stale = out / "forainetv2_oneformer3d_infos_test.pkl"
        stale.write_bytes(b"stale pkl")
        os.utime(stale, (1_000_000, 1_000_000))  # older than the scan list

    with pytest.raises(RuntimeError, match="is older than"):
        _run_with_fake_preprocess(tmp_path, monkeypatch, produce)


def test_gpu_defaults_to_the_ff3d_gpu_environment_variable(monkeypatch):
    monkeypatch.delenv("FF3D_GPU", raising=False)
    assert build_parser().parse_args(["run", "--las", "a.las"]).gpu == DEFAULT_GPU == "0"
    monkeypatch.setenv("FF3D_GPU", "7")
    assert build_parser().parse_args(["run", "--las", "a.las"]).gpu == "7"


def test_shell_guard_rejects_an_option_looking_path(fake_repo, tmp_path):
    with pytest.raises(ValueError, match="cannot be embedded"):
        plan_run(tmp_path / f"{STEM}.las", DEFAULT_CHECKPOINT,
                 fake_repo / "-x" / "out", repo=fake_repo)


def test_convert_subcommand_writes_ply_sidecar_and_classification(tmp_path):
    las = write_two_cone_las(tmp_path / f"cones_{STEM}.las")
    ply = tmp_path / "test_data" / "cones.ply"
    sidecar = tmp_path / "out" / "cones.sidecar.json"
    rc = main(["convert", "--las", str(las), "--ply", str(ply), "--sidecar", str(sidecar)])
    assert rc == 0
    assert ply.is_file()
    data = json.loads(sidecar.read_text())
    assert data["origin"] == [381300.0, 5828300.0] and data["epsg"] == 25833
    assert (tmp_path / "out" / f"cones_{STEM}_classification.npy").is_file()


def _write_fake_result_ply(input_ply, offsets_npy, result_ply):
    """Mimic batch_load + tools/test.py on the two-cone tile: centered float32
    coordinates plus semantic_pred / instance_pred / score."""
    src = PlyData.read(str(input_ply))["vertex"].data
    xyz = np.column_stack([src["x"], src["y"], src["z"]]).astype(np.float64)
    offsets = np.array([xyz[:, 0].mean(), xyz[:, 1].mean(), xyz[:, 2].min()], dtype=np.float64)
    offsets_npy.parent.mkdir(parents=True, exist_ok=True)
    np.save(offsets_npy, offsets)
    _, _, _, _, tid, sem = two_cone_points()
    result = np.empty(len(src), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"),
                                       ("semantic_pred", "i4"), ("instance_pred", "i4"),
                                       ("score", "f4")])
    result["x"], result["y"], result["z"] = (xyz - offsets).T.astype(np.float32)
    result["semantic_pred"] = sem.astype(np.int32)
    result["instance_pred"] = tid.astype(np.int32)
    result["score"] = np.where(tid >= 0, 0.9, -1.0).astype(np.float32)
    result_ply.parent.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(result, "vertex")], text=False, byte_order="<").write(
        str(result_ply))


def test_run_executes_the_docker_steps_and_then_the_host_steps(tmp_path, monkeypatch, capsys):
    pytest.importorskip("shapely")
    pytest.importorskip("geopandas")
    repo = tmp_path / "repo"
    repo.mkdir()
    las = write_two_cone_las(tmp_path / f"{STEM}.las")
    out = repo / "work_dirs" / f"tegel-{STEM}"
    input_ply = repo / "data/ForAINetV2/test_data" / f"{STEM}.ply"
    offsets_npy = repo / "data/ForAINetV2/forainetv2_instance_data" / f"{STEM}_offsets.npy"

    calls = []

    def fake_run(argv, **kwargs):
        calls.append((argv, kwargs.get("cwd"), kwargs.get("env", {})))
        if "batch_load_ForAINetV2_data.py" in argv[2]:
            # batch_load + create_data: the centering offsets land next to the .npy exports
            offsets_npy.parent.mkdir(parents=True, exist_ok=True)
            np.save(offsets_npy.parent / f"{STEM}_vert.npy", np.zeros((1, 3)))
            (out / "forainetv2_oneformer3d_infos_test.pkl").write_bytes(b"fake pkl")
            _write_fake_result_ply(input_ply, offsets_npy, out / f"{STEM}.ply")
        return subprocess.CompletedProcess(argv, 0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    rc = main(["run", "--las", str(las), "--out", str(out), "--repo", str(repo), "--gpu", "3"])
    assert rc == 0

    assert len(calls) == 2
    assert all(argv[:2] == ["bash", "-c"] for argv, _, _ in calls)
    assert all(cwd == str(repo) for _, cwd, _ in calls)
    assert all(env["FF3D_GPU"] == "3" and env["FF3D_ROOT"] == str(repo)
               for _, _, env in calls)
    assert "tools/test.py" in calls[1][0][2]

    assert input_ply.is_file()
    assert (out / f"{STEM}.sidecar.json").is_file()
    assert (out / "scan_list.txt").read_text() == f"{STEM}\n"
    assert (out / "empty_list.txt").read_text() == ""
    assert (out / f"{STEM}.las").is_file()
    assert (out / f"{STEM}_trees.gpkg").is_file()
    assert (out / f"{STEM}_report.json").is_file()
    assert (out / f"{STEM}_report.md").is_file()
    # the benchmark pkls in data/ForAINetV2 are untouched; ours lives under <out>
    assert (out / "forainetv2_oneformer3d_infos_test.pkl").is_file()
    assert not (repo / "data/ForAINetV2/forainetv2_oneformer3d_infos_test.pkl").exists()
    report = json.loads((out / f"{STEM}_report.json").read_text())
    assert report["n_trees"] == 2


def test_georef_subcommand_writes_las_and_optional_gpkg(tmp_path):
    pytest.importorskip("shapely")
    pytest.importorskip("geopandas")
    from ff3d_geo.convert import las_to_ply

    las = write_two_cone_las(tmp_path / f"{STEM}.las")
    input_ply = tmp_path / f"{STEM}.ply"
    sidecar = tmp_path / f"{STEM}.sidecar.json"
    las_to_ply(las, input_ply, sidecar)
    offsets = tmp_path / f"{STEM}_offsets.npy"
    result_ply = tmp_path / "result.ply"
    _write_fake_result_ply(input_ply, offsets, result_ply)
    out_las = tmp_path / "out.las"
    gpkg = tmp_path / "out.gpkg"
    rc = main(["georef", "--result-ply", str(result_ply), "--sidecar", str(sidecar),
               "--offsets", str(offsets), "--out-las", str(out_las), "--gpkg", str(gpkg)])
    assert rc == 0
    assert out_las.is_file() and gpkg.is_file()


def test_parser_exposes_the_four_subcommands_and_defaults():
    parser = build_parser()
    args = parser.parse_args(["run", "--las", "a.las"])
    assert args.checkpoint == DEFAULT_CHECKPOINT
    assert args.config == DEFAULT_CONFIG
    assert args.gpu == os.environ.get("FF3D_GPU", DEFAULT_GPU)
    assert args.out is None and args.dry_run is False


def test_module_entry_point_shows_help():
    proc = subprocess.run([sys.executable, "-m", "ff3d_geo", "--help"],
                          capture_output=True, text=True, cwd=str(REPO_ROOT))
    assert proc.returncode == 0
    assert "{run,convert,georef,report}" in proc.stdout
