"""Tests for ff3d_geo.cli (``python -m ff3d_geo run/convert/georef/report``).

Needs laspy/plyfile (and shapely/pyproj/geopandas for the end-to-end path), so the
module starts with ``pytest.importorskip`` like the other geo test modules; the
system-python run (no geo libs installed) skips it instead of failing.

The planning tests build a throw-away "repo" under ``tmp_path``: ``--dry-run``
executes nothing, so the fake repo only has to exist as a directory (no
``benchmark/``, no ``tools/``).
"""

import json
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
    steps = plan_run(las, DEFAULT_CHECKPOINT, out, repo=fake_repo, gpu=5)

    assert [s.name for s in steps] == [
        "las_to_ply",
        "prepare_inputs",
        "preprocess",
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
        ' && cd /workspace && python tools/create_data_forainetv2.py forainetv2"'
    )
    assert inf.argv[:2] == ["bash", "-c"]
    assert inf.argv[2] == (
        "source benchmark/common.sh; ff3d_docker python tools/test.py "
        f"{DEFAULT_CONFIG} {DEFAULT_CHECKPOINT} --work-dir work_dirs/tegel-r12"
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
    assert f"--work-dir work_dirs/tegel-{STEM}" in steps[3].argv[2]


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
    assert len(lines) == 7
    assert lines[0].startswith("# python: las_to_ply")
    assert lines[2].startswith(f"$ cd {fake_repo} && FF3D_ROOT={fake_repo} FF3D_GPU=5 bash -c ")
    assert "batch_load_ForAINetV2_data.py --unlabeled" in lines[2]
    assert "tools/test.py" in lines[3] and "--work-dir work_dirs/tegel-r12" in lines[3]
    # nothing written: no out dir, no PLY in the repo, no scan list
    assert not out.exists()
    assert list(fake_repo.iterdir()) == []


def test_dry_run_never_touches_the_tracked_test_list(fake_repo, tmp_path):
    """G2: the tracked meta_data/test_list.txt is never a step input or output."""
    steps = plan_run(tmp_path / f"{STEM}.las", DEFAULT_CHECKPOINT,
                     fake_repo / "work_dirs/o", repo=fake_repo)
    rendered = "\n".join(s.render() for s in steps) + "\n".join(
        s.name_detail or "" for s in steps)
    assert "meta_data/test_list.txt" not in rendered


def test_execute_reports_the_failing_step_by_name(fake_repo, tmp_path, monkeypatch, capsys):
    def boom(argv, **kwargs):
        raise subprocess.CalledProcessError(returncode=3, cmd=argv)

    monkeypatch.setattr(subprocess, "run", boom)
    steps = [Step(name="inference", argv=["bash", "-c", "true"], cwd=fake_repo)]
    with pytest.raises(RuntimeError, match="step 'inference' failed with exit 3"):
        execute(steps, dry_run=False)


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
    assert args.gpu == DEFAULT_GPU
    assert args.out is None and args.dry_run is False


def test_module_entry_point_shows_help():
    proc = subprocess.run([sys.executable, "-m", "ff3d_geo", "--help"],
                          capture_output=True, text=True, cwd=str(REPO_ROOT))
    assert proc.returncode == 0
    assert "{run,convert,georef,report}" in proc.stdout
