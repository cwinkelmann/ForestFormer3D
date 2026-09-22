"""``python -m ff3d_geo``: run / convert / georef / report.

Execution model (see the Phase 3 plan, ruling G7): this CLI runs on the HOST, in a
plain CPU venv with laspy/plyfile/geopandas but no torch. The two GPU/pipeline steps
are delegated to the Docker image through ``benchmark/common.sh``'s ``ff3d_docker``
helper, which bind-mounts ``$FF3D_ROOT`` at ``/workspace``. The CLI therefore

* exports ``FF3D_ROOT=<repo>`` so the container always sees the checkout this CLI was
  pointed at, and ``FF3D_GPU=<--gpu>`` so one explicit physical GPU is pinned;
* addresses every path it hands to the container in CONTAINER coordinates
  (``/workspace/...`` or repo-relative, since ``ff3d_docker`` sets ``-w /workspace``),
  never by its host path;
* keeps its own working files (scan list, sidecar, results) inside ``--out``, which
  must live under the repo so the container can see them.

The tracked ``data/ForAINetV2/meta_data/test_list.txt`` is never read or written
(ruling G2, ruling R-P3-9): a private one-line scan list is written to
``<out>/scan_list.txt`` and passed BOTH to ``batch_load_ForAINetV2_data.py``
(``--test_scan_names_file``) and to ``tools/create_data_forainetv2.py``
(``--test-list``, with ``--splits test``), so the info pkl really lists this tile.
``<out>/empty_list.txt`` is passed as the train/val list so ``batch_load`` does not
re-export the 61 training plots.

The pkl itself is written to ``<out>`` (``--out-dir``) rather than into
``data/ForAINetV2``, so the benchmark's ``forainetv2_oneformer3d_infos_{train,val,test}
.pkl`` are left byte-identical while the Phase 2 test stages are still pending;
``tools/test.py`` is pointed at it with
``--cfg-options test_dataloader.dataset.ann_file=/workspace/<out>/...pkl``. An absolute
``ann_file`` survives the dataset's ``data_root`` join either way: mmengine's
``BaseDataset._join_prefix`` only joins when the path is relative, and even a plain
``os.path.join('data/ForAINetV2/', '/workspace/...')`` returns the absolute path
unchanged.

``--dry-run`` prints the plan -- the exact command lines, including the ``bash -c``
scripts -- and touches nothing at all.
"""

from __future__ import annotations

import argparse
import os
import re
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Callable

from ff3d_geo.convert import las_to_ply, results_to_las
from ff3d_geo.origin import parse_origin
from ff3d_geo.trees import trees_to_gpkg

DEFAULT_CONFIG = "configs/oneformer3d_qs_radius16_qp300_2many.py"
DEFAULT_CHECKPOINT = "work_dirs/clean_forestformer/epoch_3000_fix.pth"
DEFAULT_EPSG = 25833
DEFAULT_GPU = "0"  # benchmark/common.sh's own default for FF3D_GPU
REPO_ROOT = Path(__file__).resolve().parents[1]
CONTAINER_ROOT = PurePosixPath("/workspace")

# tools/prepare_safe_testfile_names.sh: the data tools strip a trailing _<digits>
# from scan names (it is the ForAINetV2 block index), so such a stem cannot survive
# the round trip through batch_load/create_data. _100m is fine; _07 is not.
_UNSAFE_STEM = re.compile(r"_\d+$")
# Paths embedded verbatim in the bash -c scripts must not need shell quoting.
_SHELL_SAFE = re.compile(r"^[A-Za-z0-9_@%+=:,./][A-Za-z0-9_@%+=:,./-]*$")


@dataclass
class Step:
    """One pipeline step: a subprocess (``argv``) or a host-side callable (``func``).

    ``name`` is the stable identifier used for timings and error messages;
    ``name_detail`` carries the paths shown to the user by ``render()``.
    """

    name: str
    argv: list[str] | None = None
    cwd: Path | None = None
    env: dict[str, str] = field(default_factory=dict)
    func: Callable[[], object] | None = None
    name_detail: str = ""

    def render(self) -> str:
        if self.argv is None:
            detail = f" {self.name_detail}" if self.name_detail else ""
            return f"# python: {self.name}{detail}"
        env = "".join(f"{k}={shlex.quote(v)} " for k, v in self.env.items())
        return f"$ cd {self.cwd} && {env}{' '.join(shlex.quote(a) for a in self.argv)}"


def _shell_literal(value: str) -> str:
    """Return ``value`` for embedding in a double-quoted bash script, or raise."""
    if not _SHELL_SAFE.match(value):
        raise ValueError(
            f"{value!r} contains characters that cannot be embedded unquoted in the "
            "container command; use a path without spaces or shell metacharacters"
        )
    return value


def _under_repo(path: Path, repo: Path, what: str) -> Path:
    """Resolve ``path`` (relative ones against ``repo``) and require it inside ``repo``."""
    resolved = Path(path)
    resolved = resolved if resolved.is_absolute() else repo / resolved
    resolved = resolved.resolve()
    try:
        resolved.relative_to(repo)
    except ValueError as exc:
        raise ValueError(
            f"{what} {resolved} is not inside the repo {repo}; the container only sees "
            f"{repo} (mounted at {CONTAINER_ROOT}), so it must live under it"
        ) from exc
    return resolved


def _repo_relative(path: Path, repo: Path, what: str) -> str:
    """POSIX path of ``path`` relative to ``repo`` -- what the container's cwd sees."""
    return _shell_literal(_under_repo(path, repo, what).relative_to(repo).as_posix())


def _container_path(path: Path, repo: Path, what: str) -> str:
    """Absolute ``/workspace/...`` path of a host path under ``repo``."""
    rel = _under_repo(path, repo, what).relative_to(repo)
    return _shell_literal((CONTAINER_ROOT / rel).as_posix())


def _docker_step(name: str, inner: str, repo: Path, gpu) -> Step:
    """A step that sources benchmark/common.sh and runs ``inner`` via ``ff3d_docker``."""
    return Step(
        name=name,
        argv=["bash", "-c", f"source benchmark/common.sh; ff3d_docker {inner}"],
        cwd=repo,
        env={"FF3D_ROOT": str(repo), "FF3D_GPU": str(gpu)},
    )


def plan_run(
    las,
    checkpoint=DEFAULT_CHECKPOINT,
    out=None,
    origin: tuple[float, float] | None = None,
    epsg: int = DEFAULT_EPSG,
    config=DEFAULT_CONFIG,
    repo=REPO_ROOT,
    gpu=DEFAULT_GPU,
    timings: dict[str, float] | None = None,
) -> list[Step]:
    """Build the ordered step list for ``run`` without executing or writing anything.

    ``checkpoint`` and ``config`` are conventionally repo-relative (``work_dirs/...``,
    ``configs/...``); absolute ones must still point inside ``repo``. ``out`` defaults
    to ``<repo>/work_dirs/tegel-<stem>``. ``timings`` (step name -> seconds) is filled
    in by :func:`execute`; the report step reads ``timings["inference"]`` from it.
    """
    repo = Path(repo).resolve()
    las = Path(las).resolve()
    stem = las.stem
    if _UNSAFE_STEM.search(stem):
        raise ValueError(
            f"scan name {stem!r} ends in _<digits>, which the ForAINetV2 data tools "
            "treat as a block index and strip; rename the tile (e.g. ..._100m)"
        )
    if origin is None:
        origin = parse_origin(las.name)
    out = _under_repo(Path(out) if out is not None else Path("work_dirs") / f"tegel-{stem}",
                      repo, "--out")
    config_rel = _repo_relative(Path(config), repo, "--config")
    checkpoint_rel = _repo_relative(Path(checkpoint), repo, "--checkpoint")
    out_rel = _repo_relative(out, repo, "--out")

    data_dir = repo / "data" / "ForAINetV2"
    instance_dir = data_dir / "forainetv2_instance_data"
    input_ply = data_dir / "test_data" / f"{stem}.ply"
    scan_list = out / "scan_list.txt"
    empty_list = out / "empty_list.txt"
    sidecar = out / f"{stem}.sidecar.json"
    result_ply = out / f"{stem}.ply"
    offsets_npy = instance_dir / f"{stem}_offsets.npy"
    vert_npy = instance_dir / f"{stem}_vert.npy"
    info_pkl = out / "forainetv2_oneformer3d_infos_test.pkl"
    out_las = out / f"{stem}.las"
    gpkg = out / f"{stem}_trees.gpkg"
    report_json = out / f"{stem}_report.json"
    report_md = out / f"{stem}_report.md"
    if timings is None:
        timings = {}

    def convert() -> None:
        out.mkdir(parents=True, exist_ok=True)
        las_to_ply(las, input_ply, sidecar, origin=origin, epsg=epsg)

    def prepare_inputs() -> None:
        scan_list.write_text(f"{stem}\n")
        empty_list.write_text("")
        # batch_load skips a scan whose _vert.npy already exists: drop stale exports
        # so a re-run really re-exports this tile.
        if instance_dir.is_dir():
            for stale in instance_dir.glob(f"{stem}_*.npy"):
                stale.unlink()

    def check_preprocess() -> None:
        # batch_load reports a failed export on stderr but create_data only PRINTS
        # "no test scans with preprocessed data, skipping ..." and writes nothing, so
        # without these checks a preprocessing miss would only surface hours later,
        # after the GPU step, as a missing result PLY.
        for path in (vert_npy, offsets_npy, info_pkl):
            if not path.is_file():
                raise RuntimeError(
                    f"preprocessing did not produce {path}; see the container output above"
                )
        if info_pkl.stat().st_mtime < scan_list.stat().st_mtime:
            raise RuntimeError(
                f"{info_pkl} is older than {scan_list}: create_data skipped this tile and "
                "a stale pkl would be used; see the container output above"
            )

    def georeference() -> None:
        results_to_las(result_ply, sidecar, offsets_npy, out_las)

    def trees() -> None:
        n = trees_to_gpkg(out_las, gpkg)
        print(f"{gpkg}: {n} trees")

    def report() -> None:
        # Imported lazily: geopandas is only needed for the reporting half.
        from ff3d_geo.report import build_report, report_markdown, write_report

        rep = build_report(out_las, gpkg, runtime_s=timings.get("inference"))
        write_report(rep, report_json, report_md)
        print(report_markdown(rep))

    preprocess_inner = (
        'bash -c "'
        "cd data/ForAINetV2 && python batch_load_ForAINetV2_data.py --unlabeled"
        f" --test_scan_names_file {_container_path(scan_list, repo, 'scan list')}"
        f" --train_scan_names_file {_container_path(empty_list, repo, 'empty list')}"
        f" --val_scan_names_file {_container_path(empty_list, repo, 'empty list')}"
        f" && cd {CONTAINER_ROOT} && python tools/create_data_forainetv2.py forainetv2"
        f" --test-list {_container_path(scan_list, repo, 'scan list')}"
        " --splits test"
        f" --out-dir {_container_path(out, repo, '--out')}"
        '"'
    )
    inference_inner = (
        f"python tools/test.py {config_rel} {checkpoint_rel} --work-dir {out_rel}"
        " --cfg-options test_dataloader.dataset.ann_file="
        f"{_container_path(info_pkl, repo, 'info pkl')}"
    )

    return [
        Step(name="las_to_ply", func=convert,
             name_detail=f"{las} -> {input_ply} (+ sidecar {sidecar})"),
        Step(name="prepare_inputs", func=prepare_inputs,
             name_detail=f"write {scan_list}, {empty_list}; rm {instance_dir}/{stem}_*.npy"),
        _docker_step("preprocess", preprocess_inner, repo, gpu),
        Step(name="check_preprocess", func=check_preprocess,
             name_detail=f"require {vert_npy}, {offsets_npy}, {info_pkl}"),
        _docker_step("inference", inference_inner, repo, gpu),
        Step(name="results_to_las", func=georeference,
             name_detail=f"{result_ply} + {offsets_npy} -> {out_las}"),
        Step(name="trees_to_gpkg", func=trees, name_detail=f"{out_las} -> {gpkg}"),
        Step(name="report", func=report, name_detail=f"-> {report_json}, {report_md}"),
    ]


def execute(steps: list[Step], dry_run: bool, timings: dict[str, float] | None = None) -> None:
    """Print each step; unless ``dry_run``, run it. Failures stop the run immediately."""
    for step in steps:
        print(step.render(), flush=True)
        if dry_run:
            continue
        started = time.monotonic()
        if step.argv is not None:
            env = dict(os.environ)
            env.update(step.env)
            try:
                subprocess.run(step.argv, cwd=str(step.cwd), env=env, check=True)
            except subprocess.CalledProcessError as exc:
                raise RuntimeError(
                    f"step {step.name!r} failed with exit {exc.returncode}: "
                    f"{step.render()}"
                ) from exc
        elif step.func is not None:
            step.func()
        if timings is not None:
            timings[step.name] = time.monotonic() - started


def _add_origin_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--origin", nargs=2, type=float, metavar=("E", "N"), default=None,
                        help="tile lower-left corner; default: parsed from the file name")
    parser.add_argument("--epsg", type=int, default=DEFAULT_EPSG)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m ff3d_geo",
        description="Georeferenced ForestFormer3D inference on ALS tiles (host side).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="convert, preprocess, infer, georeference, report")
    run.add_argument("--las", required=True, type=Path, help="input ALS tile (LAS/LAZ)")
    run.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT,
                     help="converted checkpoint, repo-relative (default: %(default)s)")
    run.add_argument("--out", type=Path, default=None,
                     help="output/work dir under the repo (default: work_dirs/tegel-<stem>)")
    run.add_argument("--config", default=DEFAULT_CONFIG, help="repo-relative model config")
    run.add_argument("--repo", type=Path, default=REPO_ROOT,
                     help="checkout to mount at /workspace (default: %(default)s)")
    run.add_argument("--gpu", default=os.environ.get("FF3D_GPU", DEFAULT_GPU),
                     help="physical GPU for ff3d_docker "
                          "(default: $FF3D_GPU, else %(default)s)")
    run.add_argument("--dry-run", action="store_true", help="print the plan, run nothing")
    _add_origin_args(run)

    conv = sub.add_parser("convert", help="LAS -> input PLY + georeferencing sidecar")
    conv.add_argument("--las", required=True, type=Path)
    conv.add_argument("--ply", required=True, type=Path)
    conv.add_argument("--sidecar", required=True, type=Path)
    _add_origin_args(conv)

    geo = sub.add_parser("georef", help="result PLY -> LAS 1.4 (recovery after inference)")
    geo.add_argument("--result-ply", required=True, type=Path)
    geo.add_argument("--sidecar", required=True, type=Path)
    geo.add_argument("--offsets", required=True, type=Path)
    geo.add_argument("--out-las", required=True, type=Path)
    geo.add_argument("--gpkg", type=Path, default=None,
                     help="also write the tree GeoPackage here")

    rep = sub.add_parser("report", help="result LAS -> tree GeoPackage + report JSON/markdown")
    rep.add_argument("--las", required=True, type=Path)
    rep.add_argument("--gpkg", required=True, type=Path,
                     help="tree GeoPackage; written from --las when it does not exist")
    rep.add_argument("--json", required=True, type=Path)
    rep.add_argument("--md", type=Path, default=None, help="default: --json with a .md suffix")
    rep.add_argument("--runtime-s", type=float, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.command == "run":
        origin = tuple(args.origin) if args.origin else None
        timings: dict[str, float] = {}
        steps = plan_run(
            args.las, args.checkpoint, args.out, origin=origin, epsg=args.epsg,
            config=args.config, repo=args.repo, gpu=args.gpu, timings=timings,
        )
        execute(steps, dry_run=args.dry_run, timings=timings)
        return 0

    if args.command == "convert":
        origin = tuple(args.origin) if args.origin else None
        sidecar = las_to_ply(args.las, args.ply, args.sidecar, origin=origin, epsg=args.epsg)
        print(f"wrote {args.ply} and {args.sidecar} ({sidecar['n_points']} points)")
        return 0

    if args.command == "georef":
        results_to_las(args.result_ply, args.sidecar, args.offsets, args.out_las)
        print(f"wrote {args.out_las}")
        if args.gpkg is not None:
            print(f"wrote {args.gpkg} ({trees_to_gpkg(args.out_las, args.gpkg)} trees)")
        return 0

    if args.command == "report":
        from ff3d_geo.report import build_report, report_markdown, write_report

        if not Path(args.gpkg).is_file():
            print(f"wrote {args.gpkg} ({trees_to_gpkg(args.las, args.gpkg)} trees)")
        md = args.md if args.md is not None else Path(args.json).with_suffix(".md")
        rep = build_report(args.las, args.gpkg, runtime_s=args.runtime_s)
        write_report(rep, args.json, md)
        print(report_markdown(rep))
        return 0

    raise AssertionError(args.command)  # pragma: no cover - argparse enforces the choices


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
