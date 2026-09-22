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
from functools import partial
from pathlib import Path, PurePosixPath
from typing import Callable

from ff3d_geo.convert import las_to_ply, results_to_las
from ff3d_geo.origin import parse_origin

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


def _run_per_tile(what: str, jobs: list[tuple[str, Callable[[], object]]]) -> None:
    """Run every tile's ``fn``, then report ALL the failures in one error.

    A batch has no resume: it shares one preprocess and one multi-hour GPU inference,
    so letting tile *k* abort the loop would cost tiles *k+1...N* their ``.las`` /
    ``_trees.gpkg`` / report even though their result PLYs are already on disk, and
    re-running would redo everything. Each tile is therefore attempted independently
    and only the ones that really failed are named.
    """
    failed: list[tuple[str, Exception]] = []
    for stem, fn in jobs:
        try:
            fn()
        except Exception as exc:  # noqa: BLE001 - re-raised below, with the stem named
            failed.append((stem, exc))
    if failed:
        detail = "".join(f"\n  {stem}: {type(exc).__name__}: {exc}" for stem, exc in failed)
        raise RuntimeError(
            f"{what} failed for {len(failed)} of {len(jobs)} tile(s); the other tiles "
            f"were written:{detail}"
        ) from failed[0][1]


def _docker_step(name: str, inner: str, repo: Path, gpu) -> Step:
    """A step that sources benchmark/common.sh and runs ``inner`` via ``ff3d_docker``.

    ``FF3D_DRY_RUN`` is forced to ``"0"`` in the step env: the CLI has its own
    ``--dry-run`` flag, and a caller who still has ``FF3D_DRY_RUN=1`` exported from
    unrelated benchmark work must not silently turn this step into a no-op.
    """
    return Step(
        name=name,
        argv=["bash", "-c", f"source benchmark/common.sh; ff3d_docker {inner}"],
        cwd=repo,
        env={"FF3D_ROOT": str(repo), "FF3D_GPU": str(gpu), "FF3D_DRY_RUN": "0"},
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

    ``las`` is one tile (path) or several (list of paths). Several tiles share ONE
    ``preprocess`` and ONE ``inference`` step: the scan list simply gets one line per
    tile, and ``tools/test.py`` writes one ``<stem>.ply`` per scan into ``--work-dir``.
    The host-side steps run per tile and keep the single-tile output names, so a batch
    of N tiles produces the same ``<out>/<stem>.{las,_trees.gpkg,_report.json,.md}`` as
    N separate runs would.

    ``checkpoint`` and ``config`` are conventionally repo-relative (``work_dirs/...``,
    ``configs/...``); absolute ones must still point inside ``repo``. ``out`` defaults
    to ``<repo>/work_dirs/tegel-<stem>`` for a single tile and is required for several.
    ``timings`` (step name -> seconds) is filled in by :func:`execute`; the report step
    reads ``timings["inference"]`` from it (split evenly over the tiles of a batch).
    """
    repo = Path(repo).resolve()
    las_paths = [Path(p).resolve()
                 for p in ([las] if isinstance(las, (str, Path)) else list(las))]
    if not las_paths:
        raise ValueError("--las needs at least one file")
    stems = [p.stem for p in las_paths]
    for stem in stems:
        if _UNSAFE_STEM.search(stem):
            raise ValueError(
                f"scan name {stem!r} ends in _<digits>, which the ForAINetV2 data tools "
                "treat as a block index and strip; rename the tile (e.g. ..._100m)"
            )
    if len(set(stems)) != len(stems):
        raise ValueError(
            "--las contains duplicate scan names; every tile of a batch shares one "
            f"scan list and one output dir, so the stems must be unique: {stems}"
        )
    if len(las_paths) > 1:
        # One origin cannot describe several tiles, and the default <out> is derived
        # from a single stem: both have to be explicit (or parsed) per tile instead.
        if origin is not None:
            raise ValueError(
                "--origin is only allowed with exactly one --las file; with several "
                "tiles each origin is parsed from its own file name"
            )
        if out is None:
            raise ValueError(
                "--out must be given when several --las files are batched (the default "
                "work_dirs/tegel-<stem> names a single tile)"
            )
    origins = [origin if origin is not None else parse_origin(p.name) for p in las_paths]
    out = _under_repo(Path(out) if out is not None else Path("work_dirs") / f"tegel-{stems[0]}",
                      repo, "--out")
    config_rel = _repo_relative(Path(config), repo, "--config")
    checkpoint_rel = _repo_relative(Path(checkpoint), repo, "--checkpoint")
    out_rel = _repo_relative(out, repo, "--out")

    data_dir = repo / "data" / "ForAINetV2"
    instance_dir = data_dir / "forainetv2_instance_data"
    input_plys = [data_dir / "test_data" / f"{s}.ply" for s in stems]
    scan_list = out / "scan_list.txt"
    empty_list = out / "empty_list.txt"
    sidecars = [out / f"{s}.sidecar.json" for s in stems]
    result_plys = [out / f"{s}.ply" for s in stems]
    offsets_npys = [instance_dir / f"{s}_offsets.npy" for s in stems]
    info_pkl = out / "forainetv2_oneformer3d_infos_test.pkl"
    # every tile's exports, each tile's pair kept together, plus the one shared pkl
    preprocess_artefacts = [
        instance_dir / f"{s}_{kind}.npy" for s in stems for kind in ("vert", "offsets")
    ] + [info_pkl]
    out_lass = [out / f"{s}.las" for s in stems]
    gpkgs = [out / f"{s}_trees.gpkg" for s in stems]
    report_jsons = [out / f"{s}_report.json" for s in stems]
    report_mds = [out / f"{s}_report.md" for s in stems]
    if timings is None:
        timings = {}

    def convert() -> None:
        out.mkdir(parents=True, exist_ok=True)
        for src, input_ply, sidecar, tile_origin in zip(
            las_paths, input_plys, sidecars, origins
        ):
            las_to_ply(src, input_ply, sidecar, origin=tile_origin, epsg=epsg)

    def prepare_inputs() -> None:
        # las_to_ply normally creates <out> first; mkdir here too so this step can
        # also be run on its own (e.g. to rebuild only the scan list).
        out.mkdir(parents=True, exist_ok=True)
        scan_list.write_text("".join(f"{s}\n" for s in stems))
        empty_list.write_text("")
        # batch_load skips a scan whose _vert.npy already exists: drop stale exports
        # so a re-run really re-exports this tile.
        if instance_dir.is_dir():
            for stem in stems:
                for stale in instance_dir.glob(f"{stem}_*.npy"):
                    stale.unlink()
        # A previous run's result PLY must not survive: if this run's inference exits
        # 0 without writing one (e.g. an empty pkl), results_to_las would otherwise
        # silently georeference the old result.
        for result_ply in result_plys:
            if result_ply.is_file():
                result_ply.unlink()

    def check_preprocess() -> None:
        # batch_load reports a failed export on stderr but create_data only PRINTS
        # "no test scans with preprocessed data, skipping ..." and writes nothing, so
        # without these checks a preprocessing miss would only surface hours later,
        # after the GPU step, as a missing result PLY.
        for path in preprocess_artefacts:
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
        _run_per_tile("results_to_las", [
            (stem, partial(results_to_las, result_ply, sidecar, offsets_npy, out_las))
            for stem, result_ply, sidecar, offsets_npy, out_las in zip(
                stems, result_plys, sidecars, offsets_npys, out_lass)
        ])

    def trees() -> None:
        # Imported lazily: geopandas is only needed for the reporting half.
        from ff3d_geo.trees import trees_to_gpkg

        def one(out_las: Path, gpkg: Path) -> None:
            print(f"{gpkg}: {trees_to_gpkg(out_las, gpkg)} trees")

        _run_per_tile("trees_to_gpkg", [
            (stem, partial(one, out_las, gpkg))
            for stem, out_las, gpkg in zip(stems, out_lass, gpkgs)
        ])

    def report() -> None:
        # Imported lazily: geopandas is only needed for the reporting half.
        from ff3d_geo.report import build_report, report_markdown, write_report

        # The whole batch went through one inference step, so charge each tile its
        # share of that step's runtime rather than the batch total.
        inference_s = timings.get("inference")
        per_tile_s = None if inference_s is None else inference_s / len(stems)

        def one(stem: str, out_las: Path, gpkg: Path, report_json: Path,
                report_md: Path) -> None:
            rep = build_report(out_las, gpkg, runtime_s=per_tile_s)
            write_report(rep, report_json, report_md)
            if len(stems) == 1:
                print(report_markdown(rep))
            else:
                # N tiles of full markdown would bury the run; one line each instead
                # (the per-tile markdown is still on disk as <stem>_report.md).
                usable = "yes" if rep["recommendation"]["first_pass_usable"] else "no"
                print(f"{stem}: {rep['n_trees']} trees, first pass usable: {usable}")

        _run_per_tile("report", [
            (stem, partial(one, stem, out_las, gpkg, report_json, report_md))
            for stem, out_las, gpkg, report_json, report_md in zip(
                stems, out_lass, gpkgs, report_jsons, report_mds)
        ])

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

    # A single tile keeps the one-line details it had before batching existed; a batch
    # would run to ~20 kB on one line, so its per-tile entries go one per indented line.
    one_tile = len(stems) == 1
    tiles = "; " if one_tile else "\n    "  # joins per-tile entries
    paths = ", " if one_tile else "\n    "  # joins a flat list of paths
    return [
        Step(name="las_to_ply", func=convert,
             name_detail=tiles.join(
                 f"{src} -> {input_ply} (+ sidecar {sidecar})"
                 for src, input_ply, sidecar in zip(las_paths, input_plys, sidecars))),
        Step(name="prepare_inputs", func=prepare_inputs,
             name_detail=f"write {scan_list}, {empty_list}; rm " + paths.join(
                 f"{instance_dir}/{s}_*.npy" for s in stems)),
        _docker_step("preprocess", preprocess_inner, repo, gpu),
        Step(name="check_preprocess", func=check_preprocess,
             name_detail="require " + paths.join(str(p) for p in preprocess_artefacts)),
        _docker_step("inference", inference_inner, repo, gpu),
        Step(name="results_to_las", func=georeference,
             name_detail=tiles.join(
                 f"{result_ply} + {offsets_npy} -> {out_las}"
                 for result_ply, offsets_npy, out_las in zip(
                     result_plys, offsets_npys, out_lass))),
        Step(name="trees_to_gpkg", func=trees,
             name_detail=tiles.join(f"{out_las} -> {gpkg}"
                                    for out_las, gpkg in zip(out_lass, gpkgs))),
        Step(name="report", func=report,
             name_detail="-> " + tiles.join(
                 f"{report_json}, {report_md}"
                 for report_json, report_md in zip(report_jsons, report_mds))),
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
    run.add_argument("--las", required=True, type=Path, nargs="+",
                     help="input ALS tile(s) (LAS/LAZ); several sub-tiles share one "
                          "preprocess and one inference step and then need --out")
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
        # A non-dry run with a wrong --las would otherwise create --out (prepare_inputs)
        # and only then die inside laspy.read with a bare FileNotFoundError; check
        # up front, before any step runs, for a clear error instead. --dry-run only
        # prints the plan and is allowed to run ahead of the input file existing.
        if not args.dry_run:
            for path in args.las:
                if not Path(path).is_file():
                    raise ValueError(f"--las {path} does not exist")
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
            from ff3d_geo.trees import trees_to_gpkg

            print(f"wrote {args.gpkg} ({trees_to_gpkg(args.out_las, args.gpkg)} trees)")
        return 0

    if args.command == "report":
        from ff3d_geo.report import build_report, report_markdown, write_report
        from ff3d_geo.trees import trees_to_gpkg

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
