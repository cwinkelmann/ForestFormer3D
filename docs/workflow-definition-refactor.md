# The workflow definition should not live inside either environment

Design note, 20 October 2026, written after running the pipeline on a host without the
Docker image (olive, two RTX A6000, via the GPU booking tools). Nothing here argues with
the two environments; it argues with where the pipeline that spans them is defined.

## The two environments are right

The model half is pinned to **numpy 1.24** (numba 0.57, mmdet3d) and the geospatial wheels
want **numpy 2** — measured on olive: the image reports `numpy 1.24.1 | numba 0.57.0 |
mmdet3d 1.1.0`, the geo venv `numpy 2.4.6 | pyproj 3.7.2 | geopandas 1.2.0`. There is no
overlap, so the split is arithmetic rather than taste, and `CLAUDE.md`'s rule is the thing
that makes it workable:

> The GPU steps never import `ff3d_geo`, and `ff3d_geo` never imports torch: the two halves
> talk through files only.

That rule should stay. The image deliberately has no `pyproj`, which is why a geo step
inside it fails — correctly — with `ModuleNotFoundError`.

## What is odd: the orchestrator is a member of one half

`ff3d_geo/cli.py` defines the eight-step pipeline **and** lives in the geo environment,
and it reaches the model half by shelling out: `_docker_step` (cli.py:182) builds
`bash -c "source benchmark/common.sh; ff3d_docker <inner>"`, with `preprocess_inner`
(cli.py:493) and `inference_inner` (cli.py:505) as strings of container-side paths.

So the pipeline definition imports pyproj, and it knows that the other half is reached
through Docker. Three consequences, each hit on 20 October:

1. **Running without the image means bypassing `run` entirely.** On a host with two conda
   environments and no image, the supported entry point cannot be used; the five per-tile
   commands had to be re-derived from `preprocess_inner` and `inference_inner` and issued
   by hand. The pipeline was reimplemented in order to run it.
2. **Steps are grouped by environment, not by what they do.** `convert` is data
   preparation that needs laspy, so it is a geo step; `preprocess` is data preparation
   that needs `segmentator`, so it is a model step. A packaging fact has become a
   structural one, and the seam is in the middle of one task.
3. **Granularity is fixed at the wrong level.** `run` issues one `tools/test.py` for a
   whole km tile, so one bad sub-tile takes the batch with it — `379_5826` lost 62 of 97
   sub-tiles that way (`docs/benchmarks/2026-09-22-tegel-berlin-2021.md`), and the recovery
   was a chunking script. Anything that interrupts the batch (a `ValueError` in spconv, an
   OOM, a preempted GPU booking) costs the same.

## The shape that fixes all three

A third layer that depends on **neither** half: the workflow as a table of steps, each
tagged with where it runs, driven by stdlib only.

```python
# depends on nothing: no torch, no pyproj, no Docker
def steps_for(cfg, unit):
    return [
        ("convert",        "geo", host_argv…, container_inner…),
        ("preprocess",     "gpu", host_argv…, container_inner…),
        ("create_data",    "gpu", host_argv…, container_inner…),
        ("inference",      "gpu", host_argv…, container_inner…),
        ("results_to_las", "geo", host_argv…, container_inner…),
    ]
```

The driver picks a runner per tag — `geo` → the host venv, `gpu` → the image, or a second
conda env, or `srun` — and runs one unit at a time. Then:

* the same definition runs under Docker, under two conda environments, and under a
  scheduler, without being rewritten;
* a bad unit costs one unit, because dispatch is per unit rather than per batch;
* `ff3d_geo run` becomes a thin wrapper over the table, which is also what a booking
  worker, a Slurm array and a plain shell loop consume;
* the environments do not change at all, and neither does the file-based contract.

A working version of exactly this exists, written by accident while getting the pipeline
to run on olive: `demo/ff3d_steps.py` in `cwinkelmann/hnee_slurm_deployment` (steps tagged
`geo`/`gpu`, one `process_one_tile`, a `docker_step` helper that also returns the command
that *stops* the step). It drove 100 sub-tiles of `3dm_33_381_5828_1_be` through a booked
GPU share on 20 October, surviving six preemptions at a cost of one sub-tile each.

## What to keep when refactoring

* The file-based contract between the halves, and the rule that neither imports the other.
* `--dry-run` printing the plan: with a step table it becomes the table, which is better.
* The batched form (`--las sub/*.las` → one preprocess and one `tools/test.py`): it is
  worth **2.90x** (2 h 54 → 1 h 00 for a km tile, `docs/benchmarks/2026-09-23-inference-profile.md`).
  Per-unit dispatch must stay *optional*, chosen by the driver, not forced. A sensible
  default is batches of ~20 units: the September recovery script found that size by hand.
* `docker/entrypoint.sh`'s CUDA assertion. It is what makes a geo step inside the image
  fail loudly instead of subtly.

## What this is not

Not an argument for one environment. Pinning the geo extras down to numpy 1.24
(geopandas ~0.14, pyproj ~3.6, rasterio ~1.3) would work and would make every geospatial
upgrade a risk to the model stack — real coupling, to save one venv.
