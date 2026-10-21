# If the stack ever has to move off torch 2.0.1

Written 2026-10-21, after the measurements in `docs/known-issues.md` ("The torch 2.0.1 pin
stays"). That section says *why we are not doing it*. This one says *how it would be done*,
so that the day a new GPU or a new dependency forces the question, nobody has to rediscover
the shape of the problem.

Short answer: there is a reasonable way, but "upgrade the dependencies" is not it. The
OpenMMLab stack this repo is built on has been unmaintained since early 2024, so above
torch 2.1 you stop *using* a framework and start *maintaining a fork* of a dead one. The
three routes below are genuinely different projects with different payoffs.

## What is actually available (checked against the indexes on 2026-10-21)

| package | latest | last release | prebuilt ceiling |
|---|---|---|---|
| torch | 2.14.1 | 2026-09-30 | -- we are on 2.0.1 (2023) |
| **spconv** (the backbone) | 2.3.8 | 2024-12-15 | `spconv-cu118`, `spconv-cu126`; **no cu128** |
| **MinkowskiEngine** | 0.5.4 | **2021-05-21** | torch < 2.1; dead upstream |
| mmcv | 2.2.0 | 2024-04-24 | OpenMMLab wheels stop at **torch 2.1** |
| mmcv-lite (same, no compiled ops) | 2.2.0 | 2024-04-24 | pure Python, any torch |
| mmdet3d | 1.4.0 | 2024-01-08 | declares `mmcv >=2.0.0rc4, <2.2.0` |
| mmdet | 3.3.0 | 2024-01-05 | -- |
| mmengine | 0.10.7 | 2025-03-04 | pure Python |
| torch-points-kernels | 0.7.0 | **2021-03-16** | source build only |
| torch-scatter / torch-cluster | 2.1.2 / 1.6.3 | 2023 | **wheels rebuilt per torch on data.pyg.org, through torch 2.14** |

Two surprises in that table. `torch-scatter`/`torch-cluster` look abandoned on PyPI but are
not: the PyG wheel index carries cp310 builds for every torch up to 2.14, so they cost
nothing. And **spconv, not MinkowskiEngine, is now the binding constraint** -- it is the
actual backbone (`SpConvUNet`), its last release is Dec 2024, and there is no cu128 wheel.
Its wheels are keyed by CUDA only, with no torch version in the name, which suggests it does
not link libtorch and would tolerate a newer torch on the same CUDA; **that assumption is
untested and is the first thing to verify.**

## How tightly we are actually coupled

10,105 lines in `oneformer3d/`. Every import of the framework, counted:

- **13** x `mmdet3d.registry` -- the `@MODELS.register_module()` decorators. Plumbing.
- **1** x `mmcv` in our entire codebase: `BaseTransform` in `transforms_3d.py`, an abstract
  class with one method.
- **1** x `MinkowskiEngine`, **1** x `torch_cluster`, **1** x `torch_points_kernels`,
  **3** x `torch_scatter`, **3** x `spconv.pytorch`.
- The rest are subclass bases and containers: `ScanNetDataset`, `Det3DDataPreprocessor`,
  `Base3DDetector`, `PointData`, `Det3DDataSample`, `BaseInstance3DBoxes`, `BasePoints`,
  `LoadAnnotations3D`, `PointSample`, `Pack3DDetInputs`.

Nothing in that list is mathematically deep. The framework supplies a registry, a config
loader, data containers and a training Runner -- not the model. That is what makes route C
below thinkable.

## Route A -- the cheap unpin: torch 2.9 + CUDA 12.6

The highest torch that every compiled dependency can still be *bought* rather than built.

1. Delete MinkowskiEngine. Replace `collate` (`oneformer3d.py:1868-1884`) with the
   packed-int64 `torch.unique` + `scatter_mean` groupby measured in `docs/known-issues.md`
   -- identical voxels to 7.6e-06, 2.5-4x faster on the card, ~30 lines.
2. `spconv-cu126`, torch 2.9.0+cu126, `torch-scatter`/`torch-cluster` from
   `data.pyg.org/whl/torch-2.9.0+cu126.html`.
3. mmdet3d 1.1.0 -> 1.4.0 (widens the box to `mmcv < 2.2.0`), mmengine 0.10.7, mmdet 3.3.0.
4. mmcv 2.1.0 **built from source** against torch 2.9 -- or, if the mmcv-lite test below
   passes, `mmcv-lite` and no compilation at all.
5. Make `torch_points_kernels` a lazy import (`panoptic_losses.py:3`); it is import-time
   only on our path and its last release is 2021.

Unknowns, in the order they would bite: does spconv-cu126 really tolerate torch 2.9; does
`import oneformer3d` survive mmcv-lite (mmdet3d imports `mmcv.ops` at module level in
places, and mmcv-lite ships no `mmcv.ops` at all -- the 20-line `_ext` stub that passed the
smoke gate does **not** prove this, because full mmcv was installed under it); does
mmdet3d 1.4.0 still expose the classes we subclass under the same paths; and
`torch.load`'s `weights_only=True` default from torch 2.6, which breaks naive loading of
mmengine checkpoints.

## Route B -- current torch (2.14)

Route A plus building **spconv + cumm** from source for cu128/cu130, and mmcv 2.1 from
source against a torch five major versions past anything it was tested on. You would own
both builds forever. Only worth it if hardware forces it -- Blackwell-class cards need
CUDA 12.8+, which is exactly where spconv's wheels stop.

## Route C -- leave the framework instead of upgrading it

Given the coupling above: keep torch, spconv, torch-scatter, torch-cluster; drop mmengine,
mmdet, mmdet3d and mmcv entirely. Replace the registry with direct construction, the config
with a dataclass, the Runner with a plain training loop, and the containers with the few
fields we use. The model, the criterion, the decoder and the tiling are already ours.

This is the only route that ends with a stack that can be upgraded again next year. It is
also the only one that touches training, so it must be gated on reproducing a training run,
not just inference. Estimate: days, not hours -- and the Runner replacement (hooks, the
epoch counter the model reads from `MessageHub`, validation, checkpoint format) is most of
it.

## What this costs regardless of route

Re-validation, which is larger than the port. The released checkpoint
(`epoch_3000_fix.pth`) was trained under this stack; a changed voxeliser, a changed
attention kernel or a changed random stream all move the outputs a little. The gate is the
28-plot benchmark against the nondeterminism noise floor in
`.claude/skills/ff3d-evaluation/SKILL.md`, not `docker/smoke.sh`. Budget a full benchmark
run and a km tile per route, in a **new image built alongside the old one** -- the old image
keeps working and is the control.

## The decision rule

Do none of this for speed. The 2.90x already banked came from scheduling, the remaining
time is spread 7-17 % over five stages, and `torch.compile` cannot reach the sparse
convolutions (`docs/benchmarks/2026-09-23-inference-profile.md`). A newer torch is worth
single-digit percent here.

Do it when one of these is true:

- **A new card.** Blackwell or newer will not run cu118 cubins, and the image already
  carries no PTX to JIT from (`docs/inference-pipeline.md`).
- **Retraining at scale** (the WINMOL project), where training throughput, not inference,
  pays for the work -- and where route C's training loop would be rewritten anyway.
- **A dependency we need arrives torch-2.x-only**, which is the usual way this becomes
  urgent without warning.

Until then the pin is the cheap option, and the reason is maintenance cost, not
impossibility.
