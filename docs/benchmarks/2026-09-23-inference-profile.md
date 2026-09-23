# Inference profile: full-plot prediction on a 100 m ALS sub-tile

Why does `ForAINetV2OneFormer3D_XAwarequery` full-plot inference take 53-60 s per
100 m Berlin ALS sub-tile (~275 k points) on an H100? Measured on 2026-09-23; one scan,
GPU 7 of `carrot`. Verdict up front: **81 % of the prediction time is a Python
per-mask loop in `pred_inst_sem_test` that issues ~590 blocking `.item()` GPU syncs
per cylinder, times 625 cylinders per tile.** The network itself (sparse UNet + query
decoder) is about 8 % of the time. A ten-line vectorisation of that loop, verified to
give identical results, should cut the per-tile time by roughly 4x.

## Environment

| | |
|---|---|
| Host | `carrot`, 224 cores, 2 TB RAM, 8x NVIDIA H100 80GB HBM3 |
| Container | `forestformer3d:cu118` (torch 2.0.1+cu118, CUDA 11.8), repo mounted at `/workspace` via `benchmark/common.sh`'s `ff3d_docker` |
| Commit | `0ce86b9` on `fix/review-findings`, config `configs/oneformer3d_qs_radius16_qp300_2many.py` (radius 16 m, voxel 0.2 m, 300 query points, `score_th` 0.4, `overlap_threshold` 0.3), checkpoint `work_dirs/clean_forestformer/epoch_3000_fix.pth` |
| Scan | `3dm_33_381_5829_E381000_N5829000_100m` (275,684 points, 99.99 x 99.99 m, z 0-39.8 m), one-scan info pkl built from `work_dirs/berlin-3dm_33_381_5829_1_be/forainetv2_oneformer3d_infos_test.pkl` |
| Reference | The production run of this very sub-tile reported `runtime_s = 53.1` (`3dm_33_381_5829_E381000_N5829000_100m_report.json`; that is the batch `tools/test.py` step's wall time divided by 100 tiles) |

**Caveat on absolute numbers.** GPU 7 was running one production Berlin queue during
every measurement (`nvidia-smi` showed the GPU at 93-99 % utilisation with a second
compute process resident). My one-scan run therefore took 219-235 s instead of the
53 s production measured for the same tile on a less loaded GPU: a sync-bound loop
suffers disproportionately when another process keeps the GPU queue full. The
*shares* between phases are what this report relies on; where I convert them to
seconds at production speed, that is an estimate and marked as such. Scratch files
(logs, `one.prof`, `pstats.txt`, `phases.json`, driver script) are in
`work_dirs/logs/profile/` on carrot.

A note on the task's premise: the "Processing regions: N/81" progress bar belongs to
the older `ForAINetV2OneFormer3D.predict` (`oneformer3d/oneformer3d.py:744`), which the
active model does not run. With `radius = 16` and `step = radius/4 = 4` the active
`_predict_full_plot` lays a 25 x 25 lattice over a 100 m tile: **625 cylinders, not 81**.

## 1. Whole-process wall time, one scan

| Measurement | Seconds |
|---|---|
| `tools/test.py` wall time, one scan (two runs) | 235.6 / 232.3 |
| of which mmengine `Epoch(test) time:` (the `predict` call) | 219.3 / 216.0 |
| `docker run` + `import oneformer3d, torch` + CUDA init, nothing else | 14.4 |
| `Runner.from_cfg` + `load_checkpoint` + dataloader first batch (driver) | 1.7 + 0.3 + 0.1 |
| `UnifiedSegMetric.compute_metrics` for one unlabeled scan | 0.38 |
| `UnifiedSegMetric.process` | 0.00 |

Per-process overhead (container start, imports, 9.1 s of `Config.fromfile` that is
really `import oneformer3d`, model build, checkpoint) is ~16 s per `tools/test.py`
invocation. Amortised over the 100 scans of a batch that is 0.16 s per tile: the
pipeline already runs many scans per process, so this is not where the time goes.

## 2. cProfile (top by cumulative and by self time, trimmed)

`python -m cProfile -o one.prof tools/test.py ...` (this run took 376.9 s in-process
under the same contention; shares are what matter).

| cumtime s | tottime s | ncalls | file:function |
|---:|---:|---:|---|
| 361.4 | 2.7 | 1 | `oneformer3d/oneformer3d.py:2132 _predict_full_plot` |
| 282.3 | 0.4 | 625 | `oneformer3d/oneformer3d.py:2279 predict_by_feat_test` |
| **280.9** | **105.7** | 625 | `oneformer3d/oneformer3d.py:2456 pred_inst_sem_test` |
| **172.3** | **172.3** | 371,398 | `{method 'item' of torch._C._TensorBase}` (365,963 of them from `pred_inst_sem_test`) |
| 20.2 | 0.0 | 625 | `oneformer3d/query_decoder.py:768 forward` (decoder) |
| 19.5 | 0.4 | 625 | `oneformer3d/oneformer3d.py:1829 extract_feat` (spconv UNet) |
| 14.3 | 14.3 | 6,250 | `{built-in method torch.where}` (10 per tile; each is a sync) |
| 13.7 | 0.4 | 625 | `torch_cluster/fps.py:19 fps` (10.1 s inside the CUDA op) |
| 12.1 | 0.0 | - | `importlib exec_module` (imports at start-up) |
| 10.3 | 0.0 | 1 | `oneformer3d/oneformer3d.py:2893 save_ply_withscore` |
| 9.9 | 1.5 | 1 | `plyfile.py:635 _write_txt` (ASCII PLY, `numpy.savetxt` over 275,684 rows: 5.4 s; plus the per-row Python tuple comprehension) |
| 9.1 | 0.0 | 1 | `mmengine/config/config.py:160 fromfile` (= `import oneformer3d`, 8.9 s) |
| 4.1 | 4.0 | 8,125 | `spconv ... sort_1d_by_key_allocator` |
| 3.4 | 3.4 | 2,547 | `{built-in method torch.tensor}` |
| 3.1 | 3.1 | 9,185 | `{method 'cpu' of torch._C._TensorBase}` |
| 1.8 | 1.8 | 195,627 | `{method 'sum' of torch._C._TensorBase}` (190,625 from `pred_inst_sem_test`) |
| 0.4 | 0.4 | 1 | `oneformer3d/unified_metric.py:38 compute_metrics` |

Callees of `pred_inst_sem_test` (625 calls): `.item()` 365,963 calls / 172.0 s,
`.sum()` 190,625 / 1.7 s, `.min()` 178,463 / 0.8 s, `mask_matrix_nms` 625 / 0.22 s.
Its 105.7 s of self time is the boolean indexing `coordinates[mask, 2]` (a `nonzero`,
which also synchronises) and `scores[i] = 0` scalar writes inside the same loop; NMS is
negligible. `merge_instances_by_score` and `SemanticVotes` do not appear in the top
list at all (0.06 s and 0.45 s, see the phase table).

## 3. Phase breakdown (driver with `cuda.synchronize()` around each phase)

`work_dirs/logs/profile/phase_driver.py` rebuilds the runner, loads the checkpoint,
takes the one sample from the test dataloader and re-runs the body of
`_predict_full_plot` with timers. Repeat 2 of 2 (repeat 1 within 1 %); 625 regions,
all non-empty, decoder ran in all 625.

| Phase | Total s | Share | Per region ms | Calls |
|---|---:|---:|---:|---:|
| `predict_by_feat_test` (`pred_sem` + `pred_inst_sem_test` + `.cpu().numpy()`) | **281.0** | **81.0 %** | 450 | 625 |
| `fps` (torch_cluster, ~10 k tree points to 300 queries) | 16.5 | 4.8 % | 26 | 625 |
| decoder (`ForAINetv2QueryDecoder_XAwarequery`) | 13.3 | 3.8 % | 21 | 625 |
| backbone (`SparseConvTensor` + `extract_feat`) | 12.4 | 3.6 % | 20 | 625 |
| mask collect (edge test, `keep.tolist()`, one `.cpu()` per kept mask) | 5.7 | 1.6 % | 9 | 625 |
| `save_ply_withscore` (CPU only, ASCII PLY) | 5.6 | 1.6 % | - | 1 |
| `collate` (MinkowskiEngine `batch_sparse_collate`) | 4.3 | 1.2 % | 7 | 625 |
| `cdist` nearest neighbour pc1 -> pc3 | 3.4 | 1.0 % | 5 | 625 |
| `grid_sample` | 1.9 | 0.5 % | 3 | 625 |
| `torch.cuda.empty_cache()` per region | 0.8 | 0.2 % | 1 | 625 |
| heads (`Embed`, `BiSemantic`, argmax) | 0.7 | 0.2 % | 1 | 625 |
| crop (`torch.where(region_mask)`) | 0.5 | 0.2 % | 1 | 625 |
| `SemanticVotes.add` | 0.45 | 0.1 % | 1 | 625 |
| `merge_instances_by_score` (4,841 index-list masks -> 399 kept) | 0.06 | 0.0 % | - | 1 |
| relabel / small-instance filter / `votes.resolve` / region lattice | 0.02 | 0.0 % | - | - |
| **Total `predict`** | **347.1** | | **555** | |

The same driver with the evaluator: `process` 0.00 s, `compute_metrics` 0.38 s for
the scan.

**GPU compute vs host work.** Kernel-bound phases (backbone, decoder, heads, fps,
collate, cdist, grid_sample) sum to 52.5 s = 15 %. `save_ply` is pure CPU, 1.6 %.
The remaining 81 % in `predict_by_feat_test` is neither: it is host-device
round-trip latency. `pred_inst_sem_test` runs `for i in range(mask_pred.size(0))`
over the ~246 candidate masks per tile, and per mask does `mask.sum().item()`, a
boolean index, `z.numel()` and `z.min().item()`: about 590 blocking syncs per tile,
370 k per scan, while the GPU does almost nothing in between. That is also why
production logs show only 50-60 % GPU utilisation with one process per GPU: the
queue is empty most of the time.

Micro-benchmark on GPU 7 at the real tile size (K = 246 masks, N = 18,278 voxels,
same contention): the loop costs **344-351 ms per tile**; the vectorised equivalent
(`zmin = torch.where(mask, z[None], inf).min(1).values; scores[(n==0)|(zmin > gz+5)] = 0`)
costs **0.1-10.6 ms** and returns bit-identical scores (`torch.equal` True). 625 x
0.345 s = 216 s, consistent with the 281 s measured for the whole function.

Everything the task suspected on the CPU side is cheap: `sample_region` never
triggers (max 27,263 voxels per tile vs the 640,000 cap), `merge_instances_by_score`
works on index lists (no dense (K, N) mask) and takes 60 ms, `SemanticVotes` 0.5 s,
`compute_metrics` 0.4 s per scan.

## 4. Region count and overlap

`generate_cylindrical_regions(points_xy, radius=16, step=radius/4=4)` puts centres
at `min + k*step` for `k = 0..floor(extent/step)`: 25 per axis for this 99.99 m
tile, 625 cylinders (a tile with extent exactly 100 m gets 26 x 26 = 676). The
ideal interior overlap is `pi*16^2 / 4^2 = 50.3` passes per point; measured on this
scan (edge effects included):

| step | regions | mean passes per point | min / max | points fed to the network | measured `predict` s |
|---:|---:|---:|---|---:|---:|
| 4.00 (radius/4, current) | 625 | **43.9** (41.4 after 0.2 m voxel grid) | 8 / 52 | 12.1 M (of 275,684) | 347.1 |
| 8.00 (radius/2) | 169 | 11.4 (10.8) | 3 / 14 | 3.15 M | 100.6 |
| 10.67 (radius/1.5) | 100 | 6.6 | 3 / 9 | - | - |
| 16.00 (radius/1) | 49 | 3.0 | 1 / 4 | - | - |

Per cylinder: 19,356 raw points on average (max 29,047), 18,278 after the 0.2 m
grid (ALS at ~27 pts/m^2 is sparser than one point per 0.2 m voxel, so the grid
barely reduces anything). Per cylinder the decoder produces ~246 candidate masks,
of which only **7.7 on average survive** `score > 0.4` and the "does not touch the
cylinder edge" test; 4,841 masks go into the merge and 399 come out (397 final
instances after the ground/size filters). So each tree is typically predicted
~12 times and the best-scoring copy wins.

At `step = radius/2` the merge kept 341 masks / 340 final instances (-14 %) on this
unlabeled tile. Whether those are lost trees or lost duplicates cannot be decided
without ground truth; see option 2.

## 5. GPU utilisation

`nvidia-smi -i 7 --query-gpu=utilization.gpu -l 1` during the runs: mean 93.3 % /
94.0 % (baseline runs, 236 / 233 samples), 98.8 % during the driver. These numbers
are dominated by the co-resident production process on GPU 7 and cannot be
attributed to my run, whose own footprint was ~1.3 GB. Production's own logs (one
process per GPU, no co-tenant) reported 50-60 %; given the sync-bound loop above,
that figure reflects an idle-most-of-the-time queue kept busy by many tiny kernels
rather than saturated compute.

## Ranked speed-up options

Estimated gains are relative to the current production 53 s per tile; "measured"
means measured here, "est." means extrapolated.

1. **Vectorise the z-filter loop in `pred_inst_sem_test`** (`oneformer3d/oneformer3d.py:2456`,
   the `for i in range(mask_pred.size(0))` block) and drop the `.item()` calls.
   Measured 345 ms -> <1 ms per tile with identical output; that block is ~80 % of
   `predict`. **Est. 53 s -> 10-15 s per tile (3.5-5x).** Risk: very low (ten lines,
   results identical; add a unit test comparing both forms). The remaining per-tile
   syncs (`torch.where` x 10, `keep.tolist()`, `ground_z_max.item()`, `.cpu()` per
   kept mask) cost ~20 s under contention and are the next tier once the loop is
   gone; batching the kept masks into one `.cpu()` per tile removes most of them.
2. **Larger lattice step** (`step_size = self.radius / 4` in `_predict_full_plot`).
   `radius/2` cuts regions 625 -> 169 and every per-region cost by 3.7x (measured
   347 -> 101 s). After option 1 the per-region cost is dominated by the network, so
   this becomes the biggest remaining lever: **est. 10-15 s -> 3-5 s per tile.** Risk:
   medium, quality. On this tile the merge produced 14 % fewer instances; every point
   is still covered by >= 3 cylinders and any tree of crown radius < ~10 m still has a
   cylinder that contains it whole, so the loss is probably duplicate suppression
   rather than misses, but it must be verified with `tools/final_eval.py` on the
   labelled ForAINetV2 test split (F1, MUCov, PQ) before changing the default.
   `radius/3` (361 regions, 1.7x fewer) is the conservative middle step. Make the
   divisor a `test_cfg` knob so the two can be compared without code edits.
3. **Binary PLY output** in `save_ply_withscore` (`text=True` plus a per-row Python
   tuple comprehension over 275,684 rows): 5.6-10.3 s per tile, CPU-only, so it does
   not shrink with GPU fixes and becomes the second-largest item after option 1.
   Build the structured array with `np.rec.fromarrays` / field assignment and write
   `PlyData(..., text=False)`: **est. -5 to -9 s per tile.** Risk: low; every consumer
   (`ff3d_geo/convert.py`, `tools/merge_prediction.py`, `tools/final_eval.py`) reads
   with `plyfile.PlyData.read`, which auto-detects binary; `tools/plyutils.read_ply`
   is binary-only already.
4. **Two processes per GPU** (today). Because the loop is host-latency bound, a second
   process fills the idle queue: **est. 1.7-1.9x throughput per GPU now**, at ~1.3-5 GB
   memory per process. Risk: low. After option 1 the run is ~50 % GPU kernels and the
   gain drops to est. 1.2-1.4x; after options 1+2 it is mostly gone. Prefer fixing
   the loop; use this only as a stop-gap for runs launched before the fix lands.
5. **Batch several cylinders per forward pass.** `collate`/`extract_feat` already carry
   a batch index and the decoder takes a list of per-sample queries, so 4-8 tiles per
   `SparseConvTensor` is a contained change. Network time is 27 s of 347 here (8 %),
   i.e. est. ~4 s of the current 53 s; small 18 k-voxel, 32-channel tiles use an H100
   poorly, so batching could roughly halve that: **est. -2 to -3 s per tile**, only
   worth it after options 1-2 when it is a larger share. Risk: medium (memory scales
   with batch, `predict_by_feat_test`/fps stay per-sample, ground-height logic is per
   tile).
6. **fps input size**: `torch_cluster.fps` over all ~10 k tree-labelled voxels costs
   22 ms per tile (13.7 s, 4 %). Pre-subsampling to e.g. 4 k points before fps would
   halve it (est. -0.5 s per tile at production speed) but changes which query points
   are chosen. Risk: low-medium; needs an F1 check. Low priority.
7. **Skip the mmengine evaluator for unlabeled runs**: `compute_metrics` is 0.38 s per
   scan (its Python per-point loops are only ~1.4 us per point), i.e. ~40 s once at the
   end of a 100-scan process, 0.7 % of the time. Worth a `test_evaluator` override in
   `ff3d_geo` for tidiness, not for speed.
8. **Per-process overhead** (~16 s: container, imports, model build) is 0.16 s per tile
   at 100 scans per process. Nothing to gain; keep batching scans per process.
9. **Not worth touching**: `merge_instances_by_score` (60 ms; index lists, no dense
   masks), `SemanticVotes` (0.5 s), `cdist` NN (3.4 s, chunked), `grid_sample` (1.9 s),
   `sample_region` (never active on ALS), `torch.cuda.empty_cache()` per region
   (0.8 s; can go, but it is 0.2 %).

Expected end state after options 1-3 (est.): ~5-8 s per 100 m tile at `radius/4`,
~3-5 s at `radius/2` if the F1 check passes, i.e. 7-15x over today's 53 s.

## Reproduction

On carrot (GPU 7; adjust `FF3D_GPU`), inside `work_dirs/logs/profile/`:

```bash
source benchmark/common.sh; export FF3D_GPU=7
ff3d_docker python tools/test.py configs/oneformer3d_qs_radius16_qp300_2many.py \
  work_dirs/clean_forestformer/epoch_3000_fix.pth --work-dir work_dirs/logs/profile/out \
  --cfg-options test_dataloader.dataset.ann_file=/workspace/work_dirs/logs/profile/one.pkl
ff3d_docker python -m cProfile -o /workspace/work_dirs/logs/profile/one.prof tools/test.py ...   # same args
ff3d_docker python work_dirs/logs/profile/phase_driver.py --repeat 2 --step-div 4
ff3d_docker python work_dirs/logs/profile/phase_driver.py --repeat 1 --step-div 2 \
  --out /workspace/work_dirs/logs/profile/phases_div2.json
```

`one.pkl` is the production info pkl with `data_list` cut to the one scan
(`mmengine.load` / `mmengine.dump`). Note that `docker run` without `-i` does not
forward stdin, so heredoc scripts must go through a file or `python -c`.
