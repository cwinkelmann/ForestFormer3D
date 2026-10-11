# What one km tile costs: measured throughput of the three methods

Every runtime figure in this chapter is measured on carrot (8 x H100 80 GB, 224 cores) over
the Berlin ALS 2021 mosaic: 57 km tiles, 1.004 billion points, **17.6 M points per tile on
average** but 3.6 M to 27.1 M in range. That spread is why the comparable unit is **seconds
per million source points**, not seconds per tile or per 100 m sub-tile -- the same method
on the lake tile and on a dense forest tile differs by 7x in wall time and not at all in
cost per point.

Two kinds of number appear below and they must not be mixed:

* **Clean**: one process, one card, nothing else of ours on the host
  (`benchmark/speed_benchmark.sh`). This is the cost of the method.
* **As run**: the production configuration, several queues per card and per host. This is
  the cost of the mosaic, and it is what a schedule should be built from.

## Clean: one process, one card, quiet host

Two tiles (25.2 M and 15.1 M points), GPU 4 idle, 2026-10-08/09. Both deep methods are
linear in the point count to the third digit, which is what makes the per-point column
trustworthy.

| method | s / M points | a mean 17.6 M tile | a 25 M tile | GPU util mean / max | GPU memory |
| :--- | ---: | ---: | ---: | ---: | ---: |
| ForestFormer3D, `region_batch=8` + `batched_decoder` | **143** | **0.70 h** | 1 h 00 | 79.8 % / 100 % | 16.2 GB |
| ForestFormer3D, before the batching (`b0570a1`) | 414 | 2.02 h | 2 h 54 | 21.4 % / 87 % | 4.2 GB |
| SegmentAnyTree, one container | 164 | 0.80 h | 1 h 09 | 24-28 % / 43-47 % | 2.7-3.6 GB |
| AMS3D, 48 CPU workers | 21-69 | 0.1-0.34 h | 29 min | -- | -- |

AMS3D is the one method that is *not* point-linear: its cost follows the number of crowns it
has to cluster (69 s/M on a tile with 38.3 k trees against 21 s/M on one with 16.2 k).

## As run: the pre-inference-mask re-runs of 2026-10-09/11

All three methods over the same ALKIS-premasked split (5.75 % of the haloed points held
back). "Per queue" is what one queue needed for one tile; "per card" divides that by the
queues sharing the card, which is the throughput a schedule sees.

| method | configuration | tiles | median per queue | s / M points, per queue | s / M points, per card | wall clock for its share |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| ForestFormer3D | 2 queues x 7 cards | 57 | 87 min | 300 | **150** | 6 h 49 |
| SegmentAnyTree | 2 queues x 3 cards | 16 | 100 min | 491 | 245 | ~8 h |
| SegmentAnyTree | 3 queues x 1 card | 31 | 174 min | 599 | **200** | 30 h 23 |
| AMS3D | 4 queues x 32 workers | 57 | 16 min | **58** | -- (CPU) | 4 h 47 |

Read against the clean column, this is the whole oversubscription story:

* **ForestFormer3D: one queue per card.** 150 s/M per card with two queues against 143 s/M
  with one -- the second queue costs about 7 % instead of buying anything. Oversubscription
  was right while one process used a fifth of the card; with the batched path it holds four
  fifths and a co-tenant only adds contention.
* **SegmentAnyTree: three containers per card.** 200 s/M per card with three against
  245 s/M with two. It uses 3-6 GB and leaves the card at 24-28 % on its own, so the
  memory, not the SMs, is the limit; three containers reached 97 % utilisation and 9.9 GB.
* **AMS3D: CPU, and cheap.** 16 min per tile at 58 s/M with four queues of 32 workers. It
  never competes for a GPU, so it can always run beside the deep models.

The premasked runs show a slightly higher s/M than the clean measurement for the same
method, because the mask removes ~6 % of the points but none of the per-sub-tile Python,
collate and post-processing: fewer points, same overheads.

## Where a tile's time goes

File mtimes of the pipeline's own outputs, one 27.1 M point tile of the ForestFormer3D
re-run (`work_dirs/logs/...`, two queues on its card):

| phase | duration | share |
| :--- | ---: | ---: |
| `las_to_ply` incl. the pre-inference mask | 62 s | 0.7 % |
| preprocess (container: superpoints, info pkl) | 5 s | 0.1 % |
| **inference** | **8,321 s** | **96.3 %** |
| `results_to_las` | 5 s | 0.1 % |
| `trees_to_gpkg` | 28 s | 0.3 % |
| report | 10 s | 0.1 % |

This is why "the GPUs look idle" during a run is almost always a queue in its 110-second
host-side envelope, and why there is nothing left to overlap by adding ForestFormer3D
queues: the host work is 2 % of a tile. It is also why the mosaic-wide stitch, which is
pure host work, is budgeted separately below.

## The steps that are not per tile

| step | cost | note |
| :--- | ---: | :--- |
| `ff3d_geo split` (100 sub-tiles, 20 m halo, 8 neighbours) | ~40 s per tile | reads the tile and its neighbours; 14 in parallel on this host |
| `premask-split` (the masked copy for SAT/AMS3D) | ~60 s per tile | 8 in parallel: 57 tiles in 11 min |
| `premask-restore` | ~5 s per tile | 57 tiles in 4 min |
| **mosaic-wide `stitch`** | **45-47 min** | ONE call over all 5,293 sub-tiles; cannot be split, ids must be unique mosaic-wide |
| `border-check` + `masks` + `buildings` per tile | ~45 s per tile | part of the stitch script |
| `instance_agreement.py`, one pair | 2 min for 57 tiles | 6 in parallel |

## Planning numbers

For a mosaic of N tiles of mean M million points on this host:

* ForestFormer3D: `N x M x 143 s` of card time, one queue per card, so
  `N x M x 143 / (3600 x cards)` hours. The 57-tile mosaic on 7 cards: **4.0 h**.
* SegmentAnyTree: `N x M x 200 s` of card time with three containers per card. The 57-tile
  mosaic on 7 cards: **8.0 h**; on one card: **56 h**.
* AMS3D: `N x M x 58 s` of CPU time at four queues of 32 workers, i.e. ~16 min per tile
  wall: **4.8 h** for 57 tiles.
* Plus, once per mosaic and per method: 45 min of stitch, and ~15 min of per-tile products.

The 57-tile ForestFormer3D re-run took 6 h 49 rather than the 4.0 h the clean figure
predicts, because it ran two queues per card -- the 7 % penalty above, plus the tail of the
largest tiles finishing alone. Build a schedule from the per-card column, then add a third
for the tail.
