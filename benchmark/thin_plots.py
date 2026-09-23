#!/usr/bin/env python3
"""Thin labelled ForAINetV2 test plots down to an airborne-laser point density.

Motivation: the released ForestFormer3D checkpoint was trained on terrestrial /
drone-laser plots with hundreds to thousands of points per square metre, while the
Berlin ALS data it performs poorly on has roughly 20-30 pts/m2.  Thinning the
*labelled* test plots to ALS density separates "density is the problem" from
"the Berlin domain is the problem" without needing Berlin ground truth.

Two thinning modes:

``uniform``
    Random subsample without replacement.  Keeps the vertical structure of the
    plot (stems, understorey, ground) and only reduces the density.  This is the
    control: it isolates density from viewing geometry.

``canopy``
    An **approximation** of what an airborne sensor sees.  Points are binned into
    0.5 m xy cells; inside each cell the points are sorted by z descending and
    only the highest ``k`` are kept, with the per-cell budget ``k`` distributed
    over the occupied cells proportionally to their point counts (at least one
    point per cell).  Under-canopy returns and stem points are therefore mostly
    dropped, the way an ALS sensor mostly sees the top of the canopy.

    This is explicitly *not* a sensor simulation: a real ALS system produces
    multiple returns per pulse, so some ground and understorey points survive
    under canopy gaps, the footprint is not a square cell, and the hit
    probability depends on the leaf area above a point rather than on rank in z.
    The approximation keeps the first-return-like bias (top of canopy over
    stems) which is the property being tested here.

Both modes keep every field of the input PLY (``x y z semantic_seg treeID`` and
any extra column), so the thinned file goes through the normal labelled
preprocessing path (``batch_load_ForAINetV2_data.py`` without ``--unlabeled``).

The target point count is ``density * area``, where ``area`` is the area of the
2D convex hull of the plot's xy coordinates (shapely).  Using the hull rather
than the bounding box matters for the non-rectangular circular plots in
ForAINetV2.

Output name: ``<dst>/<stem>_thin<density><u|c>.ply``.  The suffix deliberately
does not end in ``_<digits>``, which the ForAINetV2 data tools would strip as a
block index.

Example::

    python benchmark/thin_plots.py --src data/ForAINetV2/test_data \\
        --dst data/ForAINetV2/test_data_thin25u --density 25 --mode uniform --seed 0 \\
        --list data/ForAINetV2/meta_data/test_list.txt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# --- mode -> file-name suffix letter -----------------------------------------
MODE_SUFFIX = {"uniform": "u", "canopy": "c"}

#: xy cell size (metres) of the canopy-mode binning grid.
CELL_SIZE = 0.5


def density_tag(density: float) -> str:
    """``25`` -> ``"25"``, ``12.5`` -> ``"12p5"`` (no dot: it would split the stem)."""
    text = f"{float(density):g}"
    return text.replace(".", "p").replace("-", "m")


def thinned_name(stem: str, density: float, mode: str) -> str:
    """Scan name of the thinned copy of ``stem``."""
    return f"{stem}_thin{density_tag(density)}{MODE_SUFFIX[mode]}"


#: Above this point count, ``hull_area`` reduces the input to hull candidates
#: before handing it to shapely (shapely 1.8 builds a MultiPoint point by point).
HULL_CANDIDATE_LIMIT = 20_000
#: Bin width (m) of that reduction.
HULL_BIN = 0.1


def _extreme_indices(key: np.ndarray, value: np.ndarray) -> np.ndarray:
    """For each distinct ``key``, the indices of the smallest and largest ``value``."""
    order = np.lexsort((value, key))
    sorted_key = key[order]
    boundary = sorted_key[1:] != sorted_key[:-1]
    first = np.concatenate([[True], boundary])
    last = np.concatenate([boundary, [True]])
    return np.concatenate([order[first], order[last]])


def hull_area(x: np.ndarray, y: np.ndarray) -> float:
    """Area (m2) of the 2D convex hull of the xy footprint."""
    from shapely.geometry import MultiPoint

    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size < 3:
        return 0.0
    if x.size > HULL_CANDIDATE_LIMIT:
        # Every convex-hull vertex is, within one bin, the topmost or bottommost
        # point of its x column or the leftmost/rightmost of its y row. Keeping
        # only those few thousand candidates is what makes this affordable on
        # multi-million-point plots; the hull can shrink by at most HULL_BIN at
        # the rim, far below the precision the density target needs.
        xb = np.floor(x / HULL_BIN).astype(np.int64)
        yb = np.floor(y / HULL_BIN).astype(np.int64)
        keep = np.unique(np.concatenate([_extreme_indices(xb, y),
                                         _extreme_indices(yb, x)]))
        x, y = x[keep], y[keep]
    return float(MultiPoint(np.column_stack([x, y])).convex_hull.area)


def uniform_indices(n_points: int, target: int, rng: np.random.Generator) -> np.ndarray:
    """Indices of a random subsample of size ``target`` without replacement."""
    if target >= n_points:
        return np.arange(n_points)
    return np.sort(rng.choice(n_points, size=target, replace=False))


def _allocate(counts: np.ndarray, target: int, rng: np.random.Generator) -> np.ndarray:
    """Per-cell budgets summing to ``target``, >=1 each and <= the cell's count.

    Proportional to ``counts``, then repaired to hit ``target`` exactly by
    adding to (removing from) the cells with the most (fewest) points.
    """
    total = int(counts.sum())
    if target >= total:
        return counts.copy()
    n_cells = counts.size
    if n_cells >= target:
        # Fewer points in the budget than occupied cells: one point from each of
        # ``target`` randomly chosen cells (cells are the unit of coverage here).
        k = np.zeros(n_cells, dtype=np.int64)
        k[rng.choice(n_cells, size=target, replace=False)] = 1
        return k
    k = np.clip(np.rint(counts * (target / total)).astype(np.int64), 1, counts)
    order = np.argsort(-counts, kind="stable")  # big cells absorb the remainder
    diff = target - int(k.sum())
    while diff > 0:
        room = k[order] < counts[order]
        take = min(diff, int(room.sum()))
        k[order[room][:take]] += 1
        diff -= take
    while diff < 0:
        spare = k[order] > 1
        give = min(-diff, int(spare.sum()))
        k[order[spare][-give:]] -= 1
        diff += give
    return k


def canopy_indices(x: np.ndarray, y: np.ndarray, z: np.ndarray, target: int,
                   rng: np.random.Generator, cell: float = CELL_SIZE) -> np.ndarray:
    """Indices of the highest points per 0.5 m xy cell, summing to ``target``.

    See the module docstring: this approximates an airborne view, it is not a
    sensor simulation.
    """
    n_points = x.size
    if target >= n_points:
        return np.arange(n_points)
    keys = np.column_stack([np.floor(np.asarray(x, dtype=np.float64) / cell),
                            np.floor(np.asarray(y, dtype=np.float64) / cell)]
                           ).astype(np.int64)
    _, cell_of_point, counts = np.unique(
        keys, axis=0, return_inverse=True, return_counts=True)
    cell_of_point = cell_of_point.ravel()
    budget = _allocate(counts, target, rng)

    # Sort by (cell, -z) so each cell's points are contiguous and top-down.
    order = np.lexsort((-np.asarray(z, dtype=np.float64), cell_of_point))
    starts = np.zeros(counts.size, dtype=np.int64)
    np.cumsum(counts[:-1], out=starts[1:])
    rank = np.arange(n_points) - starts[cell_of_point[order]]
    keep = rank < budget[cell_of_point[order]]
    return np.sort(order[keep])


def thin_one(src_ply: Path, dst_ply: Path, density: float, mode: str,
             seed: int) -> dict:
    """Thin one plot; returns the per-plot report dict."""
    from plyfile import PlyData, PlyElement

    data = PlyData.read(str(src_ply))
    vertex = data.elements[0].data
    x = np.asarray(vertex["x"], dtype=np.float64)
    y = np.asarray(vertex["y"], dtype=np.float64)
    z = np.asarray(vertex["z"], dtype=np.float64)
    n_points = x.size
    area = hull_area(x, y)
    target = int(round(density * area))
    rng = np.random.default_rng(seed)

    if mode == "uniform":
        keep = uniform_indices(n_points, target, rng)
    elif mode == "canopy":
        keep = canopy_indices(x, y, z, target, rng)
    else:
        raise ValueError(f"unknown mode {mode!r}")

    out = vertex[keep]  # structured array: every field is carried over
    dst_ply.parent.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(out, data.elements[0].name)],
            text=False, byte_order="<").write(str(dst_ply))
    return dict(
        scan=src_ply.stem,
        out=dst_ply.name,
        points=n_points,
        area=area,
        target=target,
        kept=int(keep.size),
        density_in=(n_points / area) if area > 0 else float("nan"),
        density_out=(keep.size / area) if area > 0 else float("nan"),
    )


def read_list(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--src", default="data/ForAINetV2/test_data",
                        help="directory holding the labelled test PLYs")
    parser.add_argument("--dst", required=True, help="output directory")
    parser.add_argument("--density", type=float, required=True,
                        help="target point density in points per square metre")
    parser.add_argument("--mode", choices=sorted(MODE_SUFFIX), required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--list", dest="scan_list", default=None,
                        help="file with one scan stem per line; default: every "
                             "*.ply in --src")
    parser.add_argument("--out-list", default=None,
                        help="write the thinned scan names here (one per line)")
    args = parser.parse_args(argv)

    src = Path(args.src)
    dst = Path(args.dst)
    stems = (read_list(Path(args.scan_list)) if args.scan_list
             else sorted(p.stem for p in src.glob("*.ply")))
    if not stems:
        print(f"no scans found in {src}", file=sys.stderr)
        return 1

    rows, names = [], []
    print(f"{'scan':<46} {'points':>10} {'area_m2':>9} {'target':>8} "
          f"{'kept':>8} {'in/m2':>8} {'out/m2':>7}")
    for stem in stems:
        src_ply = src / f"{stem}.ply"
        if not src_ply.is_file():
            print(f"missing: {src_ply}", file=sys.stderr)
            return 1
        name = thinned_name(stem, args.density, args.mode)
        row = thin_one(src_ply, dst / f"{name}.ply", args.density, args.mode,
                       args.seed)
        rows.append(row)
        names.append(name)
        print(f"{row['scan'][:46]:<46} {row['points']:>10d} {row['area']:>9.0f} "
              f"{row['target']:>8d} {row['kept']:>8d} {row['density_in']:>8.1f} "
              f"{row['density_out']:>7.1f}")

    total_in = sum(r["points"] for r in rows)
    total_out = sum(r["kept"] for r in rows)
    print(f"\n{len(rows)} plots: {total_in} -> {total_out} points "
          f"({100.0 * total_out / max(total_in, 1):.2f} % kept), mode={args.mode}, "
          f"density={args.density} pts/m2, seed={args.seed}")
    if args.out_list:
        Path(args.out_list).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_list).write_text("".join(f"{n}\n" for n in names))
        print(f"wrote scan list: {args.out_list}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
