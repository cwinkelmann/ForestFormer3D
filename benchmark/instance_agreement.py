#!/usr/bin/env python3
"""Instance-level agreement between two segmentations of the same point cloud.

Both inputs are result LAS files in the ff3d contract (``treeID`` int32, -1 = none)
over the *same points in the same order* -- e.g. ``work_dirs/berlin-<T>/<T>.las``
(ForestFormer3D) and ``work_dirs/sat-<T>/<T>.las`` (SegmentAnyTree via
``benchmark/sat_to_ff3d.py``), which are both concatenations of the same 100 m
sub-tiles. The script checks that x/y/z agree before comparing anything.

Matching follows ``benchmark/instance_diagnostics.py`` in spirit (one-to-one pairs,
hit at IoU >= ``--iou``, 0.5) but is built for km tiles with 25-40 k instances per
side, where the dense contingency table and the Hungarian assignment that script uses
(``(n_a+1) x (n_b+1)`` float64, O(n^3)) are out of reach. Only overlapping pairs are
enumerated (``np.unique`` over the joint id of every point that carries an instance in
both), and a pair is matched when the two instances are each other's best-IoU partner
and the IoU reaches the threshold. For thresholds >= 0.5 that is the maximum matching:
an instance can have at most one partner above IoU 0.5 (two would each need more than
half of the union), so mutual best only differs from the Hungarian solution on exact
ties. Because there is no ground truth, the result is an *agreement* between two
methods, not an accuracy of either: a low figure says they disagree, not which one is
wrong.

Reported (also as JSON with ``--json``):

  n_a, n_b          instances in A and B
  matched           one-to-one pairs with IoU >= threshold
  matched_frac_a/b  matched / n_a, matched / n_b
  iou_median        median IoU of the matched pairs
  frag_b_per_a      mean number of B instances that put >= 20 % of their own points
                    into one A instance (how many pieces B cuts an A tree into)
  frag_a_per_b      the same the other way round
  split_a_frac      fraction of A instances cut into >= 2 B pieces
  split_b_frac      fraction of B instances cut into >= 2 A pieces
  pts_a, pts_b      points with an instance in A / B, and in both

    python3 benchmark/instance_agreement.py --a work_dirs/berlin-<T>/<T>.las --b work_dirs/sat-<T>/<T>.las \\
        --label-a ForestFormer3D --label-b SegmentAnyTree --json <out>.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "benchmark") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "benchmark"))

from instance_diagnostics import MATCH_IOU, OVERLAP_FRACTION, compact  # noqa: E402


def overlapping_pairs(a: np.ndarray, n_a: int, b: np.ndarray, n_b: int):
    """``(ia, ib, inter, size_a, size_b)`` for every pair of instances sharing a point.

    ``a``/``b`` are compacted id arrays (-1 = none). Sparse counterpart of
    ``instance_diagnostics.contingency``: memory grows with the number of overlapping
    pairs, not with ``n_a * n_b``.
    """
    both = (a >= 0) & (b >= 0)
    key = a[both].astype(np.int64) * (n_b + 1) + b[both]
    uniq, inter = np.unique(key, return_counts=True)
    ia = uniq // (n_b + 1)
    ib = uniq % (n_b + 1)
    size_a = np.bincount(a[a >= 0], minlength=n_a).astype(np.float64)
    size_b = np.bincount(b[b >= 0], minlength=n_b).astype(np.float64)
    return ia, ib, inter.astype(np.float64), size_a, size_b


def agreement(ids_a: np.ndarray, ids_b: np.ndarray, iou_thr: float = MATCH_IOU) -> dict:
    """Agreement statistics between two instance labelings of the same points."""
    if ids_a.shape != ids_b.shape:
        raise ValueError(f"id arrays differ in length: {ids_a.shape} vs {ids_b.shape}")
    if iou_thr < 0.5:
        raise ValueError("iou_thr below 0.5 would need a full assignment; use >= 0.5")
    a, n_a = compact(ids_a)
    b, n_b = compact(ids_b)
    out = {
        "n_points": int(a.size),
        "n_a": n_a, "n_b": n_b,
        "pts_a": int((a >= 0).sum()), "pts_b": int((b >= 0).sum()),
        "pts_both": int(((a >= 0) & (b >= 0)).sum()),
        "iou_threshold": iou_thr,
    }
    if n_a == 0 or n_b == 0:
        out.update(matched=0, matched_frac_a=0.0, matched_frac_b=0.0, iou_median=None,
                   frag_b_per_a=0.0, frag_a_per_b=0.0, split_a_frac=0.0, split_b_frac=0.0)
        return out

    ia, ib, inter, size_a, size_b = overlapping_pairs(a, n_a, b, n_b)
    iou = inter / (size_a[ia] + size_b[ib] - inter)

    # best partner of every A instance and of every B instance (first pair wins a tie:
    # np.unique sorted the pairs by (ia, ib), and lexsort keeps that order among equals)
    best_a = np.full(n_a, -1)
    best_b = np.full(n_b, -1)
    order = np.lexsort((np.arange(iou.size), -iou))
    seen_a = np.zeros(n_a, bool)
    seen_b = np.zeros(n_b, bool)
    for k in order:
        if not seen_a[ia[k]]:
            best_a[ia[k]] = k
            seen_a[ia[k]] = True
        if not seen_b[ib[k]]:
            best_b[ib[k]] = k
            seen_b[ib[k]] = True
    mutual = (best_a[ia] == np.arange(iou.size)) & (best_b[ib] == np.arange(iou.size))
    hit = mutual & (iou >= iou_thr)
    matched = int(hit.sum())

    frag_b = np.bincount(ia[inter / size_b[ib] >= OVERLAP_FRACTION], minlength=n_a)  # B pieces per A
    frag_a = np.bincount(ib[inter / size_a[ia] >= OVERLAP_FRACTION], minlength=n_b)  # A pieces per B

    out.update(
        matched=matched,
        matched_frac_a=matched / n_a,
        matched_frac_b=matched / n_b,
        iou_median=float(np.median(iou[hit])) if matched else None,
        frag_b_per_a=float(frag_b.mean()),
        frag_a_per_b=float(frag_a.mean()),
        split_a_frac=float((frag_b >= 2).mean()),
        split_b_frac=float((frag_a >= 2).mean()),
    )
    return out


def read_ids(path) -> tuple[np.ndarray, np.ndarray]:
    import laspy

    las = laspy.read(str(path))
    xyz = np.column_stack([np.asarray(las.x), np.asarray(las.y), np.asarray(las.z)])
    return np.asarray(las.treeID, dtype=np.int64), xyz


def format_table(res: dict, label_a: str, label_b: str) -> str:
    med = "n/a" if res["iou_median"] is None else f"{res['iou_median']:.3f}"
    rows = [
        ("points", f"{res['n_points']:,}"),
        (f"instances {label_a} / {label_b}", f"{res['n_a']:,} / {res['n_b']:,}"),
        (f"points with an instance {label_a} / {label_b} / both",
         f"{res['pts_a']:,} / {res['pts_b']:,} / {res['pts_both']:,}"),
        (f"matched pairs (IoU >= {res['iou_threshold']})", f"{res['matched']:,}"),
        (f"matched share of {label_a} / {label_b}",
         f"{100 * res['matched_frac_a']:.1f} % / {100 * res['matched_frac_b']:.1f} %"),
        ("median IoU of matched pairs", med),
        (f"{label_b} pieces per {label_a} instance (>= 20 % overlap)", f"{res['frag_b_per_a']:.2f}"),
        (f"{label_a} pieces per {label_b} instance", f"{res['frag_a_per_b']:.2f}"),
        (f"{label_a} instances split by {label_b} (>= 2 pieces)", f"{100 * res['split_a_frac']:.1f} %"),
        (f"{label_b} instances split by {label_a}", f"{100 * res['split_b_frac']:.1f} %"),
    ]
    return "\n".join(["| Metric | Value |", "|---|---|"] + [f"| {k} | {v} |" for k, v in rows])


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--a", required=True, type=Path)
    p.add_argument("--b", required=True, type=Path)
    p.add_argument("--label-a", default="A")
    p.add_argument("--label-b", default="B")
    p.add_argument("--iou", type=float, default=MATCH_IOU, help="match threshold, >= 0.5")
    p.add_argument("--json", type=Path, default=None)
    args = p.parse_args(argv)

    ids_a, xyz_a = read_ids(args.a)
    ids_b, xyz_b = read_ids(args.b)
    if xyz_a.shape != xyz_b.shape or not np.allclose(xyz_a, xyz_b, atol=0.002):
        raise SystemExit(f"{args.a} and {args.b} do not hold the same points in the same order")
    res = agreement(ids_a, ids_b, args.iou)
    res.update(a=str(args.a), b=str(args.b), label_a=args.label_a, label_b=args.label_b)
    print(format_table(res, args.label_a, args.label_b))
    if args.json:
        args.json.write_text(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
