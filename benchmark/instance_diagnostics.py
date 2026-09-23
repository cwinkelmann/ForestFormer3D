#!/usr/bin/env python3
"""Over- and under-segmentation diagnostics for ForestFormer3D result PLYs.

`tools/final_eval.py` reports F1 / coverage; it does not say *how* a run fails.
This script answers the two questions the Berlin ALS results raised:

  * does the model split single trees into several instances (over-segmentation)?
  * are the predicted instances too short (crowns only, stems missed)?

It reads the result PLYs written by `tools/test.py` (fields ``x y z
semantic_pred instance_pred score semantic_gt instance_gt``), so it needs no
separate ground-truth files: the GT travelled with the prediction.

Per plot and pooled over plots it reports

  gt          number of GT trees (``instance_gt`` normalized: ground and
              unannotated points, which share raw id 0, become -1)
  pred        number of predicted instances (``instance_pred >= 0``)
  matched     GT trees with an optimally matched prediction at IoU >= 0.5
              (Hungarian assignment on the full IoU matrix)
  frag/tree   mean number of predictions that put >= 20 % of THEIR OWN points
              inside one GT tree -- the fragment count of that tree
  split       fraction of GT trees with >= 2 such fragments
  merged      fraction of predictions that cover >= 20 % of two or more GT trees
  h_pred      median predicted-instance height (max z of the instance minus the
              plot's min z)
  h_gt        the same for GT trees

Usage::

    python benchmark/instance_diagnostics.py \\
        --run full=work_dirs/bench-release-fixed \\
        --run thin-uniform=work_dirs/logs/thin/uniform/out \\
        --run thin-canopy=work_dirs/logs/thin/canopy/out --per-plot
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

#: A prediction counts as a "fragment" of a GT tree when this share of the
#: prediction's own points falls inside that tree; a prediction "merges" two GT
#: trees when it covers this share of each of them.
OVERLAP_FRACTION = 0.20
#: IoU at which an optimally matched (prediction, GT tree) pair counts as a hit.
MATCH_IOU = 0.50


def compact(ids: np.ndarray) -> tuple[np.ndarray, int]:
    """Map ids >= 0 to 0..K-1 (order of first appearance), keep negatives as -1."""
    ids = np.asarray(ids).astype(np.int64)
    out = np.full(ids.shape, -1, dtype=np.int64)
    valid = ids >= 0
    if not valid.any():
        return out, 0
    uniq, inverse = np.unique(ids[valid], return_inverse=True)
    out[valid] = inverse
    return out, int(uniq.size)


def normalize_gt(semantic_gt: np.ndarray, instance_gt: np.ndarray) -> np.ndarray:
    """GT tree ids: -1 for ground / unannotated, else 0..G-1.

    Mirrors ``oneformer3d.labels.normalize_instance_gt`` (kept standalone here so
    the script stays numpy-only and runs outside the container). The test-set GT
    uses the raw ``treeID`` convention where ground (semantic 0) and unannotated
    points both carry id 0; that id is not a tree. ``raw`` is detected the same
    way ``looks_raw`` does.
    """
    semantic_gt = np.asarray(semantic_gt)
    instance_gt = np.asarray(instance_gt).astype(np.int64)
    ground = semantic_gt == 0
    raw = bool((instance_gt[ground] == 0).any()) if ground.any() \
        else not bool((instance_gt < 0).any())
    bad = ground | (instance_gt < 0)
    if raw:
        bad = bad | (instance_gt == 0)
    return compact(np.where(bad, -1, instance_gt))[0]


def contingency(pred: np.ndarray, n_pred: int, gt: np.ndarray, n_gt: int):
    """(inter, pred_sizes, gt_sizes) for compacted id arrays (-1 = none)."""
    flat = (pred + 1) * (n_gt + 1) + (gt + 1)
    table = np.bincount(flat, minlength=(n_pred + 1) * (n_gt + 1))
    table = table.reshape(n_pred + 1, n_gt + 1)
    inter = table[1:, 1:].astype(np.float64)
    pred_sizes = table[1:, :].sum(axis=1).astype(np.float64)
    gt_sizes = table[:, 1:].sum(axis=0).astype(np.float64)
    return inter, pred_sizes, gt_sizes


def diagnose_plot(path: Path) -> dict | None:
    """Diagnostics for one result PLY, or None when it carries no ground truth."""
    from plyfile import PlyData

    data = PlyData.read(str(path)).elements[0].data
    names = set(data.dtype.names)
    if not {"semantic_gt", "instance_gt", "instance_pred"} <= names:
        return None
    z = np.asarray(data["z"], dtype=np.float64)
    pred, n_pred = compact(np.asarray(data["instance_pred"]))
    gt = normalize_gt(np.asarray(data["semantic_gt"]),
                      np.asarray(data["instance_gt"]))
    n_gt = int(gt.max()) + 1 if gt.size and gt.max() >= 0 else 0

    inter, pred_sizes, gt_sizes = contingency(pred, n_pred, gt, n_gt)
    union = pred_sizes[:, None] + gt_sizes[None, :] - inter
    iou = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)

    matched = 0
    if n_pred and n_gt:
        from scipy.optimize import linear_sum_assignment

        rows, cols = linear_sum_assignment(-iou)
        matched = int((iou[rows, cols] >= MATCH_IOU).sum())

    # fragments: predictions whose OWN points are >=20 % inside this GT tree
    with np.errstate(invalid="ignore", divide="ignore"):
        share_of_pred = np.divide(inter, pred_sizes[:, None],
                                  out=np.zeros_like(inter),
                                  where=pred_sizes[:, None] > 0)
        share_of_gt = np.divide(inter, gt_sizes[None, :],
                                out=np.zeros_like(inter),
                                where=gt_sizes[None, :] > 0)
    frags = (share_of_pred >= OVERLAP_FRACTION).sum(axis=0) if n_pred else \
        np.zeros(n_gt)
    covers = (share_of_gt >= OVERLAP_FRACTION).sum(axis=1) if n_gt else \
        np.zeros(n_pred)

    z_min = float(z.min()) if z.size else 0.0
    pred_h = np.array([z[pred == i].max() - z_min for i in range(n_pred)]) \
        if n_pred else np.zeros(0)
    gt_h = np.array([z[gt == i].max() - z_min for i in range(n_gt)]) \
        if n_gt else np.zeros(0)

    return dict(
        scan=path.stem,
        n_gt=n_gt,
        n_pred=n_pred,
        matched=matched,
        frags=frags.astype(np.int64),
        covers=covers.astype(np.int64),
        pred_h=pred_h,
        gt_h=gt_h,
    )


def summarize(plots: list[dict]) -> dict:
    """Pool per-plot diagnostics into one row."""
    frags = np.concatenate([p["frags"] for p in plots]) if plots else np.zeros(0)
    covers = np.concatenate([p["covers"] for p in plots]) if plots else np.zeros(0)
    pred_h = np.concatenate([p["pred_h"] for p in plots]) if plots else np.zeros(0)
    gt_h = np.concatenate([p["gt_h"] for p in plots]) if plots else np.zeros(0)
    n_gt = int(sum(p["n_gt"] for p in plots))
    n_pred = int(sum(p["n_pred"] for p in plots))
    return dict(
        plots=len(plots),
        n_gt=n_gt,
        n_pred=n_pred,
        matched=int(sum(p["matched"] for p in plots)),
        match_rate=(sum(p["matched"] for p in plots) / n_gt) if n_gt else float("nan"),
        frag_per_tree=float(frags.mean()) if frags.size else float("nan"),
        split_frac=float((frags >= 2).mean()) if frags.size else float("nan"),
        merged_frac=float((covers >= 2).mean()) if covers.size else float("nan"),
        pred_h_median=float(np.median(pred_h)) if pred_h.size else float("nan"),
        gt_h_median=float(np.median(gt_h)) if gt_h.size else float("nan"),
    )


HEADER = (f"{'run':<14} {'plots':>5} {'gt':>6} {'pred':>6} {'matched':>7} "
          f"{'match%':>7} {'frag/tree':>9} {'split%':>7} {'merged%':>8} "
          f"{'h_pred':>7} {'h_gt':>6}")


def format_row(label: str, s: dict) -> str:
    return (f"{label:<14} {s['plots']:>5d} {s['n_gt']:>6d} {s['n_pred']:>6d} "
            f"{s['matched']:>7d} {100 * s['match_rate']:>6.1f}% "
            f"{s['frag_per_tree']:>9.2f} {100 * s['split_frac']:>6.1f}% "
            f"{100 * s['merged_frac']:>7.1f}% {s['pred_h_median']:>7.1f} "
            f"{s['gt_h_median']:>6.1f}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--run", action="append", required=True, metavar="LABEL=DIR",
                        help="a directory of result PLYs, repeatable")
    parser.add_argument("--per-plot", action="store_true",
                        help="also print one line per plot")
    parser.add_argument("--json", dest="json_out", default=None,
                        help="write the pooled rows (and per-plot rows) here")
    args = parser.parse_args(argv)

    runs = []
    for spec in args.run:
        if "=" not in spec:
            print(f"--run needs LABEL=DIR, got {spec!r}", file=sys.stderr)
            return 2
        label, _, directory = spec.partition("=")
        runs.append((label, Path(directory)))

    out: dict = {"overlap_fraction": OVERLAP_FRACTION, "match_iou": MATCH_IOU,
                 "runs": {}}
    print(HEADER)
    for label, directory in runs:
        plys = sorted(directory.glob("*.ply"))
        if not plys:
            print(f"{label}: no PLYs in {directory}", file=sys.stderr)
            return 1
        plots, skipped = [], []
        for ply in plys:
            row = diagnose_plot(ply)
            if row is None:
                skipped.append(ply.name)
            else:
                plots.append(row)
        if skipped:
            print(f"{label}: {len(skipped)} PLY(s) without GT fields, skipped",
                  file=sys.stderr)
        pooled = summarize(plots)
        out["runs"][label] = {
            "dir": str(directory),
            "pooled": pooled,
            "per_plot": [
                dict(scan=p["scan"], n_gt=p["n_gt"], n_pred=p["n_pred"],
                     matched=p["matched"],
                     frag_per_tree=float(p["frags"].mean()) if p["frags"].size else None,
                     split_frac=float((p["frags"] >= 2).mean()) if p["frags"].size else None,
                     merged_frac=float((p["covers"] >= 2).mean()) if p["covers"].size else None,
                     pred_h_median=float(np.median(p["pred_h"])) if p["pred_h"].size else None,
                     gt_h_median=float(np.median(p["gt_h"])) if p["gt_h"].size else None)
                for p in plots
            ],
        }
        print(format_row(label, pooled))
        if args.per_plot:
            for p in out["runs"][label]["per_plot"]:
                print(f"  {p['scan'][:44]:<44} gt={p['n_gt']:>4d} pred={p['n_pred']:>4d} "
                      f"matched={p['matched']:>4d} frag={p['frag_per_tree']:.2f} "
                      f"split={100 * p['split_frac']:.0f}% "
                      f"h_pred={p['pred_h_median']:.1f} h_gt={p['gt_h_median']:.1f}")

    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_out).write_text(json.dumps(out, indent=2))
        print(f"wrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
