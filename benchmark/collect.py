#!/usr/bin/env python3
"""Collect the Phase 2 benchmark results into docs/benchmarks/<date>-carrot-ff3d.md.

Inputs (under --root, or given explicitly with --release-old/--release-fixed/--train-old/
--train-fixed):
  work_dirs/bench-release-{old,fixed}/evaluation_total_test.txt
      Released checkpoint (epoch_3000) inference, scored by tools/final_eval.py. The script
      opens this file with mode 'w' (see tools/final_eval.py line 65), so a normal run leaves
      exactly ONE result block; parse_final_eval() still takes the LAST block defensively, in
      case a stale multi-block file from an older tools/final_eval.py is encountered.
  work_dirs/bench-{old,fixed}-200/test/evaluation_total_test.txt
      The 200-epoch run's own test-split score (epoch_200), same file shape as above.
  work_dirs/bench-{old,fixed}-200/<timestamp>/vis_data/scalars.json
      mmengine scalars, one JSON object per line: train records carry 'loss'/'epoch'/'iter'/
      'lr'/...; val records carry the metric keys (F1, mIoU, ...), optionally prefixed with
      '<dataset>/', plus 'step' (== epoch). Some metric keys may be absent on a given line;
      every reader here tolerates that instead of KeyError-ing.
  work_dirs/bench-{old,fixed}-200/<timestamp>/<timestamp>.log
      mmengine's own text log; used only for wall-clock / sec-per-epoch (parse_log_wallclock).
  work_dirs/bench-{old,fixed}-200/train.log
      The nohup wrapper log for the training process itself. NOT parsed here: the mmengine
      log above already carries the same per-epoch timestamps in a stable, greppable format.

Output: docs/benchmarks/<date>-carrot-ff3d.md. Missing inputs never crash main(): tables show
'n/a' and the success-criteria section shows PENDING for whatever wasn't found (pass
--allow-missing to still write a partial report; without it, main() exits 1).

Usage:  python benchmark/collect.py --root /workspace --date 2026-09-25
Pure Python + numpy; no torch or mmengine imports.
"""
import argparse
import datetime as dt
import json
import math
import os
import re
import sys
from pathlib import Path

import numpy as np

# evaluation_total_test.txt label -> report key, restricted to the SCALAR lines of the global
# block (no per-class list lines, no '(things)'/'(stuff)' breakdowns). Each comment points at
# the log_string(...) call in the CURRENT tools/final_eval.py that produces that line.
FINAL_EVAL_LABELS = {
    'Semantic Segmentation oAcc': 'oAcc',                  # tools/final_eval.py:453
    'Semantic Segmentation mAcc': 'mAcc',                  # tools/final_eval.py:454
    'Semantic Segmentation mIoU': 'mIoU',                  # tools/final_eval.py:456
    'Binary Semantic Segmentation mIoU': 'mIoU_binary',    # tools/final_eval.py:483
    'Instance Segmentation mMUCov': 'mMUCov',              # tools/final_eval.py:549
    'Instance Segmentation mMWCov': 'mMWCov',              # tools/final_eval.py:551
    'Instance Segmentation mPrecision': 'mPrecision',      # tools/final_eval.py:553
    'Instance Segmentation mRecall': 'mRecall',            # tools/final_eval.py:555
    'Instance Segmentation F1 score': 'F1',                # tools/final_eval.py:556
    'Instance Segmentation meanRQ': 'mRQ',                 # tools/final_eval.py:558
    'Instance Segmentation meanSQ': 'mSQ',                 # tools/final_eval.py:560
    'Instance Segmentation meanPQ': 'mPQ',                 # tools/final_eval.py:562
    'Instance Segmentation mean PQ star': 'mPQ_star',      # tools/final_eval.py:564
}
RELEASE_COLUMNS = ['F1', 'mPrecision', 'mRecall', 'mPQ', 'mIoU', 'mIoU_binary', 'mMUCov', 'mMWCov']
LOG_TRAIN_RE = re.compile(
    r'^(\d{4}/\d{2}/\d{2} \d{2}:\d{2}:\d{2}) - mmengine - INFO - Epoch\(train\)\s*\[(\d+)\]\[\s*(\d+)/(\d+)\]')
TS_FMT = '%Y/%m/%d %H:%M:%S'


def parse_final_eval(path):
    """Scalar metrics of the LAST block in evaluation_total_test.txt.

    tools/final_eval.py truncates the file (mode 'w') on every run, so in normal operation the
    file holds exactly one block. This still resolves to the last block's values when more than
    one is present (e.g. a stale file written by an older version of the script), so a naive
    'first match wins' reader never silently reports a superseded run.
    """
    out = {}
    for line in Path(path).read_text().splitlines():
        if ': ' not in line:
            continue
        label, value = line.split(': ', 1)
        key = FINAL_EVAL_LABELS.get(label.strip())
        if key is None:
            continue
        try:
            out[key] = float(value.strip())
        except ValueError:
            continue
    return out


def parse_scalars(paths):
    """Return (train_records, val_records) from mmengine vis_data/scalars.json files.

    A record with both 'loss' and 'epoch' is a train record. Anything else that carries 'step'
    is treated as a val record (regardless of which metric keys it does or doesn't carry, so a
    val line missing some metric is still counted rather than dropped). Any 'prefix/' on a val
    record's keys is stripped, so 'ForAINetV2/F1' and plain 'F1' both come out as 'F1'.
    """
    train, val = [], []
    for p in paths:
        for line in Path(p).read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if 'loss' in rec and 'epoch' in rec:
                train.append(rec)
                continue
            if 'step' in rec:
                val.append({k.split('/', 1)[-1]: v for k, v in rec.items()})
    return train, val


def loss_per_epoch(train):
    by_epoch = {}
    for r in train:
        by_epoch.setdefault(int(r['epoch']), []).append(float(r['loss']))
    return {e: float(np.mean(v)) for e, v in sorted(by_epoch.items())}


def val_curve(val):
    curve = {}
    for r in val:
        epoch = int(r['step'])
        curve[epoch] = {k: float(v) for k, v in r.items() if k != 'step' and isinstance(v, (int, float))}
    return dict(sorted(curve.items()))


def parse_log_wallclock(paths):
    """Wall clock between the first and last 'Epoch(train)' log line, divided by the number
    of epochs spanned (validation time is included, which is what a user waits for)."""
    first = last = None
    for p in paths:
        for line in Path(p).read_text().splitlines():
            m = LOG_TRAIN_RE.match(line)
            if not m:
                continue
            ts = dt.datetime.strptime(m.group(1), TS_FMT)
            epoch = int(m.group(2))
            if first is None:
                first = (ts, epoch)
            last = (ts, epoch)
    if first is None:
        return {'first_ts': None, 'last_ts': None, 'first_epoch': None, 'last_epoch': None,
                'wall_s': None, 'sec_per_epoch': None}
    wall = (last[0] - first[0]).total_seconds()
    spanned = last[1] - first[1]
    return {'first_ts': first[0], 'last_ts': last[0], 'first_epoch': first[1], 'last_epoch': last[1],
            'wall_s': wall, 'sec_per_epoch': (wall / spanned) if spanned > 0 else None}


def summarize_training(work_dir):
    work_dir = Path(work_dir)
    scalars = sorted(work_dir.glob('*/vis_data/scalars.json'))
    logs = sorted(work_dir.glob('*/*.log'))
    if not scalars:
        return None
    train, val = parse_scalars(scalars)
    losses = loss_per_epoch(train)
    curve = val_curve(val)
    nonfinite_epochs = sorted(e for e, l in losses.items() if not math.isfinite(l))
    have_f1 = sorted((e, m['F1']) for e, m in curve.items() if 'F1' in m)
    best = max(have_f1, key=lambda t: t[1]) if have_f1 else None
    last = have_f1[-1] if have_f1 else None
    test_file = work_dir / 'test' / 'evaluation_total_test.txt'
    return {
        'epochs_done': max(losses) if losses else 0,
        'final_loss': losses[max(losses)] if losses else None,
        'nonfinite_epochs': nonfinite_epochs,
        'losses': losses,
        'val': curve,
        'best_val': best,
        'last_val': last,
        'wallclock': parse_log_wallclock(logs),
        'test': parse_final_eval(test_file) if test_file.exists() else None,
    }


def _f(x, nd=4):
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return 'n/a'
    return f'{x:.{nd}f}'


def render(date, release, training, epochs_target=200):
    lines = [f'# ForestFormer3D benchmark on carrot ({date})', '',
             '## Environment', '',
             'Hardware: carrot (H100), Docker image `forestformer3d:cu118` — the image (CUDA/Python/'
             'library versions) is the ONLY thing held constant between the two variants. Dataset: '
             'ForAINetV2 test split (28 plots), scored in both cases with the FIXED `tools/final_eval.py`. '
             'The **old** variant runs the ORIGINAL `main` @ 6a75c37 `tools/test.py`/`tools/train.py` '
             '*together with its own original config* (not the fixed config), checked out into a git '
             'worktree inside the same image — so code AND config both differ from the fixed variant, not '
             'code alone. Spec: `docs/superpowers/specs/2026-09-22-ff3d-fixes-benchmark-tegel-design.md` section 5.',
             '', '## Table 1: released checkpoint epoch_3000, old vs fixed inference', '',
             '| variant | ' + ' | '.join(RELEASE_COLUMNS) + ' |',
             '|' + '---|' * (len(RELEASE_COLUMNS) + 1)]
    for variant in ('old', 'fixed'):
        m = release.get(variant)
        cells = [_f(m.get(c)) if m else 'n/a' for c in RELEASE_COLUMNS]
        lines.append(f'| {variant} | ' + ' | '.join(cells) + ' |')
    lines += ['', '## Table 2: 200-epoch training, old vs fixed', '',
              '| variant | epochs | sec/epoch | final train loss | best val F1 (epoch) | last val F1 (epoch) | test F1 (epoch_200) | test mIoU | non-finite loss epochs |',
              '|---|---|---|---|---|---|---|---|---|']
    for variant in ('old', 'fixed'):
        s = training.get(variant)
        if not s:
            lines.append(f'| {variant} | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |')
            continue
        best = f"{_f(s['best_val'][1])} ({s['best_val'][0]})" if s['best_val'] else 'n/a'
        last = f"{_f(s['last_val'][1])} ({s['last_val'][0]})" if s['last_val'] else 'n/a'
        test_f1 = _f(s['test'].get('F1')) if s['test'] else 'n/a'
        test_miou = _f(s['test'].get('mIoU')) if s['test'] else 'n/a'
        nonfinite = ', '.join(map(str, s['nonfinite_epochs'])) if s['nonfinite_epochs'] else 'none'
        lines.append(f"| {variant} | {s['epochs_done']} | {_f(s['wallclock']['sec_per_epoch'], 1)} | "
                     f"{_f(s['final_loss'])} | {best} | {last} | {test_f1} | {test_miou} | {nonfinite} |")
    epochs = sorted({e for s in training.values() if s for e in s['val']})
    if epochs:
        lines += ['', '### Validation curve (val split, every val_interval epochs)', '',
                  '| epoch | old val F1 | fixed val F1 | old train loss | fixed train loss |', '|---|---|---|---|---|']
        for e in epochs:
            row = [str(e)]
            for variant in ('old', 'fixed'):
                s = training.get(variant)
                row.append(_f(s['val'][e].get('F1')) if s and e in s['val'] else 'n/a')
            for variant in ('old', 'fixed'):
                s = training.get(variant)
                row.append(_f(s['losses'].get(e)) if s else 'n/a')
            lines.append('| ' + ' | '.join(row) + ' |')
    for variant in ('old', 'fixed'):
        s = training.get(variant)
        w = s['wallclock'] if s else None
        if w and w['first_ts']:
            lines.append('')
            lines.append(f"Wall clock {variant}: {w['first_ts']:%Y-%m-%d %H:%M} to {w['last_ts']:%Y-%m-%d %H:%M} "
                         f"({w['wall_s'] / 3600:.2f} h over epochs {w['first_epoch']}..{w['last_epoch']}, validation included).")
    lines += ['', '## Success criteria', '']
    old, fixed = release.get('old'), release.get('fixed')
    if old and fixed:
        ok = fixed.get('F1', float('-inf')) >= old.get('F1', float('inf'))
        lines.append(f"- Released checkpoint: fixed F1 {_f(fixed.get('F1'))} >= old F1 {_f(old.get('F1'))}: {'PASS' if ok else 'FAIL'}")
    else:
        lines.append('- Released checkpoint: PENDING (missing evaluation_total_test.txt for old and/or fixed release run)')
    s = training.get('fixed')
    if s:
        ok = s['epochs_done'] >= epochs_target and not s['nonfinite_epochs'] and len(s['val']) > 0
        lines.append(f"- Fixed 200-epoch run: {s['epochs_done']}/{epochs_target} epochs, "
                     f"{'no non-finite loss' if not s['nonfinite_epochs'] else 'non-finite loss at epochs ' + str(s['nonfinite_epochs'])}, "
                     f"{len(s['val'])} val points: {'PASS' if ok else 'FAIL'}")
    else:
        lines.append('- Fixed 200-epoch run: PENDING (no scalars.json found under bench-fixed-200)')
    s = training.get('old')
    if s:
        lines.append(f"- Old 200-epoch run (informational): {s['epochs_done']}/{epochs_target} epochs"
                     + (', non-finite loss at epochs ' + str(s['nonfinite_epochs']) if s['nonfinite_epochs'] else ''))
    else:
        lines.append('- Old 200-epoch run (informational): PENDING (no scalars.json found under bench-old-200)')
    lines.append('')
    return '\n'.join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--root', default=os.environ.get('FF3D_ROOT', '.'),
                         help='base dir holding work_dirs/ (defaults for all *-old/-fixed paths below)')
    parser.add_argument('--date', default=dt.date.today().isoformat())
    parser.add_argument('--out', default=None, help='default: <root>/docs/benchmarks/<date>-carrot-ff3d.md')
    parser.add_argument('--allow-missing', action='store_true', help='write n/a/PENDING instead of failing')
    parser.add_argument('--release-old', default=None,
                         help='evaluation_total_test.txt for the old-code release eval; '
                              'default <root>/work_dirs/bench-release-old/evaluation_total_test.txt')
    parser.add_argument('--release-fixed', default=None,
                         help='evaluation_total_test.txt for the fixed-code release eval; '
                              'default <root>/work_dirs/bench-release-fixed/evaluation_total_test.txt')
    parser.add_argument('--train-old', default=None,
                         help='old-code 200-epoch work_dir; default <root>/work_dirs/bench-old-200')
    parser.add_argument('--train-fixed', default=None,
                         help='fixed-code 200-epoch work_dir; default <root>/work_dirs/bench-fixed-200')
    args = parser.parse_args(argv)
    root = Path(args.root)
    out = Path(args.out) if args.out else root / 'docs' / 'benchmarks' / f'{args.date}-carrot-ff3d.md'

    release_paths = {
        'old': Path(args.release_old) if args.release_old
        else root / 'work_dirs' / 'bench-release-old' / 'evaluation_total_test.txt',
        'fixed': Path(args.release_fixed) if args.release_fixed
        else root / 'work_dirs' / 'bench-release-fixed' / 'evaluation_total_test.txt',
    }
    train_dirs = {
        'old': Path(args.train_old) if args.train_old else root / 'work_dirs' / 'bench-old-200',
        'fixed': Path(args.train_fixed) if args.train_fixed else root / 'work_dirs' / 'bench-fixed-200',
    }

    missing = []
    release = {}
    for variant, f in release_paths.items():
        if f.exists():
            release[variant] = parse_final_eval(f)
        else:
            missing.append(str(f))
    training = {}
    for variant, d in train_dirs.items():
        s = summarize_training(d) if d.exists() else None
        if s is None:
            missing.append(str(d / '*/vis_data/scalars.json'))
        training[variant] = s
    if missing and not args.allow_missing:
        for m in missing:
            print(f'missing: {m}', file=sys.stderr)
        print('use --allow-missing to write a partial report', file=sys.stderr)
        return 1
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render(args.date, release, training))
    print(f'wrote {out}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
