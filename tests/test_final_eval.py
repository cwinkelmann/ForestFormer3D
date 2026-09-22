import ast
import subprocess
import sys
from pathlib import Path

import pytest

plyfile = pytest.importorskip("plyfile")
pytest.importorskip("scipy")

import numpy as np
from plyfile import PlyData, PlyElement

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / 'tools' / 'final_eval.py'


def write_result_ply(path, semantic_gt, semantic_pred, instance_gt, instance_pred):
    n = len(semantic_gt)
    vertex = np.zeros(n, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
                                ('semantic_pred', 'i4'), ('instance_pred', 'i4'), ('score', 'f4'),
                                ('semantic_gt', 'i4'), ('instance_gt', 'i4')])
    vertex['x'] = np.arange(n)
    vertex['semantic_gt'], vertex['semantic_pred'] = semantic_gt, semantic_pred
    vertex['instance_gt'], vertex['instance_pred'] = instance_gt, instance_pred
    PlyData([PlyElement.describe(vertex, 'vertex')], text=True).write(str(path))


def parse(log_text, key):
    for line in log_text.splitlines():
        if line.startswith(key + ':'):
            return ast.literal_eval(line.split(':', 1)[1].strip())
    raise AssertionError(f'{key} not in log')


def test_binary_semantic_totals_are_summed_over_files(tmp_path):
    # file A: 8 ground/ground, 2 ground predicted as tree, no instances
    write_result_ply(tmp_path / 'a.ply',
                     semantic_gt=[0] * 10, semantic_pred=[0] * 8 + [1] * 2,
                     instance_gt=[-1] * 10, instance_pred=[-1] * 10)
    # file B: 10 tree points in two trees, perfectly predicted.
    # GT instance ids start at 1 (not 0): this file has no ground points, so
    # the raw-vs-normalized heuristic in labels.looks_raw falls back to "no -1
    # seen -> raw convention", under which a GT instance id of 0 would be
    # treated as the raw "unannotated" marker and dropped (see R3 / the
    # normalization test below). Real files always carry ground points, so
    # this ambiguity does not arise there.
    write_result_ply(tmp_path / 'b.ply',
                     semantic_gt=[1] * 10, semantic_pred=[1] * 10,
                     instance_gt=[1] * 5 + [2] * 5, instance_pred=[0] * 5 + [1] * 5)
    proc = subprocess.run([sys.executable, str(SCRIPT), str(tmp_path)],
                          capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    log = (tmp_path / 'evaluation_total_test.txt').read_text()
    iou = parse(log, 'Binary Semantic Segmentation IoU')
    # summed over both files: class 1 (non-tree) tp 8, gt 10, pred 8 -> 0.8;
    # class 2 (tree) tp 10, gt 10, pred 12 -> 10/12
    assert iou[1] == pytest.approx(0.8)
    assert iou[2] == pytest.approx(10 / 12)
    assert parse(log, 'Binary Semantic Segmentation oAcc') == pytest.approx(18 / 20)
    assert parse(log, 'Instance Segmentation F1 score') == pytest.approx(1.0)


def test_rerun_overwrites_instead_of_accumulating(tmp_path):
    # evaluation_total_test.txt must hold exactly one fresh result block per
    # run: a re-run on the same output directory (which run_release_eval.sh
    # does, at minimum across an old and a fixed pass sharing a dir, and
    # routinely while debugging) must not leave a stale block from a
    # previous run alongside the new one, or a naive "first match" reader
    # (e.g. Phase 2's collect.py) would silently pick up stale numbers.
    write_result_ply(tmp_path / 'a.ply',
                     semantic_gt=[1] * 10, semantic_pred=[1] * 10,
                     instance_gt=[1] * 5 + [2] * 5, instance_pred=[0] * 5 + [1] * 5)

    for _ in range(2):
        proc = subprocess.run([sys.executable, str(SCRIPT), str(tmp_path)],
                              capture_output=True, text=True)
        assert proc.returncode == 0, proc.stderr

    log = (tmp_path / 'evaluation_total_test.txt').read_text()
    assert log.count('Instance Segmentation F1 score:') == 1, log


def test_empty_directory_exits_with_message(tmp_path):
    proc = subprocess.run([sys.executable, str(SCRIPT), str(tmp_path)],
                          capture_output=True, text=True)
    assert proc.returncode == 1
    assert 'no .ply result files' in proc.stderr
    assert not (tmp_path / 'evaluation_total_test.txt').exists()


def test_raw_gt_ground_and_unannotated_zero_ids_are_not_spurious_instances(tmp_path):
    # Raw test-set GT convention: ground points (semantic 0) and unannotated
    # points within an annotated region both carry instance id 0. Without
    # normalizing the GT, final_eval would treat that shared id 0 as a real
    # (spurious) GT tree instance, inflating total_gt_ins and dragging recall
    # down even though nothing was actually missed.
    semantic_gt = [0] * 5 + [1] * 3 + [1] * 5
    semantic_pred = [0] * 5 + [1] * 3 + [1] * 5
    # 5 ground points (raw marker 0) + 3 unannotated tree points (raw marker
    # 0) + 5 points of one real tree (raw id 5).
    instance_gt = [0] * 5 + [0] * 3 + [5] * 5
    # nothing predicted for ground/unannotated points; the one real tree is
    # predicted perfectly as instance 0.
    instance_pred = [-1] * 5 + [-1] * 3 + [0] * 5
    write_result_ply(tmp_path / 'c.ply', semantic_gt, semantic_pred, instance_gt, instance_pred)

    proc = subprocess.run([sys.executable, str(SCRIPT), str(tmp_path)],
                          capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    log = (tmp_path / 'evaluation_total_test.txt').read_text()
    # Only one real GT instance exists (the 5-point tree); the spurious id-0
    # cluster made of ground + unannotated points must not count as a second
    # one, so recall/precision/F1 all come out perfect.
    assert parse(log, 'Instance Segmentation F1 score') == pytest.approx(1.0)
