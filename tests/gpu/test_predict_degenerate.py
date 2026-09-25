"""Degenerate cylinder regions must not abort full-plot inference.

A 100 m ALS sub-tile with almost nothing in it (water, bare ground, a handful of
stray returns) produced cylinder regions whose voxels all vanished inside the
spconv UNet's stride-2 downsampling, and the ``ValueError`` killed the whole
``tools/test.py`` process -- with it every other sub-tile of the km tile. Predict
must now skip such a region, and still write a result PLY for the scan.

The log lines asserted here are the ones ``docs/known-issues.md`` tells an
operator to grep for, so they are part of the contract.

``build_model`` and ``synthetic_plot`` are fixtures from tests/gpu/conftest.py.
"""
import logging

import numpy as np
import pytest
import torch
from mmdet3d.structures import Det3DDataSample
from plyfile import PlyData

pytestmark = pytest.mark.gpu

# region_step_factor=1.0 puts the cylinder centres 16 m apart, so a cluster at the
# origin and an outlier 48 m away land in separate regions and the test runs a
# handful of forward passes instead of a few hundred.
COARSE = dict(region_step_factor=1.0)


def collinear_trio(x=48.0, y=48.0):
    """Three nearly collinear points, far enough away to own a cylinder alone."""
    return torch.tensor([[x, y, 0.0],
                         [x + 0.01, y - 0.01, 9.0],
                         [x - 0.01, y + 0.01, 18.0]], dtype=torch.float32)


def run_predict(model, tmp_path, points, name):
    sample = Det3DDataSample()
    sample.set_metainfo({'lidar_path': str(tmp_path / 'points' / f'{name}.bin')})
    sample.eval_ann_info = None
    with torch.no_grad():
        result = model.predict({'points': [points.cuda()]}, [sample])
    return result[0].pred_pts_seg


def read_ply(path, n):
    assert path.exists(), f'{path} was not written'
    vertex = PlyData.read(str(path))['vertex']
    assert vertex.count == n
    return vertex


def warnings_containing(caplog, needle):
    return [r.getMessage() for r in caplog.records
            if r.levelno >= logging.WARNING and needle in r.getMessage()]


def test_degenerate_region_does_not_abort_the_scan(
        tmp_path, mm_caplog, build_model, synthetic_plot):
    """A normal cluster plus one region holding three nearly collinear points."""
    out_dir = tmp_path / 'out'
    model = build_model(out_dir, **COARSE)
    cluster, _, _ = synthetic_plot(n_ground=4000, n_trees=2)
    points = torch.cat([cluster, collinear_trio()])
    n = points.shape[0]

    with mm_caplog.at_level(logging.WARNING):
        seg = run_predict(model, tmp_path, points, 'plot_trio')

    inst = np.asarray(seg['pts_instance_mask'][1])
    sem = np.asarray(seg['pts_semantic_mask'][1])
    assert len(inst) == n and len(sem) == n
    # the healthy part is still segmented
    assert (inst[:-3] >= 0).any()
    # the skipped region contributed nothing: its points stay nodata
    assert (inst[-3:] == -1).all()
    assert (sem[-3:] == -1).all()

    skips = warnings_containing(mm_caplog, 'skipping degenerate region')
    assert skips, 'the degenerate region was never reported'
    # scan name, region index and point count, as known-issues.md promises
    assert all('plot_trio' in m and 'points' in m for m in skips)
    assert any('min_region_points=64' in m for m in skips)
    summary = warnings_containing(mm_caplog, 'cylinder regions')
    assert len(summary) == 1, summary
    assert 'were segmented' in summary[0] and 'were degenerate' in summary[0]

    vertex = read_ply(out_dir / 'plot_trio.ply', n)
    assert np.array_equal(vertex['instance_pred'], inst)
    assert np.array_equal(vertex['semantic_pred'], sem)


def test_per_region_warnings_are_capped_at_three(
        tmp_path, mm_caplog, build_model, synthetic_plot):
    """Many degenerate regions must not put one WARNING each into a km-tile log."""
    out_dir = tmp_path / 'out'
    model = build_model(out_dir, **COARSE)
    cluster, _, _ = synthetic_plot(n_ground=4000, n_trees=2)
    # a grid of isolated vertical needles, each one its own degenerate region
    needles = torch.cat([collinear_trio(x=32.0 * (i + 1), y=32.0 * (j + 1))
                         for i in range(3) for j in range(3)])
    points = torch.cat([cluster, needles])

    with mm_caplog.at_level(logging.WARNING):
        run_predict(model, tmp_path, points, 'plot_needles')

    skips = warnings_containing(mm_caplog, 'skipping degenerate region')
    assert len(skips) == 3, f'expected the 3-line cap, got {len(skips)}'
    summary = warnings_containing(mm_caplog, 'cylinder regions')
    assert len(summary) == 1
    # the summary carries the real total, and its four counts reconcile
    counts = [int(w) for w in summary[0].replace(',', ' ').split() if w.isdigit()]
    total, used, degenerate, rejected, empty = counts[-5:]
    assert used + degenerate + rejected + empty == total
    assert degenerate > 3


def test_scan_with_only_five_points_writes_an_unlabelled_ply(
        tmp_path, mm_caplog, build_model):
    """No region survives: the scan still gets a complete, all-nodata result."""
    out_dir = tmp_path / 'out'
    model = build_model(out_dir, **COARSE)
    points = torch.tensor([[0.0, 0.0, 0.0],
                           [1.0, 0.5, 2.0],
                           [2.0, 1.0, 4.0],
                           [3.0, 1.5, 6.0],
                           [4.0, 2.0, 8.0]], dtype=torch.float32)

    with mm_caplog.at_level(logging.WARNING):
        seg = run_predict(model, tmp_path, points, 'plot_five')

    inst = np.asarray(seg['pts_instance_mask'][1])
    sem = np.asarray(seg['pts_semantic_mask'][1])
    assert inst.tolist() == [-1] * 5
    assert sem.tolist() == [-1] * 5

    nothing = warnings_containing(mm_caplog, 'no usable cylinder region')
    assert len(nothing) == 1, nothing
    assert 'all-unlabelled' in nothing[0] and '5 points' in nothing[0]

    vertex = read_ply(out_dir / 'plot_five.ply', 5)
    assert (np.asarray(vertex['instance_pred']) == -1).all()
    assert (np.asarray(vertex['semantic_pred']) == -1).all()


def test_spconv_value_error_is_caught_per_region(
        tmp_path, monkeypatch, mm_caplog, build_model, synthetic_plot):
    """With the pre-filter disabled, a ValueError out of spconv only loses its region."""
    import oneformer3d.oneformer3d as ofm

    out_dir = tmp_path / 'out'
    model = build_model(out_dir, **COARSE)
    cluster, _, _ = synthetic_plot(n_ground=4000, n_trees=2)
    points = torch.cat([cluster, collinear_trio()])
    n = points.shape[0]

    # disable the cheap pre-filter so the degenerate region reaches the backbone
    monkeypatch.setattr(ofm, 'degenerate_region_reason', lambda *a, **k: None)
    real_extract_feat = model.extract_feat
    raised = []

    def flaky(x):
        # the three-point region is the only one with a handful of voxels
        if x.features.shape[0] < 64:
            raised.append(int(x.features.shape[0]))
            raise ValueError('Your points vanished here, this usually because you '
                             'provide conv params that may ignore some input points.')
        return real_extract_feat(x)

    monkeypatch.setattr(model, 'extract_feat', flaky)
    with mm_caplog.at_level(logging.WARNING):
        seg = run_predict(model, tmp_path, points, 'plot_flaky')

    assert raised, 'the fake spconv failure never fired'
    rejects = warnings_containing(mm_caplog, 'spconv rejected region')
    assert rejects, 'the spconv failure was not reported'
    assert len(rejects) <= 3, 'the spconv warning is not rate-limited'
    assert 'Your points vanished here' in rejects[0]

    inst = np.asarray(seg['pts_instance_mask'][1])
    assert len(inst) == n
    assert (inst[:-3] >= 0).any()
    assert (inst[-3:] == -1).all()
    read_ply(out_dir / 'plot_flaky.ply', n)


def test_other_exceptions_are_not_swallowed(
        tmp_path, monkeypatch, build_model, synthetic_plot):
    """Only ValueError is treated as a degenerate region."""
    model = build_model(tmp_path / 'out', **COARSE)
    points, _, _ = synthetic_plot(n_ground=4000, n_trees=2)

    def boom(x):
        raise RuntimeError('CUDA out of memory')

    monkeypatch.setattr(model, 'extract_feat', boom)
    with pytest.raises(RuntimeError, match='out of memory'):
        run_predict(model, tmp_path, points, 'plot_boom')
