"""Degenerate cylinder regions must not abort full-plot inference.

A 100 m ALS sub-tile with almost nothing in it (water, bare ground, a handful of
stray returns) produced cylinder regions whose voxels all vanished inside the
spconv UNet's stride-2 downsampling, and the ``ValueError`` killed the whole
``tools/test.py`` process -- with it every other sub-tile of the km tile. Predict
must now skip such a region, and still write a result PLY for the scan.
"""
from pathlib import Path

import numpy as np
import pytest
import torch
from mmengine.config import Config
from mmengine.registry import init_default_scope
from mmdet3d.registry import MODELS
from mmdet3d.structures import Det3DDataSample
from plyfile import PlyData

pytestmark = pytest.mark.gpu

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG = str(REPO_ROOT / 'configs' / 'oneformer3d_qs_radius16_qp300_2many.py')


def build_model(output_dir, **test_cfg_overrides):
    """Same fixture as tests/gpu/test_predict_full_plot.py, with a coarse lattice.

    ``region_step_factor=1.0`` puts the cylinder centres 16 m apart, so a cluster
    at the origin and an outlier 48 m away land in separate regions and the test
    runs a handful of forward passes instead of a few hundred.
    """
    import oneformer3d  # noqa: F401  registers the model / decoder / criterion classes

    init_default_scope('mmdet3d')
    cfg = Config.fromfile(CONFIG)
    cfg.model.test_cfg.output_dir = str(output_dir)
    # untrained scores are negative raw logits; both thresholds must let them through
    cfg.model.test_cfg.inst_score_thr = -1e6
    cfg.model.test_cfg.score_th = -1e6
    cfg.model.test_cfg.region_step_factor = 1.0
    for k, v in test_cfg_overrides.items():
        cfg.model.test_cfg[k] = v
    torch.manual_seed(0)
    np.random.seed(0)
    model = MODELS.build(cfg.model).cuda()
    head = model.BiSemantic[1]
    with torch.no_grad():
        head.weight.zero_()
        head.bias.copy_(torch.tensor([-1.0, 1.0], device=head.bias.device))
    model.eval()
    return model


def healthy_cluster(n_ground=4000, n_trees=2, pts_per_tree=1500, seed=0):
    """A small plot at the origin: ground plus two trees, footprint ~7.5 x 7.5 m."""
    g = torch.Generator().manual_seed(seed)
    pts = [torch.rand((n_ground, 3), generator=g) * torch.tensor([30.0, 30.0, 0.3])]
    for t in range(n_trees):
        centre = torch.tensor([5.0 + 6.0 * t, 15.0, 0.0])
        pts.append(torch.rand((pts_per_tree, 3), generator=g)
                   * torch.tensor([2.0, 2.0, 10.0]) + centre)
    points = torch.cat(pts).float()
    points[:, :2] *= 0.25
    return points


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


def test_degenerate_region_does_not_abort_the_scan(tmp_path):
    """A normal cluster plus one region holding three nearly collinear points."""
    out_dir = tmp_path / 'out'
    model = build_model(out_dir)
    points = torch.cat([healthy_cluster(), collinear_trio()])
    n = points.shape[0]

    seg = run_predict(model, tmp_path, points, 'plot_trio')

    inst = np.asarray(seg['pts_instance_mask'][1])
    sem = np.asarray(seg['pts_semantic_mask'][1])
    assert len(inst) == n and len(sem) == n
    # the healthy part is still segmented
    assert (inst[:-3] >= 0).any()
    # the skipped region contributed nothing: its points stay nodata
    assert (inst[-3:] == -1).all()
    assert (sem[-3:] == -1).all()

    vertex = read_ply(out_dir / 'plot_trio.ply', n)
    assert np.array_equal(vertex['instance_pred'], inst)
    assert np.array_equal(vertex['semantic_pred'], sem)


def test_scan_with_only_five_points_writes_an_unlabelled_ply(tmp_path):
    """No region survives: the scan still gets a complete, all-nodata result."""
    out_dir = tmp_path / 'out'
    model = build_model(out_dir)
    points = torch.tensor([[0.0, 0.0, 0.0],
                           [1.0, 0.5, 2.0],
                           [2.0, 1.0, 4.0],
                           [3.0, 1.5, 6.0],
                           [4.0, 2.0, 8.0]], dtype=torch.float32)

    seg = run_predict(model, tmp_path, points, 'plot_five')

    inst = np.asarray(seg['pts_instance_mask'][1])
    sem = np.asarray(seg['pts_semantic_mask'][1])
    assert inst.tolist() == [-1] * 5
    assert sem.tolist() == [-1] * 5

    vertex = read_ply(out_dir / 'plot_five.ply', 5)
    assert (np.asarray(vertex['instance_pred']) == -1).all()
    assert (np.asarray(vertex['semantic_pred']) == -1).all()


def test_spconv_value_error_is_caught_per_region(tmp_path, monkeypatch):
    """With the pre-filter disabled, a ValueError out of spconv only loses its region."""
    import oneformer3d.oneformer3d as ofm

    out_dir = tmp_path / 'out'
    model = build_model(out_dir)
    points = torch.cat([healthy_cluster(), collinear_trio()])
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
    seg = run_predict(model, tmp_path, points, 'plot_flaky')

    assert raised, 'the fake spconv failure never fired'
    inst = np.asarray(seg['pts_instance_mask'][1])
    assert len(inst) == n
    assert (inst[:-3] >= 0).any()
    assert (inst[-3:] == -1).all()
    read_ply(out_dir / 'plot_flaky.ply', n)


def test_other_exceptions_are_not_swallowed(tmp_path, monkeypatch):
    """Only ValueError is treated as a degenerate region."""
    model = build_model(tmp_path / 'out')
    points = healthy_cluster()

    def boom(x):
        raise RuntimeError('CUDA out of memory')

    monkeypatch.setattr(model, 'extract_feat', boom)
    with pytest.raises(RuntimeError, match='out of memory'):
        run_predict(model, tmp_path, points, 'plot_boom')
