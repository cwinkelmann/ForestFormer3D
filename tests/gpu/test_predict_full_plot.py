"""Full-plot predict(): merged arrays of length N in pred_pts_seg and a .ply on disk."""
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
    import oneformer3d  # noqa: F401  registers the model / decoder / criterion classes

    init_default_scope('mmdet3d')
    cfg = Config.fromfile(CONFIG)
    cfg.model.test_cfg.output_dir = str(output_dir)
    # untrained scores are negative raw logits; both thresholds must let them through
    # so this gate tests plumbing, not quality
    cfg.model.test_cfg.inst_score_thr = -1e6
    cfg.model.test_cfg.score_th = -1e6
    for k, v in test_cfg_overrides.items():
        cfg.model.test_cfg[k] = v
    torch.manual_seed(0)
    np.random.seed(0)
    model = MODELS.build(cfg.model).cuda()
    # Untrained BiSemantic head: make every voxel "wood" so query sampling has candidates.
    head = model.BiSemantic[1]
    with torch.no_grad():
        head.weight.zero_()
        head.bias.copy_(torch.tensor([-1.0, 1.0], device=head.bias.device))
    model.eval()
    return model


def synthetic_plot(n_ground=6000, n_trees=4, pts_per_tree=1500, seed=0):
    """4 trees on a ground plane; x/y scaled by 0.25 so the whole footprint stays
    inside the 15.5 m inner band of every tile (radius 16 m minus the 0.5 m edge
    filter) and no untrained mask ever gets discarded by the edge filter, while the
    4 m tiling step still produces a multi-tile lattice.
    """
    g = torch.Generator().manual_seed(seed)
    pts = [torch.rand((n_ground, 3), generator=g) * torch.tensor([30.0, 30.0, 0.3])]
    sem = [torch.zeros(n_ground, dtype=torch.int64)]
    inst = [torch.full((n_ground,), -1, dtype=torch.int64)]
    for t in range(n_trees):
        centre = torch.tensor([5.0 + 6.0 * t, 15.0, 0.0])
        pts.append(torch.rand((pts_per_tree, 3), generator=g) * torch.tensor([2.0, 2.0, 10.0]) + centre)
        sem.append(torch.ones(pts_per_tree, dtype=torch.int64))
        inst.append(torch.full((pts_per_tree,), t + 1, dtype=torch.int64))
    points = torch.cat(pts).float()
    points[:, :2] *= 0.25  # shrink x/y only, so the footprint clears the edge band
    return points, torch.cat(sem).numpy(), torch.cat(inst).numpy()


def make_sample(tmp_path, with_gt):
    points, sem_gt, inst_gt = synthetic_plot()
    sample = Det3DDataSample()
    sample.set_metainfo({'lidar_path': str(tmp_path / 'points' / 'plot_a.bin')})
    sample.eval_ann_info = ({'pts_semantic_mask': sem_gt, 'pts_instance_mask': inst_gt}
                            if with_gt else None)
    return {'points': [points.cuda()]}, [sample], points.shape[0]


@pytest.mark.parametrize('with_gt', [True, False])
def test_full_plot_predict_returns_merged_arrays_and_writes_ply(tmp_path, with_gt):
    out_dir = tmp_path / 'out'
    model = build_model(out_dir)
    inputs, samples, n = make_sample(tmp_path, with_gt)
    with torch.no_grad():
        result = model.predict(inputs, samples)
    seg = result[0].pred_pts_seg
    assert len(seg['pts_instance_mask'][1]) == n
    assert len(seg['pts_semantic_mask'][1]) == n
    assert len(seg['instance_scores']) == n
    inst = np.asarray(seg['pts_instance_mask'][1])
    sem = np.asarray(seg['pts_semantic_mask'][1])
    assert inst.min() >= -1 and sem.min() >= -1 and sem.max() < 3
    assert (inst >= 0).any()                               # not a zero-instance pass
    ids = np.unique(inst[inst >= 0])
    assert ids.tolist() == list(range(len(ids)))          # contiguous 0..K-1
    assert not np.any((sem == 0) & (inst >= 0))            # ground carries no instance

    ply_path = out_dir / 'plot_a.ply'
    assert ply_path.exists()
    vertex = PlyData.read(str(ply_path))['vertex']
    assert vertex.count == n
    names = set(vertex.data.dtype.names)
    assert {'x', 'y', 'z', 'semantic_pred', 'instance_pred', 'score'} <= names
    assert ({'semantic_gt', 'instance_gt'} <= names) == with_gt
    assert np.array_equal(vertex['instance_pred'], inst)


def test_crop_mode_still_works(tmp_path):
    model = build_model(tmp_path, full_plot=False)
    inputs, samples, n = make_sample(tmp_path, with_gt=True)
    with torch.no_grad():
        result = model.predict(inputs, samples)
    seg = result[0].pred_pts_seg
    assert len(seg['pts_semantic_mask'][1]) == n
    assert not (tmp_path / 'plot_a.ply').exists()
