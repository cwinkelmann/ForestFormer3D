"""loss() reads the epoch from mmengine's MessageHub, not from an `epoch` kwarg."""
import pytest
import torch
from mmengine.config import Config
from mmengine.logging import MessageHub
from mmengine.registry import init_default_scope
from mmdet3d.registry import MODELS
from mmdet3d.structures import Det3DDataSample, PointData

from oneformer3d.structures import InstanceData_

pytestmark = pytest.mark.gpu
CONFIG = 'configs/oneformer3d_qs_radius16_qp300_2many.py'
WARMUP_KEYS = {'discriminative_loss', 'semantic_loss_bi'}


def build_model(prepare_epoch):
    init_default_scope('mmdet3d')
    cfg = Config.fromfile(CONFIG)
    cfg.model.prepare_epoch = prepare_epoch
    model = MODELS.build(cfg.model).cuda()
    model.train()
    return model


def synthetic_batch(seed=0, n_trees=3, pts_per_tree=800, n_ground=1500):
    g = torch.Generator().manual_seed(seed)
    pts, sem, inst = [], [], []
    ground = torch.rand((n_ground, 3), generator=g) * torch.tensor([12.0, 12.0, 0.2])
    pts.append(ground)
    sem.append(torch.zeros(n_ground, dtype=torch.long))
    inst.append(torch.full((n_ground,), -1, dtype=torch.long))
    for t in range(n_trees):
        centre = torch.tensor([3.0 + 3.0 * t, 6.0, 0.0])
        tree = torch.rand((pts_per_tree, 3), generator=g) * torch.tensor([1.5, 1.5, 8.0]) + centre
        pts.append(tree)
        sem.append(torch.ones(pts_per_tree, dtype=torch.long))
        inst.append(torch.full((pts_per_tree,), t, dtype=torch.long))
    points = torch.cat(pts).float().cuda()
    sem = torch.cat(sem).cuda()
    inst = torch.cat(inst).cuda()
    sample = Det3DDataSample()
    sample.gt_pts_seg = PointData(
        pts_instance_mask=inst,
        pts_semantic_mask=sem,
        instance_mask=(inst >= 0).cpu().numpy(),
        ratio_inspoint={t: 1.0 for t in range(n_trees)})
    sample.gt_instances_3d = InstanceData_(
        labels_3d=torch.ones(n_trees, dtype=torch.long).cuda())
    return {'points': [points]}, [sample]


def run_loss(model, epoch):
    MessageHub.get_current_instance().update_info('epoch', epoch)
    inputs, samples = synthetic_batch()
    return model.loss(inputs, samples)          # no epoch kwarg


def test_warmup_branch_below_prepare_epoch():
    losses = run_loss(build_model(prepare_epoch=1000), epoch=5)
    assert set(losses) == WARMUP_KEYS


def test_full_branch_above_prepare_epoch():
    losses = run_loss(build_model(prepare_epoch=1000), epoch=1500)
    assert WARMUP_KEYS < set(losses)            # criterion keys added


def test_prepare_epoch_zero_is_not_falsy():
    losses = run_loss(build_model(prepare_epoch=0), epoch=1)
    assert WARMUP_KEYS < set(losses)


def test_prepare_epoch_none_disables_gate():
    losses = run_loss(build_model(prepare_epoch=None), epoch=0)
    assert WARMUP_KEYS < set(losses)


def test_missing_hub_epoch_defaults_to_zero():
    hub = MessageHub.get_current_instance()
    if 'epoch' in hub.runtime_info:
        hub.runtime_info.pop('epoch')
    inputs, samples = synthetic_batch()
    losses = build_model(prepare_epoch=1000).loss(inputs, samples)
    assert set(losses) == WARMUP_KEYS


def test_zero_foreground_does_not_divide_by_zero():
    """All voxels predicted background -> no queries, no ZeroDivisionError."""
    model = build_model(prepare_epoch=None)
    with torch.no_grad():                       # force the binary head to 'background'
        model.BiSemantic[1].weight.zero_()
        model.BiSemantic[1].bias.copy_(torch.tensor([10.0, -10.0]))
    inputs, samples = synthetic_batch()
    losses = model.loss(inputs, samples)
    assert set(losses) == WARMUP_KEYS
