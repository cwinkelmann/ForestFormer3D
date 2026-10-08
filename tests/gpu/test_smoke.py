"""Phase 0 gate: the CUDA image can build the model, run one loss() step and one predict().

Runs only inside the Docker image: `docker/smoke.sh` or, in the container,
`pytest -m gpu tests/gpu/test_smoke.py`.

Phase 1 changed two things here: loss() lost its `epoch` kwarg, and predict()
selects full-plot tiling with test_cfg.full_plot instead of 'test' in lidar_path.

The synthetic plot's footprint (~5 m) is kept well inside a single cylinder's inner
band (radius 16 m minus the 0.5 m edge filter) so no untrained mask ever touches the
edge, while the 4 m tiling step still produces a 2x2 tile lattice.
"""
from pathlib import Path

import numpy as np
import pytest
import torch

pytestmark = pytest.mark.gpu

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG = REPO_ROOT / "configs" / "oneformer3d_qs_radius16_qp300_2many.py"
N_POINTS = 5000
# The .ply written by full-plot predict() is named after this file's stem.
LIDAR_PATH = "data/ForAINetV2/plots/smoke_plot.ply"


def make_synthetic_plot(n_points=N_POINTS, seed=0):
    """A flat ground disc (radius 2.5 m) with two fake trees.

    x/y is scaled by 0.25 relative to the "natural" 10 m-radius layout (ground disc
    radius 2.5 m, trees at (-1, -1) and (1, 1), stem/crown x/y jitter scaled too; z is
    untouched) so the whole plot's footprint stays inside the 15.5 m inner band of
    every tile and no untrained mask ever gets discarded by the edge filter.

    Returns points float32[N,3]; semantic int64[N] (0 ground, 1 wood, 2 leaf);
    instance int64[N] (-1 ground, 0 and 1 for the trees).
    """
    rng = np.random.default_rng(seed)
    n_ground = n_points // 2
    n_tree = (n_points - n_ground) // 2

    r = 10.0 * np.sqrt(rng.random(n_ground))
    a = 2.0 * np.pi * rng.random(n_ground)
    ground = np.stack([r * np.cos(a), r * np.sin(a), rng.normal(0.0, 0.05, n_ground)], 1)
    pts = [ground]
    sem = [np.zeros(n_ground, dtype=np.int64)]
    ins = [np.full(n_ground, -1, dtype=np.int64)]

    for inst_id, (cx, cy) in enumerate([(-4.0, -4.0), (4.0, 4.0)]):
        n_stem = n_tree // 4
        n_crown = n_tree - n_stem
        stem = np.stack([cx + rng.normal(0.0, 0.1, n_stem),
                         cy + rng.normal(0.0, 0.1, n_stem),
                         12.0 * rng.random(n_stem)], 1)
        direction = rng.normal(size=(n_crown, 3))
        direction /= np.linalg.norm(direction, axis=1, keepdims=True)
        crown = np.array([cx, cy, 12.0]) + direction * 3.0 * np.cbrt(rng.random((n_crown, 1)))
        pts += [stem, crown]
        sem += [np.ones(n_stem, dtype=np.int64), np.full(n_crown, 2, dtype=np.int64)]
        ins += [np.full(n_stem, inst_id, dtype=np.int64), np.full(n_crown, inst_id, dtype=np.int64)]

    points = np.concatenate(pts).astype(np.float32)
    points[:, :2] *= 0.25  # shrink x/y only, so the footprint clears the edge band
    semantic = np.concatenate(sem)
    instance = np.concatenate(ins)
    assert points.shape == (n_points, 3)
    return points, semantic, instance


def make_sample(semantic, instance):
    """A Det3DDataSample shaped like the output of Pack3DDetInputs_ for one plot."""
    from mmdet3d.structures import Det3DDataSample, PointData
    from oneformer3d.structures import InstanceData_

    sample = Det3DDataSample()
    sample.set_metainfo(dict(lidar_path=LIDAR_PATH))

    gt_pts_seg = PointData()
    gt_pts_seg.pts_semantic_mask = torch.from_numpy(semantic).cuda()
    gt_pts_seg.pts_instance_mask = torch.from_numpy(instance).cuda()
    gt_pts_seg.instance_mask = semantic != 0          # numpy bool: loss() calls torch.from_numpy
    gt_pts_seg.ratio_inspoint = {0: 1.0, 1: 1.0}      # both trees fully inside the "crop"
    sample.gt_pts_seg = gt_pts_seg

    gt_instances_3d = InstanceData_()
    gt_instances_3d.labels_3d = torch.tensor([1, 1]).cuda()  # semantic class per instance
    sample.gt_instances_3d = gt_instances_3d

    sample.eval_ann_info = dict(pts_semantic_mask=semantic, pts_instance_mask=instance)
    return sample


@pytest.fixture(scope="module")
def plot():
    return make_synthetic_plot()


@pytest.fixture(scope="module")
def model():
    import oneformer3d  # noqa: F401  registers the model, decoder, criterion classes
    from mmdet3d.registry import MODELS
    from mmengine.config import Config
    from mmengine.registry import init_default_scope

    cfg = Config.fromfile(str(CONFIG))
    init_default_scope("mmdet3d")
    torch.manual_seed(0)
    np.random.seed(0)
    net = MODELS.build(cfg.model).cuda()

    # Untrained BiSemantic head: make every voxel "wood" so query sampling has candidates.
    head = net.BiSemantic[1]  # Seq(MLP, Linear(num_channels, 2), LogSoftmax)
    with torch.no_grad():
        head.weight.zero_()
        head.bias.copy_(torch.tensor([-1.0, 1.0], device=head.bias.device))
    return net


def test_loss_is_finite_and_backpropagates(model, plot):
    points, semantic, instance = plot
    model.train()
    inputs = dict(points=[torch.from_numpy(points).cuda()])
    samples = [make_sample(semantic, instance)]

    # epoch > prepare_epoch (1000 in the config) exercises the query decoder and the
    # instance criterion, not only the discriminative / binary-semantic losses.
    from mmengine.logging import MessageHub
    MessageHub.get_current_instance().update_info('epoch', 2000)
    losses = model.loss(inputs, samples)

    assert "discriminative_loss" in losses and "semantic_loss_bi" in losses
    tensors = {k: v for k, v in losses.items() if torch.is_tensor(v)}
    assert len(tensors) > 2, f"criterion losses missing: {sorted(losses)}"
    for name, value in tensors.items():
        assert torch.isfinite(value).all(), f"{name} is not finite: {value}"

    total = sum(v.sum() for v in tensors.values())
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    optimizer.zero_grad()
    total.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads, "no parameter received a gradient"
    assert all(torch.isfinite(g).all() for g in grads)
    optimizer.step()


def test_predict_writes_nonempty_instance_map(model, plot, tmp_path):
    from plyfile import PlyData

    points, semantic, instance = plot
    model.eval()
    model.test_cfg["output_dir"] = str(tmp_path)  # what tools/test.py does with --work-dir
    model.test_cfg["full_plot"] = True            # tiled whole-plot inference
    # untrained scores are negative raw logits; both thresholds must let them through
    # so this gate tests plumbing, not quality
    model.test_cfg["inst_score_thr"] = -1e6
    model.test_cfg["score_th"] = -1e6

    inputs = dict(points=[torch.from_numpy(points).cuda()])
    samples = [make_sample(semantic, instance)]
    with torch.no_grad():
        out = model.predict(inputs, samples)

    assert len(out) == 1
    ply_path = tmp_path / "smoke_plot.ply"
    assert ply_path.exists(), f"predict() did not write {ply_path}"

    vertex = PlyData.read(str(ply_path))["vertex"]
    assert len(vertex) == N_POINTS
    instance_pred = np.asarray(vertex["instance_pred"])
    semantic_pred = np.asarray(vertex["semantic_pred"])
    assert (instance_pred >= 0).sum() > 0, "no instance survived tiling and merging"
    assert set(np.unique(semantic_pred)).issubset({-1, 0, 1, 2})
    assert np.array_equal(np.asarray(vertex["instance_gt"]), instance)
