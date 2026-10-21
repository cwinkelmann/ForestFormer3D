"""`pred_inst_sem_test` vectorisation must not change a single output value.

The profile in docs/benchmarks/2026-09-23-inference-profile.md found that the
per-mask Python loop in `ForAINetV2OneFormer3D_XAwarequery.pred_inst_sem_test`
was 81 % of full-plot prediction time (~590 blocking `.item()` syncs per tile).
It was replaced by batched tensor ops. This test keeps the ORIGINAL
implementation (copied verbatim from commit 96558eb, only renamed and dedented)
next to the new one and asserts identical outputs on randomised inputs of
realistic size.

Run with `pytest -m gpu tests/gpu` inside the container.
"""
import pytest

torch = pytest.importorskip('torch')

from mmengine.config import ConfigDict

from oneformer3d.mask_matrix_nms import mask_matrix_nms
from oneformer3d.oneformer3d import ForAINetV2OneFormer3D_XAwarequery

pytestmark = pytest.mark.gpu

K = 246          # candidate masks per tile, as measured in the profile
N = 40_000       # points per tile (the profile measured ~18 k voxels)


class _Stub:
    """`pred_inst_sem_test` only ever touches `self.test_cfg`."""

    def __init__(self, test_cfg):
        self.test_cfg = test_cfg


def _test_cfg(**overrides):
    """The active config's `model.test_cfg`, minus the keys this method ignores."""
    cfg = dict(
        topk_insts=300,
        inst_score_thr=0.0,
        npoint_thr=10,
        obj_normalization=True,
        obj_normalization_thr=0.01,
        sp_score_thr=0.15,
        nms=True,
        matrix_nms_kernel='linear',
        num_sem_cls=3,
        stuff_cls=[0],
        thing_cls=[0],
    )
    cfg.update(overrides)
    return ConfigDict(cfg)


# ---------------------------------------------------------------------------
# Original implementation, verbatim from
# `git show 96558eb:oneformer3d/oneformer3d.py` (lines 2456-2557), dedented to
# module level and renamed. Do not "clean up" -- it is the reference.
# ---------------------------------------------------------------------------
def pred_inst_sem_test_original(self, pred_masks, pred_scores, #pred_labels,
              superpoints, score_threshold, sem_res, coordinates, ground_z_max, queries):
    """Predict instance masks for a single scene.

    Args:
        pred_masks (Tensor): of shape (n_queries, n_points).
        pred_scores (Tensor): of shape (n_queris, 1).
        pred_labels (Tensor): of shape (n_queries, n_instance_classes + 1).
        superpoints (Tensor): of shape (n_raw_points,).
        score_threshold (float): minimal score for predicted object.

    Returns:
        Tuple:
            Tensor: mask_preds of shape (n_preds, n_raw_points),
            Tensor: labels of shape (n_preds,),
            Tensor: scors of shape (n_preds,).
    """
    #scores = F.softmax(pred_labels, dim=-1)[:, :-1]
    #scores *= pred_scores
    scores = pred_scores

    labels = torch.arange(
        1,
        device=scores.device).unsqueeze(0).repeat(
            queries.shape[0],
            1).flatten(0, 1)

    scores, topk_idx = scores.flatten(0, 1).topk(
        min(self.test_cfg.topk_insts, queries.shape[0]), sorted=False)
    labels = labels[topk_idx]

    topk_idx = torch.div(topk_idx, 1, rounding_mode='floor') #self.num_classes, rounding_mode='floor')
    mask_pred = pred_masks
    mask_pred = mask_pred[topk_idx]
    mask_pred_sigmoid = mask_pred.sigmoid()

    queries_select = queries[topk_idx]
    if self.test_cfg.get('obj_normalization', None):
        mask_pred_thr = mask_pred_sigmoid > \
            self.test_cfg.obj_normalization_thr
        mask_scores = (mask_pred_sigmoid * mask_pred_thr).sum(1) / \
            (mask_pred_thr.sum(1) + 1e-6)
        scores = scores * mask_scores

    if self.test_cfg.get('nms', None):
        kernel = self.test_cfg.matrix_nms_kernel
        scores, labels, mask_pred_sigmoid, keep_inds = mask_matrix_nms(
            mask_pred_sigmoid, labels, scores, kernel=kernel)
    else:
        keep_inds = torch.arange(scores.shape[0], device=scores.device)

    queries_select = queries_select[keep_inds]
    mask_pred = mask_pred_sigmoid > self.test_cfg.sp_score_thr
    mask_pred = mask_pred[:, superpoints]

    # Loop through each mask
    # Ensure stuff_cls is a tensor and move it to the same device as mask_sem_res
    stuff_cls_tensor = torch.tensor(self.test_cfg.stuff_cls, device=sem_res.device)
    #for i in range(mask_pred.size(0)):
    #    mask = mask_pred[i]
    #    # Get the semantic categories of the points within the mask
    #    mask_sem_res = sem_res[mask == 1]
    #    # Check if the majority of the points' semantic categories belong to stuff_cls
    #    if torch.isin(mask_sem_res, stuff_cls_tensor).sum().item() > mask_sem_res.size(0) / 2:
    #        # If true, set the corresponding score to 0
    #        scores[i] = 0

    # Compute the binary mask for stuff_cls
    is_stuff = torch.isin(sem_res, stuff_cls_tensor).float()
    # Multiply mask_pred by the binary mask and sum along the columns
    mask_scores = (mask_pred * is_stuff).sum(dim=1)
    # Calculate the number of points in each mask
    num_points_in_mask = mask_pred.sum(dim=1)
    # Set scores to 0 where the majority of points are stuff_cls
    scores[mask_scores > (num_points_in_mask / 2)] = 0

    # Filter instances based on z values
    for i in range(mask_pred.size(0)):
        mask = mask_pred[i]
        if mask.sum().item() == 0:
            scores[i] = 0
            continue
        z_values = coordinates[mask, 2]  # Get z values where mask is True
        if z_values.numel() > 0 and z_values.min().item() > ground_z_max + 5:
            scores[i] = 0

    # score_thr
    score_mask = scores > score_threshold
    scores = scores[score_mask]
    labels = labels[score_mask]
    mask_pred = mask_pred[score_mask]
    queries_select = queries_select[score_mask]

    # npoint_thr
    mask_pointnum = mask_pred.sum(1)
    npoint_mask = mask_pointnum > self.test_cfg.npoint_thr
    scores = scores[npoint_mask]
    labels = labels[npoint_mask]
    mask_pred = mask_pred[npoint_mask]
    queries_select = queries_select[npoint_mask]

    return mask_pred, labels, scores, queries_select


def _make_inputs(seed, device, n_masks=K, n_points=N, ground_z_max=2.0,
                 all_high=False, empty_masks=True, ties=True):
    """Random but realistic inputs, in the shapes `predict_by_feat_test` passes."""
    g = torch.Generator(device='cpu').manual_seed(seed)
    # logits centred so sigmoid > sp_score_thr selects a sparse subset of points
    pred_masks = (torch.randn(n_masks, n_points, generator=g) * 3 - 3).to(device)
    pred_scores = torch.rand(n_masks, 1, generator=g).to(device)
    coordinates = torch.rand(n_points, 3, generator=g).to(device)
    coordinates[:, 2] *= 30.0                        # z in [0, 30)
    queries = torch.randn(n_masks, 32, generator=g).to(device)
    superpoints = torch.arange(n_points, device=device)
    sem_res = torch.randint(0, 3, (n_points,), generator=g).to(device)

    if empty_masks and n_masks >= 5:
        pred_masks[:5] = -50.0                       # masks with no point at all
    if all_high and n_masks >= 15:
        # masks living entirely above ground_z_max + 5
        high = coordinates[:, 2] > ground_z_max + 10
        pred_masks[5:15] = -50.0
        pred_masks[5:15][:, high] = 50.0
    if ties and n_masks >= 30:
        pred_scores[20:30] = pred_scores[20]         # score ties (NMS ordering)
        # identical z for many points -> ties inside the per-mask min
        coordinates[:max(1, n_points // 10), 2] = ground_z_max + 7.0

    return dict(pred_masks=pred_masks, pred_scores=pred_scores,
                superpoints=superpoints, sem_res=sem_res,
                coordinates=coordinates, ground_z_max=ground_z_max,
                queries=queries)


def _run_both(stub, kw, score_threshold=0.0):
    """Both versions mutate `scores` in place, so each gets its own copies."""
    def call(fn):
        return fn(stub, kw['pred_masks'].clone(), kw['pred_scores'].clone(),
                  kw['superpoints'], score_threshold, kw['sem_res'],
                  kw['coordinates'].clone(), kw['ground_z_max'],
                  kw['queries'].clone())

    return (call(pred_inst_sem_test_original),
            call(ForAINetV2OneFormer3D_XAwarequery.pred_inst_sem_test))


def _assert_identical(old, new):
    for name, a, b in zip(('mask_pred', 'labels', 'scores', 'queries_select'),
                          old, new):
        assert a.shape == b.shape, f'{name}: {a.shape} != {b.shape}'
        assert a.dtype == b.dtype, f'{name}: {a.dtype} != {b.dtype}'
        assert torch.equal(a, b), f'{name} differs'


@pytest.mark.parametrize('seed', [0, 1, 2, 17, 2024])
def test_identical_on_realistic_inputs(seed):
    stub = _Stub(_test_cfg())
    kw = _make_inputs(seed, torch.device('cuda'))
    old, new = _run_both(stub, kw)
    assert old[0].shape[0] > 0, 'these inputs should keep at least one mask'
    _assert_identical(old, new)


def test_identical_with_masks_entirely_above_ground():
    """Whole cloud above ground_z_max + 5: the height test fires for every mask."""
    stub = _Stub(_test_cfg())
    kw = _make_inputs(7, torch.device('cuda'), all_high=True)
    kw['coordinates'][:, 2] = kw['coordinates'][:, 2].clamp(min=20.0)
    old, new = _run_both(stub, kw)
    _assert_identical(old, new)
    assert old[2].numel() == 0, 'expected every mask to be dropped'


def test_identical_without_ground_points():
    """`ground_z_max` is inf when a tile has no ground: only empty masks drop."""
    stub = _Stub(_test_cfg())
    kw = _make_inputs(3, torch.device('cuda'), ground_z_max=float('inf'))
    old, new = _run_both(stub, kw)
    _assert_identical(old, new)


def test_identical_when_nothing_is_kept():
    """Edge case: a score threshold above every score keeps zero masks."""
    stub = _Stub(_test_cfg())
    kw = _make_inputs(5, torch.device('cuda'), n_masks=16, n_points=1_000)
    old, new = _run_both(stub, kw, score_threshold=1.0)
    _assert_identical(old, new)
    assert old[0].shape[0] == 0


def test_identical_with_a_single_mask():
    stub = _Stub(_test_cfg())
    kw = _make_inputs(11, torch.device('cuda'), n_masks=1, n_points=500,
                      empty_masks=False, ties=False)
    old, new = _run_both(stub, kw)
    _assert_identical(old, new)


def test_identical_when_every_mask_is_empty():
    stub = _Stub(_test_cfg())
    kw = _make_inputs(13, torch.device('cuda'), n_masks=8, n_points=200)
    kw['pred_masks'][:] = -50.0
    old, new = _run_both(stub, kw)
    _assert_identical(old, new)
    assert old[0].shape[0] == 0


def test_identical_for_a_tile_that_forces_several_chunks():
    """The new code chunks the (K, N) temporary; the result must not depend on it."""
    stub = _Stub(_test_cfg())
    kw = _make_inputs(23, torch.device('cuda'), n_masks=64, n_points=300_000)
    old, new = _run_both(stub, kw)
    _assert_identical(old, new)


def test_identical_at_a_float32_rounding_boundary():
    """The height threshold is compared in float64, exactly as the loop was.

    The loop evaluated `z_values.min().item() > ground_z_max + 5` in float64:
    `.item()` widens the float32 minimum, and `ground_z_max` is itself a float64
    widening of a float32 (`predict_by_feat_test` gets it from `.max().item()`).
    Comparing in float32 instead would round the threshold to the nearest
    float32; here it rounds *up*, so a mask whose lowest point sits exactly on
    the rounded value is dropped by the loop but kept by a float32 comparison.
    Measure-zero on real data, but it is the one input where the two forms can
    disagree, so it is pinned.
    """
    device = torch.device('cuda')
    stub = _Stub(_test_cfg())

    # ground_z_max as predict_by_feat_test produces it: .item() of a float32
    ground_z_max = torch.tensor(12.3456789, dtype=torch.float32).item()
    threshold_f32 = torch.tensor(ground_z_max + 5, dtype=torch.float32).item()
    assert threshold_f32 > ground_z_max + 5, \
        'this probe needs a threshold that float32 rounds up'

    n_masks, n_points, per_mask = 4, 200, 50
    coordinates = torch.zeros(n_points, 3, device=device)
    coordinates[:, 2] = threshold_f32 + 10.0
    # mask 0's lowest point sits exactly on the float32-rounded threshold
    coordinates[:per_mask, 2] = threshold_f32

    pred_masks = torch.full((n_masks, n_points), -50.0, device=device)
    for i in range(n_masks):
        pred_masks[i, i * per_mask:(i + 1) * per_mask] = 50.0
    pred_scores = torch.full((n_masks, 1), 0.9, device=device)
    queries = torch.randn(n_masks, 32, device=device)

    kw = dict(pred_masks=pred_masks, pred_scores=pred_scores,
              superpoints=torch.arange(n_points, device=device),
              sem_res=torch.ones(n_points, dtype=torch.long, device=device),
              coordinates=coordinates, ground_z_max=ground_z_max,
              queries=queries)
    old, new = _run_both(stub, kw)
    _assert_identical(old, new)
    assert old[0].shape[0] == 0, 'the loop drops every mask on this input'
