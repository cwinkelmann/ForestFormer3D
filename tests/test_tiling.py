import importlib.util
import math

import pytest

torch_missing = importlib.util.find_spec('torch') is None
pytestmark = pytest.mark.skipif(torch_missing, reason='torch not installed')

if not torch_missing:
    import torch
    from oneformer3d.tiling import (SemanticVotes, degenerate_region_reason,
                                    generate_cylindrical_regions,
                                    merge_instances_by_score, relabel_contiguous,
                                    sample_region)


def test_lower_scoring_mask_does_not_steal_points():
    n = 20
    masks = torch.zeros((2, n), dtype=torch.bool)
    masks[0, 0:10] = True                 # tree A, score 0.9
    masks[1, 8:18] = True                 # tree B, score 0.5, overlaps 2/10 = 20%
    labels, kept = merge_instances_by_score(masks, torch.tensor([0.9, 0.5]), 0.3)
    assert kept.tolist() == [0, 1]
    assert labels[0:10].tolist() == [0] * 10        # points 8 and 9 stay with A
    assert labels[10:18].tolist() == [1] * 8
    assert labels[18:].tolist() == [-1, -1]


def test_mask_overlapping_half_is_dropped():
    n = 20
    masks = torch.zeros((2, n), dtype=torch.bool)
    masks[0, 0:10] = True
    masks[1, 5:15] = True                 # 5 of 10 points already taken = 50% > 0.3
    labels, kept = merge_instances_by_score(masks, torch.tensor([0.9, 0.4]), 0.3)
    assert kept.tolist() == [0]
    assert labels[10:15].tolist() == [-1] * 5


def test_merge_visits_by_descending_score_regardless_of_input_order():
    n = 12
    masks = torch.zeros((2, n), dtype=torch.bool)
    masks[0, 0:8] = True                  # low score first in the input
    masks[1, 4:12] = True                 # high score
    labels, kept = merge_instances_by_score(masks, torch.tensor([0.2, 0.8]), 0.5)
    assert kept.tolist() == [1, 0]
    assert labels[4:12].tolist() == [0] * 8          # label 0 <-> kept[0] == mask 1
    assert labels[0:4].tolist() == [1] * 4


def test_merge_accepts_index_lists():
    # Non-boundary overlap: mask 0 has 4 points, 1 already assigned (0.25 <= 0.3), so
    # it is kept and claims only its unassigned points. (A prior version of this test
    # used a 1-of-3 overlap, i.e. 1/3 > 0.3, which should have been dropped -- that was
    # an invalid boundary case, not a real bug in merge_instances_by_score.)
    labels, kept = merge_instances_by_score(
        [torch.tensor([0, 1, 2, 4]), torch.tensor([2, 3])], torch.tensor([0.5, 0.9]),
        overlap_threshold=0.3, num_points=5)
    assert kept.tolist() == [1, 0]
    assert labels.tolist() == [1, 1, 0, 0, 1]


def test_merge_with_no_masks():
    labels, kept = merge_instances_by_score([], torch.tensor([]), 0.3, num_points=4)
    assert labels.tolist() == [-1] * 4 and kept.numel() == 0


def test_merge_with_no_dense_masks_keeps_the_masks_device():
    masks = torch.zeros((0, 5), dtype=torch.bool)
    labels, kept = merge_instances_by_score(masks, torch.tensor([]), 0.3)
    assert labels.device == masks.device
    assert labels.tolist() == [-1] * 5
    assert kept.numel() == 0


def test_exact_cap_sampling_returns_all_points():
    pts = torch.arange(30, dtype=torch.float32).reshape(10, 3)
    idx = torch.arange(10)
    out_pts, out_idx = sample_region(pts, idx, num_points=10)
    assert torch.equal(out_pts, pts) and torch.equal(out_idx, idx)


def test_over_cap_sampling_is_without_replacement():
    pts = torch.arange(300, dtype=torch.float32).reshape(100, 3)
    idx = torch.arange(100)
    g = torch.Generator().manual_seed(0)
    out_pts, out_idx = sample_region(pts, idx, num_points=40, generator=g)
    assert out_pts.shape == (40, 3)
    assert len(set(out_idx.tolist())) == 40
    assert torch.equal(out_pts, pts[out_idx])


def test_region_grid_covers_bounding_box_corners():
    radius, step = 16.0, 4.0
    xy = torch.tensor([[0.0, 0.0], [30.0, 30.0], [30.0, 0.0], [0.0, 30.0], [11.0, 17.0]])
    regions = generate_cylindrical_regions(xy, radius, step)
    assert regions.shape == (64, 2)                   # 8 x 8 lattice: 0,4,...,28
    assert torch.equal(regions[0], torch.tensor([0.0, 0.0]))
    for corner in xy[:4]:
        d = torch.sqrt(((regions - corner) ** 2).sum(1))
        assert d.min() <= radius


def test_region_grid_rejects_bad_step():
    with pytest.raises(ValueError):
        generate_cylindrical_regions(torch.zeros((3, 2)), 16.0, 0.0)


def test_semantic_votes_label_spaces():
    votes = SemanticVotes(num_points=5, num_classes=3)
    votes.add(torch.tensor([0, 0, 1]), torch.tensor([2, 2, 1]))            # 3-class votes
    votes.add_binary(torch.tensor([0, 2, 2, 3, 3, 3]),
                     torch.tensor([True, True, True, False, False, True]))
    out = votes.resolve()
    assert out[0].item() == 2      # 3-class majority wins, binary votes ignored
    assert out[1].item() == 1
    assert out[2].item() == 1      # only binary votes, foreground majority
    assert out[3].item() == 0      # only binary votes, background majority
    assert out[4].item() == -1     # no votes at all


def test_semantic_votes_rejects_out_of_range_labels():
    votes = SemanticVotes(num_points=2, num_classes=3)
    with pytest.raises(ValueError):
        votes.add(torch.tensor([0]), torch.tensor([3]))


def test_relabel_contiguous():
    out = relabel_contiguous(torch.tensor([-1, 7, 3, 7, -1, 3]))
    assert out.tolist() == [-1, 1, 0, 1, -1, 0]


# --- degenerate_region_reason ------------------------------------------------
# The guard that decides which cylinder regions of a production tile reach the
# sparse backbone at all. See docs/known-issues.md ("Degenerate cylinder regions").

VOXEL = 0.2


def _blob(n, seed=0, scale=(4.0, 4.0, 10.0)):
    """A normal, three-dimensional region: n points filling a box."""
    g = torch.Generator().manual_seed(seed)
    return torch.rand((n, 3), generator=g) * torch.tensor(scale)


def test_too_few_points_is_degenerate():
    reason = degenerate_region_reason(_blob(10), VOXEL, min_points=64)
    assert reason is not None
    assert 'only 10 points' in reason and 'min_region_points=64' in reason


def test_min_points_threshold_is_inclusive():
    """`min_points` points are enough; one fewer is not."""
    assert degenerate_region_reason(_blob(64), VOXEL, min_points=64) is None
    assert degenerate_region_reason(_blob(63), VOXEL, min_points=64) is not None


def test_a_single_repeated_point_is_degenerate():
    points = torch.zeros((100, 3)) + torch.tensor([3.0, -7.0, 1.0])
    reason = degenerate_region_reason(points, VOXEL, min_points=64)
    assert reason is not None
    assert 'voxel span [1, 1, 1]' in reason


def test_a_vertical_line_is_degenerate():
    z = torch.linspace(0.0, 30.0, 100)
    points = torch.stack([torch.full_like(z, 2.0), torch.full_like(z, 5.0), z], dim=1)
    reason = degenerate_region_reason(points, VOXEL, min_points=64)
    assert reason is not None
    assert 'a point or a line' in reason


def test_a_horizontal_line_is_degenerate():
    x = torch.linspace(0.0, 30.0, 100)
    points = torch.stack([x, torch.full_like(x, 5.0), torch.full_like(x, 5.0)], dim=1)
    assert degenerate_region_reason(points, VOXEL, min_points=64) is not None


def test_a_flat_plane_is_usable():
    """`min_spatial_shape` pads the thin axis, so a ground-only region is fine."""
    g = torch.Generator().manual_seed(1)
    points = torch.rand((500, 3), generator=g) * torch.tensor([20.0, 20.0, 0.0])
    assert degenerate_region_reason(points, VOXEL, min_points=64) is None


def test_a_normal_blob_is_usable():
    assert degenerate_region_reason(_blob(5000), VOXEL, min_points=64) is None


def test_span_follows_the_backbone_quantization():
    """Two points 0.02 m apart across a voxel boundary occupy two voxels.

    The span must be `floor(max/v) - floor(min/v) + 1`, not `floor(extent/v) + 1`,
    which would call this column one voxel thick and flag the region as a line.
    """
    x = torch.tensor([0.19, 0.21]).repeat(50)
    y = torch.tensor([0.19, 0.21]).repeat(50)
    z = torch.linspace(0.0, 10.0, 100)
    points = torch.stack([x, y, z], dim=1)
    assert degenerate_region_reason(points, VOXEL, min_points=64) is None


def test_extra_feature_columns_are_ignored():
    points = torch.cat([_blob(200), torch.rand((200, 2))], dim=1)
    assert degenerate_region_reason(points, VOXEL, min_points=64) is None
