"""Pure-torch helpers for tiled full-plot inference.

This module intentionally imports only torch so it can be unit-tested on a
machine without mmengine / mmdet3d / spconv.
"""
from __future__ import annotations

import torch


def generate_cylindrical_regions(points_xy: torch.Tensor, radius: float,
                                 step: float) -> torch.Tensor:
    """Lattice of cylinder centres covering the xy bounding box.

    Centres are ``x_min + k * step`` for every ``k`` with the value ``<= x_max``
    (same for y): the lattice the original ``predict()`` loop produced.

    Args:
        points_xy: (N, >=2) tensor on any device; only the first two columns are used.
        radius: cylinder radius in metres. ``step`` must not exceed it.
        step: lattice spacing in metres, > 0.

    Returns:
        (M, 2) float32 tensor of centres on CPU, x-major order.
    """
    if step <= 0:
        raise ValueError(f'step must be > 0, got {step}')
    if step > radius:
        raise ValueError(f'step ({step}) larger than radius ({radius}) leaves gaps')
    if points_xy.numel() == 0:
        return torch.zeros((0, 2), dtype=torch.float32)
    xy = points_xy[:, :2].detach().to('cpu', torch.float64)
    mins = xy.min(0).values
    maxs = xy.max(0).values
    counts = torch.floor((maxs - mins) / step).long() + 1
    xs = mins[0] + step * torch.arange(int(counts[0]), dtype=torch.float64)
    ys = mins[1] + step * torch.arange(int(counts[1]), dtype=torch.float64)
    gx, gy = torch.meshgrid(xs, ys, indexing='ij')
    return torch.stack((gx.reshape(-1), gy.reshape(-1)), dim=1).to(torch.float32)


def sample_region(points: torch.Tensor, indices: torch.Tensor, num_points: int,
                  generator: torch.Generator | None = None
                  ) -> tuple[torch.Tensor, torch.Tensor]:
    """Cap a region at ``num_points`` points, sampling without replacement.

    Handles ``N < num_points``, ``N == num_points`` (both returned unchanged) and
    ``N > num_points``. ``generator`` must be a CPU generator.
    """
    n = points.shape[0]
    if indices.shape[0] != n:
        raise ValueError(f'points ({n}) and indices ({indices.shape[0]}) differ in length')
    if n <= num_points:
        return points, indices
    perm = torch.randperm(n, generator=generator)[:num_points]
    return points[perm.to(points.device)], indices[perm.to(indices.device)]


def merge_instances_by_score(masks, scores: torch.Tensor, overlap_threshold: float,
                             num_points: int | None = None
                             ) -> tuple[torch.Tensor, torch.Tensor]:
    """Assign every point to at most one instance, best score first.

    Masks are visited by descending score. A mask only claims points that are
    still unassigned; it is dropped entirely when the fraction of its points
    that are already assigned exceeds ``overlap_threshold``.

    Args:
        masks: ``BoolTensor`` of shape (K, N), or a sequence of K 1-D index
            tensors (global point indices); the latter needs ``num_points``.
        scores: (K,) scores.
        overlap_threshold: drop a mask when ``taken.mean() > threshold``.
        num_points: N, required for the index-list form.

    Returns:
        labels: (N,) long, -1 = unassigned, else 0..J-1 in visiting order.
        kept: (J,) long, original mask indices; label ``j`` belongs to ``kept[j]``.
    """
    if isinstance(masks, torch.Tensor):
        if masks.dim() != 2 or masks.dtype != torch.bool:
            raise ValueError('dense masks must be a bool tensor of shape (K, N)')
        num_points = masks.shape[1]
        device = masks.device
        index_lists = [masks[k].nonzero(as_tuple=False).flatten() for k in range(masks.shape[0])]
    else:
        if num_points is None:
            raise ValueError('num_points is required when masks is a list of index tensors')
        index_lists = [torch.as_tensor(m, dtype=torch.long).flatten() for m in masks]
        device = index_lists[0].device if index_lists else torch.device('cpu')
    scores = torch.as_tensor(scores, dtype=torch.float32).flatten()
    if len(index_lists) != scores.numel():
        raise ValueError(f'{len(index_lists)} masks but {scores.numel()} scores')
    labels = torch.full((num_points,), -1, dtype=torch.long, device=device)
    taken = torch.zeros(num_points, dtype=torch.bool, device=device)
    kept: list[int] = []
    order = torch.sort(scores.cpu(), descending=True, stable=True).indices
    for k in order.tolist():
        pts = index_lists[k].to(device)
        if pts.numel() == 0:
            continue
        already = taken[pts]
        if already.float().mean().item() > overlap_threshold:
            continue
        labels[pts[~already]] = len(kept)
        taken[pts] = True
        kept.append(k)
    return labels, torch.tensor(kept, dtype=torch.long)


class SemanticVotes:
    """Per-point semantic votes with two separate label spaces.

    ``add`` records votes from the 3-class head, ``add_binary`` records
    foreground/background votes from the binary head (used for tiles where the
    decoder did not run). ``resolve`` prefers 3-class votes; points with only
    binary votes get 1 (wood) when foreground wins, else 0 (ground); points
    without any vote get -1. Counts live on CPU.
    """

    def __init__(self, num_points: int, num_classes: int = 3):
        self.num_points = int(num_points)
        self.num_classes = int(num_classes)
        self.counts = torch.zeros((self.num_points, self.num_classes), dtype=torch.int32)
        self.binary = torch.zeros((self.num_points, 2), dtype=torch.int32)

    def _check(self, indices: torch.Tensor, values: torch.Tensor, upper: int):
        if indices.shape != values.shape:
            raise ValueError('indices and values must have the same shape')
        if values.numel() and (values.min() < 0 or values.max() >= upper):
            raise ValueError(f'labels must be in [0, {upper}), got [{values.min()}, {values.max()}]')

    def add(self, indices: torch.Tensor, labels: torch.Tensor) -> None:
        indices = torch.as_tensor(indices, dtype=torch.long).cpu().flatten()
        labels = torch.as_tensor(labels, dtype=torch.long).cpu().flatten()
        self._check(indices, labels, self.num_classes)
        self.counts.index_put_((indices, labels),
                               torch.ones_like(indices, dtype=torch.int32), accumulate=True)

    def add_binary(self, indices: torch.Tensor, is_foreground: torch.Tensor) -> None:
        indices = torch.as_tensor(indices, dtype=torch.long).cpu().flatten()
        fg = torch.as_tensor(is_foreground).cpu().flatten().long()
        self._check(indices, fg, 2)
        self.binary.index_put_((indices, fg),
                               torch.ones_like(indices, dtype=torch.int32), accumulate=True)

    def resolve(self) -> torch.Tensor:
        out = torch.full((self.num_points,), -1, dtype=torch.long)
        has_class_votes = self.counts.sum(1) > 0
        out[has_class_votes] = self.counts[has_class_votes].argmax(1)
        only_binary = ~has_class_votes & (self.binary.sum(1) > 0)
        out[only_binary] = (self.binary[only_binary, 1] > self.binary[only_binary, 0]).long()
        return out


def relabel_contiguous(labels: torch.Tensor) -> torch.Tensor:
    """Renumber instance ids to 0..K-1 (ascending old id); -1 stays -1."""
    out = torch.full_like(labels, -1)
    valid = labels >= 0
    if valid.any():
        _, inverse = torch.unique(labels[valid], return_inverse=True)
        out[valid] = inverse
    return out
