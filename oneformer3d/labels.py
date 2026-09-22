"""Pure-numpy helpers for normalizing instance-segmentation ground truth.

This module intentionally imports only numpy so it can be unit-tested on a
machine without torch / mmengine / mmdet3d.
"""
from __future__ import annotations

import numpy as np


def _compact(instance: np.ndarray) -> tuple[np.ndarray, dict[int, int]]:
    """Core of :func:`compact_instance_ids`.

    Returns the compacted array together with the ``{old_id: new_id}`` mapping
    that produced it, so callers can re-key side tables (``ratio_inspoint``)
    with exactly the same mapping.
    """
    instance = np.asarray(instance).astype(np.int64)
    out = np.full(instance.shape, -1, dtype=np.int64)
    valid = instance >= 0
    if not np.any(valid):
        return out, {}
    valid_vals = instance[valid]
    unique, first_idx, inverse = np.unique(valid_vals, return_index=True, return_inverse=True)
    # `unique` is sorted by value, not by first appearance in `valid_vals`;
    # re-rank so the id that appears first in the array becomes 0, etc.
    appearance_order = np.argsort(first_idx, kind='stable')
    new_id = np.empty(len(unique), dtype=np.int64)
    new_id[appearance_order] = np.arange(len(unique))
    out[valid] = new_id[inverse]
    return out, {int(old): int(new) for old, new in zip(unique, new_id)}


def compact_instance_ids(instance: np.ndarray) -> np.ndarray:
    """Renumber instance ids to 0..K-1 in order of first appearance.

    ``-1`` is preserved wherever it occurs (treated as "no instance"); every
    other value, including 0, is a valid instance id and gets remapped.

    Args:
        instance: (N,) integer array. Values < 0 are treated as -1.

    Returns:
        (N,) int64 array with -1 preserved and other ids compacted.
    """
    return _compact(instance)[0]


def compact_instance_ids_with_ratio(
        instance: np.ndarray,
        ratio_inspoint: dict | None = None) -> tuple[np.ndarray, dict | None]:
    """:func:`compact_instance_ids`, re-keying ``ratio_inspoint`` alongside it.

    ``ratio_inspoint`` maps instance id -> fraction of that instance's points
    that survived the cylinder crop. Every transform that subsamples points
    renumbers the ids, so the dict has to follow: ``filter_stuff_masks`` and
    ``get_iou_with_crop`` look the ratios up *by id* (strictly), and a stale
    key silently scales the wrong instance's IoU or raises.

    Entries whose id no longer occurs in ``instance`` are dropped.

    Returns:
        ``(new_mask, new_ratio)``. ``new_ratio`` is ``None`` if and only if
        ``ratio_inspoint`` was ``None``; the return is always a 2-tuple so
        call sites can unpack unconditionally.
    """
    out, id_map = _compact(instance)
    if ratio_inspoint is None:
        return out, None
    new_ratio = {id_map[int(old)]: float(ratio)
                 for old, ratio in ratio_inspoint.items() if int(old) in id_map}
    return out, new_ratio


def normalize_instance_gt(semantic: np.ndarray, instance: np.ndarray,
                          raw: bool = True) -> np.ndarray:
    """Mark non-instance points as -1 and compact the remaining instance ids.

    A point is *not* a real instance (and becomes -1) when:
      - ``semantic == 0`` (the ground class), or
      - ``instance < 0`` (already marked unannotated), or
      - ``raw`` is True and ``instance == 0``.

    The ``instance == 0`` rule only applies when ``raw=True``: the dataset's
    raw treeID convention uses 0 to mean "unannotated", so id 0 is not a real
    tree. After one pass through this function (or through the crop
    pipeline), valid instance ids start at 0 and 0 *is* a real instance id;
    callers on that path must pass ``raw=False`` or every surviving instance
    would collide with "unannotated" and get wiped out on a second pass.

    Callers on the raw path (metric / final_eval on test-set ground truth,
    which still uses the raw treeID convention) must use ``raw=True``.
    Callers downstream of the crop pipeline (ids already compacted, 0 is a
    real instance) must use ``raw=False``. With ``raw=False`` this function is
    idempotent: calling it again on its own output returns an equal array.

    Args:
        semantic: (N,) semantic class ids; 0 is the ground class.
        instance: (N,) raw or already-normalized instance ids.
        raw: whether ``instance == 0`` means "unannotated" (True, default) or
            is a valid instance id (False).

    Returns:
        (N,) int64 array: -1 for non-instance points, else 0..K-1 in order of
        first appearance.
    """
    semantic = np.asarray(semantic)
    instance = np.asarray(instance).astype(np.int64)
    if semantic.shape != instance.shape:
        raise ValueError(
            f'semantic {semantic.shape} and instance {instance.shape} shapes differ')
    unannotated = (semantic == 0) | (instance < 0)
    if raw:
        unannotated = unannotated | (instance == 0)
    masked = np.where(unannotated, -1, instance)
    return compact_instance_ids(masked)
