"""PLY vertex layout for result point clouds.

Kept in its own module (numpy + plyfile only, no torch / spconv / mmdet3d) so the
CPU test suite can exercise the array layout without a GPU install. The model's
``save_ply_withscore`` builds its ``PlyElement`` here and writes it as binary
little-endian; every reader in this repo goes through ``plyfile.PlyData.read``
(or ``tools/plyutils.read_ply``, which is binary-only), so both formats load.
"""
import numpy as np
from plyfile import PlyElement

#: Fields written for every result point. ``semantic_pred``/``instance_pred``
#: are int32, ``score`` is the float32 score of the point's instance (-1 for
#: unassigned points).
RESULT_PLY_DTYPE = [
    ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
    ('semantic_pred', 'i4'), ('instance_pred', 'i4'), ('score', 'f4'),
]

#: Appended when ground truth is available (labelled scans only).
RESULT_PLY_GT_DTYPE = [('semantic_gt', 'i4'), ('instance_gt', 'i4')]


def result_ply_element(points, semantic_pred, instance_pred, scores,
                       semantic_gt=None, instance_gt=None):
    """Build the ``vertex`` element of a result PLY.

    Args:
        points (array): (N, >=3) coordinates; only the first three columns are
            written, cast to float32.
        semantic_pred (array): (N,) predicted semantic class.
        instance_pred (array): (N,) predicted instance id (-1 = unassigned).
        scores (array): (N,) per-point instance score.
        semantic_gt (array, optional): (N,) ground truth semantic class.
        instance_gt (array, optional): (N,) ground truth instance id. Both GT
            arrays must be given together or the GT fields are omitted.

    Returns:
        plyfile.PlyElement: named ``vertex``, ready for ``PlyData([el], ...)``.
    """
    points = np.asarray(points)
    dtype = list(RESULT_PLY_DTYPE)
    columns = [points[:, 0], points[:, 1], points[:, 2],
               semantic_pred, instance_pred, scores]
    if semantic_gt is not None and instance_gt is not None:
        dtype += RESULT_PLY_GT_DTYPE
        columns += [semantic_gt, instance_gt]

    vertex = np.empty(points.shape[0], dtype=dtype)
    for (name, _), column in zip(dtype, columns):
        vertex[name] = column
    return PlyElement.describe(vertex, 'vertex')
