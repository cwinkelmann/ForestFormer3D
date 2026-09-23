"""Result PLYs are written as binary little-endian with unchanged field values.

`save_ply_withscore` used to build its vertex array with a per-row Python tuple
comprehension and write ASCII (`numpy.savetxt` over every point): 5-10 s per
100 m tile, see docs/benchmarks/2026-09-23-inference-profile.md. The array part
now lives in `oneformer3d/ply_io.py` (numpy + plyfile, no torch) so it can be
tested here on the CPU. These tests pin the field names, dtypes and values
against the old row-wise construction and check that plyfile reads the binary
file back unchanged.
"""
import numpy as np
import pytest

pytest.importorskip('plyfile')
from plyfile import PlyData  # noqa: E402

ply_io = pytest.importorskip('oneformer3d.ply_io')

DTYPE = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
         ('semantic_pred', 'i4'), ('instance_pred', 'i4'), ('score', 'f4')]
GT_DTYPE = DTYPE + [('semantic_gt', 'i4'), ('instance_gt', 'i4')]


def _legacy_vertex(points, semantic_pred, instance_pred, scores,
                   semantic_gt=None, instance_gt=None):
    """The pre-vectorisation construction, kept as the reference."""
    dtype = list(DTYPE)
    if semantic_gt is not None and instance_gt is not None:
        dtype += [('semantic_gt', 'i4'), ('instance_gt', 'i4')]
        return np.array(
            [tuple(points[i]) + (semantic_pred[i], instance_pred[i], scores[i],
                                 semantic_gt[i], instance_gt[i])
             for i in range(points.shape[0])], dtype=dtype)
    return np.array(
        [tuple(points[i]) + (semantic_pred[i], instance_pred[i], scores[i])
         for i in range(points.shape[0])], dtype=dtype)


def _sample(n=500, seed=0, float64=False):
    rng = np.random.default_rng(seed)
    points = rng.uniform(-50, 50, size=(n, 3))
    if not float64:
        points = points.astype(np.float32)
    semantic_pred = rng.integers(0, 3, size=n).astype(np.int32)
    instance_pred = rng.integers(-1, 40, size=n).astype(np.int32)
    scores = rng.uniform(0, 1, size=n).astype(np.float32)
    scores[instance_pred < 0] = -1.0
    return points, semantic_pred, instance_pred, scores


def test_fields_and_dtypes_unchanged():
    points, sem, inst, score = _sample()
    el = ply_io.result_ply_element(points, sem, inst, score)
    assert el.name == 'vertex'
    assert el.data.dtype == np.dtype(DTYPE)


def test_values_match_the_legacy_row_wise_construction():
    points, sem, inst, score = _sample(seed=1)
    new = ply_io.result_ply_element(points, sem, inst, score).data
    old = _legacy_vertex(points, sem, inst, score)
    assert new.dtype == old.dtype
    for name, _ in DTYPE:
        assert np.array_equal(new[name], old[name]), name


def test_values_match_legacy_for_float64_points():
    """`points` may arrive as float64; both paths cast to float32 the same way."""
    points, sem, inst, score = _sample(seed=2, float64=True)
    new = ply_io.result_ply_element(points, sem, inst, score).data
    old = _legacy_vertex(points, sem, inst, score)
    for name, _ in DTYPE:
        assert np.array_equal(new[name], old[name]), name


def test_ground_truth_fields_are_appended():
    points, sem, inst, score = _sample(seed=3)
    sem_gt = (sem + 1).astype(np.int32)
    inst_gt = (inst + 2).astype(np.int32)
    new = ply_io.result_ply_element(points, sem, inst, score, sem_gt,
                                    inst_gt).data
    old = _legacy_vertex(points, sem, inst, score, sem_gt, inst_gt)
    assert new.dtype == np.dtype(GT_DTYPE)
    for name, _ in GT_DTYPE:
        assert np.array_equal(new[name], old[name]), name


@pytest.mark.parametrize('with_gt', [False, True])
def test_binary_round_trip(tmp_path, with_gt):
    points, sem, inst, score = _sample(n=2000, seed=4)
    extra = {}
    if with_gt:
        extra = dict(semantic_gt=(sem + 1).astype(np.int32),
                     instance_gt=(inst + 2).astype(np.int32))
    el = ply_io.result_ply_element(points, sem, inst, score,
                                   extra.get('semantic_gt'),
                                   extra.get('instance_gt'))
    path = tmp_path / 'result.ply'
    PlyData([el], text=False, byte_order='<').write(str(path))

    assert path.read_bytes()[:64].splitlines()[1] == b'format binary_little_endian 1.0'

    back = PlyData.read(str(path))['vertex'].data
    assert back.dtype.names == el.data.dtype.names
    assert np.array_equal(back['x'], points[:, 0].astype(np.float32))
    assert np.array_equal(back['y'], points[:, 1].astype(np.float32))
    assert np.array_equal(back['z'], points[:, 2].astype(np.float32))
    assert np.array_equal(back['semantic_pred'], sem)
    assert np.array_equal(back['instance_pred'], inst)
    assert np.array_equal(back['score'], score)
    for name, value in extra.items():
        assert np.array_equal(back[name], value)


def test_binary_and_ascii_files_carry_the_same_values(tmp_path):
    """Switching the encoding must not change a single value a reader sees."""
    points, sem, inst, score = _sample(n=300, seed=5)
    el = ply_io.result_ply_element(points, sem, inst, score)
    binary, ascii_ = tmp_path / 'b.ply', tmp_path / 'a.ply'
    PlyData([el], text=False, byte_order='<').write(str(binary))
    PlyData([el], text=True).write(str(ascii_))

    a = PlyData.read(str(ascii_))['vertex'].data
    b = PlyData.read(str(binary))['vertex'].data
    for name, _ in DTYPE:
        assert np.array_equal(a[name], b[name]), name


def test_empty_cloud():
    points = np.zeros((0, 3), dtype=np.float32)
    empty = np.zeros((0,), dtype=np.int32)
    el = ply_io.result_ply_element(points, empty, empty,
                                   np.zeros((0,), dtype=np.float32))
    assert el.data.shape == (0,)
    assert el.data.dtype == np.dtype(DTYPE)
