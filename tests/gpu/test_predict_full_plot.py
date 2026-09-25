"""Full-plot predict(): merged arrays of length N in pred_pts_seg and a .ply on disk.

`build_model` and `synthetic_plot` are fixtures from tests/gpu/conftest.py.
"""
import numpy as np
import pytest
import torch
from mmdet3d.structures import Det3DDataSample
from plyfile import PlyData

pytestmark = pytest.mark.gpu


def make_sample(tmp_path, with_gt, synthetic_plot):
    points, sem_gt, inst_gt = synthetic_plot()
    sample = Det3DDataSample()
    sample.set_metainfo({'lidar_path': str(tmp_path / 'points' / 'plot_a.bin')})
    sample.eval_ann_info = ({'pts_semantic_mask': sem_gt, 'pts_instance_mask': inst_gt}
                            if with_gt else None)
    return {'points': [points.cuda()]}, [sample], points.shape[0]


@pytest.mark.parametrize('with_gt', [True, False])
def test_full_plot_predict_returns_merged_arrays_and_writes_ply(
        tmp_path, with_gt, build_model, synthetic_plot):
    out_dir = tmp_path / 'out'
    model = build_model(out_dir)
    inputs, samples, n = make_sample(tmp_path, with_gt, synthetic_plot)
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


def test_crop_mode_still_works(tmp_path, build_model, synthetic_plot):
    model = build_model(tmp_path, full_plot=False)
    inputs, samples, n = make_sample(tmp_path, True, synthetic_plot)
    with torch.no_grad():
        result = model.predict(inputs, samples)
    seg = result[0].pred_pts_seg
    assert len(seg['pts_semantic_mask'][1]) == n
    assert not (tmp_path / 'plot_a.ply').exists()


def test_region_step_factor_reaches_the_lattice_and_still_segments(
        tmp_path, monkeypatch, build_model, synthetic_plot):
    """`model.test_cfg.region_step_factor` sets the cylinder pitch to radius * factor.

    0.25 (the default) must keep today's `radius / 4`; 0.5 must halve the pitch
    in each axis, i.e. lay down roughly a quarter of the cylinders, and still
    produce a non-empty instance map.
    """
    import oneformer3d.oneformer3d as ofm

    steps = []
    real = ofm.generate_cylindrical_regions

    def spy(points_xy, radius, step):
        steps.append(step)
        return real(points_xy, radius, step)

    monkeypatch.setattr(ofm, 'generate_cylindrical_regions', spy)

    model = build_model(tmp_path / 'out', region_step_factor=0.5)
    inputs, samples, n = make_sample(tmp_path, False, synthetic_plot)
    with torch.no_grad():
        result = model.predict(inputs, samples)

    assert steps == [model.radius * 0.5]
    n_coarse = len(real(inputs['points'][0][:, :2], model.radius, model.radius * 0.5))
    n_default = len(real(inputs['points'][0][:, :2], model.radius, model.radius * 0.25))
    assert n_coarse < n_default

    inst = np.asarray(result[0].pred_pts_seg['pts_instance_mask'][1])
    assert len(inst) == n
    assert (inst >= 0).any()
