"""UnifiedSegMetric: id 0 is a real tree, -1 is ignored, empty scenes score zero."""
import numpy as np
import pytest
from mmengine.logging import MMLogger

from oneformer3d.unified_metric import UnifiedSegMetric

pytestmark = pytest.mark.gpu


def metric():
    MMLogger.get_instance('test_metric')
    return UnifiedSegMetric(thing_class_inds=[1, 2], stuff_class_inds=[0])


def three_tree_scene():
    """30 tree points (ids 0,1,2 x 10) + 10 ground points (-1). Ids start at 0.

    This is the *crop* (validation) convention: the pipeline already
    normalized the GT, so ground is -1 and instance id 0 is a real tree.
    """
    sem = np.array([0] * 10 + [1] * 30)
    inst = np.array([-1] * 10 + [0] * 10 + [1] * 10 + [2] * 10)
    return sem, inst


def three_tree_scene_raw():
    """The same scene in the *raw* full-plot convention from eval_ann_info.

    Ground and unannotated points carry treeID 0; real trees carry arbitrary
    positive treeIDs and there is no -1 anywhere.
    """
    sem = np.array([0] * 10 + [1] * 30)
    inst = np.array([0] * 10 + [7] * 10 + [3] * 10 + [42] * 10)
    return sem, inst


def as_result(sem_gt, inst_gt, sem_pred, inst_pred):
    return ({'pts_semantic_mask': sem_gt, 'pts_instance_mask': inst_gt},
            {'pts_semantic_mask': [sem_pred, sem_pred],
             'pts_instance_mask': [inst_pred, inst_pred]})


def test_instance_id_zero_is_a_valid_tree():
    sem, inst = three_tree_scene()
    m = metric().compute_metrics([as_result(sem, inst, sem.copy(), inst.copy())])
    assert m['mPrecision'] == pytest.approx(1.0)
    assert m['mRecall'] == pytest.approx(1.0)
    assert m['mMUCov'] == pytest.approx(1.0)
    assert m['mPQ'] == pytest.approx(1.0)


def test_raw_full_plot_gt_is_normalized_before_scoring():
    """Raw GT (ground/unannotated == treeID 0, no -1) must score identically."""
    sem, inst_raw = three_tree_scene_raw()
    _, inst_norm = three_tree_scene()
    # The prediction always uses the normalized convention (that is what
    # predict() emits); only the GT arrives raw here.
    m = metric().compute_metrics([as_result(sem, inst_raw, sem.copy(), inst_norm.copy())])
    assert m['mPrecision'] == pytest.approx(1.0)
    assert m['mRecall'] == pytest.approx(1.0)
    assert m['mMUCov'] == pytest.approx(1.0)
    assert m['mPQ'] == pytest.approx(1.0)


def test_raw_gt_ignores_vegetation_without_tree_id():
    """Vegetation with raw treeID 0 is unannotated, not a giant instance."""
    sem = np.array([0] * 10 + [2] * 10 + [1] * 10)
    inst_raw = np.array([0] * 10 + [0] * 10 + [5] * 10)
    # Prediction: one tree covering exactly the annotated tree.
    inst_pred = np.array([-1] * 10 + [-1] * 10 + [0] * 10)
    m = metric().compute_metrics([as_result(sem, inst_raw, sem.copy(), inst_pred)])
    assert m['mPrecision'] == pytest.approx(1.0)
    assert m['mRecall'] == pytest.approx(1.0)
    assert m['mMUCov'] == pytest.approx(1.0)


def test_unused_ctor_args_are_gone():
    with pytest.raises(TypeError):
        UnifiedSegMetric(thing_class_inds=[1, 2], stuff_class_inds=[0], min_num_points=1)


def test_scene_without_predictions_counts_zero_coverage():
    sem, inst = three_tree_scene()
    perfect = as_result(sem, inst, sem.copy(), inst.copy())
    empty = as_result(sem, inst, sem.copy(), np.full_like(inst, -1))
    m = metric().compute_metrics([perfect, empty])
    assert m['mMUCov'] == pytest.approx(0.5)     # mean(1.0, 0.0), not mean(1.0)
    assert m['mMWCov'] == pytest.approx(0.5)
    assert m['mRecall'] == pytest.approx(0.5)


def test_scene_without_gt_is_skipped():
    sem, inst = three_tree_scene()
    m = metric().compute_metrics(
        [as_result(sem, inst, sem.copy(), inst.copy()),
         (None, {'pts_semantic_mask': [sem, sem], 'pts_instance_mask': [inst, inst]})])
    assert m['mPrecision'] == pytest.approx(1.0)
