"""`model.test_cfg.region_batch`: several cylinders per backbone pass, same results.

`build_model` and `synthetic_plot` are fixtures from tests/gpu/conftest.py.
"""
import numpy as np
import pytest
import torch
from mmdet3d.structures import Det3DDataSample

from oneformer3d.tiling import split_batched_inverse_mapping

pytestmark = pytest.mark.gpu


def test_split_batched_inverse_mapping_ranks_rows_within_each_region():
    # 5 voxel rows, regions interleaved: rows 0,2,3 belong to region 1, rows 1,4 to region 0
    row_batch = torch.tensor([1, 0, 1, 1, 0])
    # region 0 has 3 points (rows 4, 1, 1), region 1 has 4 points (rows 0, 3, 2, 0)
    inverse = torch.tensor([4, 1, 1, 0, 3, 2, 0])
    r0, r1 = split_batched_inverse_mapping(row_batch, inverse, [3, 4])
    assert r0.tolist() == [1, 0, 0]          # row 4 is region 0's second row, row 1 its first
    assert r1.tolist() == [0, 2, 1, 0]       # rows 0, 2, 3 -> ranks 0, 1, 2


def test_split_batched_inverse_mapping_refuses_crossed_regions():
    row_batch = torch.tensor([0, 1])
    with pytest.raises(ValueError, match='another region'):
        split_batched_inverse_mapping(row_batch, torch.tensor([1, 1]), [1, 1])
    with pytest.raises(ValueError, match='mapped points'):
        split_batched_inverse_mapping(row_batch, torch.tensor([0, 1]), [1, 2])


def test_split_batched_inverse_mapping_single_region_is_identity():
    row_batch = torch.zeros(6, dtype=torch.long)
    inverse = torch.tensor([5, 3, 3, 0, 2, 1, 4])
    (only,) = split_batched_inverse_mapping(row_batch, inverse, [7])
    assert torch.equal(only, inverse)


def _predict(model, tmp_path, points, name):
    sample = Det3DDataSample()
    sample.set_metainfo({'lidar_path': str(tmp_path / 'points' / f'{name}.bin')})
    sample.eval_ann_info = None
    with torch.no_grad():
        result = model.predict({'points': [points.cuda()]}, [sample])
    seg = result[0].pred_pts_seg
    return (np.asarray(seg['pts_semantic_mask'][1]), np.asarray(seg['pts_instance_mask'][1]),
            np.asarray(seg['instance_scores']))


def test_batched_backbone_features_match_the_single_region_features(build_model, synthetic_plot):
    """The voxel rows come back in another order inside a batch, but after the
    split inverse mapping every POINT sees the same backbone features (up to
    float rounding) as when its region runs alone."""
    import spconv.pytorch as spconv

    model = build_model('/tmp/unused')
    points, _, _ = synthetic_plot()
    pts = points.cuda()
    regions = []
    for cx, cy in [(3.0, 3.5), (5.0, 3.5), (4.0, 4.0)]:
        idx = torch.where(((pts[:, 0] - cx) ** 2 + (pts[:, 1] - cy) ** 2) <= model.radius ** 2)[0]
        pc2, _ = model.grid_sample(pts[idx], idx, 0.2)
        regions.append(pc2)
    with torch.no_grad():
        singles = []
        for pc in regions:
            c, f, inv, sh = model.collate([pc])
            singles.append(model.extract_feat(spconv.SparseConvTensor(f, c, sh, 1))[0][inv.long()])
        c, f, inv, sh = model.collate(regions)
        feats = model.extract_feat(spconv.SparseConvTensor(f, c, sh, len(regions)))
        maps = split_batched_inverse_mapping(c[:, 0], inv, [p.shape[0] for p in regions])
    assert len(feats) == len(regions)
    for single, feat, m in zip(singles, feats, maps):
        batched = feat[m]
        assert batched.shape == single.shape
        assert torch.allclose(batched, single, atol=1e-5, rtol=1e-5)


def test_region_batch_gives_the_single_region_result(tmp_path, monkeypatch, build_model, synthetic_plot):
    """Batched and one-at-a-time inference agree on the labelling to within the
    model's own run-to-run noise, and the batched run really does push several
    regions through the backbone at once.

    The sparse backbone is not bit-reproducible: two single-region runs of the
    untrained test model on this plot disagree on ~0.3 % of the points (ties in
    near-random logits flip with the summation order), and batched vs single
    measured 0.23 % -- the same noise. The point-feature test above is the exact
    check; this one guards the plumbing end to end.
    """
    points, _, _ = synthetic_plot()
    batch_sizes = {}

    def spy(model, key):
        real = model.extract_feat
        sizes = batch_sizes.setdefault(key, [])

        def wrapped(x):
            sizes.append(int(x.batch_size))
            return real(x)

        monkeypatch.setattr(model, 'extract_feat', wrapped)

    single = build_model(tmp_path / 'single', region_batch=1)
    spy(single, 'single')
    sem1, inst1, _ = _predict(single, tmp_path, points, 'plot_single')
    sem1b, inst1b, _ = _predict(single, tmp_path, points, 'plot_single_again')

    batched = build_model(tmp_path / 'batched', region_batch=4)
    spy(batched, 'batched')
    sem4, inst4, _ = _predict(batched, tmp_path, points, 'plot_batched')

    assert set(batch_sizes['single']) == {1}
    assert max(batch_sizes['batched']) == 4
    # the single model ran twice: every region segmented once per run
    assert 2 * sum(batch_sizes['batched']) == sum(batch_sizes['single'])
    assert 2 * len(batch_sizes['batched']) < len(batch_sizes['single'])

    noise = max(float((sem1 != sem1b).mean()), float((inst1 != inst1b).mean()))
    assert (inst1 >= 0).any()
    assert inst4.max() == inst1.max()                                   # same number of instances
    assert float((sem4 != sem1).mean()) <= max(0.01, 3 * noise)
    assert float((inst4 != inst1).mean()) <= max(0.01, 3 * noise)


def test_region_batch_must_be_positive(tmp_path, build_model, synthetic_plot):
    points, _, _ = synthetic_plot()
    model = build_model(tmp_path / 'out', region_batch=0)
    with pytest.raises(ValueError, match='region_batch'):
        _predict(model, tmp_path, points, 'plot_zero')


def _backbone_regions(model, points, centres):
    """Backbone features + inverse maps of a few cylinders, each run alone (as a list)."""
    import spconv.pytorch as spconv

    pts = points.cuda()
    feats, maps = [], []
    with torch.no_grad():
        for cx, cy in centres:
            idx = torch.where(((pts[:, 0] - cx) ** 2 + (pts[:, 1] - cy) ** 2) <= model.radius ** 2)[0]
            pc2, _ = model.grid_sample(pts[idx], idx, 0.2)
            c, f, inv, sh = model.collate([pc2])
            feats.append(model.extract_feat(spconv.SparseConvTensor(f, c, sh, 1))[0])
            maps.append(inv.long())
    return feats, maps


def test_forward_padded_matches_the_per_scene_decoder(build_model, synthetic_plot):
    """The padded batched decoder gives every scene the masks and scores the per-scene
    decoder gives it, for scenes of different sizes and query counts."""
    model = build_model('/tmp/unused')
    points, _, _ = synthetic_plot()
    feats, _ = _backbone_regions(model, points, [(3.0, 3.5), (5.0, 3.5), (4.0, 4.0)])
    g = torch.Generator(device='cpu').manual_seed(0)
    queries = [f[torch.randperm(f.shape[0], generator=g)[:n].cuda()] for f, n in zip(feats, (300, 120, 37))]
    with torch.no_grad():
        ref = model.decoder(feats, [q.clone() for q in queries])
        out = model.decoder.forward_padded(feats, [q.clone() for q in queries])
    assert len(out['masks']) == 3 and len(out['aux_outputs']) == len(ref['aux_outputs'])
    for i in range(3):
        assert out['masks'][i].shape == ref['masks'][i].shape
        assert torch.allclose(out['masks'][i], ref['masks'][i], atol=2e-3, rtol=1e-3), \
            f'scene {i}: max |diff| {(out["masks"][i] - ref["masks"][i]).abs().max().item()}'
        assert torch.allclose(out['scores'][i], ref['scores'][i], atol=2e-3, rtol=1e-3)
    # a single scene takes the ordinary path
    with torch.no_grad():
        one = model.decoder.forward_padded(feats[:1], [queries[0].clone()])
        one_ref = model.decoder(feats[:1], [queries[0].clone()])
    assert torch.equal(one['masks'][0], one_ref['masks'][0])


def test_batched_query_sampling_picks_query_point_num_per_region(tmp_path, monkeypatch, build_model, synthetic_plot):
    """One fps call per batch still yields min(query_point_num, n_tree_points) queries per
    region, each drawn from that region's own tree points."""
    import oneformer3d.oneformer3d as ofm

    model = build_model(tmp_path / 'out', region_batch=4)
    seen = []
    real = model.decoder.forward_padded

    def spy(feats, queries):
        seen.append([(f.shape[0], q.shape[0]) for f, q in zip(feats, queries)])
        return real(feats, queries)

    monkeypatch.setattr(model.decoder, 'forward_padded', spy)
    points, _, _ = synthetic_plot()
    _predict(model, tmp_path, points, 'plot_fps')
    assert seen, 'the batched decoder path never ran'
    for batch in seen:
        assert len(batch) > 1
        for n_points, n_queries in batch:
            assert 1 <= n_queries <= min(model.query_point_num, n_points)


def test_batched_decoder_can_be_switched_off(tmp_path, monkeypatch, build_model, synthetic_plot):
    model = build_model(tmp_path / 'out', region_batch=4, batched_decoder=False)
    called = []
    monkeypatch.setattr(model.decoder, 'forward_padded', lambda *a, **k: called.append(1))
    points, _, _ = synthetic_plot()
    sem, inst, _ = _predict(model, tmp_path, points, 'plot_nopad')
    assert not called and (inst >= 0).any()
