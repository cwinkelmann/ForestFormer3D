"""GPU-only tests: run with `pytest -m gpu tests/gpu` inside the Docker image (docker/smoke.sh).

Pytest imports every test module before applying `-m`, so on a machine without torch
(the Mac) the modules in this directory are not collected at all.
"""
import importlib.util
from pathlib import Path

import pytest

collect_ignore_glob = [] if importlib.util.find_spec("torch") else ["test_*.py"]

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG = str(REPO_ROOT / 'configs' / 'oneformer3d_qs_radius16_qp300_2many.py')


@pytest.fixture(scope="session", autouse=True)
def require_cuda():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.fail("tests/gpu needs a CUDA device; run them through docker/smoke.sh")


@pytest.fixture
def mm_caplog(caplog):
    """`caplog`, but it also sees `mmengine.print_log(..., logger='current')`.

    `MMLogger` does not propagate to the root logger, so pytest's own handler
    never sees the model's warnings; attach it to the current instance instead.
    """
    import logging

    from mmengine.logging import MMLogger

    logger = MMLogger.get_current_instance()
    caplog.handler.setLevel(logging.NOTSET)
    logger.addHandler(caplog.handler)
    try:
        yield caplog
    finally:
        logger.removeHandler(caplog.handler)


@pytest.fixture
def build_model():
    """Factory: ``build_model(output_dir, **test_cfg_overrides) -> model``.

    Builds the released full-plot config untrained on the GPU, with both score
    thresholds opened up (untrained scores are negative raw logits) so the tests
    gate plumbing, not quality. Shared by every test that drives ``predict()``.
    """
    import numpy as np
    import torch
    from mmengine.config import Config
    from mmengine.registry import init_default_scope
    from mmdet3d.registry import MODELS

    def _build(output_dir, **test_cfg_overrides):
        import oneformer3d  # noqa: F401  registers model / decoder / criterion classes

        init_default_scope('mmdet3d')
        cfg = Config.fromfile(CONFIG)
        cfg.model.test_cfg.output_dir = str(output_dir)
        cfg.model.test_cfg.inst_score_thr = -1e6
        cfg.model.test_cfg.score_th = -1e6
        for k, v in test_cfg_overrides.items():
            cfg.model.test_cfg[k] = v
        torch.manual_seed(0)
        np.random.seed(0)
        model = MODELS.build(cfg.model).cuda()
        # Untrained BiSemantic head: make every voxel "wood" so query sampling
        # has candidates.
        head = model.BiSemantic[1]
        with torch.no_grad():
            head.weight.zero_()
            head.bias.copy_(torch.tensor([-1.0, 1.0], device=head.bias.device))
        model.eval()
        return model

    return _build


@pytest.fixture
def synthetic_plot():
    """Factory: ``synthetic_plot(...) -> (points, sem_gt, inst_gt)``.

    ``n_trees`` trees on a ground plane; x/y scaled by 0.25 so the whole footprint
    stays inside the 15.5 m inner band of every tile (radius 16 m minus the 0.5 m
    edge filter) and no untrained mask is discarded by the edge filter, while the
    4 m tiling step still produces a multi-tile lattice.
    """
    import torch

    def _make(n_ground=6000, n_trees=4, pts_per_tree=1500, seed=0):
        g = torch.Generator().manual_seed(seed)
        pts = [torch.rand((n_ground, 3), generator=g) * torch.tensor([30.0, 30.0, 0.3])]
        sem = [torch.zeros(n_ground, dtype=torch.int64)]
        inst = [torch.full((n_ground,), -1, dtype=torch.int64)]
        for t in range(n_trees):
            centre = torch.tensor([5.0 + 6.0 * t, 15.0, 0.0])
            pts.append(torch.rand((pts_per_tree, 3), generator=g)
                       * torch.tensor([2.0, 2.0, 10.0]) + centre)
            sem.append(torch.ones(pts_per_tree, dtype=torch.int64))
            inst.append(torch.full((pts_per_tree,), t + 1, dtype=torch.int64))
        points = torch.cat(pts).float()
        points[:, :2] *= 0.25  # shrink x/y only, so the footprint clears the edge band
        return points, torch.cat(sem).numpy(), torch.cat(inst).numpy()

    return _make
