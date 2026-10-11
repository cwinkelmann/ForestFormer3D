"""Small config helpers shared by tools/test.py (importable without mmengine)."""


def resolve_output_dir(test_cfg, work_dir):
    """Where inference writes result .ply files.

    ``model.test_cfg.output_dir`` wins when it was set (e.g. via
    ``--cfg-options model.test_cfg.output_dir=...``); otherwise ``work_dir``.
    """
    value = test_cfg.get('output_dir', None) if test_cfg is not None else None
    return value if value else work_dir
