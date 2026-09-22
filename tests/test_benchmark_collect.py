# tests/test_benchmark_collect.py
"""CPU-only tests for benchmark/collect.py against synthetic final_eval, scalars.json and
mmengine log fixtures. numpy only, no torch, no mmengine."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("collect", REPO / "benchmark" / "collect.py")
collect = importlib.util.module_from_spec(spec)
spec.loader.exec_module(collect)

FINAL_EVAL_BLOCK = """Semantic Segmentation oAcc: {oacc}
Semantic Segmentation mAcc: 0.88
Semantic Segmentation IoU: [0.0, 0.95, 0.8, 0.7]
Semantic Segmentation mIoU: {miou}

Binary Semantic Segmentation oAcc: 0.97
Binary Semantic Segmentation mAcc: 0.95
Binary Semantic Segmentation IoU: [0.0, 0.9, 0.9]
Binary Semantic Segmentation mIoU: 0.9

Instance Segmentation for Offset:
Instance Segmentation MUCov: [0.7]
Instance Segmentation mMUCov: 0.7
Instance Segmentation MWCov: [0.72]
Instance Segmentation mMWCov: 0.72
Instance Segmentation Precision: [0.81]
Instance Segmentation mPrecision: 0.81
Instance Segmentation Recall: [0.79]
Instance Segmentation mRecall: 0.79
Instance Segmentation F1 score: {f1}
Instance Segmentation RQ: [1 1]
Instance Segmentation meanRQ: 1.0
Instance Segmentation SQ: [0.8 0.9]
Instance Segmentation meanSQ: 0.85
Instance Segmentation PQ: [0.8 0.9]
Instance Segmentation meanPQ: 0.85
Instance Segmentation PQ star: [0.8 0.9]
Instance Segmentation mean PQ star: 0.85
Instance Segmentation RQ (things): [1]
Instance Segmentation meanRQ (things): 1.0
Instance Segmentation SQ (things): [0.9]
Instance Segmentation meanSQ (things): 0.9
Instance Segmentation PQ (things): [0.9]
Instance Segmentation meanPQ (things): 0.9
Instance Segmentation RQ (stuff): [1]
Instance Segmentation meanRQ (stuff): 1.0
Instance Segmentation SQ (stuff): [0.8]
Instance Segmentation meanSQ (stuff): 0.8
Instance Segmentation PQ (stuff): [0.8]
Instance Segmentation meanPQ (stuff): 0.8
"""


def write_final_eval(d: Path, f1: float, miou: float = 0.81, oacc: float = 0.9, two_blocks: bool = False):
    """Write evaluation_total_test.txt the way tools/final_eval.py does: it opens the file
    with mode 'w' (see tools/final_eval.py line 65), so a normal run leaves exactly ONE block.
    two_blocks=True hand-builds a stale multi-block file (e.g. left over from an older version
    of the script) to exercise collect.py's defensive 'last block wins' parsing."""
    d.mkdir(parents=True, exist_ok=True)
    text = ""
    if two_blocks:
        text += FINAL_EVAL_BLOCK.format(f1=0.1, miou=0.1, oacc=0.1)
    text += FINAL_EVAL_BLOCK.format(f1=f1, miou=miou, oacc=oacc)
    (d / "evaluation_total_test.txt").write_text(text)


def write_training(work: Path, epochs: int, loss0: float, f1s: dict, nan_at: int | None = None,
                   inf_at: int | None = None, ts0="2026/09/23 00:00:00", sec_per_epoch=60):
    """23 iterations per epoch, one train record per epoch (mmengine logs at end of epoch
    when len(dataloader) <= logger interval), val record every 20 epochs with step=epoch."""
    import datetime as dt
    ts_dir = work / "20260923_000000"
    vis = ts_dir / "vis_data"
    vis.mkdir(parents=True)
    t0 = dt.datetime.strptime(ts0, "%Y/%m/%d %H:%M:%S")
    scalars, log = [], []
    for e in range(1, epochs + 1):
        if nan_at == e:
            loss = float("nan")
        elif inf_at == e:
            loss = float("inf")
        else:
            loss = loss0 / e
        scalars.append({"lr": 1e-4, "data_time": 0.05, "loss": loss, "time": 2.5, "epoch": e,
                        "memory": 20000, "step": 23 * e})
        t = t0 + dt.timedelta(seconds=sec_per_epoch * e)
        loss_s = "nan" if loss != loss else f"{loss:.4f}"
        log.append(f"{t:%Y/%m/%d %H:%M:%S} - mmengine - INFO - Epoch(train)  [{e}][23/23]  "
                   f"lr: 1.0000e-04  eta: 0:00:00  time: 2.5000  data_time: 0.0500  memory: 20000  loss: {loss_s}")
        if e in f1s:
            scalars.append({"mIoU": 0.8, "mIoU_binary": 0.9, "mMWCov": 0.6, "mMUCov": 0.6, "mPrecision": 0.7,
                            "mRecall": 0.7, "F1": f1s[e], "mPQ": 0.5, "mSQ": 0.7, "mRQ": 0.6,
                            "data_time": 0.1, "time": 3.0, "step": e})
            log.append(f"{t:%Y/%m/%d %H:%M:%S} - mmengine - INFO - Epoch(val) [{e}][15/15]    F1: {f1s[e]:.4f}")
    (vis / "scalars.json").write_text("\n".join(json.dumps(s) for s in scalars) + "\n")
    (ts_dir / "20260923_000000.log").write_text("\n".join(log) + "\n")


@pytest.fixture
def root(tmp_path):
    wd = tmp_path / "work_dirs"
    write_final_eval(wd / "bench-release-fixed", f1=0.82)
    write_final_eval(wd / "bench-release-old", f1=0.80)
    write_training(wd / "bench-old-200", 200, 10.0, {20: 0.30, 40: 0.40, 200: 0.55})
    write_training(wd / "bench-fixed-200", 200, 9.0, {20: 0.35, 40: 0.45, 200: 0.60})
    write_final_eval(wd / "bench-old-200" / "test", f1=0.50)
    write_final_eval(wd / "bench-fixed-200" / "test", f1=0.58)
    return tmp_path


def test_parse_final_eval_one_block(root):
    """Normal case: tools/final_eval.py truncates the file, so there is exactly one block."""
    m = collect.parse_final_eval(root / "work_dirs/bench-release-old/evaluation_total_test.txt")
    assert m["F1"] == pytest.approx(0.80)
    assert m["oAcc"] == pytest.approx(0.9)
    assert m["mIoU"] == pytest.approx(0.81)


def test_parse_final_eval_last_block_wins(tmp_path):
    """Defensive case: a hand-built stale multi-block file must resolve to the LAST block."""
    d = tmp_path / "bench-release-fixed"
    write_final_eval(d, f1=0.82, two_blocks=True)
    m = collect.parse_final_eval(d / "evaluation_total_test.txt")
    assert m["F1"] == pytest.approx(0.82)
    assert m["mIoU"] == pytest.approx(0.81)
    assert m["oAcc"] == pytest.approx(0.9)
    assert m["mPQ"] == pytest.approx(0.85)       # 'meanPQ', not 'meanPQ (things)'
    assert m["mIoU_binary"] == pytest.approx(0.9)
    assert m["mPrecision"] == pytest.approx(0.81)
    assert "MUCov" not in m                      # list-valued lines are skipped


def test_parse_scalars_and_curves(root):
    paths = sorted((root / "work_dirs/bench-fixed-200").glob("*/vis_data/scalars.json"))
    train, val = collect.parse_scalars(paths)
    assert len(train) == 200 and len(val) == 3
    losses = collect.loss_per_epoch(train)
    assert losses[1] == pytest.approx(9.0) and losses[200] == pytest.approx(0.045)
    curve = collect.val_curve(val)
    assert curve[20]["F1"] == pytest.approx(0.35) and curve[200]["F1"] == pytest.approx(0.60)


def test_val_prefix_is_stripped(tmp_path):
    p = tmp_path / "scalars.json"
    p.write_text(json.dumps({"ForAINetV2/F1": 0.5, "ForAINetV2/mIoU": 0.7, "step": 20}) + "\n")
    _, val = collect.parse_scalars([p])
    assert val[0]["F1"] == 0.5 and val[0]["mIoU"] == 0.7 and val[0]["step"] == 20


def test_val_record_tolerates_missing_metric_keys(tmp_path):
    """A val record without 'epoch' but with 'step' is still counted, even if it lacks the
    F1 key entirely (e.g. a metric that failed to compute for that eval)."""
    p = tmp_path / "scalars.json"
    p.write_text(json.dumps({"mIoU": 0.7, "step": 20}) + "\n")
    _, val = collect.parse_scalars([p])
    assert len(val) == 1
    assert val[0]["mIoU"] == 0.7 and "F1" not in val[0]
    curve = collect.val_curve(val)
    assert curve[20]["mIoU"] == pytest.approx(0.7) and "F1" not in curve[20]


def test_wallclock_from_log(root):
    logs = sorted((root / "work_dirs/bench-old-200").glob("*/*.log"))
    log_path = logs[0]
    # mmengine left-pads the iteration counter to align digit widths, e.g. "[ 5/23]"; rewrite
    # the first log line to use that padded form so LOG_TRAIN_RE is proven tolerant of it. The
    # padding only touches the iteration fraction (unused by parse_log_wallclock), not the
    # epoch number or timestamp, so the expected results are unchanged.
    text = log_path.read_text()
    first_line, rest = text.split("\n", 1)
    assert "[23/23]" in first_line
    log_path.write_text(first_line.replace("[23/23]", "[ 5/23]") + "\n" + rest)
    w = collect.parse_log_wallclock(logs)
    assert w["first_epoch"] == 1 and w["last_epoch"] == 200
    assert w["wall_s"] == pytest.approx(199 * 60)
    assert w["sec_per_epoch"] == pytest.approx(60.0)


def test_summarize_training_detects_nan(tmp_path):
    write_training(tmp_path / "w", 30, 5.0, {20: 0.2}, nan_at=25)
    s = collect.summarize_training(tmp_path / "w")
    assert s["nonfinite_epochs"] == [25]
    assert s["epochs_done"] == 30
    assert s["best_val"] == (20, pytest.approx(0.2))


def test_summarize_training_detects_infinite_loss(tmp_path):
    """A loss diverging to +/-inf round-trips through JSON as Infinity/-Infinity and must be
    flagged just like NaN — math.isnan() alone would miss it and report PASS."""
    write_training(tmp_path / "w", 10, 5.0, {}, inf_at=6)
    s = collect.summarize_training(tmp_path / "w")
    assert s["nonfinite_epochs"] == [6]
    assert s["epochs_done"] == 10


def test_summarize_training_no_val_points_does_not_crash(tmp_path):
    write_training(tmp_path / "w", 5, 5.0, {})
    s = collect.summarize_training(tmp_path / "w")
    assert s["best_val"] is None and s["last_val"] is None
    assert s["val"] == {}


def test_main_writes_report_with_pass_fail(root, capsys):
    out = root / "docs/benchmarks/2026-09-25-carrot-ff3d.md"
    rc = collect.main(["--root", str(root), "--date", "2026-09-25", "--out", str(out)])
    assert rc == 0
    text = out.read_text()
    assert "# ForestFormer3D benchmark on carrot (2026-09-25)" in text
    assert "## Environment" in text
    assert "code AND config both differ" in text
    assert "| fixed | 0.8200 |" in text and "| old | 0.8000 |" in text
    assert "| fixed | 200 | 60.0 |" in text
    assert "Released checkpoint: fixed F1 0.8200 >= old F1 0.8000: PASS" in text
    assert "Fixed 200-epoch run: 200/200 epochs, no non-finite loss, 3 val points: PASS" in text
    assert "| 20 | 0.3000 | 0.3500 |" in text


def test_main_infinite_loss_fails_criteria(tmp_path):
    """An Infinity loss in the fixed 200-epoch run must flip the success criterion to FAIL,
    not be silently treated as finite (the original bug: math.isnan(inf) is False)."""
    wd = tmp_path / "work_dirs"
    write_final_eval(wd / "bench-release-fixed", f1=0.82)
    write_final_eval(wd / "bench-release-old", f1=0.80)
    write_training(wd / "bench-old-200", 200, 10.0, {20: 0.30, 40: 0.40, 200: 0.55})
    write_training(wd / "bench-fixed-200", 200, 9.0, {20: 0.35, 40: 0.45, 200: 0.60}, inf_at=100)
    write_final_eval(wd / "bench-old-200" / "test", f1=0.50)
    write_final_eval(wd / "bench-fixed-200" / "test", f1=0.58)
    out = tmp_path / "docs/benchmarks/2026-09-25-carrot-ff3d.md"
    rc = collect.main(["--root", str(tmp_path), "--date", "2026-09-25", "--out", str(out)])
    assert rc == 0
    text = out.read_text()
    assert "non-finite loss at epochs [100]" in text
    assert "Fixed 200-epoch run: 200/200 epochs, non-finite loss at epochs [100], 3 val points: FAIL" in text


def test_main_explicit_dir_args_override_root(root):
    """--release-old/--release-fixed/--train-old/--train-fixed override the --root defaults."""
    out = root / "docs/benchmarks/explicit.md"
    rc = collect.main([
        "--root", str(root / "nonexistent"), "--date", "2026-09-25", "--out", str(out),
        "--release-old", str(root / "work_dirs/bench-release-old/evaluation_total_test.txt"),
        "--release-fixed", str(root / "work_dirs/bench-release-fixed/evaluation_total_test.txt"),
        "--train-old", str(root / "work_dirs/bench-old-200"),
        "--train-fixed", str(root / "work_dirs/bench-fixed-200"),
    ])
    assert rc == 0
    text = out.read_text()
    assert "| fixed | 0.8200 |" in text and "| old | 0.8000 |" in text


def test_main_missing_dir_fails_unless_allowed(tmp_path):
    out = tmp_path / "r.md"
    rc = collect.main(["--root", str(tmp_path), "--date", "2026-09-25", "--out", str(out)])
    assert rc == 1
    rc = collect.main(["--root", str(tmp_path), "--date", "2026-09-25", "--out", str(out), "--allow-missing"])
    assert rc == 0
    text = out.read_text()
    assert "n/a" in text
    assert "PENDING" in text
    assert "Released checkpoint: PENDING" in text
    assert "Fixed 200-epoch run: PENDING" in text
