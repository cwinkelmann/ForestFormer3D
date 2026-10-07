"""benchmark/tile_timings.py: the queue-log parser for per-tile, per-method timings."""

import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmark"))

from tile_timings import collect, rows, summary  # noqa: E402

T = "3dm_33_378_5828_1_be"


def _logs(root: Path):
    (root / "work_dirs/logs/sat").mkdir(parents=True)
    (root / "work_dirs/logs/ams3d").mkdir(parents=True)
    (root / "work_dirs/logs/berlin-gpu4-block1.log").write_text(
        f"=== 2026-10-07T15:55:20 {T}: split ===\n"
        f"=== 2026-10-07T15:55:40 {T}: run (30 sub-tiles) ===\n"
        f"=== 2026-10-07T16:25:40 {T}: tile done (run 1800s) ===\n"
        f"=== 2026-10-07T16:25:41 3dm_33_380_5825_1_be: split ===\n"
        f"=== 2026-10-07T16:26:00 3dm_33_380_5825_1_be: run (50 sub-tiles) ===\n")
    (root / "work_dirs/logs/sat/sat-gpu4-q0-57.log").write_text(
        f"=== 2026-10-07T20:00:00+02:00 {T}: sat run (30 sub-tiles, GPU 4) ===\n"
        f"[2026-10-07 20:00:01][x] - =================================================\n"
        f"=== 2026-10-07T21:00:00+02:00 {T}: convert (30 files, run 3600s) ===\n")
    (root / "work_dirs/logs/ams3d/ams3d-q0-57.log").write_text(
        f"=== 2026-10-07T20:00:00 {T}: ams3d --subtiles (30 sub-tiles, 48 workers, config C) ===\n"
        f"=== 2026-10-07T20:10:00 {T}: tile done (run 600s, 30 results) ===\n")


def test_collect_reads_all_three_queue_formats_and_the_gpu(tmp_path):
    _logs(tmp_path)
    d = collect(tmp_path)
    assert d[T]["ff3d"] == {"start": "2026-10-07T15:55:40", "subtiles": 30, "gpu": 4, "log": "berlin-gpu4-block1.log",
                            "end": "2026-10-07T16:25:40", "run_s": 1800}
    assert d[T]["sat"]["run_s"] == 3600 and d[T]["sat"]["subtiles"] == 30 and d[T]["sat"]["gpu"] == 4
    assert d[T]["ams3d"]["run_s"] == 600 and d[T]["ams3d"]["gpu"] is None
    assert d["3dm_33_380_5825_1_be"]["ff3d"]["end"] is None            # still running


def test_rows_and_summary_report_rates_and_running_tiles(tmp_path):
    _logs(tmp_path)
    table = rows(collect(tmp_path), now=datetime(2026, 10, 7, 16, 36, 0))
    by = {(r["tile"], r["method"]): r for r in table}
    assert by[(T, "ff3d")]["s_per_subtile"] == 60.0 and by[(T, "ff3d")]["status"] == "done"
    assert by[(T, "sat")]["s_per_subtile"] == 120.0 and by[(T, "ams3d")]["s_per_subtile"] == 20.0
    run = by[("3dm_33_380_5825_1_be", "ff3d")]
    assert run["status"] == "running" and run["elapsed_s"] == 600 and run["s_per_subtile"] == 12.0
    s = summary(table)
    assert s["ff3d"] == {"tiles": 1, "hours_total": 0.5, "s_per_subtile_median": 60.0, "s_per_subtile_min": 60.0,
                         "s_per_subtile_max": 60.0, "running": 1}
    assert rows(collect(tmp_path), running_only=True)[0]["tile"] == "3dm_33_380_5825_1_be"


def test_latest_start_wins_and_stale_starts_are_abandoned(tmp_path):
    _logs(tmp_path)
    (tmp_path / "work_dirs/logs/berlin-r13-gpu0-old.log").write_text(
        f"=== 2026-09-23T08:10:00 {T}: run (30 sub-tiles) ===\n"           # killed in September, no end
        f"=== 2026-09-23T08:10:00 3dm_33_381_5825_1_be: run (10 sub-tiles) ===\n"
        f"=== 2026-09-23T09:10:00 3dm_33_381_5825_1_be: tile done (run 3600s) ===\n")
    d = collect(tmp_path)
    assert d[T]["ff3d"]["start"] == "2026-10-07T15:55:40" and d[T]["ff3d"]["run_s"] == 1800   # October run kept
    table = rows(d, now=datetime(2026, 10, 7, 16, 36, 0))
    by = {(r["tile"], r["method"]): r for r in table}
    assert by[("3dm_33_381_5825_1_be", "ff3d")]["status"] == "done"
    old = rows(collect(tmp_path), now=datetime(2026, 10, 9, 0, 0, 0))
    assert {r["status"] for r in old if r["tile"] == "3dm_33_380_5825_1_be"} == {"abandoned"}
    assert all(r["tile"] != "3dm_33_380_5825_1_be" for r in rows(collect(tmp_path), now=datetime(2026, 10, 9), running_only=True))
