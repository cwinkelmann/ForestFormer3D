#!/usr/bin/env python3
"""Per-tile, per-method processing times, read from the queue logs on carrot.

The three production queues write the same kind of marker lines:

    === <ts> <tile>: run (83 sub-tiles) ===                       berlin_run_gpu.sh  (FF3D)
    === <ts> <tile>: tile done (run 4891s) ===
    === <ts> <tile>: sat run (83 sub-tiles, GPU 3) ===           sat_run_gpu.sh     (SAT)
    === <ts> <tile>: convert (83 files, run 6861s) ===
    === <ts> <tile>: ams3d --subtiles (83 sub-tiles, 48 workers, config C) ===   ams3d_run_cpu.sh
    === <ts> <tile>: tile done (run 822s, 83 results) ===

This script collects them into one table: per tile and method the start time, the run
seconds, the number of sub-tiles, seconds per sub-tile, the GPU (from the log file name),
and for a tile that has started but not finished the elapsed time so far. Stdlib only, so
it runs on carrot's system python:

    python3 benchmark/tile_timings.py                     # table of every tile seen
    python3 benchmark/tile_timings.py --tiles 3dm_33_378_5828_1_be ... --json work_dirs/logs/tile_timings.json
    python3 benchmark/tile_timings.py --running           # only what is in flight

A tile that ran more than once (a retry) keeps its LAST completed run. The two concurrency
columns say how many of our runs shared the GPU / the host while a tile ran (time-weighted):
per-sub-tile seconds are production throughput under that sharing, not a clean benchmark,
and only rows with similar concurrency compare.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

ABANDONED_AFTER_S = 24 * 3600
LINE = re.compile(r"^=== (?P<ts>\S+) (?P<tile>3dm_33_\d+_\d+_1_be): (?P<what>.*?) ===\s*$")
GPU_IN_NAME = re.compile(r"gpu(\d+)")
METHODS = {
    "ff3d": ("work_dirs/logs", "berlin-*gpu*.log", r"^run \((\d+) sub-tiles\)", r"^tile done \(run (\d+)s\)"),
    "sat": ("work_dirs/logs/sat", "sat-gpu*.log", r"^sat run \((\d+) sub-tiles", r"^convert \((\d+) files, run (\d+)s\)"),
    "ams3d": ("work_dirs/logs/ams3d", "ams3d-q*.log", r"^ams3d --subtiles \((\d+) sub-tiles", r"^tile done \(run (\d+)s"),
}


def parse_ts(s: str) -> datetime:
    """``2026-10-05T15:11:57`` (naive, carrot local) or ``...+02:00``."""
    t = datetime.fromisoformat(s)
    return t.replace(tzinfo=None) if t.tzinfo else t


def collect(root: Path) -> dict:
    """{tile: {method: {start, end, run_s, subtiles, gpu, log}}} from every queue log."""
    out: dict = {}
    for method, (sub, pattern, start_re, done_re) in METHODS.items():
        for log in sorted((root / sub).glob(pattern)):
            gpu = GPU_IN_NAME.search(log.name)
            gpu = int(gpu.group(1)) if gpu else None
            for line in log.read_text(errors="replace").splitlines():
                m = LINE.match(line)
                if not m:
                    continue
                tile, what, ts = m.group("tile"), m.group("what"), parse_ts(m.group("ts"))
                rec = out.setdefault(tile, {}).setdefault(method, {})
                s = re.match(start_re, what)
                d = re.match(done_re, what)
                if s:
                    # the LATEST start wins (a retry, or the halo re-run after the seamed one),
                    # whatever order the log files are read in
                    if "start" not in rec or ts.isoformat() >= rec["start"]:
                        rec.update({"start": ts.isoformat(), "subtiles": int(s.group(1)), "gpu": gpu,
                                    "log": log.name, "end": None, "run_s": None})
                elif d and "start" in rec and ts.isoformat() >= rec["start"] and rec.get("end") is None:
                    rec.update({"end": ts.isoformat(), "run_s": int(d.groups()[-1])})
                    if "subtiles" not in rec and len(d.groups()) > 1:
                        rec["subtiles"] = int(d.group(1))
    return out


def concurrency(data: dict, now: datetime) -> dict:
    """For every (tile, method) run: how many of OUR runs overlapped it in time -- on the same
    GPU and on the host as a whole -- averaged over its duration. A tile that ran alone scores
    1.0 on both; three queues sharing GPU 4 score about 3. The per-sub-tile seconds are only
    comparable between runs with similar values here."""
    runs = []
    for tile, methods in data.items():
        for method, rec in methods.items():
            if "start" not in rec:
                continue
            a = datetime.fromisoformat(rec["start"])
            b = datetime.fromisoformat(rec["end"]) if rec.get("end") else now
            if (b - a).total_seconds() > ABANDONED_AFTER_S and not rec.get("end"):
                continue
            runs.append((tile, method, rec.get("gpu"), a, b))
    out = {}
    for tile, method, gpu, a, b in runs:
        dur = max((b - a).total_seconds(), 1.0)
        same_gpu = host = 0.0
        for t2, m2, g2, a2, b2 in runs:
            ov = (min(b, b2) - max(a, a2)).total_seconds()
            if ov <= 0:
                continue
            host += ov / dur
            if gpu is not None and g2 == gpu:
                same_gpu += ov / dur
        out[(tile, method)] = {"gpu_concurrency": round(same_gpu, 1) if gpu is not None else None, "host_concurrency": round(host, 1)}
    return out


def rows(data: dict, tiles=None, now: datetime | None = None, running_only=False) -> list[dict]:
    now = now or datetime.now()
    conc = concurrency(data, now)
    out = []
    for tile in sorted(data):
        if tiles and tile not in tiles:
            continue
        for method in METHODS:
            rec = data[tile].get(method)
            if not rec or "start" not in rec:
                continue
            running = rec.get("end") is None
            start = datetime.fromisoformat(rec["start"])
            # a start with no end after a day is a killed or abandoned run, not work in flight
            abandoned = running and (now - start).total_seconds() > ABANDONED_AFTER_S
            if running_only and (not running or abandoned):
                continue
            secs = rec["run_s"] if not running else int((now - start).total_seconds())
            n = rec.get("subtiles")
            out.append({"tile": tile, "method": method, "gpu": rec.get("gpu"), "start": rec["start"],
                        "subtiles": n, "run_s": rec["run_s"], "elapsed_s": secs,
                        "s_per_subtile": round(secs / n, 1) if n else None,
                        "status": "abandoned" if abandoned else ("running" if running else "done"),
                        **conc.get((tile, method), {"gpu_concurrency": None, "host_concurrency": None})})
    return out


def summary(table: list[dict]) -> dict:
    out = {}
    for method in METHODS:
        done = [r for r in table if r["method"] == method and r["status"] == "done" and r["subtiles"]]
        if not done:
            continue
        per = sorted(r["s_per_subtile"] for r in done)
        out[method] = {"tiles": len(done), "hours_total": round(sum(r["run_s"] for r in done) / 3600, 2),
                       "s_per_subtile_median": per[len(per) // 2], "s_per_subtile_min": per[0], "s_per_subtile_max": per[-1],
                       "running": sum(1 for r in table if r["method"] == method and r["status"] == "running")}
    return out


def fmt_hm(seconds) -> str:
    if seconds is None:
        return "–"
    h, m = divmod(int(seconds) // 60, 60)
    return f"{h}:{m:02d}"


def markdown(table: list[dict], summ: dict) -> str:
    lines = ["| tile | method | GPU | start | sub-tiles | run | s / sub-tile | queues on GPU / host | status |",
             "|---|---|---:|---|---:|---:|---:|---:|---|"]
    for r in table:
        cq = r.get("gpu_concurrency"); ch = r.get("host_concurrency")
        lines.append(f"| {r['tile'][7:15]} | {r['method']} | {r['gpu'] if r['gpu'] is not None else 'cpu'} | {r['start'][5:16].replace('T', ' ')} | "
                     f"{r['subtiles'] or '–'} | {fmt_hm(r['elapsed_s'])} | {r['s_per_subtile'] or '–'} | "
                     f"{cq if cq is not None else '–'} / {ch if ch is not None else '–'} | {r['status']} |")
    lines += ["", "| method | tiles done | hours total | s / sub-tile median (min–max) | running |", "|---|---:|---:|---:|---:|"]
    for m, s in summ.items():
        lines.append(f"| {m} | {s['tiles']} | {s['hours_total']} | {s['s_per_subtile_median']} ({s['s_per_subtile_min']}–{s['s_per_subtile_max']}) | {s['running']} |")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=Path("."), help="repo root holding work_dirs/logs")
    ap.add_argument("--tiles", nargs="*", default=None, help="restrict to these tile stems")
    ap.add_argument("--running", action="store_true", help="only tiles in flight")
    ap.add_argument("--json", type=Path, default=None, help="also write the table and summary as JSON")
    a = ap.parse_args(argv)
    data = collect(a.root)
    table = rows(data, set(a.tiles) if a.tiles else None, running_only=a.running)
    summ = summary(table)
    print(markdown(table, summ))
    if a.json:
        a.json.write_text(json.dumps({"generated": datetime.now(timezone.utc).isoformat(), "rows": table, "summary": summ}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
