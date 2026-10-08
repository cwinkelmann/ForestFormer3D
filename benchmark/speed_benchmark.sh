#!/usr/bin/env bash
# Clean per-method speed benchmark on carrot: one tile at a time, one process on one GPU,
# nothing else of ours on the host -- the number the report quotes as "cost per km tile".
#
#   nohup bash benchmark/speed_benchmark.sh 4 3dm_33_381_5828_1_be 3dm_33_379_5829_1_be \
#     > work_dirs/logs/bench/speed-$(date +%Y%m%d-%H%M%S).log 2>&1 < /dev/null &
#
# The production queues run several tiles per host and two SegmentAnyTree containers per
# GPU, so their per-sub-tile seconds (benchmark/tile_timings.py, concurrency columns) are
# throughput under sharing. This script first waits until none of our queues or chains is
# running, then for every tile runs ForestFormer3D (`ff3d_geo run` over the tile's existing
# haloed split, GPU $1), SegmentAnyTree (one container, same GPU) and AMS3D (48 workers,
# CPU) strictly one after the other, sampling `nvidia-smi` utilisation every 10 s meanwhile.
# Outputs go to work_dirs/bench-<T>-{ff3d,sat,ams3d}/ (production results untouched) and the
# record to work_dirs/logs/bench/speed-<TS>.json: method, tile, sub-tiles, seconds, seconds
# per sub-tile, mean/max GPU utilisation, host load before and after.
#
# Environment: FF3D_ROOT, GEO_VENV, BENCH_WAIT (1 = wait for a quiet host, default),
# BENCH_WORKERS (48), BENCH_METHODS ("ff3d sat ams3d"; a subset re-times one method, e.g.
# BENCH_METHODS=ff3d after a ForestFormer3D optimisation, against the same tile's old row).
set -uo pipefail
GPU="${1:?usage: speed_benchmark.sh <gpu> <tile stem>...}"; shift
[ "$#" -gt 0 ] || { echo "no tiles given" >&2; exit 2; }
FF3D_ROOT="${FF3D_ROOT:-/raid/cwinkelmann/ForestFormer3D}"
GEO_VENV="${GEO_VENV:-/raid/cwinkelmann/ff3d-geo-venv}"
BENCH_WORKERS="${BENCH_WORKERS:-48}"
BENCH_METHODS="${BENCH_METHODS:-ff3d sat ams3d}"
cd "$FF3D_ROOT" || exit 1
source "$GEO_VENV/bin/activate"
CK=work_dirs/clean_forestformer/epoch_3000_fix.pth
TS=$(date +%Y%m%d-%H%M%S)
LOG=work_dirs/logs/bench; mkdir -p "$LOG"
REC="$LOG/speed-$TS.json"; echo "[]" > "$REC"
log() { echo "=== $(date +%FT%T) $*"; }
want() { case " $BENCH_METHODS " in *" $1 "*) return 0 ;; esac; return 1; }
for m in $BENCH_METHODS; do
  case $m in ff3d|sat|ams3d) ;; *) echo "unknown method in BENCH_METHODS: $m" >&2; exit 2 ;; esac
done

quiet() {   # none of our production queues/chains alive
  for p in berlin_run_gpu sat_run_gpu ams3d_run_cpu berlin_extend_mosaic potree_rebuild_mosaic add_ptf berlin_stitch ams3d_stitch; do
    P="${p}[.]sh"; [ "$(pgrep -f "$P" | wc -l)" = "0" ] || return 1
  done
  return 0
}
if [ "${BENCH_WAIT:-1}" = "1" ]; then
  until quiet; do log "host busy with our own queues, waiting"; sleep 600; done
fi
log "benchmark on GPU $GPU, methods: $BENCH_METHODS, tiles: $*; host load $(cut -d' ' -f1-3 /proc/loadavg)"

sample_gpu() {   # background sampler -> file with one utilisation value per 10 s
  while true; do nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits -i "$GPU" >> "$1"; sleep 10; done
}
record() {   # method tile subtiles seconds samplefile
  python3 - "$REC" "$1" "$2" "$3" "$4" "$5" "$GPU" <<'EOF'
import json, sys
rec, method, tile, n, secs, samples, gpu = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5]), sys.argv[6], int(sys.argv[7])
util, mem = [], []
try:
    for line in open(samples):
        u, m = line.strip().split(","); util.append(float(u)); mem.append(float(m))
except FileNotFoundError:
    pass
rows = json.load(open(rec))
rows.append({"method": method, "tile": tile, "gpu": gpu if method != "ams3d" else None, "subtiles": n, "seconds": secs,
             "s_per_subtile": round(secs / n, 1) if n else None,
             "gpu_util_mean": round(sum(util) / len(util), 1) if util else None, "gpu_util_max": max(util) if util else None,
             "gpu_mem_max_mib": max(mem) if mem else None, "load_after": open("/proc/loadavg").read().split()[0]})
json.dump(rows, open(rec, "w"), indent=1)
print(f"  {method} {tile}: {secs}s over {n} sub-tiles = {secs / max(n, 1):.1f} s/sub-tile, GPU util mean {rows[-1]['gpu_util_mean']}")
EOF
}

for T in "$@"; do
  N=$(ls inputs/berlin/sub/$T/*.las 2>/dev/null | wc -l)
  [ "$N" -gt 0 ] || { echo "!!! $T: no haloed split under inputs/berlin/sub/$T"; continue; }

  # --- ForestFormer3D: one process, this GPU
  if want ff3d; then
  log "$T ff3d ($N sub-tiles)"; S=$LOG/gpu-$TS-ff3d-$T.csv; sample_gpu "$S" & SP=$!
  t0=$(date +%s)
  python -m ff3d_geo run --las inputs/berlin/sub/$T/*.las --checkpoint $CK --out work_dirs/bench-$T-ff3d --gpu "$GPU" \
    > "$LOG/ff3d-$T-$TS.log" 2>&1 || echo "!!! $T ff3d reported failures"
  kill $SP 2>/dev/null; record ff3d "$T" "$N" $(( $(date +%s) - t0 )) "$S"
  fi

  # --- SegmentAnyTree: one container, this GPU (SAT_STITCH=1 keeps the per-sub-tile contract LAS)
  if want sat; then
  log "$T sat"; S=$LOG/gpu-$TS-sat-$T.csv; sample_gpu "$S" & SP=$!
  t0=$(date +%s)
  SAT_STITCH=1 SAT_OUT_PREFIX=work_dirs/bench-sat- bash benchmark/sat_run_gpu.sh "$GPU" "$T" > "$LOG/sat-$T-$TS.log" 2>&1 || echo "!!! $T sat failed"
  kill $SP 2>/dev/null
  R=$(grep -h "tile done (run" "$LOG/sat-$T-$TS.log" | sed "s/.*run //;s/s).*//" | tail -1)
  record sat "$T" "$N" "${R:-$(( $(date +%s) - t0 ))}" "$S"
  mv "work_dirs/bench-sat-$T" "work_dirs/bench-$T-sat" 2>/dev/null
  fi

  # --- AMS3D: CPU only, 48 workers, GPU idle
  if want ams3d; then
  log "$T ams3d"; t0=$(date +%s)
  AMS3D_WORKERS=$BENCH_WORKERS AMS3D_OUT_PREFIX=work_dirs/bench-ams3d- bash benchmark/ams3d_run_cpu.sh "$T" > "$LOG/ams3d-$T-$TS.log" 2>&1 || echo "!!! $T ams3d failed"
  R=$(grep -h "tile done (run" "$LOG/ams3d-$T-$TS.log" | sed "s/.*run //;s/s,.*//" | tail -1)
  record ams3d "$T" "$N" "${R:-$(( $(date +%s) - t0 ))}" /dev/null
  mv "work_dirs/bench-ams3d-$T" "work_dirs/bench-$T-ams3d" 2>/dev/null
  fi
done
log "benchmark done -> $REC"
python3 - "$REC" <<'EOF'
import json, sys
rows = json.load(open(sys.argv[1]))
print("| method | tile | sub-tiles | seconds | s / sub-tile | GPU util mean / max | GPU mem max (MiB) |")
print("|---|---|---:|---:|---:|---:|---:|")
for r in rows:
    print(f"| {r['method']} | {r['tile'][7:15]} | {r['subtiles']} | {r['seconds']} | {r['s_per_subtile']} | "
          f"{r['gpu_util_mean'] if r['gpu_util_mean'] is not None else '–'} / {r['gpu_util_max'] if r['gpu_util_max'] is not None else '–'} | "
          f"{r['gpu_mem_max_mib'] if r['gpu_mem_max_mib'] is not None else '–'} |")
EOF
