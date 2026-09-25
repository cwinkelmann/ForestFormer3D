#!/usr/bin/env bash
# Wait until the GPU host is genuinely idle, then exec the given command.
#
#   bash benchmark/wait_idle.sh -- bash benchmark/run_train_200.sh fixed
#
# Checks ONE GPU index, $FF3D_GPU (default 0, from common.sh -- the server has 8 GPUs shared
# with other users, so we only ever watch/use our own):
#   1. `nvidia-smi --query-compute-apps=pid ... -i $FF3D_GPU`: zero rows = no compute
#      processes on that GPU.
#   2. `docker ps`: no RUNNING container whose image starts with $FF3D_IMAGE_PREFIX
#      (default "forestformer3d"). We best-effort narrow this to containers actually using
#      $FF3D_GPU via `docker inspect`'s HostConfig.DeviceRequests; when that can't be
#      determined (e.g. --gpus all, or inspect itself fails), we do NOT rule the container
#      out -- an undeterminable container counts as busy, never as idle.
#
# "Idle" = both checks come back clear for $FF3D_IDLE_SAMPLES consecutive samples (default 3),
# $FF3D_IDLE_INTERVAL seconds apart (default 30). One empty sample is not idle: a neighbour's
# job between stages can look momentarily empty.
#
# A FAILED query (nvidia-smi or docker ps itself erroring, not "found zero") ABORTS the
# script rather than being treated as idle -- exit 2, message on stderr. A silent failure
# must never masquerade as an idle GPU.
#
# --once: take a single sample and exit immediately: 0 idle, 1 busy, 2 error. For tests and
# for cron-style / scripted single checks (no waiting, no launched command).
#
# FF3D_NVIDIA_SMI: path to the nvidia-smi binary (default: "nvidia-smi" on $PATH). Overridden
# in tests to point at a fake script that simulates idle/busy/failing output.
set -u
source "$(dirname "$0")/common.sh"
# common.sh's own `set -euo pipefail` is in effect after the source above. This script wants
# -u but NOT -e: it inspects the exit codes of nvidia-smi/docker/sample_once itself (a failed
# query must become exit 2, not an abrupt death), so turn -e and pipefail back off here --
# the `set -u` on the line above was silently overridden before.
set +e +o pipefail

FF3D_IDLE_SAMPLES="${FF3D_IDLE_SAMPLES:-3}"
FF3D_IDLE_INTERVAL="${FF3D_IDLE_INTERVAL:-30}"
FF3D_IMAGE_PREFIX="${FF3D_IMAGE_PREFIX:-forestformer3d}"
NVIDIA_SMI="${FF3D_NVIDIA_SMI:-nvidia-smi}"
MAX_WAIT="${MAX_WAIT:-$((48*3600))}"

now() { date '+%Y-%m-%dT%H:%M:%S'; }   # portable (GNU and BSD date)

# check_container_busy <container-id>: 0 (true) if this container should count as using
# $FF3D_GPU, 1 (false) if we could positively confirm it uses ONLY a different GPU.
# Undeterminable => busy (0), by design: we never let a container we can't account for make
# the host look idle.
check_container_busy() {
  local cid="$1" devids
  # $FF3D_DOCKER is deliberately UNQUOTED here and below, exactly as in common.sh
  # (ff3d_docker/ff3d_docker_old): it may legitimately hold two words, "sudo docker".
  devids=$($FF3D_DOCKER inspect \
    -f '{{range .HostConfig.DeviceRequests}}{{range .DeviceIDs}}{{.}},{{end}}{{end}}' \
    "$cid" 2>/dev/null) || return 0
  [ -n "$devids" ] || return 0
  case ",$devids," in
    *",$FF3D_GPU,"*) return 0 ;;
    *) return 1 ;;
  esac
}

# sample_once: takes one sample. Prints one status line. Returns 0 idle, 1 busy, 2 error
# (an ERROR line has already gone to stderr in that case).
sample_once() {
  local gpu_out gpu_count docker_out cid image busy=0

  if ! gpu_out=$("$NVIDIA_SMI" --query-compute-apps=pid --format=csv,noheader -i "$FF3D_GPU" 2>&1); then
    echo "ERROR: nvidia-smi query failed (gpu $FF3D_GPU): $gpu_out" >&2
    return 2
  fi
  gpu_count=0
  if [ -n "$gpu_out" ]; then
    gpu_count=$(printf '%s\n' "$gpu_out" | grep -c .)
  fi

  if ! docker_out=$($FF3D_DOCKER ps --format '{{.ID}}|{{.Image}}' 2>&1); then
    echo "ERROR: docker ps failed: $docker_out" >&2
    return 2
  fi

  if [ -n "$docker_out" ]; then
    while IFS='|' read -r cid image; do
      [ -n "$cid" ] || continue
      case "$image" in
        "${FF3D_IMAGE_PREFIX}"*)
          if check_container_busy "$cid"; then
            busy=1
          fi
          ;;
      esac
    done <<EOF
$docker_out
EOF
  fi

  if [ "$gpu_count" -eq 0 ] && [ "$busy" -eq 0 ]; then
    echo "$(now) sample: idle (gpu $FF3D_GPU: 0 compute apps, no matching container)"
    return 0
  else
    echo "$(now) sample: busy (gpu $FF3D_GPU: $gpu_count compute apps, container busy=$busy)"
    return 1
  fi
}

ONCE=0
if [ "${1:-}" = "--once" ]; then
  ONCE=1
  shift
fi

if [ "$ONCE" -eq 1 ]; then
  sample_once
  exit $?
fi

[ "${1:-}" = "--" ] && shift
[ $# -gt 0 ] || { echo "usage: wait_idle.sh [--once] -- <command...>" >&2; exit 2; }

streak=0
waited=0
echo "$(now) watcher started: need ${FF3D_IDLE_SAMPLES}x${FF3D_IDLE_INTERVAL}s idle on gpu $FF3D_GPU"
while [ "$waited" -lt "$MAX_WAIT" ]; do
  rc=0
  sample_once || rc=$?
  case "$rc" in
    0)
      streak=$((streak+1))
      echo "$(now) idle ${streak}/${FF3D_IDLE_SAMPLES}"
      if [ "$streak" -ge "$FF3D_IDLE_SAMPLES" ]; then
        echo "$(now) IDLE -- launching: $*"
        exec "$@"
      fi
      ;;
    1)
      [ "$streak" -gt 0 ] && echo "$(now) busy again -- reset"
      streak=0
      ;;
    2)
      echo "$(now) ABORT: query failed, see error above" >&2
      exit 2
      ;;
  esac
  sleep "$FF3D_IDLE_INTERVAL"
  waited=$((waited+FF3D_IDLE_INTERVAL))
done
echo "$(now) gave up after ${MAX_WAIT}s -- never idle" >&2
exit 1
