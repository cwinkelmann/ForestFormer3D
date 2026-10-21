# Where the geo half lives, and whether we are already inside the image.
#
# Sourced by common.sh and by every benchmark script that needs the pure-Python half, so
# these two decisions exist once instead of once per script. Before this file, nine scripts
# each re-derived `ff3d-geo-venv` and honoured only $GEO_VENV, while common.sh exported an
# $FF3D_GEO_VENV that nothing read -- so the documented variable had no effect, and the
# in-image guard existed in two scripts out of eight.
#
# Deliberately sets no shell options. The queue scripts run with `set -uo pipefail` and no
# `-e` on purpose, so that one bad tile cannot abort the loop; a sourced file must not take
# that decision away from them.

FF3D_ROOT="${FF3D_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
# FF3D_GEO_VENV is the name; GEO_VENV is the older spelling and still wins when it is the
# one that is set, so nothing that already exports it breaks. Both end up with the same
# value, because scripts refer to either.
FF3D_GEO_VENV="${FF3D_GEO_VENV:-${GEO_VENV:-$(dirname "$FF3D_ROOT")/ff3d-geo-venv}}"
GEO_VENV="$FF3D_GEO_VENV"
export FF3D_ROOT FF3D_GEO_VENV GEO_VENV

# True when this process is ALREADY inside forestformer3d:cu118, which carries both halves.
# FF3D_IN_CONTAINER says so explicitly; otherwise the runtime's own marker file answers it,
# and the two runtimes write different ones -- docker touches /.dockerenv, podman writes
# /run/.containerenv. Checking only docker's would leave a podman container looking like a
# host, so it would try to `docker run` its GPU steps from inside a container with no docker.
#
# This MUST agree with ff3d_geo.cli._in_image, including for mixed case and stray spaces:
# tests/test_benchmark_geo_env.py runs both over the same table of values. A disagreement
# splits the run -- the script skipping the venv while ff3d_geo shells out to docker, or the
# reverse.
ff3d_in_image() {
  if [ -n "${FF3D_IN_CONTAINER+set}" ]; then
    case "$(printf '%s' "$FF3D_IN_CONTAINER" | tr -d '[:space:]' | tr '[:upper:]' '[:lower:]')" in
      ""|0|false|no) return 1 ;;
      *)             return 0 ;;
    esac
  fi
  [ -f /.dockerenv ] || [ -f /run/.containerenv ]
}

# Make the geo half importable: activate the host venv, or do nothing inside the image.
# Either way FF3D_IN_CONTAINER is pinned afterwards, so ff3d_geo cannot decide differently
# from the script that called it.
ff3d_activate_geo() {
  if ff3d_in_image; then
    export FF3D_IN_CONTAINER=1
    echo "=== inside the image: no host venv, the GPU steps run in place ==="
    return 0
  fi
  export FF3D_IN_CONTAINER=0
  if [ ! -r "$FF3D_GEO_VENV/bin/activate" ]; then
    echo "!!! no geo venv at $FF3D_GEO_VENV -- set FF3D_GEO_VENV, create it, or run the" >&2
    echo "    whole thing inside the image (docs/inference-pipeline.md)" >&2
    return 1
  fi
  # shellcheck disable=SC1091
  source "$FF3D_GEO_VENV/bin/activate"
}
