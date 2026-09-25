#!/bin/sh
# Serve the mounted Potree site, after checking it is actually there.
set -eu

SITE="${FF3D_SITE:-/site}"

if [ ! -d "$SITE" ]; then
    echo "!!! $SITE is not a directory -- mount the site, e.g. -v /path/to/berlin_potree_v2:/site:ro" >&2
    exit 2
fi
if [ ! -f "$SITE/data/tiles.json" ]; then
    echo "!!! $SITE has no data/tiles.json -- that is a Potree site built by" >&2
    echo "    benchmark/build_potree_site.py, not a bare octree directory." >&2
    exit 2
fi
if [ ! -d "$SITE/pointclouds" ]; then
    echo "!!! $SITE has no pointclouds/ -- the octrees are missing" >&2
    exit 2
fi

if [ "${FF3D_REFRESH_INDEX:-0}" = "1" ]; then
    if [ -w "$SITE" ]; then
        cp /app/potree_index.html "$SITE/index.html"
        echo "== refreshed $SITE/index.html from the image"
    else
        echo "!!! FF3D_REFRESH_INDEX=1 but $SITE is read-only; keeping the site's own index.html" >&2
    fi
fi

TILES=$(python3 - "$SITE/data/tiles.json" <<'PY'
import json, sys
m = json.load(open(sys.argv[1]))
t = m.get("tiles", [])
methods = {"ff3d": len(t)}
for r in t:
    for k in r.get("variants", {}):
        methods[k] = methods.get(k, 0) + 1
print(", ".join(f"{k}: {v}" for k, v in methods.items()))
PY
)
echo "== serving $SITE on ${FF3D_BIND:-0.0.0.0}:${FF3D_PORT:-8080}  ($TILES)"

# Overridable so the script can be exercised outside the image (see the Dockerfile).
exec python3 "${FF3D_SERVER:-/app/serve_potree.py}" \
    --root "$SITE" \
    --port "${FF3D_PORT:-8080}" \
    --bind "${FF3D_BIND:-0.0.0.0}"
