# Potree viewer server: nginx + one config. The site (index.html, build/, libs/,
# pointclouds*/, data/) is NOT in the image; compose.yaml bind-mounts it read-only
# at /usr/share/nginx/html. See docs/benchmarks/2026-09-23-potree-viewer.md.
FROM nginx:1.27-alpine
COPY nginx.conf /etc/nginx/nginx.conf
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s \
    CMD wget -q -O /dev/null http://127.0.0.1/data/tiles.json || exit 1
