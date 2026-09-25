# Serves the Berlin Potree site (benchmark/build_potree_site.py output).
#
# The site is tens of gigabytes of octrees, so NOTHING of it is baked in: the image
# carries only the Range-capable server and the viewer page, and the site is mounted at
# /site. Potree fetches every octree node as a byte range out of one large octree.bin,
# and a server that ignores Range answers 200 with the whole file -- the viewer then
# decodes the wrong bytes and renders scattered blobs with nothing in the console to say
# why. benchmark/serve_potree.py answers 206 and is threaded; that is the whole reason
# this is not an nginx one-liner.
#
#   docker build -f docker/potree/Dockerfile -t ff3d-potree:latest .
#   docker run --rm -p 8081:8080 \
#       -v /Volumes/2TB/winmol/ALS_Data/berlin_potree_v2:/site:ro ff3d-potree:latest
#
# The mount is read-only on purpose: the server never writes, and a typo in the path
# then fails loudly instead of quietly creating an empty directory to serve.
FROM python:3.12-slim

# No pip install: serve_potree.py is standard library only.
WORKDIR /app
COPY benchmark/serve_potree.py /app/serve_potree.py
# The viewer page ships WITH the image so the frontend is versioned alongside it.
# refresh-index.sh copies it over the mounted site's index.html when that is writable
# and FF3D_REFRESH_INDEX=1; by default the site's own page is left alone.
COPY benchmark/potree_index.html /app/potree_index.html
COPY docker/potree/entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

ENV FF3D_SITE=/site \
    FF3D_PORT=8080 \
    FF3D_BIND=0.0.0.0 \
    FF3D_REFRESH_INDEX=0
EXPOSE 8080

# Fails while the site is missing or empty, which is the usual mistake (wrong -v path).
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD python3 -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8080/data/tiles.json',timeout=4).status==200 else 1)"

ENTRYPOINT ["/app/entrypoint.sh"]
