# Potree Viewer in Docker Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Serve the existing Berlin Potree site (`index.html` + `build/` + `libs/` + `pointclouds*/` + `data/`) from a Docker container that bind-mounts the site folder read-only, on the Mac today and on carrot or another Linux host later, replacing the ad-hoc `python3 benchmark/serve_potree.py`.

**Architecture:** One tiny image, `ff3d-potree`, built from `nginx:1.27-alpine` plus a hand-written `nginx.conf`. The container holds NO site content: the whole site root is a single read-only bind mount at `/usr/share/nginx/html`, selected per host with a `POTREE_SITE` variable in a Compose `.env` file. nginx answers HTTP Range requests natively (Potree reads every octree node as a byte range out of `octree.bin`), so the Python range server becomes a no-Docker fallback only.

**Tech Stack:** Docker Engine / Docker Desktop 29, Docker Compose v5 (`compose.yaml`, `.env`), nginx 1.27 alpine, pytest for the repo's CPU tests, Python `urllib` for the container smoke test.

**Spec:** No separate spec file; the design is the "Design" section below (written 2026-09-25 from the state of `/Volumes/2TB/winmol/ALS_Data/berlin_potree_v2`, `benchmark/serve_potree.py`, `benchmark/build_potree_site.py` and `docs/benchmarks/2026-09-23-potree-viewer.md`).

## Design

### What exists

- The site is fully static and every URL in `index.html` is **relative** (`build/potree/...`, `libs/...`, `data/tiles.json`, `pointclouds/<tile>/metadata.json`, `pointclouds_sat/...`, `pointclouds_ams3d/...`). Nothing needs a hostname, port or path prefix.
- Two site roots exist on the 2TB volume: `berlin_potree/` (11 tiles, 8.5 GB octrees) and `berlin_potree_v2/` (33 tiles, three methods, 22 GB octrees + 1.3 GB `data/`). Both share the same `build/`, `libs/`, `README.md`. The v2 site is the current one.
- `benchmark/build_potree_site.py` regenerates `<site>/index.html` from the tracked template `benchmark/potree_index.html` and writes `<site>/data/`. The site directory is therefore the natural unit to mount: splitting "viewer runtime in the image, data mounted" would fight that script and gain nothing.
- Serving requirement (verified in a browser on 2026-09-23): the server MUST answer `Range: bytes=a-b` with `206 Partial Content`. `python3 -m http.server` returns `200` + whole file and the cloud renders as scattered blobs with no error. `benchmark/serve_potree.py` fixes that but is single-host, single-user, Python-speed.
- The 2TB volume is exFAT, mounted `noowners`; it carries AppleDouble `._*` sidecar files next to every real file. Docker Desktop shares `/Volumes` by default (Settings ▸ Resources ▸ File sharing); a read-only mount avoids every permission question.
- Docker on the Mac: engine 29.2.1, Compose v5.0.2 installed; the daemon was NOT running when this plan was written (start Docker Desktop first). carrot has Docker too (`benchmark/common.sh` `ff3d_docker`).

### Decisions

1. **Image = nginx + config only.** `docker/potree/Dockerfile` copies `docker/potree/nginx.conf` over `/etc/nginx/nginx.conf`. No site files are baked in; `docker compose build` takes seconds and the image never needs rebuilding when tiles are added.
2. **One mount, the site root, read-only.** `${POTREE_SITE}:/usr/share/nginx/html:ro`. To view the v1 site instead, change `POTREE_SITE` (or run a second Compose project with `-p potree-v1` and another port).
3. **Config via `.env` next to `compose.yaml`.** `POTREE_SITE` (required, no default: a wrong default silently serves the wrong mosaic), `POTREE_PORT` (default 8080), `POTREE_BIND` (default `127.0.0.1`; set `0.0.0.0` on carrot to share on the LAN/VPN). `docker/potree/.env.example` is tracked, `docker/potree/.env` is git-ignored.
4. **nginx settings that matter for Potree:** `sendfile on` + `tcp_nopush on` (byte ranges out of multi-GB files), `max_ranges 1` (Potree only ever sends one range), `gzip off` for `*.bin` (a compressed body has no byte-exact offsets), gzip on for text/JS/JSON/GeoJSON, explicit MIME types `geojson → application/geo+json` and `wasm → application/wasm` (the `laz-perf.wasm` and `sql-wasm.wasm` workers need the right type for `WebAssembly.instantiateStreaming`), `location ~ /\._ { return 404; }` so the exFAT sidecars are never served, `access_log off` (a single tile view is thousands of range requests).
5. **Alternative considered and rejected:** baking `build/`, `libs/`, `index.html` into the image and mounting only `pointclouds*/` and `data/`. Rejected because `build_potree_site.py` writes `index.html` into the site root, `build/`+`libs/` are not in the repo (they are the Potree 1.8.2 release), and a three-mount layout is more to get wrong for no benefit.
6. **Keep `benchmark/serve_potree.py`** as the no-Docker fallback; docs point to the container first.

### Out of scope

TLS, authentication, a public deployment, PotreeConverter in a container, changes to `build_potree_site.py` or `potree_index.html` (both have UNRELATED uncommitted edits in the working tree right now: never `git add -A`, add files by name).

## Global Constraints

- Base image exactly `nginx:1.27-alpine`; image tag `ff3d-potree:latest`; container name `ff3d-potree`.
- The site mount is read-only (`:ro`) and is the only volume.
- Compose file is `docker/potree/compose.yaml` (Compose v2+ file format, no `version:` key). Compose reads `.env` from the directory of the compose file, so every `docker compose` command below runs from `docker/potree/` or passes `-f docker/potree/compose.yaml --project-directory docker/potree`.
- No secrets or host paths hard-coded in tracked files; host paths live only in the git-ignored `.env`.
- Tests under `tests/` must pass on the Mac WITHOUT Docker running (`python3 -m pytest tests`): anything needing the daemon skips cleanly with a reason.
- Commits: conventional prefix (`feat:`, `docs:`, `test:`), no `Co-Authored-By`/`Claude-Session` trailers (user rule), push only to the `fork` remote on branch `fix/review-findings`.

## Review Focus

1. **Range request on `octree.bin`** must return `206` with `Content-Range: bytes a-b/size` and exactly the requested bytes; a `200` here silently breaks the viewer. Pinned by Task 2 `test_range_request_returns_206_and_exact_bytes`.
2. **`._*` AppleDouble files** on the exFAT volume must never be served (`404`), otherwise a stray `libs/._potree.js` style path could shadow a real request in a directory listing or confuse a copy. Pinned by Task 2 `test_appledouble_sidecar_is_404`.
3. **Missing or empty `POTREE_SITE`** must fail `docker compose up` with a readable message, not start nginx over an empty anonymous volume that serves a 403/404 for `/`. Pinned by Task 1 `test_compose_requires_potree_site` (checks the `:?` interpolation form) and Task 3 step 2 (manual: run without `.env`, expect the error).
4. **`.geojson` and `.wasm` MIME types**: the crown/tree overlays are fetched as JSON and the LAZ workers instantiate wasm; wrong types show up as console errors, not as a hard failure. Pinned by Task 2 `test_mime_types`.
5. **`.bin` served uncompressed** even when the browser sends `Accept-Encoding: gzip`. Pinned by Task 2 `test_bin_is_not_gzipped`.

---

### Task 1: nginx config, Dockerfile, Compose file and hygiene tests

**Files:**
- Create: `docker/potree/nginx.conf`
- Create: `docker/potree/Dockerfile`
- Create: `docker/potree/compose.yaml`
- Create: `docker/potree/.env.example`
- Modify: `.gitignore` (append one line)
- Test: `tests/test_potree_docker.py`

**Interfaces:**
- Produces: the directory `docker/potree/` with the four files above; env variables `POTREE_SITE` (required), `POTREE_PORT` (default `8080`), `POTREE_BIND` (default `127.0.0.1`); image `ff3d-potree:latest`; container path `/usr/share/nginx/html`. Task 2 builds the image from `docker/potree/`; Task 3 runs it.

- [ ] **Step 1: Write the failing hygiene tests**

```python
# tests/test_potree_docker.py
"""The Potree viewer container (docker/potree/): config hygiene without Docker.

The site is static and path-relative; the one hard requirement is HTTP Range
support (Potree reads octree nodes as byte ranges out of octree.bin), so these
tests pin the nginx settings that keep ranges byte-exact, plus the Compose
contract (read-only site mount selected by POTREE_SITE).
"""
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
POTREE_DIR = REPO_ROOT / "docker" / "potree"
NGINX_CONF = POTREE_DIR / "nginx.conf"
DOCKERFILE = POTREE_DIR / "Dockerfile"
COMPOSE = POTREE_DIR / "compose.yaml"
ENV_EXAMPLE = POTREE_DIR / ".env.example"


def _text(path: Path) -> str:
    assert path.is_file(), f"{path.relative_to(REPO_ROOT)} is missing"
    return path.read_text()


def test_files_exist():
    for path in (NGINX_CONF, DOCKERFILE, COMPOSE, ENV_EXAMPLE):
        assert path.is_file(), path.relative_to(REPO_ROOT)


def test_nginx_serves_byte_exact_ranges():
    conf = _text(NGINX_CONF)
    assert re.search(r"^\s*sendfile\s+on;", conf, re.M)
    assert re.search(r"^\s*max_ranges\s+1;", conf, re.M)
    # a gzip-compressed body has no byte-exact offsets: .bin must stay raw
    bin_block = re.search(r"location\s+~\s+\\\.bin\$\s*\{([^}]*)\}", conf)
    assert bin_block, "no `location ~ \\.bin$` block"
    assert re.search(r"gzip\s+off;", bin_block.group(1))


def test_nginx_mime_types_for_overlays_and_wasm():
    conf = _text(NGINX_CONF)
    assert "include" in conf and "mime.types" in conf
    assert re.search(r"application/geo\+json\s+geojson;", conf)
    assert re.search(r"application/wasm\s+wasm;", conf)


def test_nginx_hides_appledouble_sidecars():
    conf = _text(NGINX_CONF)
    assert re.search(r"location\s+~\s+/\\\._\s*\{\s*return\s+404;", conf)


def test_nginx_root_matches_compose_mount():
    conf = _text(NGINX_CONF)
    compose = _text(COMPOSE)
    assert re.search(r"^\s*root\s+/usr/share/nginx/html;", conf, re.M)
    assert ":/usr/share/nginx/html:ro" in compose


def test_dockerfile_is_nginx_plus_config_only():
    lines = [l.strip() for l in _text(DOCKERFILE).splitlines()
             if l.strip() and not l.strip().startswith("#")]
    assert lines[0] == "FROM nginx:1.27-alpine"
    assert any(l.startswith("COPY nginx.conf /etc/nginx/nginx.conf") for l in lines)
    assert not any(l.startswith("ADD") for l in lines)
    assert not any("pointclouds" in l or "index.html" in l for l in lines), \
        "site content must be mounted, never baked into the image"


def test_compose_requires_potree_site():
    compose = _text(COMPOSE)
    assert "version:" not in compose, "Compose v2 files have no version key"
    assert re.search(r"\$\{POTREE_SITE:\?", compose), \
        "POTREE_SITE must use the `:?` form so a missing value fails `up`"
    assert re.search(r"\$\{POTREE_PORT:-8080\}", compose)
    assert re.search(r"\$\{POTREE_BIND:-127\.0\.0\.1\}", compose)
    assert "image: ff3d-potree:latest" in compose
    assert "container_name: ff3d-potree" in compose
    assert compose.count("- \"${POTREE_SITE") + compose.count("- ${POTREE_SITE") == 1, \
        "exactly one volume: the site root"


def test_env_example_documents_every_variable():
    env = _text(ENV_EXAMPLE)
    for var in ("POTREE_SITE", "POTREE_PORT", "POTREE_BIND"):
        assert re.search(rf"^{var}=", env, re.M), var
    assert "berlin_potree_v2" in env


def test_env_file_is_gitignored():
    ignore = _text(REPO_ROOT / ".gitignore")
    assert re.search(r"^docker/potree/\.env$", ignore, re.M)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd /Users/christian/ForestFormer3D && python3 -m pytest tests/test_potree_docker.py -q`
Expected: 9 failed, first assertion `docker/potree/nginx.conf is missing`.

- [ ] **Step 3: Write `docker/potree/nginx.conf`**

```nginx
# nginx for the Potree viewer (docker/potree/). The site root is a read-only
# bind mount; nothing here is site specific.
#
# Potree 2 reads every octree node as an HTTP byte range out of one large
# octree.bin, so the server must answer `Range:` with 206 and byte-exact
# offsets. nginx does that natively; the two things that would break it are
# gzip on the binaries (no byte-exact offsets in a compressed body) and a
# proxy/CDN that strips Range, neither of which applies here.

worker_processes auto;

events {
    worker_connections 1024;
}

http {
    include       /etc/nginx/mime.types;
    default_type  application/octet-stream;
    types {
        application/geo+json  geojson;   # data/<tile>_trees|_crowns.geojson
        application/wasm      wasm;      # libs/copc/laz-perf.wasm, sql.js
    }

    sendfile          on;
    tcp_nopush        on;
    keepalive_timeout 65;

    # thousands of range requests per tile view: log errors only
    access_log off;
    error_log  /dev/stderr warn;

    gzip            on;
    gzip_min_length 1024;
    gzip_types      text/css text/javascript application/javascript
                    application/json application/geo+json;

    server {
        listen 80;
        root   /usr/share/nginx/html;
        index  index.html;

        # Potree sends exactly one range per request
        max_ranges 1;

        # AppleDouble sidecars (._foo) the exFAT volume carries next to every file
        location ~ /\._ { return 404; }

        # octree.bin / hierarchy.bin: never compress, keep offsets byte-exact
        location ~ \.bin$ {
            gzip off;
        }

        location / {
            try_files $uri $uri/ =404;
        }
    }
}
```

- [ ] **Step 4: Write `docker/potree/Dockerfile`**

```dockerfile
# Potree viewer server: nginx + one config. The site (index.html, build/, libs/,
# pointclouds*/, data/) is NOT in the image; compose.yaml bind-mounts it read-only
# at /usr/share/nginx/html. See docs/benchmarks/2026-09-23-potree-viewer.md.
FROM nginx:1.27-alpine
COPY nginx.conf /etc/nginx/nginx.conf
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s \
    CMD wget -q -O /dev/null http://127.0.0.1/data/tiles.json || exit 1
```

- [ ] **Step 5: Write `docker/potree/compose.yaml`**

```yaml
# Potree viewer over a mounted site folder.
#
#   cp .env.example .env      # set POTREE_SITE to the site root
#   docker compose up -d --build
#   open http://localhost:${POTREE_PORT:-8080}/
#
# Compose reads .env from this directory, so run the commands from docker/potree/
# (or pass -f docker/potree/compose.yaml --project-directory docker/potree).
services:
  potree:
    build: .
    image: ff3d-potree:latest
    container_name: ff3d-potree
    restart: unless-stopped
    ports:
      - "${POTREE_BIND:-127.0.0.1}:${POTREE_PORT:-8080}:80"
    volumes:
      - "${POTREE_SITE:?set POTREE_SITE in docker/potree/.env to the site root, e.g. /Volumes/2TB/winmol/ALS_Data/berlin_potree_v2}:/usr/share/nginx/html:ro"
```

- [ ] **Step 6: Write `docker/potree/.env.example`**

```bash
# Copy to docker/potree/.env (git-ignored) and edit.

# The site root: the folder holding index.html, build/, libs/, pointclouds*/, data/.
# Mac (2TB volume, 33-tile mosaic):
POTREE_SITE=/Volumes/2TB/winmol/ALS_Data/berlin_potree_v2
# carrot would be e.g. POTREE_SITE=/raid/cwinkelmann/potree/berlin_potree_v2

# Host port the viewer is published on.
POTREE_PORT=8080

# 127.0.0.1 = this machine only. Use 0.0.0.0 on carrot to reach it over the VPN.
POTREE_BIND=127.0.0.1
```

- [ ] **Step 7: Git-ignore the real `.env`**

Append to `.gitignore` under the "Local tooling" section:

```
docker/potree/.env
```

- [ ] **Step 8: Run the tests to verify they pass**

Run: `cd /Users/christian/ForestFormer3D && python3 -m pytest tests/test_potree_docker.py -q`
Expected: `9 passed`.

- [ ] **Step 9: Syntax-check the nginx config if the daemon is up (optional here, mandatory in Task 2)**

Run (start Docker Desktop first if `docker info` fails):
```bash
cd /Users/christian/ForestFormer3D/docker/potree && docker build -t ff3d-potree:latest . && docker run --rm ff3d-potree:latest nginx -t
```
Expected: last two lines `nginx: the configuration file /etc/nginx/nginx.conf syntax is ok` and `... test is successful`. If it reports a duplicate `types` or unknown directive, fix `nginx.conf` before committing.

- [ ] **Step 10: Commit (by file name, the working tree has unrelated edits)**

```bash
cd /Users/christian/ForestFormer3D
git add docker/potree/nginx.conf docker/potree/Dockerfile docker/potree/compose.yaml docker/potree/.env.example .gitignore tests/test_potree_docker.py
git commit -m "feat: Potree viewer as an nginx container over a read-only site mount"
```

---

### Task 2: Container smoke test (Range, MIME, sidecars) that skips without Docker

**Files:**
- Modify: `tests/test_potree_docker.py` (append a second section)

**Interfaces:**
- Consumes: `docker/potree/` from Task 1 (image `ff3d-potree:latest`, container root `/usr/share/nginx/html`).
- Produces: a fixture `potree_container` yielding a base URL; Task 3 uses the same manual checks against the real site.

- [ ] **Step 1: Append the failing container tests**

```python
# --- appended to tests/test_potree_docker.py -----------------------------------
import gzip
import json
import os
import shutil
import subprocess
import time
import urllib.error
import urllib.request


def _docker_available() -> bool:
    if shutil.which("docker") is None:
        return False
    try:
        return subprocess.run(["docker", "info"], capture_output=True, timeout=15).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


needs_docker = pytest.mark.skipif(not _docker_available(),
                                  reason="docker daemon not reachable (start Docker Desktop)")

OCTREE = bytes(range(256)) * 8  # 2048 bytes, byte i == i % 256


@pytest.fixture(scope="module")
def fake_site(tmp_path_factory) -> Path:
    site = tmp_path_factory.mktemp("potree_site")
    (site / "index.html").write_text("<html><body>potree</body></html>")
    (site / "data").mkdir()
    # comfortably above nginx's gzip_min_length (1024) so the gzip test is meaningful
    (site / "data" / "tiles.json").write_text(json.dumps(
        {"crs": "EPSG:25833", "tiles": [{"tile": f"3dm_33_{i}_1_be" * 4} for i in range(60)]}))
    (site / "data" / "t_crowns.geojson").write_text('{"type":"FeatureCollection","features":[]}')
    (site / "libs").mkdir()
    (site / "libs" / "laz-perf.wasm").write_bytes(b"\0asm\1\0\0\0")
    (site / "libs" / "._laz-perf.wasm").write_bytes(b"\0\5\26\7AppleDouble")
    pc = site / "pointclouds" / "tile"
    pc.mkdir(parents=True)
    (pc / "octree.bin").write_bytes(OCTREE)
    return site


@pytest.fixture(scope="module")
def potree_container(fake_site):
    if not _docker_available():
        pytest.skip("docker daemon not reachable (start Docker Desktop)")
    subprocess.run(["docker", "build", "-q", "-t", "ff3d-potree:latest", str(POTREE_DIR)],
                   check=True, capture_output=True)
    cid = subprocess.run(
        ["docker", "run", "-d", "--rm", "-p", "127.0.0.1:0:80",
         "-v", f"{fake_site}:/usr/share/nginx/html:ro", "ff3d-potree:latest"],
        check=True, capture_output=True, text=True).stdout.strip()
    try:
        port = subprocess.run(["docker", "port", cid, "80/tcp"], check=True,
                              capture_output=True, text=True).stdout.strip().rsplit(":", 1)[-1]
        base = f"http://127.0.0.1:{port}"
        for _ in range(50):  # wait for nginx to listen
            try:
                urllib.request.urlopen(base + "/index.html", timeout=1).read()
                break
            except (urllib.error.URLError, ConnectionError):
                time.sleep(0.1)
        yield base
    finally:
        subprocess.run(["docker", "rm", "-f", cid], capture_output=True)


def _get(base: str, path: str, headers: dict | None = None):
    req = urllib.request.Request(base + path, headers=headers or {})
    try:
        with urllib.request.urlopen(req, timeout=5) as r:
            return r.status, dict(r.headers), r.read()
    except urllib.error.HTTPError as e:
        return e.code, dict(e.headers), e.read()


@needs_docker
def test_range_request_returns_206_and_exact_bytes(potree_container):
    status, headers, body = _get(potree_container, "/pointclouds/tile/octree.bin",
                                 {"Range": "bytes=100-199"})
    assert status == 206
    assert headers["Content-Range"] == f"bytes 100-199/{len(OCTREE)}"
    assert body == OCTREE[100:200]
    assert headers["Accept-Ranges"] == "bytes"


@needs_docker
def test_open_ended_and_suffix_ranges(potree_container):
    status, headers, body = _get(potree_container, "/pointclouds/tile/octree.bin",
                                 {"Range": "bytes=2000-"})
    assert (status, body) == (206, OCTREE[2000:])
    status, headers, body = _get(potree_container, "/pointclouds/tile/octree.bin",
                                 {"Range": "bytes=-16"})
    assert (status, body) == (206, OCTREE[-16:])


@needs_docker
def test_bin_is_not_gzipped(potree_container):
    status, headers, body = _get(potree_container, "/pointclouds/tile/octree.bin",
                                 {"Accept-Encoding": "gzip"})
    assert status == 200
    assert "Content-Encoding" not in headers
    assert body == OCTREE


@needs_docker
def test_json_is_gzipped_when_asked(potree_container):
    status, headers, body = _get(potree_container, "/data/tiles.json", {"Accept-Encoding": "gzip"})
    assert status == 200
    assert headers.get("Content-Encoding") == "gzip"
    assert json.loads(gzip.decompress(body))["crs"] == "EPSG:25833"
    assert len(json.loads(gzip.decompress(body))["tiles"]) == 60


@needs_docker
def test_mime_types(potree_container):
    for path, mime in (("/data/tiles.json", "application/json"),
                       ("/data/t_crowns.geojson", "application/geo+json"),
                       ("/libs/laz-perf.wasm", "application/wasm"),
                       ("/index.html", "text/html")):
        status, headers, _ = _get(potree_container, path)
        assert status == 200, path
        assert headers["Content-Type"].split(";")[0] == mime, path


@needs_docker
def test_appledouble_sidecar_is_404(potree_container):
    status, _, _ = _get(potree_container, "/libs/._laz-perf.wasm")
    assert status == 404
    status, _, _ = _get(potree_container, "/libs/laz-perf.wasm")
    assert status == 200


@needs_docker
def test_root_serves_index(potree_container):
    status, headers, body = _get(potree_container, "/")
    assert status == 200 and b"potree" in body
```

- [ ] **Step 2: Run without Docker to verify the skip is clean**

Run (with Docker Desktop stopped, or `DOCKER_HOST=unix:///nonexistent`):
`cd /Users/christian/ForestFormer3D && DOCKER_HOST=unix:///nonexistent python3 -m pytest tests/test_potree_docker.py -q -rs`
Expected: `9 passed, 7 skipped`, each skip reason `docker daemon not reachable (start Docker Desktop)`.

- [ ] **Step 3: Start Docker Desktop and run the container tests**

Run: `open -a Docker` then wait until `docker info` succeeds, then
`cd /Users/christian/ForestFormer3D && python3 -m pytest tests/test_potree_docker.py -q -rs`
Expected: `16 passed`. If `test_bin_is_not_gzipped` fails with `Content-Encoding: gzip`, the `location ~ \.bin$` block is not matching: check the regex escaping in `nginx.conf`. If `test_mime_types` fails on `.geojson` with `application/octet-stream`, the extra `types {}` block was not merged: move the two lines into a copy of `mime.types` instead.

- [ ] **Step 4: Run the whole CPU suite to make sure nothing else broke**

Run: `cd /Users/christian/ForestFormer3D && python3 -m pytest tests -q`
Expected: all previously passing tests still pass; the new file adds 16 passed (or 9 passed + 7 skipped without Docker).

- [ ] **Step 5: Commit**

```bash
cd /Users/christian/ForestFormer3D
git add tests/test_potree_docker.py
git commit -m "test: Potree container answers byte ranges, MIME types and hides ._ sidecars"
```

---

### Task 3: Run against the real 33-tile site, verify in a browser, document

**Files:**
- Create: `docker/potree/.env` (git-ignored, local only)
- Modify: `docs/benchmarks/2026-09-23-potree-viewer.md` (section "Serving and two viewer fixes")
- Modify: `.claude/skills/ff3d-outputs-and-viewers/SKILL.md` (section 7 "Potree web viewer")
- Modify: `.claude/skills/segmentanytree-inference/SKILL.md:162` (the `serve_potree.py` line)
- Modify: `CLAUDE.md` ("Skills" list entry for `ff3d-outputs-and-viewers` mentions the viewer; add the container in one clause)
- Modify (off-repo, both copies): `/Volumes/2TB/winmol/ALS_Data/berlin_potree_v2/README.md` and `/Volumes/2TB/winmol/ALS_Data/berlin_potree/README.md`, section "Start"

**Interfaces:**
- Consumes: `docker/potree/compose.yaml` and `.env.example` from Task 1.

- [ ] **Step 1: Create the local `.env` and start the container**

```bash
cd /Users/christian/ForestFormer3D/docker/potree
cp .env.example .env            # POTREE_SITE already points at berlin_potree_v2, port 8080
docker compose up -d --build
docker compose ps
```
Expected: service `potree`, container `ff3d-potree`, state `running` (health `starting` then `healthy` within ~35 s), port `127.0.0.1:8080->80/tcp`. If the mount fails with "path not shared", add `/Volumes/2TB` under Docker Desktop ▸ Settings ▸ Resources ▸ File sharing and retry.

- [ ] **Step 2: Verify the required-variable guard**

```bash
cd /Users/christian/ForestFormer3D/docker/potree
mv .env .env.bak && docker compose config >/dev/null; echo "exit $?"; mv .env.bak .env
```
Expected: an error containing `set POTREE_SITE in docker/potree/.env` and a non-zero exit. No container is started.

- [ ] **Step 3: Verify the real site over HTTP from Python (curl is blocked in this session)**

```bash
cd /Users/christian/ForestFormer3D && python3 - <<'EOF'
import json, urllib.request
base = "http://127.0.0.1:8080"
m = json.load(urllib.request.urlopen(base + "/data/tiles.json"))
print("tiles:", len(m["tiles"]), "first:", m["tiles"][0]["pointcloud"])
meta = json.load(urllib.request.urlopen(base + "/" + m["tiles"][0]["pointcloud"]))
print("points:", meta["points"])
octree = m["tiles"][0]["pointcloud"].rsplit("/", 1)[0] + "/octree.bin"
req = urllib.request.Request(base + "/" + octree, headers={"Range": "bytes=0-1023"})
with urllib.request.urlopen(req) as r:
    print(r.status, r.headers["Content-Range"], len(r.read()))
EOF
```
Expected: `tiles: 33`, a `points:` count in the tens of millions, and `206 bytes 0-1023/<size> 1024`.

- [ ] **Step 4: Verify the rendering in a browser**

Open `http://localhost:8080/` in Chrome (or drive it with the Playwright MCP tools: navigate, wait for the tile list, take a screenshot). Expected: the first tile renders as a coherent point cloud coloured by tree id, the right-hand panel lists 33 tiles, and the console has no `could not read data/tiles.json` and no `Range`/`206` related errors. Scattered blobs = ranges not honoured; go back to Task 2.

- [ ] **Step 5: Update the repo docs**

In `docs/benchmarks/2026-09-23-potree-viewer.md`, section "Serving and two viewer fixes", replace the `serve_potree.py` code block with:

````markdown
The site must be served by something that answers HTTP Range requests (see below).
The supported way is the nginx container in `docker/potree/`, which bind-mounts the
site folder read-only and needs no site-specific build:

```bash
cd docker/potree && cp .env.example .env     # POTREE_SITE = site root, POTREE_PORT, POTREE_BIND
docker compose up -d --build                  # http://localhost:8080/
docker compose logs -f                        # errors only; access log is off
docker compose down
```

`benchmark/serve_potree.py --root <site> --port 8080` remains as the no-Docker fallback:

```bash
python3 benchmark/serve_potree.py --root /Volumes/2TB/winmol/ALS_Data/berlin_potree_v2 --port 8080
```
````

Keep the paragraph explaining WHY ranges matter (blobs, `http.server`) as it is.

In `.claude/skills/ff3d-outputs-and-viewers/SKILL.md` section 7, replace the `python3 -m http.server 8080` block with the same `docker compose` block plus one line: "`python3 -m http.server` does NOT work (no Range support: scattered blobs); `benchmark/serve_potree.py` is the no-Docker fallback." Add to the layout paragraph: "`pointclouds_sat/` and `pointclouds_ams3d/` hold the other two methods' octrees; the v2 site (`berlin_potree_v2/`) is the 33-tile one." Add a "Sharing from carrot" line: copy the site folder to `/raid/cwinkelmann/potree/berlin_potree_v2`, set `POTREE_SITE` to it and `POTREE_BIND=0.0.0.0` in `docker/potree/.env` on carrot, `docker compose up -d --build`, open `http://carrot:8080/` over the VPN (or an ssh `-L 8080:localhost:8080` tunnel with the default bind).

In `.claude/skills/segmentanytree-inference/SKILL.md` line 162, replace the `serve_potree.py` command with `cd docker/potree && docker compose up -d --build   # or: python3 benchmark/serve_potree.py --root <site> --port 8080`.

In `CLAUDE.md`, "Skills" list, `ff3d-outputs-and-viewers` entry: change "QGIS and the Potree viewer" to "QGIS and the Potree viewer (served by the `docker/potree/` nginx container over a read-only site mount)".

- [ ] **Step 6: Update the two site READMEs on the 2TB volume (same content, both copies)**

Replace the "Start" section's code block in `/Volumes/2TB/winmol/ALS_Data/berlin_potree_v2/README.md` and `/Volumes/2TB/winmol/ALS_Data/berlin_potree/README.md` with:

````markdown
```bash
# with Docker (recommended): serves this folder read-only with nginx
cd ~/ForestFormer3D/docker/potree && cp -n .env.example .env
# edit POTREE_SITE in .env to this folder, then
docker compose up -d --build            # open http://localhost:8080/

# without Docker
python3 ~/ForestFormer3D/benchmark/serve_potree.py --root . --port 8080
```
````

Keep the sentence about `file://` not working. Also fix the "Notes and limits" tip so it says "Serve with the `docker/potree` container or `serve_potree.py`, not `python3 -m http.server`".

- [ ] **Step 7: Run the docs hygiene tests and commit**

Run: `cd /Users/christian/ForestFormer3D && python3 -m pytest tests -q`
Expected: all pass (some repo tests grep docs and skills for stale commands; if one fails on the replaced `http.server` line, update that test's expectation to the compose command).

```bash
cd /Users/christian/ForestFormer3D
git add docs/benchmarks/2026-09-23-potree-viewer.md .claude/skills/ff3d-outputs-and-viewers/SKILL.md .claude/skills/segmentanytree-inference/SKILL.md CLAUDE.md
git commit -m "docs: serve the Potree site from the docker/potree container"
git push fork fix/review-findings
```

- [ ] **Step 8: Leave the container running or stop it**

`cd docker/potree && docker compose down` stops it; `restart: unless-stopped` means it otherwise comes back with Docker Desktop, which is what you want for a viewer you open often.

---

## Self-review notes

- Design coverage: image (T1), mount + env contract (T1), Range/MIME/sidecar/gzip behaviour (T2), real-site + browser verification and docs on both sides (T3). The carrot deployment is documentation only, on purpose: same files, different `.env`.
- Placeholders: none; every config file is given in full.
- Name consistency: `POTREE_SITE`/`POTREE_PORT`/`POTREE_BIND`, `ff3d-potree:latest`, `ff3d-potree`, `/usr/share/nginx/html` are spelled the same in T1 files, T1 tests, T2 fixture and T3 docs.
- Review Focus 1–5 are each pinned by a named test in T1 or T2; item 3 also has the manual check in T3 step 2.
