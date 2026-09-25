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


def test_env_example_does_not_default_to_the_exfat_volume():
    # On the Mac, Docker Desktop hangs any bind mount from /Volumes/2TB (exFAT via FSKit)
    # and wedges the daemon: `cp .env.example .env && docker compose up` must not do that.
    env = _text(ENV_EXAMPLE)
    site = re.search(r"^POTREE_SITE=(.*)$", env, re.M).group(1).strip()
    assert site == "", "POTREE_SITE must be left empty so `:?` forces an explicit choice"
    assert "exFAT" in env
    compose = _text(COMPOSE)
    assert "/Volumes/" not in compose, "the compose error text must not suggest the exFAT volume"


def test_docs_tell_a_linux_copy_to_open_the_modes():
    # The exFAT tree is 0700 throughout; `rsync -a` keeps that and nginx (uid 101) gets 403.
    skill = _text(REPO_ROOT / ".claude" / "skills" / "ff3d-outputs-and-viewers" / "SKILL.md")
    assert "--chmod=" in skill
    doc = _text(REPO_ROOT / "docs" / "benchmarks" / "2026-09-23-potree-viewer.md")
    assert "--chmod=" in doc


def test_env_file_is_gitignored():
    ignore = _text(REPO_ROOT / ".gitignore")
    assert re.search(r"^docker/potree/\.env$", ignore, re.M)


# --- container smoke tests: skipped cleanly when the docker daemon is not reachable ---
import gzip  # noqa: E402
import json  # noqa: E402
import shutil  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402
import urllib.error  # noqa: E402
import urllib.request  # noqa: E402


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
    # nginx advertises Accept-Ranges on the 200 response, not on the 206 (RFC 9110 leaves it optional there)


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
    assert headers["Accept-Ranges"] == "bytes"
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
