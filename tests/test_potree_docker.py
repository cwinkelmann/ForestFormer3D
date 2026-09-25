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
