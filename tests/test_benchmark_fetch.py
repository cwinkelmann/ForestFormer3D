"""Tests for benchmark/fetch_zenodo.sh.

Covers the two sourcing modes:
  * cache mode: FF3D_ZENODO_CACHE points at a directory holding files.tsv (tab-separated
    key, url, md5:<hex>, size) plus the archives themselves -- nothing is downloaded, the
    cache directory must not be written to, and archives are unpacked straight from it.
  * API mode (fallback, used only when the cache is absent/incomplete): the Zenodo REST
    record shape (files[].key/size/checksum/links.self) is mimicked by a local
    http.server; files are downloaded into $FF3D_ROOT/work_dirs/.zenodo/downloads/.

No network, no torch, no numpy -- stdlib only.
"""
import hashlib
import json
import os
import subprocess
import threading
import zipfile
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "benchmark" / "fetch_zenodo.sh"

ARCHIVE_SPECS = {
    "train_val_data.zip": [
        ("train_val_data/CULS_plot_2_annotated.ply", b"ply-train"),
        ("train_val_data/CULS_plot_3_annotated.ply", b"ply-val"),
    ],
    "test_data.zip": [
        ("test_data/NIBIO_NIBIO_plot_1_annotated_test.ply", b"ply-test"),
    ],
    "clean_forestformer.zip": [
        ("clean_forestformer/epoch_3000_fix.pth", b"not-a-real-checkpoint"),
    ],
}
KEYS = list(ARCHIVE_SPECS)


def _md5(p: Path) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest()


def _write_archives(dest_dir: Path) -> dict:
    """Writes the three archives into dest_dir. Returns {key: md5}."""
    md5s = {}
    for key, members in ARCHIVE_SPECS.items():
        p = dest_dir / key
        with zipfile.ZipFile(p, "w") as z:
            for arcname, content in members:
                z.writestr(arcname, content)
        md5s[key] = _md5(p)
    return md5s


def _snapshot(d: Path):
    return {
        str(p.relative_to(d)): p.stat().st_mtime_ns
        for p in sorted(d.rglob("*"))
        if p.is_file()
    }


def _run(args, root, env_extra):
    env = dict(os.environ, FF3D_ROOT=str(root), **env_extra)
    return subprocess.run(
        ["bash", str(SCRIPT), *args], env=env, capture_output=True, text=True
    )


# --- cache mode ---------------------------------------------------------------


@pytest.fixture
def cache_dir(tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    md5s = _write_archives(cache)
    lines = []
    for key in KEYS:
        size = (cache / key).stat().st_size
        lines.append(f"{key}\thttp://example.invalid/{key}\tmd5:{md5s[key]}\t{size}")
    (cache / "files.tsv").write_text("\n".join(lines) + "\n")
    return cache, md5s


def test_list_from_cache(tmp_path, cache_dir):
    cache, md5s = cache_dir
    root = tmp_path / "root"
    r = _run(["--list"], root, {"FF3D_ZENODO_CACHE": str(cache)})
    assert r.returncode == 0, r.stdout + r.stderr
    lines = [l.split("\t") for l in r.stdout.strip().splitlines()]
    assert [l[0] for l in lines] == KEYS
    for key, size, md5, url in lines:
        assert md5 == md5s[key]
        assert len(md5) == 32
    # --list is side-effect-free: it must not even create FF3D_ROOT/work_dirs
    assert not (root / "work_dirs").exists()


def test_dry_run_cache_mode_does_not_touch_disk(tmp_path, cache_dir):
    cache, _ = cache_dir
    root = tmp_path / "root"
    r = _run(["--dry-run"], root, {"FF3D_ZENODO_CACHE": str(cache)})
    assert r.returncode == 0, r.stdout + r.stderr
    assert "dry-run" in r.stdout
    for key in KEYS:
        assert key in r.stdout
    assert not (root / "data").exists()
    # --dry-run is side-effect-free: it must not even create FF3D_ROOT/work_dirs
    # (not just the specific data/checkpoint subdirs)
    assert not (root / "work_dirs").exists()


def test_cache_mode_unpacks_and_leaves_cache_untouched(tmp_path, cache_dir):
    cache, md5s = cache_dir
    before = _snapshot(cache)

    root = tmp_path / "root"
    r = _run([], root, {"FF3D_ZENODO_CACHE": str(cache)})
    assert r.returncode == 0, r.stdout + r.stderr

    data = root / "data" / "ForAINetV2"
    assert (data / "train_val_data" / "CULS_plot_2_annotated.ply").read_bytes() == b"ply-train"
    assert (data / "train_val_data" / "CULS_plot_3_annotated.ply").read_bytes() == b"ply-val"
    assert (data / "test_data" / "NIBIO_NIBIO_plot_1_annotated_test.ply").read_bytes() == b"ply-test"
    ckpt = root / "work_dirs" / "clean_forestformer" / "epoch_3000_fix.pth"
    assert ckpt.read_bytes() == b"not-a-real-checkpoint"

    zenodo_dir = root / "work_dirs" / ".zenodo"
    for key in KEYS:
        assert (zenodo_dir / f"{key}.unpacked-{md5s[key]}").exists()

    # the cache directory itself must be byte-for-byte and mtime-for-mtime unchanged
    after = _snapshot(cache)
    assert after == before

    # second run: no re-download (there is nothing to download), no re-unpack
    r2 = _run([], root, {"FF3D_ZENODO_CACHE": str(cache)})
    assert r2.returncode == 0, r2.stdout + r2.stderr
    assert r2.stdout.count("already unpacked") == 3
    assert "\nunpacked:" not in ("\n" + r2.stdout)
    assert _snapshot(cache) == before


def test_cache_wrong_md5_fails(tmp_path, cache_dir):
    cache, md5s = cache_dir
    # corrupt the recorded md5 for one entry without touching the archive itself
    tsv = cache / "files.tsv"
    lines = tsv.read_text().splitlines()
    bad_lines = []
    for line in lines:
        key, url, md5field, size = line.split("\t")
        if key == "test_data.zip":
            md5field = "md5:" + ("0" * 32)
        bad_lines.append("\t".join([key, url, md5field, size]))
    tsv.write_text("\n".join(bad_lines) + "\n")

    root = tmp_path / "root"
    r = _run([], root, {"FF3D_ZENODO_CACHE": str(cache)})
    assert r.returncode != 0
    assert "md5 mismatch" in r.stdout + r.stderr


# --- API mode (fallback, cache absent) -----------------------------------------


@pytest.fixture
def zenodo(tmp_path):
    """A local http.server mimicking the Zenodo REST record shape."""
    srv_dir = tmp_path / "srv"
    srv_dir.mkdir()
    md5s = _write_archives(srv_dir)

    handler = partial(SimpleHTTPRequestHandler, directory=str(srv_dir))
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    port = httpd.server_address[1]
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()

    files = []
    for key in KEYS:
        p = srv_dir / key
        files.append(
            {
                "key": key,
                "size": p.stat().st_size,
                "checksum": "md5:" + md5s[key],
                "links": {"self": f"http://127.0.0.1:{port}/{key}"},
            }
        )
    (srv_dir / "record.json").write_text(json.dumps({"files": files}))
    try:
        yield f"http://127.0.0.1:{port}/record.json", md5s
    finally:
        httpd.shutdown()


def _no_cache_env(tmp_path, api_url):
    return {
        "FF3D_ZENODO_CACHE": str(tmp_path / "no-such-cache"),
        "ZENODO_API": api_url,
    }


def test_list_from_api(tmp_path, zenodo):
    api_url, _ = zenodo
    root = tmp_path / "root"
    r = _run(["--list"], root, _no_cache_env(tmp_path, api_url))
    assert r.returncode == 0, r.stderr
    lines = [l.split("\t") for l in r.stdout.strip().splitlines()]
    assert [l[0] for l in lines] == KEYS
    assert all(len(l) == 4 and len(l[2]) == 32 for l in lines)
    # --list is side-effect-free: it must not even create FF3D_ROOT/work_dirs
    assert not (root / "work_dirs").exists()


def test_dry_run_api_mode_does_not_touch_disk(tmp_path, zenodo):
    api_url, _ = zenodo
    root = tmp_path / "root"
    r = _run(["--dry-run"], root, _no_cache_env(tmp_path, api_url))
    assert r.returncode == 0, r.stdout + r.stderr
    assert "dry-run" in r.stdout
    for key in KEYS:
        assert key in r.stdout
    assert not (root / "data").exists()
    # --dry-run is side-effect-free: it must not download, and must not even create
    # FF3D_ROOT/work_dirs (the downloads dir and .zenodo marker dir both live under it)
    assert not (root / "work_dirs").exists()


def test_api_fetch_places_files_and_is_idempotent(tmp_path, zenodo):
    api_url, _ = zenodo
    root = tmp_path / "root"
    env = _no_cache_env(tmp_path, api_url)

    r = _run([], root, env)
    assert r.returncode == 0, r.stdout + r.stderr
    data = root / "data" / "ForAINetV2"
    assert (data / "train_val_data" / "CULS_plot_2_annotated.ply").read_bytes() == b"ply-train"
    assert (data / "train_val_data" / "CULS_plot_3_annotated.ply").exists()
    assert (data / "test_data" / "NIBIO_NIBIO_plot_1_annotated_test.ply").read_bytes() == b"ply-test"
    assert (root / "work_dirs" / "clean_forestformer" / "epoch_3000_fix.pth").read_bytes() == b"not-a-real-checkpoint"
    assert (root / "work_dirs" / ".zenodo" / "downloads" / "train_val_data.zip").exists()
    assert "train_val_data: 2 ply" in r.stdout
    assert "test_data: 1 ply" in r.stdout

    r2 = _run([], root, env)
    assert r2.returncode == 0, r2.stdout + r2.stderr
    assert r2.stdout.count("already present") == 3
    assert r2.stdout.count("already unpacked") == 3


def test_api_md5_mismatch_fails(tmp_path, zenodo):
    api_url, _ = zenodo
    root = tmp_path / "root"
    env = _no_cache_env(tmp_path, api_url)

    # simulate a pre-existing corrupt download that the resume+md5-check must catch
    dl = root / "work_dirs" / ".zenodo" / "downloads"
    dl.mkdir(parents=True)
    (dl / "test_data.zip").write_bytes(b"corrupt-partial")

    r = _run([], root, env)
    assert r.returncode != 0
    assert "md5 mismatch" in r.stdout + r.stderr


def test_bash_syntax():
    r = subprocess.run(["bash", "-n", str(SCRIPT)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
