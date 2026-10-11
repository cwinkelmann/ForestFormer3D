#!/usr/bin/env python3
"""Static file server with HTTP Range support, for the Potree site.

Potree 2.0 fetches every octree node as a byte range out of one large
``octree.bin`` (``Range: bytes=<start>-<end>``). Python's ``http.server``
ignores the Range header and answers ``200`` with the *whole* file, so the
viewer decodes the wrong bytes: point positions come out scrambled, the cloud
renders as scattered blobs, and nothing in the console says why. This server
answers ``206 Partial Content`` and is threaded, so the many parallel node
requests do not queue behind one another.

    python3 benchmark/serve_potree.py --root /Volumes/2TB/winmol/ALS_Data/berlin_potree
    # then open http://localhost:8080/

Only single ranges are implemented (``bytes=a-b``, ``bytes=a-``, ``bytes=-n``),
which is all Potree sends.
"""
from __future__ import annotations

import argparse
import os
import re
from functools import partial
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

_RANGE = re.compile(r"^bytes=(\d*)-(\d*)$")


class RangeRequestHandler(SimpleHTTPRequestHandler):
    """SimpleHTTPRequestHandler plus single-range ``206`` responses."""

    def send_head(self):  # noqa: D102 - mirrors the base class
        header = self.headers.get("Range")
        if not header:
            return self._with_accept_ranges(super().send_head)

        path = self.translate_path(self.path)
        if os.path.isdir(path):
            return self._with_accept_ranges(super().send_head)

        match = _RANGE.match(header.strip())
        if not match:
            self.send_error(HTTPStatus.BAD_REQUEST, "invalid Range header")
            return None

        try:
            size = os.path.getsize(path)
            handle = open(path, "rb")
        except OSError:
            self.send_error(HTTPStatus.NOT_FOUND, "File not found")
            return None

        first, last = match.group(1), match.group(2)
        if first == "":  # bytes=-n -> the last n bytes
            if last == "":
                handle.close()
                self.send_error(HTTPStatus.BAD_REQUEST, "invalid Range header")
                return None
            length = min(int(last), size)
            start = size - length
            end = size - 1
        else:
            start = int(first)
            end = int(last) if last != "" else size - 1
            end = min(end, size - 1)

        if start > end or start >= size:
            handle.close()
            self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
            self.send_header("Content-Range", f"bytes */{size}")
            self.end_headers()
            return None

        handle.seek(start)
        self.send_response(HTTPStatus.PARTIAL_CONTENT)
        self.send_header("Content-Type", self.guess_type(path))
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.send_header("Content-Length", str(end - start + 1))
        self.send_header("Last-Modified", self.date_time_string(int(os.stat(path).st_mtime)))
        self.end_headers()
        return _Slice(handle, end - start + 1)

    def _with_accept_ranges(self, call):
        original = self.send_header

        def send_header(key, value):
            original(key, value)
            if key.lower() == "content-length":
                original("Accept-Ranges", "bytes")

        self.send_header = send_header  # type: ignore[method-assign]
        try:
            return call()
        finally:
            self.send_header = original  # type: ignore[method-assign]

    def log_message(self, fmt, *args):  # quieter: one line per failure only
        if args and str(args[1]).startswith(("4", "5")):
            super().log_message(fmt, *args)


class _Slice:
    """File-like wrapper that yields at most ``remaining`` bytes."""

    def __init__(self, handle, remaining: int) -> None:
        self._handle = handle
        self._remaining = remaining

    def read(self, amount: int = -1) -> bytes:
        if self._remaining <= 0:
            return b""
        if amount is None or amount < 0:
            amount = self._remaining
        chunk = self._handle.read(min(amount, self._remaining))
        self._remaining -= len(chunk)
        return chunk

    def close(self) -> None:
        self._handle.close()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", default="/Volumes/2TB/winmol/ALS_Data/berlin_potree",
                    help="directory to serve (default: the Berlin Potree site)")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--bind", default="127.0.0.1")
    args = ap.parse_args(argv)

    root = os.path.abspath(args.root)
    if not os.path.isdir(root):
        raise SystemExit(f"--root {root} is not a directory")

    handler = partial(RangeRequestHandler, directory=root)
    server = ThreadingHTTPServer((args.bind, args.port), handler)
    print(f"serving {root} at http://{args.bind}:{args.port}/ (Range requests supported)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
