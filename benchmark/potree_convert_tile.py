#!/usr/bin/env python3
"""Convert one ForestFormer3D tile LAS into a Potree 2 octree.

PotreeConverter 2.1.x aborts on these files with

    nlohmann::detail::type_error … invalid UTF-8 byte at index 32

because ForestFormer3D writes an extra-bytes description that fills all 32 bytes of
the field with no null terminator, and the converter keeps reading into the binary
that follows. This script therefore streams a copy of the LAS with those descriptions
null-terminated, converts the copy, and deletes it again. Nothing else is touched, so
``treeID``, ``semantic`` and ``score`` survive as native octree attributes.

    python3 benchmark/potree_convert_tile.py \
        --las work_dirs/berlin-<tile>/<tile>.las \
        --out work_dirs/logs/potree/out/<tile> \
        --potree-converter work_dirs/logs/potree/PotreeConverter_linux_x64/PotreeConverter
"""

from __future__ import annotations

import argparse
import os
import shutil
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

HEADER_READ = 1 << 16


def patched_copy(src: Path, dst: Path) -> list[str]:
    """Copy ``src`` to ``dst``, null-terminating extra-bytes descriptions."""
    fixed = []
    with open(src, "rb") as f:
        head = bytearray(f.read(HEADER_READ))
        hsize = struct.unpack_from("<H", head, 94)[0]
        offd = struct.unpack_from("<I", head, 96)[0]
        nvlr = struct.unpack_from("<I", head, 100)[0]
        if offd > HEADER_READ:
            raise SystemExit(f"{src}: point data starts past {HEADER_READ} bytes")
        o = hsize
        for _ in range(nvlr):
            uid = head[o + 2:o + 18].split(b"\x00")[0]
            rid = struct.unpack_from("<H", head, o + 18)[0]
            length = struct.unpack_from("<H", head, o + 20)[0]
            if uid == b"LASF_Spec" and rid == 4:
                for k in range(length // 192):
                    rec = o + 54 + k * 192
                    name = bytes(head[rec + 4:rec + 36]).split(b"\x00")[0].decode("ascii", "replace")
                    d0 = rec + 160
                    desc = bytes(head[d0:d0 + 32])
                    if b"\x00" not in desc:
                        head[d0:d0 + 32] = (desc[:31].rstrip() + b"\x00").ljust(32, b"\x00")
                        fixed.append(name)
            o += 54 + length
        with open(dst, "wb") as g:
            g.write(head)
            shutil.copyfileobj(f, g, 1 << 24)
    return fixed


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--las", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path, help="octree output directory")
    p.add_argument("--potree-converter", required=True, type=Path)
    p.add_argument("--projection", default="EPSG:25833")
    p.add_argument("--scratch", type=Path, default=None,
                   help="where to put the patched copy (default: system temp)")
    a = p.parse_args()

    scratch = a.scratch or Path(tempfile.gettempdir())
    scratch.mkdir(parents=True, exist_ok=True)
    tmp = scratch / f"{a.las.stem}.patched.las"

    print(f"[{a.las.stem}] copying + patching -> {tmp}", flush=True)
    fixed = patched_copy(a.las, tmp)
    if fixed:
        print(f"[{a.las.stem}] null-terminated descriptions of: {', '.join(fixed)}", flush=True)
    try:
        print(f"[{a.las.stem}] PotreeConverter -> {a.out}", flush=True)
        r = subprocess.run(
            [str(a.potree_converter), str(tmp), "-o", str(a.out), "--projection", a.projection],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        sys.stdout.write(r.stdout[-2000:])
    finally:
        os.remove(tmp)

    ok = (a.out / "metadata.json").exists()
    print(f"[{a.las.stem}] rc={r.returncode} metadata.json={ok}", flush=True)
    return 0 if ok and r.returncode == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
