#!/usr/bin/env python3
"""Assemble the Berlin ForestFormer3D benchmark write-ups into ONE PDF report.

The project's reporting lives in ``docs/benchmarks/*.md`` -- one document per study,
written as the work happened. This script concatenates a curated, ordered subset of them
into a single PDF with a title page and a table of contents, via pandoc + xelatex (both
installed on the Mac; see ``.claude/skills/ff3d-pdf-report/SKILL.md``).

Each source document becomes one chapter: its own headings are shifted down one level and
a chapter heading is inserted from its first H1, so the TOC reads study-by-study. Image
references are resolved against ``docs/benchmarks``; a reference to a file that does not
exist is REPLACED by a visible note rather than left in, because a missing image aborts
xelatex with an error that names a temp file and nothing else.

    python benchmark/build_report_pdf.py                    # docs/benchmarks/report/berlin-ff3d-report.pdf
    python benchmark/build_report_pdf.py --out /tmp/r.pdf --docs seamless-ids segmentanytree-berlin

The default document order is the narrative order, not the file-name order: what the
model is and how it was validated, then the Berlin runs, then the method comparisons.
"""

from __future__ import annotations

import argparse
import datetime as dt
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DOCS = REPO / "docs" / "benchmarks"

# Narrative order. Keys are the file names without the date prefix and ``.md``;
# RUNBOOK-* and the session log are deliberately not in a reader-facing report.
DEFAULT_ORDER = [
    "data-sources",            # where every input comes from, licences, what is NOT used
    "carrot-ff3d",             # the released model reproduced on the H100 host
    "inference-profile",       # where inference time goes, region_step_factor
    "als-density-eval",        # thinning study: what ALS density the model tolerates
    "tegel-als",               # first real ALS plots (r12, r13)
    "tegel-berlin-2021",       # Berlin ALS 2021 km tiles around Tegel
    "spandau-berlin-2021",     # ... and Spandau
    "berlin-visual-report",    # figures
    "berlin-dop-overlay",      # orthophoto overlays
    "seamless-ids",            # halo + mosaic-wide stitch
    "segmentanytree-berlin",   # comparison method 1
    "ams3d-berlin",            # comparison method 2
    "potree-viewer",           # how to look at it
]

IMG_RE = re.compile(r"!\[([^\]]*)\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")


def find_doc(key: str) -> Path:
    hits = sorted(DOCS.glob(f"*{key}.md"))
    if not hits:
        raise FileNotFoundError(f"no docs/benchmarks/*{key}.md")
    return hits[-1]


def chapter(path: Path, warn=print) -> str:
    """One document as a chapter: H1 -> chapter title, all headings shifted down one."""
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    title = path.stem
    in_code = False
    for l in lines:
        if re.match(r"^\s*(`{3,}|~{3,})", l):
            in_code = not in_code
        elif not in_code and l.startswith("# "):
            title = l[2:].strip()
            break
    body = []
    in_code = False
    # Fences may be indented and may be ~~~: a "# comment" inside a bash block is not a
    # heading, and treating it as one turned shell comments into report chapters.
    fence = re.compile(r"^\s*(`{3,}|~{3,})")
    heading = re.compile(r"^#{1,6} ")
    for line in lines:
        if fence.match(line):
            in_code = not in_code
        if not in_code and heading.match(line):
            if line.startswith("# "):
                continue                      # the chapter heading replaces the H1
            line = "#" + line                 # shift every other heading down one level
        body.append(line)
    md = "\n".join(body)

    def fix_img(m: re.Match) -> str:
        alt, src = m.group(1), m.group(2)
        if src.startswith("http"):
            return m.group(0)
        if not (DOCS / src).exists():
            warn(f"  {path.name}: image not found, replaced by a note: {src}")
            return f"*[figure missing: `{src}`]*"
        return m.group(0)

    md = IMG_RE.sub(fix_img, md)
    stamp = dt.datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d")
    return (f"\n\n\\newpage\n\n# {title}\n\n"
            f"*Source: `docs/benchmarks/{path.name}`, last edited {stamp}.*\n\n{md}\n")


def title_block(docs: list[Path]) -> str:
    today = dt.date.today().isoformat()
    return (
        "---\n"
        "title: \"ForestFormer3D on the Berlin ALS 2021 tiles\"\n"
        "subtitle: \"Tree segmentation at km-tile scale: validation, seamless ids, and two comparison methods\"\n"
        f"date: \"{today}\"\n"
        "lang: en\n"
        "toc: true\n"
        "toc-depth: 2\n"
        "numbersections: true\n"
        "documentclass: report\n"
        "geometry: \"a4paper, margin=22mm\"\n"
        "mainfont: \"Helvetica Neue\"\n"
        "monofont: \"Menlo\"\n"
        "fontsize: 10pt\n"
        "linkcolor: NavyBlue\n"
        "colorlinks: true\n"
        # No header-includes: pandoc already loads longtable/booktabs/fvextra for its own
        # templates, and hand-written LaTeX in YAML is where the first build failed
        # ("There's no line here to end" from an escaped backslash sequence).
        "---\n\n"
        "This report is assembled by `benchmark/build_report_pdf.py` from the working write-ups in "
        f"`docs/benchmarks/`. {len(docs)} studies, in narrative order. No ground truth exists for "
        "the Berlin tiles; figures labelled *agreement* compare two methods and say nothing about "
        "which one is right.\n"
    )


def build(keys: list[str], out: Path, keep_md: bool = False) -> Path:
    for tool in ("pandoc", "xelatex"):
        if not shutil.which(tool):
            raise RuntimeError(f"{tool} not on PATH")
    docs = [find_doc(k) for k in keys]
    md = title_block(docs) + "".join(chapter(d) for d in docs)
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as td:
        src = Path(td) / "report.md"
        src.write_text(md, encoding="utf-8")
        if keep_md:
            shutil.copy(src, out.with_suffix(".md"))
        cmd = ["pandoc", str(src), "-o", str(out), "--pdf-engine=xelatex",
               "--resource-path", str(DOCS), "--from", "markdown+pipe_tables+grid_tables",
               "--wrap=preserve"]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            tail = "\n".join(r.stderr.splitlines()[-25:])
            raise RuntimeError(f"pandoc failed ({r.returncode}):\n{tail}")
    return out


def pdf_pages(path: Path) -> int:
    """Page count via pypdf (xelatex writes compressed object streams, so the bytes
    cannot simply be grepped for page objects)."""
    from pypdf import PdfReader

    return len(PdfReader(str(path)).pages)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=DOCS / "report" / "berlin-ff3d-report.pdf")
    ap.add_argument("--docs", nargs="+", default=DEFAULT_ORDER, help="doc keys in order (default: the narrative)")
    ap.add_argument("--keep-md", action="store_true", help="also write the assembled markdown next to the PDF")
    a = ap.parse_args(argv)
    out = build(a.docs, a.out, a.keep_md)
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB, {pdf_pages(out)} pages, {len(a.docs)} chapters)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
