#!/usr/bin/env python3
"""Assemble the Berlin ForestFormer3D benchmark write-ups into ONE PDF report.

The project's reporting lives in ``docs/benchmarks/*.md`` -- one document per study,
written as the work happened. This script concatenates a curated, ordered subset of them
into a single PDF with a title page and a table of contents, via pandoc + xelatex (both
installed on the Mac; see ``.claude/skills/ff3d-pdf-report/SKILL.md``). The report proper
(``MAIN``: summary, introduction, data sources, methods, the generated analytics chapter,
discussion) is followed by an appendix (``APPENDIX``) holding the working studies.

Each source document becomes one chapter: its first H1 is the chapter title, its H2/H3 are
the sections and subsections (manual "1." numbering is stripped; pandoc numbers them). Image
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

# The report proper: written for a reader, in this order. Numbers inside these documents
# are {{placeholders}} filled from assets/analytics/analytics.json at build time, so the
# summary never drifts from the generated analytics chapter.
MAIN = [
    "summary",                 # executive summary
    "introduction",            # question, study area, data at a glance
    "data-sources",            # where every input comes from, licences, what is NOT used
    "methods",                 # the pipeline, the three methods, references, metrics
    "berlin-analytics",        # results: comparisons and analytics across the mosaic (generated)
    "discussion",              # findings, limitations, outlook
]
# The studies in detail, as they were written while the work happened; appended as an
# appendix with their source lines. RUNBOOK-* and the session log stay out.
APPENDIX = [
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
DEFAULT_ORDER = MAIN + APPENDIX
ANALYTICS_JSON = DOCS / "assets" / "analytics" / "analytics.json"
PLACEHOLDER_RE = re.compile(r"\{\{([A-Za-z0-9_.\-]+)(?:\|([a-z0-9]+))?\}\}")


def lookup(data, dotted: str):
    """``methods.ff3d.height_q.1`` -> the value; KeyError when any step is missing."""
    cur = data
    for part in dotted.split("."):
        if isinstance(cur, list):
            cur = cur[int(part)]
        elif isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            raise KeyError(dotted)
    return cur


def render_value(v, fmt: str | None) -> str:
    if fmt == "pct":
        return f"{100 * float(v):.1f} %"
    if fmt == "pct0":
        return f"{100 * float(v):.0f} %"
    if fmt == "int":
        return f"{int(round(float(v))):,}"
    if fmt and fmt.isdigit():
        return f"{float(v):,.{int(fmt)}f}"
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, int):
        return f"{v:,}"
    if isinstance(v, float):
        return f"{v:,.1f}"
    return str(v)


def fill_placeholders(md: str, data: dict | None, warn=print) -> str:
    """Replace ``{{key.path}}`` / ``{{key.path|fmt}}`` with values from the analytics JSON;
    a key that does not exist becomes a visible ``[n/a: key]`` so the gap is seen, not hidden."""
    def sub(m: re.Match) -> str:
        key, fmt = m.group(1), m.group(2)
        try:
            return render_value(lookup(data or {}, key), fmt)
        except (KeyError, IndexError, ValueError, TypeError):
            warn(f"  placeholder not found in analytics.json: {key}")
            return f"[n/a: {key}]"
    return PLACEHOLDER_RE.sub(sub, md)


IMG_RE = re.compile(r"!\[([^\]]*)\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")


def find_doc(key: str) -> Path:
    hits = sorted(DOCS.glob(f"*{key}.md"))
    if not hits:
        raise FileNotFoundError(f"no docs/benchmarks/*{key}.md")
    return hits[-1]


def chapter(path: Path, warn=print, source_line: bool = True, data: dict | None = None) -> str:
    """One document as a chapter: H1 -> chapter title, all headings shifted down one."""
    text = fill_placeholders(path.read_text(encoding="utf-8"), data, warn)
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
    manual_no = re.compile(r"^(#{2,6}) \d+[a-z]?\. ")   # "## 3b. Title": pandoc numbers sections itself
    for line in lines:
        if fence.match(line):
            in_code = not in_code
        if not in_code and heading.match(line):
            if line.startswith("# "):
                continue                      # the chapter heading replaces the H1
            line = manual_no.sub(r"\1 ", line)  # H2 stays a section: pandoc maps H1 -> chapter, H2 -> section
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
    src = f"*Source: `docs/benchmarks/{path.name}`, last edited {stamp}.*\n\n" if source_line else ""
    return f"\n\n\\newpage\n\n# {title}\n\n{src}{md}\n"


def title_block(n_main: int, n_appendix: int) -> str:
    today = dt.date.today().isoformat()
    return (
        "---\n"
        "title: \"Individual tree segmentation on Berlin's 2021 airborne laser scanning\"\n"
        "subtitle: \"ForestFormer3D, SegmentAnyTree and AMS3D on 33 km² of Tegel and Spandau forest: "
        "a seamless mosaic, method comparison, and checks against the forest inventory and the tree cadastre\"\n"
        "author: \"Christian Winkelmann\"\n"
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
        f"The report has {n_main} chapters and an appendix of {n_appendix} working studies, the write-ups "
        "made while the work was done, reproduced as they are with their source files named. No ground "
        "truth exists for the Berlin tiles: every figure labelled *agreement* compares two methods and "
        "says nothing about which one is right. Everything is regenerated from the data by "
        "`benchmark/berlin_analytics.py` and assembled by `benchmark/build_report_pdf.py`.\n"
    )


APPENDIX_BREAK = ("\n\n\\newpage\n\n\\appendix\n\n"
                  "\\part*{Appendix: the studies in detail}\n"
                  "\\addcontentsline{toc}{part}{Appendix: the studies in detail}\n\n"
                  "The chapters that follow are the working write-ups, one per study, in the order the "
                  "work happened. They carry the operational detail -- machines, commands, run times, "
                  "per-tile tables -- that the report proper summarises.\n")


def build(keys: list[str], out: Path, keep_md: bool = False) -> Path:
    for tool in ("pandoc", "xelatex"):
        if not shutil.which(tool):
            raise RuntimeError(f"{tool} not on PATH")
    import json
    data = json.loads(ANALYTICS_JSON.read_text()) if ANALYTICS_JSON.exists() else None
    if data is None:
        print(f"  no {ANALYTICS_JSON}: placeholders stay unfilled (run benchmark/berlin_analytics.py)")
    main_keys = [k for k in keys if k in MAIN or k not in APPENDIX]
    app_keys = [k for k in keys if k in APPENDIX]
    md = title_block(len(main_keys), len(app_keys))
    md += "".join(chapter(find_doc(k), source_line=False, data=data) for k in main_keys)
    if app_keys:
        md += APPENDIX_BREAK + "".join(chapter(find_doc(k), source_line=True, data=data) for k in app_keys)
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
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB, {pdf_pages(out)} pages, "
          f"{sum(k in MAIN or k not in APPENDIX for k in a.docs)} chapters + {sum(k in APPENDIX for k in a.docs)} appendix studies)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
