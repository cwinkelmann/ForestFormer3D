---
name: ff3d-pdf-report
description: Use when asked for a PDF report of the ForestFormer3D / Berlin ALS work, to rebuild it after a benchmark write-up changes, or to add a study to it - benchmark/build_report_pdf.py assembles docs/benchmarks/*.md into one pandoc+xelatex PDF with a title page, TOC and one chapter per study, and this skill says how to build, verify, extend and where the known traps are.
---

# The PDF report of the Berlin ForestFormer3D work

The project's reporting is written as it happens, one markdown document per study under
`docs/benchmarks/`. There was no PDF until 2026-10-05; the report is **assembled from those
documents**, never written separately, so it can be rebuilt in a minute after any write-up
changes and it never drifts from them.

## Build

```bash
cd /Users/christian/ForestFormer3D
.venv-cpu/bin/python benchmark/build_report_pdf.py --keep-md
#  -> docs/benchmarks/report/berlin-ff3d-report.pdf  (and .md, the assembled source)
```

Needs pandoc (`/opt/homebrew/bin/pandoc`, 3.1) and xelatex (TeX Live 2022 in
`/Library/TeX/texbin`), both present on the Mac; `pypdf` in `.venv-cpu` for the page count.
Fonts are Helvetica Neue / Menlo via fontspec, so the build is Mac-only as written. The
output directory is git-ignored: the PDF is ~13 MB and regenerable.

What the builder does per document: the first H1 (outside code fences) becomes the chapter
title, every other heading is shifted down one level, a `\newpage` precedes each chapter, a
"Source: ..., last edited" line is inserted, and an image reference whose file does not
exist is replaced by a visible *[figure missing: ...]* note -- a missing image otherwise
aborts xelatex with an error naming only a temp file.

## Structure: the report proper, then an appendix

Since 2026-10-05 the PDF has two parts. `MAIN` in the script is the reader-facing report
-- `summary` (executive summary), `introduction` (question, study area figure
`assets/analytics/analytics_study_area.png`, data at a glance), `data-sources`, `methods`,
`berlin-analytics` (the GENERATED results chapter) and `discussion` (findings, limitations,
outlook) -- without source lines. `APPENDIX` holds the working studies as they were written
(`carrot-ff3d`, `inference-profile`, `als-density-eval`, `tegel-als`, `tegel-berlin-2021`,
`spandau-berlin-2021`, `berlin-visual-report`, `berlin-dop-overlay`, `seamless-ids`,
`segmentanytree-berlin`, `ams3d-berlin`, `potree-viewer`), each with its "Source: ..., last
edited" line, behind a `\appendix` part page. `RUNBOOK-*.md` and the session log stay out.

**Numbers in the hand-written chapters are placeholders**, filled at build time from
`docs/benchmarks/assets/analytics/analytics.json` (written by `benchmark/berlin_analytics.py`):
`{{methods.ff3d.trees}}`, `{{agreement.ff3d_sat.matched_frac_a|pct}}`, `{{stands_r|2}}`,
`{{methods.ff3d.height_q.1}}` (list index), formats `pct`, `pct0`, `int`, or a digit count.
An unknown key renders as a visible `[n/a: key]` and a warning, never silently. So after
the mosaic changes: run `berlin_analytics.py`, then rebuild, and the summary follows the
data. Never write a number by hand into summary/introduction/discussion that the analytics
JSON has.

Headings: the first H1 is the chapter, H2/H3 stay sections/subsections (pandoc numbers
them; a manual "## 3b. Title" prefix is stripped). To add a study, write it as
`docs/benchmarks/<date>-<key>.md` with one H1 and put `<key>` into `APPENDIX` (or `MAIN`
if it is reader-facing); `--docs` builds an ad-hoc subset.

## Verify before handing it over

Count chapters from the **PDF outline**, not by grepping the markdown: shell comments
inside fenced code blocks start with `# ` and pandoc correctly leaves them alone, so a
`grep -c '^# '` reports ~30 for a 12-chapter report and misleads.

```bash
.venv-cpu/bin/python - <<'PY'
from pypdf import PdfReader
r = PdfReader("docs/benchmarks/report/berlin-ff3d-report.pdf")
top = [o for o in r.outline if not isinstance(o, list)]
print(len(r.pages), "pages |", len(top), "chapters |", sum(len(p.images) for p in r.pages), "images")
for o in top: print(" -", o.title)
PY
```

Expected on 2026-10-05 (evening): ~105 pages, 6 chapters + 12 appendix studies (7 top-level
outline entries, the last being the appendix part), 25 images. Two figures referenced by `2026-09-22-tegel-als.md`
(`assets/2026-09-22-tegel-r12-qgis.png`, `-r13-qgis.png`) do not exist in
`docs/benchmarks/assets/` and appear as *figure missing* notes; they are QGIS screenshots
the user takes by hand -- drop them into `assets/` and rebuild.

## Traps already hit

- **No hand-written LaTeX in the YAML header.** A `header-includes` block with
  `\usepackage{fvextra}...commandchars=\\\{\}` failed with "There's no line here to end";
  pandoc's own template already loads longtable, booktabs and fvextra. Keep the YAML to
  pandoc variables (`documentclass`, `geometry`, fonts, `toc`).
- **Fences can be indented or `~~~`.** The heading shift must track `^\s*(```|~~~)`, or a
  `# comment` inside a bash block becomes a chapter.
- **`mdls -name kMDItemNumberOfPages` returns null** on a file Spotlight has not indexed
  yet, i.e. always right after the build. Use pypdf.
- Wide pipe tables render as `longtable` and wrap; the per-tile result tables in the
  Berlin runs are the widest and still fit A4 at 10 pt with 22 mm margins. If a new table
  overflows, that chapter is the place to shorten column headers, not the builder.

Report-level conventions follow the docs: no ground truth exists for the Berlin tiles, so
the title page states that every "agreement" figure compares two methods and says nothing
about which is right; keep that sentence when editing `title_block()`.
