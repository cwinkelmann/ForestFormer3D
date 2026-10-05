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

## Which documents, in what order

`DEFAULT_ORDER` in the script is the narrative, not the file-name order:

1. `carrot-ff3d` -- the released model reproduced on the H100 host
2. `inference-profile` -- where inference time goes, `region_step_factor`
3. `als-density-eval` -- the thinning study
4. `tegel-als`, `tegel-berlin-2021`, `spandau-berlin-2021` -- the Berlin runs
5. `berlin-visual-report`, `berlin-dop-overlay` -- figures and orthophoto overlays
6. `seamless-ids` -- halo + mosaic-wide stitch
7. `segmentanytree-berlin`, `ams3d-berlin` -- the two comparison methods
8. `potree-viewer`

`RUNBOOK-*.md` and the session log are deliberately NOT in it: a reader-facing report is
not an operations manual. To add a study, write it as `docs/benchmarks/<date>-<key>.md`
with one H1 and put `<key>` into `DEFAULT_ORDER` where it belongs in the story; `--docs`
builds an ad-hoc subset without touching the default.

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

Expected on 2026-10-05: 78 pages, 12 chapters, 17 images, every chapter title a study
name. Two figures referenced by `2026-09-22-tegel-als.md`
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
