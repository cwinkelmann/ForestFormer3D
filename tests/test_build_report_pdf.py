"""benchmark/build_report_pdf.py: chapter assembly and the {{placeholder}} filling (no pandoc)."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmark"))

from build_report_pdf import (  # noqa: E402
    APPENDIX,
    DEFAULT_ORDER,
    MAIN,
    chapter,
    fill_placeholders,
    lookup,
    render_value,
)

DATA = {"methods": {"ff3d": {"trees": 832130, "height_q": [4.3, 18.2, 29.4]}},
        "agreement": {"ff3d_sat": {"matched_frac_a": 0.5377, "iou_median": 0.7419}},
        "stands_r": 0.5}


def test_order_is_main_then_appendix_without_overlap():
    assert DEFAULT_ORDER == MAIN + APPENDIX
    assert not set(MAIN) & set(APPENDIX)
    assert MAIN[0] == "summary" and "berlin-analytics" in MAIN and "potree-viewer" in APPENDIX


def test_lookup_walks_dicts_and_lists():
    assert lookup(DATA, "methods.ff3d.trees") == 832130
    assert lookup(DATA, "methods.ff3d.height_q.1") == 18.2
    with pytest.raises(KeyError):
        lookup(DATA, "methods.sat.trees")


def test_render_value_formats():
    assert render_value(832130, None) == "832,130"
    assert render_value(0.5377, "pct") == "53.8 %"
    assert render_value(0.5377, "pct0") == "54 %"
    assert render_value(0.7419, "3") == "0.742"
    assert render_value(18.26, None) == "18.3"
    assert render_value(1234.6, "int") == "1,235"


def test_fill_placeholders_substitutes_and_flags_unknown_keys():
    warnings = []
    md = "{{methods.ff3d.trees}} trees, {{agreement.ff3d_sat.matched_frac_a|pct}} matched, r {{stands_r|2}}, {{nope.key}}"
    out = fill_placeholders(md, DATA, warn=warnings.append)
    assert out == "832,130 trees, 53.8 % matched, r 0.50, [n/a: nope.key]"
    assert warnings == ["  placeholder not found in analytics.json: nope.key"]
    assert fill_placeholders("no placeholders", None) == "no placeholders"


def test_chapter_keeps_h2_as_sections_strips_manual_numbers_and_code_comments(tmp_path):
    doc = tmp_path / "2026-10-05-x.md"
    doc.write_text("# My study\n\nintro {{stands_r|2}}\n\n## 1. First\n\n```bash\n# not a heading\n```\n\n### 3b. Sub\n\n![f](assets/missing.png)\n")
    warnings = []
    md = chapter(doc, warn=warnings.append, source_line=False, data=DATA)
    assert "\\newpage\n\n# My study\n" in md
    assert "Source:" not in md
    assert "\n## First\n" in md and "\n### Sub\n" in md
    assert "# not a heading" in md                       # untouched inside the fence
    assert "intro 0.50" in md
    assert "*[figure missing: `assets/missing.png`]*" in md and any("image not found" in w for w in warnings)
    with_src = chapter(doc, warn=lambda m: None, source_line=True, data=DATA)
    assert "*Source: `docs/benchmarks/2026-10-05-x.md`" in with_src
