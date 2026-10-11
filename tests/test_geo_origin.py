"""Tests for ff3d_geo.origin.parse_origin.

Note: parse_origin is pure Python (stdlib only), so this module needs no
``pytest.importorskip``. Every other ``tests/test_geo_*.py`` module that
touches a geo library (laspy, shapely, pyproj, geopandas, ...) must start
with ``pytest.importorskip("<library>")`` for each library it needs, so the
system-python test run (no geo libs installed) skips it instead of failing.
"""

import pytest

from ff3d_geo.origin import parse_origin


def test_parse_origin_from_tegel_name():
    assert parse_origin("r12_tegel_E381300_N5828300_100m.las") == (381300.0, 5828300.0)


def test_parse_origin_uses_only_the_file_name():
    assert parse_origin("/data/E1_N2/r13_spandau_E376400_N5827400_100m.las") == (376400.0, 5827400.0)


def test_parse_origin_rejects_names_without_token():
    with pytest.raises(ValueError, match="E<int>_N<int>"):
        parse_origin("plot_42.las")
