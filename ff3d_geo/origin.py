"""Tile origin (lower-left corner easting/northing) encoded in a file name."""

import re
from pathlib import Path

_ORIGIN_RE = re.compile(r"E(\d+)_N(\d+)")


def parse_origin(name: str) -> tuple[float, float]:
    """Return (easting, northing) parsed from an ``E<int>_N<int>`` token.

    ``name`` may be a bare file name or a path; only the final path component
    is searched. Raises ``ValueError`` when the token is absent.
    """
    match = _ORIGIN_RE.search(Path(name).name)
    if match is None:
        raise ValueError(
            f"no E<int>_N<int> origin token in {name!r}; pass --origin E N"
        )
    return float(match.group(1)), float(match.group(2))
