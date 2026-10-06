#!/usr/bin/env bash
# Cut the WINMOL 2025 drone orthomosaics (R12 Tegelsee 9.6 cm, R13 Spandau 10 cm, EPSG:32633)
# into Berlin km tiles for the viewer's `drone2025` drape: EPSG:25833, 0.4 m (the drape texture
# is 1792 px per km, so finer is wasted), RGB + alpha, transparent outside the survey footprint
# (the mosaics carry black and white borders that would otherwise paint the tile).
#
#   bash benchmark/cut_drone_ortho_tiles.sh [out dir]          # Mac, MacPorts GDAL
#
# Reads the footprint GeoPackages next to the mosaics to decide which km tiles to cut; writes
# <out>/3dm_33_<E>_<N>_1_be.tif. Environment: GDAL_BIN (/opt/local/bin), DRONE_RES (0.4), PYTHON
# (a python with geopandas; default the repo's .venv-cpu).
set -uo pipefail
OUT="${1:-/Volumes/2TB/winmol/ALS_Data/berlin_drone_2025}"
GDAL="${GDAL_BIN:-/opt/local/bin}"
RES="${DRONE_RES:-0.4}"
PY="${PYTHON:-$(cd "$(dirname "$0")/.." && pwd)/.venv-cpu/bin/python}"
B=/Volumes/2TB/winmol/training_data/WINDWURF_Tegel
mkdir -p "$OUT"
cut_area() {   # $1 = mosaic tif, $2 = footprint gpkg, $3 = label
  local tif="$1" fp="$2" label="$3"
  # km tiles the footprint intersects (ogrinfo -so -q prints no extent line; geopandas does it)
  local tiles
  tiles=$("$PY" - "$fp" <<'PYEOF'
import sys, geopandas as gpd
from shapely.geometry import box
g = gpd.read_file(sys.argv[1], layer="footprint").to_crs(25833).union_all()
minx, miny, maxx, maxy = g.bounds
out = []
for e in range(int(minx // 1000), int(maxx // 1000) + 1):
    for n in range(int(miny // 1000), int(maxy // 1000) + 1):
        if g.intersection(box(e * 1000, n * 1000, e * 1000 + 1000, n * 1000 + 1000)).area > 100:
            out.append(f"{e} {n}")
print("\n".join(out))
PYEOF
)
  while read -r e n; do
      [ -n "$e" ] || continue
      T="3dm_33_${e}_${n}_1_be"
      [ -f "$OUT/$T.tif" ] && { echo "  $T exists"; continue; }
      echo "=== $(date +%T) $label $T"
      "$GDAL/gdalwarp" -q -overwrite -t_srs EPSG:25833 -te $((e * 1000)) $((n * 1000)) $((e * 1000 + 1000)) $((n * 1000 + 1000)) \
        -tr "$RES" "$RES" -r average -cutline "$fp" -cl footprint -dstalpha -srcnodata 0 \
        -co COMPRESS=LZW -co TILED=YES -co PHOTOMETRIC=RGB "$tif" "$OUT/$T.tif" \
        || { echo "!!! $T failed"; rm -f "$OUT/$T.tif"; continue; }
      # drop tiles the footprint does not actually reach (alpha all zero)
      if "$GDAL/gdalinfo" -stats "$OUT/$T.tif" 2>/dev/null | awk '/Band 4/{b=1} b && /STATISTICS_MAXIMUM=/{print $0; exit}' | grep -q "MAXIMUM=0$"; then
        echo "  $T outside the footprint, dropped"; rm -f "$OUT/$T.tif" "$OUT/$T.tif.aux.xml"
      fi
      rm -f "$OUT/$T.tif.aux.xml"
  done <<< "$tiles"
}
cut_area "$B/Revier_12/ortho/result_Res1.2_QGIS_preview_1to8.tif" "$B/Revier_12/ortho/R12_footprint.gpkg" R12
cut_area "$B/Revier_13/Ortho/result_Res10_COG.tif" "$B/Revier_13/Ortho/R13_footprint.gpkg" R13
echo "=== $(date +%T) done: $(ls "$OUT"/3dm_33_*.tif | wc -l) tiles in $OUT"
