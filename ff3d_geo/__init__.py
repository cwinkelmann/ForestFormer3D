"""Geospatial glue for running ForestFormer3D on georeferenced ALS tiles.

Pure Python (numpy, laspy, plyfile, shapely, pyproj, geopandas); no CUDA.
Modules: origin, convert, grid, trees, baseline, report, cli.

This module performs no eager imports of geo libraries, so
``import ff3d_geo.origin`` (and other pure-Python submodules) works even
when laspy/shapely/pyproj/geopandas are not installed.
"""

__version__ = "0.1.0"
