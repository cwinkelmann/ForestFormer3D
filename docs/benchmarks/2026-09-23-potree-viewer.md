# Potree web viewer for the Berlin ALS 2021 tiles (2026-09-23)

**What.** An offline Potree site for exploring the ForestFormer3D segmentation of the
Berlin ALS 2021 tiles around Tegel at full resolution: the per-tile point clouds as
Potree 2 octrees coloured by tree id / semantic class / elevation / instance score /
ALS class, plus draped CHM, instance mask and the 2021 (leaf-off) and 2025 (leaf-on)
orthophotos, crown outlines and clickable tree markers.

**Where.** `/Volumes/2TB/winmol/ALS_Data/berlin_potree/` on the 2 TB volume; its
`README.md` documents every control, the attribute mapping and how to add a tile.

```bash
cd /Volumes/2TB/winmol/ALS_Data/berlin_potree && python3 benchmark/serve_potree.py --root /Volumes/2TB/winmol/ALS_Data/berlin_potree
# then http://localhost:8080/
```

## Build

| step | script |
|---|---|
| LAS → Potree 2 octree (on carrot) | `benchmark/potree_convert_tile.py` + PotreeConverter 2.1.1 (Linux x64 release) |
| overlays, manifest, `index.html`  | `benchmark/build_potree_site.py` |
| viewer page (template)            | `benchmark/potree_index.html` |

```bash
# on carrot, per tile (seconds each; the octree is ~ the size of the LAS)
python3 benchmark/potree_convert_tile.py \
    --las work_dirs/berlin-<tile>/<tile>.las \
    --out work_dirs/logs/potree/out/<tile> \
    --potree-converter work_dirs/logs/potree/PotreeConverter_linux_x64/PotreeConverter

# on the Mac, after rsyncing out/<tile>/ into <site>/pointclouds/<tile>/
.venv-cpu/bin/python benchmark/build_potree_site.py \
    --site /Volumes/2TB/winmol/ALS_Data/berlin_potree \
    --ff3d-dir /Volumes/2TB/winmol/ALS_Data/berlin_als_2021_ff3d \
    --dop2021-dir /Volumes/2TB/winmol/ALS_Data/berlin_dop_2021/dop_2021_rgb \
    --dop2025-dir /Volumes/2TB/winmol/ALS_Data/berlin_dop_2025_sommer
```

## Findings worth keeping

* **Extra attributes survive the conversion.** PotreeConverter 2.1.1 carries the LAS
  extra bytes through into the octree, so `treeID` (int32, −1 = none), `semantic`
  (uint8) and `score` (float) are native Potree attributes and appear in
  `metadata.json`. No `point_source_id` / `user_data` fallback encoding was needed.
* **PotreeConverter aborts on our LAS files** with
  `nlohmann::detail::type_error … invalid UTF-8 byte at index 32: 0xC0`. Cause: the
  `treeID` extra-bytes description `ForestFormer3D instance, -1 none` is exactly 32
  bytes and fills the field with no null terminator, and the converter reads past it
  into the binary that follows. `potree_convert_tile.py` streams a copy of the LAS with
  those descriptions null-terminated and converts that; the originals are untouched.
  Worth fixing at the source in the writer that produces these LAS files.
* **Categorical colouring in Potree needs a texture trick.** Potree colours a scalar
  attribute by sampling a 1-D gradient texture, and its point-cloud renderer only
  uploads textures created by its own bundled three.js build (`instanceof` checks). The
  viewer therefore takes a texture Potree made itself and swaps in a 1-row, 8192-wide
  canvas of random colours with nearest filtering. Tiles hold 15–25 k trees, so 2–3
  consecutive ids share a colour slot; consecutive ids are ~50 m apart (median), so
  trees sharing a colour are essentially never adjacent.

## Coverage

All eleven km² tiles are in the site: 379_5828, 379_5829, 380_5828, 380_5829,
381_5828, 381_5829, 381_5830, 382_5828, 382_5829, 383_5828, 383_5829 — the last one
finished inferring on carrot while the site was being built and was converted last.
227 M points, 300 452 trees, 8.5 GB of octrees and 374 MB of overlays; conversion took
4–10 s per tile on carrot and the octree came out roughly the size of the source LAS.

## Not verified

The page was checked by serving the folder over `python3 -m http.server` and fetching
`index.html`, `data/tiles.json` and an octree `metadata.json`, and by reading the Potree
1.8.2 sources for every API it calls. It has **not** been opened in a browser, so the
rendering itself — octree display, the LUT texture upload, the draped planes and the
marker picking — is unverified.

## Serving and two viewer fixes (verified in a browser)

The site must be served by something that answers HTTP Range requests. Potree 2.0 reads every
octree node as a byte range out of one large `octree.bin`; `python3 -m http.server` ignores the
`Range` header and returns the whole file with `200`, so the viewer decodes the wrong bytes and
the cloud appears as scattered blobs with no error anywhere. `benchmark/serve_potree.py` answers
`206 Partial Content` and is threaded for the parallel node requests:

```bash
python3 benchmark/serve_potree.py --root /Volumes/2TB/winmol/ALS_Data/berlin_potree --port 8080
```

Two fixes in `benchmark/potree_index.html`, both found by driving the page in a headless browser:

- **proj4 has no EPSG:25833.** Potree's per-frame update passes the point cloud's CRS to proj4 for
  its map view; an unknown code throws, the exception kills the requestAnimationFrame loop, and the
  page freezes after a few frames. The page now registers the definition before the viewer starts.
- **Extra attributes are not normalised.** Potree 1.8.2 assumes extra attributes arrive in [0,1],
  but its decoder only rescales types larger than four bytes, so `treeID` (int32), `semantic`
  (uint8) and `score` (float) reach the shader raw and every tree clamps to a single colour.
  Setting the attribute's `initialRange` to [0,1] makes the renderer compute the right scale and
  offset, and the per-tree colours appear.

Empty crown geometries (24 to 155 per tile, trees whose hull degenerates) also aborted the vector
build; the crown loop now skips them.
