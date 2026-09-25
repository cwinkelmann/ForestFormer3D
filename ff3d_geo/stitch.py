"""Stitch halo sub-tile results into seamless, mosaic-wide tree ids.

``ff3d_geo.split`` cuts a km tile into ``size_m`` cores plus a ``buffer_m`` halo and
records, per sub-tile point, where it came from (``<stem>_ident.npy``, see
:data:`IDENT_DTYPE`) together with ``split_manifest.json``. The model then segments each
sub-tile on its own, so the same physical tree gets a different local ``treeID`` in every
sub-tile that saw it.

This module puts them back together:

1. every point is keyed by its global identity ``(source tile, index in that file)``
   packed into one int64 (:func:`ident_keys`), so a halo point of one sub-tile and a
   core point of its neighbour are recognisably the same point;
2. for each adjacent sub-tile pair (8-neighbourhood) the instances are matched by their
   IoU over the shared points (:func:`match_instances`) and the matches are unioned
   (:class:`UnionFind`), so one tree straddling a grid line becomes one component;
3. every component gets one dense global id, ordered pseudo-randomly (hash of the
   component root) so consecutive ids are spatially unrelated -- the Potree viewer maps
   ids onto 8192 colour slots by position, and a spatially ordered id scheme paints one
   sub-tile in one colour (see the design doc's section 1a);
4. each km tile that OWNS sub-tiles is rewritten in ITS OWN point order with the labels
   of the sub-tile whose CORE owns each point, so no point is written twice and the km
   tiles remain a partition. A point no sub-tile core claimed keeps the nodata values
   ``treeID = -1 / semantic = 255 / score = -1`` (the older ``ff3d_geo.merge`` path left
   such points out of the merged LAS entirely). A km tile that only supplied halo points
   to a neighbour's split owns nothing and is not written at all;
5. the tree table is built ONCE over the whole mosaic and each tree's row is written to
   the single km tile holding most of its points, measured over ALL of its points. A
   tree unified across a km border would otherwise get a partial row in both tiles'
   ``<T>_trees.gpkg``, double-counting it and describing each half as a whole tree.

Everything that touches point arrays is vectorised: a km tile is 25 M points over 100
sub-tiles with 20 m halos (~1.96x its core each), so a per-instance Python pass over
points would cost hours.
"""

from __future__ import annotations

import json
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path

import geopandas as gpd
import laspy
import numpy as np
from shapely.geometry import MultiPoint, Point

from ff3d_geo.convert import result_point_header
from ff3d_geo.report import build_report, recommend, write_report
from ff3d_geo.split import IDENT_DTYPE
from ff3d_geo.trees import TREE_COLUMNS, ground_surface

__all__ = [
    "IDENT_DTYPE",
    "Mosaic",
    "UnionFind",
    "adjacent_pairs",
    "ident_keys",
    "load_mosaic",
    "match_instances",
    "stitch",
]

#: CRS of the Berlin ALS mosaic; the km-tile output is written with it.
EPSG = 25833

#: Point dimensions copied verbatim from the source km tile when it has them.
_COPY_DIMS = ("intensity", "return_number", "number_of_returns")

#: Chunk size for reading/writing a km tile; bounds peak memory independent of tile size.
_CHUNK_POINTS = 2_000_000

#: How many sub-tile result LAS files to keep decoded in memory at once. Each adjacent
#: pair needs both sides, and a sub-tile has up to 8 neighbours, so a small cache turns
#: the ~8 reads per sub-tile into ~1.
_CACHE_SIZE = 32

#: Point-order check (see ``_SubtileCache._check_point_order``): compare every
#: ``_ORDER_STRIDE``-th point of a result LAS with the split sub-tile it came from, to
#: within ``_ORDER_TOL_M``. The tolerance only has to absorb the pipeline's float32
#: centering round trip (sub-millimetre) while still catching a reordering, which moves
#: a point by metres.
_ORDER_STRIDE = 10_000
_ORDER_TOL_M = 0.5

#: Knuth's multiplicative constant, used to shuffle component ids (see step 3).
_MIX64 = 0x9E3779B97F4A7C15

#: IoU above which a losing candidate counts as a "runner up" in ``match_instances``'s
#: optional statistics: the rate at which the greedy one-to-one rule leaves a one-sided
#: over-segmentation as two ids. Purely diagnostic, it never changes a match.
RUNNER_UP_IOU = 0.2

_KEY_INDEX_MASK = np.int64(0xFFFFFFFF)

#: Nodata values for a source point no sub-tile core claimed.
NODATA_TREE_ID = np.int32(-1)
NODATA_SEMANTIC = np.uint8(255)
NODATA_SCORE = np.float32(-1.0)


def ident_keys(ident: np.ndarray, tile_map) -> np.ndarray:
    """Pack an ``IDENT_DTYPE`` array into one int64 key per point.

    ``ident["tile"]`` indexes the sub-tile's OWN manifest sources; ``tile_map`` maps
    those local indices onto the mosaic-wide source order (``Mosaic.sources``, sorted by
    key), so keys from two different manifests are comparable. The key is
    ``global_tile << 32 | index``, which is unique per physical point and fits an int64
    because ``index`` is a uint32.
    """
    tile_map = np.asarray(tile_map, dtype=np.int64)
    tile = tile_map[np.asarray(ident["tile"], dtype=np.int64)]
    return (tile << np.int64(32)) | np.asarray(ident["index"], dtype=np.int64)


def match_instances(
    keys_a: np.ndarray,
    labels_a: np.ndarray,
    keys_b: np.ndarray,
    labels_b: np.ndarray,
    iou_threshold: float = 0.5,
    min_shared: int = 20,
    stats: dict | None = None,
) -> list[tuple[int, int, float, int]]:
    """Match the instances of two sub-tiles over the points they have in common.

    Only the keys present in BOTH arrays take part. For every pair ``(la, lb)`` of
    non-negative labels that co-occur on a shared point,
    ``iou = n_ab / (n_a + n_b - n_ab)`` where ``n_a`` and ``n_b`` are the label's sizes
    *on the shared points only* -- the parts of an instance outside the overlap say
    nothing about whether the two sub-tiles saw the same tree, and counting them would
    make the IoU depend on the halo width.

    Returns ``(la, lb, iou, n_ab)`` for the pairs with ``n_ab >= min_shared`` and
    ``iou >= iou_threshold``, best first. The list is a one-to-one matching: pairs are
    taken greedily in ``-iou`` order (ties by ``(la, lb)``, so the result is
    deterministic) and a label already matched is skipped, so an instance of A joins the
    single best instance of B rather than merging two over-segmented halves of B into
    one component. Above ``iou_threshold = 0.5`` that can only ever discard exact ties
    (two labels of B can each reach IoU 0.5 with the same label of A only when they
    partition it exactly), so it matters only for the lower thresholds.

    Fully vectorised: the pair counting packs ``(la, lb)`` into one int64 and uses a
    single ``np.unique``, rather than ``np.unique(..., axis=1)`` (which sorts rows via
    a void view) or a Python loop over instances.

    ``stats``, when given, is a dict this function accumulates into: it counts, per
    matched pair, how many OTHER candidates above :data:`RUNNER_UP_IOU` share one of the
    pair's two labels (``"runner_up_histogram"``, ``{n_runner_ups: n_pairs}``) and how
    many pairs were matched at all. ``stitch`` writes it to ``stitch.json``: the
    histogram is the 1:N rate of the greedy one-to-one rule on real data, i.e. how often
    a one-sided over-segmentation was left as two ids. It costs nothing -- the IoU of
    every co-occurring label pair is computed anyway.
    """
    keys_a = np.asarray(keys_a)
    keys_b = np.asarray(keys_b)
    labels_a = np.asarray(labels_a, dtype=np.int64)
    labels_b = np.asarray(labels_b, dtype=np.int64)

    _, index_a, index_b = np.intersect1d(keys_a, keys_b, return_indices=True)
    if index_a.size == 0:
        return []
    shared_a = labels_a[index_a]
    shared_b = labels_b[index_b]

    # Label sizes over the shared points (each label counted regardless of what the
    # other sub-tile says about the same point).
    def _sizes(labels: np.ndarray) -> dict[int, int]:
        positive = labels[labels >= 0]
        if positive.size == 0:
            return {}
        values, counts = np.unique(positive, return_counts=True)
        return dict(zip(values.tolist(), counts.tolist()))

    size_a = _sizes(shared_a)
    size_b = _sizes(shared_b)
    if not size_a or not size_b:
        return []

    both = (shared_a >= 0) & (shared_b >= 0)
    if not np.any(both):
        return []
    packed = (shared_a[both] << np.int64(32)) | shared_b[both]
    pairs, n_ab = np.unique(packed, return_counts=True)
    la_all = (pairs >> np.int64(32)).astype(np.int64)
    lb_all = (pairs & _KEY_INDEX_MASK).astype(np.int64)

    # Every co-occurring label pair, scored. Kept down to RUNNER_UP_IOU (not just
    # iou_threshold) so the runner-up statistics below see the near misses too.
    floor_iou = min(iou_threshold, RUNNER_UP_IOU) if stats is not None else iou_threshold
    candidates: list[tuple[int, int, float, int]] = []
    for la, lb, n in zip(la_all.tolist(), lb_all.tolist(), n_ab.tolist()):
        if n < min_shared:
            continue
        union = size_a[la] + size_b[lb] - n
        iou = n / union if union > 0 else 0.0
        if iou >= floor_iou:
            candidates.append((int(la), int(lb), float(iou), int(n)))

    candidates.sort(key=lambda p: (-p[2], p[0], p[1]))
    used_a: set[int] = set()
    used_b: set[int] = set()
    matched: list[tuple[int, int, float, int]] = []
    for la, lb, iou, n in candidates:
        if iou < iou_threshold:
            continue
        if la in used_a or lb in used_b:
            continue
        used_a.add(la)
        used_b.add(lb)
        matched.append((la, lb, iou, n))

    if stats is not None:
        _record_runner_ups(stats, candidates, matched)
    return matched


def _record_runner_ups(stats: dict, candidates: list, matched: list) -> None:
    """Accumulate, per matched pair, how many other candidates share one of its labels."""
    by_a: dict[int, int] = {}
    by_b: dict[int, int] = {}
    for la, lb, iou, _n in candidates:
        if iou >= RUNNER_UP_IOU:
            by_a[la] = by_a.get(la, 0) + 1
            by_b[lb] = by_b.get(lb, 0) + 1
    histogram = stats.setdefault("runner_up_histogram", {})
    for la, lb, iou, _n in matched:
        # A candidate other than the pair itself shares at most ONE of the two labels,
        # so it is counted once; the pair itself is counted in both indices.
        self_counted = 2 if iou >= RUNNER_UP_IOU else 0
        others = by_a.get(la, 0) + by_b.get(lb, 0) - self_counted
        histogram[others] = histogram.get(others, 0) + 1
    stats["n_matched"] = stats.get("n_matched", 0) + len(matched)


class UnionFind:
    """Disjoint-set over arbitrary hashable, order-comparable nodes.

    The root of a merged component is always the SMALLER of the two roots, so the
    component representative depends only on the set of nodes and not on the order the
    unions arrived in: the global id assignment below hashes the root, and a run-to-run
    varying root would reshuffle every id.
    """

    def __init__(self) -> None:
        self._parent: dict = {}

    def find(self, node):
        """Return ``node``'s component root, adding it as its own root if unseen."""
        parent = self._parent
        root = parent.setdefault(node, node)
        while root != parent[root]:
            root = parent[root]
        # Path compression: point every node on the way at the root.
        while node != root:
            parent[node], node = root, parent[node]
        return root

    def union(self, a, b) -> bool:
        """Merge the components of ``a`` and ``b``; ``False`` if already together."""
        root_a = self.find(a)
        root_b = self.find(b)
        if root_a == root_b:
            return False
        root, child = (root_a, root_b) if root_a < root_b else (root_b, root_a)
        self._parent[child] = root
        return True

    def roots(self) -> set:
        return {self.find(node) for node in list(self._parent)}


@dataclass
class Mosaic:
    """The sub-tiles of one or more ``split`` runs, with their source km tiles.

    ``sources`` is sorted by ``key`` and unique; a sub-tile's ``tile_map`` maps its own
    manifest's source order onto this list, so ``ident_keys`` produces mosaic-wide keys.
    """

    size_m: int
    buffer_m: float
    sources: list[dict] = field(default_factory=list)
    subtiles: list[dict] = field(default_factory=list)

    def source_index(self, key: str) -> int:
        for i, source in enumerate(self.sources):
            if source["key"] == key:
                return i
        raise KeyError(key)


def load_mosaic(manifest_paths) -> Mosaic:
    """Merge ``split_manifest.json`` files into one :class:`Mosaic`.

    All manifests must agree on ``size_m`` and ``buffer_m`` (sub-tiles cut on different
    grids cannot be matched by adjacency). A stem listed twice, or one source key given
    two different point counts, is rejected: both would silently corrupt the per-km
    output (the same core written twice, or an output array of the wrong length).
    """
    manifest_paths = [Path(p) for p in manifest_paths]
    if not manifest_paths:
        raise ValueError("load_mosaic: no manifests given")

    size_m: int | None = None
    buffer_m: float | None = None
    sources: dict[str, dict] = {}
    raw: list[tuple[Path, dict]] = []

    for path in manifest_paths:
        manifest = json.loads(path.read_text())
        if size_m is None:
            size_m, buffer_m = int(manifest["size_m"]), float(manifest["buffer_m"])
        elif (int(manifest["size_m"]), float(manifest["buffer_m"])) != (size_m, buffer_m):
            raise ValueError(
                f"load_mosaic: {path} was split with size_m={manifest['size_m']} "
                f"buffer_m={manifest['buffer_m']} but a previous manifest used "
                f"size_m={size_m} buffer_m={buffer_m}; sub-tiles on different grids "
                "cannot be stitched together"
            )
        for source in manifest["sources"]:
            key = source["key"]
            known = sources.get(key)
            if known is None:
                sources[key] = {
                    "key": key,
                    "path": str(source["path"]),
                    "n_points": int(source["n_points"]),
                }
            elif known["n_points"] != int(source["n_points"]):
                raise ValueError(
                    f"load_mosaic: source {key} has {known['n_points']} points in one "
                    f"manifest and {source['n_points']} in {path}; the ident indices of "
                    "the two splits refer to different files"
                )
        raw.append((path, manifest))

    ordered = [sources[key] for key in sorted(sources)]
    global_index = {source["key"]: i for i, source in enumerate(ordered)}

    subtiles: list[dict] = []
    seen_stems: dict[str, Path] = {}
    for path, manifest in raw:
        keys = [source["key"] for source in manifest["sources"]]
        tile_map = np.array([global_index[key] for key in keys], dtype=np.int64)
        for sub in manifest["subtiles"]:
            stem = sub["stem"]
            if stem in seen_stems:
                raise ValueError(
                    f"load_mosaic: sub-tile {stem} is listed by both {seen_stems[stem]} "
                    f"and {path}; its core would be written twice"
                )
            seen_stems[stem] = path
            subtiles.append({
                "stem": stem,
                "origin": (int(sub["origin"][0]), int(sub["origin"][1])),
                "source": keys[int(sub["source"])],
                "n_points": int(sub["n_points"]),
                "n_core": int(sub["n_core"]),
                "manifest_dir": path.parent,
                "tile_map": tile_map,
            })

    subtiles.sort(key=lambda sub: sub["stem"])
    return Mosaic(size_m=int(size_m), buffer_m=float(buffer_m), sources=ordered,
                  subtiles=subtiles)


def adjacent_pairs(mosaic: Mosaic) -> list[tuple[str, str]]:
    """Unordered stem pairs of sub-tiles whose cores touch (8-neighbourhood).

    Two cores are adjacent when their origins differ by exactly one ``size_m`` step in x
    and/or y, i.e. they share an edge or a corner. Looked up through an origin index
    rather than by comparing every pair, so a mosaic of many km tiles stays linear.
    """
    step = mosaic.size_m
    by_origin: dict[tuple[int, int], list[str]] = {}
    for sub in mosaic.subtiles:
        by_origin.setdefault(sub["origin"], []).append(sub["stem"])

    pairs: set[tuple[str, str]] = set()
    for (x0, y0), stems in by_origin.items():
        for dx in (-step, 0, step):
            for dy in (-step, 0, step):
                if dx == 0 and dy == 0:
                    continue
                for other in by_origin.get((x0 + dx, y0 + dy), ()):
                    for stem in stems:
                        if stem != other:
                            pairs.add((stem, other) if stem < other else (other, stem))
    return sorted(pairs)


@dataclass
class _Subtile:
    """One sub-tile's decoded result, aligned with its ``_ident.npy`` sidecar."""

    stem: str
    source: str
    keys: np.ndarray
    labels: np.ndarray
    semantic: np.ndarray
    score: np.ndarray
    core: np.ndarray


class _SubtileCache:
    """Read a sub-tile's result LAS + ident sidecar, keeping the last few in memory."""

    def __init__(self, mosaic: Mosaic, result_paths: dict[str, Path], maxsize: int = _CACHE_SIZE):
        self._mosaic = mosaic
        self._result_paths = result_paths
        self._by_stem = {sub["stem"]: sub for sub in mosaic.subtiles}
        self._cache: "OrderedDict[str, _Subtile]" = OrderedDict()
        self._maxsize = maxsize

    def __call__(self, stem: str) -> _Subtile:
        cached = self._cache.get(stem)
        if cached is not None:
            self._cache.move_to_end(stem)
            return cached
        loaded = self._load(stem)
        self._cache[stem] = loaded
        while len(self._cache) > self._maxsize:
            self._cache.popitem(last=False)
        return loaded

    def _load(self, stem: str) -> _Subtile:
        sub = self._by_stem[stem]
        las_path = self._result_paths[stem]
        ident_path = Path(sub["manifest_dir"]) / f"{stem}_ident.npy"
        if not ident_path.is_file():
            raise FileNotFoundError(
                f"stitch: no point identity sidecar {ident_path} for sub-tile {stem}; "
                "it is written by ff3d_geo.split next to split_manifest.json"
            )
        ident = np.load(ident_path)
        las = laspy.read(str(las_path))
        n_points = int(las.header.point_count)
        if n_points != sub["n_points"]:
            raise ValueError(
                f"stitch: {las_path} has {n_points} points but the manifest records "
                f"{sub['n_points']} for {stem}; the result does not belong to this split"
            )
        if len(ident) != n_points:
            raise ValueError(
                f"stitch: {ident_path} has {len(ident)} entries but {las_path} has "
                f"{n_points} points; point order would not match"
            )

        x = np.asarray(las.x, dtype=np.float64)
        y = np.asarray(las.y, dtype=np.float64)
        x0, y0 = sub["origin"]
        self._check_point_order(stem, sub, las_path, x, y)
        size = self._mosaic.size_m
        core = (x >= x0) & (x < x0 + size) & (y >= y0) & (y < y0 + size)
        return _Subtile(
            stem=stem,
            source=sub["source"],
            keys=ident_keys(ident, sub["tile_map"]),
            labels=np.asarray(las.treeID, dtype=np.int32),
            semantic=np.asarray(las.semantic, dtype=np.uint8),
            score=np.asarray(las.score, dtype=np.float32),
            core=core,
        )

    def _check_point_order(self, stem: str, sub: dict, las_path: Path,
                           x: np.ndarray, y: np.ndarray) -> None:
        """Verify the result LAS is in the SAME point order as its split sub-tile.

        The ident sidecar maps result point ``i`` to source point ``ident[i]``, so every
        stitched id is wrong -- silently -- if anything between ``split`` and
        ``results_to_las`` ever reorders points. The count checks in :meth:`_load` and
        the ``own.sum() == n_core`` guard in :func:`stitch` are both order-blind (a
        permutation that keeps every point inside its own halo box passes them), so
        compare a strided sample of coordinates against the split sub-tile LAS, which
        the km-tile workflow keeps on disk next to ``split_manifest.json`` until the
        stitch. A sub-tile LAS that has been cleaned up already is skipped rather than
        refused: the check is an integrity guard, not a new input requirement.
        """
        split_las = Path(sub["manifest_dir"]) / f"{stem}.las"
        if not split_las.is_file() or len(x) == 0:
            return
        x0, y0 = sub["origin"]
        n = len(x)
        indices = np.unique(np.append(np.arange(0, n, _ORDER_STRIDE), n - 1))
        with laspy.open(str(split_las)) as reader:
            for i in indices:
                reader.seek(int(i))
                point = reader.read_points(1)
                sx = float(np.asarray(point.x)[0]) + x0
                sy = float(np.asarray(point.y)[0]) + y0
                if abs(sx - x[i]) > _ORDER_TOL_M or abs(sy - y[i]) > _ORDER_TOL_M:
                    raise ValueError(
                        f"stitch: point {i} of {las_path} is at ({x[i]:.3f}, {y[i]:.3f}) "
                        f"but point {i} of {split_las} is at ({sx:.3f}, {sy:.3f}); the "
                        f"result of {stem} is not in its split sub-tile's point order, so "
                        f"{stem}_ident.npy would map every label to the wrong source point"
                    )


def _find_results(mosaic: Mosaic, results_dirs) -> dict[str, Path]:
    """``stem -> <dir>/<stem>.las``, first directory that has it."""
    dirs = [Path(d) for d in results_dirs]
    found: dict[str, Path] = {}
    for sub in mosaic.subtiles:
        stem = sub["stem"]
        for directory in dirs:
            candidate = directory / f"{stem}.las"
            if candidate.is_file():
                found[stem] = candidate
                break
        else:
            raise FileNotFoundError(
                f"stitch: no result LAS for sub-tile {stem} in "
                f"{[str(d) for d in dirs]} (expected {stem}.las)"
            )
    return found


def _global_ids(uf: UnionFind, nodes: list[tuple[str, int]]) -> dict[tuple[str, int], int]:
    """Assign dense ids ``0..N-1`` to the components of ``nodes``, in shuffled order.

    Note this is a Weyl (low-discrepancy) permutation rather than a hash: consecutive
    ids come from ranks a fixed few steps apart. That is what the brief asked for and it
    meets the goal -- a sub-tile's ~300 trees land in ~300 distinct LUT slots and each
    slot mixes several sub-tiles -- so please do not "fix" it into an avalanche mixer
    without re-checking the viewer's colouring; the ids are written into published LAS
    files and changing the permutation renumbers every tree.

    The order is the multiplicative hash of the component root's RANK among the sorted
    roots, not of anything spatial: ids that follow each other must not belong to
    neighbouring trees, because the Potree viewer buckets the id range into 8192 colour
    slots and a spatially ordered id scheme then gives each sub-tile its own tint.
    """
    roots = sorted({uf.find(node) for node in nodes})
    order = sorted(range(len(roots)), key=lambda rank: (rank * _MIX64) & ((1 << 63) - 1))
    gid_of_root = {roots[rank]: gid for gid, rank in enumerate(order)}
    return {node: gid_of_root[uf.find(node)] for node in nodes}


def _lookup_tables(gids: dict[tuple[str, int], int]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Group the id map by sub-tile into ``(sorted local labels, global ids)`` arrays.

    Built once for the whole mosaic: scanning the id map per sub-tile instead would be
    quadratic in a mosaic of many km tiles (sub-tiles x nodes).
    """
    by_stem: dict[str, list[tuple[int, int]]] = {}
    for (stem, label), gid in gids.items():
        by_stem.setdefault(stem, []).append((label, gid))
    tables: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for stem, entries in by_stem.items():
        entries.sort()
        tables[stem] = (
            np.array([label for label, _ in entries], dtype=np.int64),
            np.array([gid for _, gid in entries], dtype=np.int32),
        )
    return tables


def _label_lookup(tables: dict[str, tuple[np.ndarray, np.ndarray]], stem: str,
                  labels: np.ndarray) -> np.ndarray:
    """Map a sub-tile's local labels onto global ids, vectorised (-1 stays -1)."""
    local, mapped = tables.get(stem, (np.empty(0, np.int64), np.empty(0, np.int32)))
    if local.size == 0:
        return np.full(len(labels), -1, dtype=np.int32)
    labels = np.asarray(labels, dtype=np.int64)
    position = np.searchsorted(local, labels)
    position = np.clip(position, 0, local.size - 1)
    hit = (labels >= 0) & (local[position] == labels)
    return np.where(hit, mapped[position], -1).astype(np.int32)


def _write_source_las(source: dict, out_las: Path, tree_id: np.ndarray,
                      semantic: np.ndarray, score: np.ndarray,
                      epsg: int = EPSG) -> None:
    """Rewrite a source km tile with the stitched labels, in its own point order."""
    source_path = Path(source["path"])
    out_las.parent.mkdir(parents=True, exist_ok=True)
    tmp_las = out_las.with_name(out_las.name + ".tmp")
    try:
        with laspy.open(str(source_path)) as reader:
            header = result_point_header(epsg, [0.001, 0.001, 0.001],
                                         np.floor(reader.header.mins))
            source_dims = {d.name for d in reader.header.point_format.dimensions}
            copy_dims = [d for d in _COPY_DIMS if d in source_dims]
            start = 0
            with laspy.open(str(tmp_las), mode="w", header=header) as writer:
                for points in reader.chunk_iterator(_CHUNK_POINTS):
                    n_chunk = len(points)
                    stop = start + n_chunk
                    record = laspy.ScaleAwarePointRecord.zeros(
                        n_chunk,
                        point_format=header.point_format,
                        scales=header.scales,
                        offsets=header.offsets,
                    )
                    record.x = np.asarray(points.x)
                    record.y = np.asarray(points.y)
                    record.z = np.asarray(points.z)
                    record.classification = np.asarray(points.classification)
                    for dim in copy_dims:
                        setattr(record, dim, np.asarray(getattr(points, dim)))
                    record.treeID = tree_id[start:stop]
                    record.semantic = semantic[start:stop]
                    record.score = score[start:stop]
                    writer.write_points(record)
                    start = stop
            if start != len(tree_id):
                raise ValueError(
                    f"stitch: {source_path} yielded {start} points but its manifest "
                    f"entry records {len(tree_id)}"
                )
        tmp_las.replace(out_las)
    except BaseException:
        tmp_las.unlink(missing_ok=True)
        raise


def _hull_vertices(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """The 2D convex hull of a tree's points, as the few vertices that define it.

    Keeping the hull instead of the points makes the per-tile parts mergeable: the hull
    of a union is the hull of the union of the hulls, so a tree split over two km tiles
    gets its true crown area without either tile holding the other's points.
    """
    points = np.column_stack([np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)])
    if len(points) <= 3:
        return points
    hull = MultiPoint(points).convex_hull
    if hull.geom_type == "Polygon":
        return np.asarray(hull.exterior.coords)[:-1]
    if hull.geom_type == "LineString":
        return np.asarray(hull.coords)
    return points[:1]


def _hull_area(points: np.ndarray) -> float:
    """Convex hull area of ``(n, 2)`` points; 0.0 when they are collinear or fewer than 3.

    Same rule as :func:`ff3d_geo.trees.trees_to_gpkg`: a degenerate hull is a LineString
    or a Point, not a Polygon, and scores no crown area.
    """
    if len(points) < 3:
        return 0.0
    hull = MultiPoint(points).convex_hull
    return float(hull.area) if hull.geom_type == "Polygon" else 0.0


def _tile_tree_parts(las_path: Path, ground_grid_m: float = 1.0):
    """Per-tree partial aggregates for one km tile, plus its ground grid and CRS.

    The aggregates are chosen so that two tiles' parts for the same global id can be
    merged EXACTLY (see :func:`_mosaic_tree_rows`): counts and score sums add, ``top_z``
    and ``min_z`` are extrema, crown hulls compose, and the "lowest metre" band is kept
    as points because the stem position is a median over it. The band is taken at this
    tile's own ``min_z``, which is always at or above the global one, so it is a superset
    of the points the merged band needs.
    """
    las = laspy.read(str(las_path))
    x = np.asarray(las.x, dtype=np.float64)
    y = np.asarray(las.y, dtype=np.float64)
    z = np.asarray(las.z, dtype=np.float64)
    tree_id = np.asarray(las.treeID, dtype=np.int64)
    semantic = np.asarray(las.semantic, dtype=np.int64)
    classification = np.asarray(las.classification, dtype=np.int64)
    score = np.asarray(las.score, dtype=np.float64)

    ground, extent = ground_surface(x, y, z, semantic, classification, ground_grid_m)

    # One sort instead of a `tree_id == tid` pass per tree (see trees.trees_to_gpkg).
    order = np.argsort(tree_id, kind="stable")
    sorted_ids = tree_id[order]
    first_real = int(np.searchsorted(sorted_ids, 0, side="left"))
    ids, starts = np.unique(sorted_ids[first_real:], return_index=True)
    starts = starts + first_real
    stops = np.append(starts[1:], sorted_ids.size)

    parts: dict[int, dict] = {}
    for tid, start, stop in zip(ids.tolist(), starts.tolist(), stops.tolist()):
        member = order[start:stop]
        tx, ty, tz = x[member], y[member], z[member]
        low = tz <= tz.min() + 1.0
        parts[int(tid)] = {
            "n": int(member.size),
            "score_sum": float(score[member].sum()),
            "top_z": float(tz.max()),
            "min_z": float(tz.min()),
            "low": np.column_stack([tx[low], ty[low], tz[low]]),
            "hull": _hull_vertices(tx, ty),
        }
    return parts, ground, extent, las.header.parse_crs()


def _mosaic_tree_rows(parts_by_tile: dict, ground_by_tile: dict, extent_by_tile: dict):
    """Assign every global tree id to ONE owner tile and build its row from all its points.

    A tree unified across a km border has points in two source tiles. Running
    ``trees_to_gpkg`` per tile would then give it a partial row in each: mosaic-wide
    counts would double it and every attribute (height, crown area, n_points) would
    describe a fragment. Here each id goes to the tile holding the majority of its
    points -- ties to the tile holding its highest point, then to the first tile name, so
    the choice is deterministic -- and its row is computed over ALL of its points.

    The height is taken against the ground grid of the tile whose extent CONTAINS the
    stem, which is not always the owner: the owner holds most of the tree's points, but
    the median of its lowest metre can sit a few metres across the km border. Looking
    that up in the owner's grid would go through ``GridExtent.index``'s silent clamping
    and take the ground of the tile's edge cell instead, a small but systematic height
    bias on exactly the cross-border trees. The owner is the fallback when no tile's
    extent contains the stem.

    Returns ``({tile stem: [row, ...]}, n_cross_km_trees)``.
    """
    tiles_of: dict[int, list[str]] = {}
    for stem, parts in parts_by_tile.items():
        for gid in parts:
            tiles_of.setdefault(gid, []).append(stem)

    rows_by_tile: dict[str, list[dict]] = {stem: [] for stem in parts_by_tile}
    n_cross_km_trees = 0
    for gid in sorted(tiles_of):
        stems = sorted(tiles_of[gid])
        if len(stems) > 1:
            n_cross_km_trees += 1
        pieces = [parts_by_tile[stem][gid] for stem in stems]
        owner = min(stems, key=lambda s: (-parts_by_tile[s][gid]["n"],
                                          -parts_by_tile[s][gid]["top_z"], s))

        n_points = sum(p["n"] for p in pieces)
        top_z = max(p["top_z"] for p in pieces)
        min_z = min(p["min_z"] for p in pieces)
        low = np.vstack([p["low"] for p in pieces])
        low = low[low[:, 2] <= min_z + 1.0]
        stem_x = float(np.median(low[:, 0]))
        stem_y = float(np.median(low[:, 1]))
        ground_stem = owner
        if not bool(extent_by_tile[owner].contains(stem_x, stem_y)):
            for candidate in stems:
                if bool(extent_by_tile[candidate].contains(stem_x, stem_y)):
                    ground_stem = candidate
                    break
        ix, iy = extent_by_tile[ground_stem].index(stem_x, stem_y)
        ground_z = float(ground_by_tile[ground_stem][iy, ix])
        rows_by_tile[owner].append({
            "tree_id": int(gid),
            "x": stem_x,
            "y": stem_y,
            "top_z": top_z,
            "height": top_z - ground_z,
            "crown_area_m2": _hull_area(np.vstack([p["hull"] for p in pieces])),
            "n_points": n_points,
            "mean_score": sum(p["score_sum"] for p in pieces) / n_points,
        })
    return rows_by_tile, n_cross_km_trees


def _write_tree_table(rows: list[dict], crs, gpkg_path: Path) -> int:
    """Write ``rows`` as layer ``trees``, with the columns ``trees_to_gpkg`` produces."""
    frame = {column: [row[column] for row in rows] for column in TREE_COLUMNS}
    geoms = [Point(row["x"], row["y"]) for row in rows]
    gdf = gpd.GeoDataFrame(frame, geometry=gpd.GeoSeries(geoms, crs=crs), crs=crs)
    if not rows:
        gdf = gdf.astype({"tree_id": "int64", "x": "float64", "y": "float64",
                          "top_z": "float64", "height": "float64",
                          "crown_area_m2": "float64", "n_points": "int64",
                          "mean_score": "float64"})
    gpkg_path = Path(gpkg_path)
    gpkg_path.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(str(gpkg_path), driver="GPKG", layer="trees")
    return len(rows)


def stitch(manifests, results_dirs, out_dir, iou_threshold: float = 0.5,
           min_shared: int = 20, runtime_s: float | None = None,
           epsg: int = EPSG) -> dict:
    """Unify sub-tile instances over their halo overlaps and rewrite the km tiles.

    ``manifests`` are ``split_manifest.json`` paths, ``results_dirs`` the directories
    holding ``<stem>.las`` per sub-tile (searched in order; a missing one raises
    ``FileNotFoundError`` naming the stem). Writes per source km tile ``<out>/<S>.las``
    (all of its points, in its own order, with the extra dims of
    :func:`ff3d_geo.convert.result_point_header` and the CRS ``epsg``, which a caller
    should set to whatever ``las_to_ply`` used), ``<S>_trees.gpkg`` and
    ``<S>_report.json``/``.md``, plus ``<out>/stitch_ids.npy`` (the
    ``(gid, stem, local)`` id map) and ``<out>/stitch.json``.

    Every tree appears in exactly ONE ``<S>_trees.gpkg`` -- the tile holding most of its
    points -- with attributes measured over all of its points across tiles, so the tables
    of a mosaic concatenate to exactly ``n_trees`` rows and the per-tile reports count
    each tree once. ``stitch.json``'s ``n_cross_km_trees`` says how many trees have
    points in more than one km tile.

    Only the sources that OWN at least one sub-tile are written. ``Mosaic.sources`` also
    holds the km tiles that merely supplied halo points to somebody else's split
    (``split_las(..., neighbours=[...])`` lists them); no sub-tile core covers those, so
    writing them would produce a full-size km tile of pure nodata and a report claiming
    zero trees. They are named in ``stitch.json``'s ``neighbour_only_sources`` instead.

    A tree on a km-tile border ends up with the SAME global id in both km tiles, which
    is the whole point of the exercise. A point keeps the nodata values
    ``-1 / 255 / -1.0`` when no sub-tile core claimed it -- because its sub-tile was
    dropped by ``split``'s ``min_points``, or because the run simply has no result for
    that part of the km tile.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_paths = [Path(m) for m in manifests]

    mosaic = load_mosaic(manifest_paths)
    result_paths = _find_results(mosaic, results_dirs)
    load = _SubtileCache(mosaic, result_paths)
    source_of = {sub["stem"]: sub["source"] for sub in mosaic.subtiles}

    uf = UnionFind()
    match_stats: dict = {}
    n_pairs_tested = 0
    n_unified = 0
    n_cross_km = 0
    for stem_a, stem_b in adjacent_pairs(mosaic):
        a = load(stem_a)
        b = load(stem_b)
        n_pairs_tested += 1
        cross = source_of[stem_a] != source_of[stem_b]
        for la, lb, _iou, _n in match_instances(
            a.keys, a.labels, b.keys, b.labels,
            iou_threshold=iou_threshold, min_shared=min_shared, stats=match_stats,
        ):
            if uf.union((stem_a, la), (stem_b, lb)):
                n_unified += 1
                if cross:
                    n_cross_km += 1

    # Only instances that own CORE points become trees: a halo-only instance is the
    # neighbouring sub-tile's business and would otherwise be counted twice.
    nodes: list[tuple[str, int]] = []
    for sub in mosaic.subtiles:
        data = load(sub["stem"])
        core_labels = data.labels[data.core]
        for label in np.unique(core_labels[core_labels >= 0]).tolist():
            nodes.append((sub["stem"], int(label)))
    gids = _global_ids(uf, nodes)
    tables = _lookup_tables(gids)
    n_trees = len(set(gids.values()))

    # The stem field is sized to the longest stem actually present: a fixed "U64" would
    # silently TRUNCATE a longer one (a custom split --prefix), and a truncated stem
    # makes the id map unjoinable with the sub-tile results.
    stem_width = max((len(stem) for stem, _ in nodes), default=1)
    id_map = np.empty(len(nodes),
                      dtype=[("gid", "u4"), ("stem", f"U{stem_width}"), ("local", "i4")])
    for i, (stem, label) in enumerate(sorted(nodes)):
        id_map[i] = (gids[(stem, label)], stem, label)
    np.save(out_dir / "stitch_ids.npy", id_map)

    owning_sources = {sub["source"] for sub in mosaic.subtiles}
    neighbour_only = [s["key"] for s in mosaic.sources if s["key"] not in owning_sources]

    tiles: dict[str, dict] = {}
    for source in mosaic.sources:
        if source["key"] not in owning_sources:
            continue
        source_stem = Path(source["path"]).stem
        n_points = int(source["n_points"])
        tree_id = np.full(n_points, NODATA_TREE_ID, dtype=np.int32)
        semantic = np.full(n_points, NODATA_SEMANTIC, dtype=np.uint8)
        score = np.full(n_points, NODATA_SCORE, dtype=np.float32)
        global_tile = np.int64(mosaic.source_index(source["key"]))

        for sub in mosaic.subtiles:
            if sub["source"] != source["key"]:
                continue
            data = load(sub["stem"])
            # A core point always comes from this source, but the halo does not, so the
            # ownership mask is core AND "from this km tile" before indexing by index.
            own = data.core & ((data.keys >> np.int64(32)) == global_tile)
            # The core box is re-derived here from the RESULT LAS coordinates, which
            # went through split's local shift, the model's float32 PLY and
            # results_to_las; split computed it on the source coordinates. They agree
            # today, but a point that flipped side would silently be written twice (last
            # writer wins below) or not at all, so the count is checked against the
            # manifest -- cheaper and sharper than any coordinate tolerance.
            if int(own.sum()) != sub["n_core"]:
                raise ValueError(
                    f"stitch: sub-tile {sub['stem']} owns {int(own.sum())} points of "
                    f"{source['key']} but split recorded n_core={sub['n_core']}; core "
                    "membership no longer round-trips through the result LAS "
                    "coordinates, so points would be written twice or not at all"
                )
            index = (data.keys[own] & _KEY_INDEX_MASK).astype(np.int64)
            tree_id[index] = _label_lookup(tables, sub["stem"], data.labels[own])
            semantic[index] = data.semantic[own]
            score[index] = data.score[own]

        out_las = out_dir / f"{source_stem}.las"
        _write_source_las(source, out_las, tree_id, semantic, score, epsg=epsg)
        tiles[source_stem] = {
            "las": str(out_las),
            "gpkg": str(out_dir / f"{source_stem}_trees.gpkg"),
            "n_points": n_points,
        }

    # The tree table is built ONCE over the whole mosaic and then split by owner tile:
    # a tree unified across a km border has points in two km tiles and must be counted,
    # and measured, once. Only after every km LAS is on disk, since the parts are read
    # back from them.
    parts_by_tile, ground_by_tile, extent_by_tile, crs_by_tile = {}, {}, {}, {}
    for source_stem, tile in tiles.items():
        (parts_by_tile[source_stem], ground_by_tile[source_stem],
         extent_by_tile[source_stem], crs_by_tile[source_stem]) = _tile_tree_parts(tile["las"])
    rows_by_tile, n_cross_km_trees = _mosaic_tree_rows(
        parts_by_tile, ground_by_tile, extent_by_tile)

    for source_stem, tile in tiles.items():
        out_las = Path(tile["las"])
        out_gpkg = Path(tile["gpkg"])
        tile["n_trees_in_tile"] = _write_tree_table(
            rows_by_tile[source_stem], crs_by_tile[source_stem], out_gpkg)
        report = build_report(out_las, out_gpkg, runtime_s=runtime_s)
        # build_report counts the distinct treeIDs in the LAS, which counts a km-border
        # tree in BOTH of the tiles it reaches into. The tree table is the authority
        # here, so the report reports the trees this tile OWNS (the mosaic's per-tile
        # counts then add up to n_trees) and keeps the LAS figure alongside it.
        report["n_trees_in_las"] = int(report["n_trees"])
        report["n_trees"] = int(tile["n_trees_in_tile"])
        report["recommendation"] = recommend(report)
        write_report(report, out_dir / f"{source_stem}_report.json",
                     out_dir / f"{source_stem}_report.md")

    info = {
        "n_trees": int(n_trees),
        "n_pairs_tested": int(n_pairs_tested),
        "n_unified": int(n_unified),
        "n_cross_km": int(n_cross_km),
        "n_cross_km_trees": int(n_cross_km_trees),
        "tiles": tiles,
    }
    (out_dir / "stitch.json").write_text(json.dumps({
        **{k: v for k, v in info.items() if k != "tiles"},
        "n_subtiles": len(mosaic.subtiles),
        "n_sources": len(mosaic.sources),
        "neighbour_only_sources": neighbour_only,
        "size_m": mosaic.size_m,
        "buffer_m": mosaic.buffer_m,
        "epsg": int(epsg),
        "iou_threshold": float(iou_threshold),
        "min_shared": int(min_shared),
        # How often the greedy one-to-one rule left a second plausible partner on the
        # table: {number of other candidates above RUNNER_UP_IOU sharing a label with
        # the matched pair: number of matched pairs}.
        "runner_up_iou": RUNNER_UP_IOU,
        "runner_up_histogram": {str(k): v for k, v in
                                sorted(match_stats.get("runner_up_histogram", {}).items())},
        "manifests": [str(Path(m).resolve()) for m in manifest_paths],
        "tiles": tiles,
    }, indent=2))
    return info
