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
4. each source km tile is rewritten in ITS OWN point order with the labels of the
   sub-tile whose CORE owns each point, so no point is written twice and the km tiles
   remain a partition. A core point whose sub-tile was dropped by ``split``'s
   ``min_points`` keeps the nodata values ``treeID = -1 / semantic = 255 / score = -1``
   (the older ``ff3d_geo.merge`` path left such points out of the merged LAS entirely).

Everything that touches point arrays is vectorised: a km tile is 25 M points over 100
sub-tiles with 20 m halos (~1.96x its core each), so a per-instance Python pass over
points would cost hours.
"""

from __future__ import annotations

import json
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path

import laspy
import numpy as np

from ff3d_geo.convert import result_point_header
from ff3d_geo.report import build_report, write_report
from ff3d_geo.split import IDENT_DTYPE
from ff3d_geo.trees import trees_to_gpkg

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

#: Knuth's multiplicative hash constant, used to shuffle component ids (see step 3).
_MIX64 = 0x9E3779B97F4A7C15

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

    candidates: list[tuple[int, int, float, int]] = []
    for la, lb, n in zip(la_all.tolist(), lb_all.tolist(), n_ab.tolist()):
        if n < min_shared:
            continue
        union = size_a[la] + size_b[lb] - n
        iou = n / union if union > 0 else 0.0
        if iou >= iou_threshold:
            candidates.append((int(la), int(lb), float(iou), int(n)))

    candidates.sort(key=lambda p: (-p[2], p[0], p[1]))
    used_a: set[int] = set()
    used_b: set[int] = set()
    matched: list[tuple[int, int, float, int]] = []
    for la, lb, iou, n in candidates:
        if la in used_a or lb in used_b:
            continue
        used_a.add(la)
        used_b.add(lb)
        matched.append((la, lb, iou, n))
    return matched


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
                      semantic: np.ndarray, score: np.ndarray) -> None:
    """Rewrite a source km tile with the stitched labels, in its own point order."""
    source_path = Path(source["path"])
    out_las.parent.mkdir(parents=True, exist_ok=True)
    tmp_las = out_las.with_name(out_las.name + ".tmp")
    try:
        with laspy.open(str(source_path)) as reader:
            header = result_point_header(EPSG, [0.001, 0.001, 0.001],
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


def stitch(manifests, results_dirs, out_dir, iou_threshold: float = 0.5,
           min_shared: int = 20, runtime_s: float | None = None) -> dict:
    """Unify sub-tile instances over their halo overlaps and rewrite the km tiles.

    ``manifests`` are ``split_manifest.json`` paths, ``results_dirs`` the directories
    holding ``<stem>.las`` per sub-tile (searched in order; a missing one raises
    ``FileNotFoundError`` naming the stem). Writes per source km tile ``<out>/<S>.las``
    (all of its points, in its own order, with the extra dims of
    :func:`ff3d_geo.convert.result_point_header`), ``<S>_trees.gpkg`` and
    ``<S>_report.json``/``.md``, plus ``<out>/stitch_ids.npy`` (the
    ``(gid, stem, local)`` id map) and ``<out>/stitch.json``.

    A tree on a km-tile border ends up with the SAME global id in both km tiles, which
    is the whole point of the exercise; a core point no sub-tile claimed (its sub-tile
    was dropped by ``split``'s ``min_points``) keeps ``-1 / 255 / -1.0``.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_paths = [Path(m) for m in manifests]

    mosaic = load_mosaic(manifest_paths)
    result_paths = _find_results(mosaic, results_dirs)
    load = _SubtileCache(mosaic, result_paths)
    source_of = {sub["stem"]: sub["source"] for sub in mosaic.subtiles}

    uf = UnionFind()
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
            iou_threshold=iou_threshold, min_shared=min_shared,
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

    id_map = np.empty(len(nodes), dtype=[("gid", "u4"), ("stem", "U64"), ("local", "i4")])
    for i, (stem, label) in enumerate(sorted(nodes)):
        id_map[i] = (gids[(stem, label)], stem, label)
    np.save(out_dir / "stitch_ids.npy", id_map)

    tiles: dict[str, dict] = {}
    for source in mosaic.sources:
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
            index = (data.keys[own] & _KEY_INDEX_MASK).astype(np.int64)
            tree_id[index] = _label_lookup(tables, sub["stem"], data.labels[own])
            semantic[index] = data.semantic[own]
            score[index] = data.score[own]

        out_las = out_dir / f"{source_stem}.las"
        _write_source_las(source, out_las, tree_id, semantic, score)
        out_gpkg = out_dir / f"{source_stem}_trees.gpkg"
        n_trees_in_tile = trees_to_gpkg(out_las, out_gpkg)
        report = build_report(out_las, out_gpkg, runtime_s=runtime_s)
        write_report(report, out_dir / f"{source_stem}_report.json",
                     out_dir / f"{source_stem}_report.md")
        tiles[source_stem] = {
            "las": str(out_las),
            "gpkg": str(out_gpkg),
            "n_points": n_points,
            "n_trees_in_tile": int(n_trees_in_tile),
        }

    info = {
        "n_trees": int(n_trees),
        "n_pairs_tested": int(n_pairs_tested),
        "n_unified": int(n_unified),
        "n_cross_km": int(n_cross_km),
        "tiles": tiles,
    }
    (out_dir / "stitch.json").write_text(json.dumps({
        **{k: v for k, v in info.items() if k != "tiles"},
        "n_subtiles": len(mosaic.subtiles),
        "n_sources": len(mosaic.sources),
        "size_m": mosaic.size_m,
        "buffer_m": mosaic.buffer_m,
        "iou_threshold": float(iou_threshold),
        "min_shared": int(min_shared),
        "manifests": [str(Path(m).resolve()) for m in manifest_paths],
        "tiles": tiles,
    }, indent=2))
    return info
