"""APEX-T M3: experimental topology/curved-path screening.

Pixels are {'x','y','value','extra'} dictionaries.  The third column is treated
only as a detector SIGNAL, *not* as calibrated energy.  The `extra`/time column
is never consulted.

M is a heuristic candidate-ranking score [0, 1], NOT a pile-up probability.
The geometry searches for multiple continuous pixel skeleton paths; it does NOT
split a mildly curved path into multiple linear RANSAC fits.
"""
from __future__ import annotations

from collections import deque
from itertools import combinations
import math
import numpy as np

VERSION = "M3.0-curves-signal"
NEIGHBORS = [(dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if dx or dy]


def _thin_zhang_suen(mask: np.ndarray) -> np.ndarray:
    """Thin a binary image to a 1px skeleton, using only NumPy."""
    a = np.pad(mask.astype(bool), 2)
    # Avoid runaway on huge pathological clusters; iterative changes settle fast.
    for _ in range(max(10, min(140, max(a.shape) * 2))):
        changed = False
        for stage in (0, 1):
            p = a[1:-1, 1:-1]
            p2, p3 = a[:-2, 1:-1], a[:-2, 2:]
            p4, p5 = a[1:-1, 2:], a[2:, 2:]
            p6, p7 = a[2:, 1:-1], a[2:, :-2]
            p8, p9 = a[1:-1, :-2], a[:-2, :-2]
            around = (p2, p3, p4, p5, p6, p7, p8, p9)
            degree = sum(z.astype(np.uint8) for z in around)
            transitions = sum((~around[i]) & around[(i+1) % 8]
                              for i in range(8))
            remove = p & (degree >= 2) & (degree <= 6) & (transitions == 1)
            if stage == 0:
                remove &= ~(p2 & p4 & p6) & ~(p4 & p6 & p8)
            else:
                remove &= ~(p2 & p4 & p8) & ~(p2 & p6 & p8)
            if remove.any():
                p[remove] = False
                changed = True
        if not changed:
            break
    return a[2:-2, 2:-2]


def _make_graph(skeleton: np.ndarray) -> dict[tuple[int,int], set]:
    ys, xs = np.nonzero(skeleton)
    nodes = {(int(y), int(x)) for y, x in zip(ys, xs)}
    graph = {}
    for y, x in nodes:
        edges = set()
        for dy, dx in NEIGHBORS:
            q = (y + dy, x + dx)
            if q not in nodes:
                continue
            # Remove diagonal shortcut across a right-angled pixel corner.
            # Without this, a simple staircase curve produces dozens of
            # spurious branch nodes in an 8-neighbor graph.
            if dy and dx and ((y+dy, x) in nodes or (y, x+dx) in nodes):
                continue
            edges.add(q)
        graph[y, x] = edges
    return graph


def _prune_spurs(g: dict, min_length: int) -> dict:
    """Remove short skeleton terminal twigs; don't remove main curves."""
    g = {k: set(v) for k, v in g.items()}
    for _ in range(5):
        drop = set()
        for start in (n for n in g if len(g[n]) == 1):
            chain = [start]
            prev, cur = None, start
            while len(chain) <= min_length:
                nxt = [n for n in g[cur] if n != prev]
                if len(nxt) != 1:
                    break
                prev, cur = cur, nxt[0]
                chain.append(cur)
                if len(g[cur]) != 2:
                    break
            if len(g[cur]) >= 3 and len(chain) - 1 < min_length:
                drop.update(chain[:-1])
        if not drop:
            break
        for node in drop:
            for other in g[node]:
                if other not in drop:
                    g[other].discard(node)
            del g[node]
    return g


def _components(g):
    left = set(g)
    groups = []
    while left:
        start = next(iter(left)); q = deque([start]); left.remove(start)
        comp = {start}
        while q:
            for n in g[q.popleft()]:
                if n in left:
                    left.remove(n); comp.add(n); q.append(n)
        groups.append(comp)
    return groups


def _shortest_path(g, source, target):
    queue = deque([source]); previous = {source: None}
    while queue:
        cur = queue.popleft()
        if cur == target:
            path = []
            while cur is not None:
                path.append(cur); cur = previous[cur]
            path.reverse()
            return path
        for next_node in g[cur]:
            if next_node not in previous:
                previous[next_node] = cur; queue.append(next_node)
    return []


def _smoothness(path) -> float:
    """Smooth *curved* paths score well; sharp corners are penalized."""
    p = np.array([(x, y) for y, x in path], dtype=float)
    if len(p) < 10:
        return 0.0
    # Compare tangents on both sides of a point over a 4-pixel baseline.
    step = 4
    v1 = p[step:-step] - p[:-2*step]
    v2 = p[2*step:] - p[step:-step]
    n1, n2 = np.linalg.norm(v1, axis=1), np.linalg.norm(v2, axis=1)
    active = (n1 > 0.1) & (n2 > 0.1)
    if not active.any():
        return 0.0
    cos = np.clip((v1[active] * v2[active]).sum(axis=1) /
                  (n1[active] * n2[active]), -1, 1)
    angle = np.degrees(np.arccos(cos))
    # Allow smooth curves of up to 35 degrees local rotation, penalize elbows.
    return float(np.exp(-np.mean(np.clip(angle-35, 0, None))/25.0))


def _pairs(endpoints):
    """All pairings of at most 8 endpoints (max 105 pairings)."""
    if not endpoints:
        yield []
        return
    first = endpoints[0]
    for k in range(1, len(endpoints)):
        for rest in _pairs(endpoints[1:k] + endpoints[k+1:]):
            yield [(first, endpoints[k])] + rest


def _select_paths(g, endpoints):
    if len(endpoints) < 4 or len(endpoints) % 2 or len(endpoints) > 8:
        return []
    # Cache paths and bend penalties by endpoint pair.
    cache = {}
    for a, b in combinations(endpoints, 2):
        path = _shortest_path(g, a, b)
        if path:
            cache[frozenset((a, b))] = path
    best = (-1e9, [])
    for arrangement in _pairs(endpoints):
        trajectories = []
        for a, b in arrangement:
            p = cache.get(frozenset((a, b)))
            if not p or len(p) < 12:
                break
            trajectories.append(p)
        else:
            # Strongly prefer smooth paths of reasonable lengths with small
            # overlap; crossing paths may share the center of the skeleton.
            smooth = np.mean([_smoothness(p) for p in trajectories])
            lengths = [len(p) for p in trajectories]
            union = len(set().union(*(set(p) for p in trajectories)))
            redundancy = (sum(lengths) - union) / max(1, sum(lengths))
            coverage = union / max(1, len(g))
            quality = 2*smooth + coverage - 1.5*redundancy
            if quality > best[0]:
                best = (quality, trajectories)
    return best[1]


def _signal_peak(cluster, junction_xy):
    """Support from a LOCAL signal excess near a real intersection only."""
    if junction_xy is None:
        return (0.0, None)
    points = np.array([(p['x'], p['y']) for p in cluster], dtype=float)
    sig = np.array([p['value'] for p in cluster], dtype=float)
    if len(sig) == 0 or not np.isfinite(sig).all() or (sig < 0).any():
        return (0.0, None)
    distance = np.linalg.norm(points - np.asarray(junction_xy), axis=1)
    core = sig[distance <= 2.5]
    arms = sig[(distance >= 4) & (distance <= 10)]
    if len(core) < 1 or len(arms) < 5:
        return (0.0, None)
    # Use medians rather than a single saturated pixel max.
    baseline = float(np.median(arms))
    central = float(np.median(core))
    if baseline <= 0:
        return (0.0, None)
    ratio = central / baseline
    support = float(np.clip((ratio-1.2)/1.8, 0, 1))
    return support, float(ratio)


def score_m(cluster, *, min_spur=4, max_tracks=4, **_ignored):
    """Return experimental score and CURVED track paths for visualization.

    Do not compare M3 scores with M1/M2. A lone curved trajectory gets M=0
    even if it could be approximated by multiple RANSAC straight lines.
    """
    result = dict(M=0.0, version=VERSION, status='too_few_pixels',
                  n_pixels=len(cluster), n_tracks=0, geometry='none',
                  paths=[], skeleton=[], signal_peak_ratio=None,
                  signal_support=0.0, path_smoothness=0.0,
                  coverage=0.0, junction=None, signal_units='unknown')
    if len(cluster) < 16:
        return result
    pts = np.array([(p['x'], p['y']) for p in cluster], dtype=int)
    if len(np.unique(pts, axis=0)) != len(pts):
        raise ValueError('Duplicate coordinates inside cluster')
    low = pts.min(axis=0)
    shifted = pts - low + 2
    width, height = np.ptp(shifted, axis=0) + 5
    mask = np.zeros((height, width), dtype=bool)
    mask[shifted[:, 1], shifted[:, 0]] = True
    thin = _thin_zhang_suen(mask)
    g = _prune_spurs(_make_graph(thin), min_spur)
    if not g:
        result['status'] = 'no_skeleton'
        return result
    offset = np.array([low[0]-2, low[1]-2])
    result['skeleton'] = [(x + int(offset[0]), y + int(offset[1]))
                          for y, x in g]
    comps = _components(g)
    # Segments with multiple disconnected components are unusual for CLOG
    # clusters. Treat their paths as independent candidates, not pile-up proof.
    paths = []
    for comp in comps:
        ends = [n for n in comp if len(g[n]) == 1]
        if len(ends) == 2:
            p = _shortest_path(g, ends[0], ends[1])
            if p and len(p) >= 10:
                paths.append(p)
        elif len(ends) >= 4 and len(ends) % 2 == 0:
            paths.extend(_select_paths({n: g[n] for n in comp}, ends))
    def to_xy(p):
        return [(int(x+offset[0]), int(y+offset[1])) for y, x in p]
    result['paths'] = [to_xy(p) for p in paths]
    result['n_tracks'] = len(paths)
    if len(paths) < 2:
        result['geometry'] = 'single_curve' if paths else 'unresolved'
        result['status'] = 'single_or_unresolved'
        return result
    if len(paths) > max_tracks:
        result['status'] = 'too_many_branches'
        return result

    # How well do 2-4 smooth paths explain a single skeleton?
    count = len(set().union(*(set(p) for p in paths)))
    coverage = count / max(1, len(g))
    smooth = float(np.mean([_smoothness(p) for p in paths]))
    sizes = [len(p) for p in paths]
    balance = min(sizes)/max(sizes)
    result.update(coverage=float(coverage), path_smoothness=smooth)
    if coverage < 0.68 or smooth < 0.55 or balance < 0.2:
        result['status'] = 'weak_multiple_paths'
        return result

    # Distinct paths cannot be practically identical; shared skeleton nodes
    # are allowed at crossings, not along a substantial common segment.
    redundancy = (sum(sizes) - count)/max(1, sum(sizes))
    if redundancy > 0.22:
        result['status'] = 'overlapping_paths'
        return result

    # The main path directions are estimated globally to distinguish parallel
    # tracks and crossings; actual displayed paths remain curved.
    vectors = []
    for p in paths:
        first = np.array((p[0][1], p[0][0]), float)
        last = np.array((p[-1][1], p[-1][0]), float)
        v = last - first
        v /= max(1e-9, np.linalg.norm(v))
        vectors.append(v)
    theta = math.degrees(math.acos(float(np.clip(abs(np.dot(vectors[0], vectors[1])), 0, 1))))
    result['geometry'] = ('parallel_curves' if theta < 18 else 'crossing_or_curves')
    # Find a *real shared junction* or a close approach between paths.
    intersection = set(paths[0]) & set(paths[1])
    junction_xy = None
    if intersection:
        near = np.array([[x, y] for y, x in intersection], float)
        junction_xy = near.mean(axis=0)
    signal_support, peak_ratio = _signal_peak(cluster,
        None if junction_xy is None else junction_xy + offset)
    result['signal_peak_ratio'] = peak_ratio
    result['signal_support'] = signal_support
    if junction_xy is not None:
        result['junction'] = (float(junction_xy[0] + offset[0]),
                              float(junction_xy[1] + offset[1]))

    # Energy never substitutes for geometric evidence. Without a calibrated
    # reference the *sum* and *max* of detector signals are diagnostic only.
    geometry = float(coverage * smooth * np.sqrt(balance) * (1-redundancy))
    # M deliberately stays GEOMETRIC. Signal is separate evidence E and the
    # local peak is only diagnostic, avoiding double-counting signal in M + E.
    result['M'] = float(np.clip(geometry, 0, 1))
    result['status'] = 'multi_curve_candidate'
    return result
