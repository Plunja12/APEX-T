"""Prototype topology anomaly score T for Timepix clusters.

T is a *ranking heuristic*, not a probability of pile-up.
The skeleton is computed from the ORIGINAL active pixel coordinates, no
signal percentile threshold, no timestamps. T and M share geometry and
are statistically dependent; do not add them as independent evidence.
"""
from __future__ import annotations

from collections import deque
import numpy as np
def skeletonize(mask: np.ndarray) -> np.ndarray:
    """Zhang-Suen thinning, 8-connected, using only NumPy.

    Return a one-pixel-wide approximate centre-line of a binary image.
    This is a heuristic thinning implementation, not a pixel-perfect
    replacement for scikit-image's skeletonize.
    """
    image = np.asarray(mask, dtype=bool).copy()
    if image.ndim != 2:
        raise ValueError("Skeleton input must be a 2D binary image")
    if not image.any():
        return image

    # Each loop alternates two removal conditions. Evaluate all marked
    # pixels simultaneously to avoid direction-dependent results.
    while True:
        changed = False
        for step in (0, 1):
            p = np.pad(image, 1, mode="constant")
            n = p[:-2, 1:-1]   # P2: north
            ne = p[:-2, 2:]    # P3: northeast
            e = p[1:-1, 2:]    # P4: east
            se = p[2:, 2:]     # P5: southeast
            so = p[2:, 1:-1]   # P6: south
            sw = p[2:, :-2]   # P7: southwest
            w = p[1:-1, :-2]  # P8: west
            nw = p[:-2, :-2]  # P9: northwest
            ring = (n, ne, e, se, so, sw, w, nw)
            count = sum(arr.astype(np.uint8) for arr in ring)
            transitions = sum((~ring[i]) & ring[(i+1) % 8]
                              for i in range(8))
            remove = image & (count >= 2) & (count <= 6) & (transitions == 1)
            if step == 0:
                remove &= ~(n & e & so) & ~(e & so & w)
            else:
                remove &= ~(n & e & w) & ~(n & so & w)
            if remove.any():
                image[remove] = False
                changed = True
        if not changed:
            break
    return image

VERSION_T = 'T1.1-zhang-suen-numpy'
_OFFSETS = [(dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if dx or dy]


def _graph(skel: np.ndarray) -> dict[tuple[int, int], set[tuple[int, int]]]:
    """8-connectivity without diagonal corner shortcuts."""
    nodes = {(int(y), int(x)) for y, x in zip(*np.nonzero(skel))}
    graph = {}
    for y, x in nodes:
        neighbors = set()
        for dy, dx in _OFFSETS:
            q = (y+dy, x+dx)
            if q not in nodes:
                continue
            if dx and dy and ((y+dy, x) in nodes or (y, x+dx) in nodes):
                continue
            neighbors.add(q)
        graph[y, x] = neighbors
    return graph


def _prune_spurs(graph: dict, min_arm: int) -> dict:
    """Remove terminal twigs shorter than min_arm until stable."""
    g = {node: set(neighbors) for node, neighbors in graph.items()}
    for _ in range(12):
        remove = set()
        for end in [n for n in g if len(g[n]) == 1]:
            chain = [end]
            previous = None
            current = end
            while len(chain) - 1 < min_arm:
                nxt = [n for n in g[current] if n != previous]
                if len(nxt) != 1:
                    break
                previous, current = current, nxt[0]
                chain.append(current)
                if len(g[current]) != 2:
                    break
            # Only trim if a short twig actually reaches a junction.
            if len(g[current]) >= 3 and len(chain) - 1 < min_arm:
                remove.update(chain[:-1])
        if not remove:
            break
        for node in remove:
            for nb in g[node]:
                if nb not in remove:
                    g[nb].remove(node)
            del g[node]
    return g


def _junction_groups(graph: dict) -> list[set]:
    """Merge adjacent high-degree pixels into a single physical junction."""
    branches = {n for n in graph if len(graph[n]) >= 3}
    groups = []
    while branches:
        start = branches.pop()
        group = {start}
        queue = deque([start])
        while queue:
            y, x = queue.popleft()
            nearby = {n for n in branches if abs(n[0]-y) <= 2 and abs(n[1]-x) <= 2}
            for nb in nearby:
                branches.remove(nb)
                group.add(nb)
                queue.append(nb)
        groups.append(group)
    return groups


def score_t(cluster: list[dict], *, min_spur: int = 5) -> dict:
    """Compute skeleton topology without touching signal/timing fields.

    Returns T (0..1) and pixel coordinates for display.
    Interpretation (provisional): 3 strong arms ~0.45; 4 ~0.80;
    5+ ~0.95; additional junction groups raise score slightly.
    """
    result = {'T': 0.0, 'version_t': VERSION_T, 'status_t': 'too_small',
              'skeleton_t': [], 'endpoints_t': [], 'junctions_t': [],
              'n_endpoints_t': 0, 'n_junctions_t': 0,
              'n_skeleton_pixels_t': 0}
    if len(cluster) < 8:
        return result
    pts = np.asarray([(p['x'], p['y']) for p in cluster], dtype=int)
    if (pts < 0).any() or not np.isfinite(pts).all():
        result['status_t'] = 'invalid_coordinates'
        return result
    unique = np.unique(pts, axis=0)
    if len(unique) != len(pts):
        result['status_t'] = 'duplicate_coordinates'
        return result
    x0, y0 = unique.min(axis=0)
    xy = unique - (x0, y0) + 2
    width, height = np.ptp(xy, axis=0) + 5
    if width * height > 256 * 256 * 4:
        result['status_t'] = 'too_large'
        return result
    mask = np.zeros((int(height), int(width)), dtype=bool)
    mask[xy[:, 1], xy[:, 0]] = True
    skel = skeletonize(mask)
    g = _prune_spurs(_graph(skel), min_arm=min_spur)
    def to_xy(p):
        y, x = p
        return (int(x+x0-2), int(y+y0-2))
    result['skeleton_t'] = [to_xy(p) for p in g]
    result['n_skeleton_pixels_t'] = len(g)
    if not g:
        result['status_t'] = 'no_skeleton'
        return result
    ends = [p for p, ns in g.items() if len(ns) == 1]
    junctions = _junction_groups(g)
    result['endpoints_t'] = [to_xy(p) for p in ends]
    result['junctions_t'] = [to_xy(min(group)) for group in junctions]
    result['n_endpoints_t'] = len(ends)
    result['n_junctions_t'] = len(junctions)
    if not junctions:
        result['status_t'] = 'no_branching'
        return result
    # Count endpoints which have a meaningful uninterrupted arm reaching a junction.
    # A short terminal twig should not count as another incident particle.
    reached = 0
    for end in ends:
        prev = None
        cur = end
        length = 0
        while True:
            nxt = [n for n in g[cur] if n != prev]
            if len(g[cur]) >= 3:
                if length >= min_spur:
                    reached += 1
                break
            if len(nxt) != 1:
                break
            prev, cur = cur, nxt[0]
            length += 1
    if reached < 3:
        result['status_t'] = 'no_three_supported_arms'
        return result
    branch_strength = 0.45 if reached == 3 else 0.80 if reached == 4 else 0.95
    extra = min(0.05, 0.025 * max(0, len(junctions)-1))
    result['T'] = float(min(1.0, branch_strength + extra))
    result['status_t'] = f'{reached}_supported_arms'
    return result
