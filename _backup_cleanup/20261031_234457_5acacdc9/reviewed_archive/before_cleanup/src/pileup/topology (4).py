"""APEX-T M v2: fast geometric screening for 2-4 linear track candidates.

Accepts [{"x": int, "y": int, "value": float, "extra": ...}, ...].
Only uses x/y. This is NOT a calibrated pile-up probability.
Unlike v1, detects separated parallel lines as well as crossing lines.
"""

from __future__ import annotations

import numpy as np

VERSION = "M2.0"


def _line_distances(points, center, direction):
    delta = points - center
    return np.abs(delta[:, 0] * direction[1] - delta[:, 1] * direction[0])


def _fit_one(points, available, rng, trials, tolerance, min_support=6):
    """Vectorized random-pair line proposals; refine the winner with PCA."""
    subset = points[available]
    n = len(subset)
    if n < min_support:
        return None

    pairs = rng.integers(0, n, size=(trials * 2, 2))
    start = subset[pairs[:, 0]]
    delta = subset[pairs[:, 1]] - start
    lengths = np.hypot(delta[:, 0], delta[:, 1])
    keep = lengths >= 5.0
    if not np.any(keep):
        return None
    start = start[keep][:trials]
    direction = delta[keep][:trials] / lengths[keep][:trials, None]

    # (proposals, pixels) distances in one NumPy operation rather than
    # hundreds of slow Python loops / SVD calls.
    offsets = subset[None, :, :] - start[:, None, :]
    distances = np.abs(offsets[:, :, 0] * direction[:, None, 1] -
                       offsets[:, :, 1] * direction[:, None, 0])
    inside = distances <= tolerance
    counts = inside.sum(axis=1)
    if counts.max(initial=0) < min_support:
        return None
    losses = np.where(inside, distances, 0).sum(axis=1)
    objective = counts - losses / (n + 1.0) * 0.01
    best = int(np.argmax(objective))
    inliers = subset[inside[best]]
    center = inliers.mean(axis=0)

    # Principal axis = eigenvector associated with largest covariance eigenvalue.
    covariance = (inliers - center).T @ (inliers - center)
    _, eigenvectors = np.linalg.eigh(covariance)
    v = eigenvectors[:, -1]
    d = _line_distances(subset, center, v)
    mask = d <= tolerance
    if mask.sum() < min_support:
        return None
    support = subset[mask]
    span = np.ptp((support - center) @ v)
    if span < 6.0:
        return None
    return {"center": center, "direction": v,
            "indices": available[mask], "span": float(span)}


def _arms(points, indices, line, cross):
    along = (points[indices] - cross) @ line["direction"]
    neg = along[along < -1.5]
    pos = along[along > 1.5]
    if len(neg) < 2 or len(pos) < 2:
        return 0.0
    if -np.min(neg) < 3.0 or np.max(pos) < 3.0:
        return 0.0
    return float(min(len(neg), len(pos)) / max(len(neg), len(pos)))


def _parallel_check(points, line_a, line_b, tolerance):
    """Check if two parallel tracks are resolved as distinct parallel ridges.

    A continuous thick stripe must not automatically become two tracks:
    require spatial separation, longitudinal overlap, and a relatively empty
    gap between the two fitted centerlines.
    """
    v = line_a["direction"].copy()
    if np.dot(v, line_b["direction"]) < 0:
        v = -v
    normal = np.array([-v[1], v[0]])
    off_a = float(line_a["center"] @ normal)
    off_b = float(line_b["center"] @ normal)
    separation = abs(off_a - off_b)
    if separation < max(2.75, 2 * tolerance + 0.6):
        return 0.0, separation, 1.0

    along_a = points[line_a["indices"]] @ v
    along_b = points[line_b["indices"]] @ v
    low = max(float(along_a.min()), float(along_b.min()))
    high = min(float(along_a.max()), float(along_b.max()))
    overlap = max(0.0, high - low)
    overlap_fraction = overlap / max(1e-8, min(np.ptp(along_a), np.ptp(along_b)))
    if overlap < 5.0 or overlap_fraction < 0.4:
        return 0.0, separation, 1.0

    selected = points[(points @ v >= low) & (points @ v <= high)]
    across = selected @ normal
    left, right = sorted((off_a, off_b))
    # How many pixels occupy the valley between the two ridges?
    in_middle = (across > left + tolerance) & (across < right - tolerance)
    in_ridges = ((abs(across - left) <= tolerance) |
                 (abs(across - right) <= tolerance))
    gap_ratio = float(in_middle.sum() / max(1, int(in_ridges.sum())))
    if gap_ratio > 0.32:
        return 0.0, separation, gap_ratio

    separation_factor = float(np.clip((separation - 2.0) / 2.0, 0, 1))
    shape = overlap_fraction * (1 - min(gap_ratio / 0.75, 1)) * separation_factor
    return float(shape), separation, gap_ratio


def score_m(cluster, *, trials=64, tolerance=0.85, seed=42, max_tracks=4):
    """Return heuristic M in [0,1] and diagnostics for crossing/parallel tracks.

    M2.0 scores are NOT comparable with the old crossing-only M1 scores.
    No timestamps, energy or pixel amplitudes are consulted.
    """
    result = {"M": 0.0, "version": VERSION, "status": "too_few_pixels",
              "n_pixels": len(cluster), "n_tracks": 0, "geometry": "none",
              "angle_deg": 0.0, "coverage": 0.0, "balance": 0.0,
              "arm_balance": 0.0, "separation_px": 0.0,
              "gap_ratio": 0.0, "lines": [], "intersection": None}
    if len(cluster) < 16:
        return result
    points = np.array([(p["x"], p["y"]) for p in cluster], dtype=float)
    if len(np.unique(points, axis=0)) < len(points):
        raise ValueError("Duplicate pixel coordinates in cluster")
    rng = np.random.default_rng(seed)
    available = np.arange(len(points))
    lines = []

    for _ in range(max_tracks):
        model = _fit_one(points, available, rng, trials, tolerance)
        if model is None:
            break
        lines.append(model)
        available = np.setdiff1d(available, model["indices"], assume_unique=True)
        if len(available) < 6:
            break
    result["n_tracks"] = len(lines)  # number of raw fitted lines, not true particles
    if len(lines) < 2:
        result["status"] = "fewer_than_two_lines"
        return result

    # Count supporting pixels for each line without double-counting overlaps.
    ds = np.vstack([_line_distances(points, l["center"], l["direction"])
                    for l in lines])
    owner = np.argmin(ds, axis=0)
    near = np.min(ds, axis=0) <= tolerance
    coverage = float(near.mean())
    result["coverage"] = coverage
    if coverage < 0.55:
        result["status"] = "low_coverage"
        return result

    groups = [np.flatnonzero(near & (owner == i)) for i in range(len(lines))]
    best = (0.0, None)
    for i in range(len(lines)):
        for j in range(i+1, len(lines)):
            a, b = lines[i], lines[j]
            n1, n2 = len(groups[i]), len(groups[j])
            if min(n1, n2) < 6:
                continue
            balance = 2 * min(n1, n2) / (n1 + n2)
            angle = float(np.degrees(np.arccos(np.clip(
                abs(float(a["direction"] @ b["direction"])), 0, 1))))
            score, geometry, cross, arms = 0.0, "none", None, 0.0
            sep, gap = 0.0, 0.0
            if angle >= 16.0:
                # A crossing must have significant support on both sides of
                # the common point for both lines (distinguishes X from Y).
                matrix = np.column_stack((a["direction"], -b["direction"]))
                ts = np.linalg.solve(matrix, b["center"] - a["center"])
                cross = a["center"] + ts[0] * a["direction"]
                arms = min(_arms(points, groups[i], a, cross),
                           _arms(points, groups[j], b, cross))
                shape = arms * np.clip((angle - 16.0) / 24.0, 0, 1)
                score = balance * shape
                geometry = "crossing"
            elif angle <= 12.0:
                shape, sep, gap = _parallel_check(points, a, b, tolerance)
                score = balance * shape
                geometry = "parallel"
            if score > best[0]:
                best = (float(score), (i, j, geometry, angle, cross, arms,
                                       balance, sep, gap))

    if best[1] is None:
        result["status"] = "no_valid_pair"
        return result
    i, j, geometry, angle, cross, arms, balance, sep, gap = best[1]
    # More valid supporting lines can improve confidence, but cannot create it.
    bonus = 1.0 + min(0.15, 0.075 * max(0, len(lines) - 2))
    result.update({"M": float(np.clip(coverage * best[0] * bonus, 0, 1)),
                   "status": f"{geometry}_candidate", "geometry": geometry,
                   "angle_deg": angle, "arm_balance": arms,
                   "balance": float(balance), "separation_px": sep,
                   "gap_ratio": gap, "intersection": cross,
                   "lines": [lines[k] for k in [i, j] +
                             [k for k in range(len(lines)) if k not in (i, j)]]})
    return result
