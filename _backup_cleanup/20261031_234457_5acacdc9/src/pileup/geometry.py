"""Experimental geometric anomaly score G for Timepix pixel clusters.

Only x/y coordinates are used. Pixel signal and timestamp are not used.
G is a screening HEURISTIC, not a probability of pile-up.

Features:
- local thickening along the original-mask skeleton (80% weight)
- non-convexity of the pixel footprint (20% weight)

Single curved tracks, delta-electron tracks, and pixel artifacts can score high.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.spatial import ConvexHull, QhullError

from pileup.topology import score_t

VERSION_G = 'G1.0-width-solidity'


def _hull_area(mask: np.ndarray) -> float:
    """Convex hull of pixel squares, with coordinates in pixel units."""
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return 0.0
    # Each pixel occupies a unit square around the pixel's integer center.
    xy = np.column_stack([xs, ys]).astype(float)
    corners = np.vstack([xy + delta for delta in
                         ((-.5, -.5), (-.5, .5), (.5, -.5), (.5, .5))])
    try:
        return float(ConvexHull(corners).volume)  # 2D volume == area
    except QhullError:
        return float(len(xs))


def score_g(cluster: list[dict], *, topology: dict | None = None) -> dict:
    """Calculate geometric anomaly score and per-pixel visualization features.

    Returns G in [0,1], width_component / concavity_component in [0,1],
    solidity, median_width, p90_width, and coordinates of locally thick pixels.
    The optional topology avoids recalculating skeleton T.
    """
    result = dict(G=0., version_g=VERSION_G, status_g='too_small',
                  width_component=0., concavity_component=0.,
                  solidity=1., median_width=0., p90_width=0.,
                  thick_spots_g=[], n_thick_spots_g=0)
    if len(cluster) < 8:
        return result
    xy = np.asarray([(p['x'], p['y']) for p in cluster], dtype=int)
    if xy.ndim != 2 or xy.shape[1] != 2 or np.any(xy < 0):
        result['status_g'] = 'invalid_coordinates'
        return result
    xy = np.unique(xy, axis=0)
    if not len(xy):
        return result
    x0, y0 = xy.min(axis=0)
    coords = xy - [x0, y0] + 2
    w, h = np.ptp(coords, axis=0) + 5
    if int(w) * int(h) > 256 * 256 * 4:
        result['status_g'] = 'too_large'
        return result
    mask = np.zeros((int(h), int(w)), dtype=bool)
    mask[coords[:, 1], coords[:, 0]] = True

    # Convex hull includes pixel square corners so thin lines get sensible area.
    area = float(mask.sum())
    hull_area = max(area, _hull_area(mask))
    solidity = float(np.clip(area / hull_area, 0., 1.))
    concavity = float(np.clip((0.65 - solidity) / 0.5, 0., 1.))

    if topology is None:
        topology = score_t(cluster)
    skel_points = np.asarray(topology.get('skeleton_t', []), dtype=int).reshape(-1, 2)
    radii = distance_transform_edt(mask)
    if len(skel_points):
        sx = skel_points[:, 0] - x0 + 2
        sy = skel_points[:, 1] - y0 + 2
        inside = ((sx >= 0) & (sx < w) & (sy >= 0) & (sy < h))
        sx, sy, skel_points = sx[inside], sy[inside], skel_points[inside]
        distances = radii[sy, sx]
    else:
        distances = np.empty(0, dtype=float)
    if len(distances) >= 5:
        median = float(np.median(distances))
        p90 = float(np.percentile(distances, 90))
        ratio = p90 / max(median, 1e-6)
        # An upper-quantile bulge has to be substantial to dominate G.
        width = float(np.clip((ratio - 1.) / 1.2, 0., 1.))
        hot = skel_points[distances >= max(1.3 * median, 1.45)]
        result['thick_spots_g'] = [tuple(map(int, pt)) for pt in hot]
        result['n_thick_spots_g'] = len(hot)
    else:
        median, p90, width = 0., 0., 0.
    result.update(G=float(np.clip(.80 * width + .20 * concavity, 0., 1.)),
                  status_g='ok', width_component=width,
                  concavity_component=concavity, solidity=solidity,
                  median_width=2 * median, p90_width=2 * p90)
    return result
