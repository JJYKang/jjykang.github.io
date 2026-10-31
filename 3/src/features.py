"""Interest-point selection for automatic stitching."""
import numpy as np


def anms(points, strengths, count=500, c_robust=0.9):
    """Return selected indices and all suppression radii for n-by-2 (x, y) points.

    Each radius is the distance to the nearest point j satisfying
    strength[i] < c_robust * strength[j]. Points with no such neighbor have
    infinite radius. Keep the largest radii; break ties by strength, then input
    order. Pairwise distances are computed in blocks to bound memory usage.
    """
    points = np.asarray(points, dtype=float)
    strengths = np.asarray(strengths, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or strengths.shape != (len(points),):
        raise ValueError("Provide n-by-2 points and n strengths")
    if not np.isfinite(points).all() or not np.isfinite(strengths).all() or np.any(strengths <= 0):
        raise ValueError("Points must be finite and strengths finite and positive")
    if not isinstance(count, (int, np.integer)) or count < 0 or not 0 < c_robust < 1:
        raise ValueError("Use a nonnegative integer count and 0 < c_robust < 1")
    radius_squared = np.full(len(points), np.inf)
    for start in range(0, len(points), 128):
        stop = min(start + 128, len(points))
        dx = points[start:stop, 0, None] - points[None, :, 0]
        dy = points[start:stop, 1, None] - points[None, :, 1]
        distance_squared = dx * dx + dy * dy
        stronger = strengths[start:stop, None] < c_robust * strengths[None, :]
        distance_squared[~stronger] = np.inf
        radius_squared[start:stop] = distance_squared.min(axis=1)
    order = np.lexsort((np.arange(len(points)), -strengths, -radius_squared))
    return order[:count], np.sqrt(radius_squared)
