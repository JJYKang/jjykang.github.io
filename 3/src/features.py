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


def extract_descriptors(image, points, sigma=2.0, min_std=1e-8):
    """Return normalized n-by-64 descriptors and their input-point indices.

    Use integer (x, y) corners, extract [y-20:y+20, x-20:x+20], blur each
    40-by-40 patch, and sample every fifth pixel to obtain an 8-by-8 patch.
    Border points and patches with standard deviation <= min_std are omitted.
    """
    image = np.asarray(image, dtype=float)
    points = np.asarray(points, dtype=float)
    if image.ndim != 2 or not np.isfinite(image).all():
        raise ValueError("Provide a finite grayscale image")
    if (points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all()
            or not np.equal(points, np.round(points)).all()):
        raise ValueError("Provide n-by-2 integer pixel coordinates")
    if not np.isfinite(sigma) or sigma <= 0 or not np.isfinite(min_std) or min_std < 0:
        raise ValueError("Use positive sigma and nonnegative min_std")
    radius = int(np.ceil(3 * sigma))
    offsets = np.arange(-radius, radius + 1)
    kernel = np.exp(-offsets ** 2 / (2 * sigma ** 2))
    kernel /= kernel.sum()
    descriptors, indices = [], []
    for index, (x, y) in enumerate(points.astype(int)):
        if x < 20 or y < 20 or x + 20 > image.shape[1] or y + 20 > image.shape[0]:
            continue
        patch = image[y - 20:y + 20, x - 20:x + 20]
        # Separable Gaussian smoothing with reflected patch boundaries.
        for axis in (0, 1):
            padding = [(0, 0), (0, 0)]
            padding[axis] = (radius, radius)
            padded = np.pad(patch, padding, mode="reflect")
            patch = np.apply_along_axis(lambda line: np.convolve(line, kernel, mode="valid"), axis, padded)
        small = patch[2::5, 2::5].ravel()
        centered = small - small.mean()
        std = small.std()
        if std <= min_std:
            continue
        descriptors.append(centered / std)
        indices.append(index)
    return np.asarray(descriptors, dtype=float).reshape(-1, 64), np.asarray(indices, dtype=int)


def match_descriptors(descriptors1, descriptors2, ratio_threshold=0.5):
    """Return matching index pairs and their nearest/second-nearest SSD ratios.

    Query each descriptor in the first image against the second image. Accept
    a match when its SSD ratio is strictly below ratio_threshold. Two identical
    nearest descriptors give an ambiguous match and are rejected.
    """
    a = np.asarray(descriptors1, dtype=float)
    b = np.asarray(descriptors2, dtype=float)
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[1] or a.shape[1] == 0:
        raise ValueError("Provide two descriptor matrices with the same feature dimension")
    if not np.isfinite(a).all() or not np.isfinite(b).all() or not 0 < ratio_threshold < 1:
        raise ValueError("Descriptors must be finite and 0 < ratio_threshold < 1")
    if len(a) == 0 or len(b) < 2:
        return np.empty((0, 2), dtype=int), np.empty(0, dtype=float)
    # Squared distances without allocating an n-by-m-by-64 difference array.
    distances = np.maximum(np.sum(a * a, axis=1)[:, None]
                           + np.sum(b * b, axis=1)[None, :] - 2 * (a @ b.T), 0)
    nearest = np.argsort(distances, axis=1, kind="stable")[:, :2]
    rows = np.arange(len(a))
    first = distances[rows, nearest[:, 0]]
    second = distances[rows, nearest[:, 1]]
    ratios = np.divide(first, second, out=np.full(len(a), np.inf), where=second > 1e-12)
    keep = ratios < ratio_threshold
    return np.column_stack((rows[keep], nearest[keep, 0])), ratios[keep]
