"""Four-point RANSAC homography estimation."""
from itertools import combinations
import numpy as np
from homography import computeH


def reprojection_errors(source, target, h):
    """Forward Euclidean reprojection distances in target-image pixels."""
    projected = np.column_stack((source, np.ones(len(source)))) @ h.T
    with np.errstate(divide="ignore", invalid="ignore"):
        projected = projected[:, :2] / projected[:, 2:]
        errors = np.linalg.norm(projected - target, axis=1)
    return np.where(np.isfinite(errors), errors, np.inf)


def _nondegenerate(points):
    scale_squared = np.sum(np.ptp(points, axis=0) ** 2)
    for a, b, c in combinations(points, 3):
        u, v = b - a, c - a
        if abs(u[0] * v[1] - u[1] * v[0]) <= 1e-6 * scale_squared:
            return False
    return scale_squared > 0


def ransac_homography(source, target, threshold=3.0, max_iterations=5000,
                      confidence=0.999, seed=671):
    """Return (H, inliers, errors, trials) for an n-by-2 correspondence set.

    Sample four distinct pairs, reject collinear configurations, score by
    forward reprojection distance, then refit on the consensus. H maps source
    to target. Sampling is reproducible for a fixed seed.
    """
    source, target = np.asarray(source, dtype=float), np.asarray(target, dtype=float)
    if (source.ndim != 2 or source.shape[1] != 2 or source.shape != target.shape
            or len(source) < 4 or not np.isfinite(source).all() or not np.isfinite(target).all()):
        raise ValueError("Provide at least four finite corresponding (x, y) pairs")
    if (not np.isfinite(threshold) or threshold <= 0 or not 0 < confidence < 1
            or not isinstance(max_iterations, (int, np.integer)) or max_iterations < 1):
        raise ValueError("Use positive threshold/iteration limit and 0 < confidence < 1")
    rng = np.random.default_rng(seed)
    best_h, best_inliers = None, None
    best_score = (0, -np.inf)
    limit, trials = max_iterations, 0
    while trials < limit:
        trials += 1
        sample = rng.choice(len(source), 4, replace=False)
        if not _nondegenerate(source[sample]) or not _nondegenerate(target[sample]):
            continue
        try:
            h = computeH(source[sample], target[sample])
        except (ValueError, np.linalg.LinAlgError):
            continue
        errors = reprojection_errors(source, target, h)
        inliers = errors < threshold
        count = int(inliers.sum())
        score = (count, -float(np.median(errors[inliers])) if count else -np.inf)
        if count >= 4 and score > best_score:
            best_h, best_inliers, best_score = h, inliers, score
            probability = (count / len(source)) ** 4
            needed = 1 if probability >= 1 else int(np.ceil(np.log1p(-confidence) / np.log1p(-probability)))
            limit = min(limit, max(50, needed))
    if best_h is None:
        raise ValueError("RANSAC could not find a nondegenerate four-point model")
    # Reclassify after each least-squares refit to refine the consensus.
    h, inliers = best_h, best_inliers
    for _ in range(10):
        try:
            fitted = computeH(source[inliers], target[inliers])
        except (ValueError, np.linalg.LinAlgError):
            break
        errors = reprojection_errors(source, target, fitted)
        updated = errors < threshold
        if updated.sum() < 4:
            break
        h = fitted
        if np.array_equal(updated, inliers):
            break
        inliers = updated
    errors = reprojection_errors(source, target, h)
    return h, errors < threshold, errors, trials
