"""Least-squares homography estimation from corresponding image points."""
import numpy as np


def _normalize(points):
    # Center and scale coordinates to keep the linear system well conditioned.
    center = points.mean(axis=0)
    distance = np.linalg.norm(points - center, axis=1).mean()
    if distance <= np.finfo(float).eps:
        raise ValueError("Correspondences must contain distinct points")
    scale = np.sqrt(2) / distance
    transform = np.array([[scale, 0, -scale * center[0]],
                          [0, scale, -scale * center[1]], [0, 0, 1]])
    return (points - center) * scale, transform


def computeH(im1_pts, im2_pts):
    """Return H mapping (x, y) points in image 1 to image 2, with H[2, 2] = 1."""
    source = np.asarray(im1_pts, dtype=np.float64)
    target = np.asarray(im2_pts, dtype=np.float64)
    if (source.ndim != 2 or source.shape[1] != 2 or target.shape != source.shape
            or len(source) < 4):
        raise ValueError("Provide matching n-by-2 point arrays with n >= 4")
    if not np.isfinite(source).all() or not np.isfinite(target).all():
        raise ValueError("Point coordinates must be finite")
    source, t1 = _normalize(source)
    target, t2 = _normalize(target)
    x, y = source.T
    u, v = target.T
    # Each pair contributes two equations for the eight unknown entries of H.
    a = np.zeros((2 * len(source), 8))
    a[0::2, :3] = np.column_stack((x, y, np.ones(len(x))))
    a[1::2, 3:6] = np.column_stack((x, y, np.ones(len(x))))
    a[0::2, 6:] = -np.column_stack((u * x, u * y))
    a[1::2, 6:] = -np.column_stack((v * x, v * y))
    h, _, rank, _ = np.linalg.lstsq(a, target.ravel(), rcond=None)
    if rank < 8:
        raise ValueError("Correspondences do not determine a unique homography")
    normalized_h = np.append(h, 1).reshape(3, 3)
    result = np.linalg.inv(t2) @ normalized_h @ t1
    if abs(result[2, 2]) < np.finfo(float).eps * np.linalg.norm(result):
        raise ValueError("Homography cannot be scaled to H[2, 2] = 1")
    return result / result[2, 2]
