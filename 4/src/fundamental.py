"""Fundamental matrix estimation from corresponding image points."""
import numpy as np


def lls_eight_point_alg(points1, points2):
    """Return rank-two F satisfying p2.T @ F @ p1 approximately equal to zero.

    Inputs are matching n-by-3 homogeneous point arrays, with n >= 8.
    Fit directly in pixel coordinates without centering or rescaling them.
    F has unit Frobenius norm, with its largest-magnitude entry positive.
    """
    first = np.asarray(points1, dtype=np.float64)
    second = np.asarray(points2, dtype=np.float64)
    if (first.ndim != 2 or first.shape[1] != 3 or second.shape != first.shape
            or len(first) < 8):
        raise ValueError("Provide matching n-by-3 point arrays with n >= 8")
    if not np.isfinite(first).all() or not np.isfinite(second).all():
        raise ValueError("Point coordinates must be finite")
    if np.any(first[:, 2] == 0) or np.any(second[:, 2] == 0):
        raise ValueError("Image points must have a nonzero homogeneous coordinate")
    first = first / first[:, 2:3]
    second = second / second[:, 2:3]

    # Each outer product supplies one row for p2.T @ F @ p1 = 0.
    a = (second[:, :, None] * first[:, None, :]).reshape(-1, 9)
    # An 8-by-9 system needs the full V.T to retain its null-space vector.
    _, singular_values, vt = np.linalg.svd(a, full_matrices=(len(first) == 8))
    tolerance = singular_values[0] * max(a.shape) * np.finfo(float).eps
    if np.count_nonzero(singular_values > tolerance) < 8:
        raise ValueError("Correspondences do not determine a unique fundamental matrix")
    f = vt[-1].reshape(3, 3)

    # The closest rank-two matrix is obtained by removing the smallest value.
    u, singular_values, vt = np.linalg.svd(f)
    singular_values[-1] = 0
    f = (u * singular_values) @ vt

    # F is defined only up to scale; fix the scale and sign for saved results.
    f /= np.linalg.norm(f)
    if f.flat[np.argmax(np.abs(f))] < 0:
        f = -f
    return f
