"""Camera pose estimation and projective triangulation."""
import numpy as np


def estimate_initial_RT(E):
    """Return four 3-by-4 [R | t] candidates as a 4-by-3-by-4 array.

    E follows p2.T @ E @ p1 = 0 for calibrated camera coordinates.
    Each pose maps points from Camera 1 into Camera 2: X2 = R @ X1 + t.
    Translation has unit length because its magnitude is not determined by E.
    """
    essential = np.asarray(E, dtype=np.float64)
    if (essential.shape != (3, 3) or not np.isfinite(essential).all()
            or not np.any(essential)):
        raise ValueError("Provide a finite, nonzero 3-by-3 essential matrix")
    essential = essential / np.abs(essential).max()
    u, values, vt = np.linalg.svd(essential)
    if values[1] <= 3 * np.finfo(float).eps * values[0]:
        raise ValueError("The essential matrix must have at least rank two")
    w = np.array([[0., -1, 0], [1, 0, 0], [0, 0, 1]])
    translation = u[:, 2]
    candidates = []
    for middle in (w, w.T):
        rotation = u @ middle @ vt
        # Orthogonality alone permits det(R) = -1; retain proper rotations.
        if np.linalg.det(rotation) < 0:
            rotation = -rotation
        for sign in (1, -1):
            candidates.append(np.column_stack((rotation, sign * translation)))
    return np.stack(candidates)
