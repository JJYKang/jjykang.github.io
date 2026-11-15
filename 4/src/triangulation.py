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


def linear_estimate_3d_point(image_points, camera_matrices):
    """Return a length-three DLT estimate from M image points and cameras, M >= 2.

    image_points has shape (M, 2); camera_matrices has shape (M, 3, 4).
    Cameras map homogeneous world points to image coordinates. The estimate
    is expressed in their shared world frame; positive depth is not required.
    """
    points = np.asarray(image_points, dtype=np.float64)
    cameras = np.asarray(camera_matrices, dtype=np.float64)
    if (points.ndim != 2 or points.shape[1] != 2 or len(points) < 2
            or cameras.shape != (len(points), 3, 4)):
        raise ValueError("Provide matching M-by-2 points and M-by-3-by-4 cameras, M >= 2")
    if not np.isfinite(points).all() or not np.isfinite(cameras).all():
        raise ValueError("Image points and camera matrices must be finite")

    # Eliminate each image's projective scale using two cross-product equations.
    a = np.empty((2 * len(points), 4))
    a[0::2] = points[:, 0:1] * cameras[:, 2] - cameras[:, 0]
    a[1::2] = points[:, 1:2] * cameras[:, 2] - cameras[:, 1]
    _, values, vt = np.linalg.svd(a, full_matrices=False)
    if values[2] <= max(a.shape) * np.finfo(float).eps * values[0]:
        raise ValueError("Camera observations do not determine a unique 3D point")
    homogeneous = vt[-1]
    if abs(homogeneous[3]) <= 16 * np.finfo(float).eps:
        raise ValueError("The triangulated point is at infinity")
    depths = cameras[:, 2] @ homogeneous
    tolerance = 16 * np.finfo(float).eps * np.linalg.norm(cameras[:, 2], axis=1)
    if np.any(np.abs(depths) <= tolerance):
        raise ValueError("The triangulated point has an undefined image projection")
    return homogeneous[:3] / homogeneous[3]
