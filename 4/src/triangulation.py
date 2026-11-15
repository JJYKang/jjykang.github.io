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


def _project_point(point_3d, camera_matrices):
    """Validate a finite world point and cameras, then return projective coordinates."""
    point = np.asarray(point_3d, dtype=np.float64)
    cameras = np.asarray(camera_matrices, dtype=np.float64)
    if point.shape != (3,) or not np.isfinite(point).all():
        raise ValueError("Provide a finite length-three 3D point")
    if (cameras.ndim != 3 or cameras.shape[1:] != (3, 4) or len(cameras) < 1
            or not np.isfinite(cameras).all()):
        raise ValueError("Provide finite M-by-3-by-4 camera matrices, M >= 1")
    scales = np.max(np.abs(cameras), axis=(1, 2), keepdims=True)
    if np.any(scales == 0):
        raise ValueError("Camera matrices must be nonzero")
    cameras = cameras / scales
    projected = cameras[:, :, :3] @ point + cameras[:, :, 3]
    if np.any(projected[:, 2] == 0):
        raise ValueError("The 3D point has an undefined image projection")
    return projected, cameras


def reprojection_error(point_3d, image_points, camera_matrices):
    """Return projected-minus-observed residuals as (dx1, dy1, dx2, dy2, ...)."""
    projected, _ = _project_point(point_3d, camera_matrices)
    points = np.asarray(image_points, dtype=np.float64)
    if points.shape != (len(projected), 2) or not np.isfinite(points).all():
        raise ValueError("Provide finite M-by-2 image points matching the cameras")
    predicted = projected[:, :2] / projected[:, 2:3]
    return (predicted - points).reshape(-1)


def jacobian(point_3d, camera_matrices):
    """Return the analytic 2M-by-3 reprojection Jacobian, ordered x1, y1, x2, y2."""
    projected, cameras = _project_point(point_3d, camera_matrices)
    predicted = projected[:, :2] / projected[:, 2:3]
    # Quotient rule: d(y_i/y_3)/dX = (P_i - (y_i/y_3)*P_3) / y_3.
    derivatives = (cameras[:, :2, :3]
                   - predicted[:, :, None] * cameras[:, 2:3, :3])
    derivatives /= projected[:, 2, None, None]
    return derivatives.reshape(-1, 3)


def nonlinear_estimate_3d_point(image_points, camera_matrices):
    """Refine the DLT estimate with exactly ten undamped Gauss-Newton updates."""
    point = linear_estimate_3d_point(image_points, camera_matrices)
    for _ in range(10):
        error = reprojection_error(point, image_points, camera_matrices)
        derivatives = jacobian(point, camera_matrices)
        # Solving J*step ~= error avoids explicitly forming and inverting J.T*J.
        step, _, rank, _ = np.linalg.lstsq(derivatives, error, rcond=None)
        if rank < 3:
            raise ValueError("The reprojection Jacobian does not constrain a unique 3D update")
        point = point - step
        if not np.isfinite(point).all():
            raise ValueError("Gauss-Newton produced a nonfinite 3D point")
    _project_point(point, camera_matrices)
    return point


def _pose_depth_counts(candidates, image_points, K):
    """Count positive depths in both camera frames, excluding failed triangulations."""
    observations = np.asarray(image_points, dtype=np.float64)
    k = np.asarray(K, dtype=np.float64)
    if (observations.ndim != 3 or observations.shape[1:] != (2, 2)
            or len(observations) == 0 or not np.isfinite(observations).all()):
        raise ValueError("Provide finite N-by-2-by-2 correspondences, N >= 1")
    if (k.shape != (3, 3) or not np.isfinite(k).all()
            or np.linalg.matrix_rank(k) < 3):
        raise ValueError("Provide a finite, invertible 3-by-3 intrinsic matrix")
    first_camera = k @ np.column_stack((np.eye(3), np.zeros(3)))
    counts = np.zeros(len(candidates), dtype=int)
    failures = np.zeros(len(candidates), dtype=int)
    for index, pose in enumerate(candidates):
        cameras = np.stack((first_camera, k @ pose))
        for matches in observations:
            try:
                point = nonlinear_estimate_3d_point(matches, cameras)
            except (ValueError, np.linalg.LinAlgError):
                failures[index] += 1
                continue
            second_point = pose[:, :3] @ point + pose[:, 3]
            if point[2] > 0 and second_point[2] > 0:
                counts[index] += 1
    return counts, failures


def estimate_RT_from_E(E, image_points, K):
    """Return the 3-by-4 pose with the most triangulated points in front of both cameras.

    image_points has shape (N, 2, 2), with one (x, y) observation per camera.
    K is the shared intrinsic matrix; X2 = R @ X1 + t, with unit translation.
    Failed triangulations receive no vote. Tied maxima use the first candidate;
    if no point has positive depth in both cameras, the pose is rejected.
    """
    candidates = estimate_initial_RT(E)
    counts, _ = _pose_depth_counts(candidates, image_points, K)
    best = int(np.argmax(counts))
    if counts[best] == 0:
        raise ValueError("No candidate has triangulated points in front of both cameras")
    return candidates[best].copy()
