import numpy as np
from fundamental import _prepare_points


def build_measurement_matrix(points_im1, points_im2):
    # return centered measurements and row means, ordered (x1, y1, x2, y2)
    first, second = _prepare_points(points_im1, points_im2, min_points=4)
    measurements = np.vstack((first[:, :2].T, second[:, :2].T))
    translation = measurements.mean(axis=1, keepdims=True)
    return measurements - translation, translation


def factorization_method(points_im1, points_im2):
    # factor centered measurements into affine motion and 3D structure
    measurements, _ = build_measurement_matrix(points_im1, points_im2)
    u, values, vt = np.linalg.svd(measurements, full_matrices=False)
    # Split the retained singular values evenly between motion and structure.
    root_values = np.sqrt(values[:3])
    motion = u[:, :3] * root_values
    structure = root_values[:, None] * vt[:3]
    return structure, motion
