import numpy as np
from fundamental import _prepare_points


def compute_epipole(F):
    # find the Image 2 epipole; pass F.T for Image 1
    f = np.asarray(F, dtype=np.float64)
    if f.shape != (3, 3) or not np.isfinite(f).all() or not np.any(f):
        raise ValueError("Provide a finite, nonzero 3-by-3 fundamental matrix")
    f = f / np.abs(f).max()
    u, values, _ = np.linalg.svd(f)
    tolerance = 3 * np.finfo(float).eps * values[0]
    if values[1] <= tolerance or values[2] > 1e-10 * values[0]:
        raise ValueError("The fundamental matrix must have rank two")

    # The last left singular vector is the right null vector of F.T.
    epipole = u[:, -1]
    if abs(epipole[2]) > 16 * np.finfo(float).eps:
        return epipole / epipole[2]

    # Parallel viewing geometry puts the epipole at infinity.
    epipole[2] = 0
    epipole /= np.linalg.norm(epipole[:2])
    if epipole[np.argmax(np.abs(epipole))] < 0:
        epipole = -epipole
    return epipole


def compute_H(e, im):
    # center the image, rotate the epipole onto x, then send it to infinity
    epipole = np.asarray(e, dtype=np.float64)
    if (epipole.shape != (3,) or not np.isfinite(epipole).all()
            or not np.any(epipole)):
        raise ValueError("Provide a finite, nonzero homogeneous epipole")
    image = np.asarray(im)
    if image.ndim < 2 or min(image.shape[:2]) == 0:
        raise ValueError("Provide an image with nonzero height and width")
    height, width = image.shape[:2]
    epipole = epipole / np.abs(epipole).max()
    if epipole[2] < 0 or (epipole[2] == 0
                          and epipole[np.argmax(np.abs(epipole))] < 0):
        epipole = -epipole
    t = np.array([[1., 0, -width / 2],
                  [0, 1, -height / 2], [0, 0, 1]])
    centered = t @ epipole
    radius = np.hypot(centered[0], centered[1])
    if radius <= np.finfo(float).eps * np.linalg.norm(centered):
        raise ValueError("The image-centered construction requires an off-center epipole")

    # Preserve the epipole's left/right direction, as in the handout.
    if abs(centered[0]) <= 16 * np.finfo(float).eps * radius:
        centered[0] = 0
    alpha = 1 if centered[0] >= 0 else -1
    cosine, sine = alpha * centered[:2] / radius
    r = np.array([[cosine, sine, 0], [-sine, cosine, 0], [0, 0, 1]])
    rotated = r @ centered
    g = np.eye(3)
    g[2, 0] = -rotated[2] / rotated[0]
    t_inverse = np.array([[1., 0, width / 2],
                          [0, 1, height / 2], [0, 0, 1]])
    return t_inverse @ g @ r @ t


def compute_matching_homographies(e2, F, im2, points1, points2):
    # rectify both views and fit an x correction that preserves scanlines
    first, second = _prepare_points(points1, points2, min_points=3)
    h2 = compute_H(e2, im2)
    epipole = np.asarray(e2, dtype=np.float64)
    epipole = epipole / np.abs(epipole).max()
    f = np.asarray(F, dtype=np.float64)
    if f.shape != (3, 3) or not np.isfinite(f).all() or not np.any(f):
        raise ValueError("Provide a finite, nonzero 3-by-3 fundamental matrix")
    f = f / np.abs(f).max()
    f = f / np.linalg.norm(f)
    if np.linalg.norm(f.T @ epipole) > 1e-10 * np.linalg.norm(epipole):
        raise ValueError("The second epipole must satisfy F.T @ e2 = 0")

    x, y, z = epipole
    cross = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    m = cross @ f + np.outer(epipole, np.ones(3))
    if np.linalg.matrix_rank(m) < 3:
        raise ValueError("The epipole and fundamental matrix give a singular mapping")
    base = h2 @ m
    mapped1 = first @ base.T
    mapped2 = second @ h2.T
    if np.any(mapped1[:, 2] == 0) or np.any(mapped2[:, 2] == 0):
        raise ValueError("A correspondence maps to infinity during rectification")
    mapped1 = mapped1 / mapped1[:, 2:3]
    mapped2 = mapped2 / mapped2[:, 2:3]

    # Fit a1*x1 + a2*y1 + a3 to x2 without changing the rectified scanlines.
    a, _, rank, _ = np.linalg.lstsq(mapped1, mapped2[:, 0], rcond=None)
    if rank < 3 or abs(a[0]) <= np.finfo(float).eps * np.linalg.norm(a):
        raise ValueError("Correspondences do not determine an invertible affine correction")
    affine = np.eye(3)
    affine[0] = a
    return affine @ base, h2
