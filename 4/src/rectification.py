"""Epipoles and homographies for stereo image rectification."""
import numpy as np


def compute_epipole(F):
    """Return the epipole e satisfying F.T @ e = 0.

    For p2.T @ F @ p1 = 0, this is the epipole in Image 2.
    Pass F.T for Image 1. Finite epipoles have the form (x, y, 1);
    epipoles at infinity have a unit direction vector (x, y, 0).
    """
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
    """Return H mapping epipole e to horizontal infinity, using the image center.

    Compose T_inverse @ G @ R @ T: center the image, rotate the epipole
    onto the x-axis, send it to infinity, then restore the image center.
    Epipoles already at infinity require rotation only in the centered frame.
    """
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
