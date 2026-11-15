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
