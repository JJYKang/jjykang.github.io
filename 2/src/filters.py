"""Finite differences and gradient-based edges for Part 1.2."""
import numpy as np
from scipy.signal import convolve2d
from convolution import DX, DY


def finite_differences(image):
    """Compute signed x/y differences with symmetric boundary extension."""
    dx = convolve2d(image, DX, mode="same", boundary="symm")
    dy = convolve2d(image, DY, mode="same", boundary="symm")
    return dx, dy


def gradient_magnitude(dx, dy):
    return np.sqrt(dx ** 2 + dy ** 2)


def binarize_edges(magnitude, threshold):
    return magnitude > threshold
