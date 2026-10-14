"""Finite differences and Gaussian derivatives for Parts 1.2 and 1.3."""
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


def gaussian_kernel(size=9, sigma=2.0):
    """Construct a normalized 2D Gaussian from OpenCV's 1D kernel."""
    import cv2
    if size < 1 or size % 2 == 0 or not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("size must be positive and odd; sigma must be positive")
    g = cv2.getGaussianKernel(size, sigma)
    return g @ g.T


def derivative_gaussian_kernels(gaussian):
    """Full convolution retains the complete support of each DoG kernel."""
    dog_x = convolve2d(gaussian, DX, mode="full")
    dog_y = convolve2d(gaussian, DY, mode="full")
    return dog_x, dog_y


def gaussian_derivatives(image, gaussian):
    """Compare blur-then-differentiate with one DoG convolution per direction.

    Extend the original image once, far enough for BOTH filters. Crop only
    after filtering, so intermediate boundary handling does not break
    convolution associativity at the image frame.
    """
    py, px = gaussian.shape[0] // 2 + 1, gaussian.shape[1] // 2 + 1
    padded = np.pad(image, ((py, py), (px, px)), mode="symmetric")
    crop = (slice(py, -py), slice(px, -px))
    blurred = convolve2d(padded, gaussian, mode="same")
    dx = convolve2d(blurred, DX, mode="same")[crop]
    dy = convolve2d(blurred, DY, mode="same")[crop]
    dog_x, dog_y = derivative_gaussian_kernels(gaussian)
    direct_dx = convolve2d(padded, dog_x, mode="same")[crop]
    direct_dy = convolve2d(padded, dog_y, mode="same")[crop]
    return blurred[crop], dx, dy, direct_dx, direct_dy
