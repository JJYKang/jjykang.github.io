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


def gradient_orientation(dx, dy):
    """Angles in [0, 2*pi), computed without built-in angle functions.

    Reduce to the first octant, approximate atan by its alternating series,
    then restore the octant and quadrant. Image y increases downwards.
    """
    ax, ay = np.abs(dx), np.abs(dy)
    largest = np.maximum(ax, ay)
    ratio = np.divide(np.minimum(ax, ay), largest,
                      out=np.zeros_like(largest), where=largest > 0)
    # atan(r) = pi/4 + atan((r-1)/(r+1)) for larger ratios.
    # This limits the series argument to abs(z) <= sqrt(2)-1.
    reduced = ratio > np.sqrt(2) - 1
    z = np.where(reduced, (ratio - 1) / (ratio + 1), ratio)
    angle = np.zeros_like(z)
    for n in range(12):
        angle += (-1) ** n * z ** (2 * n + 1) / (2 * n + 1)
    angle = np.where(reduced, np.pi / 4 + angle, angle)
    angle = np.where(ay > ax, np.pi / 2 - angle, angle)
    angle = np.where(dx < 0, np.pi - angle, angle)
    angle = np.where(dy < 0, 2 * np.pi - angle, angle)
    return np.where(largest > 0, angle % (2 * np.pi), 0)


def orientation_rgb(dx, dy, value_scale=None):
    """HSV: hue = orientation, saturation = 1, value = gradient strength."""
    import cv2
    angle = gradient_orientation(dx, dy)
    magnitude = gradient_magnitude(dx, dy)
    if value_scale is None:
        value_scale = np.percentile(magnitude, 99)
    value = np.clip(magnitude / value_scale, 0, 1) if value_scale > 0 else np.zeros_like(magnitude)
    hsv = np.stack((angle * 180 / np.pi, np.ones_like(angle), value), axis=-1)
    # OpenCV float HSV uses degrees for H and [0,1] for S and V.
    return cv2.cvtColor(hsv.astype(np.float32), cv2.COLOR_HSV2RGB)
