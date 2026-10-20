"""Spatial filters, gradients, and unsharp masking."""
import numpy as np
from scipy.signal import convolve2d

# Finite differences along columns and rows.
DX = np.array([[1.0, 0.0, -1.0]])
DY = DX.T.copy()


def finite_differences(image):
    """Compute signed x/y differences."""
    # Mirror the border to avoid artificial edges against black padding.
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
    g = cv2.getGaussianKernel(size, sigma)
    # The outer product gives a 2D kernel with weights summing to one.
    return g @ g.T


def derivative_gaussian_kernels(gaussian):
    """Combine smoothing and differentiation into DoG kernels."""
    # Full convolution keeps the outer kernel coefficients.
    dog_x = convolve2d(gaussian, DX, mode="full")
    dog_y = convolve2d(gaussian, DY, mode="full")
    return dog_x, dog_y


def gaussian_derivatives(image, gaussian):
    """Compute two-stage and single-DoG derivatives."""
    # Pad for both filters; crop only at the end to preserve equivalence.
    py, px = gaussian.shape[0] // 2 + 1, gaussian.shape[1] // 2 + 1
    padded = np.pad(image, ((py, py), (px, px)), mode="symmetric")
    crop = (slice(py, -py), slice(px, -px))
    # Two stages: smooth, then differentiate.
    blurred = convolve2d(padded, gaussian, mode="same")
    dx = convolve2d(blurred, DX, mode="same")[crop]
    dy = convolve2d(blurred, DY, mode="same")[crop]
    # One stage: apply each combined DoG kernel.
    dog_x, dog_y = derivative_gaussian_kernels(gaussian)
    direct_dx = convolve2d(padded, dog_x, mode="same")[crop]
    direct_dy = convolve2d(padded, dog_y, mode="same")[crop]
    return blurred[crop], dx, dy, direct_dx, direct_dy


def gradient_orientation(dx, dy):
    """Compute angles in [0, 2*pi) without built-in angle functions."""
    # Reduce to the first octant; leave zero gradients at zero.
    ax, ay = np.abs(dx), np.abs(dy)
    largest = np.maximum(ax, ay)
    ratio = np.divide(np.minimum(ax, ay), largest,
                      out=np.zeros_like(largest), where=largest > 0)
    # Approximate atan on [0, 1]; the correction improves a linear estimate.
    angle = ratio * (np.pi / 4 + 0.273 * (1 - ratio))
    # Restore the octant and quadrant.
    angle = np.where(ay > ax, np.pi / 2 - angle, angle)
    angle = np.where(dx < 0, np.pi - angle, angle)
    angle = np.where(dy < 0, 2 * np.pi - angle, angle)
    # Wrap a full turn back to zero.
    return np.where(largest > 0, angle % (2 * np.pi), 0)


def orientation_rgb(dx, dy, value_scale=None):
    """Map direction to hue and gradient strength to brightness."""
    import cv2
    angle = gradient_orientation(dx, dy)
    magnitude = gradient_magnitude(dx, dy)
    # Limit strong outliers when setting display brightness.
    if value_scale is None:
        value_scale = np.percentile(magnitude, 99)
    value = np.clip(magnitude / value_scale, 0, 1) if value_scale > 0 else np.zeros_like(magnitude)
    # Float HSV uses degrees for hue and [0, 1] for saturation/value.
    hsv = np.stack((angle * 180 / np.pi, np.ones_like(angle), value), axis=-1)
    return cv2.cvtColor(hsv.astype(np.float32), cv2.COLOR_HSV2RGB)


def filter_rgb(image, kernel):
    """Apply a spatial kernel independently to each color channel."""
    return np.stack([convolve2d(image[:, :, c], kernel, mode="same", boundary="symm")
                     for c in range(3)], axis=-1)


def unsharp_mask(image, gaussian, amount):
    """Sharpen with the single kernel (1 + amount) * impulse - amount * G."""
    kernel = -amount * gaussian.copy()
    # The center impulse preserves the original image contribution.
    kernel[kernel.shape[0] // 2, kernel.shape[1] // 2] += 1 + amount
    return filter_rgb(image, kernel)
