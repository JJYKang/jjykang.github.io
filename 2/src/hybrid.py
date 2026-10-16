"""Landmark alignment and hybrid-image filtering."""
import cv2
import numpy as np
from filters import gaussian_kernel, filter_rgb


def align_image(image, source_points, target_points, shape):
    """Match two landmarks using translation, rotation, and uniform scale."""
    def triangle(points):
        a, b = np.asarray(points, dtype=np.float32)
        v = b - a
        # A perpendicular third point constrains the transform to uniform scale.
        return np.float32([a, b, a + [-v[1], v[0]]])

    transform = cv2.getAffineTransform(triangle(source_points), triangle(target_points))
    h, w = shape[:2]
    aligned = cv2.warpAffine(image, transform, (w, h))
    mask = cv2.warpAffine(np.ones(image.shape[:2], dtype=np.uint8), transform,
                          (w, h), flags=cv2.INTER_NEAREST)
    return aligned, mask


def gaussian_blur(image, sigma):
    # Keep three standard deviations on either side of the Gaussian.
    size = 2 * int(np.ceil(3 * sigma)) + 1
    return filter_rgb(image, gaussian_kernel(size, sigma))


def hybrid_image(low_image, high_image, sigma_low, sigma_high):
    """Combine a low-pass image with the other's high-pass residual."""
    low = gaussian_blur(low_image, sigma_low)
    high = high_image - gaussian_blur(high_image, sigma_high)
    return low, high, low + high


def grayscale_rgb(image):
    # Repeat luminance across channels so color variants use the same pipeline.
    gray = image @ np.array([0.299, 0.587, 0.114])
    return np.repeat(gray[:, :, None], 3, axis=2)


def log_spectrum(image):
    gray = image @ np.array([0.299, 0.587, 0.114])
    # Center the DC component; log1p also handles zero-valued frequencies.
    return np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(gray))))
