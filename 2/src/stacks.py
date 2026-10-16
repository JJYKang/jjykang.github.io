"""Gaussian and Laplacian stacks without downsampling."""
import cv2
import numpy as np
from scipy.ndimage import convolve1d


def gaussian_stack(image, levels=6, sigma=2):
    """Keep the original and progressively smoothed images at full size."""
    stack = [np.asarray(image, dtype=np.float64)]
    for level in range(1, levels):
        # Double the blur scale at each level to separate frequency bands.
        scale = sigma * 2 ** (level - 1)
        radius = int(np.ceil(3 * scale))
        kernel = cv2.getGaussianKernel(2 * radius + 1, scale).ravel()
        # Separable spatial convolution leaves color channels independent.
        blurred = convolve1d(stack[-1], kernel, axis=0, mode="reflect")
        blurred = convolve1d(blurred, kernel, axis=1, mode="reflect")
        stack.append(blurred)
    return np.stack(stack)


def laplacian_stack(gaussian):
    """Store adjacent Gaussian differences and the final low-pass residual."""
    bands = [gaussian[i] - gaussian[i + 1] for i in range(len(gaussian) - 1)]
    # The residual is needed to reconstruct the original by summation.
    return np.stack(bands + [gaussian[-1]])
