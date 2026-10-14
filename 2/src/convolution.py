import numpy as np


def zero_pad(image, padding):
    """Pad a 2D array with ((top, bottom), (left, right)) zeros."""
    return np.pad(image, padding, mode="constant", constant_values=0)


def _prepare(image, kernel, mode):
    image = np.asarray(image, dtype=np.float64)
    kernel = np.asarray(kernel, dtype=np.float64)
    if image.ndim != 2 or kernel.ndim != 2 or not image.size or not kernel.size:
        raise ValueError("image and kernel must be nonempty 2D arrays")
    if not np.isfinite(image).all() or not np.isfinite(kernel).all():
        raise ValueError("image and kernel must contain finite values")
    kh, kw = kernel.shape
    if mode == "same":
        # Even kernels need the extra zero before the image to match SciPy.
        padding = ((kh // 2, (kh - 1) // 2), (kw // 2, (kw - 1) // 2))
    elif mode == "full":
        padding = ((kh - 1, kh - 1), (kw - 1, kw - 1))
    elif mode == "valid":
        if image.shape[0] < kh or image.shape[1] < kw:
            raise ValueError("valid mode requires image to be at least as large as kernel")
        padding = ((0, 0), (0, 0))
    else:
        raise ValueError("mode must be 'same', 'full', or 'valid'")
    padded = zero_pad(image, padding)
    output = np.zeros((padded.shape[0] - kh + 1, padded.shape[1] - kw + 1))
    # Flip BOTH axes: without this, the operation is cross-correlation.
    return padded, kernel[::-1, ::-1], output


def convolve_four_loops(image, kernel, mode="same"):
    """Explicit loops over output rows/columns and kernel rows/columns."""
    padded, flipped, output = _prepare(image, kernel, mode)
    kh, kw = flipped.shape
    for y in range(output.shape[0]):
        for x in range(output.shape[1]):
            for ky in range(kh):
                for kx in range(kw):
                    output[y, x] += padded[y + ky, x + kx] * flipped[ky, kx]
    return output


def convolve_two_loops(image, kernel, mode="same"):
    """Loop over output pixels; NumPy sums each elementwise window product."""
    padded, flipped, output = _prepare(image, kernel, mode)
    kh, kw = flipped.shape
    for y in range(output.shape[0]):
        for x in range(output.shape[1]):
            window = padded[y:y + kh, x:x + kw]
            output[y, x] = np.sum(window * flipped)
    return output


def box_filter(size=9):
    """Normalized square averaging filter (9 by 9 for Part 1.1)."""
    if not isinstance(size, (int, np.integer)) or size < 1:
        raise ValueError("size must be a positive integer")
    return np.ones((size, size), dtype=np.float64) / size ** 2


DX = np.array([[1.0, 0.0, -1.0]])
DY = DX.T.copy()
