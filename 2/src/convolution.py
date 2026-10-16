import numpy as np

def zero_pad(image, padding):
    """Add zero borders: ((top, bottom), (left, right))."""
    return np.pad(image, padding, mode="constant", constant_values=0)

def _prepare(image, kernel):
    # Use floats to preserve weighted sums.
    image = np.asarray(image, dtype=np.float64)
    kernel = np.asarray(kernel, dtype=np.float64)
    kh, kw = kernel.shape
    # Half-kernel padding keeps the output size unchanged for odd kernels.
    padding = ((kh // 2, kh // 2), (kw // 2, kw // 2))
    padded = zero_pad(image, padding)
    output = np.zeros((padded.shape[0] - kh + 1, padded.shape[1] - kw + 1))
    # Flip both axes for convolution, rather than correlation.
    return padded, kernel[::-1, ::-1], output

def convolve_four_loops(image, kernel):
    """Convolve using four loops."""
    padded, flipped, output = _prepare(image, kernel)
    kh, kw = flipped.shape
    # Visit each output pixel.
    for y in range(output.shape[0]):
        for x in range(output.shape[1]):
            # Accumulate its weighted neighborhood.
            for ky in range(kh):
                for kx in range(kw):
                    output[y, x] += padded[y + ky, x + kx] * flipped[ky, kx]
    return output

def convolve_two_loops(image, kernel):
    """Convolve using two loops and NumPy window sums."""
    padded, flipped, output = _prepare(image, kernel)
    kh, kw = flipped.shape
    for y in range(output.shape[0]):
        for x in range(output.shape[1]):
            # NumPy replaces the two kernel loops.
            window = padded[y:y + kh, x:x + kw]
            output[y, x] = np.sum(window * flipped)
    return output