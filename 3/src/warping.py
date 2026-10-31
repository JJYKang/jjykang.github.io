"""Inverse projective warping with NumPy interpolation."""
import numpy as np


def warp_bounds(image_shape, h):
    """Return ((height, width), (left, top)) enclosing transformed pixel centers."""
    height, width = image_shape[:2]
    corners = np.array([[0, 0, 1], [width - 1, 0, 1],
                        [width - 1, height - 1, 1], [0, height - 1, 1]]) @ h.T
    denominator = corners[:, 2]
    if denominator.min() <= 0 <= denominator.max():
        raise ValueError("Image crosses the projective horizon; specify a finite output canvas")
    xy = corners[:, :2] / denominator[:, None]
    low = np.floor(xy.min(axis=0)).astype(int)
    high = np.ceil(xy.max(axis=0)).astype(int)
    return (int(high[1] - low[1] + 1), int(high[0] - low[0] + 1)), tuple(low)


def _warp(image, h, method, output_shape, origin):
    image = np.asarray(image, dtype=np.float64)
    h = np.asarray(h, dtype=np.float64)
    if image.ndim not in (2, 3) or min(image.shape[:2]) == 0:
        raise ValueError("Provide a nonempty grayscale or multichannel image")
    if h.shape != (3, 3) or not np.isfinite(h).all():
        raise ValueError("H must be a finite 3-by-3 matrix")
    if output_shape is None:
        if origin is not None:
            raise ValueError("Specify output_shape when specifying origin")
        output_shape, origin = warp_bounds(image.shape, h)
    elif origin is None:
        origin = (0, 0)
    height, width = output_shape
    if height <= 0 or width <= 0:
        raise ValueError("Output dimensions must be positive")
    inverse = np.linalg.inv(h)
    result = np.zeros((height, width) + image.shape[2:], dtype=np.float64)
    mask = np.zeros((height, width), dtype=bool)
    # Process row blocks to limit temporary coordinate-array memory.
    for top in range(0, height, 128):
        bottom = min(top + 128, height)
        y, x = np.mgrid[top:bottom, :width].astype(float)
        x += origin[0]
        y += origin[1]
        denominator = inverse[2, 0] * x + inverse[2, 1] * y + inverse[2, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            sx = (inverse[0, 0] * x + inverse[0, 1] * y + inverse[0, 2]) / denominator
            sy = (inverse[1, 0] * x + inverse[1, 1] * y + inverse[1, 2]) / denominator
        # A small tolerance preserves boundary pixels after matrix roundoff.
        valid = (np.isfinite(sx) & np.isfinite(sy)
                 & (sx >= -1e-8) & (sx <= image.shape[1] - 1 + 1e-8)
                 & (sy >= -1e-8) & (sy <= image.shape[0] - 1 + 1e-8))
        sx = np.clip(np.where(valid, sx, 0), 0, image.shape[1] - 1)
        sy = np.clip(np.where(valid, sy, 0), 0, image.shape[0] - 1)
        if method == "nearest":
            values = image[np.floor(sy + 0.5).astype(int), np.floor(sx + 0.5).astype(int)]
        else:
            x0, y0 = np.floor(sx).astype(int), np.floor(sy).astype(int)
            x1 = np.minimum(x0 + 1, image.shape[1] - 1)
            y1 = np.minimum(y0 + 1, image.shape[0] - 1)
            dx, dy = sx - x0, sy - y0
            if image.ndim == 3:
                dx, dy = dx[..., None], dy[..., None]
            values = ((1 - dx) * (1 - dy) * image[y0, x0]
                      + dx * (1 - dy) * image[y0, x1]
                      + (1 - dx) * dy * image[y1, x0]
                      + dx * dy * image[y1, x1])
        block = result[top:bottom]
        block[valid] = values[valid]
        mask[top:bottom] = valid
    return result, mask, origin


def warpImageNearestNeighbor(im, H, output_shape=None, origin=None):
    """Return (warped, valid_mask, (left, top)) using rounded source coordinates.

    H maps source to destination. output_shape is (height, width); origin is
    the destination coordinate of output pixel (0, 0). By default the canvas
    encloses the transformed image. Pixel centers have integer coordinates.
    """
    return _warp(im, H, "nearest", output_shape, origin)


def warpImageBilinear(im, H, output_shape=None, origin=None):
    """Return (warped, valid_mask, origin) using four weighted source neighbors.

    Uses the same canvas and coordinate conventions as warpImageNearestNeighbor.
    """
    return _warp(im, H, "bilinear", output_shape, origin)
