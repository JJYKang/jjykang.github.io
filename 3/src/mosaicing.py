"""Shared-canvas image mosaics with feathered weighted averaging."""
import numpy as np
from warping import warp_bounds, warpImageBilinear


def feather_weights(shape):
    """Linear distance to the nearest source-image border, with positive edges."""
    height, width = shape[:2]
    y, x = np.mgrid[:height, :width]
    distance = np.minimum(np.minimum(x + 1, width - x), np.minimum(y + 1, height - y))
    return distance / distance.max()


def create_mosaic(images, homographies):
    """Map RGB images to a shared projection and return blended and direct results.

    Each homography maps one source to the reference coordinate system.
    Returns (blended, direct, valid_mask, origin). The direct comparison lets
    later images overwrite earlier images wherever their validity masks overlap.
    """
    if len(images) != len(homographies) or not images:
        raise ValueError("Provide one homography per image")
    bounds = [warp_bounds(image.shape, h) for image, h in zip(images, homographies)]
    left = min(origin[0] for shape, origin in bounds)
    top = min(origin[1] for shape, origin in bounds)
    right = max(origin[0] + shape[1] for shape, origin in bounds)
    bottom = max(origin[1] + shape[0] for shape, origin in bounds)
    shape, origin = (bottom - top, right - left), (left, top)
    weighted = np.zeros(shape + (3,), dtype=float)
    total_weight = np.zeros(shape, dtype=float)
    direct = np.zeros_like(weighted)
    valid = np.zeros(shape, dtype=bool)
    for image, h in zip(images, homographies):
        # Warp color and its source-space feather weight with the same mapping.
        packed = np.dstack((image, feather_weights(image.shape)))
        warped, mask, _ = warpImageBilinear(packed, h, shape, origin)
        weight = warped[..., 3]
        weighted += warped[..., :3] * weight[..., None]
        total_weight += weight
        direct[mask] = warped[..., :3][mask]
        valid |= mask
    blended = np.divide(weighted, total_weight[..., None],
                        out=np.zeros_like(weighted), where=total_weight[..., None] > 0)
    return blended, direct, valid, origin
