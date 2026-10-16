"""Multiresolution blending with Gaussian mask and Laplacian image stacks."""
from stacks import gaussian_stack, laplacian_stack


def multiresolution_blend(a, b, mask, levels=6, sigma=2):
    """Blend each frequency band, then sum the bands and residual."""
    lap_a = laplacian_stack(gaussian_stack(a, levels, sigma))
    lap_b = laplacian_stack(gaussian_stack(b, levels, sigma))
    # Broadcast the same spatial weights across RGB channels.
    masks = gaussian_stack(mask, levels, sigma)[:, :, :, None]
    weighted_a = masks * lap_a
    weighted_b = (1 - masks) * lap_b
    result = (weighted_a + weighted_b).sum(axis=0)
    return result, weighted_a, weighted_b
