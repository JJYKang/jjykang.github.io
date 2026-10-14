"""Generate Part 1.3 Gaussian, DoG, and orientation results."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
from PIL import Image
from filters import (gaussian_kernel, derivative_gaussian_kernels, gaussian_derivatives,
                     gradient_magnitude, binarize_edges, orientation_rgb)

SIZE = 9
SIGMA = 2.0
THRESHOLD = 0.08


def main():
    # Load grayscale intensities in [0, 1].
    with Image.open(ROOT / "data" / "cameraman.png") as source:
        image = np.asarray(source.convert("L"), dtype=np.float64) / 255
    # Build the kernels and compute both filtering methods.
    gaussian = gaussian_kernel(SIZE, SIGMA)
    dog_x, dog_y = derivative_gaussian_kernels(gaussian)
    blurred, dx, dy, direct_dx, direct_dy = gaussian_derivatives(image, gaussian)
    # Form gradient magnitudes and binary edge maps.
    magnitude = gradient_magnitude(dx, dy)
    direct_magnitude = gradient_magnitude(direct_dx, direct_dy)
    edges = binarize_edges(magnitude, THRESHOLD)
    direct_edges = binarize_edges(direct_magnitude, THRESHOLD)
    # Measure equivalence before display scaling.
    errors = {"dx": float(np.max(np.abs(dx - direct_dx))),
              "dy": float(np.max(np.abs(dy - direct_dy))),
              "magnitude": float(np.max(np.abs(magnitude - direct_magnitude))),
              "different_edge_pixels": int(np.count_nonzero(edges != direct_edges))}
    out = ROOT / "media" / "part1" / "1_3"
    out.mkdir(parents=True, exist_ok=True)
    # Use matching display scales for both methods and both DoG kernels.
    dog_scale = max(np.max(np.abs(dog_x)), np.max(np.abs(dog_y)))
    displays = {"blurred": blurred, "dx": 0.5 + dx / 2, "dy": 0.5 + dy / 2,
                "magnitude": magnitude / np.sqrt(2), "edges": edges.astype(float),
                "dog_dx": 0.5 + direct_dx / 2, "dog_dy": 0.5 + direct_dy / 2,
                "dog_magnitude": direct_magnitude / np.sqrt(2),
                "dog_edges": direct_edges.astype(float),
                "gaussian_kernel": gaussian / gaussian.max(),
                "dog_kernel_x": 0.5 + dog_x / (2 * dog_scale),
                "dog_kernel_y": 0.5 + dog_y / (2 * dog_scale)}
    for name, display in displays.items():
        result = Image.fromarray(np.round(np.clip(display, 0, 1) * 255).astype(np.uint8))
        if "kernel" in name:
            # Enlarge kernel cells without interpolation.
            result = result.resize((result.width * 32, result.height * 32), Image.Resampling.NEAREST)
        result.save(out / f"{name}.png")
    # Color the smoothed gradients by direction and strength.
    colors = orientation_rgb(direct_dx, direct_dy)
    Image.fromarray(np.round(colors * 255).astype(np.uint8)).save(out / "orientations.png")
    # Build a direction key using unit vectors and the same color mapping.
    y, x = np.mgrid[-128:129, -128:129]
    radius = np.sqrt(x ** 2 + y ** 2)
    unit_x = np.divide(x, radius, out=np.zeros_like(radius), where=radius > 0)
    unit_y = np.divide(y, radius, out=np.zeros_like(radius), where=radius > 0)
    key = orientation_rgb(unit_x, unit_y, value_scale=1)
    key[(radius > 120) | (radius < 45)] = 0  # Show only the ring.
    Image.fromarray(np.round(key * 255).astype(np.uint8)).save(out / "orientation_key.png")
    # Export parameters and comparison errors for the webpage.
    data = {"size": SIZE, "sigma": SIGMA, "threshold": THRESHOLD, "errors": errors}
    (out / "results.js").write_text("window.dogData = " + json.dumps(data) + ";\n", encoding="utf-8")
    print(f"Saved Part 1.3 and orientation results to {out}")
    print(f"DoG comparison: {errors}")


if __name__ == "__main__":
    main()
