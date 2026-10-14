"""Generate Part 1.3 Gaussian smoothing and DoG results.

Run from 2/: python scripts/part1_3.py --size 9 --sigma 2 --threshold 0.08
"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
from PIL import Image
from filters import (gaussian_kernel, derivative_gaussian_kernels, gaussian_derivatives,
                     gradient_magnitude, binarize_edges)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, default=ROOT / "data" / "cameraman.png")
    parser.add_argument("--size", type=int, default=9)
    parser.add_argument("--sigma", type=float, default=2.0)
    parser.add_argument("--threshold", type=float, default=0.08)
    args = parser.parse_args()
    if not np.isfinite(args.threshold) or args.threshold < 0:
        parser.error("--threshold must be finite and nonnegative")
    with Image.open(args.image) as source:
        image = np.asarray(source.convert("L"), dtype=np.float64) / 255
    gaussian = gaussian_kernel(args.size, args.sigma)
    dog_x, dog_y = derivative_gaussian_kernels(gaussian)
    blurred, dx, dy, direct_dx, direct_dy = gaussian_derivatives(image, gaussian)
    magnitude = gradient_magnitude(dx, dy)
    direct_magnitude = gradient_magnitude(direct_dx, direct_dy)
    edges = binarize_edges(magnitude, args.threshold)
    direct_edges = binarize_edges(direct_magnitude, args.threshold)
    errors = {"dx": float(np.max(np.abs(dx - direct_dx))),
              "dy": float(np.max(np.abs(dy - direct_dy))),
              "magnitude": float(np.max(np.abs(magnitude - direct_magnitude))),
              "different_edge_pixels": int(np.count_nonzero(edges != direct_edges))}
    # Verification of equivalence is a required result for Part 1.3.
    out = ROOT / "media" / "part1" / "1_3"
    out.mkdir(parents=True, exist_ok=True)
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
            result = result.resize((result.width * 32, result.height * 32), Image.Resampling.NEAREST)
        result.save(out / f"{name}.png")
    data = {"size": args.size, "sigma": args.sigma, "threshold": args.threshold, "errors": errors}
    (out / "results.js").write_text("window.dogData = " + json.dumps(data) + ";\n", encoding="utf-8")
    print(json.dumps(data, indent=2))


if __name__ == "__main__":
    main()
