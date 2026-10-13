"""Generate the cameraman derivatives, gradient magnitude, and edge image.

Run from 2/: python scripts/part1_2.py --threshold 0.2
"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
from PIL import Image
from filters import finite_differences, gradient_magnitude, binarize_edges


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, default=ROOT / "data" / "cameraman.png")
    parser.add_argument("--threshold", type=float, default=0.2)
    args = parser.parse_args()
    if not np.isfinite(args.threshold) or args.threshold < 0:
        parser.error("--threshold must be finite and nonnegative")
    with Image.open(args.image) as source:
        image = np.asarray(source.convert("L"), dtype=np.float64) / 255
    dx, dy = finite_differences(image)
    magnitude = gradient_magnitude(dx, dy)
    edges = binarize_edges(magnitude, args.threshold)
    out = ROOT / "media" / "part1" / "1_2"
    out.mkdir(parents=True, exist_ok=True)
    # Fixed display scales; thresholding always uses the raw floating-point magnitude.
    displays = {"original": image, "dx": 0.5 + dx / 2, "dy": 0.5 + dy / 2,
                "magnitude": magnitude / np.sqrt(2), "edges": edges.astype(float)}
    for name, display in displays.items():
        pixels = np.round(np.clip(display, 0, 1) * 255).astype(np.uint8)
        Image.fromarray(pixels).save(out / f"{name}.png")
    (out / "results.js").write_text(
        "window.finiteDifferenceData = " + json.dumps({"threshold": args.threshold}) + ";\n",
        encoding="utf-8")
    print(f"Saved Part 1.2 results to {out}; threshold = {args.threshold:g}")


if __name__ == "__main__":
    main()
