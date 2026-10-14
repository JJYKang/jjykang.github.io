"""Generate Part 1.2 derivatives, magnitude, and binary edges."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
from PIL import Image
from filters import finite_differences, gradient_magnitude, binarize_edges

THRESHOLD = 0.2


def main():
    # Load grayscale intensities in [0, 1].
    with Image.open(ROOT / "data" / "cameraman.png") as source:
        image = np.asarray(source.convert("L"), dtype=np.float64) / 255
    # Differentiate, combine directions, then threshold the magnitude.
    dx, dy = finite_differences(image)
    magnitude = gradient_magnitude(dx, dy)
    edges = binarize_edges(magnitude, THRESHOLD)
    out = ROOT / "media" / "part1" / "1_2"
    out.mkdir(parents=True, exist_ok=True)
    # Scale only for display; thresholding uses the raw magnitude.
    displays = {"original": image, "dx": 0.5 + dx / 2, "dy": 0.5 + dy / 2,
                "magnitude": magnitude / np.sqrt(2), "edges": edges.astype(float)}
    for name, display in displays.items():
        pixels = np.round(np.clip(display, 0, 1) * 255).astype(np.uint8)
        Image.fromarray(pixels).save(out / f"{name}.png")
    # Keep webpage labels consistent with the generated images.
    (out / "results.js").write_text(
        "window.finiteDifferenceData = " + json.dumps({"threshold": THRESHOLD}) + ";\n",
        encoding="utf-8")
    print(f"Saved Part 1.2 results to {out}")


if __name__ == "__main__":
    main()
