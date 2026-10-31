"""Generate Harris and ANMS corner overlays for the three house photos."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
from PIL import Image, ImageDraw
from harris import get_harris_corners
from features import anms

OUT = ROOT / "media" / "part_b" / "b1"
MAX_DIMENSION = 1024
EDGE_DISCARD = 20
COUNT = 500
C_ROBUST = 0.9


def corner_overlay(image, points, path):
    marked = image.copy()
    draw = ImageDraw.Draw(marked)
    for x, y in points:
        draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill="#ffdf00", outline="black", width=1)
    marked.save(path, quality=95)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    records = []
    for index in (1, 2, 3):
        filename = f"mosaic3-{index}.jpg"
        with Image.open(ROOT / "data" / filename) as source:
            image = source.convert("RGB")
            image.thumbnail((MAX_DIMENSION, MAX_DIMENSION), Image.Resampling.LANCZOS)
        gray = np.asarray(image, dtype=float) @ np.array([0.299, 0.587, 0.114]) / 255
        response, coords = get_harris_corners(gray, edge_discard=EDGE_DISCARD)
        # Supplied Harris returns 2-by-n (y, x); shared geometry uses n-by-2 (x, y).
        strengths = response[coords[0], coords[1]]
        positive = np.isfinite(strengths) & (strengths > 0)
        points = coords.T[positive, ::-1]
        strengths = strengths[positive]
        selected, radii = anms(points, strengths, COUNT, C_ROBUST)
        corner_overlay(image, points, OUT / f"image_{index}_harris.jpg")
        corner_overlay(image, points[selected], OUT / f"image_{index}_anms.jpg")
        # Save the exact detections for descriptor extraction in B.2.
        np.savez_compressed(OUT / f"image_{index}_corners.npz", points=points,
                            strengths=strengths, radii=radii, selected=selected,
                            image_shape=np.array(gray.shape))
        record = {"image": index, "source": filename, "shape": list(gray.shape),
                  "harris_count": len(points), "anms_count": len(selected)}
        records.append(record)
        print(f"{filename}: {len(points)} Harris corners -> {len(selected)} ANMS corners")
    data = {"max_dimension": MAX_DIMENSION, "edge_discard": EDGE_DISCARD,
            "count": COUNT, "c_robust": C_ROBUST, "records": records}
    (OUT / "results.json").write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    (OUT / "results.js").write_text("window.cornerData = " + json.dumps(data) + ";\n", encoding="utf-8")


if __name__ == "__main__":
    main()
