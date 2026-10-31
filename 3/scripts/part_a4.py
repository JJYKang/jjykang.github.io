"""Generate the manual clinic mosaic and direct-overlay comparison."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
from PIL import Image
from homography import computeH
from mosaicing import create_mosaic
from part_a2 import POINTS_1, POINTS_2

OUT = ROOT / "media" / "part_a" / "a4"
MAX_DIMENSION = 1024
DISPLAY_CROP = (0, 450, 2250, 1180)  # left, top, right, bottom on the mosaic canvas


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    images, scales = [], []
    for index in (1, 2):
        with Image.open(ROOT / "data" / f"mosaic2-{index}.jpg") as source:
            source = source.convert("RGB")
            original_width, original_height = source.size
            source.thumbnail((MAX_DIMENSION, MAX_DIMENSION), Image.Resampling.LANCZOS)
            sx, sy = source.width / original_width, source.height / original_height
            # Match the pixel-center convention of Pillow's resize operation.
            scales.append(np.array([[sx, 0, (sx - 1) / 2], [0, sy, (sy - 1) / 2], [0, 0, 1]]))
            images.append(np.asarray(source, dtype=float) / 255)
    # Reuse A.2's manual correspondences and keep image 1 as the reference.
    h12 = computeH(POINTS_1, POINTS_2)
    h21 = scales[0] @ np.linalg.inv(h12) @ np.linalg.inv(scales[1])
    transforms = [np.eye(3), h21]
    blended, direct, valid, origin = create_mosaic(images, transforms)
    left, top, right, bottom = DISPLAY_CROP
    if not valid[top:bottom, left:right].all():
        raise ValueError("Display crop contains empty canvas pixels")
    for name, result in [("mosaic", blended), ("direct", direct)]:
        rgba = np.dstack((np.round(np.clip(result, 0, 1) * 255).astype(np.uint8), valid.astype(np.uint8) * 255))
        output = Image.fromarray(rgba)
        output.save(OUT / f"{name}.png")
        output.crop(DISPLAY_CROP).save(OUT / f"{name}_crop.png")
    data = {"sources": ["mosaic2-1.jpg", "mosaic2-2.jpg"], "reference": "mosaic2-1.jpg",
            "input_shapes": [list(image.shape[:2]) for image in images],
            "output_shape": list(blended.shape[:2]), "origin": [int(v) for v in origin],
            "homographies": [h.tolist() for h in transforms], "display_crop": list(DISPLAY_CROP)}
    (OUT / "results.json").write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {blended.shape[1]}x{blended.shape[0]} clinic mosaic to {OUT}")


if __name__ == "__main__":
    main()
