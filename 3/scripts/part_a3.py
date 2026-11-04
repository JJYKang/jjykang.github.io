"""Generate two rectifications and compare interpolation quality and runtime."""
import json
from pathlib import Path
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
from PIL import Image, ImageDraw
from homography import computeH
from warping import warpImageNearestNeighbor, warpImageBilinear

OUT = ROOT / "media" / "part_a" / "a3"
REPEATS = 5
# Corners in original images: top-left, top-right, bottom-right, bottom-left.
# Destination rectangles set the displayed width and height of each object.
EXAMPLES = [
    ("sign", "rectification-1.jpg", [[740, 274], [1583, 459], [1583, 1029], [761, 1246]],
     (600, 600), (90, 435, 210, 515)),
    ("window", "rectification-2.jpg", [[489, 516], [1512, 191], [1529, 1278], [470, 1051]],
     (600, 900), (150, 200, 270, 280)),
]


def save_image(image, path, mask=None):
    pixels = np.round(np.clip(image, 0, 1) * 255).astype(np.uint8)
    if mask is not None:
        pixels = np.dstack((pixels, mask.astype(np.uint8) * 255))
    Image.fromarray(pixels).save(path)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    records = []
    for name, filename, corners, shape, detail in EXAMPLES:
        corners = np.array(corners, dtype=float)
        with Image.open(ROOT / "data" / filename) as source:
            source = source.convert("RGB")
            image = np.asarray(source, dtype=np.float64) / 255
            marked = source.copy()
        draw = ImageDraw.Draw(marked)
        draw.line([tuple(p) for p in corners] + [tuple(corners[0])], fill="#ffdf00", width=3)
        for x, y in corners:
            draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill="#ffdf00")
        # Show a source crop around the selected quadrilateral.
        low = np.maximum(np.floor(corners.min(axis=0)).astype(int) - 35, 0)
        high = np.minimum(np.ceil(corners.max(axis=0)).astype(int) + 36, source.size)
        marked.crop((*low, *high)).save(OUT / f"{name}_source.png")
        height, width = shape
        target = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]])
        h = computeH(corners, target)
        record = {"name": name, "source": filename, "corners": corners.tolist(),
                  "output_shape": list(shape), "homography": h.tolist(), "milliseconds": {}}
        for method, function in [("nearest", warpImageNearestNeighbor), ("bilinear", warpImageBilinear)]:
            function(image, h, output_shape=shape)  # Untimed warm-up.
            times = []
            for _ in range(REPEATS):
                start = perf_counter()
                result, mask, _ = function(image, h, output_shape=shape)
                times.append((perf_counter() - start) * 1000)
            record["milliseconds"][method] = float(np.median(times))
            save_image(result, OUT / f"{name}_{method}.png", mask)
            left, top, right, bottom = detail
            crop = Image.fromarray(np.round(np.clip(result[top:bottom, left:right], 0, 1) * 255).astype(np.uint8))
            # Enlarge both detail crops equally to display individual output pixels.
            crop.resize((crop.width * 4, crop.height * 4), Image.Resampling.NEAREST).save(OUT / f"{name}_{method}_detail.png")
        records.append(record)
        print(name, record["milliseconds"])
    data = {"repeats": REPEATS, "records": records}
    (OUT / "results.json").write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    (OUT / "results.js").write_text("window.warpingData = " + json.dumps(data) + ";\n", encoding="utf-8")


if __name__ == "__main__":
    main()
