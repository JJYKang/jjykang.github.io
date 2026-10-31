"""Generate manual correspondences and the recovered clinic homography."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from homography import computeH

# Manually selected (x, y) landmarks in the original 2048x1536 images.
# Each row pairs the same landmark in mosaic2-1.jpg and mosaic2-2.jpg.
POINTS_1 = np.array([
    [1232, 373], [1523, 358],
    [1067, 605], [1185, 601], [1070, 694], [1183, 692],
    [1482, 598], [1633, 594], [1484, 691], [1631, 691],
    [1634, 793], [1916, 905],
], dtype=float)
POINTS_2 = np.array([
    [403, 332], [709, 358],
    [201, 574], [352, 579], [204, 674], [350, 679],
    [681, 595], [812, 600], [683, 684], [811, 687],
    [811, 779], [1037, 870],
], dtype=float)


def main():
    out = ROOT / "media" / "part_a" / "a2"
    out.mkdir(parents=True, exist_ok=True)
    h = computeH(POINTS_1, POINTS_2)
    projected = np.column_stack((POINTS_1, np.ones(len(POINTS_1)))) @ h.T
    projected = projected[:, :2] / projected[:, 2:]
    errors = np.linalg.norm(projected - POINTS_2, axis=1)
    # Numbered markers identify the same manually chosen point in each image.
    font = ImageFont.load_default(size=25)
    for index, points in [(1, POINTS_1), (2, POINTS_2)]:
        with Image.open(ROOT / "data" / f"mosaic2-{index}.jpg") as source:
            image = source.convert("RGB")
        draw = ImageDraw.Draw(image)
        for number, (x, y) in enumerate(points, 1):
            draw.ellipse((x - 7, y - 7, x + 7, y + 7), fill="#ffdf00", outline="black", width=2)
            draw.text((x + 10, y - 30), str(number), font=font,
                      fill="#ffdf00", stroke_width=2, stroke_fill="black")
        image.save(out / f"points_{index}.jpg", quality=95)
    data = {"source": "mosaic2-1.jpg", "target": "mosaic2-2.jpg",
            "points_1": POINTS_1.tolist(), "points_2": POINTS_2.tolist(),
            "homography": h.tolist(), "errors_px": errors.tolist(),
            "rmse_px": float(np.sqrt(np.mean(errors ** 2)))}
    (out / "results.json").write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    (out / "results.js").write_text("window.homographyData = " + json.dumps(data) + ";\n", encoding="utf-8")
    print(h)
    print(f"Reprojection RMSE: {data['rmse_px']:.3f} px")
    print("Point errors:", np.round(errors, 2))


if __name__ == "__main__":
    main()
