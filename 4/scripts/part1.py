"""Generate linear and normalized eight-point estimates for both supplied pairs."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from fundamental import (lls_eight_point_alg, normalized_eight_point_alg,
                         compute_distance_to_epipolar_lines)

DATA = ROOT / "assignment 4 code" / "data"
OUT = ROOT / "media" / "part1"


def load_image_points(path):
    """Read the supplied (y, x) text rows as homogeneous (x, y, 1) points."""
    with path.open() as source:
        count = int(source.readline())
        points = np.loadtxt(source, ndmin=2)
    if points.shape != (count, 2):
        raise ValueError(f"Unexpected point data in {path}")
    return np.column_stack((points[:, 1], points[:, 0], np.ones(count)))


def line_segment(line, width, height):
    """Clip ax + by + c = 0 to the image, including vertical lines."""
    a, b, c = line
    candidates = []
    if b != 0:
        candidates.extend((x, (-c - a*x) / b) for x in (0, width-1))
    if a != 0:
        candidates.extend(((-c - b*y) / a, y) for y in (0, height-1))
    visible = []
    for x, y in candidates:
        if 0 <= x <= width-1 and 0 <= y <= height-1:
            if not any(np.allclose((x, y), point) for point in visible):
                visible.append((x, y))
    return visible[:2] if len(visible) >= 2 else None


def save_epipolar_plot(name, method, first, second, f):
    """Draw corresponding epipolar lines in red and measured points in blue."""
    panels = []
    font = ImageFont.load_default(size=16)
    for index, points, lines in [(1, first, second @ f),
                                 (2, second, first @ f.T)]:
        with Image.open(DATA / name / f"image{index}.jpg") as source:
            image = source.convert("RGB")
        draw = ImageDraw.Draw(image)
        for line in lines:
            segment = line_segment(line, *image.size)
            if segment is not None:
                draw.line(segment, fill="#ed333b", width=1)
        for x, y, _ in points:
            draw.ellipse((x-2, y-2, x+2, y+2), fill="#0066ff")
        panel = Image.new("RGB", (image.width, image.height+30), "white")
        panel.paste(image, (0, 30))
        ImageDraw.Draw(panel).text((8, 5), f"Image {index}", fill="black", font=font)
        panels.append(panel)
    sheet = Image.new("RGB", (sum(p.width for p in panels)+12,
                              max(p.height for p in panels)), "white")
    sheet.paste(panels[0], (0, 0))
    sheet.paste(panels[1], (panels[0].width+12, 0))
    filename = f"{name}_{method}_epipolar.png"
    sheet.save(OUT / filename)
    return filename


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    records = []
    for name in ("set1", "set2"):
        first = load_image_points(DATA / name / "pt_2D_1.txt")
        second = load_image_points(DATA / name / "pt_2D_2.txt")
        for method, estimate in [("linear", lls_eight_point_alg),
                                 ("normalized", normalized_eight_point_alg)]:
            f = estimate(first, second)
            values = np.linalg.svd(f, compute_uv=False)
            if not (values[1] > 1e-12 and values[2] < 1e-12):
                raise ValueError(f"Fundamental matrix for {name}/{method} is not rank two")
            np.save(OUT / f"{name}_{method}_F.npy", f)
            distance1 = compute_distance_to_epipolar_lines(first, second, f)
            distance2 = compute_distance_to_epipolar_lines(second, first, f.T)
            plot = save_epipolar_plot(name, method, first, second, f)
            records.append({"dataset": name, "method": method,
                            "correspondences": len(first),
                            "fundamental_matrix": f.tolist(),
                            "singular_values": values.tolist(),
                            "mean_distance_image1_px": distance1,
                            "mean_distance_image2_px": distance2,
                            "epipolar_plot": plot})
            print(f"{name}, {method}: {len(first)} correspondences")
            print(np.array2string(f, precision=8))
            print("Singular values:", values)
            print(f"Mean line distances: image 1 = {distance1:.6f} px, "
                  f"image 2 = {distance2:.6f} px")

    result = {"methods": ["linear", "normalized"],
              "constraint": "p2.T @ F @ p1 = 0",
              "scale": "unit Frobenius norm; largest-magnitude entry positive",
              "records": records}
    (OUT / "results.json").write_text(json.dumps(result, indent=2) + "\n",
                                      encoding="utf-8")


if __name__ == "__main__":
    main()
