"""Match adjacent house-image descriptors and draw accepted feature pairs."""
import colorsys
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
from PIL import Image, ImageDraw
from features import match_descriptors

OUT = ROOT / "media" / "part_b" / "b3"
RATIO_THRESHOLD = 0.5  # Ratio of squared distances (SSD), as in Figure 6b.
PAIRS = [(1, 2), (2, 3)]


def draw_matches(images, points1, points2, path):
    gap = 20
    offset = images[0].width + gap
    canvas = Image.new("RGB", (offset + images[1].width, max(im.height for im in images)), "white")
    canvas.paste(images[0], (0, 0))
    canvas.paste(images[1], (offset, 0))
    draw = ImageDraw.Draw(canvas)
    for index, (p, q) in enumerate(zip(points1, points2)):
        color = tuple(round(v * 255) for v in colorsys.hsv_to_rgb((index * 0.618034) % 1, 0.75, 1))
        x, y = p
        u, v = q + [offset, 0]
        draw.line((x, y, u, v), fill=color, width=1)
        for px, py in [(x, y), (u, v)]:
            draw.ellipse((px - 3, py - 3, px + 3, py + 3), fill=color, outline="black", width=1)
    canvas.save(path, quality=95)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    images, features = {}, {}
    for index in (1, 2, 3):
        with np.load(ROOT / "media" / "part_b" / "b2" / f"image_{index}_descriptors.npz") as saved:
            features[index] = {key: saved[key] for key in ("points", "descriptors", "image_shape")}
        with Image.open(ROOT / "data" / f"mosaic3-{index}.jpg") as source:
            height, width = features[index]["image_shape"]
            images[index] = source.convert("RGB").resize((int(width), int(height)), Image.Resampling.LANCZOS)
    records = []
    for first, second in PAIRS:
        a, b = features[first], features[second]
        matches, ratios = match_descriptors(a["descriptors"], b["descriptors"], RATIO_THRESHOLD)
        points1, points2 = a["points"][matches[:, 0]], b["points"][matches[:, 1]]
        name = f"pair_{first}_{second}"
        np.savez_compressed(OUT / f"{name}_matches.npz", matches=matches, ratios=ratios,
                            points1=points1, points2=points2)
        draw_matches([images[first], images[second]], points1, points2, OUT / f"{name}_matches.jpg")
        records.append({"pair": [first, second], "queries": len(a["descriptors"]),
                        "matches": len(matches), "median_ratio": float(np.median(ratios)) if len(ratios) else None})
        print(f"Images {first}-{second}: {len(matches)} matches from {len(a['descriptors'])} queries")
    data = {"ratio_threshold": RATIO_THRESHOLD, "distance": "SSD", "records": records}
    (OUT / "results.json").write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    (OUT / "results.js").write_text("window.matchData = " + json.dumps(data) + ";\n", encoding="utf-8")


if __name__ == "__main__":
    main()
