"""Extract normalized patch descriptors and illustrate six features in image 2."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from features import extract_descriptors

OUT = ROOT / "media" / "part_b" / "b2"
SIGMA = 2.0
# Locations used to choose six spatially separated examples from image 2.
EXAMPLE_LOCATIONS = np.array([[280, 300], [610, 320], [740, 400],
                              [550, 440], [500, 630], [850, 250]])


def make_examples(image, gray, points, descriptors):
    chosen = []
    for location in EXAMPLE_LOCATIONS:
        distances = np.sum((points - location) ** 2, axis=1).astype(float)
        distances[chosen] = np.inf
        chosen.append(int(np.argmin(distances)))
    marked = image.copy()
    draw = ImageDraw.Draw(marked)
    font = ImageFont.load_default(size=20)
    sheet = Image.new("RGB", (960, 410), "white")
    labels = ImageDraw.Draw(sheet)
    labels.text((8, 4), "40x40 source patches", fill="black", font=font)
    labels.text((8, 215), "8x8 normalized descriptors", fill="black", font=font)
    records = []
    for number, index in enumerate(chosen, 1):
        x, y = points[index].astype(int)
        draw.rectangle((x - 20, y - 20, x + 19, y + 19), outline="#ffdf00", width=2)
        draw.text((x - 20, y - 43), str(number), fill="#ffdf00", font=font,
                  stroke_width=2, stroke_fill="black")
        raw = gray[y - 20:y + 20, x - 20:x + 20]
        # One fixed display scale: descriptor values [-3, 3] map to black/white.
        normalized = np.clip(0.5 + descriptors[index].reshape(8, 8) / 6, 0, 1)
        for patch, top in [(raw, 52), (normalized, 244)]:
            tile = Image.fromarray(np.round(np.clip(patch, 0, 1) * 255).astype(np.uint8))
            tile = tile.resize((152, 152), Image.Resampling.NEAREST).convert("RGB")
            sheet.paste(tile, ((number - 1) * 160 + 4, top))
        labels.text(((number - 1) * 160 + 4, 30), f"{number}: ({x}, {y})", fill="black", font=ImageFont.load_default(size=15))
        records.append({"number": number, "descriptor_index": index, "point": [int(x), int(y)]})
    marked.save(OUT / "examples_locations.jpg", quality=95)
    sheet.save(OUT / "examples_descriptors.png")
    return records


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    settings = json.loads((ROOT / "media" / "part_b" / "b1" / "results.json").read_text())
    records, examples = [], []
    for record in settings["records"]:
        index = record["image"]
        with Image.open(ROOT / "data" / record["source"]) as source:
            image = source.convert("RGB")
            image.thumbnail((settings["max_dimension"], settings["max_dimension"]), Image.Resampling.LANCZOS)
        gray = np.asarray(image, dtype=float) @ np.array([0.299, 0.587, 0.114]) / 255
        with np.load(ROOT / "media" / "part_b" / "b1" / f"image_{index}_corners.npz") as corners:
            np.testing.assert_array_equal(corners["image_shape"], gray.shape)
            points = corners["points"][corners["selected"]]
        descriptors, valid_indices = extract_descriptors(gray, points, sigma=SIGMA)
        points = points[valid_indices]
        np.savez_compressed(OUT / f"image_{index}_descriptors.npz", points=points,
                            descriptors=descriptors, anms_indices=valid_indices,
                            image_shape=np.array(gray.shape))
        records.append({"image": index, "source": record["source"], "count": len(points),
                        "max_abs_mean": float(np.max(np.abs(descriptors.mean(axis=1)))),
                        "max_std_error": float(np.max(np.abs(descriptors.std(axis=1) - 1)))})
        if index == 2:
            examples = make_examples(image, gray, points, descriptors)
        print(f"{record['source']}: {len(points)} normalized 64-value descriptors")
    data = {"sigma": SIGMA, "patch_size": 40, "descriptor_size": 8,
            "sample_step": 5, "records": records, "examples": examples}
    (OUT / "results.json").write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
