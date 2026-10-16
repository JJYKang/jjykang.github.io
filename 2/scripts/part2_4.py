"""Generate two curved-mask blends, a color comparison, and band contributions."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
from PIL import Image, ImageOps
from blending import multiresolution_blend
from hybrid import align_image, grayscale_rgb

CUSTOM = ROOT / "data" / "custom"
OUT = ROOT / "media" / "part2" / "2_4"
LEVELS = 6
SIGMA = 2


def load_rgb(filename):
    with Image.open(CUSTOM / filename) as source:
        source = ImageOps.exif_transpose(source).convert("RGB")
        source.thumbnail((640, 640))
        return np.asarray(source, dtype=np.float64) / 255


def save_image(image, path):
    # Preserve signed bands during computation; clip only for export.
    pixels = np.round(np.clip(image, 0, 1) * 255).astype(np.uint8)
    Image.fromarray(pixels).save(path)


def main():
    for name, file_a, file_b in [("drinks", "blend1-A.jpg", "blend1-B.jpg"),
                                ("landscapes", "blend2-A.jpg", "blend2-B.jpg")]:
        a, b = load_rgb(file_a), load_rgb(file_b)
        if name == "drinks":
            # Match cup centers and heights, then crop to shared content.
            a, valid = align_image(a, [(255, 130), (255, 525)],
                                  [(240, 165), (240, 525)], b.shape)
            crop = (slice(50, 627), slice(10, 443))
            if not valid[crop].all():
                raise ValueError("Cup crop includes an empty alignment border")
            a, b = a[crop], b[crop]
        else:
            # Match the canvas sizes without stretching either landscape.
            top = (a.shape[0] - b.shape[0]) // 2
            a = a[top:top + b.shape[0]]

        # White selects A, black selects B; a curved seam keeps both subjects.
        height, width = a.shape[:2]
        y, x = np.mgrid[:height, :width]
        amplitude = 0.04 if name == "drinks" else 0.055
        seam = width * (0.5 + amplitude * np.sin(2 * np.pi * y / height))
        mask = (x < seam).astype(float)
        result, weighted_a, weighted_b = multiresolution_blend(a, b, mask, LEVELS, SIGMA)
        direct = mask[:, :, None] * a + (1 - mask[:, :, None]) * b
        out = OUT / name
        out.mkdir(parents=True, exist_ok=True)
        for label, image in [("input_a", a), ("input_b", b), ("mask", mask),
                             ("direct", direct), ("blend", result)]:
            save_image(image, out / f"{label}.png")

        if name == "drinks":
            # Compare color using the same mask and frequency weights.
            save_image(grayscale_rgb(result), out / "grayscale.png")

            # Figure 10 style: selected bands and reconstructed contributions.
            combined = weighted_a + weighted_b
            panels = []
            for level in [0, 2, 4]:
                row = [weighted_a[level], weighted_b[level], combined[level]]
                scale = max(np.abs(image).max() for image in row)
                panels.extend([0.5 + image / (2 * scale) for image in row])
            panels.extend([weighted_a.sum(axis=0), weighted_b.sum(axis=0), result])
            for letter, panel in zip("abcdefghijkl", panels):
                save_image(panel, out / f"process_{letter}.png")
        print(f"Saved {name}: {width}x{height}, {LEVELS} stack levels")


if __name__ == "__main__":
    main()
