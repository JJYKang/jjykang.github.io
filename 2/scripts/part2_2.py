"""Generate three aligned hybrids, Fourier analysis, and color comparisons."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
from PIL import Image, ImageOps
from hybrid import align_image, hybrid_image, grayscale_rgb, log_spectrum

PROVIDED = ROOT / "data" / "provided" / "syde671 assignment 2 hybrid" / "hybrid"
CUSTOM = ROOT / "data" / "custom"

# Landmarks are measured after resizing each input to at most 512 pixels.
# Crops use (left, top, right, bottom) in the low-image coordinates.
PAIRS = [
    ("derek_nutmeg", PROVIDED / "DerekPicture.jpg", PROVIDED / "nutmeg.jpg",
     [(149, 172), (220, 167)], [(220, 105), (273, 134)], (85, 70, 290, 355), 5, 2),
    ("pair1", CUSTOM / "pair1_A.jpg", CUSTOM / "pair1_B.jpg",
     [(160, 226), (222, 223)], [(160, 181), (219, 179)], (95, 115, 290, 345), 3, 2),
    ("pair2", CUSTOM / "pair2_A.jpg", CUSTOM / "pair2_B.jpg",
     [(222, 222), (162, 298)], [(224, 172), (184, 272)], (80, 150, 300, 400), 3, 1),
]


def load_rgb(path):
    with Image.open(path) as source:
        source = ImageOps.exif_transpose(source).convert("RGB")
        source.thumbnail((512, 512))
        return np.asarray(source, dtype=np.float64) / 255


def save_rgb(image, path):
    # Keep signed filter values until export.
    Image.fromarray(np.round(np.clip(image, 0, 1) * 255).astype(np.uint8)).save(path)


def main():
    for name, path_a, path_b, points_a, points_b, crop, sigma_low, sigma_high in PAIRS:
        original_a, original_b = load_rgb(path_a), load_rgb(path_b)
        aligned_b, valid = align_image(original_b, points_b, points_a, original_a.shape)
        left, top, right, bottom = crop
        # Crop to shared image content, excluding empty warp borders.
        if not valid[top:bottom, left:right].all():
            raise ValueError(f"{name}: crop contains an empty alignment border")
        a = original_a[top:bottom, left:right]
        b = aligned_b[top:bottom, left:right]
        low, high, both_color = hybrid_image(a, b, sigma_low, sigma_high)
        # Keep color in the low band; use luminance for high-frequency detail.
        gray_high = grayscale_rgb(high)
        hybrid = low + gray_high
        out = ROOT / "media" / "part2" / "2_2" / name
        out.mkdir(parents=True, exist_ok=True)
        for label, image in [("original_a", original_a), ("original_b", original_b), ("hybrid", hybrid)]:
            save_rgb(image, out / f"{label}.png")

        if name == "pair1":
            # Show the complete filtering and frequency process for this example.
            save_rgb(a, out / "aligned_a.png")
            save_rgb(b, out / "aligned_b.png")
            save_rgb(low, out / "low.png")
            save_rgb(0.5 + 2 * gray_high, out / "high.png")
            stages = {"input_a": a, "input_b": b, "low": low, "high": gray_high, "hybrid": hybrid}
            spectra = {label: log_spectrum(image) for label, image in stages.items()}
            scale = max(s.max() for s in spectra.values())
            for label, spectrum in spectra.items():
                Image.fromarray(np.round(spectrum / scale * 255).astype(np.uint8)).save(out / f"fft_{label}.png")

            # Compare color in each frequency band with the grayscale baseline.
            gray_low = grayscale_rgb(low)
            for label, image in [("gray", gray_low + gray_high),
                                 ("high_color", gray_low + high),
                                 ("both_color", both_color)]:
                save_rgb(image, out / f"{label}.png")
        print(f"{name}: {a.shape[1]}x{a.shape[0]}, sigma low/high = {sigma_low}/{sigma_high}")


if __name__ == "__main__":
    main()
