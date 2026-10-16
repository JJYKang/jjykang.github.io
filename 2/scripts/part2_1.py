"""Generate unsharp masking and blur-then-sharpen results for Part 2.1."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
from PIL import Image, ImageOps
from filters import gaussian_kernel, filter_rgb, unsharp_mask

SIZE = 9
SIGMA = 2.0
AMOUNT = 2


def save_rgb(image, path):
    # Clip only for export; filtering keeps signed and out-of-range values.
    pixels = np.round(np.clip(image, 0, 1) * 255).astype(np.uint8)
    Image.fromarray(pixels).save(path)


def main():
    gaussian = gaussian_kernel(SIZE, SIGMA)
    inputs = [("taj", ROOT / "data" / "provided" / "syde671 assn2 part 2 images" / "taj.jpg"),
              ("custom", ROOT / "data" / "custom" / "sharp.jpg")]
    for name, path in inputs:
        # Preserve orientation and aspect ratio; use RGB intensities in [0, 1].
        with Image.open(path) as source:
            source = ImageOps.exif_transpose(source).convert("RGB")
            source.thumbnail((640, 640))
            image = np.asarray(source, dtype=np.float64) / 255
        out = ROOT / "media" / "part2" / "2_1" / name
        out.mkdir(parents=True, exist_ok=True)

        # Separate low and high frequencies for the explanation.
        blurred = filter_rgb(image, gaussian)
        high = image - blurred
        save_rgb(image, out / "original.png")
        save_rgb(blurred, out / "blurred.png")
        save_rgb(0.5 + 4 * high, out / "high_frequency.png")

        # Show sharpening strength on the Taj; use amount 2 for the custom photo.
        amounts = [1, 2, 4] if name == "taj" else [AMOUNT]
        for amount in amounts:
            sharpened = unsharp_mask(image, gaussian, amount)
            save_rgb(sharpened, out / f"sharpened_{amount}.png")
            error = np.max(np.abs(sharpened - (image + amount * high)))
            print(f"{name}, amount {amount}: single-kernel difference = {error:.2e}")

        if name == "custom":
            # Start from the blurred image to evaluate lost-detail recovery.
            resharpened = unsharp_mask(blurred, gaussian, AMOUNT)
            save_rgb(resharpened, out / "resharpened.png")
    print("Saved Part 2.1 results.")


if __name__ == "__main__":
    main()
