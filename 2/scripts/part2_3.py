"""Generate fruit stacks and the Figure 3.42 apple/orange example."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
from PIL import Image, ImageDraw
from stacks import gaussian_stack, laplacian_stack

INPUT = ROOT / "data" / "provided" / "syde671 assn2 part 2 images"
OUT = ROOT / "media" / "part2" / "2_3"
LEVELS = 6
SIGMA = 2


def display_image(array):
    # Clip only for display; keep signed bands for reconstruction.
    return Image.fromarray(np.round(np.clip(array, 0, 1) * 255).astype(np.uint8))


def save_stack_sheet(stack, name, laplacian=False, band_scale=1):
    # Thumbnails are for the webpage; stack arrays keep their full dimensions.
    sheet = Image.new("RGB", (960, 510), "white")
    draw = ImageDraw.Draw(sheet)
    for level, image in enumerate(stack):
        if laplacian and level < LEVELS - 1:
            image = 0.5 + image / (2 * band_scale)
        tile = display_image(image).resize((320, 229), Image.Resampling.LANCZOS)
        x, y = (level % 3) * 320, (level // 3) * 255
        label = f"L{level}" if laplacian else f"G{level}"
        if laplacian and level == LEVELS - 1:
            label += " (low-pass residual)"
        draw.text((x + 8, y + 5), label, fill="black")
        sheet.paste(tile, (x, y + 26))
    sheet.save(OUT / f"{name}.png")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    gaussian, laplacian = {}, {}
    for name, filename in [("apple", "burt_apple.png"), ("orange", "burt_orange.png")]:
        with Image.open(INPUT / filename) as source:
            source = source.convert("RGB")
            source.thumbnail((480, 480))
            image = np.asarray(source, dtype=np.float64) / 255
        gaussian[name] = gaussian_stack(image, LEVELS, SIGMA)
        laplacian[name] = laplacian_stack(gaussian[name])
        # Verify that the signed bands and residual recover the input.
        error = np.max(np.abs(laplacian[name].sum(axis=0) - image))
        print(f"{name}: {LEVELS} levels at {image.shape}; reconstruction error {error:.2e}")

    # Use one signed scale for both fruit stack sheets.
    band_scale = max(np.abs(laplacian[name][:-1]).max() for name in laplacian)
    for name in gaussian:
        save_stack_sheet(gaussian[name], f"{name}_gaussian")
        save_stack_sheet(laplacian[name], f"{name}_laplacian", True, band_scale)

    # Smooth a left/right mask at the same scales as the images.
    height, width = gaussian["apple"].shape[1:3]
    mask = np.zeros((height, width))
    mask[:, :width // 2] = 1
    masks = gaussian_stack(mask, LEVELS, SIGMA)[:, :, :, None]
    apple_bands = masks * laplacian["apple"]
    orange_bands = (1 - masks) * laplacian["orange"]
    combined = apple_bands + orange_bands

    # Figure 3.42: levels 0, 2, 4, followed by reconstructed contributions.
    panels = []
    for level in [0, 2, 4]:
        row = [apple_bands[level], orange_bands[level], combined[level]]
        # Match all three displays within a frequency row.
        scale = max(np.abs(image).max() for image in row)
        panels.extend([0.5 + image / (2 * scale) for image in row])
    panels.extend([apple_bands.sum(axis=0), orange_bands.sum(axis=0), combined.sum(axis=0)])
    for letter, panel in zip("abcdefghijkl", panels):
        display_image(panel).save(OUT / f"figure_{letter}.png")
    print(f"Saved Part 2.3 stacks and Figure 3.42 panels to {OUT}")


if __name__ == "__main__":
    main()
