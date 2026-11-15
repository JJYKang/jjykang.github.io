import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from fundamental import normalized_eight_point_alg
from rectification import compute_epipole, compute_matching_homographies
from part1 import DATA, load_image_points
from warping import warp_bounds, warpImageBilinear

OUT = ROOT / "media" / "part2"


def save_rectified_pair(images, homographies, points):
    # warp both images at the same scale and vertical origin; add scanline guides
    bounds = [warp_bounds(image.shape, h) for image, h in zip(images, homographies)]
    top = min(origin[1] for _, origin in bounds)
    bottom = max(origin[1]+shape[0]-1 for shape, origin in bounds)
    height = bottom-top+1
    panels, display_points, records = [], [], []
    font = ImageFont.load_default(size=16)
    for index, (image, h, transformed, (shape, origin)) in enumerate(
            zip(images, homographies, points, bounds), 1):
        origin = (int(origin[0]), int(top))
        output_shape = (int(height), shape[1])
        warped, valid, _ = warpImageBilinear(image, h, output_shape, origin)
        warped[~valid] = 255
        pixels = Image.fromarray(np.rint(np.clip(warped, 0, 255)).astype(np.uint8))
        filename = f"set1_rectified_{index}.png"
        pixels.save(OUT / filename)
        panel = Image.new("RGB", (pixels.width, pixels.height+30), "white")
        panel.paste(pixels, (0, 30))
        ImageDraw.Draw(panel).text((8, 5), f"Image {index}", fill="black", font=font)
        panels.append(panel)
        display_points.append(transformed[:, :2] - np.array(origin))
        records.append({"image": index, "filename": filename,
                        "origin_xy": list(origin), "shape_hw": list(output_shape),
                        "valid_pixels": int(valid.sum())})

    gap = 12
    sheet = Image.new("RGB", (sum(panel.width for panel in panels)+gap,
                              int(height)+30), "white")
    sheet.paste(panels[0], (0, 0))
    sheet.paste(panels[1], (panels[0].width+gap, 0))
    draw = ImageDraw.Draw(sheet)
    selected = np.argsort(display_points[1][:, 1])[
        np.linspace(0, len(points[1])-1, 6, dtype=int)]
    for index in selected:
        y = display_points[1][index, 1]+30
        draw.line((0, y, sheet.width-1, y), fill="#ed333b", width=1)
        for offset, transformed in zip((0, panels[0].width+gap), display_points):
            x, y = transformed[index]+[offset, 30]
            draw.ellipse((x-3, y-3, x+3, y+3), fill="#0066ff")
    sheet.save(OUT / "set1_rectified_pair.png")
    return {"images": records, "pair_filename": "set1_rectified_pair.png",
            "pair_size_wh": list(sheet.size), "guide_point_indices": selected.tolist()}


def main():
    first = load_image_points(DATA / "set1" / "pt_2D_1.txt")
    second = load_image_points(DATA / "set1" / "pt_2D_2.txt")
    f = normalized_eight_point_alg(first, second)
    e1 = compute_epipole(f.T)
    e2 = compute_epipole(f)
    residual1 = np.linalg.norm(f @ e1) / np.linalg.norm(e1)
    residual2 = np.linalg.norm(f.T @ e2) / np.linalg.norm(e2)
    if max(residual1, residual2) > 1e-12:
        raise ValueError("Epipoles do not satisfy their null-space constraints")

    images = []
    for index in (1, 2):
        with Image.open(DATA / "set1" / f"image{index}.jpg") as image:
            images.append(np.asarray(image.convert("RGB")))
    h1, h2 = compute_matching_homographies(e2, f, images[1], first, second)
    mapped_epipole = h2 @ e2
    mapped_epipole /= np.linalg.norm(mapped_epipole)
    if max(abs(mapped_epipole[1:])) > 1e-12:
        raise ValueError("Homography does not map the epipole to horizontal infinity")

    transformed1 = first @ h1.T
    transformed2 = second @ h2.T
    transformed1 /= transformed1[:, 2:3]
    transformed2 /= transformed2[:, 2:3]
    y_errors = transformed1[:, 1] - transformed2[:, 1]
    rectified_f = np.linalg.inv(h2).T @ f @ np.linalg.inv(h1)
    rectified_f /= np.linalg.norm(rectified_f)
    # Horizontal, corresponding scanlines have the canonical epipolar constraint.
    canonical = np.array([[0., 0, 0], [0, 0, -1], [0, 1, 0]]) / np.sqrt(2)
    canonical_error = min(np.linalg.norm(rectified_f-canonical),
                          np.linalg.norm(rectified_f+canonical))
    if canonical_error > 1e-8:
        raise ValueError("Matching homographies do not produce horizontal epipolar geometry")

    OUT.mkdir(parents=True, exist_ok=True)
    np.save(OUT / "set1_H1.npy", h1)
    np.save(OUT / "set1_H2.npy", h2)
    rectified_images = save_rectified_pair(images, (h1, h2),
                                          (transformed1, transformed2))
    result = {"dataset": "set1", "fundamental_matrix": f.tolist(),
              "epipole_image1": e1.tolist(), "epipole_image2": e2.tolist(),
              "relative_null_residual_image1": float(residual1),
              "relative_null_residual_image2": float(residual2),
              "homography_image1": h1.tolist(), "homography_image2": h2.tolist(),
              "mapped_epipole_image2": mapped_epipole.tolist(),
              "transformed_points_image1": transformed1.tolist(),
              "transformed_points_image2": transformed2.tolist(),
              "mean_vertical_disparity_px": float(np.abs(y_errors).mean()),
              "rms_vertical_disparity_px": float(np.sqrt(np.mean(y_errors**2))),
              "max_vertical_disparity_px": float(np.abs(y_errors).max()),
              "rectified_fundamental_matrix": rectified_f.tolist(),
              "canonical_matrix_error": float(canonical_error),
              "rectified_images": rectified_images}
    (OUT / "results.json").write_text(json.dumps(result, indent=2) + "\n",
                                      encoding="utf-8")
    print("Image 1 epipole:", e1)
    print("Image 2 epipole:", e2)
    print("Null-space residuals:", residual1, residual2)
    print("Image 1 homography:\n", h1)
    print("Image 2 homography:\n", h2)
    print("Mapped epipole (unit homogeneous vector):", mapped_epipole)
    print("Mean vertical disparity (px):", result["mean_vertical_disparity_px"])
    print("Canonical rectified-matrix error:", canonical_error)


if __name__ == "__main__":
    main()
