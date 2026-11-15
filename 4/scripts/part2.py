"""Compute and check Set 1 epipoles and matching homographies for 2(a-c)."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
from PIL import Image
from fundamental import normalized_eight_point_alg
from rectification import compute_epipole, compute_matching_homographies
from part1 import DATA, load_image_points

OUT = ROOT / "media" / "part2"


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

    with Image.open(DATA / "set1" / "image2.jpg") as image:
        h1, h2 = compute_matching_homographies(e2, f, np.asarray(image), first, second)
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
              "canonical_matrix_error": float(canonical_error)}
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
