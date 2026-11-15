"""Compute Set 1 epipoles and the Image 2 rectifying homography for 2(a-b)."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
from PIL import Image
from fundamental import normalized_eight_point_alg
from rectification import compute_epipole, compute_H
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
        h2 = compute_H(e2, np.asarray(image))
    mapped_epipole = h2 @ e2
    mapped_epipole /= np.linalg.norm(mapped_epipole)
    if max(abs(mapped_epipole[1:])) > 1e-12:
        raise ValueError("Homography does not map the epipole to horizontal infinity")

    OUT.mkdir(parents=True, exist_ok=True)
    np.save(OUT / "set1_H2.npy", h2)
    result = {"dataset": "set1", "fundamental_matrix": f.tolist(),
              "epipole_image1": e1.tolist(), "epipole_image2": e2.tolist(),
              "relative_null_residual_image1": float(residual1),
              "relative_null_residual_image2": float(residual2),
              "homography_image2": h2.tolist(),
              "mapped_epipole_image2": mapped_epipole.tolist()}
    (OUT / "results.json").write_text(json.dumps(result, indent=2) + "\n",
                                      encoding="utf-8")
    print("Image 1 epipole:", e1)
    print("Image 2 epipole:", e2)
    print("Null-space residuals:", residual1, residual2)
    print("Image 2 homography:\n", h2)
    print("Mapped epipole (unit homogeneous vector):", mapped_epipole)


if __name__ == "__main__":
    main()
