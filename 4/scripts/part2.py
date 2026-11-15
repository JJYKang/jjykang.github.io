"""Compute and check the Set 1 epipoles for Problem 2(a)."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
from fundamental import normalized_eight_point_alg
from rectification import compute_epipole
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

    OUT.mkdir(parents=True, exist_ok=True)
    result = {"dataset": "set1", "fundamental_matrix": f.tolist(),
              "epipole_image1": e1.tolist(), "epipole_image2": e2.tolist(),
              "relative_null_residual_image1": float(residual1),
              "relative_null_residual_image2": float(residual2)}
    (OUT / "results.json").write_text(json.dumps(result, indent=2) + "\n",
                                      encoding="utf-8")
    print("Image 1 epipole:", e1)
    print("Image 2 epipole:", e2)
    print("Null-space residuals:", residual1, residual2)


if __name__ == "__main__":
    main()
