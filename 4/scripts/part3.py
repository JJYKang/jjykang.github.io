"""Generate full-set and subset affine factorizations for Problem 3(a)."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
from factorization import build_measurement_matrix, factorization_method
from part1 import DATA, load_image_points

OUT = ROOT / "media" / "part3"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    records = []
    for name in ("set1", "set1_subset"):
        first = load_image_points(DATA / name / "pt_2D_1.txt")
        second = load_image_points(DATA / name / "pt_2D_2.txt")
        structure, motion = factorization_method(first, second)
        measurements, translation = build_measurement_matrix(first, second)
        values = np.linalg.svd(measurements, compute_uv=False)
        residual = measurements - motion @ structure
        # With four measurement rows, only the fourth singular value is discarded.
        if not np.isclose(np.linalg.norm(residual), values[3], rtol=1e-10, atol=1e-10):
            raise ValueError(f"Factorization for {name} is not the rank-three SVD fit")
        filename = f"{name}_factorization.npz"
        np.savez_compressed(OUT / filename, structure=structure, motion=motion,
                            translation=translation, singular_values=values,
                            points1=first, points2=second)
        record = {"dataset": name, "correspondences": len(first),
                  "singular_values": values.tolist(),
                  "coordinate_reprojection_rmse_px": float(np.sqrt(np.mean(residual**2))),
                  "factorization_file": filename}
        records.append(record)
        print(f"{name}: structure {structure.shape}, motion {motion.shape}")
        print("Singular values:", values)
        print("Coordinate reprojection RMSE (px):", record["coordinate_reprojection_rmse_px"])
    result = {"measurement_row_order": ["x1", "y1", "x2", "y2"],
              "factorization": "centered_measurements = motion @ structure + residual",
              "records": records}
    (OUT / "results.json").write_text(json.dumps(result, indent=2) + "\n",
                                      encoding="utf-8")


if __name__ == "__main__":
    main()
