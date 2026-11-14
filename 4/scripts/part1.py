"""Generate linear and normalized eight-point estimates for both supplied pairs."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
from fundamental import lls_eight_point_alg, normalized_eight_point_alg

DATA = ROOT / "assignment 4 code" / "data"
OUT = ROOT / "media" / "part1"


def load_image_points(path):
    """Read the supplied (y, x) text rows as homogeneous (x, y, 1) points."""
    with path.open() as source:
        count = int(source.readline())
        points = np.loadtxt(source, ndmin=2)
    if points.shape != (count, 2):
        raise ValueError(f"Unexpected point data in {path}")
    return np.column_stack((points[:, 1], points[:, 0], np.ones(count)))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    records = []
    for name in ("set1", "set2"):
        first = load_image_points(DATA / name / "pt_2D_1.txt")
        second = load_image_points(DATA / name / "pt_2D_2.txt")
        for method, estimate in [("linear", lls_eight_point_alg),
                                 ("normalized", normalized_eight_point_alg)]:
            f = estimate(first, second)
            values = np.linalg.svd(f, compute_uv=False)
            if not (values[1] > 1e-12 and values[2] < 1e-12):
                raise ValueError(f"Fundamental matrix for {name}/{method} is not rank two")
            np.save(OUT / f"{name}_{method}_F.npy", f)
            records.append({"dataset": name, "method": method,
                            "correspondences": len(first),
                            "fundamental_matrix": f.tolist(),
                            "singular_values": values.tolist()})
            print(f"{name}, {method}: {len(first)} correspondences")
            print(np.array2string(f, precision=8))
            print("Singular values:", values)

    result = {"methods": ["linear", "normalized"],
              "constraint": "p2.T @ F @ p1 = 0",
              "scale": "unit Frobenius norm; largest-magnitude entry positive",
              "records": records}
    (OUT / "results.json").write_text(json.dumps(result, indent=2) + "\n",
                                      encoding="utf-8")


if __name__ == "__main__":
    main()
