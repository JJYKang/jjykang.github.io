"""Generate affine factorizations and ground-truth comparisons for Problem 3."""
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
from factorization import build_measurement_matrix, factorization_method
from part1 import DATA, load_image_points

# Support the ignored local package directory used for this checkout's plots.
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".preview" / "matplotlib"))
try:
    import matplotlib
except ModuleNotFoundError:
    sys.path.insert(0, str(ROOT / ".deps"))
    import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = ROOT / "media" / "part3"


def align_similarity(points, target):
    """Fit translation, uniform scale, and orthogonal orientation for display only."""
    center = points.mean(axis=0)
    target_center = target.mean(axis=0)
    centered = points - center
    centered_target = target - target_center
    u, _, vt = np.linalg.svd(centered.T @ centered_target)
    orientation = u @ vt
    scale = np.sum((centered @ orientation) * centered_target) / np.sum(centered**2)
    aligned = scale * centered @ orientation + target_center
    return aligned, scale, orientation


def save_comparison(name, structure):
    """Compare reconstruction and ground truth, with an edge view for the subset."""
    truth = np.loadtxt(DATA / name / "pt_3D.txt", skiprows=1)
    if truth.shape != (structure.shape[1], 3):
        raise ValueError("Ground-truth points do not match the reconstructed points")
    aligned, scale, orientation = align_similarity(structure.T, truth)
    errors = np.linalg.norm(aligned-truth, axis=1)
    rmse = np.sqrt(np.mean(errors**2))
    extent = np.sqrt(np.mean(np.sum((truth-truth.mean(axis=0))**2, axis=1)))
    np.savez_compressed(OUT / f"{name}_comparison.npz", ground_truth=truth,
                        display_aligned_structure=aligned, errors=errors)

    combined = np.vstack((aligned, truth))
    center = (combined.min(axis=0)+combined.max(axis=0)) / 2
    radius = np.ptp(combined, axis=0).max() * .55
    colors = plt.get_cmap("viridis")(np.linspace(0, 1, len(truth)))
    views = [(20, -60)] if name == "set1" else [(20, -60), (0, 0)]
    fig = plt.figure(figsize=(10, 4.8*len(views)), layout="constrained")
    for row, (elevation, azimuth) in enumerate(views):
        for column, (points, title) in enumerate(
                [(aligned, "Factorization (display aligned)"), (truth, "Ground truth")]):
            ax = fig.add_subplot(len(views), 2, row*2+column+1, projection="3d")
            ax.scatter(*points.T, c=colors, s=30, edgecolors="white", linewidths=.3,
                       depthshade=False)
            for setter, mid in zip((ax.set_xlim, ax.set_ylim, ax.set_zlim), center):
                setter(mid-radius, mid+radius)
            ax.set_box_aspect((1, 1, 1))
            ax.view_init(elev=elevation, azim=azimuth)
            if row == 1:
                title += " — edge-on"
            ax.set(xlabel="X", ylabel="Y", zlabel="Z", title=title)
            ax.tick_params(labelsize=8)
            if row == 1:
                ax.set_proj_type("ortho")
                ax.set_xticks([])
                ax.set_xlabel("")
    filename = f"{name}_reconstruction_comparison.png"
    fig.savefig(OUT / filename, dpi=180)
    plt.close(fig)
    return {"plot": filename, "display_alignment": "translation, uniform scale, orthogonal orientation",
            "scale": float(scale), "orientation": orientation.tolist(),
            "allows_reflection": True, "point_rmse": float(rmse),
            "relative_point_rmse": float(rmse / extent),
            "ground_truth_extent_xyz": np.ptp(truth, axis=0).tolist(),
            "aligned_structure_extent_xyz": np.ptp(aligned, axis=0).tolist(),
            "views_elevation_azimuth": [list(view) for view in views]}


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
        record["ground_truth_comparison"] = save_comparison(name, structure)
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
