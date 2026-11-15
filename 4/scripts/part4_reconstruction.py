import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
if (ROOT / ".deps").is_dir():
    sys.path.append(str(ROOT / ".deps"))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".preview" / "matplotlib"))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from sfm_utils import Frame, bundle_adjustment, merge_all_frames, triangulate_dense_matches

DATA = ROOT / "assignment 4 code" / "data" / "statue"
OUT = ROOT / "media" / "part4"
FOCAL_LENGTH = 719.5459
PLOT_LIMITS = np.array([[-2.25, 2.25], [-2.5, 2.], [2.5, 7.]])
PLOT_VIEW = (-100, 90)


def save_plot(points):
    visible = ((points >= PLOT_LIMITS[:, 0]) & (points <= PLOT_LIMITS[:, 1])).all(axis=1)
    shown = points[visible]
    fig = plt.figure(figsize=(8, 8), layout="constrained")
    ax = fig.add_subplot(111, projection="3d")
    scatter = ax.scatter(*shown.T, c=shown[:, 2], cmap="viridis", s=.4,
                         linewidths=0, depthshade=False, rasterized=True)
    ax.set(xlim=PLOT_LIMITS[0], ylim=PLOT_LIMITS[1], zlim=PLOT_LIMITS[2],
           xlabel="X", ylabel="Y", zlabel="Z")
    ax.set_box_aspect((1, 1, 1))
    ax.set_xticks([-2, -1, 0, 1, 2])
    ax.set_yticks([-2, -1, 0, 1, 2])
    ax.set_zticks([3, 4, 5, 6, 7])
    ax.view_init(*PLOT_VIEW)
    fig.colorbar(scatter, ax=ax, shrink=.55, pad=.02, label="Depth Z (relative units)")
    fig.savefig(OUT / "statue_reconstruction.png", dpi=180)
    plt.close(fig)
    return int(visible.sum())


def normalize_reconstruction_frame(frame):
    # set Camera 1 to the world origin and its baseline to Camera 2 to unit length
    reference = frame.motion[0].copy()
    rotation, translation = reference[:, :3], reference[:, 3]
    frame.structure = frame.structure @ rotation.T + translation
    for pose in frame.motion:
        pose[:, :3] = pose[:, :3] @ rotation.T
        pose[:, 3] = pose[:, 3] - pose[:, :3] @ translation
    scale = 1 / np.linalg.norm(frame.motion[1, :, 3])
    frame.structure *= scale
    frame.motion[:, :, 3] *= scale
    return float(scale)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.plot_only:
        with np.load(OUT / "statue_reconstruction.npz") as saved:
            plotted = save_plot(saved["dense_structure"])
        result = json.loads((OUT / "reconstruction_results.json").read_text())
        result.update({"plotted_point_count": plotted, "plot_limits_xyz": PLOT_LIMITS.tolist(),
                       "plot_elevation_azimuth": list(PLOT_VIEW), "plot_color": "depth Z"})
        (OUT / "reconstruction_results.json").write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
        return
    started = time.perf_counter()
    sparse = np.load(DATA / "matches_subset.npy", allow_pickle=True, encoding="latin1")[0]
    fundamental = np.load(DATA / "fundamental_matrices.npy", allow_pickle=True, encoding="latin1")[0]
    dense = np.load(DATA / "dense_matches.npy", allow_pickle=True, encoding="latin1")
    image_paths = sorted((DATA / "images").glob("*.jpg"))
    with Image.open(image_paths[0]) as image:
        width, height = image.size
    frames = []
    for index in range(len(image_paths)-1):
        frame = Frame(sparse[index].T, FOCAL_LENGTH, fundamental[index], width, height)
        bundle_adjustment(frame)
        frames.append(frame)
        print(f"Initialized and adjusted pair {index+1}-{index+2}", flush=True)
    merged = merge_all_frames(frames)
    if len(merged.motion) != len(image_paths):
        raise ValueError("The reconstruction does not contain every camera")
    frame_scale = normalize_reconstruction_frame(merged)
    print(f"Merged {len(merged.motion)} cameras and {len(merged.structure)} sparse points", flush=True)

    clouds, records = [], []
    for index, matches in enumerate(dense):
        observations = matches.T.reshape(-1, 2, 2)
        cameras = np.stack((merged.K @ merged.motion[index], merged.K @ merged.motion[index+1]))
        points, valid = triangulate_dense_matches(observations, cameras)
        safe = np.where(valid[:, None], points, 0)
        projected = np.einsum('mij,nj->nmi', cameras[:, :, :3], safe) + cameras[None, :, :, 3]
        with np.errstate(divide="ignore", invalid="ignore"):
            predicted = projected[:, :, :2]/projected[:, :, 2:3]
            errors = np.linalg.norm(predicted-observations, axis=2)
        # Keep the same 10-pixel reprojection threshold used by the sparse framework.
        keep = valid & (projected[:, :, 2] > 0).all(axis=1) & (errors <= 10).all(axis=1)
        clouds.append(points[keep])
        records.append({"pair": [image_paths[index].stem, image_paths[index+1].stem],
                        "input_matches": len(matches.T), "valid_triangulations": int(valid.sum()),
                        "retained_points": int(keep.sum()),
                        "mean_reprojection_error_px": float(errors[keep].mean())})
        print(f"Dense pair {index+1}-{index+2}: {keep.sum()}/{len(keep)} points retained", flush=True)
    cloud = np.vstack(clouds)
    np.savez_compressed(OUT / "statue_reconstruction.npz", dense_structure=cloud,
                        sparse_structure=merged.structure, camera_motion=merged.motion,
                        intrinsic_matrix=merged.K, match_idx=merged.match_idx,
                        match_points=merged.match_points)
    plotted = save_plot(cloud)
    result = {"image_count": len(image_paths), "dense_pair_count": len(records),
              "sparse_point_count": len(merged.structure), "dense_point_count": len(cloud),
              "world_frame": "first camera, with first-to-second camera baseline equal to one",
              "bundle_adjustment_scale_normalization": frame_scale,
              "reprojection_threshold_px": 10, "pairs": records,
              "plot": "statue_reconstruction.png", "plot_limits_xyz": PLOT_LIMITS.tolist(),
              "plotted_point_count": plotted, "plot_color": "depth Z",
              "plot_elevation_azimuth": list(PLOT_VIEW), "seconds": time.perf_counter()-started}
    (OUT / "reconstruction_results.json").write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
    print(f"Saved {len(cloud)} dense points in {result['seconds']:.1f} seconds", flush=True)


if __name__ == "__main__":
    main()
