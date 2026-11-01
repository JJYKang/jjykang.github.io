"""Estimate robust homographies and compare manual and automatic house mosaics."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
from PIL import Image
from homography import computeH
from ransac import ransac_homography, reprojection_errors
from mosaicing import create_mosaic
from warping import warp_bounds
from part_b3 import draw_matches

OUT = ROOT / "media" / "part_b" / "b4"
THRESHOLD = 3.0
MAX_ITERATIONS = 5000
CONFIDENCE = 0.999
SEED = 671
COMPARISON_VIEW = (-500, 150, 1350, 700)  # Reference-image coordinates.
AUTOMATIC_VIEW = (-600, 100, 2200, 700)
# Manually selected landmarks at 1024x768, independent of automatic matches.
# Each source is mapped to the reference image 2.
MANUAL = {
    1: (np.array([[608,282],[609,365],[478,379],[563,377],[478,434],
                  [563,434],[694,317],[678,442],[750,375],[751,480]], dtype=float),
        np.array([[257,266],[257,354],[94,375],[202,375],[94,435],
                  [202,435],[351,318],[336,440],[395,376],[397,471]], dtype=float)),
    3: (np.array([[194,255],[248,259],[194,293],[248,297],[352,264],
                  [416,268],[352,308],[416,312],[176,364],[246,369],
                  [336,371],[415,375]], dtype=float),
        np.array([[615,263],[664,262],[615,299],[664,299],[764,260],
                  [823,260],[764,299],[823,298],[600,366],[671,367],
                  [743,369],[818,369]], dtype=float)),
}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    pil_images = []
    for index in (1, 2, 3):
        with Image.open(ROOT / "data" / f"mosaic3-{index}.jpg") as source:
            image = source.convert("RGB")
            image.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
            pil_images.append(image)
    images = [np.asarray(image, dtype=float) / 255 for image in pil_images]
    records, automatic_pairs = [], []
    for first, second in [(1, 2), (2, 3)]:
        with np.load(ROOT / "media" / "part_b" / "b3" / f"pair_{first}_{second}_matches.npz") as saved:
            p, q = saved["points1"], saved["points2"]
        h, inliers, errors, trials = ransac_homography(
            p, q, THRESHOLD, MAX_ITERATIONS, CONFIDENCE, SEED)
        automatic_pairs.append(h)
        name = f"pair_{first}_{second}"
        np.savez_compressed(OUT / f"{name}_ransac.npz", homography=h, points1=p, points2=q,
                            inliers=inliers, errors=errors)
        draw_matches([pil_images[first - 1], pil_images[second - 1]], p[inliers], q[inliers],
                     OUT / f"{name}_inliers.jpg")
        record = {"pair": [first, second], "matches": len(p), "inliers": int(inliers.sum()),
                  "rmse_px": float(np.sqrt(np.mean(errors[inliers] ** 2))),
                  "trials": trials, "homography": h.tolist()}
        records.append(record)
        print(f"Images {first}-{second}: {record['inliers']}/{len(p)} inliers; RMSE {record['rmse_px']:.3f} px")
    automatic = [automatic_pairs[0], np.eye(3), np.linalg.inv(automatic_pairs[1])]
    manual = [computeH(*MANUAL[1]), np.eye(3), computeH(*MANUAL[3])]
    manual_records = []
    for index in (1, 3):
        p, q = MANUAL[index]
        error = reprojection_errors(p, q, manual[index - 1])
        draw_matches([pil_images[index - 1], pil_images[1]], p, q,
                     OUT / f"manual_{index}_to_2.jpg")
        manual_records.append({"source_image": index, "points1": p.tolist(), "points2": q.tolist(),
                               "rmse_px": float(np.sqrt(np.mean(error ** 2)))})
    # Use one canvas for both estimates so scale and framing match.
    bounds = [warp_bounds(images[i].shape, transforms[i])
              for transforms in (manual, automatic) for i in range(3)]
    left = min(point[0] for shape, point in bounds)
    top = min(point[1] for shape, point in bounds)
    right = max(point[0] + shape[1] for shape, point in bounds)
    bottom = max(point[1] + shape[0] for shape, point in bounds)
    shape, origin = (bottom - top, right - left), (left, top)
    for name, transforms in [("manual", manual), ("automatic", automatic)]:
        mosaic, _, valid, _ = create_mosaic(images, transforms, shape, origin)
        rgba = np.dstack((np.round(np.clip(mosaic, 0, 1) * 255).astype(np.uint8), valid.astype(np.uint8) * 255))
        output = Image.fromarray(rgba)
        output.save(OUT / f"{name}_mosaic.png")
        def save_view(view, filename):
            x0, y0, x1, y1 = view
            box = (x0 - left, y0 - top, x1 - left, y1 - top)
            if not valid[box[1]:box[3], box[0]:box[2]].all():
                raise ValueError(f"{filename} contains empty canvas pixels")
            output.crop(box).save(OUT / filename)
        save_view(COMPARISON_VIEW, f"{name}_comparison.png")
        if name == "automatic":
            save_view(AUTOMATIC_VIEW, "automatic_crop.png")
    data = {"threshold_px": THRESHOLD, "max_iterations": MAX_ITERATIONS, "confidence": CONFIDENCE,
            "seed": SEED, "reference_image": 2, "records": records, "manual_points": manual_records,
            "automatic_homographies": [h.tolist() for h in automatic],
            "manual_homographies": [h.tolist() for h in manual],
            "output_shape": [int(v) for v in shape], "origin": [int(v) for v in origin],
            "comparison_view": list(COMPARISON_VIEW), "automatic_view": list(AUTOMATIC_VIEW)}
    (OUT / "results.json").write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    (OUT / "results.js").write_text("window.ransacData = " + json.dumps(data) + ";\n", encoding="utf-8")
    print("Canvas", shape, origin, "manual errors", [r["rmse_px"] for r in manual_records])


if __name__ == "__main__":
    main()
