"""Generate pose candidates and check linear triangulation for Problem 4(a-b)."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
from triangulation import estimate_initial_RT, linear_estimate_3d_point

DATA = ROOT / "assignment 4 code" / "data"
OUT = ROOT / "media" / "part4"
FOCAL_LENGTH = 719.5459
EXAMPLE_RT = np.array([[0.9736, -0.0988, -0.2056, 0.9994],
                       [0.1019, 0.9948, 0.0045, -0.0089],
                       [0.2041, -0.0254, 0.9786, 0.0331]])


def check_linear_triangulation(k):
    """Run the same rounded-pose, first-match check as the supplied p4.py."""
    matches = np.load(DATA / "statue" / "matches_subset.npy",
                      allow_pickle=True, encoding="latin1")[0]
    image_points = matches[0][:, 0].reshape(2, 2)
    cameras = np.stack((k @ np.column_stack((np.eye(3), np.zeros(3))), k @ EXAMPLE_RT))
    estimated = linear_estimate_3d_point(image_points, cameras)
    expected = np.array([0.6774, -1.1029, 4.6621])
    difference = np.abs(estimated-expected)
    # The starter reference is approximate: these supplied inputs differ by
    # about 0.0029 in summed absolute coordinates under homogeneous DLT.
    if difference.sum() > 5e-3:
        raise ValueError("Linear triangulation is not close to the supplied reference point")
    filename = "linear_triangulation_check.npz"
    np.savez_compressed(OUT / filename, image_points=image_points,
                        camera_matrices=cameras, estimated_point=estimated,
                        expected_point=expected)
    print("Linear 3D estimate:", estimated)
    print("Sum of differences from supplied point:", difference.sum())
    return {"check_file": filename, "estimated_point": estimated.tolist(),
            "expected_point": expected.tolist(),
            "sum_absolute_difference": float(difference.sum()),
            "max_absolute_difference": float(difference.max())}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    matrices = np.load(DATA / "statue" / "fundamental_matrices.npy",
                       allow_pickle=True, encoding="latin1")[0]
    k = np.diag([FOCAL_LENGTH, FOCAL_LENGTH, 1.])
    records = []
    for index, f in enumerate(matrices):
        essential = k.T @ f @ k
        candidates = estimate_initial_RT(essential)
        rotations = candidates[:, :, :3]
        orthogonality = np.max(np.abs(rotations.transpose(0, 2, 1) @ rotations - np.eye(3)))
        determinants = np.linalg.det(rotations)
        lengths = np.linalg.norm(candidates[:, :, 3], axis=1)
        if (orthogonality > 1e-12 or not np.allclose(determinants, 1, atol=1e-12, rtol=0)
                or not np.allclose(lengths, 1, atol=1e-12, rtol=0)):
            raise ValueError("Pose candidates must have proper rotations and unit translations")
        filename = f"pair_{index+1}_{index+2}_initial_RT.npy"
        np.save(OUT / filename, candidates)
        record = {"pair": [f"B{21+index}", f"B{22+index}"],
                  "essential_matrix": essential.tolist(),
                  "candidate_poses": candidates.tolist(), "candidate_file": filename,
                  "rotation_determinants": determinants.tolist(),
                  "max_rotation_orthogonality_error": float(orthogonality),
                  "translation_lengths": lengths.tolist()}
        if index == 0:
            errors = np.max(np.abs(candidates-EXAMPLE_RT), axis=(1, 2))
            best = int(np.argmin(errors))
            if errors[best] > 5.1e-5:
                raise ValueError("No candidate agrees with the supplied four-decimal example")
            record["example_match_index"] = best
            record["example_max_absolute_difference"] = float(errors[best])
            print("First pair candidates:\n", candidates)
            print("Maximum difference from supplied example:", errors[best])
        records.append(record)
        print(f"Pair {index+1}-{index+2}: proper rotations and unit translations verified")
    result = {"focal_length": FOCAL_LENGTH, "intrinsic_matrix": k.tolist(),
              "pose_convention": "X2 = R @ X1 + t; translation has unit length",
              "example_pose": EXAMPLE_RT.tolist(), "records": records,
              "linear_triangulation_check": check_linear_triangulation(k)}
    (OUT / "results.json").write_text(json.dumps(result, indent=2) + "\n",
                                      encoding="utf-8")


if __name__ == "__main__":
    main()
