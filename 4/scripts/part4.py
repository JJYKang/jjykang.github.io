import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
from triangulation import (estimate_initial_RT, linear_estimate_3d_point,
                           reprojection_error, jacobian, nonlinear_estimate_3d_point,
                           estimate_RT_from_E, _pose_depth_counts)

DATA = ROOT / "assignment 4 code" / "data"
OUT = ROOT / "media" / "part4"
FOCAL_LENGTH = 719.5459
EXAMPLE_RT = np.array([[0.9736, -0.0988, -0.2056, 0.9994],
                       [0.1019, 0.9948, 0.0045, -0.0089],
                       [0.2041, -0.0254, 0.9786, 0.0331]])


def check_linear_triangulation(k):
    # run the same rounded-pose, first-match check as the supplied p4.py
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


def check_reprojection():
    # verify the supplied residual/Jacobian and a central-difference derivative
    with np.load(OUT / "linear_triangulation_check.npz") as inputs:
        point = inputs["expected_point"]
        image_points = inputs["image_points"]
        cameras = inputs["camera_matrices"]
    error = reprojection_error(point, image_points, cameras)
    derivatives = jacobian(point, cameras)
    expected_error = np.array([-0.0095458, -0.5171407, 0.0059307, 0.501631])
    expected_jacobian = np.array([[154.33943931, 0., -22.42541691],
                                  [0., 154.33943931, 36.51165089],
                                  [141.87950588, -14.27738422, -56.20341644],
                                  [21.9792766, 149.50628901, 32.23425643]])
    error_difference = np.abs(error-expected_error)
    jacobian_difference = np.abs(derivatives-expected_jacobian)
    # Residual references are truncated to six/seven decimals; J has eight.
    if error_difference.max() > 1e-6 or jacobian_difference.max() > 1e-8:
        raise ValueError("Reprojection results do not match the supplied reference values")
    numerical = np.empty_like(derivatives)
    for axis in range(3):
        step = np.zeros(3)
        step[axis] = 1e-6 * max(1., abs(point[axis]))
        numerical[:, axis] = (reprojection_error(point+step, image_points, cameras)
                              - reprojection_error(point-step, image_points, cameras)) / (2*step[axis])
    finite_difference_error = np.abs(derivatives-numerical).max()
    if finite_difference_error > 1e-5:
        raise ValueError("Analytic Jacobian does not agree with central differences")
    filename = "reprojection_check.npz"
    np.savez_compressed(OUT / filename, point=point, image_points=image_points,
                        camera_matrices=cameras, residual=error, jacobian=derivatives,
                        numerical_jacobian=numerical, expected_residual=expected_error,
                        expected_jacobian=expected_jacobian)
    print("Reprojection residual:", error)
    print("Residual/Jacobian reference differences:", error_difference.sum(), jacobian_difference.sum())
    print("Maximum finite-difference Jacobian error:", finite_difference_error)
    return {"check_file": filename, "residual": error.tolist(),
            "jacobian": derivatives.tolist(),
            "residual_reference_sum_absolute_difference": float(error_difference.sum()),
            "jacobian_reference_sum_absolute_difference": float(jacobian_difference.sum()),
            "max_finite_difference_jacobian_error": float(finite_difference_error)}


def check_nonlinear_triangulation():
    # compare DLT and ten-step Gauss-Newton on the supplied four-view test
    image_points = np.load(DATA / "unit_test_image_matches.npy")
    cameras = np.load(DATA / "unit_test_camera_matrix.npy")
    linear = linear_estimate_3d_point(image_points, cameras)
    nonlinear = nonlinear_estimate_3d_point(image_points, cameras)
    linear_error = reprojection_error(linear, image_points, cameras)
    nonlinear_error = reprojection_error(nonlinear, image_points, cameras)
    linear_norm = np.linalg.norm(linear_error)
    nonlinear_norm = np.linalg.norm(nonlinear_error)
    if not nonlinear_norm < linear_norm:
        raise ValueError("Nonlinear triangulation does not improve the supplied test's reprojection error")
    gradient = jacobian(nonlinear, cameras).T @ nonlinear_error
    filename = "nonlinear_triangulation_check.npz"
    np.savez_compressed(OUT / filename, image_points=image_points,
                        camera_matrices=cameras, linear_point=linear,
                        nonlinear_point=nonlinear, linear_residual=linear_error,
                        nonlinear_residual=nonlinear_error, final_gradient=gradient)
    print("Linear/nonlinear reprojection error norms:", linear_norm, nonlinear_norm)
    print("Final Gauss-Newton point:", nonlinear)
    return {"check_file": filename, "iterations": 10,
            "linear_point": linear.tolist(), "nonlinear_point": nonlinear.tolist(),
            "linear_reprojection_error_norm": float(linear_norm),
            "nonlinear_reprojection_error_norm": float(nonlinear_norm),
            "final_gradient_norm": float(np.linalg.norm(gradient))}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    matrices = np.load(DATA / "statue" / "fundamental_matrices.npy",
                       allow_pickle=True, encoding="latin1")[0]
    matches = np.load(DATA / "statue" / "matches_subset.npy",
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
        observations = matches[index].T.reshape(-1, 2, 2)
        pose = estimate_RT_from_E(essential, observations, k)
        counts, failures = _pose_depth_counts(candidates, observations, k)
        selected = int(np.argmin(np.linalg.norm(candidates-pose, axis=(1, 2))))
        if counts[selected] != counts.max() or counts[selected] == 0:
            raise ValueError("Selected pose does not maximize positive depth in both cameras")
        pose_file = f"pair_{index+1}_{index+2}_selected_RT.npy"
        np.save(OUT / pose_file, pose)
        record.update({"correspondences": len(observations),
                       "positive_depth_counts": counts.tolist(),
                       "failed_triangulations": failures.tolist(),
                       "selected_candidate_index": selected,
                       "selected_pose": pose.tolist(), "selected_pose_file": pose_file})
        if index == 0:
            errors = np.max(np.abs(candidates-EXAMPLE_RT), axis=(1, 2))
            best = int(np.argmin(errors))
            if errors[best] > 5.1e-5:
                raise ValueError("No candidate agrees with the supplied four-decimal example")
            record["example_match_index"] = best
            record["example_max_absolute_difference"] = float(errors[best])
            if not np.allclose(pose, EXAMPLE_RT, atol=5.1e-5, rtol=0):
                raise ValueError("Depth-selected pose does not match the supplied example")
            # Repeat the starter's separate one-correspondence pose-selection check.
            test_points = np.load(DATA / "unit_test_image_matches.npy")[:2][None, :, :]
            test_pose = estimate_RT_from_E(essential, test_points, k)
            if not np.allclose(test_pose, EXAMPLE_RT, atol=5.1e-5, rtol=0):
                raise ValueError("The supplied single-match check selected the wrong pose")
            record["single_match_example_max_difference"] = float(np.abs(test_pose-EXAMPLE_RT).max())
            print("First pair candidates:\n", candidates)
            print("Maximum difference from supplied example:", errors[best])
        records.append(record)
        print(f"Pair {index+1}-{index+2}: proper rotations and unit translations verified")
        print("Positive-depth counts:", counts, "; selected candidate:", selected)
    linear_check = check_linear_triangulation(k)
    result = {"focal_length": FOCAL_LENGTH, "intrinsic_matrix": k.tolist(),
              "pose_convention": "X2 = R @ X1 + t; translation has unit length",
              "example_pose": EXAMPLE_RT.tolist(), "records": records,
              "linear_triangulation_check": linear_check,
              "reprojection_check": check_reprojection(),
              "nonlinear_triangulation_check": check_nonlinear_triangulation()}
    (OUT / "results.json").write_text(json.dumps(result, indent=2) + "\n",
                                      encoding="utf-8")


if __name__ == "__main__":
    main()
