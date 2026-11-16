from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(__file__).resolve().parent / "data"
OUTPUT = ROOT / "media" / "part4"
OUTPUT.mkdir(parents=True, exist_ok=True)

import sys
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.image import imread
from sfm_utils import Frame, bundle_adjustment, merge_all_frames

def estimate_initial_RT(E):
    U, _, Vt = np.linalg.svd(E)
    W = np.array([[0., -1, 0], [1, 0, 0], [0, 0, 1]])
    candidates = []
    # combine two rotations with opposite translation directions
    for middle in (W, W.T):
        R = U @ middle @ Vt
        R *= np.linalg.det(R)  # enforce a proper rotation
        for sign in (1, -1):
            candidates.append(np.column_stack((R, sign * U[:, 2])))
    return np.array(candidates)

def linear_estimate_3d_point(image_points, camera_matrices):
    # stack two DLT constraints per image
    A = np.empty((2 * len(image_points), 4))
    A[0::2] = image_points[:, 0:1] * camera_matrices[:, 2] - camera_matrices[:, 0]
    A[1::2] = image_points[:, 1:2] * camera_matrices[:, 2] - camera_matrices[:, 1]
    # solve the homogeneous system and divide by the fourth coordinate
    _, _, Vt = np.linalg.svd(A, full_matrices=False)
    return Vt[-1, :3] / Vt[-1, 3]

def reprojection_error(point_3d, image_points, camera_matrices):
    # project into each image and subtract the observed coordinates
    projected = camera_matrices @ np.append(point_3d, 1)
    predicted = projected[:, :2] / projected[:, 2:3]
    return (predicted - image_points).reshape(-1)

def jacobian(point_3d, camera_matrices):
    projected = camera_matrices @ np.append(point_3d, 1)
    predicted = projected[:, :2] / projected[:, 2:3]
    # apply the quotient rule to each projected coordinate
    J = (camera_matrices[:, :2, :3]
         - predicted[:, :, None] * camera_matrices[:, 2:3, :3])
    J /= projected[:, 2, None, None]
    return J.reshape(-1, 3)

def nonlinear_estimate_3d_point(image_points, camera_matrices):
    point = linear_estimate_3d_point(image_points, camera_matrices)  # DLT initialization
    # perform exactly ten Gauss-Newton updates
    for _ in range(10):
        error = reprojection_error(point, image_points, camera_matrices)
        J = jacobian(point, camera_matrices)
        point -= np.linalg.lstsq(J, error, rcond=None)[0]
    return point

def estimate_RT_from_E(E, image_points, K):
    candidates = estimate_initial_RT(E)
    counts = np.zeros(4, dtype=int)
    first_camera = K @ np.column_stack((np.eye(3), np.zeros(3)))
    for index, RT in enumerate(candidates):
        cameras = np.array([first_camera, K @ RT])
        for matches in image_points:
            point = nonlinear_estimate_3d_point(matches, cameras)
            second_point = RT[:, :3] @ point + RT[:, 3]  # Camera 2 coordinates
            # count points in front of both cameras
            if point[2] > 0 and second_point[2] > 0:
                counts[index] += 1
    return candidates[np.argmax(counts)]

if __name__ == '__main__':
    run_pipeline = True

    image_data_dir = str(DATA / 'statue')
    unit_test_camera_matrix = np.load(DATA / 'unit_test_camera_matrix.npy')
    unit_test_image_matches = np.load(DATA / 'unit_test_image_matches.npy')
    image_paths = [os.path.join(image_data_dir, 'images', x) for x in
        sorted(os.listdir(DATA / 'statue' / 'images')) if '.jpg' in x]
    focal_length = 719.5459
    matches_subset = np.load(os.path.join(image_data_dir,
        'matches_subset.npy'), allow_pickle=True, encoding='latin1')[0,:]
    dense_matches = np.load(os.path.join(image_data_dir, 'dense_matches.npy'),
                               allow_pickle=True, encoding='latin1')
    fundamental_matrices = np.load(os.path.join(image_data_dir,
        'fundamental_matrices.npy'), allow_pickle=True, encoding='latin1')[0,:]

    print("Part A: Rotation and translation candidates")

    K = np.eye(3)
    K[0,0] = K[1,1] = focal_length
    E = K.T.dot(fundamental_matrices[0]).dot(K)
    im0 = imread(image_paths[0])
    im_height, im_width, _ = im0.shape
    example_RT = np.array([[0.9736, -0.0988, -0.2056, 0.9994],
        [0.1019, 0.9948, 0.0045, -0.0089],
        [0.2041, -0.0254, 0.9786, 0.0331]])
    print("Example RT:\n", example_RT)
    estimated_RT = estimate_initial_RT(E)
    print('')
    print("Estimated RT:\n", estimated_RT)

    print('Part B: Linear triangulation')

    camera_matrices = np.zeros((2, 3, 4))
    camera_matrices[0, :, :] = K.dot(np.hstack((np.eye(3), np.zeros((3,1)))))
    camera_matrices[1, :, :] = K.dot(example_RT)
    unit_test_matches = matches_subset[0][:,0].reshape(2,2)
    estimated_3d_point = linear_estimate_3d_point(unit_test_matches.copy(),
        camera_matrices.copy())
    expected_3d_point = np.array([0.6774, -1.1029, 4.6621])
    print("Difference: ", np.fabs(estimated_3d_point - expected_3d_point).sum())

    print('Part C: Reprojection error and Jacobian')

    estimated_error = reprojection_error(
            expected_3d_point, unit_test_matches, camera_matrices)
    estimated_jacobian = jacobian(expected_3d_point, camera_matrices)
    expected_error = np.array((-0.0095458, -0.5171407,  0.0059307,  0.501631))
    print("Error Difference: ", np.fabs(estimated_error - expected_error).sum())
    expected_jacobian = np.array([[ 154.33943931, 0., -22.42541691],
         [0., 154.33943931, 36.51165089],
         [141.87950588, -14.27738422, -56.20341644],
         [21.9792766, 149.50628901, 32.23425643]])
    print("Jacobian Difference: ", np.fabs(estimated_jacobian
        - expected_jacobian).sum())

    print('Part D: Nonlinear triangulation')

    estimated_3d_point_linear = linear_estimate_3d_point(
        unit_test_image_matches.copy(), unit_test_camera_matrix.copy())
    estimated_3d_point_nonlinear = nonlinear_estimate_3d_point(
        unit_test_image_matches.copy(), unit_test_camera_matrix.copy())
    error_linear = reprojection_error(
        estimated_3d_point_linear, unit_test_image_matches,
        unit_test_camera_matrix)
    print("Linear method error:", np.linalg.norm(error_linear))
    error_nonlinear = reprojection_error(
        estimated_3d_point_nonlinear, unit_test_image_matches,
        unit_test_camera_matrix)
    print("Nonlinear method error:", np.linalg.norm(error_nonlinear))

    print("Part E: Selected camera pose")

    estimated_RT = estimate_RT_from_E(E,
        np.expand_dims(unit_test_image_matches[:2,:], axis=0), K)
    print("Example RT:\n", example_RT)
    print('')
    print("Estimated RT:\n", estimated_RT)

    if not run_pipeline:
        sys.exit()

    print('Part F: Statue reconstruction')

    # estimate and merge the camera frames
    frames = [0] * (len(image_paths) - 1)
    for i in range(len(image_paths)-1):
        frames[i] = Frame(matches_subset[i].T, focal_length,
                fundamental_matrices[i], im_width, im_height)
        bundle_adjustment(frames[i])
    merged_frame = merge_all_frames(frames)

    camera_matrices = np.zeros((2,3,4))
    # triangulate the dense matches for every adjacent pair
    dense_structure = np.zeros((0,3))
    for i in range(len(dense_matches)):
        matches = dense_matches[i]
        camera_matrices[0,:,:] = merged_frame.K.dot(
            merged_frame.motion[i,:,:])
        camera_matrices[1,:,:] = merged_frame.K.dot(
                merged_frame.motion[i+1,:,:])
        points_3d = np.zeros((matches.shape[1], 3))
        use_point = np.array([True]*matches.shape[1])
        for j in range(matches.shape[1]):
            points_3d[j,:] = nonlinear_estimate_3d_point(
                matches[:,j].reshape((2,2)), camera_matrices)
        dense_structure = np.vstack((dense_structure, points_3d[use_point,:]))

    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111, projection='3d')
    # use smaller points and depth coloring
    ax.scatter(dense_structure[:,0], dense_structure[:,1], dense_structure[:,2],
        c=dense_structure[:,2], cmap='viridis', vmin=0, vmax=10,
        depthshade=False, s=.4, linewidths=0)
    # tighten the view around the statue
    ax.set_xlim(-2.25, 2.25)
    ax.set_ylim(-1.5, 3.5)
    ax.set_zlim(0, 10)
    ax.view_init(-100, 90)

    fig.savefig(OUTPUT / 'statue_reconstruction.png', dpi=150)
    plt.close(fig)
