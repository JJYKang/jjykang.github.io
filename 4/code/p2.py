from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "assignment 4 code" / "data"
OUTPUT = ROOT / "media" / "part2"
OUTPUT.mkdir(parents=True, exist_ok=True)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread
from p1 import normalized_eight_point_alg, plot_epipolar_lines_on_images
from epipolar_utils import get_data_from_txt_file, compute_rectified_image

def compute_epipole(F):
    # solve F.T @ e = 0 and normalize the homogeneous coordinate
    U, _, _ = np.linalg.svd(F)
    e = U[:, -1]
    return e / e[2]

def compute_H(e, im):
    # move the image center to the origin
    height, width = im.shape[:2]
    T = np.array([[1., 0, -width / 2], [0, 1, -height / 2], [0, 0, 1]])
    e = T @ e
    # rotate the epipole onto the horizontal axis
    alpha = 1 if e[0] >= 0 else -1
    cosine, sine = alpha * e[:2] / np.linalg.norm(e[:2])
    R = np.array([[cosine, sine, 0], [-sine, cosine, 0], [0, 0, 1]])
    rotated = R @ e
    # map the epipole to infinity
    G = np.eye(3)
    G[2, 0] = -rotated[2] / rotated[0]
    return np.linalg.inv(T) @ G @ R @ T  # restore the image center

def compute_matching_homographies(e2, F, im2, points1, points2):
    H2 = compute_H(e2, im2)
    # construct M = [e2]_x F + e2 v.T, with v = (1, 1, 1)
    x, y, z = e2
    cross = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    M = cross @ F + np.outer(e2, np.ones(3))
    # transform correspondences into rectified coordinates
    mapped1 = points1 @ (H2 @ M).T
    mapped2 = points2 @ H2.T
    mapped1 /= mapped1[:, 2:3]
    mapped2 /= mapped2[:, 2:3]
    # fit the horizontal correction while preserving scanlines
    a = np.linalg.lstsq(mapped1, mapped2[:, 0], rcond=None)[0]
    HA = np.eye(3)
    HA[0] = a
    return HA @ H2 @ M, H2

if __name__ == '__main__':
    im_set = str(DATA / 'set1')
    im1 = imread(im_set+'/image1.jpg')
    im2 = imread(im_set+'/image2.jpg')
    points1 = get_data_from_txt_file(im_set+'/pt_2D_1.txt')
    points2 = get_data_from_txt_file(im_set+'/pt_2D_2.txt')
    assert (points1.shape == points2.shape)

    F = normalized_eight_point_alg(points1, points2)

    e1 = compute_epipole(F.T)
    e2 = compute_epipole(F)
    print("e1", e1)
    print("e2", e2)

    H1, H2 = compute_matching_homographies(e2, F, im2, points1, points2)
    print('')

    new_points1 = H1.dot(points1.T)
    new_points2 = H2.dot(points2.T)
    new_points1 /= new_points1[2,:]
    new_points2 /= new_points2[2,:]
    new_points1 = new_points1.T
    new_points2 = new_points2.T
    rectified_im1, offset1 = compute_rectified_image(im1, H1)
    rectified_im2, offset2 = compute_rectified_image(im2, H2)
    new_points1 -= offset1 + (0,)
    new_points2 -= offset2 + (0,)

    F_new = normalized_eight_point_alg(new_points1, new_points2)
    plot_epipolar_lines_on_images(new_points1, new_points2, rectified_im1, rectified_im2, F_new)
    plt.savefig(OUTPUT / 'set1_rectified_pair.png', dpi=150)
    plt.close('all')
