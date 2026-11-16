from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "assignment 4 code" / "data"
OUTPUT = ROOT / "media" / "part1"
OUTPUT.mkdir(parents=True, exist_ok=True)

import numpy as np
from matplotlib.image import imread
import matplotlib.pyplot as plt
from epipolar_utils import *

def lls_eight_point_alg(points1, points2):
    # build the correspondence equations
    A = (points2[:, :, None] * points1[:, None, :]).reshape(-1, 9)
    # solve for F using the smallest singular vector
    _, _, Vt = np.linalg.svd(A, full_matrices=(len(points1) == 8))
    F = Vt[-1].reshape(3, 3)
    U, singular_values, Vt = np.linalg.svd(F)
    singular_values[-1] = 0  # enforce the rank-two constraint
    F = (U * singular_values) @ Vt
    # fix the scale and sign for displaying F
    F /= np.linalg.norm(F)
    if F.flat[np.argmax(np.abs(F))] < 0:
        F = -F
    return F

def normalized_eight_point_alg(points1, points2):
    transforms = []
    normalized = []
    for points in (points1, points2):
        # center each set and scale its mean distance to sqrt(2)
        center = points[:, :2].mean(axis=0)
        distance = np.linalg.norm(points[:, :2] - center, axis=1).mean()
        scale = np.sqrt(2) / distance
        T = np.array([[scale, 0, -scale * center[0]],
                      [0, scale, -scale * center[1]], [0, 0, 1]])
        transforms.append(T)
        normalized.append(points @ T.T)
    F = lls_eight_point_alg(*normalized)  # includes rank-two enforcement
    F = transforms[1].T @ F @ transforms[0]  # return to pixel coordinates
    F /= np.linalg.norm(F)
    if F.flat[np.argmax(np.abs(F))] < 0:
        F = -F
    return F

def plot_epipolar_lines_on_images(points1, points2, im1, im2, F):

    def plot_epipolar_lines_on_image(points1, points2, im, F):
        im_height = im.shape[0]
        im_width = im.shape[1]
        lines = F.T.dot(points2.T)
        plt.imshow(im, cmap='gray')
        for line in lines.T:
            a,b,c = line
            xs = [1, im.shape[1]-1]
            ys = [(-c-a*x)/b for x in xs]
            plt.plot(xs, ys, 'r')
        for i in range(points1.shape[0]):
            x,y,_ = points1[i]
            plt.plot(x, y, '*b')
        plt.axis([0, im_width, im_height, 0])

    new_figsize = (10, 5)
    fig = plt.figure(figsize=new_figsize)
    plt.subplot(121)
    plot_epipolar_lines_on_image(points1, points2, im1, F)
    plt.axis('off')
    plt.subplot(122)
    plot_epipolar_lines_on_image(points2, points1, im2, F.T)
    plt.axis('off')

def compute_distance_to_epipolar_lines(points1, points2, F):
    lines = points2 @ F  # corresponding lines in Image 1
    # perpendicular distance, averaged over matching pairs
    distances = np.abs(np.sum(points1 * lines, axis=1))
    distances /= np.linalg.norm(lines[:, :2], axis=1)
    return distances.mean()

if __name__ == '__main__':
    for im_set in [str(DATA / 'set1'), str(DATA / 'set2')]:

        print("Set:", im_set)

        im1 = imread(im_set+'/image1.jpg')
        im2 = imread(im_set+'/image2.jpg')
        points1 = get_data_from_txt_file(im_set+'/pt_2D_1.txt')
        points2 = get_data_from_txt_file(im_set+'/pt_2D_2.txt')
        assert (points1.shape == points2.shape)

        F_lls = lls_eight_point_alg(points1, points2)
        print("Fundamental Matrix from LLS  8-point algorithm:\n", F_lls)
        print("Distance to lines in image 1 for LLS:", \
            compute_distance_to_epipolar_lines(points1, points2, F_lls))
        print("Distance to lines in image 2 for LLS:", \
            compute_distance_to_epipolar_lines(points2, points1, F_lls.T))

        F_normalized = normalized_eight_point_alg(points1, points2)

        pFp = [points2[i].dot(F_normalized.dot(points1[i]))
            for i in range(points1.shape[0])]
        print("p'^T F p =", np.abs(pFp).max())
        print("Fundamental Matrix from normalized 8-point algorithm:\n", \
            F_normalized)
        print("Distance to lines in image 1 for normalized:", \
            compute_distance_to_epipolar_lines(points1, points2, F_normalized))
        print("Distance to lines in image 2 for normalized:", \
            compute_distance_to_epipolar_lines(points2, points1, F_normalized.T))

        plot_epipolar_lines_on_images(points1, points2, im1, im2, F_lls)
        plot_epipolar_lines_on_images(points1, points2, im1, im2, F_normalized)

        for number, method in zip(plt.get_fignums(), ('linear', 'normalized')):
            plt.figure(number).savefig(OUTPUT / f"{Path(im_set).name}_{method}_epipolar.png", dpi=150)
        plt.close('all')
