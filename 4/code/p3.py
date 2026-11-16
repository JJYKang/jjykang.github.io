from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(__file__).resolve().parent / "data"
OUTPUT = ROOT / "media" / "part3"
OUTPUT.mkdir(parents=True, exist_ok=True)

import numpy as np
from matplotlib.image import imread
import matplotlib.pyplot as plt
from epipolar_utils import *

def factorization_method(points_im1, points_im2):
    # stack and center the image measurements
    W = np.vstack((points_im1[:, :2].T, points_im2[:, :2].T))
    W -= W.mean(axis=1, keepdims=True)
    U, singular_values, Vt = np.linalg.svd(W, full_matrices=False)
    # retain three components and split their scale between the factors
    root = np.sqrt(singular_values[:3])
    motion = U[:, :3] * root
    structure = root[:, None] * Vt[:3]
    return structure, motion

if __name__ == '__main__':
    for im_set in [str(DATA / 'set1'), str(DATA / 'set1_subset')]:

        im1 = imread(im_set+'/image1.jpg')
        im2 = imread(im_set+'/image2.jpg')
        points_im1 = get_data_from_txt_file(im_set + '/pt_2D_1.txt')
        points_im2 = get_data_from_txt_file(im_set + '/pt_2D_2.txt')
        points_3d = get_data_from_txt_file(im_set + '/pt_3D.txt')
        assert (points_im1.shape == points_im2.shape)

        structure, motion = factorization_method(points_im1, points_im2)

        fig = plt.figure(figsize=(10, 5))
        ax = fig.add_subplot(121, projection = '3d')
        scatter_3D_axis_equal(structure[0,:], structure[1,:], structure[2,:], ax)
        ax.set_title('Factorization Method')
        ax = fig.add_subplot(122, projection = '3d')
        scatter_3D_axis_equal(points_3d[:,0], points_3d[:,1], points_3d[:,2], ax)
        ax.set_title('Ground Truth')

        W = np.vstack((points_im1[:, :2].T, points_im2[:, :2].T))
        W -= W.mean(axis=1, keepdims=True)
        print(Path(im_set).name, "singular values:", np.linalg.svd(W, compute_uv=False))
        fig.savefig(OUTPUT / f"{Path(im_set).name}_reconstruction_comparison.png", dpi=150)
        plt.close(fig)
