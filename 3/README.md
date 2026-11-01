# Assignment 3

- `src/`: reusable algorithms (`homography.py` for A.2, `warping.py` for A.3).
- `scripts/`: section-specific inputs, parameters, and result generation.
- `data/`: original photographs and input data.
- `media/part_a/a2/`: numbered correspondences, matrix, and reprojection errors.
- `media/part_a/a3/`: rectification inputs, interpolation comparisons, and timings.
- `media/part_a/a4/`: clinic mosaic, direct overlay, and cropped webpage views.
- `media/part_b/b1/`: Harris/ANMS overlays and saved corner coordinates.
- `media/part_b/b2/`: normalized descriptors and six illustrated features.
- `media/part_b/b3/`: adjacent-image feature matches and visualizations.
- `media/part_b/b4/`: RANSAC inliers, manual/automatic mosaics, and comparisons.
- `index.html`, `style.css`, `page.js`: static assignment webpage.

From inside `3/`:

```sh
pip install -r requirements.txt
python scripts/part_a2.py
python scripts/part_a3.py
python scripts/part_a4.py
python scripts/part_b1.py
python scripts/part_b2.py
python scripts/part_b3.py
python scripts/part_b4.py
```

The script also runs from the portfolio root with `python 3/scripts/part_a2.py`.
Manual landmarks are fixed in the script at the original image resolution.
`computeH(im1_pts, im2_pts)` maps the first point set to the second; image
coordinates are `(x, y)` from the top-left pixel center. Generated results
include `results.json` and `results.js`, which supplies the webpage matrix
and RMSE without a Python runtime on GitHub Pages.

The A.3 script uses four manually selected corners in each of two source images
and maps them to fixed output rectangles. It benchmarks both interpolation
methods with five timed runs after a warm-up and saves the median runtimes.
The warp functions return `(image, valid_mask, origin)`, with an automatically
computed canvas by default. Set `output_shape=(height, width)` and optionally
`origin=(left, top)` to use a fixed destination canvas.

A.4 uses `src/mosaicing.py` to place the two clinic images on one canvas and
blend them with weights based on distance to each source border. Image 1 is
the reference. The inputs are resized to 1024x768, and the A.2 homography is
adjusted to those coordinates. Full transparent canvases and fixed display
crops are exported; crop coordinates and transforms are in `results.json`.

B.1 uses the supplied `src/harris.py` on 1024x768 grayscale images, with a
20-pixel boundary exclusion. `src/features.py` implements ANMS with robustness
factor 0.9 and selects 500 corners per image. Each `.npz` saves n-by-2 `(x, y)`
points, strengths, suppression radii, selected indices, and image dimensions for
the descriptor stage. The supplied Harris file is preserved unchanged.

B.2 reads the saved B.1 corners, extracts 40x40 grayscale windows, applies a
Gaussian with sigma 2 and reflected boundaries, and samples every fifth pixel
starting at patch index 2 to form 8x8 descriptors. Each descriptor is normalized
to zero mean and unit standard deviation. Border windows and patches with
standard deviation at most 1e-8 are omitted. Saved `.npz` files contain the
64-value descriptors, their `(x, y)` points, indices into the ANMS selection,
and image dimensions. Run B.1 before B.2.

B.3 matches Image 1 to Image 2 and Image 2 to Image 3. For each descriptor,
`match_descriptors` finds the nearest two target descriptors using SSD and
accepts a nearest/second-nearest SSD ratio below 0.5. Ambiguous duplicate
neighbors are rejected. Each `.npz` stores matched descriptor indices, ratios,
and paired `(x, y)` coordinates for RANSAC. Run B.2 before B.3.

B.4 implements four-point RANSAC in `src/ransac.py` with a 3-pixel forward
reprojection threshold, seed 671, confidence 0.999, and a 5,000-trial cap.
Collinear samples are skipped, and the selected consensus is refined by
least-squares fitting. Image 2 is the reference for the three-image mosaic:
Image 1 uses H12 and Image 3 uses inverse(H23). Both the automatic result and
a manual baseline use the same canvas, bilinear interpolation, and feathering.
The manual points are fixed in `scripts/part_b4.py` at 1024x768. Saved outputs
include inlier masks, errors, matrices, and full canvases; webpage views use
fixed crops in reference-image coordinates. Run B.3 before B.4.
