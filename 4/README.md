# Assignment 4

Fundamental matrix estimation, image rectification, factorization, and triangulation.
Problem 1 is implemented in `src/fundamental.py`, with result generation in
`scripts/part1.py`. Problem 2 uses `src/rectification.py` and `src/warping.py`, with
Set 1 checks and image exports in `scripts/part2.py`. Problem 3(a) uses
`src/factorization.py`, with full/subset result generation in `scripts/part3.py`.
Remaining algorithms and result scripts will be added as
each part is implemented. The supplied
`assignment 4 code/` folder is preserved unchanged.

```text
4/
  index.html                 # report, organized by PDF problem and subpart
  style.css                  # Assignment 3 styling
  requirements.txt
  syde671 Assignment 4.pdf
  assignment 4 code/         # original starter code, notebook, helpers, and data
  src/                       # reusable algorithms
  scripts/                   # section inputs, checks, plots, and exports
  media/
    part1/                   # matrices, epipolar lines, distances
    part2/                   # epipoles, homographies, rectified images
    part3/                   # reconstructions and singular values
    part4/                   # triangulation checks and reconstructed statue
```

## Planned implementation

These files will be added when implementing the corresponding section:

| Problem | Reusable code | Result script | Inputs | Outputs |
| --- | --- | --- | --- | --- |
| 1 | `src/fundamental.py` | `scripts/part1.py` | `set1`, `set2` | Both eight-point matrices, distances in both images, epipolar plots |
| 2 | `src/rectification.py`, reusing Problem 1 | `scripts/part2.py` | `set1` | Epipoles, homographies, rectified pair |
| 3 | `src/factorization.py` | `scripts/part3.py` | `set1`, `set1_subset` | Full/subset reconstructions beside ground truth, all four singular values for each |
| 4 | `src/triangulation.py` | `scripts/part4.py` | `statue`, unit-test arrays | Pose candidates, point/error/Jacobian checks, nonlinear comparison, final statue |

Follow Assignments 2 and 3: keep numerical functions independent of paths,
plotting, and exports. Keep dataset choices, fixed parameters, validation, and
figure generation in the section scripts. Use short docstrings and comments
that explain why a step is needed.

Scripts will resolve paths from `Path(__file__).resolve().parents[1]` so they
work from `4/` or the portfolio root. Read original inputs from
`ROOT / "assignment 4 code" / "data"`; do not duplicate or rename the data.
Preserve starter function names and argument conventions in our algorithms.
If completed starter files are needed for submission, prepare adapters after
implementation rather than maintaining a second algorithm implementation.

The supplied loader reads 2D text rows as `(y, x)` and returns homogeneous
`(x, y, 1)` points. The fundamental-matrix convention is `p2.T @ F @ p1 = 0`
for individual column vectors. Record coordinate, camera, scale, and array-shape
conventions beside the relevant implementation.

Problem 4 requires ten Gauss-Newton updates initialized by the linear estimate.
Use the supplied numeric examples as checks before running the full pipeline.
Audit helper imports and the final 3D plotting call for compatibility during
integration. The starter dense loop omits its last adjacent image pair; check
that bound when preparing the final reconstruction.

## Webpage and result files

Save generated figures and arrays in `media/partN/`. Write `results.json` for
the full record and `results.js` for measurements used by the webpage. Commit
the displayed assets so GitHub Pages needs no Python runtime. Add result script
tags only after those files exist.

Give every PDF subpart its own heading, including separate headings for coding
and writing questions. For coding parts, use a short descriptive phrase about
the implementation, followed by a link labeled with the filename. For example:
"SVD solution with rank-two enforcement — fundamental.py". Avoid full-sentence
introductions such as "The code builds", vague role labels, and "View code" links;
writing parts contain the requested explanations, matrices, figures, and
discussion. Keep shared inputs at the start of the problem. Add code links for
unfinished parts when their implementations exist; retain only their headings
until then. Do not add page introductions, section introductions, question
restatements, or descriptions of future work. Keep text focused on the answers
and the captions or conventions needed to interpret results.
Match Assignment 3's white page, header, blue links, responsive
image grids, and tables.

## Writing conventions

- Introduce the concrete input and what the method computes.
- Explain algorithms in short numbered steps with a bold action, followed by
  what it does and why. Include equations when they clarify the step.
- State actual dimensions, parameters, normalization, and coordinate conventions.
- Place figures and measurements beside the explanation they support.
- Describe visible results in plain sentences, then explain their cause or limits.
- Separate expected properties from observations. Write comparisons after checking outputs.
- Keep captions brief: identify the image or method and the detail to compare.

## Setup and preview

With a local Python installation, from `4/`:

```sh
python -m pip install -r requirements.txt
```

Generate the Problem 1 matrices, distance measurements, and plots from inside `4/`:

```sh
python scripts/part1.py
```

From the portfolio root, use `python 4/scripts/part1.py`. This runs both the
linear and normalized eight-point algorithms. Both accept homogeneous n-by-3
points and enforce rank two by setting the smallest singular value of the fitted
matrix to zero. The normalized method independently centers each point set and
scales its mean distance to sqrt(2), fits the matrix in those coordinates, and
returns `T2.T @ F_normalized @ T1` in pixel coordinates. Saved matrices have
unit Frobenius norm and a positive largest-magnitude entry. Outputs are
`media/part1/set1_linear_F.npy`, `set2_linear_F.npy`,
`set1_normalized_F.npy`, `set2_normalized_F.npy`, and `results.json`.
Each JSON record identifies its dataset and method.
`compute_distance_to_epipolar_lines(points1, points2, F)` returns the mean
perpendicular distance in pixels in Image 1. Swap the point sets and pass `F.T`
for Image 2. It accepts one or more pairs, rejects undefined lines, and divides
by the line-normal length so the result does not depend on the scale of F.
The script also saves four paired epipolar-line plots as
`media/part1/{dataset}_{method}_epipolar.png` and stores both image distances
in each JSON record. Red lines and blue points are drawn with Pillow at the
original image resolution; the plotting code clips lines at the image borders,
including vertical lines.
No notebook runtime is required for the static report.

Generate the Problem 2 epipoles, homographies, and rectified images from inside `4/`:

```sh
python scripts/part2.py
```

From the portfolio root, use `python 4/scripts/part2.py`. The script recomputes
the normalized Set 1 fundamental matrix and saves both epipoles and null-space
residuals in `media/part2/results.json`. `compute_epipole(F)` returns the left
null vector of F, satisfying `F.T @ e = 0`, as in the starter convention.
Image 1 uses `compute_epipole(F.T)` and Image 2 uses `compute_epipole(F)`.
Finite epipoles are scaled to `(x, y, 1)`; an epipole at infinity is returned
as a unit direction `(x, y, 0)` without division by zero.

`compute_H(e, im)` composes `T_inverse @ G @ R @ T`, using the image center
`(width / 2, height / 2)` and the rotation sign specified in the handout.
It maps the epipole to horizontal infinity and leaves the image center fixed.
An epipole already at infinity needs no projective step. An epipole at the
image center is rejected because this construction is undefined there.
`compute_matching_homographies(e2, F, im2, points1, points2)` forms
`M = [e2]_x @ F + outer(e2, [1, 1, 1])` and fits the first row of an affine
correction by least squares on the transformed x-coordinates. It returns
`H1 = HA @ H2 @ M` and `H2`, preserving the rectified y-coordinates.
The script saves `media/part2/set1_H1.npy` and `set1_H2.npy`, and records both
homographies, transformed points, vertical disparities, and the rectified
fundamental matrix in `results.json`. It checks that the transformed fundamental
matrix has the canonical horizontal-scanline form.

`src/warping.py` reuses our Assignment 3 inverse-warping implementation unchanged.
The script applies bilinear interpolation at the original pixel scale, with
separate horizontal canvas bounds but one shared vertical origin and height.
This keeps matching scanlines at the same displayed y-coordinate. White marks
uncovered pixels. Clean images are saved as `set1_rectified_1.png` and
`set1_rectified_2.png`; `set1_rectified_pair.png` adds six horizontal guides and
their corresponding point markers. Canvas origins, dimensions, and guide-point
indices are recorded in `results.json`.

Generate the Problem 3(a) full-set and subset factorizations from inside `4/`:

```sh
python scripts/part3.py
```

From the portfolio root, use `python 4/scripts/part3.py`. The measurement matrix
has rows `(x1, y1, x2, y2)`, each centered independently. The top three SVD
components are split as `motion = U3 @ sqrt(S3)` and
`structure = sqrt(S3) @ V3.T`. The starter-compatible return order is
`(structure, motion)`, with shapes `3-by-N` and `4-by-3`. This is the affine
factorization; no Euclidean metric upgrade or ground-truth alignment is applied.
Both `.npz` outputs retain structure, motion, translations, input points, and
all four singular values. `media/part3/results.json` records the singular values
and reprojection RMSE per image coordinate. The script checks that the discarded
singular value equals the Frobenius norm of the reconstruction residual.

For 3(c), the script also plots the full-set reconstruction beside the supplied
3D ground truth in `media/part3/set1_reconstruction_comparison.png`. Display
alignment fits translation, a single uniform scale, and an orthogonal orientation
(reflection allowed for the arbitrary SVD axis signs). It does not fit a general
affine transform, modify the estimated factors, or use ground truth during
factorization. Both panels have equal axis scaling, identical limits, matching
point colors, and the same view. `set1_comparison.npz` stores ground truth,
display-aligned points, and point errors; alignment parameters and RMSE are in
`results.json`. Matplotlib uses the Agg backend; the ignored `.deps/` directory
is a fallback for checkout-local packages and `.preview/` holds its font cache.

To preview from the repository root:

```sh
python -m http.server 8000
```

Open `http://localhost:8000/4/`. The static page also opens directly. `.deps/` and `.preview/` are ignored local
working directories, matching the previous assignments.

