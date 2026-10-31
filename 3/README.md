# Assignment 3

- `src/`: reusable algorithms (`homography.py` for A.2, `warping.py` for A.3).
- `scripts/`: section-specific inputs, parameters, and result generation.
- `data/`: original photographs and input data.
- `media/part_a/a2/`: numbered correspondences, matrix, and reprojection errors.
- `media/part_a/a3/`: rectification inputs, interpolation comparisons, and timings.
- `index.html`, `style.css`, `page.js`: static assignment webpage.

From inside `3/`:

```sh
pip install -r requirements.txt
python scripts/part_a2.py
python scripts/part_a3.py
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
