# Assignment 3

- `src/`: reusable algorithms (`homography.py` for A.2).
- `scripts/`: section-specific inputs, parameters, and result generation.
- `data/`: original photographs and input data.
- `media/part_a/a2/`: numbered correspondences, matrix, and reprojection errors.
- `index.html`, `style.css`, `page.js`: static assignment webpage.

From inside `3/`:

```sh
pip install -r requirements.txt
python scripts/part_a2.py
```

The script also runs from the portfolio root with `python 3/scripts/part_a2.py`.
Manual landmarks are fixed in the script at the original image resolution.
`computeH(im1_pts, im2_pts)` maps the first point set to the second; image
coordinates are `(x, y)` from the top-left pixel center. Generated results
include `results.json` and `results.js`, which supplies the webpage matrix
and RMSE without a Python runtime on GitHub Pages.
