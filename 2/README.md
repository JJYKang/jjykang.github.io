# Assignment 2

The webpage stays in `index.html`, with `style.css` and `page.js` beside it, as in A1.

```text
2/
  index.html
  style.css
  page.js
  requirements.txt
  src/
    convolution.py           # reusable NumPy convolution functions
    filters.py               # finite differences, Gaussian/DoG, edges, orientations
  scripts/
    part1_1.py               # generate section 1.1 results
    part1_2.py               # generate section 1.2 results
    part1_3.py               # generate section 1.3 and orientation results
  data/
    cameraman.png            # downloaded from the Berkeley assignment
    provided/
      syde671 assignment 2 hybrid/    # supplied folder, unchanged
      syde671 assn2 part 2 images/    # supplied folder, unchanged
    custom/                  # our own original photos and masks
  media/
    part1/
      1_1/                   # generated images, benchmarks, source snippets
      1_2/                   # cameraman derivatives, magnitude, binary edges
      1_3/                   # Gaussian/DoG results and HSV orientations
```

## Adding the remaining sections

Keep reusable algorithms in `src/`, and image choices, parameters, timing, and export logic in section scripts. Add files when implementing the relevant section:

| Sections | Reusable code | Result script | Webpage assets |
| --- | --- | --- | --- |
| 1.1 | `src/convolution.py` | `scripts/part1_1.py` | `media/part1/1_1/` |
| 1.2 | `src/filters.py` | `scripts/part1_2.py` | `media/part1/1_2/` |
| 1.3 and orientations | `src/filters.py` | `scripts/part1_3.py` | `media/part1/1_3/` |
| 2.1 | `src/filters.py` | `scripts/part2_1.py` | `media/part2/2_1/` |
| 2.2 | `src/hybrid.py` | `scripts/part2_2.py` | `media/part2/2_2/` |
| 2.3 | `src/stacks.py` | `scripts/part2_3.py` | `media/part2/2_3/` |
| 2.4 | `src/blending.py`, reusing stacks | `scripts/part2_4.py` | `media/part2/2_4/` |

Add shared loading and export helpers in `src/image_io.py` when another section needs them. Preserve supplied input filenames and internal folder structure wherever the user places them. Save aligned, filtered, Fourier, stack, and blended results in the relevant `media/` section. For sections with multiple examples, use subfolders such as `media/part2/2_2/derek_nutmeg/`.

Commit the generated assets used by the webpage so GitHub Pages needs no Python runtime. Keep benchmark data and generated code snippets beside their section's images. Section scripts resolve output paths from their own location, so they work from either the portfolio root or the A2 directory.

## Run sections 1.1, 1.2, and 1.3

From inside `2/`:

```sh
pip install -r requirements.txt
python scripts/part1_1.py
python scripts/part1_2.py
python scripts/part1_3.py
```

From the portfolio root, prefix script paths with `2/`. Section 1.1 reads `data/custom/selfie.jpg`; sections 1.2 and 1.3 read `data/cameraman.png`. The scripts use fixed assignment parameters, defined near the top of each file.

`.deps/` and `.preview/` are ignored local working directories.

Cameraman source: https://cal-cs180.github.io/fa24/hw/proj2/cameraman.png
