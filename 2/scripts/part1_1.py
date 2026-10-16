"""Generate Part 1.1 filters and SciPy comparisons using data/custom/selfie.jpg."""
import inspect
import json
from pathlib import Path
from time import perf_counter
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
from PIL import Image, ImageOps
from scipy.signal import convolve2d
import scipy
from convolution import _prepare, zero_pad, convolve_four_loops, convolve_two_loops

REPEATS = 3

def save_gray(array, path, signed=False):
    # Show zero derivatives as gray on a fixed [-1, 1] scale.
    display = 0.5 + array / 2 if signed else array
    Image.fromarray(np.round(np.clip(display, 0, 1) * 255).astype(np.uint8)).save(path)


def main():
    with Image.open(ROOT / "data" / "custom" / "selfie.jpg") as source:
        source = ImageOps.exif_transpose(source).convert("L")
        source.thumbnail((320, 320))
        image = np.asarray(source, dtype=np.float64) / 255
    out = ROOT / "media" / "part1" / "1_1"
    out.mkdir(parents=True, exist_ok=True)
    save_gray(image, out / "original.png")
    
    # Compare both implementations and SciPy for each required filter.
    records = []
    box = np.ones((9, 9)) / 81
    DX = np.array([[1.0, 0.0, -1.0]])
    DY = DX.T.copy()
    for name, kernel in [("box", box), ("dx", DX), ("dy", DY)]:
        reference = convolve2d(image, kernel, mode="same", boundary="fill", fillvalue=0)
        record = {"filter": name, "kernel": list(kernel.shape), "measurements": []}
        implementations = [("Four loops", convolve_four_loops), ("Two loops", convolve_two_loops),
                           ("SciPy", lambda a, k: convolve2d(a, k, mode="same", boundary="fill", fillvalue=0))]
        for label, function in implementations:
            function(image, kernel)  # untimed warm-up
            # Use the median of three runs to reduce timing variation.
            times = []
            for _ in range(REPEATS):
                start = perf_counter()
                result = function(image, kernel)
                times.append((perf_counter() - start) * 1000)
            # Compare raw values before display scaling.
            error = float(np.max(np.abs(result - reference)))
            record["measurements"].append({"implementation": label,
                                           "milliseconds": float(np.median(times)), "max_error": error})
            if label == "Two loops":
                save_gray(result, out / f"{name}_two_loops.png", signed=name != "box")
        records.append(record)
    data = {"shape": list(image.shape), "repeats": REPEATS,
            "numpy": np.__version__, "scipy": scipy.__version__, "records": records}
    # Export measurements and actual source snippets for the webpage.
    snippets = {f.__name__: inspect.getsource(f) for f in
                [zero_pad, _prepare, convolve_four_loops, convolve_two_loops]}
    payload = {"results": data, "snippets": snippets}
    (out / "results.js").write_text("window.convolutionData = " + json.dumps(payload) + ";\n", encoding="utf-8")
    print(f"Saved Part 1.1 images, timings, and code snippets to {out}")


if __name__ == "__main__":
    main()
