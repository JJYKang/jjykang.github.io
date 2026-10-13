"""Generate Part 1.1 assets, measurements, and source snippets.

python 2/scripts/part1_1.py                         # synthetic placeholder
python 2/scripts/part1_1.py --image path/to/me.jpg   # grayscale selfie
"""
import argparse
import inspect
import json
from pathlib import Path
from time import perf_counter
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
from PIL import Image, ImageDraw
from scipy.signal import convolve2d
import scipy
from convolution import _prepare, zero_pad, convolve_four_loops, convolve_two_loops, box_filter, DX, DY

def placeholder():
    """A geometric test target, deliberately not a photograph of the author."""
    image = Image.new("L", (192, 192), 200)
    draw = ImageDraw.Draw(image)
    for x in range(0, 192, 16):
        draw.rectangle((x, 0, x + 7, 47), fill=150)
    draw.rectangle((24, 65, 92, 151), fill=40)
    draw.ellipse((107, 63, 170, 126), fill=245)
    draw.polygon([(105, 158), (137, 130), (176, 165)], fill=85)
    draw.line((18, 178, 174, 178), fill=0, width=3)
    return np.asarray(image, dtype=np.float64) / 255


def save_gray(array, path, signed=False):
    # Both derivative images use [-1, 1]: zero is mid-gray, no per-image autoscale.
    display = 0.5 + array / 2 if signed else array
    Image.fromarray(np.round(np.clip(display, 0, 1) * 255).astype(np.uint8)).save(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    if args.image:
        with Image.open(args.image) as source:
            source = source.convert("L")
            source.thumbnail((320, 320))
            image = np.asarray(source, dtype=np.float64) / 255
    else:
        image = placeholder()
    out = ROOT / "media" / "part1" / "1_1"
    out.mkdir(parents=True, exist_ok=True)
    save_gray(image, out / "original.png")
    records = []
    for name, kernel in [("box", box_filter(9)), ("dx", DX), ("dy", DY)]:
        reference = convolve2d(image, kernel, mode="same", boundary="fill", fillvalue=0)
        record = {"filter": name, "kernel": list(kernel.shape), "measurements": []}
        implementations = [("Four loops", convolve_four_loops), ("Two loops", convolve_two_loops),
                           ("SciPy", lambda a, k: convolve2d(a, k, mode="same", boundary="fill", fillvalue=0))]
        for label, function in implementations:
            function(image, kernel)  # untimed warm-up
            times = []
            for _ in range(args.repeats):
                start = perf_counter()
                result = function(image, kernel)
                times.append((perf_counter() - start) * 1000)
            error = float(np.max(np.abs(result - reference)))
            np.testing.assert_allclose(result, reference, atol=1e-12, rtol=1e-12)
            record["measurements"].append({"implementation": label,
                                           "milliseconds": float(np.median(times)), "max_error": error})
            save_gray(result, out / f"{name}_{label.lower().replace(' ', '_')}.png", signed=name != "box")
        records.append(record)
    data = {"placeholder": args.image is None, "shape": list(image.shape), "repeats": args.repeats,
            "numpy": np.__version__, "scipy": scipy.__version__, "records": records}
    (out / "results.json").write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    # Embedded script works on GitHub Pages and when index.html is opened locally.
    snippets = {f.__name__: inspect.getsource(f) for f in
                [zero_pad, _prepare, convolve_four_loops, convolve_two_loops, box_filter]}
    payload = {"results": data, "snippets": snippets}
    (out / "results.js").write_text("window.convolutionData = " + json.dumps(payload) + ";\n", encoding="utf-8")
    print(json.dumps(data, indent=2))


if __name__ == "__main__":
    main()
