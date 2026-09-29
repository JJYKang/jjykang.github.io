import sys
import os
import glob
from colorizer import colorize, save_image


def run_one(image_path, method="pyramid", metric="ncc", auto_enhance=False, out_dir=None):
    use_pyramid = (method == "pyramid")
    rgb, g_shift, r_shift = colorize(image_path, metric=metric, use_pyramid=use_pyramid, auto_enhance=auto_enhance)

    status = " [enhanced]" if auto_enhance else ""
    # Displacements (x, y) relative to blue channel
    print(f"{os.path.basename(image_path)}{status}: Green (x, y)={(g_shift[1], g_shift[0])}, Red (x, y)={(r_shift[1], r_shift[0])}")

    if out_dir is None:
        sub = f"{method}_enhanced" if auto_enhance else method
        out_dir = os.path.join("output", sub)
    os.makedirs(out_dir, exist_ok=True)
    out_name = os.path.splitext(os.path.basename(image_path))[0] + "_aligned.jpg"
    save_image(rgb, os.path.join(out_dir, out_name))


def main():
    if len(sys.argv) < 2:
        print("Usage: python main.py <image_path | --low-res | --all> [single|pyramid] [ncc|ssd] [--enhance]")
        sys.exit(1)

    auto_enhance = "--enhance" in sys.argv
    args = [a for a in sys.argv[1:] if a != "--enhance"]

    arg = args[0]
    method = args[1].lower() if len(args) > 1 else "pyramid"
    metric = args[2].lower() if len(args) > 2 else "ncc"

    if arg == "--low-res":
        files = sorted(glob.glob("data/*v.jpg"))
    elif arg == "--all":
        files = sorted(glob.glob("data/*.jpg") + glob.glob("data/*.tif"))
    else:
        files = [arg]

    for path in files:
        run_one(path, method=method, metric=metric, auto_enhance=auto_enhance)


if __name__ == "__main__":
    main()