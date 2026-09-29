import numpy as np
from PIL import Image
import cv2


def load_image(path):
    image = Image.open(path)
    if image.mode not in ("L", "I;16", "I"):
        image = image.convert("L")
    # convert to float
    img = np.array(image, dtype=np.float32)
    # scale to [0, 1] depending on if uint16 (max is 65535) or uint8 (max is 255)
    max_val = 65535.0 if img.max() > 255.0 else 255.0
    return img / max_val


def split_channels(img):
    # we split the image vertically evenly into three channels: blue, green, red
    h = img.shape[0] // 3
    b = img[0:h, :]
    g = img[h:2 * h, :]
    r = img[2 * h:3 * h, :]
    return b, g, r


def ssd(img1, img2):
    # Sum of squared differences: lower is better
    return np.sum((img1 - img2) ** 2)


def ncc(img1, img2):
    # Normalized cross correlation: higher is better
    img1_norm = np.linalg.norm(img1)
    img2_norm = np.linalg.norm(img2)
    if img1_norm == 0 or img2_norm == 0:
        return 0.0
    return np.sum(img1 * img2) / (img1_norm * img2_norm)


def align_single_scale(base, target, metric="ncc", window=15, margin=0.10):
    best_shift = (0, 0)
    best_score = -float("inf") if metric == "ncc" else float("inf")

    # evaluate metric strictly on interior pixels to ignore borders
    h, w = base.shape
    dh, dw = int(h * margin), int(w * margin)
    base_crop = base[dh:h - dh, dw:w - dw]

    # slide window across image to find best match
    for dy in range(-window, window + 1):
        for dx in range(-window, window + 1):
            shifted = np.roll(target, shift=(dy, dx), axis=(0, 1))
            target_crop = shifted[dh:h - dh, dw:w - dw]

            if metric == "ncc":
                score = ncc(base_crop, target_crop)
                if score > best_score:
                    best_score = score
                    best_shift = (dy, dx)
            else:  # ssd
                score = ssd(base_crop, target_crop)
                if score < best_score:
                    best_score = score
                    best_shift = (dy, dx)

    return best_shift


def align_pyramid(base, target, metric="ncc", min_size=200):
    # base case: if image is small enough, run single-scale search
    if base.shape[0] <= min_size or base.shape[1] <= min_size:
        return align_single_scale(base, target, metric=metric, window=15)

    # downsample by factor of 2
    h, w = base.shape[:2]
    base_small = cv2.resize(base, (w // 2, h // 2))
    target_small = cv2.resize(target, (w // 2, h // 2))

    # recursive call for shift on downsample, then scale back up
    dy, dx = align_pyramid(base_small, target_small, metric=metric, min_size=min_size)
    dy *= 2
    dx *= 2

    # pre-shift target with coarse estimate, then refine within [-2, 2] window
    shifted_target = np.roll(target, shift=(dy, dx), axis=(0, 1))
    refine_dy, refine_dx = align_single_scale(base, shifted_target, metric=metric, window=2)

    return dy + refine_dy, dx + refine_dx

# Bells and Whistle: Automatic White Balance & Contrast Stretching
def enhance_image(rgb, margin=0.10, low_p=1.0, high_p=99.0):
    enhanced = rgb.copy()
    h, w, _ = enhanced.shape
    dh, dw = int(h * margin), int(w * margin)

    # compute statistics on interior window
    interior = enhanced[dh:h - dh, dw:w - dw]

    # Gray World assumption: average scene color should be neutral gray
    mean_r = np.mean(interior[:, :, 0])
    mean_g = np.mean(interior[:, :, 1])
    mean_b = np.mean(interior[:, :, 2])
    mean_gray = (mean_r + mean_g + mean_b) / 3.0

    # scale each channel to match the target gray level
    if mean_r > 0 and mean_g > 0 and mean_b > 0:
        enhanced[:, :, 0] *= (mean_gray / mean_r)
        enhanced[:, :, 1] *= (mean_gray / mean_g)
        enhanced[:, :, 2] *= (mean_gray / mean_b)

    # re-crop interior after white balancing to compute contrast percentiles
    interior_enh = enhanced[dh:h - dh, dw:w - dw]

    # robust contrast stretch between 1st and 99th percentiles per channel
    for c in range(3):
        p_low = np.percentile(interior_enh[:, :, c], low_p)
        p_high = np.percentile(interior_enh[:, :, c], high_p)
        if p_high > p_low:
            enhanced[:, :, c] = (enhanced[:, :, c] - p_low) / (p_high - p_low)

    return np.clip(enhanced, 0.0, 1.0)


def colorize(image_path, metric="ncc", window=15, use_pyramid=False, auto_enhance=False):
    img = load_image(image_path)
    b, g, r = split_channels(img)

    # align green and red channels relative to blue channel
    if use_pyramid:
        g_shift = align_pyramid(b, g, metric=metric)
        r_shift = align_pyramid(b, r, metric=metric)
    else:
        g_shift = align_single_scale(b, g, metric=metric, window=window)
        r_shift = align_single_scale(b, r, metric=metric, window=window)

    aligned_g = np.roll(g, shift=(g_shift[0], g_shift[1]), axis=(0, 1))
    aligned_r = np.roll(r, shift=(r_shift[0], r_shift[1]), axis=(0, 1))

    # stack into RGB (R, G, B)
    rgb = np.clip(np.stack([aligned_r, aligned_g, b], axis=-1), 0.0, 1.0)

    if auto_enhance:
        rgb = enhance_image(rgb)

    return rgb, g_shift, r_shift


def save_image(rgb, path):
    out = (np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8)
    Image.fromarray(out).save(path)

