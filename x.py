# check_image_color_modes.py
import os
import argparse
from collections import Counter
from typing import Tuple, Dict, List
import numpy as np

# Pillow is lightweight and reliable for mode detection
from PIL import Image

def classify_image(path: str) -> Tuple[str, Dict]:
    """
    Returns (label, info) where label is one of:
      - 'GRAY'           : native 1-channel image (PIL mode 'L' or '1')
      - 'RGB_COLOR'      : 3-channel RGB with not-all-equal channels
      - 'RGB_MONO'       : 3-channel RGB but all channels equal (i.e., grayscale packed in RGB)
      - 'RGBA_COLOR'     : 4-channel with RGB not-all-equal (alpha ignored for color test)
      - 'RGBA_MONO'      : 4-channel with RGB equal (alpha ignored)
      - 'OTHER'          : anything else (CMYK, P, etc.)
      - 'UNREADABLE'     : failed to open/read
    info includes mode, size, channels, and a short reason.
    """
    info = {"mode": None, "size": None, "channels": None, "reason": ""}

    try:
        with Image.open(path) as im:
            info["mode"] = im.mode
            info["size"] = im.size

            # Convert to a consistent array without changing content
            arr = np.array(im)
            if arr.ndim == 2:
                info["channels"] = 1
                return "GRAY", info

            if arr.ndim == 3:
                h, w, c = arr.shape
                info["channels"] = c

                if c == 3:
                    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
                    if np.array_equal(r, g) and np.array_equal(g, b):
                        return "RGB_MONO", info
                    else:
                        return "RGB_COLOR", info

                elif c == 4:
                    r, g, b, a = arr[..., 0], arr[..., 1], arr[..., 2], arr[..., 3]
                    if np.array_equal(r, g) and np.array_equal(g, b):
                        return "RGBA_MONO", info
                    else:
                        return "RGBA_COLOR", info

                else:
                    info["reason"] = f"Unexpected channels={c}"
                    return "OTHER", info

            info["reason"] = f"Unexpected array ndim={arr.ndim}"
            return "OTHER", info

    except Exception as e:
        info["reason"] = str(e)
        return "UNREADABLE", info


def scan_folder(root: str,
                exts=(".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")) -> List[Tuple[str, str, Dict]]:
    results = []
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if fn.lower().endswith(exts):
                path = os.path.join(dirpath, fn)
                label, meta = classify_image(path)
                results.append((path, label, meta))
    return results


def main():
    parser = argparse.ArgumentParser(description="Check whether images are RGB or grayscale (and detect RGB-mono).")
    parser.add_argument("--root", required=True, help="Root directory of images (will search recursively).")
    parser.add_argument("--expect", choices=["rgb", "gray"], default=None,
                        help="Optional expectation to highlight mismatches.")
    parser.add_argument("--csv", default="color_mode_report.csv",
                        help="Output CSV path (default: color_mode_report.csv).")
    parser.add_argument("--show_samples", type=int, default=10,
                        help="Show up to N sample file paths per category in the console.")
    args = parser.parse_args()

    results = scan_folder(args.root)
    counts = Counter(lbl for _, lbl, _ in results)

    # Print summary
    total = len(results)
    print(f"\nScanned {total} images under: {args.root}\n")
    for k in ["RGB_COLOR", "RGB_MONO", "RGBA_COLOR", "RGBA_MONO", "GRAY", "OTHER", "UNREADABLE"]:
        if counts[k]:
            print(f"{k:12s}: {counts[k]}")

    # Mismatch highlighting (optional)
    if args.expect:
        if args.expect == "rgb":
            mismatches = [p for p, lbl, _ in results if lbl in ("GRAY", "RGB_MONO", "RGBA_MONO", "OTHER", "UNREADABLE")]
            print(f"\nExpected RGB: found {len(mismatches)} potential mismatches.")
            for p in mismatches[:args.show_samples]:
                print("  -", p)
        else:  # expect gray
            mismatches = [p for p, lbl, _ in results if lbl in ("RGB_COLOR", "RGBA_COLOR", "OTHER", "UNREADABLE")]
            print(f"\nExpected GRAY: found {len(mismatches)} potential mismatches.")
            for p in mismatches[:args.show_samples]:
                print("  -", p)

    # Show a few examples per class
    print("\nSample files per category:")
    by_label = {}
    for p, lbl, meta in results:
        by_label.setdefault(lbl, []).append((p, meta))
    for lbl, items in by_label.items():
        print(f"\n[{lbl}]")
        for p, meta in items[:args.show_samples]:
            mode = meta.get("mode")
            size = meta.get("size")
            ch   = meta.get("channels")
            reason = meta.get("reason", "")
            print(f"  - {p} | mode={mode} size={size} ch={ch} {('('+reason+')') if reason else ''}")

    # Write CSV
    try:
        import csv
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["path", "label", "mode", "width", "height", "channels", "reason"])
            for p, lbl, meta in results:
                size = meta.get("size") or (None, None)
                w.writerow([p, lbl, meta.get("mode"), size[0], size[1], meta.get("channels"), meta.get("reason", "")])
        print(f"\nCSV written to: {args.csv}")
    except Exception as e:
        print(f"\nCould not write CSV: {e}")


if __name__ == "__main__":
    main()
