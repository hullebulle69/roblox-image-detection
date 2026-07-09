"""Command-line interface for the Roblox character detector.

Examples:
    python -m roblox_detector.cli image.png
    python -m roblox_detector.cli shots/ --out annotated/ --json results.json
    python -m roblox_detector.cli image.png --threshold 0.5 --onnx model.onnx
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import cv2
import numpy as np

from .detector import DetectorConfig, RobloxCharacterDetector, annotate

SUPPORTED = (".png", ".jpg", ".jpeg", ".bmp", ".webp", ".gif")


def _iter_images(path: str):
    if os.path.isdir(path):
        for name in sorted(os.listdir(path)):
            if name.lower().endswith(SUPPORTED):
                yield os.path.join(path, name)
    else:
        yield path


def _imread(path: str):
    data = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Detect Roblox characters in images.")
    p.add_argument("input", help="image file or folder")
    p.add_argument("--out", help="folder to write annotated images into")
    p.add_argument("--json", dest="json_path", help="write detections to a JSON file")
    p.add_argument("--threshold", type=float, default=0.35,
                   help="confidence threshold (0-1, default 0.35)")
    p.add_argument("--onnx", help="optional YOLO-style ONNX person model")
    p.add_argument("--no-hog", action="store_true", help="disable HOG strategy")
    p.add_argument("--quiet", action="store_true", help="only print summary")
    args = p.parse_args(argv)

    config = DetectorConfig(
        confidence_threshold=args.threshold,
        use_hog=not args.no_hog,
        onnx_model_path=args.onnx,
    )
    try:
        detector = RobloxCharacterDetector(config)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.out:
        os.makedirs(args.out, exist_ok=True)

    all_results = {}
    total = 0
    images = list(_iter_images(args.input))
    if not images:
        print(f"error: no images found at {args.input}", file=sys.stderr)
        return 2

    for path in images:
        img = _imread(path)
        if img is None:
            print(f"skip (unreadable): {path}", file=sys.stderr)
            continue
        dets = detector.detect(img)
        total += len(dets)
        all_results[os.path.basename(path)] = [d.as_dict() for d in dets]
        if not args.quiet:
            print(f"{os.path.basename(path)}: {len(dets)} character(s)")
            for i, d in enumerate(dets, 1):
                print(f"  #{i}  score={d.score:.2f}  "
                      f"box=({d.x},{d.y},{d.w},{d.h})")
        if args.out:
            out_img = annotate(img, dets)
            dest = os.path.join(args.out, os.path.basename(path))
            ok, buf = cv2.imencode(os.path.splitext(dest)[1] or ".png", out_img)
            if ok:
                buf.tofile(dest)

    if args.json_path:
        with open(args.json_path, "w", encoding="utf-8") as fh:
            json.dump(all_results, fh, indent=2)

    print(f"\nDone: {total} detection(s) across {len(images)} image(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
