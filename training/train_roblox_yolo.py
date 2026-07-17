#!/usr/bin/env python3
"""Train a YOLO model to detect Roblox characters and export it to ONNX for
the C++ tools (cpp/rcd_live, cpp/rcd_cli).

Quick start (synthetic bootstrap, no labeling needed):
    python training/synthetic_dataset.py --out datasets/roblox_synth
    python training/train_roblox_yolo.py --data datasets/roblox_synth/data.yaml

Real accuracy (fine-tune the bootstrap on labeled screenshots):
    python training/train_roblox_yolo.py --data my_screenshots/data.yaml \
        --weights runs/detect/roblox/weights/best.pt --epochs 80

The exported .onnx embeds the class names, which the C++ detector reads
automatically.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", required=True, help="YOLO data.yaml")
    ap.add_argument("--weights", default="yolov8s.pt",
                    help="starting weights (.pt) or a model yaml like "
                         "'yolov8n.yaml' to train from scratch")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=-1,
                    help="-1 = auto batch size")
    ap.add_argument("--device", default=None,
                    help="e.g. 0 for first GPU, 'cpu' to force CPU")
    ap.add_argument("--name", default="roblox")
    ap.add_argument("--no-export", action="store_true")
    args = ap.parse_args()

    from ultralytics import YOLO  # deferred: slow import

    model = YOLO(args.weights)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        name=args.name,
        exist_ok=True,
        # Screenshots vary wildly in lighting/graphics settings; lean on
        # color + scale augmentation rather than flips-only defaults.
        degrees=5.0,
        scale=0.6,
        mosaic=1.0,
        hsv_h=0.02,
        hsv_s=0.6,
        hsv_v=0.5,
    )

    metrics = model.val()
    print(f"\nvalidation: mAP50={metrics.box.map50:.3f} "
          f"mAP50-95={metrics.box.map:.3f} "
          f"precision={metrics.box.mp:.3f} recall={metrics.box.mr:.3f}")

    if not args.no_export:
        onnx_path = model.export(format="onnx", imgsz=args.imgsz, opset=12,
                                 dynamic=False, simplify=True)
        print(f"\nONNX model for the C++ tools: {onnx_path}")
        print("Run it live:   rcd_live --model", Path(onnx_path).name)
        print("Test on stills: rcd_cli screenshot.png --model",
              Path(onnx_path).name)


if __name__ == "__main__":
    main()
