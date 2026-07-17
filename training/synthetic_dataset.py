#!/usr/bin/env python3
"""Generate a synthetic Roblox-avatar dataset in YOLO format.

Renders procedural blocky R6/R15-style avatars (head, torso, limbs in flat
saturated colors) over varied synthetic backgrounds, together with
non-humanoid distractor geometry, and writes YOLO txt labels.

This is a *bootstrap* dataset: a model trained on it learns the blocky
humanoid signature well enough to find avatars in easy footage, and gives
you a working end-to-end pipeline before you invest in labeling real
screenshots. For "super great" accuracy, fine-tune the bootstrapped model
on a few hundred labeled frames of the actual games you play (see
training/README.md).

Usage:
    python training/synthetic_dataset.py --out datasets/roblox_synth \
        --train 600 --val 150 --imgsz 640 --seed 0
"""

from __future__ import annotations

import argparse
import colorsys
import math
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

CLASS_NAME = "roblox_character"

# Classic Roblox body palette (head/skin tones) + common clothing hues.
SKIN_TONES = [
    (245, 205, 48),   # classic yellow
    (234, 184, 146),  # light skin
    (204, 142, 105),
    (150, 85, 63),
    (91, 93, 105),    # grey robot
    (247, 234, 194),
]


def _sat_color(rng: random.Random, s_lo=0.45, v_lo=0.45):
    h = rng.random()
    s = rng.uniform(s_lo, 1.0)
    v = rng.uniform(v_lo, 1.0)
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    return int(r * 255), int(g * 255), int(b * 255)


def _shade(color, f):
    return tuple(max(0, min(255, int(c * f))) for c in color)


def _draw_block(draw: ImageDraw.ImageDraw, box, color, rng: random.Random):
    """A flat-shaded rounded block with a subtle darker outline, like a
    Roblox part rendered at typical game lighting."""
    x1, y1, x2, y2 = box
    if x2 <= x1 or y2 <= y1:
        return
    face = _shade(color, rng.uniform(0.92, 1.05))
    if x2 - x1 < 3 or y2 - y1 < 3:
        # Too small for rounded corners / highlight strip (they would
        # produce inverted rectangles at tiny sizes).
        draw.rectangle(box, fill=face, outline=_shade(color, 0.62))
        return
    radius = max(1, int(min(x2 - x1, y2 - y1) * rng.uniform(0.04, 0.16)))
    draw.rounded_rectangle(box, radius=radius, fill=face,
                           outline=_shade(color, 0.62), width=1)
    # Cheap top-lit gradient: a lighter strip along the upper edge.
    strip_h = max(1, int((y2 - y1) * 0.18))
    draw.rounded_rectangle((x1 + 1, y1 + 1, x2 - 1, y1 + strip_h),
                           radius=radius, fill=_shade(face, 1.10))


def draw_avatar(draw: ImageDraw.ImageDraw, cx: int, ground_y: int,
                height: int, rng: random.Random):
    """Draw one blocky humanoid; returns its tight bounding box.

    Proportions follow the R6 rig: head ~ 1/5.4 of height, torso twice as
    wide as deep, arms flush with the torso sides, legs half the torso
    width each.
    """
    h = height
    head_h = h * rng.uniform(0.16, 0.21)
    torso_h = h * rng.uniform(0.30, 0.36)
    leg_h = h - head_h - torso_h
    torso_w = h * rng.uniform(0.26, 0.34)
    arm_w = torso_w * rng.uniform(0.42, 0.55)
    head_w = torso_w * rng.uniform(0.55, 0.75)

    skin = rng.choice(SKIN_TONES)
    shirt = _sat_color(rng)
    pants = _sat_color(rng)
    arm_color = shirt if rng.random() < 0.6 else skin

    top = ground_y - h
    # Slight limb pose variation so the silhouette isn't perfectly rigid.
    arm_drop = rng.uniform(-0.06, 0.10) * torso_h
    leg_gap = max(1, int(torso_w * rng.uniform(0.02, 0.10)))

    head_box = (cx - head_w / 2, top, cx + head_w / 2, top + head_h)
    torso_box = (cx - torso_w / 2, top + head_h,
                 cx + torso_w / 2, top + head_h + torso_h)
    left_arm = (cx - torso_w / 2 - arm_w, top + head_h + arm_drop,
                cx - torso_w / 2, top + head_h + torso_h * rng.uniform(0.85, 1.0))
    right_arm = (cx + torso_w / 2, top + head_h + arm_drop,
                 cx + torso_w / 2 + arm_w, top + head_h + torso_h * rng.uniform(0.85, 1.0))
    leg_w = (torso_w - leg_gap) / 2
    legs_top = top + head_h + torso_h
    left_leg = (cx - torso_w / 2, legs_top, cx - torso_w / 2 + leg_w, ground_y)
    right_leg = (cx + torso_w / 2 - leg_w, legs_top, cx + torso_w / 2, ground_y)

    for box, color in ((left_leg, pants), (right_leg, pants),
                       (left_arm, arm_color), (right_arm, arm_color),
                       (torso_box, shirt), (head_box, skin)):
        _draw_block(draw, box, color, rng)

    # Face: two dark eyes and a mouth on the head block.
    if rng.random() < 0.85:
        ex = head_w * 0.18
        ey = top + head_h * 0.35
        er = max(1, int(head_h * 0.07))
        for sx in (-1, 1):
            draw.ellipse((cx + sx * ex - er, ey - er, cx + sx * ex + er, ey + er),
                         fill=(25, 25, 25))
        draw.arc((cx - ex, top + head_h * 0.45, cx + ex, top + head_h * 0.8),
                 20, 160, fill=(25, 25, 25), width=max(1, er // 2))

    # Occasionally a simple hat/hair brim, which also hides the head top —
    # teaches the model not to depend on a visible bare head.
    top_visual = top
    if rng.random() < 0.4:
        brim = head_w * rng.uniform(0.55, 0.8)
        hat = _sat_color(rng, 0.2, 0.2)
        hat_top = top - head_h * 0.22
        draw.rectangle((cx - brim, hat_top, cx + brim, top + head_h * 0.12),
                       fill=hat, outline=_shade(hat, 0.6))
        top_visual = hat_top

    x1 = min(left_arm[0], torso_box[0])
    x2 = max(right_arm[2], torso_box[2])
    return x1, top_visual, x2, ground_y


def draw_background(img: Image.Image, rng: random.Random):
    """Blocky game-world backdrop: sky gradient, ground plane, and random
    non-humanoid parts (walls, crates, spheres, trees)."""
    w, h = img.size
    draw = ImageDraw.Draw(img)
    sky_a, sky_b = _sat_color(rng, 0.1, 0.5), _sat_color(rng, 0.1, 0.3)
    for y in range(h):
        t = y / h
        draw.line([(0, y), (w, y)],
                  fill=tuple(int(a + (b - a) * t) for a, b in zip(sky_a, sky_b)))
    horizon = int(h * rng.uniform(0.55, 0.8))
    ground = _sat_color(rng, 0.25, 0.25)
    draw.rectangle((0, horizon, w, h), fill=ground)

    # Distractor geometry: crates, pillars, spheres — flat-colored but NOT
    # humanoid assemblies, so the model must learn structure, not just
    # "saturated flat region".
    for _ in range(rng.randint(3, 10)):
        c = _sat_color(rng)
        x = rng.randint(-40, w - 1)
        y = rng.randint(int(h * 0.2), h - 10)
        s = rng.randint(12, int(h * 0.35))
        kind = rng.random()
        if kind < 0.5:
            _draw_block(draw, (x, y, x + s, y + int(s * rng.uniform(0.4, 2.5))), c, rng)
        elif kind < 0.75:
            draw.ellipse((x, y, x + s, y + s), fill=c, outline=_shade(c, 0.6))
        else:  # "tree": trunk + blob
            tw = max(3, s // 5)
            draw.rectangle((x, y, x + tw, y + s), fill=(110, 80, 50))
            r = s // 2 + 4
            draw.ellipse((x + tw // 2 - r, y - r, x + tw // 2 + r, y + r),
                         fill=(40, rng.randint(110, 180), 60))
    return horizon


def render_scene(imgsz: int, rng: random.Random):
    img = Image.new("RGB", (imgsz, imgsz))
    horizon = draw_background(img, rng)
    draw = ImageDraw.Draw(img)

    boxes = []
    n_avatars = rng.choices([0, 1, 2, 3, 4], weights=[8, 35, 30, 17, 10])[0]
    attempts = 0
    while len(boxes) < n_avatars and attempts < 40:
        attempts += 1
        height = int(imgsz * rng.uniform(0.10, 0.55))
        # Feet somewhere on/below the horizon, plus some floaters (jumping).
        ground_y = rng.randint(max(horizon, height + 2),
                               imgsz - 1) if rng.random() < 0.85 else \
            rng.randint(height + 2, imgsz - 1)
        cx = rng.randint(int(height * 0.25), imgsz - int(height * 0.25))
        candidate = (cx - height * 0.25, ground_y - height,
                     cx + height * 0.25, ground_y)
        if any(_overlap(candidate, b) > 0.35 for b in boxes):
            continue
        box = draw_avatar(draw, cx, ground_y, height, rng)
        boxes.append(tuple(map(float, box)))

    # Global nuisance: mild blur / noise / brightness, like compressed footage.
    if rng.random() < 0.3:
        img = img.filter(ImageFilter.GaussianBlur(rng.uniform(0.4, 1.2)))
    arr = np.asarray(img).astype(np.int16)
    if rng.random() < 0.5:
        arr += rng.randint(-18, 18)
    if rng.random() < 0.5:
        arr += np.random.default_rng(rng.getrandbits(32)).integers(
            -10, 10, arr.shape, dtype=np.int16)
    img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    return img, boxes


def _overlap(a, b):
    """Fraction of the *smaller-covered* box that is overlapped — symmetric,
    so a big new avatar can't fully hide a small existing one (which would
    leave a ghost label)."""
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return max(inter / area_a if area_a > 0 else 0.0,
               inter / area_b if area_b > 0 else 0.0)


def write_split(root: Path, split: str, count: int, imgsz: int,
                rng: random.Random):
    img_dir = root / "images" / split
    lbl_dir = root / "labels" / split
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)
    for i in range(count):
        img, boxes = render_scene(imgsz, rng)
        img.save(img_dir / f"{split}_{i:05d}.jpg", quality=rng.randint(70, 95))
        lines = []
        for x1, y1, x2, y2 in boxes:
            x1, y1 = max(0.0, x1), max(0.0, y1)
            x2, y2 = min(float(imgsz), x2), min(float(imgsz), y2)
            if x2 - x1 < 4 or y2 - y1 < 4:
                continue
            cx = (x1 + x2) / 2 / imgsz
            cy = (y1 + y2) / 2 / imgsz
            bw = (x2 - x1) / imgsz
            bh = (y2 - y1) / imgsz
            lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
        (lbl_dir / f"{split}_{i:05d}.txt").write_text("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="datasets/roblox_synth")
    ap.add_argument("--train", type=int, default=600)
    ap.add_argument("--val", type=int, default=150)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    root = Path(args.out)
    write_split(root, "train", args.train, args.imgsz, rng)
    write_split(root, "val", args.val, args.imgsz, rng)

    (root / "data.yaml").write_text(
        f"path: {root.resolve()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "names:\n"
        f"  0: {CLASS_NAME}\n")
    print(f"Wrote {args.train} train / {args.val} val images to {root}")
    print(f"Dataset config: {root / 'data.yaml'}")


if __name__ == "__main__":
    main()
