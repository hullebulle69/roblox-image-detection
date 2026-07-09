"""Roblox character detection engine.

Multi-strategy classical CV pipeline tuned for the visual signature of
Roblox avatars (R6/R15/blocky rigs):

  1. Flat-color region segmentation  - avatars are made of large regions of
     near-uniform, saturated color with very little texture.
  2. Head candidate detection        - classic yellow heads plus common skin
     tones, compact round/square shapes.
  3. Part assembly                   - a head with a torso-shaped flat region
     below it (and optional limb regions beside it) is assembled into a
     character hypothesis.
  4. HOG humanoid voting             - OpenCV's HOG person detector adds an
     independent vote for humanoid silhouettes.
  5. Score fusion + soft NMS         - all evidence is fused into a single
     confidence per box, overlapping hypotheses are merged.

An optional ONNX backend (any YOLO-style person/character model exported to
ONNX) can be plugged in for neural detection; classical and neural results
are fused with the same NMS stage.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class Detection:
    """One detected character. Coordinates are pixels in the original image."""

    x: int
    y: int
    w: int
    h: int
    score: float
    label: str = "roblox_character"
    evidence: dict = field(default_factory=dict)

    @property
    def bbox(self) -> tuple[int, int, int, int]:
        return (self.x, self.y, self.w, self.h)

    def as_dict(self) -> dict:
        return {
            "x": self.x, "y": self.y, "w": self.w, "h": self.h,
            "score": round(float(self.score), 4),
            "label": self.label,
            "evidence": {k: round(float(v), 4) for k, v in self.evidence.items()},
        }


@dataclass
class DetectorConfig:
    confidence_threshold: float = 0.35
    min_char_height_frac: float = 0.04   # min character height vs image height
    max_char_height_frac: float = 0.98
    nms_iou: float = 0.35
    max_working_size: int = 1280         # long side used for analysis
    use_hog: bool = True
    use_assembly: bool = True
    use_flat_regions: bool = True
    onnx_model_path: str | None = None   # optional YOLO-style ONNX model
    onnx_input_size: int = 640
    onnx_score_threshold: float = 0.30


# ---------------------------------------------------------------------------
# Small geometry helpers
# ---------------------------------------------------------------------------


def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def _make_hog():
    """Build the default people HOG detector, or None if this OpenCV build
    lacks it (e.g. some slimmed-down headless wheels)."""
    if not hasattr(cv2, "HOGDescriptor"):
        return None
    try:
        hog = cv2.HOGDescriptor()
        hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        return hog
    except Exception:
        return None


def _merge_boxes(a, b):
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    x1, y1 = min(ax1, bx1), min(ay1, by1)
    x2, y2 = max(ax1 + aw, bx1 + bw), max(ay1 + ah, by1 + bh)
    return (x1, y1, x2 - x1, y2 - y1)


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------


class RobloxCharacterDetector:
    # HSV ranges for classic Roblox head colors: "Bright yellow" plus the
    # common skin-tone palette (Light orange, Nougat, Brick yellow, ...).
    HEAD_HSV_RANGES = [
        ((20, 90, 120), (36, 255, 255)),    # classic bright yellow
        ((8, 40, 110), (26, 200, 255)),     # tan / nougat / light orange
        ((0, 25, 140), (14, 160, 255)),     # pale skin tones
    ]

    def __init__(self, config: DetectorConfig | None = None):
        self.config = config or DetectorConfig()
        self._hog = None
        self._onnx_net = None
        if self.config.use_hog:
            self._hog = _make_hog()
        if self.config.onnx_model_path:
            self.load_onnx_model(self.config.onnx_model_path)

    # -- public API ---------------------------------------------------------

    def load_onnx_model(self, path: str) -> None:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"ONNX model not found: {path}")
        self._onnx_net = cv2.dnn.readNetFromONNX(path)
        self.config.onnx_model_path = path

    def detect(self, image_bgr: np.ndarray) -> list[Detection]:
        """Detect Roblox characters in a BGR image."""
        if image_bgr is None or image_bgr.size == 0:
            return []
        if image_bgr.ndim == 2:
            image_bgr = cv2.cvtColor(image_bgr, cv2.COLOR_GRAY2BGR)
        elif image_bgr.shape[2] == 4:
            image_bgr = cv2.cvtColor(image_bgr, cv2.COLOR_BGRA2BGR)

        orig_h, orig_w = image_bgr.shape[:2]
        scale = 1.0
        work = image_bgr
        long_side = max(orig_h, orig_w)
        if long_side > self.config.max_working_size:
            scale = self.config.max_working_size / long_side
            work = cv2.resize(image_bgr, (int(orig_w * scale), int(orig_h * scale)),
                              interpolation=cv2.INTER_AREA)

        hsv = cv2.cvtColor(work, cv2.COLOR_BGR2HSV)
        smooth = cv2.bilateralFilter(work, 7, 60, 60)
        flatness = self._flatness_map(smooth)

        candidates: list[Detection] = []
        if self.config.use_assembly:
            heads = self._find_head_candidates(work, hsv, flatness)
            candidates += self._assemble_characters(work, hsv, flatness, heads)
        if self.config.use_flat_regions:
            candidates += self._flat_region_candidates(work, smooth, flatness)
        if self._hog is not None:
            candidates += self._hog_candidates(work, flatness)
        if self._onnx_net is not None:
            candidates += self._onnx_candidates(work)

        candidates = self._filter_by_size(candidates, work.shape)
        fused = self._fuse(candidates)
        fused = [d for d in fused if d.score >= self.config.confidence_threshold]

        # Map back to original resolution.
        if scale != 1.0:
            inv = 1.0 / scale
            for d in fused:
                d.x = int(round(d.x * inv))
                d.y = int(round(d.y * inv))
                d.w = int(round(d.w * inv))
                d.h = int(round(d.h * inv))
        for d in fused:
            d.x = max(0, min(d.x, orig_w - 1))
            d.y = max(0, min(d.y, orig_h - 1))
            d.w = max(1, min(d.w, orig_w - d.x))
            d.h = max(1, min(d.h, orig_h - d.y))
        fused.sort(key=lambda d: d.score, reverse=True)
        return fused

    def detect_file(self, path: str) -> list[Detection]:
        data = np.fromfile(path, dtype=np.uint8)  # handles unicode paths
        image = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Could not decode image: {path}")
        return self.detect(image)

    # -- strategy 1: flatness map --------------------------------------------

    @staticmethod
    def _flatness_map(smooth_bgr: np.ndarray) -> np.ndarray:
        """Per-pixel measure (0..1) of how 'flat colored' the neighborhood is.

        Roblox avatars render as large regions of near-constant color, so low
        local variance is strong evidence. Computed on the smoothed image.
        """
        gray = cv2.cvtColor(smooth_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
        mean = cv2.boxFilter(gray, -1, (9, 9))
        sq_mean = cv2.boxFilter(gray * gray, -1, (9, 9))
        var = np.clip(sq_mean - mean * mean, 0, None)
        std = np.sqrt(var)
        return np.clip(1.0 - std / 24.0, 0.0, 1.0)

    # -- strategy 2: head candidates ------------------------------------------

    def _find_head_candidates(self, work, hsv, flatness) -> list[dict]:
        h, w = work.shape[:2]
        mask = np.zeros((h, w), np.uint8)
        for lo, hi in self.HEAD_HSV_RANGES:
            mask |= cv2.inRange(hsv, np.array(lo), np.array(hi))
        # Heads are flat-colored; drop textured matches (foliage, sand ...).
        mask[flatness < 0.45] = 0
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

        heads = []
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        min_area = max(16, (h * self.config.min_char_height_frac * 0.22) ** 2)
        for c in contours:
            area = cv2.contourArea(c)
            if area < min_area:
                continue
            x, y, bw, bh = cv2.boundingRect(c)
            if bw == 0 or bh == 0:
                continue
            aspect = bw / bh
            if not 0.45 <= aspect <= 2.4:      # heads are roughly square-ish
                continue
            extent = area / (bw * bh)
            if extent < 0.5:                    # heads fill their box well
                continue
            # Shape score: how close to a filled square/circle.
            aspect_score = math.exp(-((aspect - 1.15) ** 2) / 0.5)
            heads.append({
                "box": (x, y, bw, bh),
                "score": 0.5 * extent + 0.5 * aspect_score,
                "area": area,
            })
        heads.sort(key=lambda d: d["score"], reverse=True)
        return heads[:40]

    # -- strategy 3: assemble head + torso + limbs ----------------------------

    def _assemble_characters(self, work, hsv, flatness, heads) -> list[Detection]:
        """For every head candidate, verify humanoid structure below it."""
        detections = []
        sat = hsv[:, :, 1].astype(np.float32) / 255.0
        h_img, w_img = work.shape[:2]
        for head in heads:
            hx, hy, hw, hh = head["box"]
            # Roblox proportions: total height ~ 4.5-5.5 head heights,
            # torso directly below head, ~1.3-2x head width.
            body_top = hy + hh
            body_h = int(hh * 4.2)
            body_bottom = min(h_img, body_top + body_h)
            bx1 = max(0, hx - int(hw * 1.1))
            bx2 = min(w_img, hx + hw + int(hw * 1.1))
            if body_bottom - body_top < hh or bx2 - bx1 < hw:
                continue

            torso_y2 = min(h_img, body_top + int(hh * 1.8))
            torso = flatness[body_top:torso_y2, bx1:bx2]
            torso_sat = sat[body_top:torso_y2, bx1:bx2]
            legs = flatness[torso_y2:body_bottom, bx1:bx2]
            if torso.size == 0:
                continue

            torso_flat = float(torso.mean())
            torso_colorful = float((torso_sat > 0.25).mean())
            legs_flat = float(legs.mean()) if legs.size else 0.0

            # Torso should differ in color from the head (shirt vs skin).
            head_hue = self._dominant_hue(hsv[hy:hy + hh, hx:hx + hw])
            torso_hue = self._dominant_hue(hsv[body_top:torso_y2, bx1:bx2])
            hue_diff = min(abs(head_hue - torso_hue), 180 - abs(head_hue - torso_hue)) / 90.0
            hue_score = min(1.0, hue_diff * 1.6 + 0.25)

            structure = (0.45 * torso_flat + 0.2 * legs_flat
                         + 0.2 * torso_colorful + 0.15 * hue_score)
            score = 0.45 * head["score"] + 0.55 * structure
            if score < 0.30:
                continue

            # Character box: from head top to lowest flat evidence below.
            char_bottom = self._find_body_bottom(flatness, bx1, bx2, body_top,
                                                 body_bottom, hh)
            x, y = bx1, max(0, hy - int(hh * 0.15))
            detections.append(Detection(
                x=x, y=y, w=bx2 - bx1, h=char_bottom - y,
                score=float(np.clip(score, 0, 1)),
                evidence={"head": head["score"], "torso_flat": torso_flat,
                          "structure": structure, "source_assembly": 1.0},
            ))
        return detections

    @staticmethod
    def _dominant_hue(hsv_patch: np.ndarray) -> float:
        if hsv_patch.size == 0:
            return 0.0
        hist = cv2.calcHist([hsv_patch], [0], None, [36], [0, 180])
        return float(np.argmax(hist)) * 5.0 + 2.5

    @staticmethod
    def _find_body_bottom(flatness, x1, x2, top, max_bottom, head_h) -> int:
        """Scan down: the body ends where flat-color evidence fades."""
        bottom = min(top + head_h, max_bottom)
        for y in range(top, max_bottom, max(2, head_h // 6)):
            row = flatness[y:y + 2, x1:x2]
            if row.size and row.mean() > 0.5:
                bottom = min(y + 2, max_bottom)
        return max(bottom, top + head_h)

    # -- strategy 4: standalone flat-region proposals --------------------------

    def _flat_region_candidates(self, work, smooth, flatness) -> list[Detection]:
        """Blocky clusters of flat saturated color, humanoid aspect ratio.

        Catches characters whose head is occluded, hatted, or non-skin-toned.
        """
        h, w = work.shape[:2]
        hsv = cv2.cvtColor(smooth, cv2.COLOR_BGR2HSV)
        sat = hsv[:, :, 1]
        val = hsv[:, :, 2]
        mask = ((flatness > 0.55) & (sat > 55) & (val > 45)).astype(np.uint8) * 255
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

        detections = []
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        min_h = h * self.config.min_char_height_frac
        for c in contours:
            x, y, bw, bh = cv2.boundingRect(c)
            if bh < min_h or bw < 4:
                continue
            aspect = bh / bw
            if not 0.9 <= aspect <= 4.5:        # humanoid: taller than wide
                continue
            area = cv2.contourArea(c)
            extent = area / (bw * bh)
            if extent < 0.28:
                continue
            aspect_score = math.exp(-((aspect - 1.9) ** 2) / 2.2)
            region_flat = float(flatness[y:y + bh, x:x + bw].mean())
            # How many distinct flat color blocks (avatars have several parts).
            parts = self._count_color_parts(smooth[y:y + bh, x:x + bw])
            parts_score = min(1.0, parts / 3.0)
            score = (0.32 * aspect_score + 0.30 * region_flat
                     + 0.18 * min(1.0, extent * 1.6) + 0.20 * parts_score)
            score *= 0.92  # standalone flat regions are weaker evidence
            if score < 0.25:
                continue
            detections.append(Detection(
                x=x, y=y, w=bw, h=bh, score=float(np.clip(score, 0, 1)),
                evidence={"aspect": aspect_score, "flat": region_flat,
                          "parts": float(parts), "source_flat_region": 1.0},
            ))
        return detections

    @staticmethod
    def _count_color_parts(patch_bgr: np.ndarray) -> int:
        """Count distinct large flat-color blobs inside a patch (posterized)."""
        if patch_bgr.size == 0:
            return 0
        small = cv2.resize(patch_bgr, (48, 96), interpolation=cv2.INTER_AREA)
        poster = (small // 40).astype(np.uint8)
        key = (poster[:, :, 0].astype(np.int32) * 49
               + poster[:, :, 1].astype(np.int32) * 7
               + poster[:, :, 2].astype(np.int32))
        parts = 0
        for k in np.unique(key):
            m = (key == k).astype(np.uint8)
            if m.sum() < 48 * 96 * 0.04:
                continue
            n, _ = cv2.connectedComponents(m)
            parts += max(0, n - 1)
        return parts

    # -- strategy 5: HOG humanoid votes ----------------------------------------

    def _hog_candidates(self, work, flatness) -> list[Detection]:
        h, w = work.shape[:2]
        if min(h, w) < 96:
            return []
        rects, weights = self._hog.detectMultiScale(
            work, winStride=(8, 8), padding=(8, 8), scale=1.06)
        detections = []
        for (x, y, bw, bh), weight in zip(rects, np.ravel(weights)):
            # HOG boxes are padded; tighten them.
            pad_x, pad_y = int(bw * 0.12), int(bh * 0.06)
            x, y = x + pad_x, y + pad_y
            bw, bh = bw - 2 * pad_x, bh - 2 * pad_y
            x, y = max(0, x), max(0, y)
            region_flat = float(flatness[y:y + bh, x:x + bw].mean()) if bw > 0 and bh > 0 else 0
            # A humanoid silhouette that is ALSO flat-colored is very likely
            # a Roblox character; textured humanoids (real people) score low.
            score = min(1.0, 0.30 + 0.25 * float(weight)) * (0.45 + 0.55 * region_flat)
            if score < 0.2:
                continue
            detections.append(Detection(
                x=int(x), y=int(y), w=int(bw), h=int(bh),
                score=float(np.clip(score, 0, 1)),
                evidence={"hog": float(weight), "flat": region_flat,
                          "source_hog": 1.0},
            ))
        return detections

    # -- strategy 6: optional ONNX (YOLO-style) --------------------------------

    def _onnx_candidates(self, work) -> list[Detection]:
        size = self.config.onnx_input_size
        h, w = work.shape[:2]
        blob = cv2.dnn.blobFromImage(work, 1 / 255.0, (size, size),
                                     swapRB=True, crop=False)
        self._onnx_net.setInput(blob)
        out = self._onnx_net.forward()
        out = np.squeeze(out)
        if out.ndim != 2:
            return []
        if out.shape[0] < out.shape[1]:      # (84, 8400) -> (8400, 84)
            out = out.T
        detections = []
        sx, sy = w / size, h / size
        for row in out:
            cx, cy, bw, bh = row[:4]
            scores = row[4:]
            cls = int(np.argmax(scores))
            conf = float(scores[cls])
            if conf < self.config.onnx_score_threshold:
                continue
            if cls != 0:                     # class 0 = person in COCO models
                continue
            x = int((cx - bw / 2) * sx)
            y = int((cy - bh / 2) * sy)
            detections.append(Detection(
                x=x, y=y, w=int(bw * sx), h=int(bh * sy),
                score=min(1.0, conf * 1.05),
                evidence={"onnx": conf, "source_onnx": 1.0},
            ))
        return detections

    # -- fusion -----------------------------------------------------------------

    def _filter_by_size(self, dets: list[Detection], shape) -> list[Detection]:
        h = shape[0]
        min_h = h * self.config.min_char_height_frac
        max_h = h * self.config.max_char_height_frac
        return [d for d in dets if min_h <= d.h <= max_h and d.w >= 3]

    def _fuse(self, dets: list[Detection]) -> list[Detection]:
        """Greedy NMS with evidence fusion: overlapping boxes from different
        strategies boost each other's confidence instead of just vanishing."""
        dets = sorted(dets, key=lambda d: d.score, reverse=True)
        kept: list[Detection] = []
        for d in dets:
            merged = False
            for k in kept:
                if _iou(d.bbox, k.bbox) >= self.config.nms_iou:
                    # Independent strategies agreeing => boost confidence.
                    same_source = any(
                        s in k.evidence for s in d.evidence if s.startswith("source_"))
                    if not same_source:
                        k.score = float(min(1.0, k.score + d.score * 0.35))
                        k.x, k.y, k.w, k.h = _merge_boxes(k.bbox, d.bbox) \
                            if d.score > 0.5 * k.score else k.bbox
                    k.evidence.update(
                        {kk: vv for kk, vv in d.evidence.items() if kk not in k.evidence})
                    merged = True
                    break
            if not merged:
                kept.append(d)
        return kept


# ---------------------------------------------------------------------------
# Annotation helper (used by GUI and CLI)
# ---------------------------------------------------------------------------


def annotate(image_bgr: np.ndarray, detections: list[Detection]) -> np.ndarray:
    out = image_bgr.copy()
    for i, d in enumerate(detections):
        color = (60, 220, 60) if d.score >= 0.6 else (40, 170, 255)
        cv2.rectangle(out, (d.x, d.y), (d.x + d.w, d.y + d.h), color, 2)
        label = f"#{i + 1} {d.score:.0%}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        ty = d.y - 6 if d.y - th - 10 > 0 else d.y + th + 8
        cv2.rectangle(out, (d.x, ty - th - 4), (d.x + tw + 6, ty + 4), color, -1)
        cv2.putText(out, label, (d.x + 3, ty), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (10, 10, 10), 2, cv2.LINE_AA)
    return out
