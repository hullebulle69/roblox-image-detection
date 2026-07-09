"""Tests for the Roblox character detector.

Synthetic Roblox-style avatars are drawn procedurally so the tests need no
image assets and run deterministically.
"""

import os
import sys

import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from roblox_detector import (  # noqa: E402
    Detection,
    DetectorConfig,
    RobloxCharacterDetector,
    annotate,
)
from roblox_detector.detector import _iou, _merge_boxes  # noqa: E402


def draw_avatar(canvas, cx, top, head=40, shirt=(200, 60, 60), pants=(60, 60, 200)):
    """Draw a simple blocky R6-style avatar on a BGR canvas."""
    hh = head
    # Head (classic bright yellow), a flat square.
    cv2.rectangle(canvas, (cx - hh // 2, top), (cx + hh // 2, top + hh),
                  (60, 220, 235), -1)
    torso_top = top + hh
    torso_w = int(hh * 1.6)
    torso_h = int(hh * 1.4)
    cv2.rectangle(canvas, (cx - torso_w // 2, torso_top),
                  (cx + torso_w // 2, torso_top + torso_h), shirt, -1)
    # Arms.
    arm_w = int(hh * 0.5)
    cv2.rectangle(canvas, (cx - torso_w // 2 - arm_w, torso_top),
                  (cx - torso_w // 2, torso_top + torso_h), shirt, -1)
    cv2.rectangle(canvas, (cx + torso_w // 2, torso_top),
                  (cx + torso_w // 2 + arm_w, torso_top + torso_h), shirt, -1)
    # Legs.
    legs_top = torso_top + torso_h
    legs_h = int(hh * 1.4)
    cv2.rectangle(canvas, (cx - torso_w // 2, legs_top),
                  (cx - 2, legs_top + legs_h), pants, -1)
    cv2.rectangle(canvas, (cx + 2, legs_top),
                  (cx + torso_w // 2, legs_top + legs_h), pants, -1)
    return (cx - torso_w // 2 - arm_w, top,
            torso_w + 2 * arm_w, hh + torso_h + legs_h)


def make_scene(avatars=1, size=(600, 800)):
    canvas = np.full((size[0], size[1], 3), 120, np.uint8)  # flat gray sky
    # A little ground texture so it isn't a trivially uniform image.
    noise = np.random.RandomState(0).randint(-8, 8, canvas.shape, dtype=np.int16)
    canvas = np.clip(canvas.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    boxes = []
    positions = np.linspace(150, size[1] - 150, avatars)
    for cx in positions:
        boxes.append(draw_avatar(canvas, int(cx), 120, head=46))
    return canvas, boxes


# -- geometry helpers -------------------------------------------------------


def test_iou_identical():
    assert _iou((0, 0, 10, 10), (0, 0, 10, 10)) == pytest.approx(1.0)


def test_iou_disjoint():
    assert _iou((0, 0, 10, 10), (100, 100, 10, 10)) == 0.0


def test_iou_partial():
    assert 0.0 < _iou((0, 0, 10, 10), (5, 0, 10, 10)) < 1.0


def test_merge_boxes():
    assert _merge_boxes((0, 0, 10, 10), (5, 5, 10, 10)) == (0, 0, 15, 15)


# -- detection --------------------------------------------------------------


def test_detects_single_avatar():
    scene, boxes = make_scene(avatars=1)
    det = RobloxCharacterDetector(DetectorConfig(confidence_threshold=0.3))
    results = det.detect(scene)
    assert len(results) >= 1
    # Best detection should substantially overlap the drawn avatar.
    best = max((_iou(r.bbox, boxes[0]) for r in results), default=0)
    assert best > 0.25


def test_detects_multiple_avatars():
    scene, boxes = make_scene(avatars=3)
    det = RobloxCharacterDetector(DetectorConfig(confidence_threshold=0.3))
    results = det.detect(scene)
    assert len(results) >= 2  # should find most of the three


def test_empty_scene_few_false_positives():
    canvas = np.full((600, 800, 3), 120, np.uint8)
    noise = np.random.RandomState(1).randint(-6, 6, canvas.shape, dtype=np.int16)
    canvas = np.clip(canvas.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    det = RobloxCharacterDetector(DetectorConfig(confidence_threshold=0.4))
    results = det.detect(canvas)
    assert len(results) <= 1


def test_handles_none_and_empty():
    det = RobloxCharacterDetector()
    assert det.detect(None) == []
    assert det.detect(np.zeros((0, 0, 3), np.uint8)) == []


def test_handles_grayscale_and_rgba():
    scene, _ = make_scene(avatars=1)
    gray = cv2.cvtColor(scene, cv2.COLOR_BGR2GRAY)
    rgba = cv2.cvtColor(scene, cv2.COLOR_BGR2BGRA)
    det = RobloxCharacterDetector(DetectorConfig(confidence_threshold=0.3))
    det.detect(gray)   # must not raise
    det.detect(rgba)   # must not raise


def test_scores_in_range():
    scene, _ = make_scene(avatars=2)
    det = RobloxCharacterDetector()
    for r in det.detect(scene):
        assert 0.0 <= r.score <= 1.0


def test_boxes_within_bounds():
    scene, _ = make_scene(avatars=2)
    h, w = scene.shape[:2]
    det = RobloxCharacterDetector()
    for r in det.detect(scene):
        assert 0 <= r.x < w and 0 <= r.y < h
        assert r.x + r.w <= w and r.y + r.h <= h


def test_threshold_filters():
    scene, _ = make_scene(avatars=3)
    low = RobloxCharacterDetector(DetectorConfig(confidence_threshold=0.2)).detect(scene)
    high = RobloxCharacterDetector(DetectorConfig(confidence_threshold=0.75)).detect(scene)
    assert len(high) <= len(low)


def test_large_image_downscaled_and_mapped_back():
    scene, _ = make_scene(avatars=1, size=(2000, 2600))
    det = RobloxCharacterDetector(DetectorConfig(max_working_size=800,
                                                 confidence_threshold=0.3))
    results = det.detect(scene)
    for r in results:
        assert r.x + r.w <= 2600 and r.y + r.h <= 2000


def test_annotate_runs():
    scene, _ = make_scene(avatars=1)
    det = RobloxCharacterDetector()
    out = annotate(scene, det.detect(scene))
    assert out.shape == scene.shape


def test_detection_as_dict():
    d = Detection(1, 2, 3, 4, 0.5, evidence={"head": 0.9})
    data = d.as_dict()
    assert data["x"] == 1 and data["score"] == 0.5
    assert data["evidence"]["head"] == 0.9


def test_detect_file_roundtrip(tmp_path):
    scene, _ = make_scene(avatars=1)
    path = str(tmp_path / "scene.png")
    cv2.imwrite(path, scene)
    det = RobloxCharacterDetector(DetectorConfig(confidence_threshold=0.3))
    assert len(det.detect_file(path)) >= 1


def test_missing_onnx_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        RobloxCharacterDetector(DetectorConfig(onnx_model_path=str(tmp_path / "no.onnx")))
