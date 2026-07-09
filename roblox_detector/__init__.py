"""Roblox character image detector.

A multi-strategy computer-vision pipeline for locating Roblox avatars in
images and screenshots, with a desktop GUI and CLI.
"""

from .detector import (
    Detection,
    DetectorConfig,
    RobloxCharacterDetector,
    annotate,
)

__all__ = [
    "Detection",
    "DetectorConfig",
    "RobloxCharacterDetector",
    "annotate",
]

__version__ = "1.0.0"
