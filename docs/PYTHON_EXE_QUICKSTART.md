Roblox Character Detector (Python edition) — Windows quick start
================================================================

roblox-detector-py.exe is a standalone build of the Python image detector.
It finds Roblox avatars in still images / screenshots using a multi-strategy
computer-vision pipeline (flatness, head+body assembly, flat-color regions,
HOG silhouette, optional ONNX) and draws confidence-scored boxes. No Python
install required — everything is bundled in the one .exe.

This is the IMAGE-ANALYSIS tool: you give it screenshots and it annotates
them. (For the real-time GPU screen overlay, use the C++ build:
rcd_live.exe.)

Usage
-----
  Analyze one screenshot (writes test.det next to it and prints results):
      roblox-detector-py.exe shot.png --out annotated

  A whole folder, with a JSON report:
      roblox-detector-py.exe screenshots\ --out annotated --json results.json

  Tune sensitivity (0-1; lower = more boxes):
      roblox-detector-py.exe shot.png --threshold 0.5

  Add a neural model (any YOLO-style person/character ONNX):
      roblox-detector-py.exe shot.png --onnx yolov8n.onnx

  Options:  --no-hog (disable the HOG strategy for busy scenes)
            --quiet  (print only the summary)
            --help   (full option list)

First launch note
-----------------
A one-file PyInstaller exe unpacks itself on first run, so the first launch
takes a few seconds — subsequent runs are quick. If Windows SmartScreen warns
about an unsigned community build, choose "More info" -> "Run anyway".

Scope / ethics
--------------
This analyzes images you provide. It does not capture a live game, move the
mouse, aim, or click — it is not an aimbot. Automating gameplay violates the
Roblox Terms of Use.
