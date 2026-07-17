Roblox Character Detector — Windows quick start
================================================

This package contains:
  rcd_live.exe             live GPU overlay: draws boxes on characters on screen
  rcd_cli.exe              run the detector on image files
  roblox_bootstrap_320.onnx  a starter model (trained on synthetic avatars)
  onnxruntime.dll, DirectML.dll, ...  the GPU runtime (keep next to the exes)

Requirements
------------
- Windows 10 version 2004 or newer, 64-bit
- A Direct3D 12 capable GPU (virtually all GPUs from the last decade —
  NVIDIA, AMD, or Intel). No CUDA install needed; this build uses DirectML.

Run the live overlay
--------------------
1. Open a Command Prompt in this folder (Shift+Right-click -> "Open in
   Terminal" / "Open command window here").
2. Start Roblox in WINDOWED or BORDERLESS mode (not exclusive fullscreen —
   no overlay can draw over exclusive fullscreen).
3. Run:

       rcd_live.exe --model roblox_bootstrap_320.onnx

   Boxes with confidence scores appear over detected characters.
   Hotkeys:  F8 show/hide overlay   F9 pause   F10 quit

   Pick a specific monitor with  --monitor 1 , raise/lower sensitivity with
   --conf 0.5 , or force a backend with  --backend directml .

Try it on a screenshot first
----------------------------
       rcd_cli.exe myscreenshot.png --model roblox_bootstrap_320.onnx

   Writes myscreenshot.png.det.png with boxes drawn, and prints the
   detections. Add  --bench 100  to measure FPS on your machine.

Getting better accuracy
-----------------------
The bundled model is a synthetic bootstrap — good for a demo, not perfect on
every game's art style. Train a stronger one on your own screenshots using
the recipe in the training/ folder of the source repository. The exes accept
any YOLOv5/v8/v11 model exported to ONNX.

Note
----
This tool only visualizes characters already visible on your screen. It does
not read game memory, see through walls, or automate any input (no aiming or
clicking). Automating gameplay violates the Roblox Terms of Use.

If Windows SmartScreen warns about an unsigned executable, that is expected
for a community build; choose "More info" -> "Run anyway", or build from
source yourself (see cpp/README.md).
