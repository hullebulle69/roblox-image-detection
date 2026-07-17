# Roblox Character Detector — C++ live tool

Real-time, GPU-accelerated detection of Roblox characters on your screen,
with confidence-scored boxes drawn on a transparent overlay.

```
┌─ capture thread ──────────────┐   ┌─ render (main) thread ─────────┐
│ DXGI Desktop Duplication      │   │ DirectComposition overlay      │
│   → BGRA frame                │   │   click-through, per-pixel     │
│   → letterbox → CHW tensor    │   │   alpha, vsync'd, GPU-composed │
│   → ONNX Runtime inference    ├──▶│   Direct2D boxes + labels      │
│     (TensorRT│CUDA│DirectML)  │   │   + HUD (provider, FPS, ms)    │
│   → decode + NMS              │   │                                │
└───────────────────────────────┘   └────────────────────────────────┘
```

> **Scope / ethics.** Like the Python tool in this repository, this is an
> *analysis and visualization* tool: it marks characters that are already
> visible on your screen. It does not read game memory, does not see
> through walls, and contains no input automation — no aiming, clicking, or
> key sending. That functionality is intentionally absent and won't be
> added. Automating gameplay violates the Roblox Terms of Use.

## Binaries

| Binary | Platform | Purpose |
|--------|----------|---------|
| `rcd_live` | Windows 10 2004+ | Live overlay: capture → GPU inference → boxes with accuracy scores |
| `rcd_cli` | Windows/Linux/macOS | Same detector on image files: batch annotation, JSON export, benchmarking |

Both share one detector core (`src/detector/`), so `rcd_cli` is the fast way
to validate a model/backend before going live.

## Getting a model

The tools accept any YOLOv5/v8/v11-family ONNX export. For actual Roblox
accuracy, train one with the recipe in [`../training/`](../training/README.md)
— synthetic bootstrap in minutes, fine-tuned on your own screenshots for
real quality. Class names are read from the ONNX metadata automatically.

## Building (Windows, Visual Studio 2022)

1. Download an ONNX Runtime package and unzip it, e.g. one of:
   - **DirectML** (works on any DX12 GPU — NVIDIA/AMD/Intel; easiest):
     nuget `Microsoft.ML.OnnxRuntime.DirectML` (a `.nupkg` is a zip)
   - **CUDA/TensorRT** (NVIDIA): `onnxruntime-win-x64-gpu-*.zip` from
     [ONNX Runtime releases](https://github.com/microsoft/onnxruntime/releases)
   - CPU-only fallback: `onnxruntime-win-x64-*.zip`

2. Configure and build:

```powershell
cd cpp
cmake -S . -B build -DONNXRUNTIME_ROOT=C:/path/to/onnxruntime
cmake --build build --config Release
```

The ONNX Runtime DLLs are copied next to the executables automatically.

## Building (Linux — `rcd_cli` only)

```bash
cd cpp
cmake -S . -B build -DONNXRUNTIME_ROOT=/path/to/onnxruntime-linux-x64
cmake --build build
```

## Running live

```powershell
rcd_live --model roblox_yolo.onnx                 # auto-picks best GPU backend
rcd_live --model m.onnx --backend directml --conf 0.4 --monitor 1
```

- Hotkeys: **F8** show/hide overlay, **F9** pause, **F10** quit.
- Backend order under `auto`: TensorRT → CUDA → DirectML → CPU. The active
  provider is shown in the HUD along with FPS, latency, and the live
  character count.
- Run Roblox **windowed or borderless fullscreen** — exclusive fullscreen
  bypasses desktop composition, so no overlay of any kind can draw above it.
- First TensorRT run builds an engine (can take minutes); cached in
  `trt_engine_cache/` under the directory you launch from afterwards.
- The overlay is excluded from screen capture
  (`WDA_EXCLUDEFROMCAPTURE`), so the detector never sees — and re-detects —
  its own boxes. Side effect: the boxes won't appear in OBS/recordings
  either; record with a camera-style capture of the game window instead if
  you want them in footage.
- DPI scaling is handled (per-monitor v2); boxes align with pixels on
  scaled displays. Rotated monitors aren't supported yet.

## CLI examples

```bash
rcd_cli shot.png --model m.onnx                    # annotate → shot.png.det.png
rcd_cli shot.png --model m.onnx --json - --no-out  # machine-readable output
rcd_cli shot.png --model m.onnx --bench 100        # latency/FPS for this machine
rcd_cli shot.png --model coco.onnx --classes person --conf 0.5
```

## Accuracy & performance notes

- Detection quality is decided by the **model**, not the runtime — see
  [`../training/README.md`](../training/README.md). The pipeline itself is
  verified equivalent to the ultralytics reference implementation
  (identical box counts, IoU ≥ 0.99 mean, score deltas < 0.01 across the
  validation suite).
- Confidence scores are shown per box and color-graded (green ≥ 75%,
  yellow ≥ 50%, orange below).
- `--conf` trades recall vs false positives at runtime; `--iou` controls
  how aggressively overlapping boxes merge.
- Preprocessing is multithreaded on CPU; inference runs on the GPU via the
  chosen execution provider. On a midrange GPU with a `yolov8s` model at
  640px, expect well over 60 FPS end-to-end.

## Third-party code

- [ONNX Runtime](https://github.com/microsoft/onnxruntime) (MIT) — linked, not vendored
- [stb_image / stb_image_write](https://github.com/nothings/stb) (public domain) — vendored in `third_party/stb`
- [font8x8](https://github.com/dhepper/font8x8) (public domain) — vendored in `third_party/font8x8`
