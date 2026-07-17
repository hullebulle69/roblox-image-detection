# Training a Roblox character model

The C++ tools (`cpp/`) run any YOLO-family model exported to ONNX. Detection
quality is almost entirely decided by the model you feed them, so this
directory contains everything needed to produce a strong one.

Install once:

```bash
pip install ultralytics
```

## Step 0 — bootstrap with synthetic data (no labeling)

```bash
python training/synthetic_dataset.py --out datasets/roblox_synth --train 600 --val 150
python training/train_roblox_yolo.py --data datasets/roblox_synth/data.yaml \
    --weights yolov8n.yaml --epochs 60
```

`synthetic_dataset.py` renders procedural blocky avatars (R6-style
proportions, flat saturated colors, hats, occlusion, distractor geometry)
over game-like backdrops. A model trained on it already finds avatars in
easy footage and validates your whole pipeline end to end — the repository's
C++ detector was verified against exactly this.

## Step 1 — collect real screenshots

Play the games you care about and screenshot every few seconds (Win+PrtScn,
OBS, or any capture tool). Variety beats volume: different games, maps,
graphics quality levels, avatar scales, day/night lighting, crowded and
empty scenes. 300–800 frames is a solid start.

## Step 2 — label

Draw a box around every visible character and give it the single class
`roblox_character`. [Roboflow](https://roboflow.com),
[Label Studio](https://labelstud.io) and
[labelImg](https://github.com/HumanSignal/labelImg) all export YOLO format
directly. Roboflow Universe also hosts public Roblox datasets you can merge
with your own to grow the training set quickly.

Label edge cases deliberately: partially occluded avatars, avatars behind
glass, tiny distant ones, and your own character's arms — decide whether
those should count and label consistently.

## Step 3 — fine-tune

```bash
python training/train_roblox_yolo.py --data my_dataset/data.yaml \
    --weights yolov8s.pt --epochs 100 --imgsz 640
```

Tips for "super great" accuracy:

- **Start from pretrained weights** (`yolov8s.pt`); COCO pretraining helps
  even though Roblox avatars aren't in COCO.
- **Model size vs speed:** `yolov8n` > 150 FPS on midrange GPUs, `yolov8s`
  is a good accuracy default, `yolov8m` if you have GPU headroom.
- **Mine your failures:** run `rcd_cli` on new screenshots, collect the ones
  with misses or false boxes, label them, retrain. Two or three of these
  loops improve real-world accuracy more than any hyperparameter.
- **Keep a held-out val set** from games you did *not* train on to catch
  overfitting to one game's art style.

The script prints mAP/precision/recall after training and exports
`best.onnx` automatically (class names embedded — the C++ tools pick them
up without extra config).

## Step 4 — run it

```bash
# live overlay (Windows, GPU):
rcd_live --model runs/roblox/weights/best.onnx

# stills / batch / benchmarking (any OS):
rcd_cli screenshot.png --model runs/roblox/weights/best.onnx
```
