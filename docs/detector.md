# Unit detector: current state and plan

## Today

- Two YOLOv8-L detectors from KataCR (43.7 M parameters each, 896 x 576 input). Each one knows half
  of the 150 classes, so both are needed.
- About 500 ms for both on a Mac (MPS). On a GTX 1660 Ti the measured time is 600 to 670 ms per frame
  when the laptop GPU is power-limited to 30 W, and 251 ms at 80 W.
- Speed-ups in the kit that don't change the detections:
  - conv + batch-norm layers fused (15 % faster on the Mac, same detections on the 4 test images);
  - `cudnn.benchmark`;
  - the two detectors run at the same time on two CUDA streams;
  - FP16 or FP32 picked at start-up by timing both;
  - above 150 ms for both, they alternate (one per frame, reusing the other's last result).
- The detectors were trained on 2024 visuals from a 1080 x 2400 phone. They don't know the newer
  tower skins nor the Google Play Games for PC rendering.

## Options, from simplest to heaviest

### 1. Tune thresholds on real frames

The kit log keeps every detection above 0.2 confidence and two raw frames per second. With
`python bot_pc.py replay <log folder>` we can tune, on real frames: per-class thresholds, the
deploy-clock threshold and the ground point of each unit.

### 2. Fine-tune the KataCR detectors on the PC rendering (one night on the cluster)

- Data: KataCR's generative dataset
  ([Clash-Royale-Detection-Dataset](https://github.com/wty-yy/Clash-Royale-Detection-Dataset), about
  600 MB of unit crops and arenas, MIT) plus raw frames from the kit, pseudo-labelled by the current
  detectors and the twin, with a sample checked by hand.
- Method: start from the current weights and add the new visuals to KataCR's generator.
- Expected gain: fewer missed units and fewer false detections on the PC.

### 3. A newer, lighter detector: YOLO11 (one or two nights)

- YOLO11-M or YOLO11-S (Ultralytics 8.3): about 2 to 4 times faster than two YOLOv8-L at similar
  accuracy, and a single model for all classes.
- The side (own or enemy) is learned by doubling the classes (`hog-rider_0`, `hog-rider_1`) instead
  of KataCR's custom head.
- FP16 ONNX export, so it runs on any GPU with onnxruntime (CUDA, or DirectML on Windows for AMD and
  Intel cards).
- Needs a newer Ultralytics in the kit (8.1.24 today) or vision through onnxruntime.

### 4. TensorRT (later)

About 2 times faster again on NVIDIA cards, but heavy to install and the model has to be compiled for
each card.
