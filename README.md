# VIBES: Vision-language Models Guided by Bayesian Inference

<p align="center">
  <b>Efficient Far-field Anomaly Detection in Expressway Surveillance Videos via Focused VLM Reasoning Guided by Bayesian Inference</b>
</p>

<p align="center">
  <a href="#overview">Overview</a> |
  <a href="#demo">Demo</a> |
  <a href="#method">Method</a> |
  <a href="#installation">Installation</a> |
  <a href="#quick-start">Quick Start</a>
</p>

VIBES is a training-free framework for detecting and explaining far-field traffic anomalies in fixed-camera surveillance videos. Instead of sending dense global video frames directly to a vision-language model, VIBES first tracks vehicles, estimates local traffic motion, and uses kinematics-guided Bayesian inference to identify high-surprise events. Only compact event-centered visual evidence is then sent to a local VLM for semantic verification.

## Overview

<p align="center">
  <img src="assets/fig/Intro.pdf" alt="VIBES overview" width="100%">
</p>

The current implementation is organized as a strict three-stage pipeline:

1. **Detection, tracking, and Bayesian surprise.** YOLO26-L and periodic SAHI rescue detect vehicles, ByteTrack maintains trajectories, and the motion model scores abnormal lateral, longitudinal, stop, and impact evidence.
2. **Adaptive spatiotemporal crop generation.** High-surprise proposals are grouped into 8-second segments, with at most four events retained per segment. The crop stage generates focused keyframes, comparison images, storyboards, metadata, and audit views.
3. **Focused local VLM inference.** After detection and crop generation finish and the detector is released, Qwen3-VL processes each event package and writes raw event-level answers plus run-level CSV/JSON summaries.

## Demo

<p align="center"><strong>Case 1</strong></p>
<p align="center">
  <img src="assets/case/case1.gif" alt="Case 1 demo" width="100%">
</p>

<p align="center"><strong>Case 2</strong></p>
<p align="center">
  <img src="assets/case/case2.gif" alt="Case 2 demo" width="100%">
</p>

<p align="center"><strong>Case 3</strong></p>
<p align="center">
  <img src="assets/case/case3.gif" alt="Case 3 demo" width="100%">
</p>

<p align="center"><strong>Case 4</strong></p>
<p align="center">
  <img src="assets/case/case4.gif" alt="Case 4 demo" width="100%">
</p>

## Method

<p align="center">
  <img src="assets/fig/Model.pdf" alt="VIBES method overview" width="100%">
</p>

VIBES follows a zoom-in-and-reason-out design:

1. **Vehicle detection and tracking.** Full-frame YOLO detection is combined with periodic sliced detection for small far-field vehicles, followed by lightweight trajectory association.
2. **Road-aware kinematic decomposition.** Vehicle motion is projected into a local traffic-aligned frame so longitudinal and lateral behavior can be modeled separately while using nearby traffic as context.
3. **Kinematics-guided Bayesian inference.** Online normal-motion baselines are estimated from reliable trajectory evidence. Deviations from the expected motion envelope produce Bayesian-surprise-style anomaly scores, while stop and impact evidence are incorporated into the same object-centric event scoring process.
4. **Adaptive evidence extraction.** High-score trajectories are consolidated into regional event proposals. The crop stage selects event-specific temporal context and produces focused visual evidence rather than passing the full video to the VLM.
5. **Focused VLM reasoning.** Qwen3-VL receives the event comparison image and storyboard and returns a raw natural-language description for each event.

## Repository Layout

```text
VIBES/
├── README.md
├── requirements.txt
├── run.py
├── assets/
│   ├── case/
│   └── fig/
│       ├── Intro.pdf
│       ├── Model.pdf
│       ├── method.png
│       └── teaser.png
├── data/
├── models/
│   ├── yolo26l.pt
│   ├── yolo26l.engine
│   └── Qwen3-VL-8B-Instruct/
└── zerotad_adaptive/
    ├── __init__.py
    ├── adaptive_crop.py
    ├── anomaly.py
    ├── app.py
    ├── common.py
    ├── config.py
    ├── crop_cache.py
    ├── crop_stage.py
    ├── detection.py
    ├── road.py
    ├── runner.py
    ├── tracking.py
    ├── visualization.py
    └── vlm_local_inference.py
```

`data/` is the default input directory. Put the input `.mp4` files directly under this directory. `runs/` is created automatically when the pipeline starts. Model weights and TensorRT engines are not included in the repository archive.

## Installation

Create an environment and install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Install a CUDA-compatible PyTorch build for your machine when GPU inference is required.

### Prepare YOLO26-L

The default detector path in `zerotad_adaptive/config.py` is `models/yolo26l.engine`.

Download the YOLO26-L PyTorch weights:

```bash
python -c "from ultralytics import YOLO; YOLO('yolo26l.pt')"
mkdir -p models
mv yolo26l.pt models/yolo26l.pt
```

Export the model to TensorRT FP16 with the 1280-pixel input size used by the current pipeline:

```bash
yolo export model=models/yolo26l.pt format=engine imgsz=1280 device=0 quantize=16
```

After export, the detector should be available at:

```text
models/yolo26l.engine
```

If the engine is exported on another GPU or with a different TensorRT/CUDA environment, rebuild it on the target inference machine.

### Prepare Qwen3-VL

Place the local Qwen3-VL model at:

```text
models/Qwen3-VL-8B-Instruct/
```

Alternatively, provide another local model directory with `--vlm-model-path`.

## Quick Start

Put input videos directly in `data/`:

```text
data/
├── case1.mp4
├── case2.mp4
└── ...
```

Then run the full pipeline with the defaults defined in `zerotad_adaptive/config.py`:

```bash
python run.py
```

The default paths are:

```text
Input videos:   data/
YOLO engine:    models/yolo26l.engine
Qwen3-VL:       models/Qwen3-VL-8B-Instruct/
Outputs:        runs/<timestamp>/
```

All paths can still be overridden from the command line:

```bash
python run.py \
  --input-path data \
  --output-dir runs/example_run \
  --detector-model models/yolo26l.engine \
  --device cuda:0 \
  --vlm-model-path models/Qwen3-VL-8B-Instruct \
  --vlm-device cuda:0 \
  --crop-interval-seconds 8 \
  --max-events-per-segment 4
```

To run only detection, tracking, Bayesian surprise, and adaptive crop generation without loading the VLM:

```bash
python run.py --disable-vlm
```

To disable adaptive crop generation as well:

```bash
python run.py --disable-adaptive-crops --disable-vlm
```

## Outputs

A typical run is organized as:

```text
runs/<timestamp>/
├── batch_summary.csv
├── vlm_<tag>_<prompt-hash>_summary.csv
├── vlm_<tag>_<prompt-hash>_summary.json
└── <video_name>/
    ├── annotated_video.mp4
    ├── motion_scores.csv
    ├── road_direction_debug.jpg
    ├── traffic_occupancy_debug.jpg
    ├── adaptive_spatiotemporal_crops/
    │   ├── crop_manifest.csv
    │   └── segment_*/
    │       ├── event_*_vlm_comparison.jpg
    │       ├── event_*_vlm_storyboard.jpg
    │       ├── event_*.json
    │       ├── event_*_frames/
    │       └── event_*_vlm_<tag>_<prompt-hash>.json
    └── ...
```

The event-level crop folders are the main bridge between Bayesian surprise detection and VLM reasoning. They preserve the focused visual evidence and metadata used for each semantic decision.
