# six-seven

Brief experimental computer vision workspace for gesture & object detection and data collection. Don't ask why it's called six-seven. (Disclaimer: This README is AI-assisted)

## Overview
This project contains three main scripts:
- `src/capture_training_data.py`: Fast image capture tool for collecting labeled gesture datasets. Supports continuous mode, single-shot capture, adjustable FPS, and per-gesture folder organization.
- `src/detect.py`: Runs real-time YOLOv8 + DeepFace emotion analysis on webcam frames, overlaying emotion confidence scores.
- `src/infer.py`: Executes a Roboflow workflow (`InferencePipeline`) that performs detection + classification, annotates frames with bounding boxes and labels, and optionally displays reference images per detected class.

## Features
- Lightweight YOLOv8 model (`yolov8n.pt`) included for rapid prototyping.
- Gesture dataset capture with keyboard controls (`g`, `c`, space, `+/-`, `q`).
- Real-time emotion percentages (DeepFace) overlay.
- Roboflow workflow integration for higher-level composed inference.
- Reference image caching to reduce I/O overhead.

## Requirements
Create and activate a virtual environment, install dependencies:
```bash
pip install ultralytics deepface python-dotenv supervision networkx
```
If using Roboflow workflows, set your API key in a `.env` file at project root:
```
ROBOFLOW_API_KEY="YOUR_KEY"
```

## Usage

### Local scripts
Capture gesture training data:
```bash
python src/capture_training_data.py
```
Run YOLO + emotion detection:
```bash
python src/detect.py
```
Run Roboflow workflow inference:
```bash
python src/infer.py
```

### Web demo (Roboflow-hosted model)
This is a simple website-style demo: the browser captures webcam frames and sends them to a local FastAPI server, which forwards them to Roboflow using your API key (so the key never reaches the browser).

1) Create `.env` at repo root:
```
ROBOFLOW_API_KEY="YOUR_KEY"
ROBOFLOW_MODEL_ID="gesture-recognition-jemzp/5"
```

2) Install dependencies (Ubuntu / CPU):
```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install fastapi uvicorn requests python-dotenv "python-multipart>=0.0.20"
```

3) Start the server:
```bash
uvicorn src.web_demo_server:app --reload --port 8000
```

4) Open the demo page:
- http://localhost:8000/

Exit any live window with `q`.

## Data Output
Captured gesture images are saved under `public/training-data/<gesture_name>/` with timestamped filenames. Reference images for display should be placed in `public/reference-images/` named by class (e.g., `thumbs-up.png`).

## Performance Tips
- Lower frame resolution before inference for higher FPS.
- Use `FRAME_SKIP` in `infer.py` (if present) to process every Nth frame.
- Cache reference images (already implemented) to avoid repeated disk reads.
- Throttle expensive emotion analysis (every N frames) if adding to other scripts.

## Future Improvements
- Add logging of predictions to CSV / JSON.
- Integrate model training pipeline for collected gestures.
- Optional GPU acceleration / ONNX export.

## License
See `LICENSE` file.
