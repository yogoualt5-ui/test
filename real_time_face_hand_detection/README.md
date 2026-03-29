# Real-Time Face and Hand Detection

A modular Python project that performs real-time face and hand detection from a webcam feed using OpenCV + MediaPipe.

## Requirements

- Python 3.7+
- Webcam

## Project Structure

```text
real_time_face_hand_detection/
├── config/
│   └── settings.py
├── src/
│   ├── main.py
│   ├── startup_checker.py
│   ├── detectors/
│   │   ├── face_detector.py
│   │   └── hand_detector.py
│   └── utils/
│       ├── drawing.py
│       └── fps.py
├── requirements.txt
└── README.md
```

## Setup

1. Clone or download this project.
2. (Optional) Create and activate a virtual environment.
3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Adjust tunable options in `config/settings.py` (camera ID, confidence thresholds, frame size, etc.).

## Run

From the project root:

```bash
python src/main.py
```

`main.py` now runs a startup preflight that checks requirements one by one (project structure, imports, MediaPipe API shape, detector imports, and camera availability). Detection starts only if all checks pass.

```bash
# Optional: run only preflight checks
python -c "from src.startup_checker import run_startup_checks; run_startup_checks()"
```

## Controls

- Press **ESC** to exit.

## Troubleshooting

- **Camera not opening**: change `CAMERA_ID` in `config/settings.py` (try `0`, `1`, `2`).
- **Permission errors**: grant camera access in your OS privacy settings.
- **Low FPS**: reduce `FRAME_WIDTH`/`FRAME_HEIGHT`, or disable one detector.
- **Install issues**: use Python 3.9-3.11 for the most reliable MediaPipe wheels, then reinstall dependencies.
- **`AttributeError: module 'mediapipe' has no attribute 'solutions'`**: uninstall conflicting packages and reinstall official MediaPipe:

  ```bash
  pip uninstall -y mediapipe mediapipe-nightly
  pip install --no-cache-dir mediapipe opencv-python
  ```

  Also make sure there is no local file/folder named `mediapipe` in your project.
