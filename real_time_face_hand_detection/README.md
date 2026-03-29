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

## Controls

- Press **ESC** to exit.

## Troubleshooting

- **Camera not opening**: change `CAMERA_ID` in `config/settings.py` (try `0`, `1`, `2`).
- **Permission errors**: grant camera access in your OS privacy settings.
- **Low FPS**: reduce `FRAME_WIDTH`/`FRAME_HEIGHT`, or disable one detector.
- **Install issues**: ensure your Python version is compatible with installed OpenCV/MediaPipe wheels.
