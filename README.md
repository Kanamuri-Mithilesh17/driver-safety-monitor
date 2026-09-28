# Driver Safety Monitor

A computer-vision prototype for analyzing forward-facing driving video with **lane detection, ego-lane position estimation, lane-change detection, and vehicle detection using YOLO**.

## Features

- Road/lane marking detection with OpenCV
- Ego vehicle lane-position estimation
- LEFT / RIGHT lane-change detection
- Lane-change counting with temporal confirmation and cooldown
- Vehicle detection for cars, motorcycles, buses, and trucks using YOLO
- Vehicle bounding boxes and confidence scores
- Vehicle count overlay
- Annotated output video
- CSV logging of confirmed lane-change events

## Pipeline

```text
Dashcam video
     ↓
Lane marking detection ──→ Ego lane geometry
     ↓                         ↓
Lane-change tracker       Lane-change event
     
Dashcam frame
     ↓
YOLO vehicle detection
     ↓
Vehicle bounding boxes + classes
     
             ↓
     Annotated output video
             +
       Lane-change CSV
```

## Tech Stack

- Python
- OpenCV
- NumPy
- Ultralytics YOLO
- CSV / pathlib

## Project Structure

```text
driver-safety-monitor/
├── src/
│   └── driver_safety_monitor.py
├── output/
│   └── .gitkeep
├── .gitignore
├── README.md
└── requirements.txt
```

## Installation

Python 3.10+ is recommended.

```bash
pip install -r requirements.txt
```

The YOLO model is loaded with:

```python
YOLO("yolo11n.pt")
```

Ultralytics can download the model weights when the program is first run.

## Running the Project

Place your driving video in the project root and name it:

```text
input_1.mp4
```

Then run:

```bash
python src/driver_safety_monitor.py
```

The program creates:

```text
output/
├── driver_safety_output.mp4
└── lane_changes.csv
```

## Lane-Change Detection

The system estimates the lane boundaries around the ego vehicle and tracks changes in the estimated lane center over time.

A lane change is only confirmed after the detected change persists for multiple frames, with additional checks for lane-width consistency, reversal, and cooldown.

Each confirmed event is logged with:

- event number
- frame number
- video timestamp
- direction
- left-change count
- right-change count
- total-change count

## Vehicle Detection

YOLO is restricted to these vehicle classes:

| Class | COCO ID |
|---|---:|
| Car | 2 |
| Motorcycle | 3 |
| Bus | 5 |
| Truck | 7 |

The output video displays bounding boxes, class labels, and confidence scores.

## Important Limitations

This is a **computer-vision prototype**, not a production autonomous-driving or safety-critical system.

Lane detection can be affected by:

- poor or missing lane markings
- strong shadows
- road curvature
- unusual camera placement
- camera movement
- weather and lighting
- occlusion

Vehicle detection accuracy depends on the camera footage, model, lighting, and object size.

Distance estimation and collision prediction are not currently implemented.

## Future Improvements

- Multi-object vehicle tracking
- Relative vehicle distance estimation
- Time-to-collision estimation
- Following-distance warnings
- Blind-spot / adjacent-lane analysis
- Better lane tracking on curved roads
- Configurable video input from the command line
- Automated testing
- Performance optimization

## Project Status

**Working prototype.**

The current focus is reliable perception and lane-change analysis from forward-facing driving footage.
