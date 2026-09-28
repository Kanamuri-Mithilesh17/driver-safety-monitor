import csv
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


# ============================================================
# DRIVER SAFETY MONITOR
#
# FEATURES:
#   1. Lane detection
#   2. Ego lane position
#   3. Lane-change detection
#   4. LEFT / RIGHT lane-change counting
#   5. Vehicle detection using YOLO
#   6. Vehicle bounding boxes
#   7. Vehicle count
#   8. Lane-change CSV logging
#   9. Annotated output video
# ============================================================


# ============================================================
# CONFIG
# ============================================================

VIDEO_PATH = "input_1.mp4"

MODEL_NAME = "yolo11n.pt"

OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_VIDEO = OUTPUT_DIR / "driver_safety_output.mp4"
OUTPUT_CSV = OUTPUT_DIR / "lane_changes.csv"


# ============================================================
# ROAD / LANE CONFIG
# ============================================================

# Tuned for the current portrait-style video.
ROI_POLY = [
    (0.00, 0.715),
    (0.30, 0.645),
    (0.66, 0.645),
    (0.88, 0.765)
]

MEASURE_Y = 0.71

MIN_ABS_SLOPE = 0.08

# Camera/ego vehicle center.
CAMERA_X_FRAC = 0.50

MIN_LANE_W_FRAC = 0.10
MAX_LANE_W_FRAC = 0.60

CLUSTER_TOL_FRAC = 0.05


# ============================================================
# LANE CHANGE CONFIG
# ============================================================

JUMP_MIN = 0.60
JUMP_MAX = 1.50

WIDTH_TOL = 0.35

NEAR_EDGE = 0.15

REVERT_RATIO = 0.35

CONFIRM_S = 0.4

COOLDOWN_S = 2.0

MAX_GAP_S = 1.0

BANNER_S = 2.0


# ============================================================
# VEHICLE DETECTION CONFIG
# ============================================================

# COCO vehicle classes:
#
# 2  = car
# 3  = motorcycle
# 5  = bus
# 7  = truck
#
VEHICLE_CLASSES = {
    2: "CAR",
    3: "MOTORCYCLE",
    5: "BUS",
    7: "TRUCK"
}

YOLO_CONFIDENCE = 0.35

YOLO_IOU = 0.45

# Detection every frame.
# Increase to 2 or 3 if your PC is slow.
YOLO_INTERVAL = 1

# Minimum bounding-box area.
MIN_VEHICLE_AREA = 400


# ============================================================
# COLORS
# ============================================================

GREEN = (0, 255, 0)
RED = (0, 0, 255)
BLUE = (255, 0, 0)
YELLOW = (0, 255, 255)
MAGENTA = (255, 0, 255)
WHITE = (255, 255, 255)
ORANGE = (0, 165, 255)


# ============================================================
# ROAD MASK
# ============================================================

def road_mask(shape):

    h, w = shape[:2]

    mask = np.zeros((h, w), dtype=np.uint8)

    poly = np.array(
        [[
            (int(x * w), int(y * h))
            for x, y in ROI_POLY
        ]],
        dtype=np.int32
    )

    cv2.fillPoly(mask, poly, 255)

    return mask


# ============================================================
# PAINT / LANE MARKING MASK
# ============================================================

def paint_mask(frame):

    h, w = frame.shape[:2]

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Top-hat operation helps highlight bright road markings.
    k = max(9, int(w * 0.02) | 1)

    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (k, k)
    )

    tophat = cv2.morphologyEx(
        gray,
        cv2.MORPH_TOPHAT,
        kernel
    )

    white = (
        (tophat > 25) &
        (gray > 110)
    ).astype(np.uint8) * 255

    yellow = cv2.inRange(
        hsv,
        np.array([15, 80, 120], np.uint8),
        np.array([35, 255, 255], np.uint8)
    )

    mask = cv2.bitwise_or(
        white,
        yellow
    )

    return cv2.bitwise_and(
        mask,
        road_mask(frame.shape)
    )


# ============================================================
# PICK LANE SIDE
# ============================================================

def pick_side(cands, cam_x, tol):

    if not cands:
        return None

    x_star = min(
        cands,
        key=lambda c: abs(c[0] - cam_x)
    )[0]

    group = [
        c for c in cands
        if abs(c[0] - x_star) <= tol
    ]

    total = sum(
        c[1] for c in group
    )

    if total <= 0:
        return None

    x = sum(
        c[0] * c[1]
        for c in group
    ) / total

    return x, [
        c[2]
        for c in group
    ]


# ============================================================
# LANE DETECTION
# ============================================================

def detect_lane(frame):

    h, w = frame.shape[:2]

    cam_x = w * CAMERA_X_FRAC

    measure_y = h * MEASURE_Y

    mask = paint_mask(frame)

    edges = cv2.Canny(
        mask,
        50,
        150
    )

    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180,
        threshold=15,
        minLineLength=max(
            15,
            int(w * 0.02)
        ),
        maxLineGap=40
    )

    left = []
    right = []

    if lines is not None:

        for raw in lines:

            x1, y1, x2, y2 = (
                int(v)
                for v in np.asarray(raw)
                .reshape(-1)[:4]
            )

            dx = x2 - x1
            dy = y2 - y1

            if dx == 0 or dy == 0:
                continue

            slope = dy / dx

            if not (
                MIN_ABS_SLOPE
                <= abs(slope)
                <= 4.0
            ):
                continue

            length = float(
                np.hypot(dx, dy)
            )

            x_m = (
                x1
                + (measure_y - y1)
                * dx / dy
            )

            if not (
                -0.5 * w
                <= x_m
                <= 1.5 * w
            ):
                continue

            item = (
                x_m,
                length,
                (x1, y1, x2, y2)
            )

            if x_m < cam_x:
                left.append(item)
            else:
                right.append(item)

    tol = w * CLUSTER_TOL_FRAC

    return (
        pick_side(left, cam_x, tol),
        pick_side(right, cam_x, tol)
    )


# ============================================================
# LANE GEOMETRY
# ============================================================

def lane_geometry(left, right, frame_w):

    if left is None or right is None:
        return None

    left_x = left[0]

    right_x = right[0]

    width = right_x - left_x

    if not (
        frame_w * MIN_LANE_W_FRAC
        <= width
        <= frame_w * MAX_LANE_W_FRAC
    ):
        return None

    return {
        "left": left_x,
        "right": right_x,
        "center": (left_x + right_x) / 2,
        "width": width
    }


# ============================================================
# LANE CHANGE TRACKER
# ============================================================

class LaneChangeTracker:

    def __init__(self, fps):

        self.fps = fps

        self.max_gap = max(
            1,
            int(fps * MAX_GAP_S)
        )

        self.confirm_frames = max(
            1,
            int(fps * CONFIRM_S)
        )

        self.cooldown_frames = int(
            fps * COOLDOWN_S
        )

        self.last = None

        self.pending = None

        self.cooldown_until = -1

        self.left_changes = 0

        self.right_changes = 0

    @property
    def total_changes(self):

        return (
            self.left_changes
            + self.right_changes
        )

    def update(
        self,
        frame_idx,
        geometry,
        cam_x
    ):

        if (
            self.pending
            and frame_idx - self.pending["frame"]
            > 3 * self.confirm_frames
        ):

            self.pending = None

        if geometry is None:
            return None

        center = geometry["center"]

        width = geometry["width"]

        event = None

        if self.pending:

            p = self.pending

            if (
                abs(center - p["c_before"])
                < REVERT_RATIO * p["w"]
            ):

                self.pending = None

            elif (
                frame_idx - p["frame"]
                >= self.confirm_frames
            ):

                event = self._confirm(p)

                self.pending = None

                self.cooldown_until = (
                    frame_idx
                    + self.cooldown_frames
                )

        elif (
            self.last is not None
            and frame_idx >= self.cooldown_until
        ):

            lf, lc, lw = self.last

            if frame_idx - lf <= self.max_gap:

                delta = (
                    center - lc
                ) / ((lw + width) / 2)

                width_ok = (
                    abs(width - lw) / lw
                    < WIDTH_TOL
                )

                pre_offset = (
                    cam_x - lc
                ) / lw

                if (
                    JUMP_MIN
                    <= abs(delta)
                    <= JUMP_MAX
                    and width_ok
                ):

                    direction = None

                    if (
                        delta > 0
                        and pre_offset > NEAR_EDGE
                    ):
                        direction = "RIGHT"

                    elif (
                        delta < 0
                        and pre_offset < -NEAR_EDGE
                    ):
                        direction = "LEFT"

                    if direction:

                        self.pending = {
                            "frame": frame_idx,
                            "direction": direction,
                            "c_before": lc,
                            "w": lw
                        }

        self.last = (
            frame_idx,
            center,
            width
        )

        return event

    def _confirm(self, p):

        if p["direction"] == "LEFT":

            self.left_changes += 1

        else:

            self.right_changes += 1

        return {
            "event": self.total_changes,
            "frame": p["frame"],
            "video_time_s": round(
                p["frame"] / self.fps,
                2
            ),
            "direction": p["direction"],
            "left_changes": self.left_changes,
            "right_changes": self.right_changes,
            "total_changes": self.total_changes
        }


# ============================================================
# VEHICLE DETECTOR
# ============================================================

class VehicleDetector:

    def __init__(self, model):

        self.model = model

        self.last_results = []

    def detect(self, frame):

        results = self.model.predict(
            source=frame,
            conf=YOLO_CONFIDENCE,
            iou=YOLO_IOU,
            classes=list(VEHICLE_CLASSES.keys()),
            verbose=False
        )

        detections = []

        if not results:
            return detections

        result = results[0]

        if result.boxes is None:
            return detections

        boxes = result.boxes

        for i in range(len(boxes)):

            xyxy = boxes.xyxy[i].cpu().numpy()

            conf = float(
                boxes.conf[i].cpu().numpy()
            )

            cls = int(
                boxes.cls[i].cpu().numpy()
            )

            if cls not in VEHICLE_CLASSES:
                continue

            x1, y1, x2, y2 = map(
                int,
                xyxy
            )

            width = max(
                0,
                x2 - x1
            )

            height = max(
                0,
                y2 - y1
            )

            area = width * height

            if area < MIN_VEHICLE_AREA:
                continue

            detections.append({
                "class_id": cls,
                "label": VEHICLE_CLASSES[cls],
                "confidence": conf,
                "bbox": (
                    x1,
                    y1,
                    x2,
                    y2
                )
            })

        self.last_results = detections

        return detections


# ============================================================
# DRAW VEHICLES
# ============================================================

def draw_vehicles(frame, vehicles):

    for vehicle in vehicles:

        x1, y1, x2, y2 = vehicle["bbox"]

        label = vehicle["label"]

        confidence = vehicle["confidence"]

        text = (
            f"{label} "
            f"{confidence:.2f}"
        )

        # Different color depending on vehicle type.
        if label == "CAR":
            color = BLUE

        elif label == "MOTORCYCLE":
            color = YELLOW

        elif label == "BUS":
            color = ORANGE

        else:
            color = MAGENTA

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            color,
            2
        )

        # Background for label.
        (tw, th), _ = cv2.getTextSize(
            text,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            1
        )

        cv2.rectangle(
            frame,
            (x1, max(0, y1 - th - 8)),
            (x1 + tw + 5, y1),
            color,
            -1
        )

        cv2.putText(
            frame,
            text,
            (x1 + 2, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            WHITE,
            1,
            cv2.LINE_AA
        )


# ============================================================
# VEHICLE STATISTICS
# ============================================================

def vehicle_statistics(vehicles):

    stats = {
        "CAR": 0,
        "MOTORCYCLE": 0,
        "BUS": 0,
        "TRUCK": 0
    }

    for vehicle in vehicles:

        label = vehicle["label"]

        if label in stats:
            stats[label] += 1

    return stats


# ============================================================
# TEXT DRAWING
# ============================================================

def put(
    frame,
    text,
    pos,
    color=WHITE,
    scale=0.6
):

    cv2.putText(
        frame,
        text,
        pos,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        2,
        cv2.LINE_AA
    )


# ============================================================
# DRAW LANE LINES
# ============================================================

def draw_side(
    frame,
    side,
    color
):

    if side is None:
        return

    for (
        x1,
        y1,
        x2,
        y2
    ) in side[1]:

        cv2.line(
            frame,
            (x1, y1),
            (x2, y2),
            color,
            3,
            cv2.LINE_AA
        )


# ============================================================
# OVERLAY
# ============================================================

def draw_overlay(
    frame,
    geometry,
    tracker,
    video_time,
    banner,
    cam_x,
    vehicles
):

    h, w = frame.shape[:2]

    my = int(
        h * MEASURE_Y
    )

    # --------------------------------------------------------
    # Lane position
    # --------------------------------------------------------

    if geometry:

        cx = int(
            geometry["center"]
        )

        cv2.circle(
            frame,
            (cx, my),
            8,
            MAGENTA,
            -1
        )

        cv2.line(
            frame,
            (
                int(cam_x),
                my - 40
            ),
            (
                int(cam_x),
                my + 40
            ),
            RED,
            2
        )

        offset = (
            cam_x
            - geometry["center"]
        ) / geometry["width"]

        if offset > 0:
            side = "RIGHT"

        else:
            side = "LEFT"

        put(
            frame,
            f"LANE POSITION: {offset:+.2f} ({side})",
            (20, 35),
            GREEN
        )

    else:

        put(
            frame,
            "LANE POSITION: SEARCHING...",
            (20, 35),
            ORANGE
        )

    # --------------------------------------------------------
    # Lane change statistics
    # --------------------------------------------------------

    put(
        frame,
        f"LEFT: {tracker.left_changes}   "
        f"RIGHT: {tracker.right_changes}   "
        f"TOTAL: {tracker.total_changes}",
        (20, 70)
    )

    # --------------------------------------------------------
    # Time
    # --------------------------------------------------------

    put(
        frame,
        f"TIME: {video_time:.2f}s",
        (20, 105)
    )

    # --------------------------------------------------------
    # Vehicle statistics
    # --------------------------------------------------------

    stats = vehicle_statistics(
        vehicles
    )

    put(
        frame,
        f"VEHICLES: {len(vehicles)}",
        (20, 140),
        YELLOW
    )

    put(
        frame,
        f"CARS: {stats['CAR']}  "
        f"BIKES: {stats['MOTORCYCLE']}",
        (20, 175),
        WHITE
    )

    put(
        frame,
        f"BUS: {stats['BUS']}  "
        f"TRUCK: {stats['TRUCK']}",
        (20, 210),
        WHITE
    )

    # --------------------------------------------------------
    # Lane-change banner
    # --------------------------------------------------------

    if banner:

        put(
            frame,
            banner,
            (
                max(20, w // 2 - 130),
                35
            ),
            YELLOW,
            0.8
        )


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # Load YOLO
    # --------------------------------------------------------

    print(
        "Loading YOLO model..."
    )

    model = YOLO(
        MODEL_NAME
    )

    vehicle_detector = VehicleDetector(
        model
    )

    # --------------------------------------------------------
    # Open video
    # --------------------------------------------------------

    cap = cv2.VideoCapture(
        VIDEO_PATH
    )

    if not cap.isOpened():

        print(
            "ERROR: Could not open video:",
            VIDEO_PATH
        )

        return

    width = int(
        cap.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )

    height = int(
        cap.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )

    fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    if fps <= 0:
        fps = 30.0

    print(
        f"Video: {width}x{height} "
        f"@ {fps:.2f} FPS"
    )

    # --------------------------------------------------------
    # Video writer
    # --------------------------------------------------------

    writer = cv2.VideoWriter(
        str(OUTPUT_VIDEO),
        cv2.VideoWriter_fourcc(
            *"mp4v"
        ),
        fps,
        (width, height)
    )

    if not writer.isOpened():

        print(
            "ERROR: Could not create output video."
        )
        cap.release()

        return

    # --------------------------------------------------------
    # CSV
    # --------------------------------------------------------

    csv_file = open(
        OUTPUT_CSV,
        "w",
        newline="",
        encoding="utf-8"
    )

    csv_writer = csv.DictWriter(
        csv_file,
        fieldnames=[
            "event",
            "frame",
            "video_time_s",
            "direction",
            "left_changes",
            "right_changes",
            "total_changes"
        ]
    )

    csv_writer.writeheader()

    # --------------------------------------------------------
    # Lane tracker
    # --------------------------------------------------------

    tracker = LaneChangeTracker(
        fps
    )

    cam_x = (
        width
        * CAMERA_X_FRAC
    )

    banner = None

    banner_left = 0

    frame_idx = 0

    vehicles = []

    print(
        "Processing video..."
    )

    try:

        while True:

            ok, frame = cap.read()

            if not ok:
                break

            # =================================================
            # LANE DETECTION
            # =================================================

            left, right = detect_lane(
                frame
            )

            geometry = lane_geometry(
                left,
                right,
                width
            )

            # =================================================
            # LANE CHANGE DETECTION
            # =================================================

            event = tracker.update(
                frame_idx,
                geometry,
                cam_x
            )

            if event:

                csv_writer.writerow(
                    event
                )

                csv_file.flush()

                banner = (
                    "LANE CHANGE: "
                    + event["direction"]
                )

                banner_left = int(
                    fps * BANNER_S
                )

                print(
                    f"[LANE CHANGE] "
                    f"{event['direction']} "
                    f"at "
                    f"{event['video_time_s']}s "
                    f"| Total: "
                    f"{event['total_changes']}"
                )

            # =================================================
            # VEHICLE DETECTION
            # =================================================

            if (
                frame_idx
                % YOLO_INTERVAL
                == 0
            ):

                vehicles = (
                    vehicle_detector.detect(
                        frame
                    )
                )

            # =================================================
            # BANNER TIMER
            # =================================================

            if banner_left > 0:

                banner_left -= 1

            else:

                banner = None

            # =================================================
            # DRAW LANES
            # =================================================

            draw_side(
                frame,
                left,
                GREEN
            )

            draw_side(
                frame,
                right,
                GREEN
            )

            # =================================================
            # DRAW VEHICLES
            # =================================================

            draw_vehicles(
                frame,
                vehicles
            )

            # =================================================
            # DRAW INFORMATION
            # =================================================

            draw_overlay(
                frame,
                geometry,
                tracker,
                frame_idx / fps,
                banner,
                cam_x,
                vehicles
            )

            # =================================================
            # WRITE OUTPUT
            # =================================================

            writer.write(
                frame
            )

            cv2.imshow(
                "Driver Safety Monitor",
                frame
            )

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            frame_idx += 1

    finally:

        cap.release()

        writer.release()

        csv_file.close()

        cv2.destroyAllWindows()

    # ========================================================
    # FINAL RESULTS
    # ========================================================

    print()
    print(
        "=============================="
    )

    print(
        "DRIVER SAFETY MONITOR COMPLETE"
    )

    print(
        "=============================="
    )

    print(
        f"Left lane changes  : "
        f"{tracker.left_changes}"
    )

    print(
        f"Right lane changes : "
        f"{tracker.right_changes}"
    )

    print(
        f"Total lane changes : "
        f"{tracker.total_changes}"
    )

    print(
        "Output video:"
    )

    print(
        OUTPUT_VIDEO.resolve()
    )

    print(
        "CSV:"
    )

    print(
        OUTPUT_CSV.resolve()
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()
