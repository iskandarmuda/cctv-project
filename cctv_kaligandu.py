import cv2
import time
import os
from datetime import datetime
from ultralytics import YOLO

# ============================================================
# CONFIG
# ============================================================

STREAM_URL = "https://stream.serangkota.go.id/LiveApp/streams/kaligandu.m3u8"

MODEL = "yolo26n.pt"

CONFIDENCE = 0.40
IMG_SIZE = 640

# Proses 1 frame setiap N frame
PROCESS_EVERY = 2

# Folder screenshot
SCREENSHOT_DIR = "screenshots"

# Objek yang ingin dihitung
VEHICLE_CLASSES = {
    "car",
    "motorcycle",
    "bus",
    "truck"
}

# ============================================================
# CREATE FOLDER
# ============================================================

os.makedirs(SCREENSHOT_DIR, exist_ok=True)

# ============================================================
# LOAD YOLO
# ============================================================

print("Loading YOLO...")

model = YOLO(MODEL)

print("YOLO loaded")
print("Connecting to CCTV Kaligandu...")

# ============================================================
# OPEN CCTV
# ============================================================

cap = cv2.VideoCapture(STREAM_URL, cv2.CAP_FFMPEG)

if not cap.isOpened():
    print("ERROR: Tidak dapat membuka CCTV")
    print("Periksa koneksi internet / URL stream / FFmpeg")
    exit()

print("CCTV CONNECTED")

# ============================================================
# FPS VARIABLES
# ============================================================

frame_count = 0
fps = 0

fps_start = time.time()
fps_frames = 0

last_result = None

# ============================================================
# MAIN LOOP
# ============================================================

while True:

    ret, frame = cap.read()

    if not ret:
        print("Stream terputus, mencoba reconnect...")

        cap.release()

        time.sleep(2)

        cap = cv2.VideoCapture(
            STREAM_URL,
            cv2.CAP_FFMPEG
        )

        continue

    frame_count += 1
    fps_frames += 1

    # ========================================================
    # FPS CALCULATION
    # ========================================================

    elapsed = time.time() - fps_start

    if elapsed >= 1:

        fps = fps_frames / elapsed

        fps_frames = 0
        fps_start = time.time()

    # ========================================================
    # YOLO PROCESS
    # ========================================================

    if frame_count % PROCESS_EVERY == 0:

        results = model.track(
            frame,
            persist=True,
            imgsz=IMG_SIZE,
            conf=CONFIDENCE,
            tracker="bytetrack.yaml",
            verbose=False
        )

        last_result = results[0]

    # ========================================================
    # DRAW RESULT
    # ========================================================

    if last_result is not None:

        annotated = last_result.plot()

        # ==============================================
        # COUNT OBJECTS
        # ==============================================

        vehicle_count = 0
        person_count = 0

        if last_result.boxes is not None:

            for box in last_result.boxes:

                cls_id = int(box.cls[0])

                class_name = model.names[cls_id]

                if class_name in VEHICLE_CLASSES:

                    vehicle_count += 1

                if class_name == "person":

                    person_count += 1

    else:

        annotated = frame

        vehicle_count = 0
        person_count = 0

    # ========================================================
    # INFORMATION PANEL
    # ========================================================

    cv2.rectangle(
        annotated,
        (10, 10),
        (330, 120),
        (0, 0, 0),
        -1
    )

    cv2.putText(
        annotated,
        f"CCTV KALIGANDU",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.putText(
        annotated,
        f"FPS: {fps:.1f}",
        (20, 62),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    cv2.putText(
        annotated,
        f"Vehicle: {vehicle_count}",
        (20, 88),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    cv2.putText(
        annotated,
        f"Person: {person_count}",
        (20, 112),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    # ========================================================
    # DISPLAY
    # ========================================================

    cv2.imshow(
        "YOLO CCTV Kaligandu",
        annotated
    )

    # ========================================================
    # KEYBOARD
    # ========================================================

    key = cv2.waitKey(1) & 0xFF

    # Q = quit
    if key == ord("q"):

        break

    # S = screenshot
    if key == ord("s"):

        filename = datetime.now().strftime(
            "%Y%m%d_%H%M%S.jpg"
        )

        filepath = os.path.join(
            SCREENSHOT_DIR,
            filename
        )

        cv2.imwrite(
            filepath,
            annotated
        )

        print(
            f"Screenshot saved: {filepath}"
        )

# ============================================================
# CLEANUP
# ============================================================

cap.release()

cv2.destroyAllWindows()

print("Program stopped")