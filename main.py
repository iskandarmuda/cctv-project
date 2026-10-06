import cv2
import time
import sqlite3
from datetime import datetime
from ultralytics import YOLO

# =========================================================
# CONFIGURATION
# =========================================================

STREAM_URL = "https://stream.serangkota.go.id/LiveApp/streams/kaligandu.m3u8"

MODEL_PATH = "yolo26n.pt"

CONFIDENCE = 0.35
IMAGE_SIZE = 640

# Proses YOLO setiap N frame
PROCESS_EVERY = 2

# Nama CCTV
CCTV_NAME = "Kaligandu"

# Database
DB_FILE = "traffic.db"

# Kendaraan yang dihitung
VEHICLE_CLASSES = {
    "car",
    "motorcycle",
    "bus",
    "truck"
}


# =========================================================
# DATABASE
# =========================================================

conn = None


def init_database():

    global conn

    conn = sqlite3.connect(
        DB_FILE,
        timeout=30,
        check_same_thread=False
    )

    cursor = conn.cursor()

    # Performance SQLite
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.execute("PRAGMA synchronous=NORMAL")
    cursor.execute("PRAGMA cache_size=-20000")

    # Table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS traffic_counts (

            id INTEGER PRIMARY KEY AUTOINCREMENT,

            timestamp TEXT NOT NULL,

            cctv TEXT NOT NULL,

            track_id INTEGER NOT NULL,

            vehicle_type TEXT NOT NULL,

            direction TEXT,

            confidence REAL,

            frame_number INTEGER,

            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Index timestamp
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS
        idx_traffic_timestamp
        ON traffic_counts(timestamp)
    """)

    # Index CCTV
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS
        idx_traffic_cctv_timestamp
        ON traffic_counts(cctv, timestamp)
    """)

    # Index vehicle
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS
        idx_traffic_vehicle
        ON traffic_counts(vehicle_type)
    """)

    conn.commit()

    print(
        f"Database siap: {DB_FILE}"
    )


# =========================================================
# SAVE VEHICLE
# =========================================================

def save_vehicle(
    track_id,
    vehicle_type,
    confidence,
    frame_number
):

    global conn

    timestamp = datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    try:

        conn.execute("""
            INSERT INTO traffic_counts (

                timestamp,
                cctv,
                track_id,
                vehicle_type,
                direction,
                confidence,
                frame_number

            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (

            timestamp,

            CCTV_NAME,

            int(track_id),

            vehicle_type,

            "TERDETEKSI",

            float(confidence),

            int(frame_number)

        ))

        conn.commit()

        print(
            f"[DATABASE] "
            f"{timestamp} | "
            f"{CCTV_NAME} | "
            f"{vehicle_type} | "
            f"ID={track_id} | "
            f"TERDETEKSI | "
            f"CONF={confidence:.2f}"
        )

    except Exception as e:

        print(
            f"[DATABASE ERROR] {e}"
        )


# =========================================================
# CLOSE DATABASE
# =========================================================

def close_database():

    global conn

    if conn is not None:

        try:

            conn.commit()

            conn.close()

        except Exception as e:

            print(
                f"[DATABASE CLOSE ERROR] {e}"
            )


# =========================================================
# INITIALIZE DATABASE
# =========================================================

init_database()


# =========================================================
# LOAD YOLO
# =========================================================

print()
print("==============================")
print("LOADING YOLO")
print("==============================")

print(
    f"Model       : {MODEL_PATH}"
)

print(
    f"Confidence  : {CONFIDENCE}"
)

print(
    f"Image Size  : {IMAGE_SIZE}"
)

print(
    f"Process     : Every {PROCESS_EVERY} frame"
)

model = YOLO(
    MODEL_PATH
)

print(
    "YOLO loaded."
)


# =========================================================
# OPEN CCTV
# =========================================================

print()
print("==============================")
print("CONNECTING CCTV")
print("==============================")

print(
    STREAM_URL
)

cap = cv2.VideoCapture(
    STREAM_URL,
    cv2.CAP_FFMPEG
)

cap.set(
    cv2.CAP_PROP_BUFFERSIZE,
    1
)


if not cap.isOpened():

    print(
        "ERROR: CCTV tidak dapat dibuka."
    )

    close_database()

    exit()


print(
    "CCTV connected."
)


# =========================================================
# VARIABLES
# =========================================================

frame_number = 0

fps = 0

fps_start = time.time()

fps_counter = 0

last_result = None


# =========================================================
# VEHICLE TRACKING
# =========================================================

# Semua kendaraan yang pernah dihitung
counted_ids = set()


# =========================================================
# STATISTICS
# =========================================================

total_vehicle = 0

vehicle_count = {

    "car": 0,

    "motorcycle": 0,

    "bus": 0,

    "truck": 0

}


# =========================================================
# MAIN LOOP
# =========================================================

while True:

    ret, frame = cap.read()


    # =====================================================
    # RECONNECT CCTV
    # =====================================================

    if not ret:

        print()
        print(
            "CCTV disconnected."
        )

        print(
            "Reconnecting..."
        )

        cap.release()

        time.sleep(2)

        cap = cv2.VideoCapture(
            STREAM_URL,
            cv2.CAP_FFMPEG
        )

        cap.set(
            cv2.CAP_PROP_BUFFERSIZE,
            1
        )

        continue


    frame_number += 1


    # =====================================================
    # YOLO TRACKING
    # =====================================================

    if frame_number % PROCESS_EVERY == 0:

        try:

            results = model.track(

                frame,

                persist=True,

                tracker="bytetrack.yaml",

                conf=CONFIDENCE,

                imgsz=IMAGE_SIZE,

                verbose=False

            )

            if results:

                last_result = results[0]

        except Exception as e:

            print(
                f"[YOLO ERROR] {e}"
            )

            continue


    # =====================================================
    # DRAW RESULT
    # =====================================================

    if last_result is not None:

        annotated_frame = (
            last_result.plot()
        )

        boxes = last_result.boxes


        if (
            boxes is not None
            and len(boxes) > 0
        ):

            # =================================================
            # TRACK IDS
            # =================================================

            if boxes.id is not None:

                track_ids = (

                    boxes.id
                    .int()
                    .cpu()
                    .tolist()

                )

            else:

                track_ids = []


            # =================================================
            # CLASSES
            # =================================================

            classes = (

                boxes.cls
                .int()
                .cpu()
                .tolist()

            )


            # =================================================
            # CONFIDENCE
            # =================================================

            confidences = (

                boxes.conf
                .cpu()
                .tolist()

            )


            # =================================================
            # PROCESS VEHICLES
            # =================================================

            for i, cls_id in enumerate(classes):

                vehicle_type = (
                    model.names[cls_id]
                )


                # Hanya kendaraan
                if (
                    vehicle_type
                    not in VEHICLE_CLASSES
                ):

                    continue


                # Tidak ada track ID
                if (
                    i >= len(track_ids)
                ):

                    continue


                track_id = (
                    track_ids[i]
                )

                confidence = (
                    confidences[i]
                )


                # =================================================
                # HITUNG ID BARU
                # =================================================

                if track_id not in counted_ids:

                    counted_ids.add(
                        track_id
                    )

                    total_vehicle += 1

                    vehicle_count[
                        vehicle_type
                    ] += 1


                    # Simpan ke database
                    save_vehicle(

                        track_id=track_id,

                        vehicle_type=vehicle_type,

                        confidence=confidence,

                        frame_number=frame_number

                    )


                # =================================================
                # CENTER POINT
                # =================================================

                x1, y1, x2, y2 = (

                    boxes.xyxy[i]
                    .cpu()
                    .tolist()

                )


                center_x = int(
                    (x1 + x2) / 2
                )

                center_y = int(
                    (y1 + y2) / 2
                )


                # =================================================
                # CENTER
                # =================================================

                cv2.circle(

                    annotated_frame,

                    (
                        center_x,
                        center_y
                    ),

                    5,

                    (255, 255, 255),

                    -1

                )


                # =================================================
                # ID
                # =================================================

                cv2.putText(

                    annotated_frame,

                    f"ID {track_id}",

                    (
                        center_x,
                        center_y - 10
                    ),

                    cv2.FONT_HERSHEY_SIMPLEX,

                    0.5,

                    (255, 255, 255),

                    2

                )


    else:

        annotated_frame = (
            frame.copy()
        )


    # =====================================================
    # FPS
    # =====================================================

    fps_counter += 1

    elapsed = (

        time.time()
        - fps_start

    )


    if elapsed >= 1:

        fps = (

            fps_counter
            / elapsed

        )

        fps_counter = 0

        fps_start = time.time()


    # =====================================================
    # DASHBOARD
    # =====================================================

    cv2.rectangle(

        annotated_frame,

        (10, 10),

        (370, 250),

        (0, 0, 0),

        -1

    )


    cv2.putText(

        annotated_frame,

        f"CCTV: {CCTV_NAME}",

        (20, 40),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.65,

        (255, 255, 255),

        2

    )


    cv2.putText(

        annotated_frame,

        f"FPS: {fps:.1f}",

        (20, 70),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.6,

        (255, 255, 255),

        2

    )


    cv2.putText(

        annotated_frame,

        f"TOTAL KENDARAAN: {total_vehicle}",

        (20, 105),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.65,

        (0, 255, 0),

        2

    )


    cv2.putText(

        annotated_frame,

        f"CAR        : {vehicle_count['car']}",

        (20, 140),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.55,

        (255, 255, 255),

        2

    )


    cv2.putText(

        annotated_frame,

        f"MOTORCYCLE : {vehicle_count['motorcycle']}",

        (20, 170),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.55,

        (255, 255, 255),

        2

    )


    cv2.putText(

        annotated_frame,

        f"BUS        : {vehicle_count['bus']}",

        (20, 200),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.55,

        (255, 255, 255),

        2

    )


    cv2.putText(

        annotated_frame,

        f"TRUCK      : {vehicle_count['truck']}",

        (20, 230),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.55,

        (255, 255, 255),

        2

    )


    # =====================================================
    # SHOW
    # =====================================================

    cv2.imshow(

        "CCTV YOLO Vehicle Counter",

        annotated_frame

    )


    # =====================================================
    # EXIT
    # =====================================================

    key = (
        cv2.waitKey(1)
        & 0xFF
    )


    if key == ord("q"):

        break


# =========================================================
# CLEANUP
# =========================================================

print()

print(
    "Stopping CCTV..."
)

cap.release()

cv2.destroyAllWindows()

close_database()


# =========================================================
# FINAL REPORT
# =========================================================

print()

print(
    "=============================="
)

print(
    "FINAL TRAFFIC STATISTICS"
)

print(
    "=============================="
)

print(
    f"Total kendaraan : {total_vehicle}"
)

print()

print(
    f"Car        : "
    f"{vehicle_count['car']}"
)

print(
    f"Motorcycle : "
    f"{vehicle_count['motorcycle']}"
)

print(
    f"Bus        : "
    f"{vehicle_count['bus']}"
)

print(
    f"Truck      : "
    f"{vehicle_count['truck']}"
)

print()

print(
    f"Database : {DB_FILE}"
)