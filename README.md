# CCTV YOLO Vehicle Detection

Sistem deteksi kendaraan secara realtime menggunakan CCTV HLS, YOLO, ByteTrack, OpenCV, dan SQLite.

## Features

* Realtime CCTV HLS stream
* YOLO object detection
* ByteTrack vehicle tracking
* Deteksi mobil
* Deteksi motor
* Deteksi bus
* Deteksi truck
* Perhitungan FPS
* Penyimpanan hasil deteksi ke SQLite
* Dashboard informasi realtime

## Requirements

* Windows 10/11
* Python 3.12.7
* Git
* OpenCV
* Ultralytics YOLO
* PyTorch
* SQLite

## Installation

Clone repository:

```bash
git clone https://github.com/iskandarmuda/cctv-project.git
cd cctv-project
```

Buat virtual environment:

```bash
python -m venv venv
```

Aktifkan virtual environment:

```cmd
venv\Scripts\activate
```

Upgrade pip:

```cmd
python -m pip install --upgrade pip
```

Install dependencies:

```cmd
pip install -r requirements.txt
```

## Run

Jalankan:

```cmd
python main.py
```

Program akan mengambil video dari CCTV HLS dan melakukan deteksi kendaraan menggunakan YOLO.

## Database

Database menggunakan SQLite:

```text
traffic.db
```

Data kendaraan yang terdeteksi disimpan ke database.

Untuk membuka database menggunakan SQLite CLI:

```cmd
sqlite3 traffic.db
```

Kemudian:

```sql
.tables
```

Melihat data:

```sql
SELECT * FROM traffic_counts
ORDER BY id DESC
LIMIT 20;
```

## Vehicle Classes

Sistem mendeteksi:

```text
car
motorcycle
bus
truck
```

## Project Architecture

```text
CCTV HLS
    │
    ▼
OpenCV / FFmpeg
    │
    ▼
YOLO Detection
    │
    ▼
ByteTrack
    │
    ▼
Vehicle Tracking
    │
    ├── Car
    ├── Motorcycle
    ├── Bus
    └── Truck
    │
    ▼
SQLite
    │
    ▼
Traffic Statistics
```

## FPS

FPS menunjukkan jumlah frame yang diproses setiap detik.

Contoh:

```text
FPS: 15
```

berarti sistem memproses sekitar 15 frame per detik.

## License

Project ini digunakan untuk pembelajaran dan pengembangan sistem computer vision.
