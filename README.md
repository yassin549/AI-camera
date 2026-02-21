This system stores biometric data locally. Do not deploy it in contexts where legal restrictions apply without reviewing local laws and obtaining consent.

# Local RTSP Face Recognition (CPU-Optimized)

CPU-first face recognition pipeline for local RTSP cameras:
- OpenCV + FFmpeg RTSP ingestion
- MediaPipe face detection
- `face_recognition` (dlib) 128-d embeddings
- SQLite identity store with thumbnail snapshots
- Flask endpoint for identity status (`/identities`)

## Privacy and Safety
- Data stays local on disk (`identities.db`, `./faces/`).
- No cloud upload is implemented.
- You are responsible for consent, notice, and legal compliance for biometric processing.

## Project Files
- `main.py`: capture loop, reconnect logic, matching, cache, display, HTTP endpoint
- `detector.py`: MediaPipe detector wrapper
- `embedder.py`: face embedding extraction + embedding blob utilities
- `db.py`: SQLite schema and identity CRUD/matching
- `utils.py`: shared helpers (cosine, timestamps, thumbnail saving)
- `config.yaml`: runtime config
- `tests/test_db.py`: DB persistence and matching unit test
- `run.sh` / `run.bat`: run examples

## Install

### Windows
```powershell
python -m venv venv && venv\Scripts\activate
pip install -r requirements.txt
```

### Linux/macOS
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Windows dlib Tips
- `face-recognition` depends on `dlib`, which can fail to build on Windows without Visual C++ build tools.
- Prefer a prebuilt wheel when available, or install via Conda (`conda install -c conda-forge dlib face_recognition`).
- If pip build fails, install Visual Studio Build Tools and CMake, then retry.
- `face_recognition_models` expects `pkg_resources`; keep `setuptools` below 81 (already pinned in `requirements.txt`).

## Configuration
Edit `config.yaml` or set environment variable `RTSP_URL`.

`config.yaml` defaults:
- `RTSP_URL`: RTSP camera URL placeholder
- `cosine_threshold`: default `0.60`
- `skip_n`: default `2` (process every 3rd frame)
- `imgsz`: `[640, 360]`
- `db_path`: `./identities.db`
- `faces_dir`: `./faces`
- `http_port`: `8080`
- `cache_seconds`: `30`

## Run

### RTSP mode
```powershell
set RTSP_URL=rtsp://username:password@camera-ip:554/stream1
python main.py
```

or:
```bash
./run.sh
```

### Verbose logging
```bash
python main.py --verbose
```

### Demo mode
Uses `demo.mp4` if present, else webcam index `0`.
```bash
python main.py --demo
```
Use a specific video:
```bash
python main.py --demo path/to/video.mp4
```

## HTTP Endpoint
- URL: `http://127.0.0.1:8080/identities`
- Response format:
```json
{
  "identities": [
    {
      "id": 1,
      "first_seen": "2026-02-21T00:00:00+00:00",
      "last_seen": "2026-02-21T00:01:00+00:00",
      "sample_path": "./faces/1_2026-02-21_00-00-00_00-00.jpg",
      "last_score": 0.92,
      "count": 8
    }
  ],
  "count": 1
}
```

## Test
```bash
python -m pytest -q tests/test_db.py
```

## Notes
- RTSP source is opened with FFmpeg backend (`cv2.CAP_FFMPEG`) and TCP hint appended when missing.
- Reconnect backoff on decode/read failures: `0.5s -> 1s -> 2s ...` up to `10s`.
- New identities save thumbnails to `./faces/{id}_{timestamp}.jpg`.
