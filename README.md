# AI Disaster Management System (YOLO Full Setup)

End-to-end disaster surveillance system with internet dataset download, YOLO training, live camera inference, people counting, incident severity logic, and a real-time control dashboard.

## Disaster Scenarios Covered

- Fire detection
- Smoke detection
- Earthquake/collapse signal detection
- Road accident detection
- Crowd panic/stampede risk detection
- Fallen person detection
- Unsafe zone breach detection
- Survivor/people detection and live people count

## Stack

- Python + FastAPI
- OpenCV video ingestion/overlay
- Ultralytics YOLO for train + inference
- Jinja + JS dashboard UI

## Full Pipeline

1. Download datasets from internet
- Script: `scripts/download_datasets.py`

2. Build unified YOLO dataset
- Script: `scripts/build_yolo_dataset.py`
- Config: `configs/dataset_sources.yaml`

3. Train YOLO model
- Script: `scripts/train_yolo.py`
- Exports trained weights to `models/disaster_yolo.pt`

4. Run live monitoring app
- API + dashboard reads live camera/video stream
- Triggers incidents and alerts
- Tracks and shows live people count

## One-Command Setup (Windows cmd)

```bat
scripts\full_setup.cmd
```

This command does:

- Uses your existing Python environment
- Checks package versions first and installs only missing/incompatible dependencies
- Downloads internet datasets
- Builds merged YOLO dataset
- Trains YOLO model
- Starts the web app

## Manual Setup

1. Create venv

```bat
python -m venv .venv
.venv\Scripts\activate
```

2. Install deps

```bat
pip install -r requirements.txt
```

3. Create env file

```bat
copy .env.example .env
```

4. Download internet datasets

```bat
python scripts\download_datasets.py --out datasets\raw --profile coverage --max-total-gb 3.0 --max-dataset-gb 1.2 --max-files-per-dataset 10000
```

To download only flood dataset from your shared Kaggle source:

```bat
python scripts\download_datasets.py --out datasets\raw --profile coverage --only flood --max-total-gb 3.0 --max-dataset-gb 1.2
```

This targets all calamity classes with many fallback dataset alternatives while keeping total downloads under about 3 GB.
The downloader also blocks known huge datasets (for example full COCO mirrors) to avoid accidental 10-25 GB downloads.

5. Update dataset mapping file

- Edit `configs/dataset_sources.yaml`
- Set correct `image_dir`, `label_dir`, and `class_map` per downloaded source

6. Merge to YOLO format

```bat
python scripts\build_yolo_dataset.py --config configs\dataset_sources.yaml --out datasets\disaster
```

7. Train YOLO

```bat
python scripts\train_yolo.py --data datasets\disaster\data.yaml --epochs 80 --imgsz 960 --batch 8 --device 0 --workers 4 --cache disk
```

The runtime now also marks `unsafe_zone` when a person is too close to a large fire area.

8. Run monitoring app

```bat
python main.py
```

9. Open dashboard

- http://localhost:8000/

## Runtime Configuration

In `.env`:

- `CAMERA_SOURCE`: camera index (`0`) or stream/video path
- `YOLO_MODEL_PATH`: trained model path (default `models/disaster_yolo.pt`)
- `YOLO_BASE_MODEL`: fallback model if trained model is missing
- `PERSON_MODEL_PATH`: dedicated person detector model path (default `yolo11n.pt`)
- `YOLO_CONF_THRESHOLD`: confidence threshold
- `YOLO_IOU_THRESHOLD`: NMS IOU threshold
- `YOLO_IMGSZ`: inference resolution
- `YOLO_DEVICE`: `cpu` or `0` for first GPU
- `USE_HEURISTICS_ONLY`: set `false` to use YOLO
- `USE_PERSON_MODEL`: use dedicated person model for survivor detection
- `USE_DISASTER_MODEL_FOR_PERSON`: keep `false` to avoid person confusion from disaster model
- `USE_HOG_FALLBACK`: keep `false` unless you need CPU-only fallback
- `ENABLE_HEURISTIC_ASSIST`: set `true` only if YOLO misses too much
- `MIN_CONF_PERSON`, `MIN_CONF_FIRE`, `MIN_CONF_SMOKE`: stricter thresholds reduce false alarms
- `MIN_CONSECUTIVE_FRAMES`: require repeated detections before alerting

## API Endpoints

- `POST /api/start`: start camera processing
- `POST /api/stop`: stop processing
- `GET /api/status`: runtime status (includes `people_in_frame`)
- `GET /api/people_count`: live people count only
- `GET /api/dashboard`: status + incidents + analytics snapshot
- `GET /api/video_feed`: MJPEG live stream
- `GET /health`: health check

## Important Dataset Notes

- Internet dataset paths and class IDs are not identical across sources.
- `configs/dataset_sources.yaml` is the control file for remapping every dataset class to your unified YOLO classes.
- After each dataset download, verify directory layout and update the config before training.

## Tests

```bat
pytest -q
```
