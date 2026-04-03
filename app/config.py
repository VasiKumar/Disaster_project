from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional at runtime
    load_dotenv = None

if load_dotenv is not None:
    load_dotenv()


@dataclass
class Settings:
    app_name: str = "AI Disaster Management System"
    host: str = os.getenv("HOST", "0.0.0.0")
    port: int = int(os.getenv("PORT", "8000"))
    debug: bool = os.getenv("DEBUG", "false").lower() == "true"

    camera_source: str = os.getenv("CAMERA_SOURCE", "0")
    camera_id: str = os.getenv("CAMERA_ID", "CAM-01")
    location_tag: str = os.getenv("LOCATION_TAG", "Sector-A")

    # If true, runs with lightweight OpenCV heuristics only.
    use_heuristics_only: bool = os.getenv("USE_HEURISTICS_ONLY", "false").lower() == "true"
    yolo_model_path: str = os.getenv("YOLO_MODEL_PATH", "models/disaster_yolo.pt")
    yolo_base_model: str = os.getenv("YOLO_BASE_MODEL", "yolo11n.pt")
    person_model_path: str = os.getenv("PERSON_MODEL_PATH", "yolo11n.pt")
    yolo_conf_threshold: float = float(os.getenv("YOLO_CONF_THRESHOLD", "0.35"))
    yolo_iou_threshold: float = float(os.getenv("YOLO_IOU_THRESHOLD", "0.45"))
    yolo_imgsz: int = int(os.getenv("YOLO_IMGSZ", "960"))
    person_yolo_imgsz: int = int(os.getenv("PERSON_YOLO_IMGSZ", "1280"))
    yolo_device: str = os.getenv("YOLO_DEVICE", "cpu")

    # Detection quality controls.
    use_person_model: bool = os.getenv("USE_PERSON_MODEL", "true").lower() == "true"
    use_disaster_model_for_person: bool = (
        os.getenv("USE_DISASTER_MODEL_FOR_PERSON", "false").lower() == "true"
    )
    use_hog_fallback: bool = os.getenv("USE_HOG_FALLBACK", "false").lower() == "true"
    enable_heuristic_assist: bool = os.getenv("ENABLE_HEURISTIC_ASSIST", "false").lower() == "true"

    min_conf_person: float = float(os.getenv("MIN_CONF_PERSON", "0.50"))
    min_conf_person_far: float = float(os.getenv("MIN_CONF_PERSON_FAR", "0.33"))
    min_conf_fire: float = float(os.getenv("MIN_CONF_FIRE", "0.60"))
    min_conf_smoke: float = float(os.getenv("MIN_CONF_SMOKE", "0.60"))
    min_conf_flood: float = float(os.getenv("MIN_CONF_FLOOD", "0.72"))
    min_conf_generic: float = float(os.getenv("MIN_CONF_GENERIC", "0.45"))
    min_consecutive_frames: int = int(os.getenv("MIN_CONSECUTIVE_FRAMES", "2"))
    survivor_min_consecutive_frames: int = int(os.getenv("SURVIVOR_MIN_CONSECUTIVE_FRAMES", "1"))
    person_far_bbox_area_ratio: float = float(os.getenv("PERSON_FAR_BBOX_AREA_RATIO", "0.0035"))

    # Class-specific false-positive control (no retraining required).
    fire_min_bbox_area_ratio: float = float(os.getenv("FIRE_MIN_BBOX_AREA_RATIO", "0.003"))
    fire_min_color_ratio: float = float(os.getenv("FIRE_MIN_COLOR_RATIO", "0.015"))
    flood_min_bbox_area_ratio: float = float(os.getenv("FLOOD_MIN_BBOX_AREA_RATIO", "0.02"))
    flood_min_bottom_ratio: float = float(os.getenv("FLOOD_MIN_BOTTOM_RATIO", "0.45"))
    flood_min_color_ratio: float = float(os.getenv("FLOOD_MIN_COLOR_RATIO", "0.02"))
    flood_low_sat_weight: float = float(os.getenv("FLOOD_LOW_SAT_WEIGHT", "0.65"))
    flood_max_fire_color_ratio: float = float(os.getenv("FLOOD_MAX_FIRE_COLOR_RATIO", "0.08"))
    flood_high_conf_override: float = float(os.getenv("FLOOD_HIGH_CONF_OVERRIDE", "0.93"))
    water_label_strict: bool = os.getenv("WATER_LABEL_STRICT", "true").lower() == "true"
    allow_generic_water_as_flood: bool = (
        os.getenv("ALLOW_GENERIC_WATER_AS_FLOOD", "false").lower() == "true"
    )

    danger_fire_min_area_ratio: float = float(os.getenv("DANGER_FIRE_MIN_AREA_RATIO", "0.02"))
    danger_person_fire_max_center_distance_ratio: float = float(
        os.getenv("DANGER_PERSON_FIRE_MAX_CENTER_DISTANCE_RATIO", "0.18")
    )

    # Optional unsafe-zone polygons in normalized coordinates [(x, y), ...]
    unsafe_zones: Dict[str, List[Tuple[float, float]]] = field(
        default_factory=lambda: {
            "collapsed_building_zone": [(0.65, 0.55), (0.95, 0.55), (0.95, 0.95), (0.65, 0.95)]
        }
    )

    data_dir: Path = field(default_factory=lambda: Path("data"))
    incident_log_path: Path = field(default_factory=lambda: Path("data") / "incidents.json")
    model_dir: Path = field(default_factory=lambda: Path("models"))
    dataset_dir: Path = field(default_factory=lambda: Path("datasets") / "disaster")

    # Logging backends.
    log_to_file: bool = os.getenv("LOG_TO_FILE", "true").lower() == "true"
    mongodb_uri: str = os.getenv(
        "MONGODB_URI",
        "mongodb+srv://dummy_user:dummy_password@dummy-cluster.mongodb.net/?retryWrites=true&w=majority",
    )
    mongodb_db_name: str = os.getenv("MONGODB_DB_NAME", "disaster_management")
    mongodb_collection_name: str = os.getenv("MONGODB_COLLECTION_NAME", "incident_logs")


settings = Settings()
settings.data_dir.mkdir(parents=True, exist_ok=True)
settings.model_dir.mkdir(parents=True, exist_ok=True)
settings.dataset_dir.mkdir(parents=True, exist_ok=True)
if not settings.incident_log_path.exists():
    settings.incident_log_path.write_text("[]", encoding="utf-8")
