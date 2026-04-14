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
    person_zoom_pass_conf: float = float(os.getenv("PERSON_ZOOM_PASS_CONF", "0.22"))
    min_conf_fire: float = float(os.getenv("MIN_CONF_FIRE", "0.60"))
    min_conf_smoke: float = float(os.getenv("MIN_CONF_SMOKE", "0.60"))
    min_conf_flood: float = float(os.getenv("MIN_CONF_FLOOD", "0.72"))
    min_conf_generic: float = float(os.getenv("MIN_CONF_GENERIC", "0.45"))
    min_consecutive_frames: int = int(os.getenv("MIN_CONSECUTIVE_FRAMES", "2"))
    survivor_min_consecutive_frames: int = int(os.getenv("SURVIVOR_MIN_CONSECUTIVE_FRAMES", "1"))
    person_far_bbox_area_ratio: float = float(os.getenv("PERSON_FAR_BBOX_AREA_RATIO", "0.0035"))
    person_enable_zoom_pass: bool = os.getenv("PERSON_ENABLE_ZOOM_PASS", "true").lower() == "true"
    person_zoom_pass_scale: float = float(os.getenv("PERSON_ZOOM_PASS_SCALE", "1.6"))
    person_zoom_pass_trigger_count: int = int(os.getenv("PERSON_ZOOM_PASS_TRIGGER_COUNT", "1"))
    person_zoom_pass_interval: int = int(os.getenv("PERSON_ZOOM_PASS_INTERVAL", "3"))
    person_dedupe_iou: float = float(os.getenv("PERSON_DEDUPE_IOU", "0.45"))
    person_predict_augment: bool = os.getenv("PERSON_PREDICT_AUGMENT", "true").lower() == "true"
    person_max_det: int = int(os.getenv("PERSON_MAX_DET", "300"))

    # Class-specific false-positive control (no retraining required).
    fire_min_bbox_area_ratio: float = float(os.getenv("FIRE_MIN_BBOX_AREA_RATIO", "0.003"))
    fire_min_color_ratio: float = float(os.getenv("FIRE_MIN_COLOR_RATIO", "0.015"))
    flood_min_bbox_area_ratio: float = float(os.getenv("FLOOD_MIN_BBOX_AREA_RATIO", "0.02"))
    flood_min_top_ratio: float = float(os.getenv("FLOOD_MIN_TOP_RATIO", "0.18"))
    flood_min_bottom_ratio: float = float(os.getenv("FLOOD_MIN_BOTTOM_RATIO", "0.45"))
    flood_min_width_ratio: float = float(os.getenv("FLOOD_MIN_WIDTH_RATIO", "0.22"))
    flood_max_height_ratio: float = float(os.getenv("FLOOD_MAX_HEIGHT_RATIO", "0.60"))
    flood_min_aspect_ratio: float = float(os.getenv("FLOOD_MIN_ASPECT_RATIO", "1.10"))
    flood_min_color_ratio: float = float(os.getenv("FLOOD_MIN_COLOR_RATIO", "0.02"))
    flood_min_blue_ratio: float = float(os.getenv("FLOOD_MIN_BLUE_RATIO", "0.012"))
    flood_min_muddy_ratio: float = float(os.getenv("FLOOD_MIN_MUDDY_RATIO", "0.010"))
    flood_min_low_sat_ratio: float = float(os.getenv("FLOOD_MIN_LOW_SAT_RATIO", "0.12"))
    flood_max_pink_ratio: float = float(os.getenv("FLOOD_MAX_PINK_RATIO", "0.10"))
    flood_require_water_signature: bool = (
        os.getenv("FLOOD_REQUIRE_WATER_SIGNATURE", "true").lower() == "true"
    )
    flood_low_sat_weight: float = float(os.getenv("FLOOD_LOW_SAT_WEIGHT", "0.65"))
    flood_max_fire_color_ratio: float = float(os.getenv("FLOOD_MAX_FIRE_COLOR_RATIO", "0.08"))
    flood_high_conf_override: float = float(os.getenv("FLOOD_HIGH_CONF_OVERRIDE", "0.93"))
    flood_min_motion_ratio: float = float(os.getenv("FLOOD_MIN_MOTION_RATIO", "0.010"))
    flood_static_water_signature: float = float(os.getenv("FLOOD_STATIC_WATER_SIGNATURE", "0.055"))
    flood_motion_roi_start_ratio: float = float(os.getenv("FLOOD_MOTION_ROI_START_RATIO", "0.35"))
    flood_motion_diff_threshold: int = int(os.getenv("FLOOD_MOTION_DIFF_THRESHOLD", "14"))
    enable_flood_heuristic_assist: bool = (
        os.getenv("ENABLE_FLOOD_HEURISTIC_ASSIST", "true").lower() == "true"
    )
    flood_heuristic_min_bottom_ratio: float = float(os.getenv("FLOOD_HEURISTIC_MIN_BOTTOM_RATIO", "0.06"))
    flood_heuristic_min_color_score: float = float(os.getenv("FLOOD_HEURISTIC_MIN_COLOR_SCORE", "0.055"))
    flood_heuristic_high_color_score: float = float(os.getenv("FLOOD_HEURISTIC_HIGH_COLOR_SCORE", "0.095"))
    flood_heuristic_min_motion_ratio: float = float(os.getenv("FLOOD_HEURISTIC_MIN_MOTION_RATIO", "0.010"))
    flood_fusion_on_threshold: float = float(os.getenv("FLOOD_FUSION_ON_THRESHOLD", "0.58"))
    flood_fusion_off_threshold: float = float(os.getenv("FLOOD_FUSION_OFF_THRESHOLD", "0.40"))
    flood_scene_gate_on_threshold: float = float(os.getenv("FLOOD_SCENE_GATE_ON_THRESHOLD", "0.28"))
    flood_scene_gate_off_threshold: float = float(os.getenv("FLOOD_SCENE_GATE_OFF_THRESHOLD", "0.18"))
    water_label_strict: bool = os.getenv("WATER_LABEL_STRICT", "true").lower() == "true"
    allow_generic_water_as_flood: bool = (
        os.getenv("ALLOW_GENERIC_WATER_AS_FLOOD", "false").lower() == "true"
    )

    # Texture & spatial coherence checks — reject indoor objects (clothing, walls)
    # that happen to be blue/brown but have high edge density or narrow vertical shape.
    flood_max_water_edge_density: float = float(os.getenv("FLOOD_MAX_WATER_EDGE_DENSITY", "0.12"))
    flood_min_horizontal_continuity: float = float(os.getenv("FLOOD_MIN_HORIZONTAL_CONTINUITY", "0.15"))

    incident_missing_grace_frames: int = int(os.getenv("INCIDENT_MISSING_GRACE_FRAMES", "4"))

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
