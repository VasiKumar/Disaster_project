from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np

from app.config import settings
from app.schemas import Detection

try:
    from ultralytics import YOLO
except ImportError:  # pragma: no cover - optional dependency at runtime
    YOLO = None


@dataclass
class MotionState:
    prev_gray: np.ndarray | None = None


class MultiDisasterDetector:
    def __init__(self) -> None:
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        self.motion_state = MotionState()
        self.last_people_count = 0
        self.yolo_model = self._load_yolo_model()
        self.person_model = self._load_person_model()
        self.type_streaks: Dict[str, int] = {}

    def detect(self, frame: np.ndarray) -> List[Detection]:
        detections: List[Detection] = []

        yolo_events, yolo_people = self._detect_with_yolo(frame)
        detections.extend(yolo_events)

        person_events, person_boxes = self._detect_people_with_person_model(frame)
        detections.extend(person_events)

        people_boxes = person_boxes or yolo_people

        if not people_boxes and settings.use_hog_fallback:
            people_boxes = self._detect_people(frame)

        if people_boxes and not person_events:
            detections.extend(self._build_survivor_events_from_boxes(people_boxes))

        self.last_people_count = len(people_boxes)
        detections.extend(self._person_based_events(frame, people_boxes))

        # Heuristics are useful in pure-heuristic mode, but can create noise when YOLO is active.
        if settings.use_heuristics_only or settings.enable_heuristic_assist:
            detections.extend(self._detect_fire_and_smoke(frame))
            detections.extend(self._detect_earthquake(frame))
            detections.extend(self._detect_crowd_panic(frame, people_boxes))
            detections.extend(self._detect_road_accident(frame))

        detections.extend(self._detect_fire_proximity_danger(frame, detections, people_boxes))

        detections.extend(self._detect_unsafe_zone(frame, people_boxes))

        return self._apply_temporal_filter(detections)

    def _load_yolo_model(self):
        if settings.use_heuristics_only or YOLO is None:
            return None

        preferred_model_path = self._resolve_preferred_yolo_weights_path()

        try:
            return YOLO(preferred_model_path)
        except Exception:
            try:
                # Fallback to base model if custom weights are not trained yet.
                return YOLO(settings.yolo_base_model)
            except Exception:
                return None

    @staticmethod
    def _resolve_preferred_yolo_weights_path() -> str:
        configured = Path(settings.yolo_model_path)
        best_metrics_model = MultiDisasterDetector._find_best_trained_weights()

        if best_metrics_model is not None:
            return str(best_metrics_model)
        if configured.exists():
            return str(configured)
        return settings.yolo_model_path

    @staticmethod
    def _find_best_trained_weights() -> Path | None:
        runs_root = Path("runs") / "detect" / "runs" / "disaster"
        if not runs_root.exists():
            return None

        best_path: Path | None = None
        best_score = -1.0

        for results_csv in runs_root.glob("*/results.csv"):
            run_dir = results_csv.parent
            candidate = run_dir / "weights" / "best.pt"
            if not candidate.exists():
                continue

            score = MultiDisasterDetector._best_map5095_from_csv(results_csv)
            if score > best_score:
                best_score = score
                best_path = candidate

        return best_path

    @staticmethod
    def _best_map5095_from_csv(results_csv: Path) -> float:
        try:
            with results_csv.open("r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                best = -1.0
                for row in reader:
                    value = row.get("metrics/mAP50-95(B)")
                    if value is None:
                        continue
                    try:
                        best = max(best, float(value))
                    except ValueError:
                        continue
                return best
        except Exception:
            return -1.0

    def _load_person_model(self):
        if settings.use_heuristics_only or YOLO is None or not settings.use_person_model:
            return None

        try:
            return YOLO(settings.person_model_path)
        except Exception:
            return None

    def _detect_with_yolo(
        self, frame: np.ndarray
    ) -> Tuple[List[Detection], List[Tuple[int, int, int, int]]]:
        if self.yolo_model is None:
            return [], []

        detections: List[Detection] = []
        people_boxes: List[Tuple[int, int, int, int]] = []

        results = self.yolo_model.predict(
            frame,
            conf=settings.yolo_conf_threshold,
            iou=settings.yolo_iou_threshold,
            imgsz=settings.yolo_imgsz,
            device=settings.yolo_device,
            verbose=False,
        )
        if not results:
            return detections, people_boxes

        result = results[0]
        names = result.names or {}
        boxes = result.boxes
        if boxes is None:
            return detections, people_boxes

        for box in boxes:
            cls_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            label = str(names.get(cls_id, cls_id)).lower()
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
            bbox = (x1, y1, max(1, x2 - x1), max(1, y2 - y1))

            if label == "person" and settings.use_disaster_model_for_person and conf >= settings.min_conf_person:
                people_boxes.append(bbox)

            mapped = self._map_yolo_label(label)
            if mapped is None:
                continue

            if mapped == "survivor" and not settings.use_disaster_model_for_person:
                continue
            if not self._passes_confidence_threshold(mapped, conf):
                continue
            if not self._passes_semantic_validation(frame, mapped, label, bbox, conf):
                continue

            detections.append(
                Detection(
                    disaster_type=mapped,
                    confidence=conf,
                    bbox=bbox,
                    message=f"YOLO detected {mapped} ({label})",
                    metadata={"yolo_label": label, "source": "yolo"},
                )
            )

        return detections, people_boxes

    @staticmethod
    def _build_survivor_events_from_boxes(
        people_boxes: List[Tuple[int, int, int, int]]
    ) -> List[Detection]:
        detections: List[Detection] = []
        for (x, y, w, h) in people_boxes:
            detections.append(
                Detection(
                    disaster_type="survivor",
                    confidence=max(settings.min_conf_person, 0.6),
                    bbox=(x, y, w, h),
                    message="Person detected from fallback person pipeline",
                    metadata={"source": "fallback_person"},
                )
            )
        return detections

    def _detect_people_with_person_model(
        self, frame: np.ndarray
    ) -> Tuple[List[Detection], List[Tuple[int, int, int, int]]]:
        if self.person_model is None:
            return [], []

        detections: List[Detection] = []
        people_boxes: List[Tuple[int, int, int, int]] = []
        results = self.person_model.predict(
            frame,
            conf=max(settings.yolo_conf_threshold, settings.min_conf_person),
            iou=settings.yolo_iou_threshold,
            imgsz=settings.yolo_imgsz,
            device=settings.yolo_device,
            classes=[0],  # COCO person class
            verbose=False,
        )
        if not results:
            return detections, people_boxes

        boxes = results[0].boxes
        if boxes is None:
            return detections, people_boxes

        for box in boxes:
            conf = float(box.conf[0].item())
            if conf < settings.min_conf_person:
                continue
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
            bbox = (x1, y1, max(1, x2 - x1), max(1, y2 - y1))
            people_boxes.append(bbox)
            detections.append(
                Detection(
                    disaster_type="survivor",
                    confidence=conf,
                    bbox=bbox,
                    message="Person detected by dedicated person model",
                    metadata={"source": "person_model"},
                )
            )

        return detections, people_boxes

    @staticmethod
    def _passes_confidence_threshold(disaster_type: str, conf: float) -> bool:
        if disaster_type == "survivor":
            return conf >= settings.min_conf_person
        if disaster_type == "fire":
            return conf >= settings.min_conf_fire
        if disaster_type == "smoke":
            return conf >= settings.min_conf_smoke
        if disaster_type == "flood":
            return conf >= settings.min_conf_flood
        return conf >= settings.min_conf_generic

    @staticmethod
    def _passes_semantic_validation(
        frame: np.ndarray,
        disaster_type: str,
        original_label: str,
        bbox: Tuple[int, int, int, int],
        conf: float,
    ) -> bool:
        h, w = frame.shape[:2]
        x, y, bw, bh = bbox
        x2 = min(w, x + bw)
        y2 = min(h, y + bh)
        area_ratio = float((max(1, x2 - x) * max(1, y2 - y)) / max(1, w * h))

        if disaster_type == "fire":
            if area_ratio < settings.fire_min_bbox_area_ratio:
                return False

            roi = frame[max(0, y):y2, max(0, x):x2]
            if roi.size == 0:
                return False

            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            fire_mask = cv2.inRange(hsv, np.array([0, 120, 120]), np.array([35, 255, 255]))
            fire_ratio = float(np.count_nonzero(fire_mask)) / float(fire_mask.size)

            # Keep only high-color-consistency fire boxes, unless confidence is very high.
            return fire_ratio >= settings.fire_min_color_ratio or conf >= 0.90

        if disaster_type == "flood":
            if area_ratio < settings.flood_min_bbox_area_ratio:
                return False

            bottom_ratio = float((y + bh) / max(1, h))
            if bottom_ratio < settings.flood_min_bottom_ratio:
                return False

            if settings.water_label_strict:
                has_flood_word = any(token in original_label for token in ("flood", "inundation", "waterlog"))
                if ("water" in original_label) and (not has_flood_word) and conf < 0.88:
                    return False

        return True

    def _apply_temporal_filter(self, detections: List[Detection]) -> List[Detection]:
        observed_types = {d.disaster_type for d in detections}
        filtered: List[Detection] = []

        for detection in detections:
            dtype = detection.disaster_type
            self.type_streaks[dtype] = self.type_streaks.get(dtype, 0) + 1

            if dtype in {"unsafe_zone", "road_accident"}:
                filtered.append(detection)
                continue

            if self.type_streaks[dtype] >= max(settings.min_consecutive_frames, 1):
                filtered.append(detection)

        for dtype in list(self.type_streaks.keys()):
            if dtype not in observed_types:
                self.type_streaks[dtype] = 0

        return filtered

    @staticmethod
    def _map_yolo_label(label: str):
        if "person" in label:
            return "survivor"
        if "fire" in label or "flame" in label:
            return "fire"
        if "smoke" in label or "fume" in label:
            return "smoke"
        if "flood" in label or "water" in label or "inundation" in label:
            return "flood"
        if "accident" in label or "collision" in label or "crash" in label:
            return "road_accident"
        if "crowd" in label or "stampede" in label or "panic" in label:
            return "crowd_panic"
        if "fallen" in label or "lying" in label or "injured" in label:
            return "fallen_person"
        if "earthquake" in label or "debris" in label or "collapse" in label or "rubble" in label:
            return "earthquake"
        return None

    def _detect_people(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        boxes, _ = self.hog.detectMultiScale(
            frame,
            winStride=(8, 8),
            padding=(8, 8),
            scale=1.03,
        )
        return [tuple(map(int, box)) for box in boxes]

    def _person_based_events(
        self, frame: np.ndarray, people_boxes: List[Tuple[int, int, int, int]]
    ) -> List[Detection]:
        detections: List[Detection] = []

        for (x, y, w, h) in people_boxes:
            # Horizontal box is used as a lightweight fallen-person cue.
            aspect_ratio = w / max(h, 1)
            if aspect_ratio > 1.2:
                detections.append(
                    Detection(
                        disaster_type="fallen_person",
                        confidence=min(0.55 + (aspect_ratio - 1.2) * 0.2, 0.95),
                        bbox=(x, y, w, h),
                        message="Possible fallen person detected",
                        metadata={"aspect_ratio": round(aspect_ratio, 2)},
                    )
                )

        return detections

    def _detect_fire_and_smoke(self, frame: np.ndarray) -> List[Detection]:
        detections: List[Detection] = []
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Fire-like color regions
        lower_fire = np.array([0, 120, 120])
        upper_fire = np.array([35, 255, 255])
        fire_mask = cv2.inRange(hsv, lower_fire, upper_fire)
        fire_ratio = float(np.count_nonzero(fire_mask)) / fire_mask.size

        if fire_ratio > 0.04:
            detections.append(
                Detection(
                    disaster_type="fire",
                    confidence=min(0.6 + fire_ratio, 0.98),
                    message="Fire-like region detected",
                    metadata={"fire_ratio": round(fire_ratio, 4)},
                )
            )

        # Smoke-like regions: low saturation and medium-high value
        lower_smoke = np.array([0, 0, 80])
        upper_smoke = np.array([180, 70, 220])
        smoke_mask = cv2.inRange(hsv, lower_smoke, upper_smoke)
        smoke_ratio = float(np.count_nonzero(smoke_mask)) / smoke_mask.size

        if smoke_ratio > 0.20:
            detections.append(
                Detection(
                    disaster_type="smoke",
                    confidence=min(0.45 + smoke_ratio * 0.8, 0.92),
                    message="Smoke-like pattern detected",
                    metadata={"smoke_ratio": round(smoke_ratio, 4)},
                )
            )

        return detections

    def _detect_earthquake(self, frame: np.ndarray) -> List[Detection]:
        detections: List[Detection] = []
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        shake_score = 0.0
        if self.motion_state.prev_gray is not None:
            flow = cv2.calcOpticalFlowFarneback(
                self.motion_state.prev_gray,
                gray,
                None,
                pyr_scale=0.5,
                levels=3,
                winsize=15,
                iterations=3,
                poly_n=5,
                poly_sigma=1.2,
                flags=0,
            )
            mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            shake_score = float(np.percentile(mag, 85))

        if shake_score > 4.0:
            detections.append(
                Detection(
                    disaster_type="earthquake",
                    confidence=min(0.45 + shake_score / 10.0, 0.93),
                    message="Earthquake-like camera shake detected",
                    metadata={"shake_score": round(shake_score, 3), "source": "motion"},
                )
            )

        return detections

    def _detect_crowd_panic(
        self, frame: np.ndarray, people_boxes: List[Tuple[int, int, int, int]]
    ) -> List[Detection]:
        detections: List[Detection] = []
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        motion_score = 0.0
        if self.motion_state.prev_gray is not None:
            flow = cv2.calcOpticalFlowFarneback(
                self.motion_state.prev_gray,
                gray,
                None,
                pyr_scale=0.5,
                levels=3,
                winsize=15,
                iterations=3,
                poly_n=5,
                poly_sigma=1.2,
                flags=0,
            )
            mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            motion_score = float(np.mean(mag))

        self.motion_state.prev_gray = gray

        crowd_density = len(people_boxes)
        if crowd_density >= 4 and motion_score > 2.0:
            confidence = min(0.5 + (crowd_density / 20.0) + (motion_score / 20.0), 0.96)
            detections.append(
                Detection(
                    disaster_type="crowd_panic",
                    confidence=confidence,
                    message="High crowd motion anomaly detected",
                    metadata={
                        "crowd_density": crowd_density,
                        "motion_score": round(motion_score, 3),
                    },
                )
            )

        return detections

    def _detect_road_accident(self, frame: np.ndarray) -> List[Detection]:
        # Baseline proxy: abrupt edge clutter + high motion suggests crash-like scene disturbance.
        detections: List[Detection] = []
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        edge_ratio = float(np.count_nonzero(edges)) / edges.size

        if edge_ratio > 0.18:
            detections.append(
                Detection(
                    disaster_type="road_accident",
                    confidence=min(0.35 + edge_ratio, 0.8),
                    message="Possible collision scene disturbance detected",
                    metadata={"edge_ratio": round(edge_ratio, 4)},
                )
            )

        return detections

    def _detect_unsafe_zone(
        self, frame: np.ndarray, people_boxes: List[Tuple[int, int, int, int]]
    ) -> List[Detection]:
        detections: List[Detection] = []
        height, width = frame.shape[:2]

        zones_px: Dict[str, List[Tuple[int, int]]] = {}
        for zone_name, points in settings.unsafe_zones.items():
            zones_px[zone_name] = [(int(x * width), int(y * height)) for x, y in points]

        for (x, y, w, h) in people_boxes:
            center = (x + w // 2, y + h // 2)
            for zone_name, polygon in zones_px.items():
                inside = cv2.pointPolygonTest(np.array(polygon, dtype=np.int32), center, False) >= 0
                if inside:
                    detections.append(
                        Detection(
                            disaster_type="unsafe_zone",
                            confidence=0.9,
                            bbox=(x, y, w, h),
                            message=f"Person entered unsafe zone: {zone_name}",
                            metadata={"zone": zone_name},
                        )
                    )

        return detections

    def _detect_fire_proximity_danger(
        self,
        frame: np.ndarray,
        detections: List[Detection],
        people_boxes: List[Tuple[int, int, int, int]],
    ) -> List[Detection]:
        flagged: List[Detection] = []
        if not people_boxes:
            return flagged

        h, w = frame.shape[:2]
        frame_area = float(max(h * w, 1))

        fire_boxes: List[Tuple[int, int, int, int]] = []
        for d in detections:
            if d.disaster_type != "fire" or d.bbox is None:
                continue
            if d.confidence < settings.min_conf_fire:
                continue
            bx, by, bw, bh = d.bbox
            area_ratio = (bw * bh) / frame_area
            if area_ratio >= settings.danger_fire_min_area_ratio:
                fire_boxes.append(d.bbox)

        if not fire_boxes:
            return flagged

        diag = float(np.hypot(w, h))
        for pbox in people_boxes:
            px, py, pw, ph = pbox
            pcx, pcy = px + pw / 2.0, py + ph / 2.0
            for fbox in fire_boxes:
                fx, fy, fw, fh = fbox
                fcx, fcy = fx + fw / 2.0, fy + fh / 2.0
                dist_ratio = float(np.hypot(pcx - fcx, pcy - fcy) / max(diag, 1.0))
                if dist_ratio <= settings.danger_person_fire_max_center_distance_ratio:
                    flagged.append(
                        Detection(
                            disaster_type="unsafe_zone",
                            confidence=0.9,
                            bbox=pbox,
                            message="Danger zone: person too close to large fire",
                            metadata={
                                "source": "fire_proximity",
                                "distance_ratio": round(dist_ratio, 3),
                            },
                        )
                    )
                    break

        return flagged
