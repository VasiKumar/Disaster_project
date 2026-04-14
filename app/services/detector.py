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
    prev_flood_gray: np.ndarray | None = None


class MultiDisasterDetector:
    def __init__(self) -> None:
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        self.motion_state = MotionState()
        self.last_people_count = 0
        self.yolo_model = self._load_yolo_model()
        self.person_model = self._load_person_model()
        self.type_streaks: Dict[str, int] = {}
        self.frame_index = 0
        self.last_flood_motion_ratio = 0.0
        self.flood_hysteresis_active = False
        self.yolo_flood_candidates: List[Tuple[float, Tuple[int, int, int, int], str]] = []

    def detect(self, frame: np.ndarray) -> List[Detection]:
        self.frame_index += 1
        self.last_flood_motion_ratio = self._estimate_flood_motion(frame)
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

        detections.extend(self._detect_flood_fusion(frame, self.last_flood_motion_ratio))

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
        self.yolo_flood_candidates = []
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

            if (
                ("flood" in label)
                or ("inundation" in label)
                or ("waterlog" in label)
                or (settings.allow_generic_water_as_flood and ("water" in label))
            ):
                self.yolo_flood_candidates.append((conf, bbox, label))

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
        base_request_conf = max(
            0.05,
            min(
                settings.yolo_conf_threshold,
                settings.min_conf_person,
                settings.min_conf_person_far,
            ),
        )

        candidates = self._predict_person_candidates(
            frame,
            request_conf=base_request_conf,
            resize_scale=1.0,
            source="person_model",
        )

        zoom_interval = max(settings.person_zoom_pass_interval, 1)
        should_run_zoom_pass = (
            settings.person_enable_zoom_pass
            and len(candidates) <= settings.person_zoom_pass_trigger_count
            and (self.frame_index % zoom_interval == 0)
        )

        if should_run_zoom_pass:
            zoom_request_conf = max(0.05, min(base_request_conf, settings.person_zoom_pass_conf))
            candidates.extend(
                self._predict_person_candidates(
                    frame,
                    request_conf=zoom_request_conf,
                    resize_scale=max(settings.person_zoom_pass_scale, 1.0),
                    source="person_model_zoom",
                )
            )

        if not candidates:
            return detections, people_boxes

        merged_candidates = self._merge_person_candidates(candidates)

        frame_h, frame_w = frame.shape[:2]
        frame_area = float(max(frame_h * frame_w, 1))

        for conf, bbox, source in merged_candidates:
            bbox_area_ratio = float((bbox[2] * bbox[3]) / frame_area)

            required_conf = self._required_person_conf(bbox_area_ratio)

            if conf < required_conf:
                continue

            people_boxes.append(bbox)
            detections.append(
                Detection(
                    disaster_type="survivor",
                    confidence=conf,
                    bbox=bbox,
                    message="Person detected by dedicated person model",
                    metadata={
                        "source": source,
                        "bbox_area_ratio": round(bbox_area_ratio, 5),
                    },
                )
            )

        return detections, people_boxes

    @staticmethod
    def _required_person_conf(bbox_area_ratio: float) -> float:
        required_conf = settings.min_conf_person
        if bbox_area_ratio <= settings.person_far_bbox_area_ratio:
            required_conf = min(settings.min_conf_person, settings.min_conf_person_far)
        return required_conf

    def _predict_person_candidates(
        self,
        frame: np.ndarray,
        request_conf: float,
        resize_scale: float,
        source: str,
    ) -> List[Tuple[float, Tuple[int, int, int, int], str]]:
        infer_frame = frame
        scale = max(resize_scale, 1.0)
        if scale > 1.0:
            infer_frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

        results = self.person_model.predict(
            infer_frame,
            conf=request_conf,
            iou=settings.yolo_iou_threshold,
            imgsz=settings.person_yolo_imgsz,
            device=settings.yolo_device,
            classes=[0],  # COCO person class
            max_det=settings.person_max_det,
            augment=settings.person_predict_augment,
            verbose=False,
        )
        if not results:
            return []

        boxes = results[0].boxes
        if boxes is None:
            return []

        h, w = frame.shape[:2]
        candidates: List[Tuple[float, Tuple[int, int, int, int], str]] = []

        for box in boxes:
            conf = float(box.conf[0].item())
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]

            if scale > 1.0:
                x1 = int(x1 / scale)
                y1 = int(y1 / scale)
                x2 = int(x2 / scale)
                y2 = int(y2 / scale)

            x1 = max(0, min(x1, w - 1))
            y1 = max(0, min(y1, h - 1))
            x2 = max(x1 + 1, min(x2, w))
            y2 = max(y1 + 1, min(y2, h))

            bbox = (x1, y1, max(1, x2 - x1), max(1, y2 - y1))
            candidates.append((conf, bbox, source))

        return candidates

    @staticmethod
    def _bbox_iou(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
        ax, ay, aw, ah = a
        bx, by, bw, bh = b

        ax2, ay2 = ax + aw, ay + ah
        bx2, by2 = bx + bw, by + bh

        inter_x1 = max(ax, bx)
        inter_y1 = max(ay, by)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)

        inter_w = max(0, inter_x2 - inter_x1)
        inter_h = max(0, inter_y2 - inter_y1)
        inter_area = float(inter_w * inter_h)
        if inter_area <= 0:
            return 0.0

        union = float(aw * ah + bw * bh) - inter_area
        return inter_area / max(union, 1.0)

    def _merge_person_candidates(
        self, candidates: List[Tuple[float, Tuple[int, int, int, int], str]]
    ) -> List[Tuple[float, Tuple[int, int, int, int], str]]:
        sorted_candidates = sorted(candidates, key=lambda item: item[0], reverse=True)
        kept: List[Tuple[float, Tuple[int, int, int, int], str]] = []

        for conf, bbox, source in sorted_candidates:
            duplicate = False
            for _, kept_bbox, _ in kept:
                if self._bbox_iou(bbox, kept_bbox) >= settings.person_dedupe_iou:
                    duplicate = True
                    break

            if not duplicate:
                kept.append((conf, bbox, source))

        return kept

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

    def _passes_semantic_validation(
        self,
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

            has_flood_word = any(token in original_label for token in ("flood", "inundation", "waterlog"))

            width_ratio = float(bw / max(1, w))
            height_ratio = float(bh / max(1, h))
            aspect_ratio = float(bw / max(bh, 1))
            if width_ratio < settings.flood_min_width_ratio:
                return False
            if height_ratio > settings.flood_max_height_ratio:
                return False
            if aspect_ratio < settings.flood_min_aspect_ratio:
                return False

            top_ratio = float(y / max(1, h))
            bottom_ratio = float((y + bh) / max(1, h))
            if top_ratio < settings.flood_min_top_ratio:
                return False
            if bottom_ratio < settings.flood_min_bottom_ratio:
                return False

            roi = frame[max(0, y):y2, max(0, x):x2]
            if roi.size == 0:
                return False

            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            blue_mask = cv2.inRange(hsv, np.array([85, 40, 40]), np.array([140, 255, 255]))
            muddy_mask = cv2.inRange(hsv, np.array([6, 35, 35]), np.array([32, 190, 235]))
            low_sat_mask = cv2.inRange(hsv, np.array([0, 0, 70]), np.array([180, 65, 255]))
            pink_mask = cv2.inRange(hsv, np.array([135, 70, 60]), np.array([179, 255, 255]))
            fire_like_mask = cv2.inRange(hsv, np.array([0, 120, 120]), np.array([35, 255, 255]))

            pixel_count = float(max(hsv.shape[0] * hsv.shape[1], 1))
            blue_ratio = float(np.count_nonzero(blue_mask)) / pixel_count
            muddy_ratio = float(np.count_nonzero(muddy_mask)) / pixel_count
            low_sat_ratio = float(np.count_nonzero(low_sat_mask)) / pixel_count
            pink_ratio = float(np.count_nonzero(pink_mask)) / pixel_count
            fire_ratio = float(np.count_nonzero(fire_like_mask)) / pixel_count
            water_signature = max(
                blue_ratio + low_sat_ratio * 0.20,
                muddy_ratio + low_sat_ratio * settings.flood_low_sat_weight,
            )
            # Require at least blue OR muddy colour; low-saturation alone is
            # NOT sufficient because grey walls, concrete, and skin tones all
            # register as low-saturation and would pass this gate.
            has_water_signature = (
                blue_ratio >= settings.flood_min_blue_ratio
                or muddy_ratio >= settings.flood_min_muddy_ratio
                or (low_sat_ratio >= settings.flood_min_low_sat_ratio
                    and max(blue_ratio, muddy_ratio) >= settings.flood_min_blue_ratio * 0.5)
            )

            if settings.water_label_strict:
                if ("water" in original_label) and (not has_flood_word) and conf < settings.flood_high_conf_override:
                    return False

            # Pink/magenta walls are a frequent false-positive source for flood; reject them.
            if pink_ratio > settings.flood_max_pink_ratio and max(blue_ratio, muddy_ratio) < settings.flood_min_blue_ratio:
                return False

            if fire_ratio > settings.flood_max_fire_color_ratio:
                return False

            # Indoor static scenes should not pass flood unless water-color evidence is strong.
            if (
                self.last_flood_motion_ratio < settings.flood_min_motion_ratio
                and water_signature < settings.flood_static_water_signature
            ):
                return False

            if settings.flood_require_water_signature and (not has_water_signature):
                return False

            if (not settings.flood_require_water_signature) and (
                water_signature < settings.flood_min_color_ratio
            ) and (conf < settings.flood_high_conf_override):
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

            required_frames = max(settings.min_consecutive_frames, 1)
            if dtype == "survivor":
                required_frames = max(settings.survivor_min_consecutive_frames, 1)

            if self.type_streaks[dtype] >= required_frames:
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

    def _estimate_flood_motion(self, frame: np.ndarray) -> float:
        h, _ = frame.shape[:2]
        y0 = int(h * settings.flood_motion_roi_start_ratio)
        roi = frame[max(0, min(y0, h - 1)):, :]
        if roi.size == 0:
            return 0.0

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        motion_ratio = 0.0

        prev = self.motion_state.prev_flood_gray
        if prev is not None and prev.shape == gray.shape:
            diff = cv2.absdiff(prev, gray)
            motion_ratio = float(np.count_nonzero(diff >= settings.flood_motion_diff_threshold)) / float(
                max(diff.size, 1)
            )

        self.motion_state.prev_flood_gray = gray
        return motion_ratio

    def _detect_flood_heuristic(self, frame: np.ndarray, flood_motion_ratio: float) -> List[Detection]:
        h, _ = frame.shape[:2]
        y0 = int(h * settings.flood_motion_roi_start_ratio)
        roi = frame[max(0, min(y0, h - 1)):, :]
        if roi.size == 0:
            return []

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        blue_mask = cv2.inRange(hsv, np.array([85, 40, 40]), np.array([140, 255, 255]))
        muddy_mask = cv2.inRange(hsv, np.array([6, 35, 35]), np.array([32, 190, 235]))
        low_sat_mask = cv2.inRange(hsv, np.array([0, 0, 70]), np.array([180, 65, 255]))
        pink_mask = cv2.inRange(hsv, np.array([135, 70, 60]), np.array([179, 255, 255]))

        pixel_count = float(max(hsv.shape[0] * hsv.shape[1], 1))
        blue_ratio = float(np.count_nonzero(blue_mask)) / pixel_count
        muddy_ratio = float(np.count_nonzero(muddy_mask)) / pixel_count
        low_sat_ratio = float(np.count_nonzero(low_sat_mask)) / pixel_count
        pink_ratio = float(np.count_nonzero(pink_mask)) / pixel_count

        water_signature = max(
            blue_ratio + low_sat_ratio * 0.20,
            muddy_ratio + low_sat_ratio * settings.flood_low_sat_weight,
        )

        structured_mask = cv2.bitwise_or(blue_mask, muddy_mask)
        bh = structured_mask.shape[0]
        bottom_start = int(bh * 0.55)
        bottom_strip = structured_mask[bottom_start:, :]
        bottom_structured_ratio = float(np.count_nonzero(bottom_strip)) / float(max(bottom_strip.size, 1))

        if pink_ratio > settings.flood_max_pink_ratio and max(blue_ratio, muddy_ratio) < settings.flood_min_blue_ratio:
            return []

        strong_color = water_signature >= settings.flood_heuristic_min_color_score
        very_strong_color = water_signature >= settings.flood_heuristic_high_color_score
        enough_bottom = bottom_structured_ratio >= settings.flood_heuristic_min_bottom_ratio
        enough_motion = flood_motion_ratio >= settings.flood_heuristic_min_motion_ratio

        if not enough_bottom:
            return []

        if not (very_strong_color or (strong_color and enough_motion)):
            return []

        confidence = min(
            0.48
            + water_signature * 1.45
            + min(flood_motion_ratio, 0.12) * 2.20,
            0.92,
        )

        return [
            Detection(
                disaster_type="flood",
                confidence=confidence,
                message="Heuristic flood flow detected",
                metadata={
                    "source": "flood_heuristic",
                    "water_signature": round(water_signature, 4),
                    "flood_motion_ratio": round(flood_motion_ratio, 4),
                    "bottom_structured_ratio": round(bottom_structured_ratio, 4),
                },
            )
        ]

    @staticmethod
    def _norm_score(value: float, low: float, high: float) -> float:
        if high <= low:
            return 1.0 if value >= high else 0.0
        return float(np.clip((value - low) / (high - low), 0.0, 1.0))

    @staticmethod
    def _water_region_edge_density(roi_gray: np.ndarray, water_mask: np.ndarray) -> float:
        """Measure Canny edge density in water-colored regions.

        Real flood water has very smooth, uniform surfaces → low edge density.
        Indoor objects (clothing, furniture, textured walls) have far higher
        edge density.  Returns ratio of edge pixels to total water pixels.
        """
        water_pixel_count = int(np.count_nonzero(water_mask))
        if water_pixel_count < 200:
            # Too few water pixels to analyse meaningfully.
            return 1.0  # Treat as high-edge (non-water)

        # Mask edges: compute Canny on the whole ROI, then AND with water mask
        edges = cv2.Canny(roi_gray, 50, 150)
        water_edges = cv2.bitwise_and(edges, water_mask)
        edge_count = int(np.count_nonzero(water_edges))
        return float(edge_count) / float(water_pixel_count)

    @staticmethod
    def _water_horizontal_continuity(water_mask: np.ndarray) -> float:
        """Check how continuously the water-colored pixels span the frame width.

        Real flood water forms wide horizontal bands stretching across most of
        the frame.  Hanging clothes, isolated objects, or narrow patches produce
        a low horizontal continuity score.

        For each row in the water mask, compute the fraction of consecutive runs
        that span at least 50% of the frame width.  Return the fraction of rows
        in the bottom half with good horizontal span.
        """
        mask_h, mask_w = water_mask.shape[:2]
        if mask_h == 0 or mask_w == 0:
            return 0.0

        # Only look at the bottom 50% of the ROI where flood water concentrates.
        start_row = mask_h // 2
        target_span = max(int(mask_w * 0.40), 1)  # 40% of frame width
        rows_with_span = 0
        total_rows = max(mask_h - start_row, 1)

        for row_idx in range(start_row, mask_h, 2):  # step by 2 for speed
            row = water_mask[row_idx, :]
            if np.count_nonzero(row) < target_span:
                continue

            # Find the longest continuous run of nonzero pixels.
            nonzero = row > 0
            max_run = 0
            current_run = 0
            for val in nonzero:
                if val:
                    current_run += 1
                    max_run = max(max_run, current_run)
                else:
                    current_run = 0

            if max_run >= target_span:
                rows_with_span += 1

        return float(rows_with_span) / float(total_rows // 2 + 1)  # accounting for step=2

    def _detect_flood_fusion(self, frame: np.ndarray, flood_motion_ratio: float) -> List[Detection]:
        h, w_frame = frame.shape[:2]
        
        # Validating YOLO candidates first: reject boxes that are completely un-flood-like in shape or cover 100% of frame
        valid_yolo_candidates = []
        frame_area = float(max(h * w_frame, 1))
        
        for conf, bbox, label in self.yolo_flood_candidates:
            bx, by, bw, bh = bbox
            bbox_area_ratio = (bw * bh) / frame_area
            # A box that covers literally the entire screen (>85%) or is perfectly square is often an artifact of indoor false positives
            if bbox_area_ratio > 0.85:
                continue
            if bbox_area_ratio < settings.flood_min_bbox_area_ratio:
                continue
            valid_yolo_candidates.append((conf, bbox, label))

        y0 = int(h * settings.flood_motion_roi_start_ratio)
        roi = frame[max(0, min(y0, h - 1)):, :]
        if roi.size == 0:
            self.flood_hysteresis_active = False
            return []

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        blue_mask = cv2.inRange(hsv, np.array([85, 40, 40]), np.array([140, 255, 255]))
        # Increased minimum saturation for muddy mask from 35 to 60 to prevent indoor wood/skin/walls from triggering it
        muddy_mask = cv2.inRange(hsv, np.array([6, 60, 35]), np.array([32, 190, 235]))
        low_sat_mask = cv2.inRange(hsv, np.array([0, 0, 70]), np.array([180, 65, 255]))
        pink_mask = cv2.inRange(hsv, np.array([135, 70, 60]), np.array([179, 255, 255]))
        fire_like_mask = cv2.inRange(hsv, np.array([0, 120, 120]), np.array([35, 255, 255]))

        pixel_count = float(max(hsv.shape[0] * hsv.shape[1], 1))
        blue_ratio = float(np.count_nonzero(blue_mask)) / pixel_count
        muddy_ratio = float(np.count_nonzero(muddy_mask)) / pixel_count
        low_sat_ratio = float(np.count_nonzero(low_sat_mask)) / pixel_count
        pink_ratio = float(np.count_nonzero(pink_mask)) / pixel_count
        fire_ratio = float(np.count_nonzero(fire_like_mask)) / pixel_count

        water_signature = max(
            blue_ratio + (low_sat_ratio * 0.10 if blue_ratio > 0.05 else 0.0),
            muddy_ratio + (low_sat_ratio * 0.25 if muddy_ratio > 0.10 else 0.0),
        )

        structured_mask = cv2.bitwise_or(blue_mask, muddy_mask)
        bh = structured_mask.shape[0]
        bottom_start = int(bh * 0.55)
        bottom_strip = structured_mask[bottom_start:, :]
        bottom_structured_ratio = float(np.count_nonzero(bottom_strip)) / float(max(bottom_strip.size, 1))

        water_component = self._norm_score(
            water_signature,
            settings.flood_heuristic_min_color_score,
            settings.flood_heuristic_high_color_score * 1.5,
        )
        motion_component = self._norm_score(
            flood_motion_ratio,
            settings.flood_heuristic_min_motion_ratio,
            settings.flood_heuristic_min_motion_ratio * 5.0,
        )
        bottom_component = self._norm_score(
            bottom_structured_ratio,
            settings.flood_heuristic_min_bottom_ratio,
            settings.flood_heuristic_min_bottom_ratio * 3.0,
        )

        # Scene gate heavily depends on actual water signature now, not just low sat
        scene_gate = 0.60 * water_component + 0.25 * motion_component + 0.15 * bottom_component

        if pink_ratio > settings.flood_max_pink_ratio and max(blue_ratio, muddy_ratio) < settings.flood_min_blue_ratio:
            scene_gate *= 0.10
        if fire_ratio > settings.flood_max_fire_color_ratio:
            scene_gate *= 0.10

        # ── NEW: Edge-density suppression ───────────────────────────────
        # Real water is smooth; objects like clothing/furniture have lots of
        # texture edges.  Heavily penalise the scene gate when the water-
        # coloured pixels contain dense edges.
        roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        water_edge_density = self._water_region_edge_density(roi_gray, structured_mask)
        if water_edge_density > settings.flood_max_water_edge_density:
            edge_penalty = min(
                (water_edge_density - settings.flood_max_water_edge_density)
                / max(settings.flood_max_water_edge_density, 0.01)
                * 0.6,
                0.85,
            )
            scene_gate *= (1.0 - edge_penalty)

        # ── NEW: Horizontal continuity suppression ─────────────────────
        # Flood water spans the frame as broad horizontal bands.  Narrow
        # vertical columns of blue (e.g. hanging clothes) produce very low
        # horizontal continuity.
        h_continuity = self._water_horizontal_continuity(structured_mask)
        if h_continuity < settings.flood_min_horizontal_continuity:
            continuity_penalty = 1.0 - (h_continuity / max(settings.flood_min_horizontal_continuity, 0.01))
            scene_gate *= max(0.05, 1.0 - continuity_penalty * 0.75)

        yolo_conf = 0.0
        best_bbox: Tuple[int, int, int, int] | None = None
        best_label = ""
        for conf, bbox, label in valid_yolo_candidates:
            if conf > yolo_conf:
                yolo_conf = conf
                best_bbox = bbox
                best_label = label

        # Hard floor: if there is essentially no water-like colour anywhere in
        # the lower portion of the frame, reject immediately regardless of what
        # YOLO thinks.  Real floods always show *some* blue, muddy, or grey tone.
        min_water_evidence = max(blue_ratio, muddy_ratio, low_sat_ratio * 0.3)
        if min_water_evidence < 0.008:
            self.flood_hysteresis_active = False
            return []

        # Also require a minimum of structured water colour in the bottom strip
        if bottom_structured_ratio < 0.005:
            self.flood_hysteresis_active = False
            return []

        # Heuristic-only path: require strong evidence on BOTH colour and motion
        heuristic_score = 0.0
        if settings.enable_flood_heuristic_assist and water_component > 0.5 and motion_component > 0.4 and bottom_component > 0.3:
            heuristic_score = 0.20 + 0.40 * water_component + 0.25 * motion_component + 0.15 * bottom_component

        # YOLO path: scene_gate must be non-trivial.  Previously the formula
        # gave 15% "free" credit (0.15 + 0.85*gate) even when gate==0, letting
        # any YOLO detection with conf>=0.60 create a flood event in a living
        # room.  Now YOLO score is entirely gated by the scene evidence.
        if scene_gate < 0.10:
            yolo_scene_score = 0.0  # no water evidence → YOLO flood ignored
        else:
            yolo_scene_score = yolo_conf * scene_gate
        fused_score = max(yolo_scene_score, heuristic_score)

        score_threshold = (
            settings.flood_fusion_off_threshold
            if self.flood_hysteresis_active
            else settings.flood_fusion_on_threshold
        )
        scene_threshold = (
            settings.flood_scene_gate_off_threshold
            if self.flood_hysteresis_active
            else settings.flood_scene_gate_on_threshold
        )

        if fused_score < score_threshold or scene_gate < scene_threshold:
            self.flood_hysteresis_active = False
            return []

        self.flood_hysteresis_active = True
        confidence = float(min(max(fused_score, settings.min_conf_flood), 0.98))

        return [
            Detection(
                disaster_type="flood",
                confidence=confidence,
                bbox=best_bbox,
                message="Flood pattern detected",
                metadata={
                    "source": "flood_fusion",
                    "yolo_flood_conf": round(yolo_conf, 4),
                    "yolo_flood_label": best_label,
                    "water_signature": round(water_signature, 4),
                    "flood_motion_ratio": round(flood_motion_ratio, 4),
                    "scene_gate": round(scene_gate, 4),
                    "bottom_structured_ratio": round(bottom_structured_ratio, 4),
                    "fused_score": round(fused_score, 4),
                    "water_edge_density": round(water_edge_density, 4),
                    "horizontal_continuity": round(h_continuity, 4),
                },
            )
        ]

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
