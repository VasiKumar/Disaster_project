from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Iterator

import cv2

from app.config import settings
from app.schemas import Incident
from app.services.alerts import AlertDispatcher
from app.services.decision_engine import DecisionEngine
from app.services.detector import MultiDisasterDetector
from app.services.mongo_logs import mongo_log_service
from app.services.risk_regression import LinearRiskRegressor
from app.state import state


class VideoProcessor:
    def __init__(self) -> None:
        self.detector = MultiDisasterDetector()
        self.decision_engine = DecisionEngine()
        self.alert_dispatcher = AlertDispatcher()
        self.risk_regressor = LinearRiskRegressor()
        self.thread: threading.Thread | None = None
        self.stop_event = threading.Event()
        self.cap: cv2.VideoCapture | None = None
        self.cap_lock = threading.Lock()

    def start(self) -> None:
        if state.running:
            return
        self.stop_event.clear()
        state.running = True
        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        state.running = False

        with self.cap_lock:
            if self.cap is not None:
                self.cap.release()
                self.cap = None

        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=2.0)

    def _run_loop(self) -> None:
        source: int | str
        source = int(settings.camera_source) if settings.camera_source.isdigit() else settings.camera_source
        cap = cv2.VideoCapture(source)
        with self.cap_lock:
            self.cap = cap

        if not cap.isOpened():
            state.running = False
            return

        prev_ts = time.time()
        while not self.stop_event.is_set():
            ok, frame = cap.read()
            if not ok:
                break

            detections = self.detector.detect(frame)
            risk = self.risk_regressor.predict(detections, self.detector.last_people_count)
            with state.lock:
                state.people_in_frame = self.detector.last_people_count
                state.risk_score = risk.score
                state.risk_level = risk.level
                state.risk_raw = risk.raw
                state.risk_features = risk.features
                state.risk_contributions = risk.contributions

            incidents = self.decision_engine.generate_incidents(
                detections=detections,
                camera_id=settings.camera_id,
                location_tag=settings.location_tag,
            )

            detected_types, danger_tags = self._build_detection_tags(detections)
            state.resolve_missing_incidents(
                detected_types=detected_types,
                missing_grace_frames=settings.incident_missing_grace_frames,
            )

            for incident in incidents:
                state.add_incident(incident)
                self.alert_dispatcher.dispatch(incident)
                self._append_incident_log(incident)
                self._append_mongo_log(incident, detected_types, danger_tags)

            annotated = self._annotate_frame(frame, detections, incidents)
            ok_encode, jpeg = cv2.imencode(".jpg", annotated)
            if ok_encode:
                with state.lock:
                    state.latest_frame_jpeg = jpeg.tobytes()
                    state.last_frame_ts = incident_time()

            now = time.time()
            delta = max(now - prev_ts, 1e-3)
            prev_ts = now
            with state.lock:
                state.current_fps = round(1.0 / delta, 2)

        cap.release()
        with self.cap_lock:
            self.cap = None
        state.running = False

    def mjpeg_stream(self) -> Iterator[bytes]:
        while not self.stop_event.is_set():
            if not state.running:
                break

            with state.lock:
                frame = state.latest_frame_jpeg

            if frame is None:
                time.sleep(0.03)
                continue

            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"

    def _annotate_frame(self, frame, detections, incidents):
        frame_h, frame_w = frame.shape[:2]
        label_boxes: list[tuple[int, int, int, int]] = []

        for detection in detections:
            if detection.bbox:
                x, y, w, h = detection.bbox
                color = self._color_for_type(detection.disaster_type)
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                label = f"{detection.disaster_type} {detection.confidence:.2f}"
                (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
                px, py = self._place_label_position(
                    desired_x=x,
                    desired_y=max(20, y - 8),
                    text_w=tw,
                    text_h=th,
                    baseline=baseline,
                    frame_w=frame_w,
                    frame_h=frame_h,
                    occupied=label_boxes,
                )

                bg_top = max(0, py - th - baseline - 2)
                bg_bottom = min(frame_h - 1, py + baseline + 2)
                bg_left = max(0, px - 2)
                bg_right = min(frame_w - 1, px + tw + 2)
                cv2.rectangle(frame, (bg_left, bg_top), (bg_right, bg_bottom), color, -1)
                cv2.putText(
                    frame,
                    label,
                    (px, py),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    2,
                )

                label_boxes.append((bg_left, bg_top, bg_right, bg_bottom))

        top = 24
        cv2.putText(
            frame,
            f"People in frame: {self.detector.last_people_count}",
            (10, top),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (0, 255, 0),
            2,
        )
        top += 28
        cv2.putText(
            frame,
            f"Risk score: {state.risk_score:.1f} ({state.risk_level})",
            (10, top),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (0, 220, 255),
            2,
        )
        top += 28

        for incident in incidents[:5]:
            text = f"{incident.severity.upper()} | {incident.disaster_type}"
            cv2.putText(
                frame,
                text,
                (10, top),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2,
            )
            top += 24

        return frame

    @staticmethod
    def _place_label_position(
        desired_x: int,
        desired_y: int,
        text_w: int,
        text_h: int,
        baseline: int,
        frame_w: int,
        frame_h: int,
        occupied: list[tuple[int, int, int, int]],
    ) -> tuple[int, int]:
        x = max(2, min(desired_x, max(2, frame_w - text_w - 4)))
        y = max(text_h + baseline + 4, min(desired_y, frame_h - baseline - 4))

        for _ in range(20):
            left = x - 2
            top = y - text_h - baseline - 2
            right = x + text_w + 2
            bottom = y + baseline + 2

            overlap = None
            for ox1, oy1, ox2, oy2 in occupied:
                intersects = not (right < ox1 or left > ox2 or bottom < oy1 or top > oy2)
                if intersects:
                    overlap = (ox1, oy1, ox2, oy2)
                    break

            if overlap is None:
                return x, y

            x = overlap[2] + 8
            if x + text_w + 2 >= frame_w:
                x = max(2, min(desired_x, max(2, frame_w - text_w - 4)))
                y += text_h + baseline + 8
                if y >= frame_h - baseline - 4:
                    y = max(text_h + baseline + 4, desired_y - (text_h + baseline + 8))

        return x, y

    @staticmethod
    def _color_for_type(disaster_type: str):
        palette = {
            "survivor": (0, 255, 0),
            "fire": (0, 69, 255),
            "smoke": (180, 180, 180),
            "flood": (255, 180, 40),
            "earthquake": (255, 140, 0),
            "road_accident": (0, 165, 255),
            "crowd_panic": (255, 0, 255),
            "fallen_person": (255, 255, 0),
            "unsafe_zone": (0, 0, 255),
        }
        return palette.get(disaster_type, (255, 255, 255))

    def _append_incident_log(self, incident: Incident) -> None:
        if not settings.log_to_file:
            return

        path: Path = settings.incident_log_path
        existing = json.loads(path.read_text(encoding="utf-8"))
        existing.append(incident.model_dump(mode="json"))
        path.write_text(json.dumps(existing, indent=2), encoding="utf-8")

    def _append_mongo_log(
        self,
        incident: Incident,
        detected_types: list[str],
        danger_tags: list[str],
    ) -> None:
        mongo_log_service.save_incident_log(
            incident=incident,
            detected_types=detected_types,
            danger_tags=danger_tags,
            people_in_frame=self.detector.last_people_count,
        )

    @staticmethod
    def _build_detection_tags(detections) -> tuple[list[str], list[str]]:
        detected = sorted({d.disaster_type for d in detections})
        danger_tags: list[str] = []

        has_human = "survivor" in detected
        has_fire = "fire" in detected
        has_flood = "flood" in detected

        if has_fire:
            danger_tags.append("danger_fire")
        if has_flood:
            danger_tags.append("danger_flood")
        if has_human and has_fire:
            danger_tags.append("human_fire")
        if has_human and has_flood:
            danger_tags.append("human_flood")

        return detected, danger_tags


def incident_time():
    from datetime import datetime

    return datetime.utcnow()


video_processor = VideoProcessor()
