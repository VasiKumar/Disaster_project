from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timedelta
from typing import Dict, List
from uuid import uuid4

from app.schemas import Detection, Incident


class DecisionEngine:
    def __init__(self) -> None:
        self.cooldowns: Dict[str, datetime] = {}
        self.cooldown_window = timedelta(seconds=8)

    def generate_incidents(
        self,
        detections: List[Detection],
        camera_id: str,
        location_tag: str,
    ) -> List[Incident]:
        grouped: Dict[str, List[Detection]] = defaultdict(list)
        for detection in detections:
            grouped[detection.disaster_type].append(detection)

        incidents: List[Incident] = []
        for disaster_type, group in grouped.items():
            avg_conf = sum(d.confidence for d in group) / len(group)
            severity = self._severity_from_type_and_conf(disaster_type, avg_conf, len(group))

            key = f"{camera_id}:{disaster_type}"
            if self._in_cooldown(key):
                continue

            incidents.append(
                Incident(
                    id=str(uuid4()),
                    disaster_type=disaster_type,
                    severity=severity,
                    camera_id=camera_id,
                    location_tag=location_tag,
                    confidence=round(avg_conf, 3),
                    message=self._build_message(disaster_type, severity, len(group)),
                    metadata={
                        "detection_count": len(group),
                        "sample": group[0].metadata,
                    },
                )
            )
            self.cooldowns[key] = datetime.utcnow()

        return incidents

    def _in_cooldown(self, key: str) -> bool:
        now = datetime.utcnow()
        last_seen = self.cooldowns.get(key)
        return bool(last_seen and (now - last_seen) < self.cooldown_window)

    def _severity_from_type_and_conf(self, disaster_type: str, conf: float, count: int) -> str:
        base = conf * 100 + min(count * 5, 20)

        if disaster_type in {"fire", "road_accident", "crowd_panic", "earthquake", "flood"}:
            base += 15
        if disaster_type in {"fallen_person", "unsafe_zone"}:
            base += 10

        if base >= 85:
            return "critical"
        if base >= 65:
            return "high"
        if base >= 45:
            return "medium"
        return "low"

    def _build_message(self, disaster_type: str, severity: str, count: int) -> str:
        labels = {
            "survivor": "Survivor(s) detected",
            "fire": "Fire detected",
            "smoke": "Smoke detected",
            "flood": "Flood/waterlogging detected",
            "earthquake": "Earthquake/collapse signal detected",
            "road_accident": "Potential road accident detected",
            "crowd_panic": "Crowd panic/stampede risk detected",
            "fallen_person": "Fallen person detected",
            "unsafe_zone": "Unsafe zone breach detected",
        }
        return f"{labels.get(disaster_type, disaster_type)} | Severity: {severity.upper()} | Count: {count}"
