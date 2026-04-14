from __future__ import annotations

import threading
from collections import deque
from datetime import datetime
from typing import Deque, List, Optional

from app.schemas import Incident


class AppState:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.latest_frame_jpeg: Optional[bytes] = None
        self.last_frame_ts: Optional[datetime] = None
        self.current_fps: float = 0.0
        self.people_in_frame: int = 0
        self.risk_score: float = 0.0
        self.risk_level: str = "low"
        self.risk_raw: float = 0.0
        self.risk_features: dict = {}
        self.risk_contributions: dict = {}
        self.running: bool = False

        self.incidents: List[Incident] = []
        self.recent_incidents: Deque[Incident] = deque(maxlen=150)
        self.missing_type_streaks: dict[str, int] = {}

    def add_incident(self, incident: Incident) -> None:
        with self.lock:
            # Keep one active incident per type/camera so dashboard state is stable.
            for existing in self.incidents:
                if (
                    not existing.resolved
                    and existing.disaster_type == incident.disaster_type
                    and existing.camera_id == incident.camera_id
                ):
                    existing.resolved = True

            self.missing_type_streaks[incident.disaster_type] = 0
            self.incidents.append(incident)
            self.recent_incidents.appendleft(incident)

    def get_active_incidents(self) -> List[Incident]:
        with self.lock:
            return [incident for incident in self.incidents if not incident.resolved]

    def get_recent_incidents(self, limit: int = 30) -> List[Incident]:
        with self.lock:
            return list(self.recent_incidents)[:limit]

    def resolve_missing_incidents(self, detected_types: List[str], missing_grace_frames: int) -> None:
        detected = set(detected_types)
        grace = max(missing_grace_frames, 0)

        with self.lock:
            active_types = {incident.disaster_type for incident in self.incidents if not incident.resolved}

            for dtype in detected:
                self.missing_type_streaks[dtype] = 0

            for dtype in active_types:
                if dtype in detected:
                    self.missing_type_streaks[dtype] = 0
                    continue

                missed = self.missing_type_streaks.get(dtype, 0) + 1
                self.missing_type_streaks[dtype] = missed

                if missed <= grace:
                    continue

                for incident in self.incidents:
                    if not incident.resolved and incident.disaster_type == dtype:
                        incident.resolved = True

                self.missing_type_streaks.pop(dtype, None)

            # Cleanup stale counters for types with no active incident.
            for dtype in list(self.missing_type_streaks.keys()):
                if dtype not in active_types and dtype not in detected:
                    self.missing_type_streaks.pop(dtype, None)


state = AppState()
