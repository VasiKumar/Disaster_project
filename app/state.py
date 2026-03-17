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
        self.running: bool = False

        self.incidents: List[Incident] = []
        self.recent_incidents: Deque[Incident] = deque(maxlen=150)

    def add_incident(self, incident: Incident) -> None:
        with self.lock:
            self.incidents.append(incident)
            self.recent_incidents.appendleft(incident)

    def get_active_incidents(self) -> List[Incident]:
        with self.lock:
            return [incident for incident in self.incidents if not incident.resolved]

    def get_recent_incidents(self, limit: int = 30) -> List[Incident]:
        with self.lock:
            return list(self.recent_incidents)[:limit]


state = AppState()
