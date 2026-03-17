from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional, Tuple

from pydantic import BaseModel, Field

DisasterType = Literal[
    "survivor",
    "fire",
    "smoke",
    "flood",
    "earthquake",
    "road_accident",
    "crowd_panic",
    "fallen_person",
    "unsafe_zone",
]


class Detection(BaseModel):
    disaster_type: DisasterType
    confidence: float = Field(ge=0.0, le=1.0)
    bbox: Optional[Tuple[int, int, int, int]] = None
    message: str = ""
    metadata: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class Incident(BaseModel):
    id: str
    disaster_type: DisasterType
    severity: Literal["low", "medium", "high", "critical"]
    camera_id: str
    message: str
    location_tag: str
    confidence: float
    created_at: datetime = Field(default_factory=datetime.utcnow)
    resolved: bool = False
    metadata: Dict[str, Any] = Field(default_factory=dict)


class SystemStatus(BaseModel):
    running: bool
    source: str
    last_frame_ts: Optional[datetime]
    active_incidents: int
    total_incidents: int
    cameras_online: int
    fps: float
    people_in_frame: int


class AlertEvent(BaseModel):
    incident_id: str
    alert_type: Literal["dashboard", "alarm", "sms", "app"]
    dispatched: bool
    dispatched_at: datetime = Field(default_factory=datetime.utcnow)


class DashboardSnapshot(BaseModel):
    status: SystemStatus
    active_incidents: List[Incident]
    recent_incidents: List[Incident]
    metrics: Dict[str, Any]


class IncidentLogRecord(BaseModel):
    id: str
    incident_id: str
    disaster_type: DisasterType
    severity: Literal["low", "medium", "high", "critical"]
    camera_id: str
    location_tag: str
    message: str
    confidence: float
    detected_types: List[str] = Field(default_factory=list)
    danger_tags: List[str] = Field(default_factory=list)
    tags: List[str] = Field(default_factory=list)
    people_in_frame: int = 0
    created_at: datetime
    metadata: Dict[str, Any] = Field(default_factory=dict)


class IncidentLogQueryResponse(BaseModel):
    total: int
    logs: List[IncidentLogRecord]
