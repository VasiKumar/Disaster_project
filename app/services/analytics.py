from __future__ import annotations

from collections import Counter
from typing import Dict, List

from app.schemas import Incident


def build_metrics(
    active_incidents: List[Incident],
    recent_incidents: List[Incident],
    people_in_frame: int,
    risk_score: float,
    risk_level: str,
    risk_features: Dict[str, object],
    risk_contributions: Dict[str, object],
) -> Dict[str, object]:
    total_recent = len(recent_incidents)
    by_type = Counter(incident.disaster_type for incident in recent_incidents)
    by_severity = Counter(incident.severity for incident in recent_incidents)

    critical_active = sum(1 for incident in active_incidents if incident.severity == "critical")

    return {
        "total_recent_incidents": total_recent,
        "active_incidents": len(active_incidents),
        "critical_active": critical_active,
        "incident_type_distribution": dict(by_type),
        "severity_distribution": dict(by_severity),
        "survivor_count_estimate": by_type.get("survivor", 0),
        "people_in_frame": people_in_frame,
        "risk_score": risk_score,
        "risk_level": risk_level,
        "risk_features": risk_features,
        "risk_contributions": risk_contributions,
    }
