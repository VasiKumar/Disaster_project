from __future__ import annotations

from datetime import datetime
from typing import List

from app.schemas import AlertEvent, Incident


class AlertDispatcher:
    def dispatch(self, incident: Incident) -> List[AlertEvent]:
        events = [
            AlertEvent(incident_id=incident.id, alert_type="dashboard", dispatched=True),
            AlertEvent(
                incident_id=incident.id,
                alert_type="alarm",
                dispatched=incident.severity in {"high", "critical"},
                dispatched_at=datetime.utcnow(),
            ),
        ]

        if incident.severity in {"high", "critical"}:
            events.append(
                AlertEvent(
                    incident_id=incident.id,
                    alert_type="sms",
                    dispatched=True,
                    dispatched_at=datetime.utcnow(),
                )
            )
            events.append(
                AlertEvent(
                    incident_id=incident.id,
                    alert_type="app",
                    dispatched=True,
                    dispatched_at=datetime.utcnow(),
                )
            )

        return events
