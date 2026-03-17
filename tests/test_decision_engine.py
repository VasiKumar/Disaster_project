from app.services.decision_engine import DecisionEngine
from app.schemas import Detection


def test_generates_incident_from_fire_detection():
    engine = DecisionEngine()
    detections = [
        Detection(disaster_type="fire", confidence=0.9, message="fire"),
        Detection(disaster_type="fire", confidence=0.8, message="fire"),
    ]

    incidents = engine.generate_incidents(
        detections=detections,
        camera_id="CAM-01",
        location_tag="Sector-A",
    )

    assert len(incidents) == 1
    assert incidents[0].disaster_type == "fire"
    assert incidents[0].severity in {"high", "critical"}


def test_generates_incident_from_earthquake_signal():
    engine = DecisionEngine()
    detections = [
        Detection(disaster_type="earthquake", confidence=0.82, message="shake"),
    ]

    incidents = engine.generate_incidents(
        detections=detections,
        camera_id="CAM-02",
        location_tag="Sector-B",
    )

    assert len(incidents) == 1
    assert incidents[0].disaster_type == "earthquake"
    assert incidents[0].severity in {"high", "critical", "medium"}
