from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

from app.schemas import Detection


@dataclass
class RiskPrediction:
    score: float
    level: str
    raw: float
    features: Dict[str, float]
    contributions: Dict[str, float]


class LinearRiskRegressor:
    """No-training linear regressor with fixed, interpretable coefficients."""

    INTERCEPT = 5.0

    # Coefficients are hand-tuned domain weights, not learned from fit().
    COEFFICIENTS: Dict[str, float] = {
        "fire_count": 18.0,
        "smoke_count": 10.0,
        "flood_count": 16.0,
        "road_accident_count": 20.0,
        "earthquake_count": 18.0,
        "crowd_panic_count": 14.0,
        "fallen_person_count": 12.0,
        "unsafe_zone_count": 22.0,
        "survivor_count": 3.0,
        "avg_disaster_conf": 25.0,
        "people_in_frame": 1.0,
    }

    def predict(self, detections: List[Detection], people_in_frame: int) -> RiskPrediction:
        features = self._build_features(detections, people_in_frame)
        contributions: Dict[str, float] = {}

        raw = self.INTERCEPT
        for name, value in features.items():
            weight = self.COEFFICIENTS.get(name, 0.0)
            contribution = weight * value
            contributions[name] = round(contribution, 4)
            raw += contribution

        score = float(max(0.0, min(100.0, raw)))
        level = self._to_level(score)

        return RiskPrediction(
            score=round(score, 2),
            level=level,
            raw=round(float(raw), 3),
            features={k: round(v, 4) for k, v in features.items()},
            contributions=contributions,
        )

    @staticmethod
    def _to_level(score: float) -> str:
        if score >= 80:
            return "critical"
        if score >= 60:
            return "high"
        if score >= 35:
            return "medium"
        return "low"

    @staticmethod
    def _build_features(detections: List[Detection], people_in_frame: int) -> Dict[str, float]:
        types = [d.disaster_type for d in detections]
        disaster_conf = [d.confidence for d in detections if d.disaster_type != "survivor"]

        return {
            "fire_count": float(types.count("fire")),
            "smoke_count": float(types.count("smoke")),
            "flood_count": float(types.count("flood")),
            "road_accident_count": float(types.count("road_accident")),
            "earthquake_count": float(types.count("earthquake")),
            "crowd_panic_count": float(types.count("crowd_panic")),
            "fallen_person_count": float(types.count("fallen_person")),
            "unsafe_zone_count": float(types.count("unsafe_zone")),
            "survivor_count": float(types.count("survivor")),
            "avg_disaster_conf": (
                float(sum(disaster_conf) / len(disaster_conf)) if disaster_conf else 0.0
            ),
            "people_in_frame": float(max(0, people_in_frame)),
        }
