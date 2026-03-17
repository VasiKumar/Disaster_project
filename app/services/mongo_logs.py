from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Dict, List

from pymongo import DESCENDING, MongoClient

from app.config import settings
from app.schemas import Incident


class MongoLogService:
    def __init__(self) -> None:
        self._client: MongoClient | None = None
        self._collection = None
        self._available = True

    def _get_collection(self):
        if self._collection is not None:
            return self._collection
        if not self._available:
            return None

        try:
            self._client = MongoClient(settings.mongodb_uri, serverSelectionTimeoutMS=1500)
            db = self._client[settings.mongodb_db_name]
            self._collection = db[settings.mongodb_collection_name]
            self._collection.create_index([("created_at", DESCENDING)])
            self._collection.create_index([("tags", 1)])
            return self._collection
        except Exception:
            self._available = False
            self._collection = None
            return None

    def save_incident_log(
        self,
        incident: Incident,
        detected_types: List[str],
        danger_tags: List[str],
        people_in_frame: int,
    ) -> bool:
        collection = self._get_collection()
        if collection is None:
            return False

        payload = incident.model_dump(mode="python")
        tags = sorted(set(detected_types + danger_tags + [incident.disaster_type]))
        payload.update(
            {
                "incident_id": incident.id,
                "detected_types": sorted(set(detected_types)),
                "danger_tags": sorted(set(danger_tags)),
                "tags": tags,
                "people_in_frame": people_in_frame,
            }
        )

        try:
            collection.insert_one(payload)
            return True
        except Exception:
            return False

    def search_logs(
        self,
        date_str: str | None = None,
        tag: str | None = None,
        search: str | None = None,
        limit: int = 200,
    ) -> List[Dict[str, Any]]:
        collection = self._get_collection()
        if collection is None:
            return []

        query: Dict[str, Any] = {}

        if date_str:
            try:
                day_start = datetime.strptime(date_str, "%Y-%m-%d")
                day_end = day_start + timedelta(days=1)
                query["created_at"] = {"$gte": day_start, "$lt": day_end}
            except ValueError:
                pass

        if tag:
            query["tags"] = tag

        if search:
            query["message"] = {"$regex": search, "$options": "i"}

        rows = list(collection.find(query).sort("created_at", DESCENDING).limit(max(1, min(limit, 1000))))
        parsed: List[Dict[str, Any]] = []
        for row in rows:
            row["id"] = str(row.pop("_id"))
            parsed.append(row)

        return parsed


mongo_log_service = MongoLogService()
