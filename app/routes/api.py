from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse

from app.config import settings
from app.schemas import DashboardSnapshot, IncidentLogQueryResponse, IncidentLogRecord, SystemStatus
from app.services.analytics import build_metrics
from app.services.mongo_logs import mongo_log_service
from app.services.video_processor import video_processor
from app.state import state

router = APIRouter(prefix="/api", tags=["api"])


@router.post("/start")
def start_processing() -> dict:
    video_processor.start()
    return {"ok": True, "message": "Processing started"}


@router.post("/stop")
def stop_processing() -> dict:
    video_processor.stop()
    return {"ok": True, "message": "Processing stopped"}


@router.get("/status", response_model=SystemStatus)
def get_status() -> SystemStatus:
    active_incidents = state.get_active_incidents()
    return SystemStatus(
        running=state.running,
        source=settings.camera_source,
        last_frame_ts=state.last_frame_ts,
        active_incidents=len(active_incidents),
        total_incidents=len(state.incidents),
        cameras_online=1 if state.running else 0,
        fps=state.current_fps,
        people_in_frame=state.people_in_frame,
        risk_score=state.risk_score,
        risk_level=state.risk_level,
    )


@router.get("/dashboard", response_model=DashboardSnapshot)
def get_dashboard_snapshot() -> DashboardSnapshot:
    active = state.get_active_incidents()
    recent = state.get_recent_incidents()
    status = get_status()

    return DashboardSnapshot(
        status=status,
        active_incidents=active,
        recent_incidents=recent,
        metrics=build_metrics(
            active,
            recent,
            state.people_in_frame,
            state.risk_score,
            state.risk_level,
            state.risk_features,
            state.risk_contributions,
        ),
    )


@router.get("/people_count")
def get_people_count() -> dict:
    return {"people_in_frame": state.people_in_frame, "running": state.running}


@router.get("/video_feed")
def video_feed() -> StreamingResponse:
    if not state.running:
        raise HTTPException(status_code=400, detail="Video processor is not running")

    return StreamingResponse(
        video_processor.mjpeg_stream(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@router.get("/logs", response_model=IncidentLogQueryResponse)
def get_logs(
    date: str | None = Query(default=None, description="Filter by date: YYYY-MM-DD"),
    tag: str | None = Query(default=None, description="Filter by tag, e.g. human_fire"),
    search: str | None = Query(default=None, description="Text search in incident message"),
    limit: int = Query(default=100, ge=1, le=500),
) -> IncidentLogQueryResponse:
    rows = mongo_log_service.search_logs(date_str=date, tag=tag, search=search, limit=limit)
    logs = [IncidentLogRecord(**row) for row in rows]
    return IncidentLogQueryResponse(total=len(logs), logs=logs)
