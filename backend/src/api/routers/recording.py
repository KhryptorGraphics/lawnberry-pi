"""Recording API for AI training data collection (BEAD-104).

REST API endpoints for controlling perimeter recording sessions.
Used to capture multi-modal sensor data during manual mowing for
training the autonomous mowing AI model.

Endpoints:
- POST /recording/start - Start a new recording session
- POST /recording/stop - Stop current recording session
- POST /recording/pause - Pause recording
- POST /recording/resume - Resume recording
- GET /recording/status - Get current recording status
- GET /recording/sessions - List all recorded sessions
- GET /recording/sessions/{session_id} - Get specific session details
- DELETE /recording/sessions/{session_id} - Delete a session
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Depends, Query
from pydantic import BaseModel, Field

from ...services.perimeter_recorder import (
    PerimeterRecordingService,
    RecordingState,
    get_perimeter_recorder,
)

logger = logging.getLogger(__name__)
router = APIRouter()


# ----------------------- Request/Response Models -----------------------


class StartRecordingRequest(BaseModel):
    """Request to start a new recording session."""
    name: str = Field(..., description="Human-readable name for the session", min_length=1, max_length=100)
    session_type: str = Field("perimeter", description="Type of recording: perimeter, training, calibration")
    notes: str = Field("", description="Optional notes about the session", max_length=500)


class RecordingSessionResponse(BaseModel):
    """Response containing session metadata."""
    session_id: str
    name: str
    start_time: str
    end_time: Optional[str] = None
    frame_count: int
    total_distance_m: float = 0.0
    mowing_area_m2: float = 0.0
    session_type: str
    notes: str
    data_file: Optional[str] = None
    video_file: Optional[str] = None


class RecordingStatusResponse(BaseModel):
    """Response containing current recording status."""
    state: str = Field(..., description="Current state: idle, recording, paused, stopping")
    recording: bool
    session_id: Optional[str] = None
    session_name: Optional[str] = None
    frame_count: int = 0
    target_fps: float = 10.0
    avg_capture_time_ms: float = 0.0
    total_frames_recorded: int = 0
    total_sessions: int = 0
    storage_dir: str = ""
    msgpack_available: bool = True


class SessionListResponse(BaseModel):
    """Response containing list of sessions."""
    sessions: List[Dict[str, Any]]
    total_count: int


class HealthResponse(BaseModel):
    """Health check response."""
    service: str
    initialized: bool
    state: str
    details: Dict[str, Any]


# ----------------------- API Endpoints -----------------------


@router.post("/start", response_model=RecordingSessionResponse)
async def start_recording(
    request: StartRecordingRequest,
    recorder: PerimeterRecordingService = Depends(get_perimeter_recorder),
) -> RecordingSessionResponse:
    """Start a new recording session.

    Begins capturing multi-modal sensor data (GPS, IMU, cameras, etc.)
    at the configured frame rate. Data is stored using MessagePack format
    for efficient storage and later upload to Thor training server.

    Args:
        request: Session name, type, and optional notes.

    Returns:
        Session metadata including ID and start time.

    Raises:
        HTTPException: If already recording or service not initialized.
    """
    try:
        session = await recorder.start_recording(
            name=request.name,
            session_type=request.session_type,
            notes=request.notes,
        )

        logger.info(f"Started recording session: {session.name} ({session.session_id})")

        return RecordingSessionResponse(
            session_id=session.session_id,
            name=session.name,
            start_time=session.start_time.isoformat(),
            frame_count=0,
            session_type=session.session_type,
            notes=session.notes,
            data_file=str(session.data_file) if session.data_file else None,
        )

    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to start recording: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to start recording: {e}")


@router.post("/stop", response_model=RecordingSessionResponse)
async def stop_recording(
    recorder: PerimeterRecordingService = Depends(get_perimeter_recorder),
) -> RecordingSessionResponse:
    """Stop the current recording session.

    Finalizes the recording, calculates total distance traveled,
    and saves the session data to disk.

    Returns:
        Final session metadata including frame count and file path.

    Raises:
        HTTPException: If not currently recording.
    """
    try:
        session = await recorder.stop_recording()

        logger.info(f"Stopped recording session: {session.name}, {session.frame_count} frames")

        return RecordingSessionResponse(
            session_id=session.session_id,
            name=session.name,
            start_time=session.start_time.isoformat(),
            end_time=session.end_time.isoformat() if session.end_time else None,
            frame_count=session.frame_count,
            total_distance_m=session.total_distance_m,
            mowing_area_m2=session.mowing_area_m2,
            session_type=session.session_type,
            notes=session.notes,
            data_file=str(session.data_file) if session.data_file else None,
        )

    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to stop recording: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to stop recording: {e}")


@router.post("/pause")
async def pause_recording(
    recorder: PerimeterRecordingService = Depends(get_perimeter_recorder),
) -> Dict[str, str]:
    """Pause the current recording session.

    Temporarily stops frame capture without finalizing the session.
    Can be resumed with the /resume endpoint.

    Returns:
        Success message.

    Raises:
        HTTPException: If not currently recording.
    """
    try:
        await recorder.pause_recording()
        logger.info("Recording paused")
        return {"status": "paused", "message": "Recording paused successfully"}

    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to pause recording: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to pause recording: {e}")


@router.post("/resume")
async def resume_recording(
    recorder: PerimeterRecordingService = Depends(get_perimeter_recorder),
) -> Dict[str, str]:
    """Resume a paused recording session.

    Continues frame capture from where it was paused.

    Returns:
        Success message.

    Raises:
        HTTPException: If not currently paused.
    """
    try:
        await recorder.resume_recording()
        logger.info("Recording resumed")
        return {"status": "recording", "message": "Recording resumed successfully"}

    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to resume recording: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to resume recording: {e}")


@router.get("/status", response_model=RecordingStatusResponse)
async def get_recording_status(
    recorder: PerimeterRecordingService = Depends(get_perimeter_recorder),
) -> RecordingStatusResponse:
    """Get current recording status.

    Returns the current state of the recording service including
    active session info, frame counts, and performance metrics.

    Returns:
        Current recording status and statistics.
    """
    health = await recorder.health_check()

    return RecordingStatusResponse(
        state=health.get("state", "unknown"),
        recording=health.get("state") == RecordingState.RECORDING,
        session_id=health.get("current_session"),
        session_name=health.get("current_session_name"),
        frame_count=health.get("frame_count", 0),
        target_fps=health.get("target_fps", 10.0),
        avg_capture_time_ms=health.get("avg_capture_time_ms", 0.0),
        total_frames_recorded=health.get("total_frames_recorded", 0),
        total_sessions=health.get("total_sessions", 0),
        storage_dir=health.get("storage_dir", ""),
        msgpack_available=health.get("msgpack_available", False),
    )


@router.get("/sessions", response_model=SessionListResponse)
async def list_sessions(
    limit: int = Query(50, ge=1, le=200, description="Maximum sessions to return"),
    offset: int = Query(0, ge=0, description="Number of sessions to skip"),
    recorder: PerimeterRecordingService = Depends(get_perimeter_recorder),
) -> SessionListResponse:
    """List all recorded sessions.

    Returns a paginated list of recording session metadata.
    Sessions are sorted by start time (most recent first).

    Args:
        limit: Maximum number of sessions to return.
        offset: Number of sessions to skip for pagination.

    Returns:
        List of session metadata.
    """
    all_sessions = recorder.list_sessions()
    total = len(all_sessions)

    # Apply pagination
    sessions = all_sessions[offset : offset + limit]

    return SessionListResponse(
        sessions=sessions,
        total_count=total,
    )


@router.get("/sessions/{session_id}")
async def get_session(
    session_id: str,
    include_frames: bool = Query(False, description="Include frame data (large)"),
    recorder: PerimeterRecordingService = Depends(get_perimeter_recorder),
) -> Dict[str, Any]:
    """Get a specific recording session.

    Returns detailed session information including optionally the
    individual frame data.

    Args:
        session_id: UUID of the session.
        include_frames: If True, include all frame data (can be very large).

    Returns:
        Session metadata and optionally frame data.

    Raises:
        HTTPException: If session not found.
    """
    session_data = await recorder.get_session(session_id)

    if session_data is None:
        raise HTTPException(status_code=404, detail=f"Session not found: {session_id}")

    if not include_frames and "frames" in session_data:
        # Return just metadata with frame count
        session_data = {
            "session": session_data.get("session", {}),
            "frame_count": len(session_data.get("frames", [])),
        }

    return session_data


@router.delete("/sessions/{session_id}")
async def delete_session(
    session_id: str,
    recorder: PerimeterRecordingService = Depends(get_perimeter_recorder),
) -> Dict[str, str]:
    """Delete a recording session.

    Permanently removes the session data file from disk.

    Args:
        session_id: UUID of the session to delete.

    Returns:
        Success message.

    Raises:
        HTTPException: If session not found.
    """
    deleted = await recorder.delete_session(session_id)

    if not deleted:
        raise HTTPException(status_code=404, detail=f"Session not found: {session_id}")

    logger.info(f"Deleted recording session: {session_id}")
    return {"status": "deleted", "session_id": session_id}


@router.get("/health", response_model=HealthResponse)
async def health_check(
    recorder: PerimeterRecordingService = Depends(get_perimeter_recorder),
) -> HealthResponse:
    """Get recording service health status.

    Returns health information for monitoring and diagnostics.

    Returns:
        Health status and detailed metrics.
    """
    health = await recorder.health_check()

    return HealthResponse(
        service=health.get("service", "perimeter_recorder"),
        initialized=health.get("initialized", False),
        state=health.get("state", "unknown"),
        details=health,
    )


__all__ = ["router"]
