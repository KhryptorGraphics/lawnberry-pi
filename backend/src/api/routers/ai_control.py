"""AI Control API for autonomous mowing (BEAD-302/303).

REST API endpoints for controlling AI-based autonomous mowing.
Provides model loading, inference control, and status monitoring.

Endpoints:
- POST /ai/enable - Enable AI control mode
- POST /ai/disable - Disable AI control mode
- GET /ai/status - Get AI system status
- POST /ai/model - Load a new AI model
- GET /ai/metrics - Get inference performance metrics
- POST /ai/metrics/reset - Reset performance metrics
- GET /ai/health - Health check endpoint
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException, Depends, UploadFile, File
from pydantic import BaseModel, Field

from ...services.ai_inference_service import (
    AIInferenceService,
    get_ai_inference_service,
)
from ...models.action_prediction import (
    AIControlStatus,
    InferenceMetrics,
    ControlMode,
)

logger = logging.getLogger(__name__)
router = APIRouter()


# ----------------------- Request/Response Models -----------------------


class EnableAIRequest(BaseModel):
    """Request to enable AI control."""
    mode: str = Field(
        "ai_autonomous",
        description="Control mode: ai_autonomous or ai_assisted"
    )


class EnableAIResponse(BaseModel):
    """Response after enabling AI."""
    success: bool
    enabled: bool
    mode: str
    message: str


class LoadModelRequest(BaseModel):
    """Request to load a new AI model."""
    model_path: str = Field(..., description="Path to .hef model file")
    model_name: Optional[str] = Field(None, description="Optional model name override")
    model_version: Optional[str] = Field(None, description="Optional version override")


class LoadModelResponse(BaseModel):
    """Response after loading model."""
    success: bool
    model_name: str
    model_version: str
    model_path: str
    hardware_accelerated: bool
    message: str


class MetricsResponse(BaseModel):
    """Response containing inference metrics."""
    total_inferences: int
    successful_inferences: int
    failed_inferences: int
    safety_overrides: int
    avg_inference_time_ms: float
    min_inference_time_ms: float
    max_inference_time_ms: float
    avg_total_time_ms: float
    inferences_per_second: float
    target_fps: float
    meets_target_fps: bool
    success_rate: float
    high_confidence_count: int
    medium_confidence_count: int
    low_confidence_count: int
    uncertain_count: int
    avg_confidence: float
    model_name: str
    model_version: str
    hardware_accelerated: bool


class StatusResponse(BaseModel):
    """Response containing AI control status."""
    enabled: bool
    mode: str
    model_loaded: bool
    last_prediction: Optional[Dict[str, Any]] = None
    prediction_age_ms: float
    hailo_available: bool
    hailo_temperature: Optional[float] = None
    using_hardware: bool
    safety_engaged: bool
    safety_reason: Optional[str] = None
    success_rate: float
    avg_latency_ms: float
    current_fps: float


class HealthResponse(BaseModel):
    """Health check response."""
    healthy: bool
    service: str
    initialized: bool
    running: bool
    enabled: bool
    model_loaded: bool
    hardware_available: bool
    message: str


# ----------------------- Dependency Injection -----------------------


async def get_service() -> AIInferenceService:
    """Get the AI inference service instance."""
    service = get_ai_inference_service()
    if not service.initialized:
        await service.initialize()
        await service.start()
    return service


# ----------------------- API Endpoints -----------------------


@router.post("/enable", response_model=EnableAIResponse)
async def enable_ai_control(
    request: EnableAIRequest,
    service: AIInferenceService = Depends(get_service)
) -> EnableAIResponse:
    """Enable AI control mode.

    Activates AI-based autonomous control. The mower will use AI model
    predictions for steering, throttle, and blade control.

    Args:
        request: Control mode specification.

    Returns:
        Status of AI control enablement.
    """
    try:
        success = await service.enable()

        if not success:
            return EnableAIResponse(
                success=False,
                enabled=False,
                mode="manual",
                message="Failed to enable AI control - service not ready"
            )

        return EnableAIResponse(
            success=True,
            enabled=True,
            mode=request.mode,
            message="AI control enabled successfully"
        )

    except Exception as e:
        logger.error(f"Failed to enable AI control: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/disable", response_model=EnableAIResponse)
async def disable_ai_control(
    service: AIInferenceService = Depends(get_service)
) -> EnableAIResponse:
    """Disable AI control mode.

    Returns control to manual mode. AI inference will stop.

    Returns:
        Confirmation of AI control disabled.
    """
    try:
        await service.disable()

        return EnableAIResponse(
            success=True,
            enabled=False,
            mode="manual",
            message="AI control disabled - manual mode active"
        )

    except Exception as e:
        logger.error(f"Failed to disable AI control: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status", response_model=StatusResponse)
async def get_ai_status(
    service: AIInferenceService = Depends(get_service)
) -> StatusResponse:
    """Get current AI control status.

    Returns comprehensive status including model state, inference metrics,
    and safety status.

    Returns:
        Current AI control status.
    """
    try:
        status = service.get_status()

        return StatusResponse(
            enabled=status.enabled,
            mode=status.mode.value,
            model_loaded=status.model_loaded,
            last_prediction=status.last_prediction,
            prediction_age_ms=status.prediction_age_ms,
            hailo_available=status.hailo_available,
            hailo_temperature=status.hailo_temperature,
            using_hardware=status.using_hardware,
            safety_engaged=status.safety_engaged,
            safety_reason=status.safety_reason,
            success_rate=round(status.success_rate, 4),
            avg_latency_ms=round(status.avg_latency_ms, 2),
            current_fps=round(status.current_fps, 1),
        )

    except Exception as e:
        logger.error(f"Failed to get AI status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/model", response_model=LoadModelResponse)
async def load_model(
    request: LoadModelRequest,
    service: AIInferenceService = Depends(get_service)
) -> LoadModelResponse:
    """Load a new AI model.

    Loads a new VLA model from the specified path. The model must be
    in Hailo HEF format.

    Args:
        request: Model loading specification.

    Returns:
        Status of model loading.
    """
    try:
        model_path = Path(request.model_path)

        if not model_path.exists():
            raise HTTPException(
                status_code=404,
                detail=f"Model file not found: {request.model_path}"
            )

        if not model_path.suffix.lower() == ".hef":
            raise HTTPException(
                status_code=400,
                detail="Model must be in HEF format (.hef extension)"
            )

        success = await service.load_model(model_path)

        if not success:
            raise HTTPException(
                status_code=500,
                detail="Failed to load model"
            )

        health = await service.health_check()

        return LoadModelResponse(
            success=True,
            model_name=health.get("model_name", "unknown"),
            model_version=health.get("model_version", "1.0.0"),
            model_path=str(model_path),
            hardware_accelerated=health.get("hailo", {}).get("hardware_available", False),
            message="Model loaded successfully"
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/metrics", response_model=MetricsResponse)
async def get_metrics(
    service: AIInferenceService = Depends(get_service)
) -> MetricsResponse:
    """Get inference performance metrics.

    Returns detailed statistics about inference performance including
    timing, confidence distribution, and throughput.

    Returns:
        Inference performance metrics.
    """
    try:
        metrics = service.get_metrics()

        return MetricsResponse(
            total_inferences=metrics.total_inferences,
            successful_inferences=metrics.successful_inferences,
            failed_inferences=metrics.failed_inferences,
            safety_overrides=metrics.safety_overrides,
            avg_inference_time_ms=round(metrics.avg_inference_time_ms, 2),
            min_inference_time_ms=round(metrics.min_inference_time_ms, 2) if metrics.min_inference_time_ms != float('inf') else 0.0,
            max_inference_time_ms=round(metrics.max_inference_time_ms, 2),
            avg_total_time_ms=round(metrics.avg_total_time_ms, 2),
            inferences_per_second=round(metrics.inferences_per_second, 2),
            target_fps=metrics.target_fps,
            meets_target_fps=metrics.meets_target_fps,
            success_rate=round(metrics.success_rate, 4),
            high_confidence_count=metrics.high_confidence_count,
            medium_confidence_count=metrics.medium_confidence_count,
            low_confidence_count=metrics.low_confidence_count,
            uncertain_count=metrics.uncertain_count,
            avg_confidence=round(metrics.avg_confidence, 3),
            model_name=metrics.model_name,
            model_version=metrics.model_version,
            hardware_accelerated=metrics.hardware_accelerated,
        )

    except Exception as e:
        logger.error(f"Failed to get metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/metrics/reset")
async def reset_metrics(
    service: AIInferenceService = Depends(get_service)
) -> Dict[str, Any]:
    """Reset inference performance metrics.

    Clears all accumulated statistics and starts fresh.

    Returns:
        Confirmation of metrics reset.
    """
    try:
        service.reset_metrics()

        return {
            "success": True,
            "message": "Metrics reset successfully"
        }

    except Exception as e:
        logger.error(f"Failed to reset metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health", response_model=HealthResponse)
async def health_check(
    service: AIInferenceService = Depends(get_service)
) -> HealthResponse:
    """Health check for AI inference service.

    Returns service health status for monitoring.

    Returns:
        Health check result.
    """
    try:
        health = await service.health_check()

        healthy = (
            health.get("initialized", False) and
            health.get("running", False)
        )

        message = "AI inference service healthy"
        if not health.get("initialized", False):
            message = "Service not initialized"
        elif not health.get("running", False):
            message = "Service not running"
        elif not health.get("model_loaded", False):
            message = "No model loaded (using simulation)"

        return HealthResponse(
            healthy=healthy,
            service="ai_inference",
            initialized=health.get("initialized", False),
            running=health.get("running", False),
            enabled=health.get("enabled", False),
            model_loaded=health.get("model_loaded", False),
            hardware_available=health.get("hailo", {}).get("hardware_available", False),
            message=message,
        )

    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthResponse(
            healthy=False,
            service="ai_inference",
            initialized=False,
            running=False,
            enabled=False,
            model_loaded=False,
            hardware_available=False,
            message=f"Health check failed: {str(e)}",
        )


__all__ = ["router"]
