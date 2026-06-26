"""Weather endpoints (v2).

Thin HTTP layer over :mod:`backend.src.services.weather_service`. The service
prefers on-board BME280 readings, falls back to OpenWeather when configured, and
otherwise reports ``source="unavailable"`` with null measurements.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter

from ...services.weather_service import weather_service

router = APIRouter()


@router.get("/weather/current")
async def weather_current() -> dict[str, Any]:
    """Return the latest environmental snapshot.

    Shape: ``{timestamp, source, temperature_c, humidity_percent, pressure_hpa}``.
    """
    return await weather_service.get_current_async()


@router.get("/weather/planning-advice")
async def weather_planning_advice() -> dict[str, Any]:
    """Return mowing advice derived from the current conditions."""
    current = await weather_service.get_current_async()
    return weather_service.get_planning_advice(current)
