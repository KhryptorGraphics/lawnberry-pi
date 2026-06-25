"""Settings endpoints (v2).

Exposes the operator-facing settings surface consumed by the Vue dashboard:

- ``/settings``                — the versioned settings profile (optimistic concurrency)
- ``/settings/system``         — system view with HTTP caching (Last-Modified / 304)
- ``/settings/security``       — authentication security level
- ``/settings/maps``           — map provider + API key management
- ``/settings/remote-access``  — remote access provider toggles

Each store is JSON-file backed under the data directory so changes survive a
restart, with an in-memory cache for fast reads. The profile uses semantic
``profile_version`` strings and enforces latency + branding guardrails on write.
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import UTC, datetime
from email.utils import format_datetime, parsedate_to_datetime
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, Response

router = APIRouter()


# --- Persistence helpers -----------------------------------------------------


def _data_dir() -> Path:
    base = os.getenv("LAWNBERRY_DATA_DIR", "./data")
    path = Path(base)
    try:
        path.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    return path


def _load_store(filename: str, default: dict[str, Any]) -> dict[str, Any]:
    path = _data_dir() / filename
    if path.exists():
        try:
            data = json.loads(path.read_text())
            if isinstance(data, dict):
                # Merge over defaults so newly added keys are always present.
                return {**default, **data}
        except Exception:
            pass
    return dict(default)


def _save_store(filename: str, data: dict[str, Any]) -> None:
    path = _data_dir() / filename
    try:
        path.write_text(json.dumps(data, indent=2, default=str))
    except Exception:
        pass


# --- Branding checksum -------------------------------------------------------


def _branding_checksum() -> str:
    """SHA-256 of the canonical branding asset (64 hex chars).

    Used to detect tampering/mismatch of the deployed branding bundle.
    """
    for candidate in ("branding/LawnBerryPi_logo.png", "LawnBerryPi_logo.png"):
        asset = Path(candidate)
        if asset.exists():
            try:
                return hashlib.sha256(asset.read_bytes()).hexdigest()
            except Exception:
                break
    # Stable fallback so the field is always a valid 64-hex string.
    return hashlib.sha256(b"lawnberry-branding").hexdigest()


# --- Settings profile (/settings) -------------------------------------------

_PROFILE_FILE = "settings_profile_v2.json"


def _default_profile() -> dict[str, Any]:
    return {
        "profile_version": "1.0.0",
        "hardware": {
            "platform": "pi5",
            "ai_accelerator": "hailo8l",
        },
        "network": {
            "hostname": "lawnberry.local",
            "backend_port": int(os.getenv("BACKEND_PORT", "8081")),
        },
        "telemetry": {
            "cadence_hz": 5,
            "latency_targets": {"pi5_ms": 250, "pi4b_ms": 350},
        },
        "simulation_mode": os.getenv("SIM_MODE", "0") == "1",
        "ai_acceleration": "hailo8l",
        "branding_checksum": _branding_checksum(),
    }


def _parse_semver(value: Any) -> tuple[int, int, int] | None:
    try:
        major, minor, patch = str(value).split(".")
        return int(major), int(minor), int(patch)
    except Exception:
        return None


@router.get("/settings")
async def get_settings() -> dict[str, Any]:
    profile = _load_store(_PROFILE_FILE, _default_profile())
    # Keep the branding checksum aligned with the current asset.
    profile["branding_checksum"] = _branding_checksum()
    # Provide a category view for clients that group settings by area.
    profile["categories"] = {
        "telemetry": profile.get("telemetry", {}),
        "control": {},
        "maps": _load_store(_MAPS_FILE, _default_maps()),
        "camera": {},
        "ai": {"acceleration": profile.get("ai_acceleration")},
        "system": {"simulation_mode": profile.get("simulation_mode")},
    }
    return profile


@router.put("/settings")
async def put_settings(payload: dict) -> Response:
    current = _load_store(_PROFILE_FILE, _default_profile())
    current_version = current.get("profile_version", "1.0.0")

    incoming_version = payload.get("profile_version")
    incoming = _parse_semver(incoming_version)
    active = _parse_semver(current_version)
    # Optimistic concurrency: an update must carry a strictly newer version.
    if incoming is None or (active is not None and incoming <= active):
        return JSONResponse(
            status_code=409,
            content={
                "error_code": "PROFILE_VERSION_CONFLICT",
                "current_version": current_version,
                "detail": "Submitted profile_version is not newer than the active profile",
            },
        )

    # Latency guardrails: Pi 5 <= 250 ms, Pi 4B <= 350 ms.
    latency = (payload.get("telemetry") or {}).get("latency_targets") or {}
    pi5 = latency.get("pi5_ms")
    pi4b = latency.get("pi4b_ms")
    if (pi5 is not None and pi5 > 250) or (pi4b is not None and pi4b > 350):
        return JSONResponse(
            status_code=422,
            content={
                "error_code": "LATENCY_GUARDRAIL_EXCEEDED",
                "detail": "Telemetry latency targets exceed platform guardrails",
                "limits": {"pi5_ms": 250, "pi4b_ms": 350},
            },
        )

    # Branding integrity.
    expected_checksum = _branding_checksum()
    if payload.get("branding_checksum") not in (None, expected_checksum):
        return JSONResponse(
            status_code=422,
            content={
                "error_code": "BRANDING_ASSET_MISMATCH",
                "detail": "Branding checksum does not match the deployed asset",
                "remediation_url": "/docs/installation-setup-guide.md#branding-assets",
            },
        )

    updated_at = datetime.now(UTC).isoformat()
    stored = {
        **current,
        **payload,
        "branding_checksum": expected_checksum,
        "updated_at": updated_at,
    }
    _save_store(_PROFILE_FILE, stored)
    return JSONResponse(status_code=200, content=stored)


# --- System view with HTTP caching (/settings/system) -----------------------


def _profile_last_modified() -> datetime:
    path = _data_dir() / _PROFILE_FILE
    if path.exists():
        try:
            return datetime.fromtimestamp(path.stat().st_mtime, tz=UTC).replace(microsecond=0)
        except Exception:
            pass
    return datetime.now(UTC).replace(microsecond=0)


@router.get("/settings/system")
async def get_settings_system(request: Request) -> Response:
    profile = _load_store(_PROFILE_FILE, _default_profile())
    last_modified = _profile_last_modified()

    ims = request.headers.get("If-Modified-Since")
    if ims:
        try:
            since = parsedate_to_datetime(ims)
            if since.tzinfo is None:
                since = since.replace(tzinfo=UTC)
            if last_modified <= since:
                return Response(
                    status_code=304,
                    headers={"Last-Modified": format_datetime(last_modified, usegmt=True)},
                )
        except Exception:
            pass

    return JSONResponse(
        status_code=200,
        content=profile,
        headers={"Last-Modified": format_datetime(last_modified, usegmt=True)},
    )


@router.put("/settings/system")
async def put_settings_system(payload: dict) -> Response:
    """Apply a partial system settings update (e.g. telemetry cadence)."""
    profile = _load_store(_PROFILE_FILE, _default_profile())
    for key, value in payload.items():
        if key in profile and isinstance(profile.get(key), dict) and isinstance(value, dict):
            profile[key] = {**profile[key], **value}
        else:
            profile[key] = value
    profile["updated_at"] = datetime.now(UTC).isoformat()
    _save_store(_PROFILE_FILE, profile)
    try:
        from ...core.persistence import persistence

        persistence.add_audit_log("settings.update", details={"changes": list(payload.keys())})
    except Exception:
        pass
    return JSONResponse(status_code=200, content=profile)


# --- Security level (/settings/security) ------------------------------------

_SECURITY_FILE = "settings_security_v2.json"
_SECURITY_LEVELS = {"password_only", "password_totp", "google_auth", "cloudflare_tunnel_auth"}


def _default_security() -> dict[str, Any]:
    return {"level": "password_only"}


@router.get("/settings/security")
async def get_settings_security() -> dict[str, Any]:
    return _load_store(_SECURITY_FILE, _default_security())


@router.put("/settings/security")
async def put_settings_security(payload: dict) -> Response:
    level = payload.get("level")
    if level is not None and level not in _SECURITY_LEVELS:
        return JSONResponse(
            status_code=422,
            content={"detail": f"Unsupported security level: {level}"},
        )
    current = _load_store(_SECURITY_FILE, _default_security())
    updated = {**current, **payload}
    _save_store(_SECURITY_FILE, updated)
    return JSONResponse(status_code=200, content=updated)


# --- Map provider settings (/settings/maps) ---------------------------------

_MAPS_FILE = "settings_maps_v2.json"
_MAP_PROVIDERS = {"google", "osm"}


def _default_maps() -> dict[str, Any]:
    return {"provider": "google", "bypass_external": False, "api_key": None}


@router.get("/settings/maps")
async def get_settings_maps() -> dict[str, Any]:
    return _load_store(_MAPS_FILE, _default_maps())


@router.put("/settings/maps")
async def put_settings_maps(payload: dict) -> Response:
    provider = payload.get("provider")
    if provider is not None and provider not in _MAP_PROVIDERS:
        return JSONResponse(
            status_code=422,
            content={"detail": f"Unsupported map provider: {provider}"},
        )
    current = _load_store(_MAPS_FILE, _default_maps())
    updated = {**current, **payload}
    _save_store(_MAPS_FILE, updated)
    return JSONResponse(status_code=200, content=updated)


# --- Remote access settings (/settings/remote-access) -----------------------

_REMOTE_FILE = "settings_remote_access_v2.json"
_REMOTE_PROVIDERS = {"disabled", "cloudflare", "ngrok", "custom"}


def _default_remote_access() -> dict[str, Any]:
    return {"enabled": False, "provider": "disabled"}


@router.get("/settings/remote-access")
async def get_settings_remote_access() -> dict[str, Any]:
    return _load_store(_REMOTE_FILE, _default_remote_access())


@router.put("/settings/remote-access")
async def put_settings_remote_access(payload: dict) -> Response:
    provider = payload.get("provider")
    if provider is not None and provider not in _REMOTE_PROVIDERS:
        return JSONResponse(
            status_code=422,
            content={"detail": f"Unsupported remote access provider: {provider}"},
        )
    current = _load_store(_REMOTE_FILE, _default_remote_access())
    updated = {**current, **payload}
    _save_store(_REMOTE_FILE, updated)
    return JSONResponse(status_code=200, content=updated)
