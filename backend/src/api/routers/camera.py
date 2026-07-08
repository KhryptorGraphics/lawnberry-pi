"""Camera status/frame/stream REST routes.

Thin HTTP proxy over the dedicated lawnberry-camera.service's IPC socket --
this router never touches camera hardware directly. Response shapes mirror
the IPC protocol's {"status": ..., "data": ...} envelope verbatim, since the
frontend (ControlView.vue) was already written against that shape.
"""

import base64

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse, StreamingResponse

from ...services import camera_ipc_client
from ..deps import require_operator_auth

router = APIRouter()

_BOUNDARY = "lawnberryframe"


@router.get("/status")
async def camera_status() -> JSONResponse:
    return JSONResponse(await camera_ipc_client.send_command("get_status"))


@router.get("/frame")
async def camera_frame() -> JSONResponse:
    return JSONResponse(await camera_ipc_client.send_command("get_frame"))


@router.post("/start", dependencies=[Depends(require_operator_auth)])
async def camera_start() -> JSONResponse:
    result = await camera_ipc_client.send_command("start_streaming")
    if result.get("status") == "error" and "error" not in result:
        result["error"] = result.get("message", "Failed to start streaming")
    return JSONResponse(result)


@router.get("/stream.mjpeg")
async def camera_stream_mjpeg() -> StreamingResponse:
    async def generate():
        async for frame in camera_ipc_client.frame_stream():
            jpeg_b64 = frame.get("data")
            if not jpeg_b64:
                continue
            try:
                jpeg_bytes = base64.b64decode(jpeg_b64)
            except (ValueError, TypeError):
                continue
            yield (
                f"--{_BOUNDARY}\r\n"
                "Content-Type: image/jpeg\r\n"
                f"Content-Length: {len(jpeg_bytes)}\r\n\r\n"
            ).encode() + jpeg_bytes + b"\r\n"

    return StreamingResponse(
        generate(), media_type=f"multipart/x-mixed-replace; boundary={_BOUNDARY}"
    )
