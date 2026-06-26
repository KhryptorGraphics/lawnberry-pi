"""Input validation middleware.

Enforces basic constraints before handlers run:
- Content-Type must be application/json for modifying methods
- Maximum body size limit
- Rejects invalid JSON payloads early with clear errors

Note: Detailed schema validation remains with FastAPI/Pydantic models.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable

from fastapi import FastAPI, Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse, Response


class InputValidationMiddleware(BaseHTTPMiddleware):
    def __init__(
        self,
        app: FastAPI,
        *,
        max_body_bytes: int = 1_000_000,
        skip_prefixes: Iterable[str] | None = None,
    ) -> None:
        super().__init__(app)
        self._max_body = max(1024, int(max_body_bytes))
        self._skip = tuple(skip_prefixes or ("/health", "/metrics", "/docs", "/openapi.json"))

    async def dispatch(self, request: Request, call_next) -> Response:
        path = request.url.path
        if any(path.startswith(p) for p in self._skip):
            return await call_next(request)

        if request.method in {"POST", "PUT", "PATCH"}:
            body = await request.body()
            if len(body) > self._max_body:
                return JSONResponse(status_code=413, content={"detail": "Payload too large"})

            if body:
                # Only enforce the JSON contract when a body is actually present.
                # Body-less modifying requests are legitimate (e.g. the safety
                # emergency-stop endpoint) and must not be rejected with 415.
                content_type = (request.headers.get("Content-Type") or "").lower()
                if "application/json" not in content_type:
                    return JSONResponse(
                        status_code=415,
                        content={"detail": "Content-Type must be application/json"},
                    )
                try:
                    payload = json.loads(body.decode("utf-8"))
                except Exception:
                    return JSONResponse(status_code=400, content={"detail": "Invalid JSON payload"})
                forwarded_body = json.dumps(payload).encode("utf-8")
            else:
                forwarded_body = b""

            # Re-provide the (already consumed) body to downstream handlers.
            async def receive() -> dict:
                return {"type": "http.request", "body": forwarded_body}

            request._receive = receive  # type: ignore[attr-defined]

        return await call_next(request)


def register_input_validation_middleware(app: FastAPI) -> None:
    max_body = int(os.getenv("INPUT_MAX_BODY_BYTES", "1000000"))
    skip = os.getenv("INPUT_VALIDATION_SKIP", "/health,/metrics,/docs,/openapi.json")
    app.add_middleware(
        InputValidationMiddleware,
        max_body_bytes=max_body,
        skip_prefixes=[s.strip() for s in skip.split(",") if s.strip()],
    )
