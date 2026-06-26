"""Thor training-data ingest server.

Runs ON the NVIDIA Thor server and receives the recording sessions uploaded by
the Pi's :class:`backend.src.services.thor_uploader.ThorUploaderService`. It
implements that uploader's chunked protocol exactly:

- ``PUT  /api/upload/chunk``    raw chunk body + ``X-*`` metadata headers
- ``POST /api/upload/finalize`` JSON ``{task_id, session_id}``; reassembles the
  chunks, verifies the ``X-SHA256``, LZ4-decompresses when ``X-Compressed`` is
  true, and writes the session file under the ingest directory.

Config via env:
- ``THOR_INGEST_DIR``  (default ``./data/thor_ingest``)
- ``THOR_INGEST_HOST`` (default ``0.0.0.0``)
- ``THOR_INGEST_PORT`` (default ``8000``)
- ``THOR_API_KEY``     when set, requires ``Authorization: Bearer <key>``
"""

from __future__ import annotations

import json
import os
import shutil
from datetime import UTC, datetime
from pathlib import Path

from fastapi import FastAPI, Header, HTTPException, Request


def _ingest_dir() -> Path:
    base = Path(os.getenv("THOR_INGEST_DIR", "./data/thor_ingest"))
    base.mkdir(parents=True, exist_ok=True)
    return base


def _staging_dir() -> Path:
    d = _ingest_dir() / ".staging"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _sessions_dir() -> Path:
    d = _ingest_dir() / "sessions"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _check_auth(authorization: str | None) -> None:
    key = os.getenv("THOR_API_KEY", "")
    if not key:
        return
    if authorization != f"Bearer {key}":
        raise HTTPException(status_code=401, detail="Unauthorized")


def create_app() -> FastAPI:
    app = FastAPI(title="LawnBerry Thor Ingest", version="1.0.0")

    @app.get("/health")
    async def health() -> dict:
        sessions = list(_sessions_dir().glob("*.manifest.json"))
        return {
            "status": "healthy",
            "service": "thor-ingest",
            "ingest_dir": str(_ingest_dir()),
            "sessions_received": len(sessions),
        }

    @app.put("/api/upload/chunk")
    async def upload_chunk(
        request: Request,
        x_task_id: str = Header(..., alias="X-Task-ID"),
        x_session_id: str = Header(..., alias="X-Session-ID"),
        x_chunk_index: int = Header(..., alias="X-Chunk-Index"),
        authorization: str | None = Header(None),
    ) -> dict:
        _check_auth(authorization)
        body = await request.body()
        part_dir = _staging_dir() / x_task_id
        part_dir.mkdir(parents=True, exist_ok=True)
        # Index-named parts so finalize can reassemble in order (resume-safe).
        (part_dir / f"{x_chunk_index:08d}.part").write_bytes(body)
        return {
            "ok": True,
            "task_id": x_task_id,
            "session_id": x_session_id,
            "chunk_index": x_chunk_index,
            "bytes": len(body),
        }

    @app.post("/api/upload/finalize")
    async def finalize(
        payload: dict,
        x_session_id: str = Header("", alias="X-Session-ID"),
        x_sha256: str = Header("", alias="X-SHA256"),
        x_total_chunks: int = Header(0, alias="X-Total-Chunks"),
        x_compressed: str = Header("False", alias="X-Compressed"),
        authorization: str | None = Header(None),
    ) -> dict:
        _check_auth(authorization)
        task_id = payload.get("task_id")
        session_id = payload.get("session_id") or x_session_id
        if not task_id or not session_id:
            raise HTTPException(status_code=400, detail="task_id and session_id required")

        part_dir = _staging_dir() / task_id
        parts = sorted(part_dir.glob("*.part")) if part_dir.is_dir() else []
        if not parts:
            raise HTTPException(status_code=404, detail="No chunks received for task")
        if x_total_chunks and len(parts) != x_total_chunks:
            raise HTTPException(
                status_code=409,
                detail=f"Expected {x_total_chunks} chunks, received {len(parts)}",
            )

        import hashlib

        digest = hashlib.sha256()
        assembled = part_dir / "assembled.bin"
        with open(assembled, "wb") as out:
            for part in parts:
                data = part.read_bytes()
                digest.update(data)
                out.write(data)
        sha = digest.hexdigest()
        if x_sha256 and sha != x_sha256:
            shutil.rmtree(part_dir, ignore_errors=True)
            raise HTTPException(status_code=422, detail="SHA256 integrity check failed")

        compressed = str(x_compressed).lower() == "true"
        decompressed = False
        final_path = _sessions_dir() / f"{session_id}.msgpack"
        if compressed:
            try:
                import lz4.frame

                with lz4.frame.open(assembled, "rb") as src, open(final_path, "wb") as dst:
                    shutil.copyfileobj(src, dst)
                decompressed = True
            except Exception:
                # lz4 unavailable or decode failed: keep the compressed blob intact.
                final_path = _sessions_dir() / f"{session_id}.msgpack.lz4"
                shutil.copyfile(assembled, final_path)
        else:
            shutil.copyfile(assembled, final_path)

        manifest = {
            "session_id": session_id,
            "task_id": task_id,
            "received_at": datetime.now(UTC).isoformat(),
            "sha256": sha,
            "compressed_upload": compressed,
            "decompressed": decompressed,
            "stored_path": str(final_path),
            "bytes": final_path.stat().st_size,
            "chunks": len(parts),
        }
        (_sessions_dir() / f"{session_id}.manifest.json").write_text(json.dumps(manifest, indent=2))
        shutil.rmtree(part_dir, ignore_errors=True)
        return {"ok": True, **manifest}

    return app


app = create_app()


def main() -> None:
    import uvicorn

    host = os.getenv("THOR_INGEST_HOST", "0.0.0.0")
    port = int(os.getenv("THOR_INGEST_PORT", "8000"))
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":  # pragma: no cover
    main()
