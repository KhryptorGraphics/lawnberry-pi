"""Tests for the Thor ingest server against the Pi uploader's chunk protocol."""

import hashlib
import io

import httpx
import lz4.frame
import pytest

from backend.src.thor_ingest.server import create_app

BASE_URL = "http://thor-test"
CHUNK = 64 * 1024


def _compress(data: bytes) -> bytes:
    buf = io.BytesIO()
    with lz4.frame.open(buf, "wb") as f:
        f.write(data)
    return buf.getvalue()


async def _upload(client, *, task_id, session_id, blob, sha, compressed, headers_extra=None):
    chunks = [blob[i : i + CHUNK] for i in range(0, len(blob), CHUNK)] or [b""]
    base = {
        "X-Task-ID": task_id,
        "X-Session-ID": session_id,
        "X-SHA256": sha,
        "X-Total-Chunks": str(len(chunks)),
        "X-Total-Size": str(len(blob)),
        "X-Compressed": "True" if compressed else "False",
        **(headers_extra or {}),
    }
    for idx, chunk in enumerate(chunks):
        r = await client.put(
            "/api/upload/chunk",
            content=chunk,
            headers={**base, "X-Chunk-Index": str(idx), "X-Chunk-Size": str(len(chunk))},
        )
        if r.status_code != 200:
            return r
    return await client.post(
        "/api/upload/finalize",
        json={"task_id": task_id, "session_id": session_id},
        headers=base,
    )


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("THOR_INGEST_DIR", str(tmp_path))
    monkeypatch.delenv("THOR_API_KEY", raising=False)
    transport = httpx.ASGITransport(app=create_app())
    return httpx.AsyncClient(transport=transport, base_url=BASE_URL), tmp_path


@pytest.mark.asyncio
async def test_compressed_session_roundtrip(client):
    ac, tmp_path = client
    async with ac:
        original = b"mowerframe" * 5000  # ~50KB -> multiple chunks
        blob = _compress(original)
        sha = hashlib.sha256(blob).hexdigest()
        resp = await _upload(
            ac, task_id="t1", session_id="sess-1", blob=blob, sha=sha, compressed=True
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["ok"] is True
        assert body["decompressed"] is True
        stored = tmp_path / "sessions" / "sess-1.msgpack"
        assert stored.exists()
        assert stored.read_bytes() == original  # LZ4 round-trip restored the original


@pytest.mark.asyncio
async def test_uncompressed_session_roundtrip(client):
    ac, tmp_path = client
    async with ac:
        blob = b"raw-msgpack-bytes" * 1000
        sha = hashlib.sha256(blob).hexdigest()
        resp = await _upload(
            ac, task_id="t2", session_id="sess-2", blob=blob, sha=sha, compressed=False
        )
        assert resp.status_code == 200, resp.text
        assert (tmp_path / "sessions" / "sess-2.msgpack").read_bytes() == blob


@pytest.mark.asyncio
async def test_sha256_mismatch_rejected(client):
    ac, _ = client
    async with ac:
        blob = b"corrupt" * 100
        resp = await _upload(
            ac, task_id="t3", session_id="sess-3", blob=blob, sha="deadbeef", compressed=False
        )
        assert resp.status_code == 422


@pytest.mark.asyncio
async def test_finalize_without_chunks_404(client):
    ac, _ = client
    async with ac:
        resp = await ac.post(
            "/api/upload/finalize",
            json={"task_id": "missing", "session_id": "x"},
            headers={"X-Session-ID": "x"},
        )
        assert resp.status_code == 404


@pytest.mark.asyncio
async def test_auth_required_when_key_set(tmp_path, monkeypatch):
    monkeypatch.setenv("THOR_INGEST_DIR", str(tmp_path))
    monkeypatch.setenv("THOR_API_KEY", "secret-key")
    transport = httpx.ASGITransport(app=create_app())
    async with httpx.AsyncClient(transport=transport, base_url=BASE_URL) as ac:
        no_auth = await ac.put(
            "/api/upload/chunk",
            content=b"x",
            headers={"X-Task-ID": "t", "X-Session-ID": "s", "X-Chunk-Index": "0"},
        )
        assert no_auth.status_code == 401

        ok = await ac.put(
            "/api/upload/chunk",
            content=b"x",
            headers={
                "X-Task-ID": "t",
                "X-Session-ID": "s",
                "X-Chunk-Index": "0",
                "Authorization": "Bearer secret-key",
            },
        )
        assert ok.status_code == 200
