"""Tests for Thor uploader env configuration + enqueue."""

import pytest

from backend.src.services import thor_uploader as tu
from backend.src.services.thor_uploader import ThorUploaderConfig, ThorUploaderService


def test_config_from_env(monkeypatch):
    monkeypatch.setenv("THOR_BASE_URL", "http://192.168.1.64:15454")
    monkeypatch.setenv("THOR_API_KEY", "sekret")
    cfg = ThorUploaderConfig.from_env()
    assert cfg.thor_base_url == "http://192.168.1.64:15454"
    assert cfg.thor_api_key == "sekret"


def test_config_from_env_defaults(monkeypatch):
    monkeypatch.delenv("THOR_BASE_URL", raising=False)
    monkeypatch.delenv("THOR_API_KEY", raising=False)
    cfg = ThorUploaderConfig.from_env()
    assert cfg.thor_base_url  # falls back to the dataclass default
    assert cfg.thor_api_key == ""


def test_get_thor_uploader_uses_env(monkeypatch):
    monkeypatch.setattr(tu, "_uploader_instance", None)
    monkeypatch.setenv("THOR_BASE_URL", "http://thor-host:15454")
    uploader = tu.get_thor_uploader()
    assert uploader.config.thor_base_url == "http://thor-host:15454"


@pytest.mark.asyncio
async def test_queue_upload_enqueues_file(tmp_path):
    f = tmp_path / "session.msgpack"
    f.write_bytes(b"recorded-frames" * 100)
    svc = ThorUploaderService(ThorUploaderConfig(queue_persistence_path=tmp_path / "queue.json"))
    await svc.initialize()
    task_id = await svc.queue_upload(str(f), "sess-xyz")
    assert isinstance(task_id, str) and task_id
    assert any(t.session_id == "sess-xyz" for t in svc._queue.values())


@pytest.mark.asyncio
async def test_queue_upload_missing_file_raises(tmp_path):
    svc = ThorUploaderService(ThorUploaderConfig(queue_persistence_path=tmp_path / "q.json"))
    await svc.initialize()
    with pytest.raises(FileNotFoundError):
        await svc.queue_upload(str(tmp_path / "nope.msgpack"), "s")
