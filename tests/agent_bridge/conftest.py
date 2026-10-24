"""Fixtures for the agent-bridge tests.

Every worker-level test runs once per backend (``omp`` and ``claude``) through the real
backend classes, driving fake CLIs that reproduce the real tools' flags and output formats.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from bridgefakes.rig import Rig, make_git_env, make_rig

BACKENDS = ["omp", "claude"]


@pytest.fixture(params=BACKENDS)
def backend_name(request: pytest.FixtureRequest) -> str:
    return str(request.param)


@pytest.fixture
def rig(tmp_path: Path, backend_name: str) -> Iterator[Rig]:
    instance = make_rig(tmp_path, backend_name)
    try:
        yield instance
    finally:
        instance.close()


@pytest.fixture
def make_rig_for(tmp_path: Path, backend_name: str):
    """Factory for rigs with config overrides (cleans up automatically)."""
    created: list[Rig] = []

    def factory(**overrides):
        instance = make_rig(tmp_path, backend_name, **overrides)
        created.append(instance)
        return instance

    yield factory
    for instance in created:
        instance.close()


@pytest.fixture
def git_env_only(tmp_path: Path):
    return make_git_env(tmp_path)
