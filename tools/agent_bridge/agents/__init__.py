"""Pluggable agent backends: OhMyPi (``omp``, default) and Claude Code (``claude``)."""

from __future__ import annotations

from collections.abc import Mapping

from ..config import AgentConfig
from ..process import SubprocessRunner
from ..redact import Redactor
from .base import (
    AgentBackend,
    AgentReport,
    AgentRequest,
    AgentResult,
    AgentStatus,
    FailureKind,
    extract_report,
)
from .claude import ClaudeBackend
from .omp import OmpBackend

__all__ = [
    "BACKENDS",
    "AgentBackend",
    "AgentReport",
    "AgentRequest",
    "AgentResult",
    "AgentStatus",
    "ClaudeBackend",
    "FailureKind",
    "OmpBackend",
    "build_backend",
    "build_backends",
    "extract_report",
]

BACKENDS: dict[str, type[AgentBackend]] = {
    OmpBackend.name: OmpBackend,
    ClaudeBackend.name: ClaudeBackend,
}


def build_backend(
    name: str,
    cfg: AgentConfig,
    *,
    redactor: Redactor | None = None,
    runner: SubprocessRunner | None = None,
    max_log_bytes: int = 4_000_000,
    environ: Mapping[str, str] | None = None,
) -> AgentBackend:
    """Instantiate one configured backend by name."""
    try:
        cls = BACKENDS[name]
    except KeyError as exc:
        raise ValueError(f"unknown agent backend {name!r}; choose from {sorted(BACKENDS)}") from exc
    return cls(
        cfg.backend(name),
        redactor=redactor,
        runner=runner,
        command_prefix=cfg.command_prefix,
        kill_grace_s=cfg.kill_grace_seconds,
        max_log_bytes=max_log_bytes,
        environ=environ,
    )


def build_backends(
    cfg: AgentConfig,
    *,
    redactor: Redactor | None = None,
    runner: SubprocessRunner | None = None,
    max_log_bytes: int = 4_000_000,
    environ: Mapping[str, str] | None = None,
) -> dict[str, AgentBackend]:
    """Every backend named in ``cfg.allowed``."""
    return {
        name: build_backend(
            name,
            cfg,
            redactor=redactor,
            runner=runner,
            max_log_bytes=max_log_bytes,
            environ=environ,
        )
        for name in cfg.allowed
    }
