"""Bridge configuration: strict YAML schema, safe defaults, credential loading.

Credentials never live in the YAML file.  The GitHub token is read from an environment
variable (default ``LAWNBERRY_BRIDGE_GITHUB_TOKEN``) or from a ``0600`` file.
"""

from __future__ import annotations

import os
import re
import stat
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .errors import ConfigError
from .globs import PathError, normalize_glob

__all__ = [
    "AgentConfig",
    "BackendConfig",
    "BridgeConfig",
    "CheckConfig",
    "DEFAULT_PROTECTED_PATHS",
    "GitConfig",
    "GitHubConfig",
    "LabelsConfig",
    "LimitsConfig",
    "PollConfig",
    "PromptConfig",
    "PullRequestConfig",
    "ScopeConfig",
    "load_config",
    "resolve_github_token",
]

BACKEND_NAMES = ("omp", "claude")
_REPO_RE = re.compile(r"^[A-Za-z0-9_.\-]+/[A-Za-z0-9_.\-]+$")
_LOGIN_RE = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9\-]{0,38})(?:\[bot\])?$")
_ENV_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

# Paths no unattended run may ever change.  Anything touching the mower's safety chain,
# hardware drivers, firmware, deployment, secrets, CI, or the agents' own policy needs a human.
DEFAULT_PROTECTED_PATHS: tuple[str, ...] = (
    "backend/src/safety/**",
    "backend/src/drivers/**",
    "backend/src/services/motor_service.py",
    "backend/src/services/blade_service.py",
    "backend/src/services/robohat_service.py",
    "backend/src/services/tractor_service.py",
    "backend/src/services/thor_link_service.py",
    "backend/src/services/power_service.py",
    "backend/src/core/**",
    "config/hardware.yaml",
    "config/limits.yaml",
    "config/tractor.yaml",
    "config/secrets*.json",
    "config/ssl/**",
    "config/remote_access.json",
    "config/agent-bridge*.yaml",
    "robohat-rp2040-code/**",
    "systemd/**",
    "scripts/**",
    "spec/**",
    "hardware/**",
    ".github/**",
    ".gitmodules",
    ".gitattributes",
    ".env",
    "**/.env",
    "**/.env.*",
    "**/*.pem",
    "**/*.key",
    "**/id_rsa*",
    "AGENTS.md",
    "CLAUDE.md",
    ".claude/**",
    ".omp/**",
    ".agents/**",
    ".cursor/**",
    "docs/constitution.md",
    ".specify/**",
    "tools/agent_bridge/**",
    "pyproject.toml",
    "uv.lock",
    "frontend/package.json",
    "frontend/package-lock.json",
)


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class GitHubConfig(_Strict):
    api_url: str = "https://api.github.com"
    token_env: str = "LAWNBERRY_BRIDGE_GITHUB_TOKEN"
    token_file: Path | None = None
    request_timeout_seconds: float = Field(30.0, gt=0, le=300)
    max_retries: int = Field(3, ge=0, le=10)
    retry_backoff_seconds: float = Field(1.0, ge=0, le=60)
    rate_limit_max_wait_seconds: float = Field(60.0, ge=0, le=900)
    user_agent: str = "lawnberry-agent-bridge/1"

    @field_validator("api_url")
    @classmethod
    def _https(cls, value: str) -> str:
        value = value.rstrip("/")
        if not re.match(r"^https://[A-Za-z0-9.\-]+(:\d+)?(/[A-Za-z0-9._\-/]*)?$", value):
            raise ValueError("api_url must be an https URL")
        return value

    @field_validator("token_env")
    @classmethod
    def _env_name(cls, value: str) -> str:
        if not _ENV_RE.match(value):
            raise ValueError("token_env must be an environment variable name")
        return value


class LabelsConfig(_Strict):
    task: str = "agent-task"
    approved: str = "agent-approved"
    running: str = "agent-running"
    blocked: str = "agent-blocked"
    done: str = "agent-done"

    @model_validator(mode="after")
    def _distinct(self) -> LabelsConfig:
        names = [self.task, self.approved, self.running, self.blocked, self.done]
        if len(set(names)) != len(names) or any(not n.strip() for n in names):
            raise ValueError("label names must be non-empty and distinct")
        return self

    def all(self) -> tuple[str, ...]:
        return (self.task, self.approved, self.running, self.blocked, self.done)


class BackendConfig(_Strict):
    """Settings shared by every agent backend."""

    binary: str
    model: str | None = None
    thinking: Literal["off", "minimal", "low", "medium", "high", "xhigh", "max", "auto"] | None = (
        None
    )
    extra_args: list[str] = Field(default_factory=list)
    pass_env: list[str] = Field(default_factory=list)

    @field_validator("pass_env")
    @classmethod
    def _names(cls, value: list[str]) -> list[str]:
        for name in value:
            if not _ENV_RE.match(name):
                raise ValueError(f"invalid environment variable name: {name!r}")
        return value


class AgentConfig(_Strict):
    default: str = "omp"
    allowed: list[str] = Field(default_factory=lambda: list(BACKEND_NAMES))
    omp: BackendConfig = Field(default_factory=lambda: BackendConfig(binary="omp"))
    claude: BackendConfig = Field(
        default_factory=lambda: BackendConfig(binary="claude", pass_env=["ANTHROPIC_API_KEY"])
    )
    command_prefix: list[str] = Field(default_factory=list)
    kill_grace_seconds: float = Field(10.0, ge=0.1, le=120)
    cancel_poll_seconds: float = Field(15.0, ge=0.05, le=300)

    @model_validator(mode="after")
    def _known(self) -> AgentConfig:
        for name in [self.default, *self.allowed]:
            if name not in BACKEND_NAMES:
                raise ValueError(f"unknown agent backend {name!r}; choose from {BACKEND_NAMES}")
        if self.default not in self.allowed:
            raise ValueError("agent.default must be listed in agent.allowed")
        return self

    def backend(self, name: str) -> BackendConfig:
        return {"omp": self.omp, "claude": self.claude}[name]


class LimitsConfig(_Strict):
    budget_usd: float = Field(2.0, gt=0, le=1000)
    max_budget_usd: float = Field(5.0, gt=0, le=1000)
    timeout_minutes: float = Field(20.0, gt=0, le=24 * 60)
    max_timeout_minutes: float = Field(60.0, gt=0, le=24 * 60)
    max_attempts_per_issue: int = Field(3, ge=1, le=20)
    auto_retry_interrupted: int = Field(1, ge=0, le=5)
    max_publish_attempts: int = Field(5, ge=1, le=20)
    max_start_retries: int = Field(3, ge=0, le=10)
    publish_retry_delays: list[float] = Field(default_factory=lambda: [2.0, 10.0, 30.0])
    lease_ttl_seconds: int = Field(180, ge=5, le=3600)
    max_log_bytes: int = Field(4_000_000, ge=10_000, le=200_000_000)
    max_final_text_chars: int = Field(6000, ge=500, le=100_000)
    max_changed_files: int = Field(40, ge=1, le=1000)
    max_diff_bytes: int = Field(400_000, ge=1000, le=50_000_000)
    max_file_bytes: int = Field(1_000_000, ge=1000, le=100_000_000)

    @model_validator(mode="after")
    def _caps(self) -> LimitsConfig:
        if self.budget_usd > self.max_budget_usd:
            raise ValueError("limits.budget_usd must not exceed limits.max_budget_usd")
        if self.timeout_minutes > self.max_timeout_minutes:
            raise ValueError("limits.timeout_minutes must not exceed limits.max_timeout_minutes")
        return self


class ScopeConfig(_Strict):
    allowed: list[Literal["docs-only", "code"]] = Field(default_factory=lambda: ["docs-only"])
    docs_paths: list[str] = Field(default_factory=lambda: ["docs/**", "*.md"])
    code_paths: list[str] = Field(
        default_factory=lambda: ["docs/**", "*.md", "tools/**", "tests/**"]
    )

    @field_validator("docs_paths", "code_paths")
    @classmethod
    def _globs(cls, value: list[str]) -> list[str]:
        try:
            return [normalize_glob(item) for item in value]
        except PathError as exc:
            raise ValueError(str(exc)) from exc

    @field_validator("allowed")
    @classmethod
    def _non_empty(cls, value: list[str]) -> list[str]:
        if not value:
            raise ValueError("scopes.allowed must list at least one scope")
        return value


class GitConfig(_Strict):
    remote: str = "origin"
    base_branch: str = "main"
    branch_prefix: str = "agent/"
    author_name: str = "LawnBerry Agent Bridge"
    author_email: str = "agent-bridge@users.noreply.github.com"
    push_auth: Literal["ambient", "token"] = "ambient"
    fetch_before_run: bool = True
    keep_worktrees: Literal["never", "on-failure", "always"] = "on-failure"
    git_timeout_seconds: float = Field(120.0, gt=0, le=3600)

    @field_validator("branch_prefix")
    @classmethod
    def _prefix(cls, value: str) -> str:
        if not re.match(r"^[A-Za-z0-9][A-Za-z0-9._\-/]*/$", value):
            raise ValueError("branch_prefix must look like 'agent/' (ends with a slash)")
        return value

    @field_validator("remote", "base_branch")
    @classmethod
    def _ref(cls, value: str) -> str:
        if not re.match(r"^[A-Za-z0-9][A-Za-z0-9._\-/]*$", value) or ".." in value:
            raise ValueError("invalid git remote or branch name")
        return value


class PullRequestConfig(_Strict):
    # Draft pull requests are not optional: the bridge never opens a ready-for-review PR.
    draft: Literal[True] = True
    closing_keyword: Literal["Refs", "Closes", "Fixes"] = "Refs"
    title_prefix: str = "[agent] "
    labels: list[str] = Field(default_factory=list)


class PollConfig(_Strict):
    interval_seconds: float = Field(60.0, ge=1, le=3600)
    max_jobs_per_poll: int = Field(1, ge=1, le=10)
    error_backoff_seconds: float = Field(120.0, ge=1, le=3600)


class CheckConfig(_Strict):
    """A named verification command run by the bridge itself (never taken from an issue)."""

    argv: list[str] = Field(min_length=1)
    timeout_seconds: float = Field(600.0, gt=0, le=7200)
    env: dict[str, str] = Field(default_factory=dict)

    @field_validator("argv")
    @classmethod
    def _argv(cls, value: list[str]) -> list[str]:
        if any(not isinstance(item, str) or not item for item in value):
            raise ValueError("argv entries must be non-empty strings")
        return value


class PromptConfig(_Strict):
    context_files: list[str] = Field(
        default_factory=lambda: ["CLAUDE.md", ".github/copilot-instructions.md"]
    )
    max_context_chars: int = Field(20_000, ge=0, le=200_000)


class BridgeConfig(_Strict):
    repository: str
    repo_path: Path | None = None
    state_dir: Path = Path("~/.local/state/lawnberry-agent-bridge")
    # Dry run is the default.  Real execution needs dry_run: false AND --execute on the CLI.
    dry_run: bool = True
    approvers: list[str]
    task_authors: list[str] | None = None
    dedupe_by_content: bool = True
    labels: LabelsConfig = Field(default_factory=LabelsConfig)
    github: GitHubConfig = Field(default_factory=GitHubConfig)
    git: GitConfig = Field(default_factory=GitConfig)
    pull_request: PullRequestConfig = Field(default_factory=PullRequestConfig)
    poll: PollConfig = Field(default_factory=PollConfig)
    agent: AgentConfig = Field(default_factory=AgentConfig)
    limits: LimitsConfig = Field(default_factory=LimitsConfig)
    scopes: ScopeConfig = Field(default_factory=ScopeConfig)
    protected_paths: list[str] = Field(default_factory=lambda: list(DEFAULT_PROTECTED_PATHS))
    checks: dict[str, CheckConfig] = Field(default_factory=dict)
    default_checks: list[str] = Field(default_factory=list)
    prompt: PromptConfig = Field(default_factory=PromptConfig)
    extra_secret_env: list[str] = Field(default_factory=list)

    @field_validator("repository")
    @classmethod
    def _repo(cls, value: str) -> str:
        if not _REPO_RE.match(value):
            raise ValueError("repository must look like 'owner/name'")
        return value

    @field_validator("approvers", "task_authors")
    @classmethod
    def _logins(cls, value: list[str] | None) -> list[str] | None:
        if value is None:
            return value
        for login in value:
            if not _LOGIN_RE.match(login):
                raise ValueError(f"invalid GitHub login: {login!r}")
        return value

    @field_validator("approvers")
    @classmethod
    def _approvers_present(cls, value: list[str]) -> list[str]:
        if not value:
            raise ValueError("approvers must name at least one GitHub login (fail closed)")
        return value

    @field_validator("protected_paths")
    @classmethod
    def _protected(cls, value: list[str]) -> list[str]:
        try:
            return [normalize_glob(item) for item in value]
        except PathError as exc:
            raise ValueError(str(exc)) from exc

    @field_validator("extra_secret_env")
    @classmethod
    def _secret_env(cls, value: list[str]) -> list[str]:
        for name in value:
            if not _ENV_RE.match(name):
                raise ValueError(f"invalid environment variable name: {name!r}")
        return value

    @model_validator(mode="after")
    def _consistency(self) -> BridgeConfig:
        for name in self.default_checks:
            if name not in self.checks:
                raise ValueError(f"default_checks references undefined check {name!r}")
        return self

    @property
    def trusted_authors(self) -> list[str]:
        """Logins whose issues may become tasks (defaults to the approvers)."""
        return self.task_authors if self.task_authors is not None else self.approvers

    @property
    def owner(self) -> str:
        return self.repository.split("/", 1)[0]

    @property
    def repo_name(self) -> str:
        return self.repository.split("/", 1)[1]

    def resolved_state_dir(self) -> Path:
        return Path(os.path.expandvars(str(self.state_dir))).expanduser().resolve()


def load_config(path: str | os.PathLike[str]) -> BridgeConfig:
    """Load and validate a bridge configuration file.

    Relative ``repo_path``/``state_dir``/``token_file`` values are resolved against the
    directory holding the config file.
    """
    cfg_path = Path(path).expanduser()
    try:
        raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ConfigError(f"config file not found: {cfg_path}") from exc
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
        raise ConfigError(f"cannot read config {cfg_path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigError(f"config {cfg_path} must be a YAML mapping")
    base = cfg_path.resolve().parent
    _absolutize(raw, ("repo_path",), base)
    _absolutize(raw, ("state_dir",), base)
    if isinstance(raw.get("github"), dict):
        _absolutize(raw["github"], ("token_file",), base)
    try:
        return BridgeConfig.model_validate(raw)
    except Exception as exc:  # pydantic.ValidationError carries a readable summary
        raise ConfigError(f"invalid config {cfg_path}: {exc}") from exc


def _absolutize(section: dict[str, Any], keys: tuple[str, ...], base: Path) -> None:
    for key in keys:
        value = section.get(key)
        if isinstance(value, str) and value and not value.startswith(("~", "/", "$")):
            section[key] = str((base / value).resolve())


def resolve_github_token(cfg: GitHubConfig, env: Mapping[str, str] | None = None) -> str:
    """Return the GitHub token from the environment or a private file; never log it."""
    env = os.environ if env is None else env
    token = (env.get(cfg.token_env) or "").strip()
    if token:
        return token
    if cfg.token_file is not None:
        path = Path(cfg.token_file).expanduser()
        try:
            mode = path.stat().st_mode
        except OSError as exc:
            raise ConfigError(f"cannot read GitHub token file {path}: {exc}") from exc
        if mode & (stat.S_IRWXG | stat.S_IRWXO):
            raise ConfigError(f"GitHub token file {path} must not be accessible by group/others")
        token = path.read_text(encoding="utf-8").strip()
        if token:
            return token
    raise ConfigError(
        f"no GitHub token: set ${cfg.token_env} (fine-grained token) or github.token_file"
    )
