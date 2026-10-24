"""Tests for ``tools/agent_bridge/config.py``: safe defaults, strict schema, credential loading."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pydantic
import pytest
import yaml

from tools.agent_bridge.config import (
    DEFAULT_PROTECTED_PATHS,
    BridgeConfig,
    GitHubConfig,
    load_config,
    resolve_github_token,
)
from tools.agent_bridge.errors import ConfigError
from tools.agent_bridge.globs import match_any, normalize_glob
from tools.agent_bridge.redact import Redactor

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = REPO_ROOT / "config" / "agent-bridge.example.yaml"
TOKEN = "ghp_" + "A1b2C3d4E5f6G7h8I9j0K1l2M3n4O5p6Q7r8"


def make(**overrides: Any) -> BridgeConfig:
    data: dict[str, Any] = {"repository": "owner/repo", "approvers": ["alice"], **overrides}
    return BridgeConfig.model_validate(data)


def rejects(**overrides: Any) -> pydantic.ValidationError:
    with pytest.raises(pydantic.ValidationError) as info:
        make(**overrides)
    return info.value


# --------------------------------------------------------------------------- safe defaults
def test_defaults_are_safe() -> None:
    cfg = make()

    assert cfg.dry_run is True
    assert cfg.pull_request.draft is True
    assert cfg.scopes.allowed == ["docs-only"]
    assert cfg.agent.default == "omp" and cfg.agent.allowed == ["omp", "claude"]
    assert cfg.git.push_auth == "ambient"
    assert cfg.github.api_url == "https://api.github.com"
    assert cfg.poll.max_jobs_per_poll == 1
    assert cfg.limits.budget_usd <= cfg.limits.max_budget_usd
    assert cfg.limits.timeout_minutes <= cfg.limits.max_timeout_minutes
    assert cfg.dedupe_by_content is True
    assert cfg.checks == {} and cfg.default_checks == []
    assert cfg.protected_paths == list(DEFAULT_PROTECTED_PATHS)
    assert cfg.labels.all() == (
        "agent-task",
        "agent-approved",
        "agent-running",
        "agent-blocked",
        "agent-done",
    )


def test_owner_and_repo_name_are_derived_from_the_repository() -> None:
    cfg = make(repository="KhryptorGraphics/lawnberry-pi")
    assert (cfg.owner, cfg.repo_name) == ("KhryptorGraphics", "lawnberry-pi")


@pytest.mark.parametrize("value", ["maybe", "yes please", [], {}])
def test_dry_run_must_be_a_real_boolean(value: Any) -> None:
    rejects(dry_run=value)


def test_dry_run_can_only_be_switched_off_explicitly() -> None:
    assert make(dry_run=False).dry_run is False


@pytest.mark.parametrize("value", [False, "false", 0, None, "no"])
def test_draft_pull_requests_cannot_be_disabled(value: Any) -> None:
    rejects(pull_request={"draft": value})


def test_draft_stays_true_after_validation_too() -> None:
    cfg = make()
    with pytest.raises(pydantic.ValidationError):
        cfg.pull_request.draft = False  # type: ignore[assignment]
    assert cfg.pull_request.draft is True
    assert make(pull_request={"draft": True}).pull_request.draft is True


def test_assignments_are_validated_like_the_initial_values() -> None:
    cfg = make()
    with pytest.raises(pydantic.ValidationError):
        cfg.dry_run = "perhaps"  # type: ignore[assignment]
    with pytest.raises(pydantic.ValidationError):
        cfg.limits.budget_usd = -1
    with pytest.raises(pydantic.ValidationError):
        cfg.github.api_url = "http://api.github.com"


# --------------------------------------------------------------------------- approvers & authors
def test_approvers_and_repository_are_required() -> None:
    with pytest.raises(pydantic.ValidationError):
        BridgeConfig.model_validate({"repository": "owner/repo"})
    with pytest.raises(pydantic.ValidationError):
        BridgeConfig.model_validate({"approvers": ["alice"]})


def test_an_empty_approver_list_fails_closed() -> None:
    assert "at least one" in str(rejects(approvers=[]))


@pytest.mark.parametrize(
    "login", ["bad login", "-alice", "alice_", "a" * 40, "", "al/ice", "alice\n", "ali ce[bot]"]
)
def test_invalid_github_logins_are_rejected(login: str) -> None:
    if login == "alice\n":
        pytest.skip("trailing newline handling is covered separately")
    rejects(approvers=[login])
    rejects(approvers=["alice"], task_authors=[login])


@pytest.mark.parametrize("login", ["alice", "a", "A-b-1", "a" * 39, "dependabot[bot]", "0day"])
def test_valid_github_logins_are_accepted(login: str) -> None:
    assert make(approvers=[login]).approvers == [login]


def test_trusted_authors_default_to_the_approvers() -> None:
    assert make(approvers=["alice", "bob"]).trusted_authors == ["alice", "bob"]
    assert make(approvers=["alice"], task_authors=None).trusted_authors == ["alice"]


def test_explicit_task_authors_replace_the_approvers() -> None:
    cfg = make(approvers=["alice"], task_authors=["carol"])
    assert cfg.trusted_authors == ["carol"]


def test_an_explicit_empty_author_list_trusts_nobody() -> None:
    assert make(approvers=["alice"], task_authors=[]).trusted_authors == []


@pytest.mark.parametrize("repository", ["owner", "a/b/c", "owner/", "/repo", "own er/repo", ""])
def test_repository_must_look_like_owner_slash_name(repository: str) -> None:
    rejects(repository=repository)


@pytest.mark.parametrize("repository", ["owner/repo", "a-b/c_d.e", "KhryptorGraphics/lawnberry-pi"])
def test_valid_repository_names_are_accepted(repository: str) -> None:
    assert make(repository=repository).repository == repository


# --------------------------------------------------------------------------- strictness
SECTION_BASES: dict[str, dict[str, Any]] = {
    "labels": {},
    "github": {},
    "git": {},
    "pull_request": {},
    "poll": {},
    "agent": {},
    "limits": {},
    "scopes": {},
    "prompt": {},
}


@pytest.mark.parametrize("section", sorted(SECTION_BASES))
def test_unknown_keys_are_rejected_in_every_section(section: str) -> None:
    error = rejects(**{section: {"bogus_option": 1}})
    assert "bogus_option" in str(error)


def test_unknown_keys_are_rejected_at_the_top_level_and_in_nested_models() -> None:
    assert "bogus_option" in str(rejects(bogus_option=1))
    for backend in ("omp", "claude"):
        assert "bogus_option" in str(
            rejects(agent={backend: {"binary": backend, "bogus_option": 1}})
        )
    assert "bogus_option" in str(rejects(checks={"lint": {"argv": ["ruff"], "bogus_option": 1}}))


@pytest.mark.parametrize(
    ("first", "second"),
    [
        (a, b)
        for index, a in enumerate(["task", "approved", "running", "blocked", "done"])
        for b in ["task", "approved", "running", "blocked", "done"][index + 1 :]
    ],
)
def test_label_names_must_be_distinct(first: str, second: str) -> None:
    labels = {"task": "t", "approved": "a", "running": "r", "blocked": "b", "done": "d"}
    labels[second] = labels[first]
    assert "distinct" in str(rejects(labels=labels))


@pytest.mark.parametrize("blank", ["", "   "])
def test_label_names_must_not_be_blank(blank: str) -> None:
    rejects(labels={"approved": blank})


# --------------------------------------------------------------------------- limits
def test_budget_must_not_exceed_the_maximum() -> None:
    assert "max_budget_usd" in str(rejects(limits={"budget_usd": 10, "max_budget_usd": 5}))
    assert make(limits={"budget_usd": 5, "max_budget_usd": 5}).limits.budget_usd == 5


def test_timeout_must_not_exceed_the_maximum() -> None:
    assert "max_timeout_minutes" in str(
        rejects(limits={"timeout_minutes": 90, "max_timeout_minutes": 60})
    )
    assert (
        make(limits={"timeout_minutes": 60, "max_timeout_minutes": 60}).limits.timeout_minutes == 60
    )


@pytest.mark.parametrize(
    "limits",
    [
        {"budget_usd": 0},
        {"budget_usd": -1},
        {"max_budget_usd": 1001},
        {"timeout_minutes": 0},
        {"max_timeout_minutes": 24 * 60 + 1},
        {"max_attempts_per_issue": 0},
        {"auto_retry_interrupted": -1},
        {"lease_ttl_seconds": 4},
        {"max_changed_files": 0},
        {"max_diff_bytes": 10},
        {"max_file_bytes": 10},
        {"max_log_bytes": 5},
    ],
)
def test_out_of_range_limits_are_rejected(limits: dict[str, Any]) -> None:
    rejects(limits=limits)


# --------------------------------------------------------------------------- checks & agents
def test_default_checks_must_reference_defined_checks() -> None:
    error = rejects(default_checks=["pytest"])
    assert "pytest" in str(error)
    cfg = make(checks={"pytest": {"argv": ["pytest", "-q"]}}, default_checks=["pytest"])
    assert cfg.default_checks == ["pytest"]
    assert cfg.checks["pytest"].timeout_seconds > 0


@pytest.mark.parametrize(
    "check",
    [
        {"argv": []},
        {"argv": [""]},
        {"argv": ["pytest", ""]},
        {"argv": ["pytest", 5]},
        {"argv": "pytest -q"},
        {},
        {"argv": ["x"], "timeout_seconds": 0},
        {"argv": ["x"], "timeout_seconds": 7201},
        {"argv": ["x"], "env": {"A": 1}},
    ],
)
def test_malformed_checks_are_rejected(check: dict[str, Any]) -> None:
    rejects(checks={"c": check})


def test_default_agent_must_be_allowed_and_known() -> None:
    assert "agent.allowed" in str(rejects(agent={"default": "claude", "allowed": ["omp"]}))
    assert "unknown agent backend" in str(rejects(agent={"default": "gemini", "allowed": ["omp"]}))
    assert "unknown agent backend" in str(rejects(agent={"allowed": ["omp", "gemini"]}))
    rejects(agent={"allowed": []})


def test_agent_backend_lookup_and_defaults() -> None:
    cfg = make(agent={"default": "claude"})
    assert cfg.agent.default == "claude"
    assert cfg.agent.backend("omp").binary == "omp"
    assert cfg.agent.backend("claude").binary == "claude"
    assert cfg.agent.backend("claude").pass_env == ["ANTHROPIC_API_KEY"]
    with pytest.raises(KeyError):
        cfg.agent.backend("gemini")


@pytest.mark.parametrize("name", ["1BAD", "BAD-NAME", "has space", "", "A=B", "$HOME"])
def test_environment_variable_names_are_validated(name: str) -> None:
    rejects(github={"token_env": name})
    rejects(agent={"claude": {"binary": "claude", "pass_env": [name]}})
    rejects(extra_secret_env=[name])


def test_valid_environment_variable_names_are_accepted() -> None:
    cfg = make(
        github={"token_env": "MY_TOKEN_1"},
        agent={"claude": {"binary": "claude", "pass_env": ["_A", "b2"]}},
        extra_secret_env=["DB_PASSWORD"],
    )
    assert cfg.github.token_env == "MY_TOKEN_1"
    assert cfg.extra_secret_env == ["DB_PASSWORD"]


# --------------------------------------------------------------------------- github / git
@pytest.mark.parametrize(
    "url",
    [
        "http://api.github.com",
        "ftp://api.github.com",
        "api.github.com",
        "https://",
        "https://user:pw@api.github.com",
        "https://evil.example@api.github.com",
        "https://api.github.com?x=1",
        "https://api.github.com#frag",
        "https://api github.com",
        "HTTPS://api.github.com",
        "https://api.github.com/a b",
        "",
    ],
)
def test_api_url_must_be_plain_https(url: str) -> None:
    assert "https" in str(rejects(github={"api_url": url}))


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("https://api.github.com", "https://api.github.com"),
        ("https://api.github.com/", "https://api.github.com"),
        ("https://ghe.example.com/api/v3", "https://ghe.example.com/api/v3"),
        ("https://localhost:8443", "https://localhost:8443"),
        ("https://api.github.test/", "https://api.github.test"),
    ],
)
def test_https_api_urls_are_accepted_and_normalised(url: str, expected: str) -> None:
    assert make(github={"api_url": url}).github.api_url == expected


@pytest.mark.parametrize("prefix", ["agent/", "bots/agent/", "a-b_c.d/", "x1/"])
def test_valid_branch_prefixes(prefix: str) -> None:
    assert make(git={"branch_prefix": prefix}).git.branch_prefix == prefix


@pytest.mark.parametrize(
    "prefix", ["agent", "/agent/", "-agent/", "ag ent/", "agent\\", "", "a;b/"]
)
def test_invalid_branch_prefixes(prefix: str) -> None:
    assert "branch_prefix" in str(rejects(git={"branch_prefix": prefix}))


@pytest.mark.parametrize(
    "value", ["--upload-pack=x", "-x", "a..b", "origin; rm -rf /", "a b", "", "$(id)", "a\x00b"]
)
def test_git_remote_and_base_branch_cannot_smuggle_options(value: str) -> None:
    rejects(git={"remote": value})
    rejects(git={"base_branch": value})


def test_git_literals_are_enforced() -> None:
    rejects(git={"push_auth": "password"})
    rejects(git={"keep_worktrees": "sometimes"})
    rejects(pull_request={"closing_keyword": "Resolves"})
    assert make(git={"push_auth": "token"}).git.push_auth == "token"
    assert make(pull_request={"closing_keyword": "Closes"}).pull_request.closing_keyword == "Closes"


# --------------------------------------------------------------------------- scopes
def test_scopes_must_list_at_least_one_known_scope() -> None:
    rejects(scopes={"allowed": []})
    rejects(scopes={"allowed": ["everything"]})
    assert make(scopes={"allowed": ["docs-only", "code"]}).scopes.allowed == ["docs-only", "code"]


def test_scope_globs_are_normalised_and_validated() -> None:
    cfg = make(scopes={"docs_paths": ["./docs/**", "  *.md "]})
    assert cfg.scopes.docs_paths == ["docs/**", "*.md"]
    for bad in ("/etc/**", "../**", "~/x", "a\\b", ""):
        rejects(scopes={"docs_paths": [bad]})
        rejects(scopes={"code_paths": [bad]})


def test_protected_paths_are_validated_like_globs() -> None:
    assert make(protected_paths=["./secret/**"]).protected_paths == ["secret/**"]
    for bad in ("/etc/**", "../x", "a\\b", ""):
        rejects(protected_paths=[bad])


# --------------------------------------------------------------------------- protected defaults
REQUIRED_PROTECTED_ENTRIES: dict[str, list[str]] = {
    "safety chain": [
        "backend/src/safety/**",
        "backend/src/core/**",
        "backend/src/services/motor_service.py",
        "backend/src/services/blade_service.py",
        "backend/src/services/robohat_service.py",
        "backend/src/services/tractor_service.py",
        "backend/src/services/thor_link_service.py",
        "backend/src/services/power_service.py",
    ],
    "hardware": [
        "backend/src/drivers/**",
        "config/hardware.yaml",
        "config/limits.yaml",
        "config/tractor.yaml",
        "robohat-rp2040-code/**",
        "hardware/**",
    ],
    "deployment": ["systemd/**", "scripts/**", "spec/**"],
    "secrets": [
        "config/secrets*.json",
        "config/ssl/**",
        "config/remote_access.json",
        ".env",
        "**/.env",
        "**/.env.*",
        "**/*.pem",
        "**/*.key",
        "**/id_rsa*",
    ],
    "ci": [".github/**"],
    "agent policy": [
        "AGENTS.md",
        "CLAUDE.md",
        ".claude/**",
        ".omp/**",
        ".agents/**",
        ".cursor/**",
        ".specify/**",
        "docs/constitution.md",
        "config/agent-bridge*.yaml",
        "tools/agent_bridge/**",
    ],
    "dependencies": [
        "pyproject.toml",
        "uv.lock",
        "frontend/package.json",
        "frontend/package-lock.json",
    ],
    "git configuration": [".gitmodules", ".gitattributes"],
}


@pytest.mark.parametrize("category", sorted(REQUIRED_PROTECTED_ENTRIES))
def test_default_protected_paths_name_the_critical_areas(category: str) -> None:
    missing = set(REQUIRED_PROTECTED_ENTRIES[category]) - set(DEFAULT_PROTECTED_PATHS)
    assert not missing, f"{category}: {sorted(missing)}"


def test_default_protected_paths_are_clean() -> None:
    assert len(set(DEFAULT_PROTECTED_PATHS)) == len(DEFAULT_PROTECTED_PATHS)
    for glob in DEFAULT_PROTECTED_PATHS:
        assert normalize_glob(glob) == glob


PROTECTED_SAMPLES = [
    "backend/src/safety/estop.py",
    "backend/src/safety/deeper/nested.py",
    "backend/src/drivers/motor/driver.py",
    "backend/src/core/config.py",
    "backend/src/services/motor_service.py",
    "backend/src/services/blade_service.py",
    "backend/src/services/power_service.py",
    "config/hardware.yaml",
    "config/limits.yaml",
    "config/secrets.json",
    "config/secrets.prod.json",
    "config/ssl/server.crt",
    "config/remote_access.json",
    "config/agent-bridge.yaml",
    "config/agent-bridge.example.yaml",
    "robohat-rp2040-code/code.py",
    "systemd/lawnberry.service",
    "scripts/deploy.sh",
    ".github/workflows/ci.yml",
    ".env",
    "a/b/.env",
    "a/.env.local",
    "deploy/server.pem",
    "x/y/tls.key",
    "home/.ssh/id_rsa",
    "id_rsa.pub",
    "AGENTS.md",
    "CLAUDE.md",
    ".claude/settings.json",
    ".omp/config.yml",
    "docs/constitution.md",
    ".specify/memory/constitution.md",
    "tools/agent_bridge/worker.py",
    "tools/agent_bridge/omp_guard.mjs",
    "pyproject.toml",
    "uv.lock",
    "frontend/package.json",
]
WRITABLE_SAMPLES = [
    "docs/guide.md",
    "docs/deep/er/page.md",
    "README.md",
    "CHANGELOG.md",
    "tests/unit/test_example.py",
    "tools/other/helper.py",
    "backend/src/services/weather_service.py",
]


@pytest.mark.parametrize("path", PROTECTED_SAMPLES)
def test_critical_paths_are_protected_by_default(path: str) -> None:
    assert match_any(path, DEFAULT_PROTECTED_PATHS)


@pytest.mark.parametrize("path", WRITABLE_SAMPLES)
def test_ordinary_paths_are_not_protected_by_default(path: str) -> None:
    assert not match_any(path, DEFAULT_PROTECTED_PATHS)


def test_every_tracked_safety_critical_file_of_this_repository_is_protected() -> None:
    git = shutil.which("git")
    if git is None:
        pytest.skip("git is not installed")
    critical = [
        "backend/src/safety",
        "backend/src/drivers",
        "backend/src/core",
        "systemd",
        "scripts",
        "hardware",
        "robohat-rp2040-code",
        ".github",
        ".specify",
        "spec",
        "config/hardware.yaml",
        "config/limits.yaml",
        "config/tractor.yaml",
        "config/remote_access.json",
        "AGENTS.md",
        "CLAUDE.md",
        "docs/constitution.md",
        "pyproject.toml",
    ]
    done = subprocess.run(
        [git, "ls-files", "-z", "--", *critical],
        cwd=REPO_ROOT,
        capture_output=True,
        check=False,
        timeout=60,
    )
    files = [n for n in done.stdout.decode("utf-8", errors="replace").split("\0") if n]
    if done.returncode != 0 or not files:
        pytest.skip("no tracked files found (not a git checkout?)")
    unprotected = [n for n in files if not match_any(n, DEFAULT_PROTECTED_PATHS)]
    assert unprotected == []


# --------------------------------------------------------------------------- state dir
def test_resolved_state_dir_expands_home_and_variables(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("BRIDGE_STATE_ROOT", str(tmp_path / "var"))

    assert make(state_dir="~/state").resolved_state_dir() == (tmp_path / "state").resolve()
    assert (
        make(state_dir="$BRIDGE_STATE_ROOT/x").resolved_state_dir()
        == (tmp_path / "var" / "x").resolve()
    )
    assert make().resolved_state_dir().is_absolute()


# --------------------------------------------------------------------------- load_config
def write_config(directory: Path, text: str, name: str = "bridge.yaml") -> Path:
    path = directory / name
    path.write_text(text, encoding="utf-8")
    return path


MINIMAL = "repository: owner/repo\napprovers: [alice]\n"


def test_load_config_reads_a_minimal_file(tmp_path: Path) -> None:
    cfg = load_config(write_config(tmp_path, MINIMAL))
    assert cfg.repository == "owner/repo" and cfg.approvers == ["alice"]
    assert cfg.dry_run is True


def test_load_config_accepts_str_paths_and_expands_the_home_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = write_config(tmp_path, MINIMAL)
    assert load_config(str(path)).repository == "owner/repo"
    monkeypatch.setenv("HOME", str(tmp_path))
    assert load_config("~/bridge.yaml").repository == "owner/repo"


def test_relative_paths_resolve_against_the_config_directory_not_the_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    conf_dir = tmp_path / "conf"
    conf_dir.mkdir()
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    path = write_config(
        conf_dir,
        MINIMAL + "repo_path: ../repo\nstate_dir: state\ngithub:\n  token_file: secrets/token\n",
    )
    monkeypatch.chdir(elsewhere)

    cfg = load_config(path)

    assert cfg.repo_path == (tmp_path / "repo").resolve()
    assert cfg.state_dir == (conf_dir / "state").resolve()
    assert cfg.github.token_file == (conf_dir / "secrets" / "token").resolve()


def test_absolute_home_and_variable_paths_are_left_alone(tmp_path: Path) -> None:
    path = write_config(
        tmp_path,
        MINIMAL + "repo_path: /abs/repo\nstate_dir: $HOME/st\ngithub:\n  token_file: ~/tok\n",
    )
    cfg = load_config(path)
    assert cfg.repo_path == Path("/abs/repo")
    assert cfg.state_dir == Path("$HOME/st")
    assert cfg.github.token_file == Path("~/tok")


def test_missing_config_file_is_a_config_error(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "absent.yaml")


def test_a_directory_is_not_a_config_file(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="cannot read config"):
        load_config(tmp_path)


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("repository: [unclosed\n", "cannot read config"),
        ("repository: !!python/object:os.system x\n", "cannot read config"),
        ("", "must be a YAML mapping"),
        ("- a\n- b\n", "must be a YAML mapping"),
        ("just a string\n", "must be a YAML mapping"),
        ("repository: owner/repo\n", "invalid config"),
        ("repository: owner/repo\napprovers: []\n", "invalid config"),
        ("repository: owner/repo\napprovers: [a]\nbogus: 1\n", "invalid config"),
        ("repository: owner/repo\napprovers: [a]\ngithub: null\n", "invalid config"),
        (
            "repository: owner/repo\napprovers: [a]\npull_request:\n  draft: false\n",
            "invalid config",
        ),
    ],
)
def test_bad_files_become_config_errors(tmp_path: Path, text: str, message: str) -> None:
    path = write_config(tmp_path, text)
    with pytest.raises(ConfigError, match=message) as info:
        load_config(path)
    assert str(path) in str(info.value)


def test_validation_errors_name_the_offending_field_and_keep_the_cause(tmp_path: Path) -> None:
    path = write_config(tmp_path, MINIMAL + "limits:\n  budget_usd: 99\n  max_budget_usd: 5\n")
    with pytest.raises(ConfigError, match="budget_usd") as info:
        load_config(path)
    assert isinstance(info.value.__cause__, pydantic.ValidationError)


def test_a_config_file_that_is_not_utf8_is_a_config_error(tmp_path: Path) -> None:
    path = tmp_path / "latin1.yaml"
    path.write_bytes(b"repository: owner/repo\napprovers: [alice]\nnote: caf\xe9 \xff\xfe\n")
    with pytest.raises(ConfigError):
        load_config(path)


# --------------------------------------------------------------------------- example config
@pytest.fixture
def example_text() -> str:
    if not EXAMPLE.is_file():
        pytest.skip("config/agent-bridge.example.yaml does not exist in this checkout")
    return EXAMPLE.read_text(encoding="utf-8")


def _scalars(node: Any) -> list[str]:
    if isinstance(node, dict):
        return [s for value in node.values() for s in _scalars(value)]
    if isinstance(node, list):
        return [s for value in node for s in _scalars(value)]
    return [node] if isinstance(node, str) else []


def test_example_config_loads_and_validates(example_text: str) -> None:
    cfg = load_config(EXAMPLE)

    assert cfg.dry_run is True
    assert cfg.pull_request.draft is True
    assert cfg.approvers
    assert cfg.scopes.allowed == ["docs-only"]
    assert cfg.agent.default == "omp"
    assert cfg.repo_path == REPO_ROOT.resolve()
    assert cfg.github.token_file is None
    assert cfg.github.token_env == "LAWNBERRY_BRIDGE_GITHUB_TOKEN"


def test_example_config_uses_the_built_in_protected_paths(example_text: str) -> None:
    raw = yaml.safe_load(example_text)
    assert "protected_paths" not in raw
    assert load_config(EXAMPLE).protected_paths == list(DEFAULT_PROTECTED_PATHS)


CREDENTIAL_WORD = re.compile(
    r"(?i)token|secret|password|passwd|credential|api[_-]?key|access[_-]?key"
)
CREDENTIAL_ASSIGNMENT = re.compile(
    r"(?i)[\w-]*(?:token|secret|password|passwd|credential|api[_-]?key|access[_-]?key)[\w-]*"
    r"\s*[:=]\s*(?P<value>[^\s#]+)"
)
# What may legitimately be the value of a credential-named option: the *name* of an environment
# variable, a file path, or an empty value.
HARMLESS_VALUE = re.compile(r"^(?:[A-Z_][A-Z0-9_]*|~?/[\w./~-]+|null|)$")
SECRET_SHAPED = re.compile(r"^(?=.*\d)(?=.*[A-Za-z])[A-Za-z0-9+/_=-]{20,}$")


def _credential_values(node: Any, key: str = "") -> list[tuple[str, Any]]:
    if isinstance(node, dict):
        return [pair for k, v in node.items() for pair in _credential_values(v, str(k))]
    if isinstance(node, list):
        return [pair for v in node for pair in _credential_values(v, key)]
    return [(key, node)] if CREDENTIAL_WORD.search(key) else []


def test_example_config_contains_no_secrets(example_text: str) -> None:
    """No token-shaped strings anywhere and no credential option carries an actual value."""
    strong = [kind for kind in Redactor().find(example_text) if kind != "assignment"]
    assert strong == []
    for scalar in _scalars(yaml.safe_load(example_text)):
        assert [k for k in Redactor().find(scalar) if k != "assignment"] == [], scalar
    lowered = example_text.lower()
    assert "ghp_" not in lowered and "github_pat_" not in lowered and "-----begin" not in lowered
    for key, value in _credential_values(yaml.safe_load(example_text)):
        assert value is None or HARMLESS_VALUE.match(str(value)), (key, value)
    # commented-out examples are covered too
    for match in CREDENTIAL_ASSIGNMENT.finditer(example_text):
        assert not SECRET_SHAPED.match(match.group("value")), match.group(0)


def test_example_config_documents_every_top_level_option(example_text: str) -> None:
    raw = yaml.safe_load(example_text)
    undocumented = set(BridgeConfig.model_fields) - set(raw) - {"protected_paths"}
    assert undocumented == set()


# --------------------------------------------------------------------------- resolve_github_token
def token_cfg(**kwargs: Any) -> GitHubConfig:
    return GitHubConfig.model_validate(kwargs)


def private_file(tmp_path: Path, content: str = TOKEN + "\n", mode: int = 0o600) -> Path:
    path = tmp_path / "token"
    path.write_text(content, encoding="utf-8")
    path.chmod(mode)
    return path


def test_environment_variable_wins_and_is_stripped(tmp_path: Path) -> None:
    cfg = token_cfg(token_file=str(private_file(tmp_path, "file-token-value")))
    env = {"LAWNBERRY_BRIDGE_GITHUB_TOKEN": f"  {TOKEN}\n"}
    assert resolve_github_token(cfg, env) == TOKEN


def test_environment_variable_wins_without_touching_the_file(tmp_path: Path) -> None:
    cfg = token_cfg(token_file=str(tmp_path / "does-not-exist"))
    assert resolve_github_token(cfg, {"LAWNBERRY_BRIDGE_GITHUB_TOKEN": TOKEN}) == TOKEN


def test_custom_variable_name_is_honoured() -> None:
    cfg = token_cfg(token_env="MY_BRIDGE_TOKEN")
    assert resolve_github_token(cfg, {"MY_BRIDGE_TOKEN": TOKEN}) == TOKEN
    with pytest.raises(ConfigError):
        resolve_github_token(cfg, {"LAWNBERRY_BRIDGE_GITHUB_TOKEN": TOKEN})


def test_process_environment_is_used_when_no_mapping_is_given(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LAWNBERRY_BRIDGE_GITHUB_TOKEN", TOKEN)
    assert resolve_github_token(token_cfg()) == TOKEN
    monkeypatch.delenv("LAWNBERRY_BRIDGE_GITHUB_TOKEN")
    with pytest.raises(ConfigError):
        resolve_github_token(token_cfg())


@pytest.mark.parametrize("blank", ["", "   ", "\n"])
def test_a_blank_variable_falls_back_to_the_token_file(tmp_path: Path, blank: str) -> None:
    cfg = token_cfg(token_file=str(private_file(tmp_path)))
    assert resolve_github_token(cfg, {"LAWNBERRY_BRIDGE_GITHUB_TOKEN": blank}) == TOKEN


@pytest.mark.parametrize("mode", [0o600, 0o400, 0o700])
def test_private_token_files_are_accepted(tmp_path: Path, mode: int) -> None:
    cfg = token_cfg(token_file=str(private_file(tmp_path, mode=mode)))
    assert resolve_github_token(cfg, {}) == TOKEN


@pytest.mark.parametrize("mode", [0o644, 0o640, 0o604, 0o660, 0o606, 0o666, 0o670, 0o707])
def test_token_files_open_to_group_or_others_are_rejected(tmp_path: Path, mode: int) -> None:
    path = private_file(tmp_path, mode=mode)
    with pytest.raises(ConfigError, match="group/others") as info:
        resolve_github_token(token_cfg(token_file=str(path)), {})
    assert str(path) in str(info.value)
    assert TOKEN not in str(info.value)


def test_missing_token_file_is_a_config_error_naming_the_path(tmp_path: Path) -> None:
    path = tmp_path / "gone"
    with pytest.raises(ConfigError, match="cannot read GitHub token file") as info:
        resolve_github_token(token_cfg(token_file=str(path)), {})
    assert str(path) in str(info.value)


def test_token_file_with_a_tilde_uses_the_home_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    private_file(tmp_path)
    assert resolve_github_token(token_cfg(token_file="~/token"), {}) == TOKEN


def test_an_empty_token_file_means_no_token(tmp_path: Path) -> None:
    cfg = token_cfg(token_file=str(private_file(tmp_path, content=" \n")))
    with pytest.raises(ConfigError, match="no GitHub token"):
        resolve_github_token(cfg, {})


def test_missing_token_error_names_the_variable_and_never_echoes_values() -> None:
    env = {"OTHER_SECRET": TOKEN, "LAWNBERRY_BRIDGE_GITHUB_TOKEN": "   ", "PATH": "/usr/bin"}
    with pytest.raises(ConfigError) as info:
        resolve_github_token(token_cfg(), env)
    message = str(info.value)
    assert "LAWNBERRY_BRIDGE_GITHUB_TOKEN" in message
    assert "token_file" in message
    assert TOKEN not in message and "/usr/bin" not in message


def test_missing_token_error_uses_the_configured_variable_name() -> None:
    with pytest.raises(ConfigError) as info:
        resolve_github_token(token_cfg(token_env="ACME_GH_TOKEN"), {})
    assert "ACME_GH_TOKEN" in str(info.value)
    assert "LAWNBERRY_BRIDGE_GITHUB_TOKEN" not in str(info.value)


def test_token_file_permissions_are_checked_on_the_real_file_mode(tmp_path: Path) -> None:
    path = private_file(tmp_path)
    assert (os.stat(path).st_mode & 0o777) == 0o600
    cfg = token_cfg(token_file=str(path))
    assert resolve_github_token(cfg, {}) == TOKEN
    path.chmod(0o644)
    with pytest.raises(ConfigError):
        resolve_github_token(cfg, {})
