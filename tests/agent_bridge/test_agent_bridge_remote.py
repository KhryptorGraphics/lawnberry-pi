"""Tests for ``tools/agent_bridge/remote.py``: does a git remote name the configured repository?

The bridge asks GitHub's API about ``BridgeConfig.repository`` but fetches its base from, and
pushes its branches to, whatever URL the clone's remote has.  ``check_remote`` is the comparison
that ties the two together, so it has to read every spelling git accepts for the same repository
(``https``, ``ssh://``, scp-like, with or without ``.git``, any letter case), refuse everything
else, and never let a credential embedded in a URL reach a message.

``GitOps.push_urls`` supplies the other half: a push does not go where a fetch does when the
remote has a ``pushurl``, a ``pushInsteadOf`` rewrite or several URLs, and git pushes to all of
them.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from bridgefakes.rig import GitRepoEnv

from tools.agent_bridge.config import GitConfig, GitHubConfig
from tools.agent_bridge.errors import GitError
from tools.agent_bridge.gitops import GitOps
from tools.agent_bridge.remote import RemoteCheck, check_remote

REPOSITORY = "owner/repo"
HOST = "github.com"


def verdict(url: str, *, repository: str = REPOSITORY, host: str = HOST) -> RemoteCheck:
    return check_remote(url, repository=repository, web_host=host)


# --------------------------------------------------------------------------- accepted spellings
@pytest.mark.parametrize(
    "url",
    [
        "https://github.com/owner/repo",
        "https://github.com/owner/repo.git",
        "https://github.com/owner/repo/",
        "https://github.com/owner/repo.git/",
        "http://github.com/owner/repo.git",
        "https://github.com:443/owner/repo.git",
        "ssh://git@github.com/owner/repo.git",
        "ssh://git@github.com:22/owner/repo",
        "ssh://github.com/owner/repo.git",
        "git+ssh://git@github.com/owner/repo.git",
        "git://github.com/owner/repo.git",
        "git@github.com:owner/repo.git",
        "git@github.com:owner/repo",
        "git@github.com:/owner/repo.git",
        "github.com:owner/repo.git",
        "  https://github.com/owner/repo.git\n",
    ],
)
def test_every_spelling_of_the_configured_repository_is_accepted(url: str) -> None:
    found = verdict(url)

    assert found.ok is True and found.local is False
    assert found.shown == "github.com/owner/repo"


@pytest.mark.parametrize(
    "url",
    [
        "https://GitHub.COM/OWNER/Repo",
        "https://github.com/Owner/REPO.GIT",
        "git@GITHUB.com:owner/REPO.git",
        "ssh://git@Github.com/OWNER/repo.Git",
    ],
)
def test_letter_case_and_the_git_suffix_do_not_matter(url: str) -> None:
    assert verdict(url).ok is True


def test_the_configured_repository_may_use_capitals_too() -> None:
    assert verdict("https://github.com/lowercase/name.git", repository="LowerCase/Name").ok is True
    assert verdict("git@github.com:LowerCase/Name", repository="lowercase/name").ok is True


@pytest.mark.parametrize(
    "url",
    [
        "https://x-access-token:ghp_AAAABBBBCCCCDDDD@github.com/owner/repo.git",
        "https://alice:S3cr3tPassw0rd@github.com/owner/repo",
        "https://alice@github.com/owner/repo.git",
        "ssh://deploy:hunter2@github.com:22/owner/repo.git",
    ],
)
def test_credentials_in_a_matching_url_are_ignored_and_never_shown(url: str) -> None:
    found = verdict(url)

    assert found.ok is True
    assert found.shown == "github.com/owner/repo"


def test_github_dot_com_also_answers_to_its_ssh_over_https_host() -> None:
    # GitHub documents ssh.github.com:443 for networks that block port 22: same repositories.
    assert verdict("ssh://git@ssh.github.com:443/owner/repo.git").ok is True
    assert verdict("https://ssh.github.com/owner/repo.git").ok is True
    # ...but that alias belongs to github.com only.
    assert verdict("ssh://git@ssh.ghe.example/owner/repo.git", host="ghe.example").ok is False


# --------------------------------------------------------------------------- refused remotes
@pytest.mark.parametrize(
    ("url", "shown"),
    [
        ("https://github.com/fork-owner/repo.git", "github.com/fork-owner/repo"),
        ("https://github.com/owner/other-repo.git", "github.com/owner/other-repo"),
        ("git@github.com:fork-owner/repo.git", "github.com/fork-owner/repo"),
        ("ssh://git@github.com/owner/repo-fork", "github.com/owner/repo-fork"),
        ("https://github.com/owner-x/repo", "github.com/owner-x/repo"),
        ("https://github.com/owner/repo.git.bak", "github.com/owner/repo.git.bak"),
        ("https://gitlab.com/owner/repo.git", "gitlab.com/owner/repo"),
        ("git@bitbucket.org:owner/repo.git", "bitbucket.org/owner/repo"),
        ("https://github.com.evil.example/owner/repo.git", "github.com.evil.example/owner/repo"),
        ("https://notgithub.com/owner/repo", "notgithub.com/owner/repo"),
        ("https://github.com@evil.example/owner/repo.git", "evil.example/owner/repo"),
        ("https://github.com:pass@evil.example/owner/repo.git", "evil.example/owner/repo"),
        ("ssh://git@evil.example:2222/owner/repo.git", "evil.example/owner/repo"),
        ("https://github.com/owner/repo/extra.git", "github.com/owner/repo/extra"),
        ("https://github.com/prefix/owner/repo.git", "github.com/prefix/owner/repo"),
        ("https://github.com/owner", "github.com/owner"),
        ("https://github.com/", "github.com"),
        ("https://github.com", "github.com"),
    ],
)
def test_any_other_repository_or_host_is_refused(url: str, shown: str) -> None:
    found = verdict(url)

    assert found.ok is False and found.local is False
    assert found.shown == shown


@pytest.mark.parametrize(
    "url",
    [
        "",
        "   ",
        "https://",
        "https:///owner/repo",
        "ext::sh -c 'touch /tmp/pwned'",
        "fd::7/owner/repo",
        "hg::https://github.com/owner/repo",
        "https://github.com/owner/re%70o.git",
        "https://github.com/owner/repo.git?x=/owner/repo",
        "https://github.com/owner/repo.git#fragment",
        "https://[::1]/owner/repo",
        "https://github.com/owner/repo name",
        "[::1]:owner/repo",
    ],
)
def test_text_that_does_not_parse_into_host_and_repository_is_refused_and_named_generically(
    url: str,
) -> None:
    found = verdict(url)

    assert found.ok is False and found.local is False
    assert found.shown == "an unrecognised URL"


def test_a_backslash_cannot_smuggle_another_host_past_the_comparison() -> None:
    # curl and Python agree that everything before the last ``@`` of the authority is userinfo.
    found = verdict("https://github.com\\@evil.example/owner/repo")

    assert found.ok is False
    assert found.shown == "evil.example/owner/repo"


def test_a_different_enterprise_host_is_refused_even_for_the_same_owner_and_name() -> None:
    assert verdict("git@ghe.example.com:owner/repo.git", host="ghe.example.com").ok is True
    assert verdict("https://github.com/owner/repo.git", host="ghe.example.com").ok is False
    assert verdict("git@ghe.example.com:owner/repo.git", host="github.com").ok is False


# --------------------------------------------------------------------------- local remotes
@pytest.mark.parametrize(
    "url",
    [
        "/srv/git/origin.git",
        "/tmp/some dir/origin.git",
        "../origin.git",
        "./origin.git",
        "origin.git",
        "./with:colon/origin.git",
        "/tmp/with:colon/origin.git",
        "file:///srv/git/origin.git",
        "file://localhost/srv/git/origin.git",
        "FILE:///srv/git/origin.git",
    ],
)
def test_a_filesystem_remote_is_allowed_and_flagged_local(url: str) -> None:
    found = verdict(url)

    assert found.ok is True and found.local is True


def test_a_file_url_never_shows_embedded_credentials() -> None:
    found = verdict("file://alice:S3cr3tPassw0rd@localhost/srv/git/origin.git")

    assert found.local is True
    assert "S3cr3tPassw0rd" not in found.shown and "alice" not in found.shown


# --------------------------------------------------------------------------- no credential leaks
CREDENTIALED = [
    "https://alice:S3cr3tPassw0rd@evil.example/owner/repo.git",
    "https://S3cr3tPassw0rd@evil.example/owner/repo.git",
    "https://x-access-token:ghp_S3cr3tPassw0rd@github.com/fork/repo.git",
    "ssh://alice:S3cr3tPassw0rd@evil.example:2222/owner/repo.git",
    "alice:S3cr3tPassw0rd@evil.example:owner/repo.git",
    "https://github.com/owner/repo.git?token=S3cr3tPassw0rd",
    "https://github.com/owner/repo.git#S3cr3tPassw0rd",
    "https://alice:S3cr3tPassw0rd@[::1]/owner/repo.git",
    "https://alice:S3cr3tPassw0rd@/owner/repo.git",
    "https://alice:S3cr3tPassw0rd@evil.example",
]


@pytest.mark.parametrize("url", CREDENTIALED)
def test_a_credential_embedded_in_the_url_never_appears_in_the_description(url: str) -> None:
    found = verdict(url)

    assert "S3cr3tPassw0rd" not in found.shown
    assert "alice" not in found.shown and "x-access-token" not in found.shown
    assert "@" not in found.shown and ":" not in found.shown and "?" not in found.shown


# --------------------------------------------------------------------------- the web host
@pytest.mark.parametrize(
    ("api_url", "web_host"),
    [
        ("https://api.github.com", "github.com"),
        ("https://api.github.com/", "github.com"),
        ("https://api.github.test", "github.test"),
        ("https://ghe.example.com/api/v3", "ghe.example.com"),
        ("https://ghe.example.com:8443/api/v3", "ghe.example.com"),
        ("https://api.ghe.example.com", "ghe.example.com"),
        ("https://api.acme.ghe.com", "acme.ghe.com"),
        ("https://GitHub.Example.com/api/v3", "github.example.com"),
        # An API under a path is on the web host itself, even when that host starts with "api.".
        ("https://api.corp.example/api/v3", "api.corp.example"),
        ("https://api.corp.example:8443", "corp.example"),
    ],
)
def test_the_web_host_is_derived_from_the_api_url(api_url: str, web_host: str) -> None:
    assert GitHubConfig(api_url=api_url).web_host == web_host


def test_the_default_api_url_means_github_dot_com() -> None:
    assert GitHubConfig().web_host == "github.com"


# --------------------------------------------------------------------------- GitOps.push_urls
FORK = "https://github.test/fork-owner/repo.git"
OWN = "https://github.test/owner/repo.git"


@pytest.fixture
def ops(git_env_only: GitRepoEnv, tmp_path: Path) -> GitOps:
    return GitOps(
        git_env_only.clone, GitConfig(), state_dir=tmp_path / "state", environ=git_env_only.env
    )


def test_push_urls_are_the_fetch_url_when_nothing_says_otherwise(
    ops: GitOps, git_env_only: GitRepoEnv
) -> None:
    assert ops.push_urls() == [ops.remote_url()] == [str(git_env_only.origin)]


def test_a_pushurl_sends_pushes_elsewhere_while_fetches_stay_put(
    ops: GitOps, git_env_only: GitRepoEnv
) -> None:
    git_env_only.git("config", "remote.origin.pushurl", FORK)

    assert ops.remote_url() == str(git_env_only.origin)
    assert ops.push_urls() == [FORK]


def test_every_push_url_is_listed_because_git_pushes_to_all_of_them(
    ops: GitOps, git_env_only: GitRepoEnv
) -> None:
    git_env_only.git("config", "remote.origin.pushurl", OWN)
    git_env_only.git("config", "--add", "remote.origin.pushurl", FORK)

    assert ops.push_urls() == [OWN, FORK]


def test_a_second_url_is_also_a_second_push_target(ops: GitOps, git_env_only: GitRepoEnv) -> None:
    git_env_only.git("remote", "set-url", "--add", "origin", FORK)

    assert ops.remote_url() == str(git_env_only.origin)  # a fetch uses the first URL only
    assert ops.push_urls() == [str(git_env_only.origin), FORK]


def test_push_instead_of_rewrites_only_the_push_side(ops: GitOps, git_env_only: GitRepoEnv) -> None:
    git_env_only.git("config", f"url.{FORK}.pushInsteadOf", str(git_env_only.origin))

    assert ops.remote_url() == str(git_env_only.origin)
    assert ops.push_urls() == [FORK]


def test_instead_of_rewrites_the_push_side_too(ops: GitOps, git_env_only: GitRepoEnv) -> None:
    git_env_only.git("config", f"url.{OWN}.insteadOf", str(git_env_only.origin))

    assert ops.remote_url() == OWN
    assert ops.push_urls() == [OWN]


def test_push_urls_of_a_missing_remote_is_an_error(
    git_env_only: GitRepoEnv, tmp_path: Path
) -> None:
    missing = GitOps(
        git_env_only.clone,
        GitConfig(remote="nope"),
        state_dir=tmp_path / "state",
        environ=git_env_only.env,
    )

    with pytest.raises(GitError, match="No such remote"):
        missing.push_urls()
