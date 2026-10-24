"""Does a git remote point at the repository the bridge is configured for?

Approvals, labels, comments and pull requests go to ``BridgeConfig.repository`` through the
GitHub API, while the base is fetched from, and the branch pushed to, whatever URL the clone's
remote holds.  :func:`check_remote` is the comparison between the two.  It reads every spelling
git accepts for a repository on the configured host (``https://``, ``ssh://``, scp-like
``git@host:owner/name``; with or without ``.git``; any letter case) and refuses everything else.

A filesystem remote (a path or a ``file://`` URL: what test rigs and local mirrors use) cannot be
compared with a GitHub repository.  It is reported as ``local`` and the caller decides.

Nothing returned from here contains a credential.  Userinfo, query and fragment are dropped, and
a URL that does not split into a plain host and plain ``owner/name`` segments is only ever
described as ``an unrecognised URL``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from urllib.parse import urlsplit

__all__ = ["RemoteCheck", "UNRECOGNISED", "check_remote"]

UNRECOGNISED = "an unrecognised URL"

_SCHEME = re.compile(r"^([A-Za-z][A-Za-z0-9+.\-]*)://")
_HOST = re.compile(r"^[a-z0-9](?:[a-z0-9.\-]*[a-z0-9])?$")
_SEGMENT = re.compile(r"^[A-Za-z0-9_.\-]+$")
_GIT_SUFFIX = re.compile(r"\.git$", re.IGNORECASE)

# Other names the same service answers to.  GitHub documents ssh.github.com:443 for networks
# that block port 22; it serves exactly the repositories github.com does.
_ALIASES: dict[str, tuple[str, ...]] = {"github.com": ("ssh.github.com",)}


@dataclass(frozen=True)
class RemoteCheck:
    """The verdict on one remote URL.

    ``ok`` says the remote may be used: it names the configured repository on the configured
    host, or it is local.  ``local`` marks a filesystem remote (nothing was compared).  ``shown``
    names the remote for a message: ``host/owner/name`` when the URL parses into those parts,
    its path when it is local, otherwise :data:`UNRECOGNISED`.
    """

    ok: bool
    local: bool
    shown: str


def check_remote(url: str, *, repository: str, web_host: str) -> RemoteCheck:
    """Compare a git remote URL with ``repository`` (``owner/name``) hosted on ``web_host``."""
    text = url.strip()
    if not text:
        return RemoteCheck(ok=False, local=False, shown=UNRECOGNISED)
    located = _locate(text)
    if located is None:
        return RemoteCheck(ok=True, local=True, shown=_local_name(text))
    host, path = located
    parts = path.strip("/").split("/") if path.strip("/") else []
    if parts:
        parts[-1] = _GIT_SUFFIX.sub("", parts[-1])
    if not _HOST.match(host) or not all(_SEGMENT.match(part) for part in parts):
        return RemoteCheck(ok=False, local=False, shown=UNRECOGNISED)
    wanted = {web_host.lower(), *_ALIASES.get(web_host.lower(), ())}
    names_it = len(parts) == 2 and "/".join(parts).lower() == repository.lower()
    return RemoteCheck(ok=names_it and host in wanted, local=False, shown="/".join([host, *parts]))


def _locate(text: str) -> tuple[str, str] | None:
    """``(host, path)`` of a network remote, or ``None`` when git reads ``text`` as a path.

    An unparseable URL yields an empty host, which no caller accepts.  Port, userinfo, query and
    fragment never take part in the comparison; a query or fragment makes the URL unparseable
    (git cannot fetch from one anyway).
    """
    scheme = _SCHEME.match(text)
    if scheme:
        if scheme.group(1).lower() == "file":
            return None
        try:
            parts = urlsplit(text)
            host = parts.hostname or ""
        except ValueError:
            return "", ""
        if parts.query or parts.fragment:
            return "", ""
        return host.rstrip("."), parts.path
    # No scheme: git's own rule.  ``host:path`` is scp-like ssh unless a slash comes first.
    colon, slash = text.find(":"), text.find("/")
    if colon < 0 or 0 <= slash < colon:
        return None
    left, _, path = text.partition(":")
    return left.rpartition("@")[2].lower().rstrip("."), path


def _local_name(text: str) -> str:
    """A filesystem remote's path; a ``file://`` URL loses its authority (and any userinfo)."""
    if _SCHEME.match(text):
        try:
            return urlsplit(text).path or "file://"
        except ValueError:
            return "file://"
    return text[:300]
