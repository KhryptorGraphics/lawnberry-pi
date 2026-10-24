"""Approval-gated, GitHub-issue-driven local coding-agent bridge.

The owner approves a structured GitHub issue; this worker runs a local coding agent
(OhMyPi ``omp`` by default, Claude Code ``claude`` as an alternative) in an isolated git
worktree, verifies the result against a strict change policy, and opens a *draft* pull request.
It never merges, deploys, or touches mower hardware.  See ``docs/agent-bridge.md``.
"""

__version__ = "1.0.0"
