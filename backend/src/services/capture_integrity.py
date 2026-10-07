"""W1 capture-dataset integrity checks (sync, gaps, RTK-fix percentage).

A capture session is a directory containing ``manifest.json`` plus one JSONL
stream per sensor (``gps.jsonl``, ``imu.jsonl``, ``commands.jsonl``,
``camera.jsonl`` frame index). Every record carries ``t_ns`` from the Pi's
monotonic clock; GPS records also carry ``fix``. The checker is pure and
offline: it runs on the workshop before a session enters the dataset.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

REQUIRED_STREAMS = ("gps", "imu", "commands", "camera")


@dataclass(frozen=True)
class IntegrityThresholds:
    max_gap_s: dict[str, float] = field(
        default_factory=lambda: {"gps": 0.5, "imu": 0.05, "commands": 0.5, "camera": 0.2}
    )
    max_start_skew_s: float = 1.0  # streams must start/end together
    min_rtk_fixed_fraction: float = 0.95


@dataclass
class IntegrityReport:
    ok: bool
    problems: list[str]
    stats: dict[str, dict[str, float]]


def _load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def check_session(session: Path, th: IntegrityThresholds | None = None) -> IntegrityReport:
    th = th or IntegrityThresholds()
    problems: list[str] = []
    stats: dict[str, dict[str, float]] = {}
    if not (session / "manifest.json").exists():
        problems.append("missing manifest.json")
    streams: dict[str, list[int]] = {}
    gps: list[dict] = []
    for name in REQUIRED_STREAMS:
        p = session / f"{name}.jsonl"
        if not p.exists():
            problems.append(f"missing stream {name}")
            continue
        recs = _load(p)
        ts = [int(r["t_ns"]) for r in recs]
        if len(ts) < 2:
            problems.append(f"{name}: fewer than 2 records")
            continue
        if any(b <= a for a, b in zip(ts, ts[1:], strict=False)):
            problems.append(f"{name}: timestamps not strictly increasing")
        gap = max(b - a for a, b in zip(ts, ts[1:], strict=False)) / 1e9
        stats[name] = {"records": len(ts), "max_gap_s": gap, "duration_s": (ts[-1] - ts[0]) / 1e9}
        if gap > th.max_gap_s[name]:
            problems.append(f"{name}: gap {gap:.3f}s > {th.max_gap_s[name]}s")
        streams[name] = ts
        if name == "gps":
            gps = recs
    if streams:
        starts = [t[0] for t in streams.values()]
        ends = [t[-1] for t in streams.values()]
        skew = max(max(starts) - min(starts), max(ends) - min(ends)) / 1e9
        stats["sync"] = {"start_end_skew_s": skew}
        if skew > th.max_start_skew_s:
            problems.append(f"stream start/end skew {skew:.3f}s > {th.max_start_skew_s}s")
    if gps:
        frac = sum(r.get("fix") == "rtk_fixed" for r in gps) / len(gps)
        stats["gps"]["rtk_fixed_fraction"] = frac
        if frac < th.min_rtk_fixed_fraction:
            problems.append(f"rtk_fixed {frac:.1%} < {th.min_rtk_fixed_fraction:.0%}")
    return IntegrityReport(ok=not problems, problems=problems, stats=stats)
