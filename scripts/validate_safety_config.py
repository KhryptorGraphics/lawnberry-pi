#!/usr/bin/env python3
from __future__ import annotations

"""Validate safety limits configuration from CLI (T123)."""

import json  # noqa: E402

from backend.src.safety.safety_validator import validate_on_start  # noqa: E402


def main() -> int:
    ok, report = validate_on_start()
    payload = {
        "ok": ok,
        "detail": report.detail,
        "limits": report.limits,
    }
    print(json.dumps(payload, indent=2))
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
