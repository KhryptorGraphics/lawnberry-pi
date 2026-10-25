"""The UI server (frontend/proxy-trust.mjs) and the backend must agree on what an E-stop is.

The UI server's socket guard keeps a reserve for the E-stop and recognises it by method and path
(SEC3-03). Its list of paths is JavaScript, the backend's is ``middleware/emergency_paths.py``:
this test fails the day one of them changes without the other.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from backend.src.middleware.emergency_paths import EMERGENCY_STOP_PATHS

REPO = Path(__file__).resolve().parents[2]


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_the_ui_server_knows_exactly_the_backend_estop_paths():
    code = (
        "import {BACKEND_EMERGENCY_STOP_PATHS as p} from './frontend/proxy-trust.mjs';"
        "console.log(JSON.stringify([...p]))"
    )
    out = subprocess.run(
        ["node", "--input-type=module", "-e", code],
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=60,
        check=True,
    ).stdout
    assert set(json.loads(out)) == set(EMERGENCY_STOP_PATHS)
