"""W1 capture integrity: passes clean sessions, flags gaps, skew and poor RTK."""

import json

from backend.src.services.capture_integrity import check_session

RATES = {"gps": 0.1, "imu": 0.01, "commands": 0.1, "camera": 0.05}


def _session(tmp_path, *, drop=None, offset_s=0.0, float_frac=0.0):
    (tmp_path / "manifest.json").write_text("{}")
    for name, dt in RATES.items():
        n = int(10 / dt)
        lines = []
        for i in range(n):
            if drop and drop[0] == name and drop[1] <= i < drop[2]:
                continue
            rec = {"t_ns": int((i * dt + (offset_s if name == "camera" else 0)) * 1e9)}
            if name == "gps":
                rec["fix"] = "rtk_float" if i < n * float_frac else "rtk_fixed"
            lines.append(json.dumps(rec))
        (tmp_path / f"{name}.jsonl").write_text("\n".join(lines))
    return tmp_path


def test_clean_session_passes(tmp_path):
    r = check_session(_session(tmp_path))
    assert r.ok, r.problems
    assert r.stats["gps"]["rtk_fixed_fraction"] == 1.0


def test_camera_gap_flagged(tmp_path):
    r = check_session(_session(tmp_path, drop=("camera", 50, 60)))
    assert not r.ok and any(p.startswith("camera: gap") for p in r.problems)


def test_clock_skew_flagged(tmp_path):
    r = check_session(_session(tmp_path, offset_s=2.0))
    assert any("skew" in p for p in r.problems)


def test_low_rtk_fix_flagged(tmp_path):
    r = check_session(_session(tmp_path, float_frac=0.1))
    assert any(p.startswith("rtk_fixed") for p in r.problems)


def test_missing_stream_flagged(tmp_path):
    s = _session(tmp_path)
    (s / "imu.jsonl").unlink()
    assert "missing stream imu" in check_session(s).problems
