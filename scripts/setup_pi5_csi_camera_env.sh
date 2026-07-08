#!/usr/bin/env bash
set -euo pipefail

# === Pi 5 CSI camera (picamera2) venv environment setup ===
# Idempotently applies the on-device fixes needed to import/install
# picamera2 in the project venv on Ubuntu 24.04 / Python 3.12.
#
# Does NOT fix the underlying libcamera pipeline-handler gap (Ubuntu's
# libcamera0.2 build lacks the Pi5 rpi/pisp handler, so the CSI camera
# still won't enumerate after running this) — see
# docs/pi5-csi-camera-libcamera-status.md for that.
#
# Run from the repo root on the Pi: ./scripts/setup_pi5_csi_camera_env.sh

VENV_SITE_PACKAGES="/apps/lawnberry-pi/.venv/lib/python3.12/site-packages"
PTH_FILE="$VENV_SITE_PACKAGES/libcamera-multiarch.pth"
KMS_STUB="$VENV_SITE_PACKAGES/kms.py"

echo "=== libcap-dev (needed to build python-prctl, a picamera2 dep) ==="
if dpkg -l libcap-dev >/dev/null 2>&1; then
    echo "libcap-dev already installed."
else
    sudo apt-get install -y libcap-dev
fi

echo ""
echo "=== multiarch .pth shim (venv -> system python3-libcamera) ==="
if [ -f "$PTH_FILE" ]; then
    echo "$PTH_FILE already exists."
else
    echo "/usr/lib/aarch64-linux-gnu/python3.12/site-packages" > "$PTH_FILE"
    echo "Created $PTH_FILE"
fi

echo ""
echo "=== kms.py stub (satisfies picamera2's unconditional 'import kms') ==="
if [ -f "$KMS_STUB" ]; then
    echo "$KMS_STUB already exists."
else
    cat > "$KMS_STUB" <<'EOF'
"""Minimal stub satisfying picamera2's top-level `import kms as pykms`.

picamera2.previews.drm_preview unconditionally imports kms/pykms (the
python3-kms++ apt package) to support local DRM/KMS preview windows.
That package isn't installable on this host: it depends on python3 < 3.12,
but this deployment runs Ubuntu 24.04 (Python 3.12). This service never
calls start_preview(Preview.DRM) -- it's a headless MJPEG/snapshot streamer
consumed by camera_stream_service.py, not a local display -- so DrmPreview
itself is never exercised. This stub exists purely to satisfy the import
(including DrmPreview.FMT_MAP, a class-body dict literal evaluated at
import time); any real use of DRM preview raises a clear error instead of
silently no-op'ing.
"""


class PixelFormats:
    RGB888 = "RGB888"
    BGR888 = "BGR888"
    YUYV = "YUYV"
    YVYU = "YVYU"
    XRGB8888 = "XRGB8888"
    XBGR8888 = "XBGR8888"
    YUV420 = "YUV420"
    YVU420 = "YVU420"


class Card:
    def __init__(self, *args, **kwargs):
        raise RuntimeError(
            "kms/pykms is a stub on this host (python3-kms++ is not installable "
            "under Python 3.12); DRM preview is unavailable. See kms.py docstring."
        )


class ResourceManager:
    def __init__(self, *args, **kwargs):
        raise RuntimeError(
            "kms/pykms is a stub on this host (python3-kms++ is not installable "
            "under Python 3.12); DRM preview is unavailable. See kms.py docstring."
        )
EOF
    echo "Created $KMS_STUB"
fi

echo ""
echo "=== picamera2 (pip) ==="
/apps/lawnberry-pi/.venv/bin/pip show picamera2 >/dev/null 2>&1 \
    && echo "picamera2 already installed." \
    || /apps/lawnberry-pi/.venv/bin/pip install picamera2

echo ""
echo "Done. This does not fix CSI camera detection -- see"
echo "docs/pi5-csi-camera-libcamera-status.md for the remaining libcamera gap."
