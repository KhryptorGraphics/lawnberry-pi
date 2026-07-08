# Pi 5 CSI camera (Pi Cam v2 / IMX219) — libcamera status

## Update: root cause fixed and verified (built from source)

libcamera `v0.5.2+rpt20250903` (matching the installed picamera2 0.3.36's
era) was built from source with `-Dpipelines=rpi/pisp` into a self-contained
prefix, `/apps/lawnberry-pi/.local-libcamera` — **the system's apt-managed
`libcamera0.2` was never touched.** `libpisp-dev` (1.2.1) installs standalone
from the raspi apt repo with zero conflicts and was used as a build-time
dependency instead of building it too.

Verified working end to end:
- `cam -l` / `cam -c1 --capture=3` (pure libcamera, no Python) — enumerates
  and captures real frames from the IMX219 at ~30fps.
- `Picamera2.global_camera_info()` — now lists the IMX219 (it was silently
  absent before).
- `Picamera2().configure()/start()/capture_array()` — full still capture at
  the sensor's native 3280x2464.
- `lawnberry-camera.service` (the dedicated systemd unit, `systemd/lawnberry-camera.service`)
  now starts cleanly and logs "Camera streaming started successfully" against
  the CSI camera at the configured 1920x1080/30fps — no more
  "Device or resource busy".

The four env vars (`LD_LIBRARY_PATH`, `LIBCAMERA_IPA_MODULE_PATH`,
`LIBCAMERA_IPA_CONFIG_PATH`, `PYTHONPATH`) that point at the custom prefix
are on `lawnberry-camera.service` only, not system-wide and not on
`lawnberry-backend.service`.

**This does not yet reach the browser** — see "Known remaining gap" below.
The kernel-side device tree was never the problem either: `dmesg` already
showed the `-raspi` kernel flavor had fully probed the IMX219 and registered
it as `/dev/video0` before any of this work started.

## Known remaining gap: frontend camera routes don't exist server-side

`ControlView.vue` calls `/api/v2/camera/status`, `/api/v2/camera/frame`,
`/api/v2/camera/stream.mjpeg`, `/api/v2/camera/start` — **all 404 on the
live backend**, confirmed by direct curl. These paths only appear in
`middleware/rate_limiting.py`'s rate-limit config; no router anywhere
defines them. This is unrelated to the libcamera fix — it would 404 even
with a working camera. Separately, `backend/src/main.py` imports and starts
its own embedded `camera_stream_service` instance directly in the main
process (`main.py:43,138-139`), independent of (and possibly redundant
with) the dedicated `lawnberry-camera.service` unit, which listens on
`/tmp/lawnberry-camera.sock` but currently has no connected consumer either.
Which of these two is meant to back the REST routes above is an open
question, not yet decided.

The mower has two cameras (`config/hardware.yaml`): a USB stereo camera and a
CSI-connected Raspberry Pi Camera v2 (IMX219), the latter is the `primary`
camera and what "Live Camera Feed" in Manual Control is meant to use.

## Root cause

The Pi 5 needs libcamera's `rpi/pisp` pipeline handler to bind the RP1 CFE +
PiSP ISP + IMX219 chain into a usable `Camera` object. Pipeline handlers are
compiled into libcamera at build time, not loaded dynamically. Ubuntu
24.04's own `libcamera0.2` build (`0.2.0-3fakesync1build6`, from
`noble/main`) does not include it, so `Picamera2.global_camera_info()` never
lists the IMX219 — only the USB stereo camera enumerates (generic UVC
pipeline handler, needs no Pi-specific patches).

This is why the USB stereo camera works and the CSI camera doesn't, and why
it looks like "camera offline" in Manual Control if the UI/consumer expects
the primary (CSI) camera.

## Environment fixes already applied on-device (not tracked by git)

These make `picamera2` importable and installable under this venv, but do
**not** fix the root cause above — they were prerequisites for even being
able to run `Picamera2.global_camera_info()` to diagnose it:

1. **`libcamera-multiarch.pth`** in the venv's site-packages, so the venv's
   Python can see Ubuntu's system-installed `python3-libcamera` bindings:
   ```
   /apps/lawnberry-pi/.venv/lib/python3.12/site-packages/libcamera-multiarch.pth
   ```
   contents: `/usr/lib/aarch64-linux-gnu/python3.12/site-packages`

2. **`kms.py` stub** in the venv's site-packages, satisfying picamera2's
   unconditional `import kms as pykms` (the real `python3-kms++` needs
   Python < 3.12 and isn't installable here; this service never uses DRM
   preview, so a stub that raises on actual use is sufficient — see the
   docstring in the file for the full explanation).

3. **`libcap-dev`** (apt) — needed to build `python-prctl`, a `picamera2`
   dependency.

4. **`picamera2`** (pip, currently `0.3.36`) — not packaged for this venv
   otherwise.

Run `scripts/setup_pi5_csi_camera_env.sh` to (re)apply these idempotently on
a fresh deploy.

## What was investigated and ruled out for the actual fix

The obvious fix — install the Raspberry Pi Foundation's own `libcamera0.2`
build from their `archive.raspberrypi.com/debian bookworm` repo (already
configured, added for Hailo support), which does have `rpi/pisp` compiled
in — **does not work**, confirmed via `apt-get install --dry-run`:

- The raspi repo's current `python3-libcamera` (`0.5.2+rpt20250903-1~bpo12+1`)
  hard-depends on `python3 (< 3.12)`. This box runs Python 3.12.3. Not
  installable, full stop.
- Going around the Python binding and installing the C++ core directly
  (`libcamera0.5` + `rpicam-apps`) also fails to resolve: `libcamera0.5`
  wants an exact-version `libcamera-ipa` the repo isn't actually serving at
  that priority, and `rpicam-apps`/`librpicam-app1` pull in `libavcodec59`/
  `libavformat59`/`libjpeg62-turbo` versions that conflict with Ubuntu
  noble's ffmpeg/libjpeg stack.

In short: the raspi Debian **bookworm** package family is not a clean drop-in
on Ubuntu **noble** — this isn't a simple apt-priority/pin problem, it's
real ABI/version divergence across several shared libraries that other
things on this box (OpenCV, the working USB camera path) also depend on.
Force-installing individual raspi `.deb`s here risks breaking those.

## Build reproduction

```
sudo apt-get install -y libpisp-dev meson ninja-build pkg-config \
  python3-yaml python3-ply python3-jinja2 python3-pybind11 pybind11-dev \
  libyaml-dev libudev-dev libevent-dev git
git clone --branch v0.5.2+rpt20250903 --depth 1 \
  https://github.com/raspberrypi/libcamera.git
cd libcamera
meson setup build --prefix=/apps/lawnberry-pi/.local-libcamera \
  -Dpipelines=rpi/pisp -Dcam=enabled -Dqcam=disabled -Dgstreamer=disabled \
  -Ddocumentation=disabled -Dtest=false -Dpycamera=enabled -Dv4l2=enabled
ninja -C build
ninja -C build install
```

If picamera2 is ever upgraded, re-check which libcamera tag pairs with the
new version (RPi Foundation versions picamera2 and their libcamera fork as a
pair — check `raspberrypi/libcamera` tags for one dated close to the
picamera2 release) and rebuild against that tag, not blindly against `main`.

## Current live status

`lawnberry-camera.service` is running and streaming the CSI camera
successfully. The main `lawnberry-backend` process still separately starts
its own embedded `camera_stream_service` instance (`main.py:43,138-139`) —
that one does *not* have the custom-libcamera env vars, so it still only
sees the USB stereo camera. See "Known remaining gap" above for why none of
this is visible in the browser yet.
