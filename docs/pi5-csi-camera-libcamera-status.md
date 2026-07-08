# Pi 5 CSI camera (Pi Cam v2 / IMX219) — libcamera status

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

## Remaining path forward (not yet done)

Build libcamera from source (meson, `-Dpipelines=rpi/pisp`) against this
system's actual Python 3.12 and existing library versions, rather than
consuming the raspi prebuilt bookworm packages. This avoids touching any
already-installed system package. Nontrivial (compiler toolchain, DRM/CFE
kernel UAPI headers, a real build), not attempted yet.

## Current live status

`lawnberry-camera.service` (the dedicated systemd unit) is stopped as of
this investigation. The main `lawnberry-backend` process is separately
running its own embedded camera capture (visible as `"camera"` in
`GET /api/v2/hardware/robohat`) at a degraded ~1fps — likely the USB stereo
camera — which is consistent with "frame enqueue stalled" warnings in the
backend log and would look broken/frozen in the UI even though technically
live. This embedded-in-backend camera path, separate from the dedicated
`lawnberry-camera.service` unit, is itself worth a look against the
constitution's "camera-stream.service exclusively owns the camera" rule.
