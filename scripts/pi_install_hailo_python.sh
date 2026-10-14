#!/usr/bin/env bash
# Install the hailo_platform Python binding into the LawnBerry venv on the mower Pi.
#
#   bash ~/pi_install_hailo_python.sh 4.24.0 [BUILD_DIR] [VENV]
#     BUILD_DIR default ~/hailo-build/<ver> (from scripts/pi_build_hailort.sh)
#     VENV      default /apps/lawnberry-pi/.venv
#
# Needs the cp3xx _pyhailort*.so already built under
#   $BUILD/hailort/hailort/libhailort/bindings/python/platform/hailo_platform/pyhailort/
# Safe to run while services are up: it stops nothing and touches only the venv.
# Running processes keep their old imports until their next restart.
set -euo pipefail

VER="${1:?usage: $0 VERSION [BUILD_DIR] [VENV]}"
BUILD="${2:-$HOME/hailo-build/$VER}"
VENV="${3:-/apps/lawnberry-pi/.venv}"
PKG="$BUILD/hailort/hailort/libhailort/bindings/python/platform"

log() { echo "[hailo-python] $*"; }
die() { echo "[hailo-python] $*" >&2; exit 1; }

[[ -x $VENV/bin/python ]] || die "no python in venv $VENV"
[[ -d $PKG ]] || die "missing binding source: $PKG"
compgen -G "$PKG/hailo_platform/pyhailort/_pyhailort*.so" >/dev/null \
  || die "no built _pyhailort*.so under $PKG/hailo_platform/pyhailort"

log "pip install $PKG"
if ! "$VENV/bin/pip" install "$PKG"; then
  log "direct install failed; building a wheel"
  (
    cd "$PKG"
    "$VENV/bin/python" setup.py bdist_wheel
    shopt -s nullglob
    wheels=(dist/hailort-"$VER"-*.whl)
    (( ${#wheels[@]} == 1 )) || die "expected one dist/hailort-$VER-*.whl, found ${#wheels[@]}"
    "$VENV/bin/pip" install --force-reinstall --no-deps "${wheels[0]}"
  )
fi

log "verify"
"$VENV/bin/python" -c "import hailo_platform as h; assert h.__version__ == '$VER', h.__version__; print('hailo_platform', h.__version__)"
log "done: hailo_platform $VER in $VENV"
