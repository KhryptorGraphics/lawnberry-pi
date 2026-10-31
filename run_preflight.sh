#!/bin/sh
# Diagnostic preflight for the 77603 CAD profile: renders every catalogued piece and reports
# ALL failures in one pass. Publishes nothing.
set -e
cd /home/kp/thordrive/lawnberry-pi-hardware-20261030
PYTHONPATH="$PWD" exec /home/kp/thordrive/lawnberry-pi/.venv/bin/python \
    hardware/mounts/preflight_77603.py "$@"
