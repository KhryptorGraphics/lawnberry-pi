#!/usr/bin/env bash
# Build HailoRT + PCIe driver for the mower Pi's Hailo-8 from Hailo's public sources.
# Build only; installation is a separate step (scripts/pi_install_hailort.sh).
#
#   scripts/pi_build_hailort.sh [VERSION]        # default 4.24.0 (Model Zoo v2.19 / DFC 3.34)
#
# Sources: github.com/hailo-ai/hailort, github.com/hailo-ai/hailort-drivers, and the
# firmware from hailo-hailort.s3.eu-west-2.amazonaws.com (driver repo's download_firmware.sh).
# Raspberry Pi's apt archive stops at 4.20, so newer versions are built here.
set -euo pipefail

VER="${1:-4.24.0}"
ROOT="$HOME/hailo-build/$VER"
JOBS="$(nproc)"
mkdir -p "$ROOT"
cd "$ROOT"

[[ -d hailort-drivers ]] || git clone -q --depth 1 --branch "v$VER" https://github.com/hailo-ai/hailort-drivers.git
[[ -d hailort ]] || git clone -q --depth 1 --branch "v$VER" https://github.com/hailo-ai/hailort.git

echo "[hailo-build] PCIe driver for $(uname -r)"
make -C hailort-drivers/linux/pcie -j"$JOBS" all

echo "[hailo-build] firmware"
(cd hailort-drivers && ./download_firmware.sh)

echo "[hailo-build] HailoRT + hailortcli + Python bindings"
cmake -S hailort -B hailort/build -DCMAKE_BUILD_TYPE=Release -DHAILO_BUILD_PYBIND=1 \
  -DPYBIND11_PYTHON_VERSION="$(python3 -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")')" \
  -DHAILO_BUILD_EXAMPLES=0
cmake --build hailort/build --config Release -j"$JOBS"

echo "[hailo-build] done: $ROOT"
ls -la hailort-drivers/linux/pcie/hailo_pci.ko hailort-drivers/hailo8_fw."$VER".bin
find hailort/build -name "libhailort.so*" -o -name hailortcli -type f -o -name "_pyhailort*.so" | head
