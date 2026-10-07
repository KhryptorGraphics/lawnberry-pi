#!/usr/bin/env bash
# Run an Isaac Sim Python script headless in the NGC container (W3/W4 jobs).
# Host: the x86 workshop. Run from the repo mirror ($LB/repo); it is mounted at /workspace.
# $LB = ~/nvme2/lawnberrypiserver is the single data root; $LB/data is mounted at /data, so
# every artifact (assets, SDG output, models) is read and written under /data in the container.
#
#   scripts/isaacsim-headless.sh workshop/isaac_sdg.py --assets workshop/sdg_assets.json --out /data/sdg/v1
#
# ISAACSIM_GPU    nvidia-smi index (default 1 = RTX 3080 Ti on utrainn; V100 is 2, P100 0)
# ISAACSIM_IMAGE  default nvcr.io/nvidia/isaac-sim:6.0.0 (needs `docker login nvcr.io` once)
# ISAACSIM_CACHE  shader/asset caches, owned by the image's uid 1234
#                 (default ~/nvme2/lawnberrypiserver/cache/isaacsim)
# LB_DATA         host data dir mounted at /data (default ~/nvme2/lawnberrypiserver/data)
#
# The 3080 Ti gives DLSS and native annotators (~0.25 s/frame warm). For the V100 fallback
# (no RT cores) scripts must pass --/renderer/gpuEnumeration/rtxRequired=false
# (workshop/isaac_sdg.py always does; it is harmless on RTX cards).
# See docs/autonomy-toolchain-matrix.md.
set -euo pipefail

IMAGE="${ISAACSIM_IMAGE:-nvcr.io/nvidia/isaac-sim:6.0.0}"
GPU="${ISAACSIM_GPU:-1}"
CACHE="${ISAACSIM_CACHE:-$HOME/nvme2/lawnberrypiserver/cache/isaacsim}"
DATA="${LB_DATA:-$HOME/nvme2/lawnberrypiserver/data}"
SCRIPT="${1:?usage: $0 script.py [args...]}"; shift

for d in kit cache nv docs; do
  if [[ ! -d "$CACHE/$d" ]]; then
    mkdir -p "$CACHE/$d"
    sudo chown 1234:1234 "$CACHE/$d"
  fi
done

exec docker run --rm --init --runtime=nvidia --gpus "\"device=$GPU\"" \
  -e ACCEPT_EULA=Y -e PRIVACY_CONSENT=Y \
  -v "$CACHE/kit":/isaac-sim/kit/cache \
  -v "$CACHE/cache":/isaac-sim/.cache \
  -v "$CACHE/nv":/isaac-sim/.nv \
  -v "$CACHE/docs":/isaac-sim/Documents \
  -v "$DATA":/data \
  -v "$PWD":/workspace -w /workspace \
  --entrypoint /isaac-sim/python.sh \
  "$IMAGE" "$SCRIPT" "$@"
