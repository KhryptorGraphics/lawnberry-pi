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
# ISAACSIM_CACHE  shader/asset caches, owned by the container user
#                 (default ~/nvme2/lawnberrypiserver/cache/isaacsim)
# LB_DATA         host data dir mounted at /data (default ~/nvme2/lawnberrypiserver/data)
# ISAACSIM_USER   docker --user UID:GID (default: the image's isaac-sim, uid 1234). On the
#                 workshop $LB lives on an ntfs-3g mount (uid=1000,umask=002 since the
#                 2026-10-24 reboot), where uid 1234 is "other" and cannot write /data or the
#                 caches, and the image's /tmp is ubuntu(1000):775: the job dies with
#                 "PermissionError: /data/..." and "No usable temporary directory". Use
#                 ISAACSIM_USER="$(id -u):$(id -g)" (kp = 1000 = the image's `ubuntu`).
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
USER_ARGS=()
[[ -z ${ISAACSIM_USER:-} ]] || USER_ARGS=(--user "$ISAACSIM_USER")
SCRIPT="${1:?usage: $0 script.py [args...]}"; shift

for d in kit cache nv docs; do
  if [[ ! -d "$CACHE/$d" ]]; then
    mkdir -p "$CACHE/$d"
    sudo chown "${ISAACSIM_USER:-1234:1234}" "$CACHE/$d"
  fi
done

exec docker run --rm --init --runtime=nvidia --gpus "\"device=$GPU\"" \
  "${USER_ARGS[@]}" \
  -e ACCEPT_EULA=Y -e PRIVACY_CONSENT=Y \
  -v "$CACHE/kit":/isaac-sim/kit/cache \
  -v "$CACHE/cache":/isaac-sim/.cache \
  -v "$CACHE/nv":/isaac-sim/.nv \
  -v "$CACHE/docs":/isaac-sim/Documents \
  -v "$DATA":/data \
  -v "$PWD":/workspace -w /workspace \
  --entrypoint /isaac-sim/python.sh \
  "$IMAGE" "$SCRIPT" "$@"
