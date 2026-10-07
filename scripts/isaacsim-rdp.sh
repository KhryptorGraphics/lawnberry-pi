#!/usr/bin/env bash
# Launch the Isaac Sim desktop app inside the caller's xrdp session on the workshop
# (never the physical console). Headless jobs use scripts/isaacsim-headless.sh.
#
#   scripts/isaacsim-rdp.sh [isaac-sim.sh args...]
#
# ISAACSIM_GPU / ISAACSIM_IMAGE / ISAACSIM_CACHE / LB_DATA: as in scripts/isaacsim-headless.sh.
# Artifacts live under /data inside the container (host path ~/nvme2/lawnberrypiserver/data).
# rtxRequired=false is passed so the V100 fallback (ISAACSIM_GPU=2) also works; on the
# default RTX 3080 Ti it is a no-op and DLSS gives a clean viewport.
set -euo pipefail

IMAGE="${ISAACSIM_IMAGE:-nvcr.io/nvidia/isaac-sim:6.0.0}"
GPU="${ISAACSIM_GPU:-1}"
CACHE="${ISAACSIM_CACHE:-$HOME/nvme2/lawnberrypiserver/cache/isaacsim}"
DATA="${LB_DATA:-$HOME/nvme2/lawnberrypiserver/data}"

# This user's xrdp display: xorgxrdp (`Xorg :N ... -config xrdp/xorg.conf`) or the
# Xvnc backend sesman starts (`Xvnc :N ... -rfbauth .../sesman_passwd-...`).
rdp_display() {
  pgrep -u "$(id -u)" -a 'Xorg|Xvnc' | awk '/xrdp\/xorg.conf|sesman_passwd/ { for (i = 1; i <= NF; i++) if ($i ~ /^:[0-9]+$/) { print $i; exit } }'
}

DISPLAY_RDP="$(rdp_display)"
if [[ -z "$DISPLAY_RDP" ]]; then
  echo "[isaacsim-rdp] no xrdp session for $(id -un); connect with an RDP client first" >&2
  exit 1
fi
XAUTH="${XAUTHORITY:-$HOME/.Xauthority}"
# The image runs as uid 1234, which cannot read the user's mode-600 Xauthority.
# Export this display's cookie with a wildcard host family into a file it can read.
mkdir -p "$HOME/.cache"
XAUTH_CONTAINER="$HOME/.cache/isaacsim-xauth-${DISPLAY_RDP#:}"
rm -f "$XAUTH_CONTAINER"
touch "$XAUTH_CONTAINER"
xauth -f "$XAUTH" nlist "$DISPLAY_RDP" | sed -e 's/^..../ffff/' | xauth -f "$XAUTH_CONTAINER" nmerge -
chmod 644 "$XAUTH_CONTAINER"
for d in kit cache nv docs; do
  if [[ ! -d "$CACHE/$d" ]]; then
    mkdir -p "$CACHE/$d"
    sudo chown 1234:1234 "$CACHE/$d"
  fi
done

echo "[isaacsim-rdp] DISPLAY=$DISPLAY_RDP gpu=$GPU image=$IMAGE"
exec docker run --rm --init --runtime=nvidia --gpus "\"device=$GPU\"" --network host \
  -e ACCEPT_EULA=Y -e PRIVACY_CONSENT=Y \
  -e DISPLAY="$DISPLAY_RDP" -e XAUTHORITY=/tmp/.Xauthority \
  -v /tmp/.X11-unix:/tmp/.X11-unix:ro -v "$XAUTH_CONTAINER":/tmp/.Xauthority:ro \
  -v "$CACHE/kit":/isaac-sim/kit/cache \
  -v "$CACHE/cache":/isaac-sim/.cache \
  -v "$CACHE/nv":/isaac-sim/.nv \
  -v "$CACHE/docs":/isaac-sim/Documents \
  -v "$DATA":/data \
  -v "$PWD":/workspace \
  --entrypoint /isaac-sim/isaac-sim.sh \
  "$IMAGE" --/renderer/gpuEnumeration/rtxRequired=false "$@"
