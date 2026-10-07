#!/usr/bin/env bash
# Install a from-source HailoRT stack on the mower Pi (built by scripts/pi_build_hailort.sh).
#
#   ssh -J orangepi@192.168.1.140 kp@192.168.50.20
#   bash ~/pi_install_hailort.sh 4.24.0 [BUILD_DIR]     # BUILD_DIR default ~/hailo-build/<ver>
#
# WARNING: run this only with physical access to the Pi. It stops lawnberry-backend,
# which also stops lawnberry-camera (PartOf=). On 2026-10-06 the HaLow radio failed
# ~10 s after such a stop (morse_spi cmd53_read ret:-71) and the Pi was unreachable
# until power-cycled; HaLow is the only remote path to it.
#
# Installs, in order: PCIe driver module (DKMS-managed path), libhailort, hailortcli,
# the Hailo-8 firmware image, then reloads the driver and restarts the services that
# hold /dev/hailo0. Everything is reversible:
#
#   - apt pins: hailo-dkms/hailofw/hailort are held (added here if missing) so the
#     Raspberry Pi archive's 4.20.0 packages never overwrite them. Reinstalling or
#     removing those apt packages WILL overwrite /lib/firmware/hailo and
#     /usr/lib/libhailort.so.* -- the pins are what prevent that.
#   - rollback to 4.20.0:
#       sudo rm -f /usr/local/bin/hailortcli
#       sudo rm -f /usr/lib/aarch64-linux-gnu/libhailort.so.4.24.0
#       sudo ln -sfn /usr/lib/libhailort.so.4.20.0 /usr/lib/libhailort.so
#       sudo rm -f /usr/lib/aarch64-linux-gnu/libhailort.so
#       sudo ldconfig
#       sudo update-alternatives --set hailo8_fw.bin /lib/firmware/hailo/hailo8_fw.4.20.0.bin
#       sudo cp -a ~/hailo-backup/hailo_pci.ko.xz /lib/modules/$(uname -r)/updates/dkms/
#       sudo rmmod hailo_pci && sudo modprobe hailo_pci
#       sudo apt install --reinstall hailort=4.20.0-1 hailofw=4.20.0-1 hailo-dkms=4.20.0-1
#
# Firmware: the 4.24 CLI has no `fw-control upgrade` (only `identify`); `fw-update` is
# for flash-based boards. On this M.2 module the hailo_pci driver loads
# /lib/firmware/hailo/hailo8_fw.bin at probe (request_firmware_direct), so switching
# firmware means installing the versioned .bin, repointing the update-alternatives
# link, and reloading the module.
set -euo pipefail

VER="${1:?usage: $0 VERSION [BUILD_DIR]}"
BUILD="${2:-$HOME/hailo-build/$VER}"
KREL="$(uname -r)"
MULTIARCH=/usr/lib/aarch64-linux-gnu
KO_SRC="$BUILD/hailort-drivers/linux/pcie/hailo_pci.ko"
FW_SRC="$BUILD/hailo8_fw.$VER.bin"
[[ -f $FW_SRC ]] || FW_SRC="$BUILD/hailort-drivers/hailo8_fw.$VER.bin"
LIB_SRC="$BUILD/hailort/build/hailort/libhailort/src/libhailort.so.$VER"
CLI_SRC="$BUILD/hailort/build/hailort/hailortcli/hailortcli"
DKMS_KO="/lib/modules/$KREL/updates/dkms/hailo_pci.ko.xz"
SVC=(hailort.service lawnberry-backend.service)

log() { echo "[hailo-install] $*"; }
die() { echo "[hailo-install] $*" >&2; exit 1; }
su_() { echo "${SUDO_PASS:-}" | sudo -S -p "" "$@"; }

for f in "$KO_SRC" "$FW_SRC" "$LIB_SRC" "$CLI_SRC"; do
  [[ -e $f ]] || die "missing build artifact: $f (run scripts/pi_build_hailort.sh $VER first)"
done
[[ -e $DKMS_KO ]] || die "no DKMS module for kernel $KREL"

log "stopping services that hold /dev/hailo0"
su_ systemctl stop "${SVC[@]}" || true

log "apt pins"
held="$(su_ apt-mark showhold | tr '\n' ' ')"
need=""
for p in hailo-dkms hailofw hailort; do
  case " $held " in *" $p "*) ;; *) need="$need $p" ;; esac
done
if [[ -n $need ]]; then
  # shellcheck disable=SC2086
  su_ apt-mark hold $need
fi

log "backup + install kernel module (DKMS path, same filename as the 4.20 package)"
su_ mkdir -p "$HOME/hailo-backup"
su_ cp -a "$DKMS_KO" "$HOME/hailo-backup/hailo_pci.ko.xz"
xz -kc "$KO_SRC" >"/tmp/hailo_pci.$VER.ko.xz"
su_ mv -f "/tmp/hailo_pci.$VER.ko.xz" "$DKMS_KO"

log "install libhailort $VER + hailortcli (shadows /usr/bin/hailortcli; apt copy untouched)"
su_ cp -a "$LIB_SRC" "$MULTIARCH/"
su_ ln -sfn "libhailort.so.$VER" "$MULTIARCH/libhailort.so"
# /usr/local/bin/hailort_service came from the 4.20 build and is linked against
# /usr/lib/libhailort.so.4.20.0 via the /usr/lib/libhailort.so soname link; repoint
# that link at the new library so the service and the CLI agree.
su_ ln -sfn "$MULTIARCH/libhailort.so.$VER" /usr/lib/libhailort.so
su_ ldconfig
su_ install -m 755 "$CLI_SRC" /usr/local/bin/hailortcli

log "install firmware $VER and point hailo8_fw.bin at it"
su_ cp -a "$FW_SRC" "/lib/firmware/hailo/hailo8_fw.$VER.bin"
su_ update-alternatives --install /lib/firmware/hailo/hailo8_fw.bin hailo8_fw.bin \
  "/lib/firmware/hailo/hailo8_fw.$VER.bin" 500
su_ update-alternatives --set hailo8_fw.bin "/lib/firmware/hailo/hailo8_fw.$VER.bin"

log "reload driver (unconditional: a stale module in RAM keeps reporting 4.20)"
su_ rmmod hailo_pci 2>/dev/null || true
su_ modprobe hailo_pci
sleep 2
[ "$(su_ modinfo hailo_pci | awk '/^version:/{print $2}')" = "$VER" ] \
  || die "loaded module version is not $VER"
log "verify"
hailortcli --version | head -1
ident="$(hailortcli fw-control identify)"
echo "$ident"
echo "$ident" | grep -q "Firmware Version: $VER" || die "firmware did not switch to $VER"
ldd /usr/local/bin/hailortcli | grep -q "libhailort.so.$VER" || die "hailortcli not on $VER lib"
su_ dmesg | grep -i hailo | tail -10 || true

log "restart services"
su_ systemctl start "${SVC[@]}" || true
systemctl is-active "${SVC[@]}" || true
log "done: HailoRT $VER"
