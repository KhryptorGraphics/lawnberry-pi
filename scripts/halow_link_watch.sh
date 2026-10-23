#!/bin/bash
# halow-link-watch.sh - recover the shed Pi's only working link.
#
# The Pi's halow0 is a station on the Orange Pi AP. A drop is silent here: the address and
# routes stay, but nothing reaches the AP, so SSH and telemetry just stop. The escalation
# ladder is deliberate:
#   3 min down   -> restart wpa_supplicant@halow0 (rebinds the station; keeps DHCP and routes)
#   10 min down  -> reboot, at most once per REBOOT_COOLDOWN seconds
# It never touches the morse module: a live reload wedges the MM6108 and only a cold power
# cycle clears it (README-STATE.md on the AP). A reboot is the last lever this host has;
# if the card is wedged past that, the mower needs hands on it.
set -u
IFACE=${HALOW_WATCH_IFACE:-halow0}
AP=${HALOW_WATCH_AP:-192.168.50.1}
STATE=/var/lib/halow-link-watch
LOG=$STATE/link-watch.log
RELOAD_AFTER=${HALOW_WATCH_RELOAD_AFTER:-3}     # consecutive 60 s checks before restarting wpa_supplicant
REBOOT_AFTER=${HALOW_WATCH_REBOOT_AFTER:-10}    # consecutive checks before rebooting
REBOOT_COOLDOWN=${HALOW_WATCH_REBOOT_COOLDOWN:-14400}  # 4 h between reboots
DRY=${HALOW_WATCH_DRY_RUN:-0}
mkdir -p "$STATE"
exec 9>"$STATE/.lock"
flock -w 10 9 || exit 0
now=$(date +%s)
log() { echo "$(date -u -Is) $*" >> "$LOG"; }

down_file=$STATE/down
down=$(cat "$down_file" 2>/dev/null || echo 0)

# Healthy = the station is associated to our BSS and the AP answers.
assoc=$(timeout 5 iw dev "$IFACE" link 2>/dev/null | grep -c '^Connected to ')
if [ "$assoc" -gt 0 ] && timeout 5 ping -c1 -W2 -n "$AP" >/dev/null 2>&1; then
    if [ "$down" -gt 0 ]; then
        log "link back after $down failed check(s) ($(( down )) min)"
    fi
    echo 0 > "$down_file"
    echo 0 > "$STATE/reloads"
    exit 0
fi

down=$(( down + 1 ))
echo "$down" > "$down_file"
[ "$down" -eq 1 ] && log "link down: no $IFACE association to $AP"

if [ "$down" -ge "$RELOAD_AFTER" ]; then
    last_reload=$(cat "$STATE/last-reload" 2>/dev/null || echo 0)
    if [ $(( now - last_reload )) -ge 600 ]; then
        if [ "$DRY" = 1 ]; then
            log "DRY-RUN: would restart wpa_supplicant@$IFACE after $down failed check(s)"
        else
            log "restarting wpa_supplicant@$IFACE after $down failed check(s)"
            sync -f "$LOG" || true
            timeout 60 systemctl restart "wpa_supplicant@$IFACE" >> "$LOG" 2>&1 || \
                log "wpa_supplicant@$IFACE restart failed"
        fi
        echo "$now" > "$STATE/last-reload"
    fi
fi

if [ "$down" -ge "$REBOOT_AFTER" ]; then
    last_reboot=$(cat "$STATE/last-reboot" 2>/dev/null || echo 0)
    if [ $(( now - last_reboot )) -lt "$REBOOT_COOLDOWN" ]; then
        log "still down after $down checks; reboot suppressed (last reboot $(( (now - last_reboot) / 60 )) min ago)"
        exit 0
    fi
    if [ "$DRY" = 1 ]; then
        log "DRY-RUN: would reboot after $down failed check(s)"
        exit 0
    fi
    log "rebooting: link down for $down checks, last resort (no morse reload)"
    sync -f "$LOG" || true
    echo "$now" > "$STATE/last-reboot"
    systemctl reboot
fi
