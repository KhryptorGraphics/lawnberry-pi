#!/bin/bash
# halow-lan-routes.sh - keep the LAN-via-HaLow routes in place (sole owner of these routes).
#
# Why: this Pi lives in the shed. HaLow (halow0, 192.168.50.20) is its working link; home WiFi
# (wlan0) is associated but near-dead (-83 dBm, ~43% loss on 2026-10-12), yet its on-link
# 192.168.1.0/24 and 192.168.1.1 routes (metric 600) would carry all LAN traffic (Thor link,
# DNS via 192.168.1.1). Two metric-500 routes via the Orange Pi AP (192.168.50.1, which NATs
# onto the LAN) win over them. A HaLow drop removes the DHCP address and the kernel flushes
# every halow0 route; halow-lan-routes.timer runs this every 15 s to put them back.
# Not a systemd-networkd [Route] drop-in: networkd reads drop-ins only at start or reload (a
# reload reconfigures halow0, the only working link), and in wifi-halow-admin's AP mode the Pi
# itself is 192.168.50.1, where that route would fail and could leave halow0 "failed".
#
# Safe at any time: acts only while halow0 holds 192.168.50.20 (station mode), adds a route only
# when it is missing, never deletes routes or touches the link. Output goes to the journal.
set -u

IFACE=halow0
ADDR=192.168.50.20
GW=192.168.50.1
METRIC=500

ip -4 -o address show dev "$IFACE" 2>/dev/null | grep -qF "inet $ADDR/" || exit 0

rc=0
for dst in 192.168.1.0/24 192.168.1.1; do
    [ -n "$(ip -4 route show "$dst" via "$GW" dev "$IFACE" metric "$METRIC")" ] && continue
    if ip -4 route replace "$dst" via "$GW" dev "$IFACE" proto static metric "$METRIC"; then
        echo "re-added missing route $dst via $GW dev $IFACE metric $METRIC"
    else
        echo "failed to add route $dst via $GW dev $IFACE metric $METRIC" >&2
        rc=1
    fi
done
exit "$rc"
