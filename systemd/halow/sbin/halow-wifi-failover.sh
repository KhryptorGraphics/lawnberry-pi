#!/bin/bash
# Prefer HaLow when WiFi is associated but not passing traffic (edge of WiFi range).
# Only the WiFi *default route* is demoted; the on-link LAN route stays, so SSH to
# 192.168.1.31 keeps working while WiFi is healthy enough to answer. Never demotes
# unless HaLow itself answers, so it cannot leave the host with no working path.
set -u
WIFI=${WIFI_IF:-wlan0}
HALOW_GW=${HALOW_GW:-192.168.50.1}
DEMOTED_METRIC=1500

demoted=0; bad=0; good=0; orig_metric=600

# prints number of replies out of 4 pings sent via interface $1 to $2
replies() { ping -I "$1" -c 4 -i 0.5 -W 1 -q "$2" 2>/dev/null | awk '/packets transmitted/ {print $4}' | head -1; }

while sleep 2; do
    route=$(ip -4 route show default dev "$WIFI" | head -1)
    if [[ -z $route ]]; then            # WiFi down / not associated: HaLow route already wins
        (( demoted )) && echo "wifi default route gone; resetting state"
        demoted=0; bad=0; good=0
        continue
    fi
    gw=$(awk '{for(i=1;i<NF;i++) if($i=="via") {print $(i+1); exit}}' <<<"$route")
    metric=$(awk '{for(i=1;i<NF;i++) if($i=="metric") {print $(i+1); exit}}' <<<"$route")
    [[ -z $gw ]] && continue
    r=$(replies "$WIFI" "$gw"); r=${r:-0}

    if (( ! demoted )); then
        if (( r <= 1 )); then
            bad=$((bad + 1)); good=0
            if (( bad >= 2 )); then
                h=$(replies halow0 "$HALOW_GW"); h=${h:-0}
                if (( h >= 2 )); then
                    orig_metric=${metric:-600}
                    ip route del default dev "$WIFI" 2>/dev/null
                    ip route add default via "$gw" dev "$WIFI" metric "$DEMOTED_METRIC"
                    demoted=1; bad=0
                    echo "WiFi dead ($r/4 to $gw): default route moved to HaLow (wifi metric $orig_metric -> $DEMOTED_METRIC)"
                fi
            fi
        else
            bad=0
        fi
    else
        # networkd may re-add the normal-metric route on a DHCP renewal: demote again
        if [[ $metric != "$DEMOTED_METRIC" ]]; then
            ip route del default dev "$WIFI" 2>/dev/null
            ip route add default via "$gw" dev "$WIFI" metric "$DEMOTED_METRIC"
        fi
        if (( r == 4 )); then
            good=$((good + 1))
            if (( good >= 3 )); then
                ip route del default dev "$WIFI" metric "$DEMOTED_METRIC" 2>/dev/null
                ip route add default via "$gw" dev "$WIFI" proto dhcp metric "$orig_metric"
                demoted=0; good=0
                echo "WiFi healthy again: default route restored to $WIFI (metric $orig_metric)"
            fi
        else
            good=0
        fi
    fi
done
