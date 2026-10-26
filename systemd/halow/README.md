# Live-only HaLow and site units (reference copies)

Verbatim copies of the units that exist only on the deployed shed Pi (`kp@10.8.0.2`), read from
`/etc/systemd/system` and `/usr/local/sbin` read-only on 2026-10-25 (nothing was changed on the Pi).
They are **reference copies**: `install_services.sh` does not install them, because they describe one
site (HaLow radio `halow0`, AP `192.168.50.1`, Pi `192.168.50.20`) and would break any other host.

| File | What it is |
|---|---|
| `halow-bringup.service` | HaLow (Seeed WM6108) preflight and driver load; runs `/home/kp/halow/tools/halow-bringup.sh`, which is **not** copied here (it was not read) |
| `halow-failover.service`, `sbin/halow-wifi-failover.sh` | demote the WiFi default route when WiFi is associated but dead and HaLow answers |
| `halow-lan-routes.{service,timer}`, `sbin/halow-lan-routes.sh` | re-add the LAN-via-HaLow routes every 15 s |
| `wifi-halow-admin.service`, `wifi-halow-{hostapd,dnsmasq}.service` | legacy HaLow admin UI (:1337, runs from `/apps/lawnberry-pi/wifi-admin`, which is not in this repository) and its AP |
| `wificommand*.service` | HaLow command center (:1377) from `/opt/wificommand` (separate project); `ExecStart` has a user-specific node path |
| `wifi-halow-*.service.d/40-wificommand-ap.conf`, `wpa_supplicant@halow0.service.d/` | drop-ins that hand the radio between the two AP managers |
| `lawnberry-backend.service.d/thor-upload.conf` | live: `THOR_UPLOAD_ENABLED=1` (the repo unit defaults to `0`) |

Not copied: `/etc/hostapd/*`, `/etc/dnsmasq.d/*`, `/etc/wpa_supplicant/*` (they hold network
credentials) and `/home/kp/halow/tools/halow-bringup.sh`. The files here were scanned for
`pass`, `psk`, `secret`, `token` and `key`; the only hit is a comment word in `sbin/halow-wifi-failover.sh`.
