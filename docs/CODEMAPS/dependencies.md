<!-- Generated: 2026-10-06 | Files scanned: pyproject.toml, frontend/package.json, systemd/, workshop/requirements.txt | Token estimate: ~600 -->
# Dependencies

## Backend Python (pyproject.toml, Python 3.11)
- **Core:** fastapi, uvicorn, pydantic, websockets, httpx, python-multipart, PyYAML, structlog, python-dotenv, requests.
- **Auth/crypto:** PyJWT, passlib, pyotp, google-auth.
- **Geo/math:** shapely, timezonefinder, numpy, pillow.
- **Hardware:** smbus2, pyserial. The `[hardware]` extra adds adafruit-blinka, adafruit-circuitpython-vl53l0x, lgpio, python-periphery and RPi.GPIO. All hardware imports are lazy.
- **Tests:** pytest, pytest-asyncio. Lint and format with ruff only (never black).

## Frontend npm
vue, vue-router, pinia, axios, @vueuse/core, leaflet, @vue-leaflet/vue-leaflet, leaflet.gridlayer.googlemutant, @googlemaps/js-api-loader, chart.js, vue-chartjs, markdown-it, dompurify, vue-draggable-next. The server side uses express, http-proxy-middleware, compression and morgan.

## External services
- **RTK corrections:** NTRIP caster (services/ntrip_client.py, `NTRIP_*` env).
- **Maps:** Google Maps JS / tiles (`maps_settings`) and OpenStreetMap via Leaflet.
- **Weather:** core/weather_client.py with services/weather_service.py.
- **TLS:** Let's Encrypt ACME (services/acme_service.py, the cli/acme_renew timer).
- **Remote access:** services/remote_access_service.py.

## On-device runtime (Pi 5)
- **Accelerator:** Hailo-8L with HailoRT 4.24.0 (scripts/pi_build_hailort.sh, pi_install_hailort.sh); HEFs target `hailo8l`.
- **Camera:** libcamera/picamera2, owned by lawnberry-camera.service.
- **systemd units:**
  - lawnberry-{backend,frontend,camera,sensors,database,health,remote-access}.service
  - {acme-renew,cert-renewal,backup} timers
  - Never restart the backend remotely: doing so has taken HaLow down.

## HaLow network (Orange Pi AP, 192.168.1.140 / 192.168.50.1)
- **Notes of record:** `~/halow-build/README-STATE.md` on the Orange Pi; tested numbers are in docs/autonomy-toolchain-matrix.md.
- **Radio:** never reload `morse` live (`rmmod`/`modprobe`, warm reset); that wedges the card and can reset the host. TX cap 20 dBm in `/etc/modprobe.d/halow-morse.conf`; changes need a cold power cycle. `halow-guard` only stops the AP on an SPI error storm; it no longer unloads `morse`.
- **Failure evidence:** `/var/lib/halow-diag/` (SD card): `health.log` (UP/DOWN/BOOT), `evidence-*.log`, `kernel.log`, `guard.log`, `client-watch.log`.
- **Client watch (`halow-client-watch.timer`, 60 s):** records when the Pi's station disappears (`client-watch.log`, `client-watch.lastseen`) and repairs two AP-side failures: a mute association (station present but neither ICMP nor ARP answered) is deauthenticated after 3 checks, cooldown 10 min; zero stations for 3 checks restarts `halow-ap.service`, cooldown 30 min. Both stop once the client has been gone for 2 h, and neither reloads `morse`. A powered-off Pi is not fixed by either; that needs a cold power cycle.
- **/var/log is a 47 MB zram:** atop/sysstat are pruned by the date in the file name (`halow-log-prune.timer`), and the journal is capped at 8 MB. A full /var/log floods dmesg and destroys driver evidence.
- **Shed Pi 5:** HaLow is its only working link. On the Pi 5, `halow-lan-routes.timer` routes 192.168.1.0/24 via HaLow. Pressing Apply/Rollback in `wifi-halow-admin` (`:1337`) reloads networkd and restarts halow0, briefly cutting the Pi off.
- **Thor:** reaches 192.168.50.0/24 only through the route `192.168.50.0/24 via 192.168.1.140` in NetworkManager "Wired connection 2". Docker's unused `app-network` (192.168.48.0/20) overlaps it. Other LAN hosts use `ssh -J orangepi@192.168.1.140` or `192.168.1.140:2222` / `:1337`.

## Offline toolchain (workshop x86 and Thor; docs/autonomy-toolchain-matrix.md is the source of truth)
- **Isaac Sim:** nvcr.io/nvidia/isaac-sim:6.0.0 on the RTX 3080 Ti (scripts/isaacsim-headless.sh, isaacsim-rdp.sh).
- **Python env** (`$LB/env`): torch 2.7.1+cu126, transformers 4.57.1 (pinned), and ultralytics for YOLO26 training.
- **Auto-labelling:** MM Grounding DINO (workshop/autolabel.py).
- **Hailo:** Dataflow Compiler 3.34.0 (needs a Hailo login) and model zoo v2.19.0.
- **3D assets:** Objaverse CC-BY assets (workshop/objaverse_*.py).
