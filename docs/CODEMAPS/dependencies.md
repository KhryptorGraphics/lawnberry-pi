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

## Offline toolchain (workshop x86 and Thor; docs/autonomy-toolchain-matrix.md is the source of truth)
- **Isaac Sim:** nvcr.io/nvidia/isaac-sim:6.0.0 on the RTX 3080 Ti (scripts/isaacsim-headless.sh, isaacsim-rdp.sh).
- **Python env** (`$LB/env`): torch 2.7.1+cu126, transformers 4.57.1 (pinned), and ultralytics for YOLO26 training.
- **Auto-labelling:** MM Grounding DINO (workshop/autolabel.py).
- **Hailo:** Dataflow Compiler 3.34.0 (needs a Hailo login) and model zoo v2.19.0.
- **3D assets:** Objaverse CC-BY assets (workshop/objaverse_*.py).
