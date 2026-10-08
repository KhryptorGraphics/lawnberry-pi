<!-- Generated: 2026-10-06 | Files scanned: 474 | Token estimate: ~600 -->
# Data

## SQLite: `data/lawnberry.db` (core/persistence.py, 570 L)
| Table | Holds |
|---|---|
| schema_version | migration level |
| system_config | persisted settings blobs |
| map_zones, map_config | mowing zones and map configuration |
| planning_jobs | scheduled and planned mow jobs |
| telemetry_snapshots | periodic telemetry snapshots |
| hardware_telemetry_streams | per-component raw telemetry streams |
| audit_logs | operator and API audit trail |

Migrations are inline `CREATE TABLE IF NOT EXISTS` statements gated by schema_version. There is no ORM.

## Config files (`config/`, loaded at lifespan into app.state)
hardware.yaml, limits.yaml, tractor.yaml (dormant platform), logging.yaml, default.json, remote_access.json, maps_settings.example.json, nginx.conf, secrets.json (gitignored), plus `.env` (NTRIP_* and similar). On the device, `LAWNBERRY_CONFIG_DIR`, `_LOG_DIR` and `_DATA_DIR` override the locations.

## Pydantic models (`backend/src/models/`, about 45 files)
API and contract types:
- webui_contracts
- telemetry_exchange
- system_configuration
- navigation_state
- mower_data_frame
- zone
- mission
- tractor_control
- thor_link (UDP datagram schema)
- training_data
- user_session

## File datasets
- **W1 capture sessions:** `<session>/` holds {gps,imu,commands,camera}.jsonl plus JPEG frames on one monotonic clock. A session that fails the integrity gate is renamed `.rejected`.
- **Recording sessions:** routers/recording.py.
- **Workshop (offline, `$LB=/home/kp/nvme2/lawnberrypiserver` on the x86 server):** data/{captures,datasets,sdg,assets,models,hailo}. Each SDG dataset is `sdg/<version>/{images,labels,rejected,idpass}` plus `manifest.json` (seed, assets_sha256, per-class counts, rejected_frames, rejected_exposure).

## Detector class set
`workshop/classes.py`: 24 classes in hard-stop, then damage, structure, terrain, unknown order.
