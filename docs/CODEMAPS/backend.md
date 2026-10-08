<!-- Generated: 2026-10-06 | Files scanned: 196 (backend/src) | Token estimate: ~950 -->
# Backend (FastAPI, `backend/src/`)

## Middleware chain (main.py, applied in this order)
global rate limit → input_validation → security → api_key_auth → correlation → sanitization
(`middleware/*.py`, registered via `register_*` helpers). Auth is deliberately disabled project-wide.

## Router mounts (main.py)
| Prefix | Module | Main routes |
|---|---|---|
| /api/v2 | api/rest.py (891 L, legacy) | control/{drive,blade,emergency,emergency-stop}, map/{zones,locations}, hardware/robohat, docs/* |
| /api/v2 | routers/auth.py | auth/{login,logout,refresh,profile}, control/manual-unlock |
| /api/v2 | routers/telemetry.py | telemetry/{stream,export,ping}, dashboard/telemetry, **WS** ws/{telemetry,control,notifications,settings} |
| /api/v2 | routers/sensors.py | sensors/{gps,imu,tof,power,environmental,health}, dashboard/status, debug/sensors |
| /api/v2 | routers/maintenance.py | health/{liveness,readiness}, maintenance/imu, system/{selftest,timezone} |
| /api/v2/recording | routers/recording.py | start/stop/pause/resume/status, sessions[/{id}] |
| /api/v2/camera | routers/camera.py | status, start, frame, stream.mjpeg |
| /api/v2/ai | routers/ai_control.py | status, enable/disable, model, metrics, datasets |
| /api/v2 | routers/{maps,autonomy,planning,settings,weather} | map/configuration, navigation/{start,stop,pause,resume,return,status}, control/mode, planning/jobs, settings/*, weather/* |
| /api/v2 | routers/tractor.py | tractor/{state,authorize,revoke,command,left-lever,right-lever,throttle,blade,starter,stop-engine,emergency-stop,clear-emergency} |
| /api/v2 | routers/capture.py | capture/{start,stop,status} (W1 teleop logger) |
| /api/v2 | api/{motors,safety}.py | motors/drive, control/emergency_clear |
| own paths | api/{metrics,status,navigation,fusion,dashboard,health,mission}.py | /metrics, /health, /healthz, /api/v2/{mission,navigation,status,...} |
| /api/v1 | api/rest_v1.py | auth/login, status, maps/zones, mow/jobs (compat) |

## Services (`services/`, one long-lived object each, reached through `get_<x>_service()`)
| Service | Lines | Drives |
|---|---|---|
| sensor_manager | 1195 | drivers/sensors/{gps,bno085,vl53l0x,ultrasonic,ina3221,bme280,victron_vedirect,stereo_camera} |
| websocket_hub / telemetry_hub | 1126 / – | WS topic fan-out |
| navigation_service | 967 | nav/*, robohat; waypoint following, mission execution |
| robohat_service / motor_service | 733 / – | drivers/motor/robohat_rp2040 (UART) |
| blade_service | – | drivers/blade/ibt4_gpio |
| tractor_service | 318 | drivers/actuators/pca9685_driver (+ on_change hook → capture) |
| camera_stream_service | 1160 | camera + drivers/ai/{hailo_driver,yolo_postprocess} |
| ai_inference_service | – | drivers/ai/hailo_driver (HailoRT 4.24, hailo8l HEFs) |
| capture_service + capture_integrity | 470 / 78 | gps/imu/commands/camera JSONL sessions, integrity gate |
| thor_link_service | 172 | Pi↔Thor UDP protocol (models/thor_link.py); not yet wired into main.py |
| maps, mission, jobs, settings, weather, ntrip_client, acme, remote_access, perimeter_recorder, timezone, hw_selftest, calibration | – | domain logic |

## Safety (`safety/`, high-risk)
estop_handler, watchdog, interlock_validator, motor_authorization, safety_triggers, safety_monitor, tractor_safety_monitor, safety_validator (`validate_on_start`).

## Navigation algorithms (`nav/`, unit-testable)
coverage_patterns, coverage_planner, zone_coverage, path_planner, geofence_validator, obstacle_avoidance, odometry, gps_degradation, geoutils, location_features, thor_strategist.

## Core (`core/`)
config_loader, config, persistence (SQLite), ipc, message_bus, observability, health, logging, secrets_manager, env_validation, tls_status, robot_state_manager.
