import asyncio
import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI

from .api.dashboard import router as dashboard_router
from .api.fusion import router as fusion_router
from .api.health import router as health_router
from .api.metrics import router as metrics_router
from .api.mission import router as mission_router
from .api.motors import router as motors_router
from .api.navigation import router as navigation_router
from .api.rest import legacy_router as rest_legacy_router
from .api.rest import router as rest_router
from .api.rest_v1 import router as rest_v1_router
from .api.routers import ai_control as ai_control_router
from .api.routers import auth as auth_router
from .api.routers import autonomy as autonomy_router
from .api.routers import camera as camera_router
from .api.routers import capture as capture_router
from .api.routers import maintenance as maintenance_router
from .api.routers import maps as maps_router
from .api.routers import planning as planning_router
from .api.routers import recording as recording_router
from .api.routers import sensors as sensors_router
from .api.routers import settings as settings_router
from .api.routers import telemetry as telemetry_router
from .api.routers import tractor as tractor_router
from .api.routers import weather as weather_router
from .api.safety import router as safety_router
from .api.status import router as status_router
from .core.auth_posture import log_auth_posture
from .core.config_loader import ConfigLoader
from .core.env_validation import validate_environment
from .core.simulation import is_simulation_mode
from .middleware.api_key_auth import register_api_key_auth_middleware
from .middleware.correlation import register_correlation_middleware
from .middleware.input_validation import register_input_validation_middleware
from .middleware.rate_limiting import register_global_rate_limiter
from .middleware.sanitization import register_sanitization_middleware
from .middleware.security import register_security_middleware
from .nav.gps_degradation import GPSDegradationConfig, GPSDegradationMonitor
from .safety.estop_handler import EstopHandler
from .safety.obstacle_stop import ObstacleStopMonitor, configure_obstacle_guard
from .safety.safety_monitor import get_safety_monitor
from .safety.safety_triggers import set_safety_event_handler
from .safety.safety_validator import validate_on_start
from .safety.tractor_safety_monitor import TractorSafetyMonitor
from .safety.watchdog import ACTUATION_SOURCE, SAFETY_MONITOR_SOURCE, Watchdog
from .services.blade_service import get_blade_service
from .services.camera_stream_service import camera_service
from .services.emergency_stop_service import execute_emergency_stop, park_actuators
from .services.navigation_service import NavigationService
from .services.robohat_service import initialize_robohat_service, shutdown_robohat_service
from .services.thor_link_wiring import ThorLinkStartError, start_thor_link, stop_thor_link
from .services.thor_plan_follower import gnss_from_sensor_manager
from .services.tractor_service import get_tractor_service
from .services.websocket_hub import websocket_hub

# Load .env early so secrets like NTRIP_* are available under systemd
try:
    from dotenv import load_dotenv

    # Load from project root working directory
    load_dotenv(dotenv_path=os.path.join(os.getcwd(), ".env"), override=False)
except Exception:
    # Safe to continue without .env
    pass

_log = logging.getLogger(__name__)


async def _fail_closed(app: FastAPI, tractor) -> None:
    """The tractor's safety wiring did not come up: stop it and refuse every command.

    Without this an exception during start-up left the actuation API live with no watchdog
    and no tilt monitor (audit N-05). The lock-out lasts until the process restarts; the
    E-stop drives blade off / levers neutral / throttle idle and is reported if unconfirmed.
    """
    tractor.lock_out("safety wiring failed at start-up (see the backend log)")
    try:
        # The IBT-4 blade is a separate output: a failed safety wiring must switch it off too.
        get_blade_service().request_blade_off("start-up safety wiring failed")
    except Exception:
        _log.exception("Start-up fail-closed: switching the IBT-4 blade off failed")
    try:
        watchdog = getattr(app.state, "tractor_watchdog", None)
        if watchdog is not None:
            await watchdog.stop()  # the lock-out and the latch below replace its repeated stops
        await tractor.stop_supervision()
        report = await tractor.emergency_stop(source="startup_failure")
        if not report.get("complete"):
            _log.critical("Start-up fail-closed stop NOT confirmed: %s", report.get("faults"))
    except Exception:
        _log.exception("Start-up fail-closed stop failed")


async def _watchdog_stop() -> dict:
    """What a watchdog trip does: the all-platform E-stop, recorded (audit SAF-01 gap, round 3).

    The trip used to call ``tractor.emergency_stop`` alone. On a platform whose tractor is
    disabled that answers ``not_enabled``: nothing is stopped, no stop event is recorded (so a
    running mission goes on) and the platform that IS driving, the RoboHAT, never hears of it;
    the IBT-4 blade service only heard of a tractor stop. A trip means the control path that
    keeps the machine safe stopped proving it can: the whole machine stops, whichever platforms
    exist. ``execute_emergency_stop`` fans out to the tractor (when enabled), the RoboHAT and
    the IBT-4 concurrently and independently, latches the process E-stop, records the stop
    (``safety.stop_events``: every autonomous run ends, a clear never revives one), persists the
    latch, and reports what each output actually did. The watchdog repeats it every second
    while a heartbeat stays silent, and the general clear is refused while a repeat runs.
    """
    return await execute_emergency_stop(source="watchdog")


#: How long shutdown waits for cancelled mission/navigation tasks to finish before it parks.
AUTONOMY_STOP_TIMEOUT_S = 3.0


async def _stop_autonomy() -> None:
    """Stop every loop that commands the actuators, before they are parked (audit SF-06, SEC-08).

    A mission or the AI loop is a plain background task that re-commands levers and throttle
    every few hundred ms. Parking while one still runs let its next command land after the
    park, and the PCA9685 then holds that pulse after the process exits. The services' park
    latches refuse non-neutral commands from anything this misses (e.g. a ``return_home`` job
    task, which nothing tracks); stopping the known loops first keeps them from spinning
    against that refusal. Never raises; bounded by ``AUTONOMY_STOP_TIMEOUT_S``.
    """
    from .models import NavigationMode
    from .services.mission_service import current_mission_service

    nav = NavigationService.get_instance()
    mission_tasks: list[asyncio.Task] = []
    service = current_mission_service()
    if service is not None:
        # Cancel the tasks themselves: MissionService's own done-callback records "aborted".
        mission_tasks = list(service.mission_tasks.values())
        for task in mission_tasks:
            task.cancel()
    try:
        async with asyncio.timeout(AUTONOMY_STOP_TIMEOUT_S):
            if nav.navigation_state.navigation_mode in (
                NavigationMode.AI,
                NavigationMode.AI_ASSISTED,
            ):
                await nav.stop_ai_navigation()
            await nav.stop_navigation()
    except Exception:
        _log.exception("Stopping navigation on shutdown failed")
    pending = [t for t in mission_tasks if not t.done()]
    if pending:
        _done, still_running = await asyncio.wait(pending, timeout=AUTONOMY_STOP_TIMEOUT_S)
        if still_running:
            _log.critical(
                "%d mission task(s) still running after cancellation; the actuator park latches "
                "will refuse their commands",
                len(still_running),
            )


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    # Say, before anything starts, every way this process is set to skip authentication
    # (OPERATOR_AUTH_REQUIRED disabled, SIM_MODE on a production or exposed process).
    log_auth_posture()
    # Load configuration (hardware + safety limits) and attach to app.state
    loader = ConfigLoader()
    hardware_cfg, safety_limits = loader.get()
    app.state.config_loader = loader
    app.state.hardware_config = hardware_cfg
    app.state.safety_limits = safety_limits
    websocket_hub.bind_app_state(app.state)
    # IBT-4 blade: lift the shutdown park and, when armed, claim both inputs LOW and read them
    # back before anything can command them (a restart leaves them in an unknown state).
    try:
        await get_blade_service().startup()
    except Exception:
        _log.exception("IBT-4 blade start-up claim failed")

    # Small boot hook: ensure dual VL53L0X are addressed uniquely via XSHUT (left 0x29, right default 0x30)  # noqa: E501
    try:
        if not is_simulation_mode():
            tc = getattr(hardware_cfg, "tof_config", None)
            if (
                tc
                and getattr(tc, "left_shutdown_gpio", None) is not None
                and getattr(tc, "right_shutdown_gpio", None) is not None
            ):
                try:
                    # Lazy import to keep CI SIM-safe
                    from .drivers.sensors.vl53l0x_driver import (
                        ensure_pair_addressing,  # type: ignore
                    )

                    right_addr = getattr(tc, "right_address", 0x30) or 0x30
                    ok = await ensure_pair_addressing(
                        tc.left_shutdown_gpio, tc.right_shutdown_gpio, right_addr=int(right_addr)
                    )
                    _log.info(
                        "ToF XSHUT pair addressing %s (left_gpio=%s right_gpio=%s right_addr=0x%x)",
                        "completed" if ok else "skipped",
                        tc.left_shutdown_gpio,
                        tc.right_shutdown_gpio,
                        int(right_addr),
                    )
                except Exception as e:  # pragma: no cover - hardware dependent
                    _log.warning("ToF pair addressing failed: %s", e)
    except Exception:
        pass
    # GNSS-quality monitor (audit N-31): fed the real GpsReading by the navigation service,
    # read by every autonomous motion path (NavigationService._motion_gate), and, while the
    # fix is degraded and autonomy runs, stops the machine at the actuators every cycle.
    nav_service = NavigationService.get_instance()
    app.state.gps_deg_monitor = GPSDegradationMonitor(
        GPSDegradationConfig.from_limits(safety_limits.gps_degradation),
        autonomy_active=nav_service.autonomy_active,
        hold_handlers=(nav_service.hold_for_gnss,),
    )
    await app.state.gps_deg_monitor.start()

    # Validate safety limits at startup and attach report
    try:
        ok, report = validate_on_start(loader)
        app.state.safety_validation = report
    except Exception:
        _log.exception("Safety limits validation failed at startup")

    # Initialize safety monitor and wire interlock event bridge
    monitor = get_safety_monitor()

    async def _event_bridge(action, interlock):
        try:
            await monitor.handle_interlock_event(action, interlock)
        except Exception:
            pass

    # wrap event bridge to work with sync trigger calls
    def _handler(action, interlock):
        import asyncio as _asyncio

        try:
            loop = _asyncio.get_running_loop()
            loop.create_task(monitor.handle_interlock_event(action, interlock))
        except RuntimeError:
            _asyncio.run(monitor.handle_interlock_event(action, interlock))

    set_safety_event_handler(_handler)

    # Obstacle stop (audit N-11, WP-P4-01/02): ONE guard, read by the navigation loops and by
    # the tractor's actuation chokepoint, fed by a fixed-rate monitor from the real ToF /
    # ultrasonic readings. It is wired before anything can actuate. Fail-closed: a required
    # sensor that is stale or missing holds autonomous motion (manual operator drive is not
    # limited), and if the monitor dies the readings age out and the guard holds by itself.
    obstacle_guard = configure_obstacle_guard(safety_limits)
    guard_tractor = get_tractor_service()
    nav_service.attach_safety(obstacle_guard=obstacle_guard, gps_monitor=app.state.gps_deg_monitor)
    guard_tractor.motion_limiter = obstacle_guard
    obstacle_monitor = ObstacleStopMonitor(
        obstacle_guard,
        sensor_provider=lambda: getattr(app.state, "sensor_manager", None),
        hold_handlers=(nav_service.enforce_motion_hold, guard_tractor.enforce_motion_hold),
    )
    await obstacle_monitor.start()
    app.state.obstacle_guard = obstacle_guard
    app.state.obstacle_monitor = obstacle_monitor
    # Initialize hardware services (best-effort; keep SIM-safe)
    try:
        await initialize_robohat_service()
    except Exception:
        pass
    # Durable latches (audit SAF-16): whatever E-stop or interlock latch the previous process
    # left set is set again before anything can actuate. It needs the same authenticated,
    # explicit clear as any other latch, and authorization is never restored. A file that
    # cannot be read fails closed (every latch is assumed set).
    try:
        from .safety import latch_store
        from .services.robohat_service import get_robohat_service

        latch_store.restore(tractor=get_tractor_service(), robohat=get_robohat_service())
    except Exception:
        _log.exception("Restoring the persisted E-stop latches failed")
    app.state.tractor_watchdog = None  # not wired until the block below succeeds
    app.state.tractor_safety_monitor = None
    tractor = None
    # True only once the watchdog, the actuation supervision and the tilt monitor have all
    # STARTED (audit SAF-12): the monitor object existing is not the monitor running.
    tractor_monitor_started = False
    try:
        tractor = get_tractor_service()
        if tractor.enabled:
            await tractor.initialize()

            # Constitution Principle VI follow-up / audit N-04. The watchdog requires TWO
            # independent heartbeats, each fed by the code that does the work, after it did
            # it: the tractor's own actuation self-check (servo outputs read back from the
            # PCA9685, no write fault, deadman running, a latched stop really safe) and the
            # tilt monitor's completed cycle. A timeout revokes the tractor's own
            # authorization and awaits the all-platform E-stop (``_watchdog_stop``: the
            # tractor's blade off, levers neutral and throttle idle, and the RoboHAT and the
            # IBT-4 too), repeated every second while a heartbeat stays silent. A HARDWARE
            # watchdog is still required (audit O-7): this one shares the event loop it
            # supervises and cannot act if that loop or the process hangs. It exists only for
            # the tractor platform: nothing feeds a heartbeat on the differential platform,
            # whose failsafe is the RoboHAT firmware's own 2 s USB timeout.

            # A window shorter than a few self-check periods would trip on a healthy machine.
            min_window_ms = 3000.0 * tractor.supervise_interval_s
            if safety_limits.watchdog_timeout_ms < min_window_ms:
                raise ValueError(
                    f"watchdog_timeout_ms={safety_limits.watchdog_timeout_ms} is shorter than "
                    f"three actuation self-check periods ({min_window_ms:.0f} ms)"
                )

            tractor_watchdog = Watchdog(
                EstopHandler(tractor),
                timeout_ms=safety_limits.watchdog_timeout_ms,
                on_timeout=_watchdog_stop,
                required_sources=(ACTUATION_SOURCE, SAFETY_MONITOR_SOURCE),
            )
            app.state.tractor_watchdog = tractor_watchdog
            tractor.attach_heartbeat(lambda: tractor_watchdog.heartbeat(ACTUATION_SOURCE))
            app.state.tractor_safety_monitor = TractorSafetyMonitor(
                tractor,
                tractor_watchdog,
                tilt_threshold_degrees=safety_limits.tilt_threshold_degrees,
            )
            await tractor_watchdog.start()
            await tractor.start_supervision()
            await app.state.tractor_safety_monitor.start()
            tractor_monitor_started = True
    except Exception:
        _log.exception("Tractor service initialization failed")
        if tractor is not None and tractor.enabled:
            await _fail_closed(app, tractor)
    # Thor link (default OFF, config/thor_link.yaml). Needs the tractor platform and its safety
    # monitor up first: plans from the Thor go only through the local navigation path and the
    # tractor interlocks, and never start unless tilt cutoff and the watchdog are running.
    # Any failure leaves the mower on local control only; it never starts a weaker link.
    app.state.thor_link = None
    app.state.thor_link_error = None
    try:
        app.state.thor_link = await start_thor_link(
            get_tractor_service(),
            safety_monitor_running=tractor_monitor_started,
            gnss=lambda: gnss_from_sensor_manager(getattr(app.state, "sensor_manager", None)),
        )
    except Exception as exc:
        app.state.thor_link_error = str(exc)
        # A refusal (ThorLinkStartError) says why in one line; anything else is a bug and keeps
        # its traceback instead of leaving a one-line message to debug from.
        _log.critical(
            "Thor link NOT started, the mower stays on local control only: %s",
            exc,
            exc_info=not isinstance(exc, ThorLinkStartError),
        )
    # camera-stream.service is the sole camera owner in production (the
    # constitution reserves the camera to it; the backend consumes frames
    # over its IPC socket, see api/routers/camera.py). Only embed a camera
    # instance here in SIM_MODE, where there's no real device to contend
    # for and other code paths expect simulated telemetry.camera data.
    if is_simulation_mode():
        try:
            await camera_service.initialize()
            await camera_service.start_streaming()
        except Exception:
            pass
    await websocket_hub.start_telemetry_loop()
    yield
    # Shutdown
    # Everything that can command motion is stopped ahead of the park, or it could undo it: the
    # Thor link first (bounded by the actuator timeouts, a few seconds at worst), then the
    # mission and navigation loops.
    await stop_thor_link(getattr(app.state, "thor_link", None))
    await _stop_autonomy()
    # Park the actuators FIRST, before any monitor/service is torn down: blade off,
    # levers neutral, throttle idle, RoboHAT neutral + blade off. The PCA9685 holds
    # its last pulse and a GPIO relay pin's state at process exit is unspecified, so
    # shutdown must command the safe state explicitly instead of relying on either.
    # (Skipped for a SIGKILL/power loss -- only independent hardware covers those.)
    try:
        park_report = await park_actuators(source="shutdown")
        if park_report["faults"]:
            _log.critical("Shutdown park NOT fully confirmed: %s", park_report["faults"])
    except Exception:
        _log.exception("Actuator park on shutdown failed")
    set_safety_event_handler(None)
    # Detach the guard so a later lifespan (or any code reusing these singletons) can never
    # inherit a monitor-less guard that would hold motion for ever.
    if getattr(app.state, "obstacle_monitor", None):
        await app.state.obstacle_monitor.stop()
        NavigationService.get_instance().attach_safety(obstacle_guard=None)
        guard_tractor.motion_limiter = None
    if getattr(app.state, "gps_deg_monitor", None):
        await app.state.gps_deg_monitor.stop()
    # Watchdog first, then its feeders: stopping a feeder under a live watchdog would trip it.
    if getattr(app.state, "tractor_watchdog", None):
        await app.state.tractor_watchdog.stop()
    try:
        await get_tractor_service().stop_supervision()
    except Exception:
        _log.exception("Stopping the actuation supervision failed")
    if getattr(app.state, "tractor_safety_monitor", None):
        await app.state.tractor_safety_monitor.stop()
    await websocket_hub.stop_telemetry_loop()
    if is_simulation_mode():
        try:
            await camera_service.shutdown()
        except Exception:
            pass
    try:
        await shutdown_robohat_service()
    except Exception:
        pass


app = FastAPI(
    title="LawnBerry Pi v2",
    description="Autonomous robotic lawn mower backend API",
    version="2.0.0",
    lifespan=lifespan,
)

# Validate environment early
try:
    if not validate_environment():
        _log.error(
            "Environment validation failed; service will continue but endpoints may be restricted"
        )
except Exception:
    _log.exception("Environment validation crashed")

# Middleware order: global limit -> input validation -> security -> API key -> correlation -> sanitization  # noqa: E501
register_global_rate_limiter(app)
register_input_validation_middleware(app)
register_security_middleware(app)
register_api_key_auth_middleware(app)
register_correlation_middleware(app)
register_sanitization_middleware(app)

app.include_router(rest_router, prefix="/api/v2")
app.include_router(auth_router.router, prefix="/api/v2")
app.include_router(telemetry_router.router, prefix="/api/v2")
app.include_router(sensors_router.router, prefix="/api/v2")
app.include_router(maintenance_router.router, prefix="/api/v2")
app.include_router(recording_router.router, prefix="/api/v2/recording", tags=["recording"])
app.include_router(camera_router.router, prefix="/api/v2/camera", tags=["camera"])
app.include_router(ai_control_router.router, prefix="/api/v2/ai", tags=["ai"])
app.include_router(maps_router.router, prefix="/api/v2", tags=["maps"])
app.include_router(autonomy_router.router, prefix="/api/v2", tags=["autonomy"])
app.include_router(tractor_router.router, prefix="/api/v2", tags=["tractor"])
app.include_router(capture_router.router, prefix="/api/v2", tags=["capture"])
app.include_router(settings_router.router, prefix="/api/v2", tags=["settings"])
app.include_router(weather_router.router, prefix="/api/v2", tags=["weather"])
app.include_router(planning_router.router, prefix="/api/v2", tags=["planning"])
app.include_router(rest_legacy_router)
app.include_router(metrics_router)
app.include_router(status_router)
app.include_router(navigation_router)
app.include_router(motors_router, prefix="/api/v2")
app.include_router(safety_router, prefix="/api/v2")
app.include_router(fusion_router)
app.include_router(dashboard_router)
app.include_router(health_router)
app.include_router(mission_router)

app.include_router(rest_v1_router, prefix="/api/v1")


# Root endpoint with API information
@app.get("/")
def root():
    return {
        "service": "LawnBerry Pi v2 Backend API",
        "version": "2.0.0",
        "status": "running",
        "api_docs": "/docs",
        "api_redoc": "/redoc",
        "api_endpoints": {
            "health": "/health",
            "api_v2": "/api/v2",
            "websocket_telemetry": "/api/v2/ws/telemetry",
        },
        "key_endpoints": [
            "/api/v2/health/liveness",
            "/api/v2/health/readiness",
            "/api/v2/dashboard/status",
            "/api/v2/dashboard/telemetry",
            "/api/v2/docs/list",
        ],
    }
