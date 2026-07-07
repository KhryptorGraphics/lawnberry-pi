// Types mirroring real backend response shapes for control/hardware/tractor/autonomy
// endpoints. See backend/src/api/rest.py, backend/src/services/robohat_service.py,
// backend/src/services/tractor_service.py, backend/src/models/tractor_control.py,
// backend/src/services/autonomy_service.py, backend/src/api/routers/maintenance.py.

// POST /control/drive, /control/emergency (declared response_model=ControlResponseV2)
export interface ControlResponseV2 {
  accepted: boolean
  audit_id: string
  result: string
  status_reason?: string | null
  watchdog_echo?: string | null
  watchdog_latency_ms?: number | null
  safety_checks: string[]
  active_interlocks: string[]
  remediation?: Record<string, string> | null
  telemetry_snapshot?: Record<string, unknown> | null
  timestamp: string
}

// POST /control/blade — no response_model; runtime-branches between these two shapes
export interface BladeAcceptedResponse {
  accepted: boolean
  audit_id: string
  result: 'accepted' | 'rejected'
  timestamp: string
}

export interface BladeBlockedResponse {
  accepted: false
  audit_id: string
  result: 'blocked'
  status_reason: string
  remediation_url: string
  safety_checks: string[]
  active_interlocks: string[]
  timestamp: string
}

export type BladeCommandResponse = BladeAcceptedResponse | BladeBlockedResponse

// POST /control/emergency-stop (general mower E-stop alias, distinct from /tractor/emergency-stop)
export interface EmergencyStopResult {
  emergency_stop_active: boolean
  motors_stopped: boolean
  blade_disabled: boolean
  remediation: { message: string; docs_link: string }
}

// GET /hardware/robohat — loosely-typed union at the network boundary by design: the
// dataclass-only fields are present when the service is initialized and absent from
// the smaller "not initialized" fallback shape; the last two fields are injected by
// the route handler on top of the dataclass.
export interface RoboHATStatus {
  firmware_version: string
  uptime_seconds: number
  watchdog_active: boolean
  serial_connected: boolean
  health_status: string
  timestamp: string | null
  watchdog_heartbeat_ms: number | null
  safety_state: 'emergency_stop' | 'nominal'
  last_watchdog_echo?: string | null
  watchdog_latency_ms?: number
  error_count?: number
  last_error?: string | null
  motor_controller_ok?: boolean
  encoder_feedback_ok?: boolean
}

// Autonomy/navigation surface — autonomy_service.py methods return dict[str, Any], not
// Pydantic models; these mirror the concrete shape each one actually returns.
export interface AutonomyStartResult {
  status: 'started'
  mode: 'autonomous'
  mission_id: string
  waypoint_count: number
}
export interface AutonomyStopResult {
  status: 'stopped'
  mode: 'idle'
  mission_id: string | null
}
export interface AutonomyPauseResult {
  status: 'idle' | 'paused'
  mode: 'idle' | 'paused'
  mission_id: string | null
}
export interface AutonomyResumeResult {
  status: 'idle' | 'running'
  mode: 'idle' | 'autonomous'
  mission_id: string | null
}
export interface AutonomyReturnResult {
  status: 'returning' | 'error'
  mode: 'return_home' | 'idle'
  mission_id: string | null
  detail: string | null
}
export interface AutonomyStatusResult {
  mode: 'autonomous' | 'paused' | 'idle'
  active: boolean
  mission_id: string | null
  mission_status: string | null
  completion_percentage: number
}
// POST /control/mode — delegates to the autonomy service then overwrites `mode`
export type ControlModeResult = { status: string; mode: 'manual' | 'autonomous' | 'idle' } & Record<
  string,
  unknown
>

// Ride-on tractor platform
export type Transmission = 'forward' | 'neutral' | 'reverse'
export type EngineState = 'off' | 'starting' | 'running'

// GET /tractor/state — Pydantic model, serialized 1:1 via model_dump(mode="json")
export interface TractorState {
  steering: number
  throttle: number
  ground_speed: number
  gear: Transmission
  clutch: number
  blade_engaged: boolean
  engine: EngineState
  emergency_stop_active: boolean
  authorized: boolean
  interlock_reason: string | null
  last_updated: string
}

// POST /tractor/{steering,throttle,speed,clutch,gear,blade,starter,stop-engine,
// emergency-stop,clear-emergency} — all follow tractor_service.py's _reject/_ok pattern
export type TractorActuatorResponse =
  | { status: 'rejected'; reason: string }
  | ({ status: 'ok' } & Record<string, unknown>)

// POST /tractor/authorize, /tractor/revoke — ad-hoc literals, not from the service layer
export interface TractorAuthResult {
  status: 'authorized' | 'revoked'
}

// IMU calibration (maintenance)
export interface IMUCalibrationResult {
  status: string
  calibration_status?: string
  calibration_score: number
  steps: Record<string, unknown>[]
  timestamp: string
  started_at?: string
  notes?: string
}
export interface IMUCalibrationStatus {
  in_progress: boolean
  last_result?: IMUCalibrationResult
  supported?: boolean
}
