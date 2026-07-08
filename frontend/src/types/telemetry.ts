/**
 * Telemetry types for LawnBerry Pi v2
 * Hardware telemetry data structures for frontend consumption
 */
import type { TractorState } from './control'

export interface TelemetryLatencyBadge {
  latency_ms: number
  status: 'healthy' | 'warning' | 'critical'
  target_ms: number
  device: 'pi5' | 'pi4'
}

export interface RTKStatus {
  fix_type: 'no_fix' | 'gps_fix' | 'dgps_fix' | 'rtk_float' | 'rtk_fixed'
  status_message: string
  satellites: number
  hdop: number
  requires_remediation: boolean
  remediation_link?: string
}

export interface IMUOrientation {
  roll_deg: number
  pitch_deg: number
  yaw_deg: number
  quaternion_w: number
  quaternion_x: number
  quaternion_y: number
  quaternion_z: number
  calibration_sys: number
  calibration_gyro: number
  calibration_accel: number
  calibration_mag: number
  health_status: 'healthy' | 'warning' | 'fault'
  remediation_link?: string
}

export interface PowerMetrics {
  battery: {
    voltage: number
    current: number
    power: number
    soc_percent: number | null
    health: 'healthy' | 'warning' | 'fault'
  }
  solar: {
    voltage: number
    current: number
    power: number
  }
  timestamp: string
}

export interface HardwareTelemetryStream {
  timestamp: string
  component_id: string
  value: any
  status: 'healthy' | 'warning' | 'fault'
  latency_ms: number
  gps_data?: {
    latitude: number
    longitude: number
    altitude_m: number
    speed_mps: number
    heading_deg: number
    hdop: number
    satellites: number
    fix_type: string
    rtk_status_message: string | null
  }
  imu_data?: {
    roll_deg: number
    pitch_deg: number
    yaw_deg: number
    quaternion_w: number
    quaternion_x: number
    quaternion_y: number
    quaternion_z: number
    accel_x: number
    accel_y: number
    accel_z: number
    gyro_x: number
    gyro_y: number
    gyro_z: number
    calibration_sys: number
    calibration_gyro: number
    calibration_accel: number
    calibration_mag: number
  }
  power_data?: {
    battery_voltage: number
    battery_current: number
    battery_power: number
    solar_voltage: number
    solar_current: number
    solar_power: number
    battery_soc_percent: number | null
    battery_health: 'healthy' | 'warning' | 'fault'
  }
  tof_data?: {
    distance_mm: number
    range_status: string
    signal_rate: number
  }
  metadata?: Record<string, any>
  verification_artifact_id?: string | null
}

export interface TelemetryStreamResponse {
  streams: HardwareTelemetryStream[]
  count: number
  latency_stats: {
    avg_latency_ms: number
    max_latency_ms: number
    min_latency_ms: number
    stream_count: number
  }
  rtk_status?: RTKStatus
  imu_orientation?: IMUOrientation
  timestamp: string
}

export interface TelemetryExportData {
  export_timestamp: string
  filters: {
    component_id: string | null
    start_time: string | null
    end_time: string | null
  }
  statistics: {
    avg_latency_ms: number
    max_latency_ms: number
    min_latency_ms: number
    stream_count: number
  }
  stream_count: number
  streams: HardwareTelemetryStream[]
}

export interface TelemetryPingResponse {
  component_id: string
  sample_count: number
  latency_ms: number
  avg_latency_ms: number
  min_latency_ms: number
  max_latency_ms: number
  p95_latency_ms: number
  meets_target: boolean
  target_ms: number
  device: 'pi5' | 'pi4'
  remediation?: {
    message: string
    doc_link: string
  }
}

export interface DashboardTelemetry {
  timestamp: string
  latency_badge: TelemetryLatencyBadge
  rtk_status: RTKStatus | null
  imu_orientation: IMUOrientation | null
  power_metrics: PowerMetrics
  gps: {
    latitude: number | null
    longitude: number | null
    altitude: number | null
    accuracy: number | null
    satellites: number
    fix_type: string
  }
  motors: {
    drive_left_pwm: number
    drive_right_pwm: number
    blade_pwm: number
    status: 'idle' | 'running' | 'error'
  }
  safety: {
    state: 'safe' | 'warning' | 'emergency'
    messages: string[]
  }
}

// ---------------------------------------------------------------------------
// WebSocket topic payloads (services/websocket.ts onTopic/subscribe), distinct
// from the REST telemetry-stream types above. Ported from the broadcast sites in
// backend/src/services/websocket_hub.py (_broadcast_telemetry_topics /
// _broadcast_additional_topics) and backend/src/safety/safety_monitor.py
// (handle_interlock_event).
// ---------------------------------------------------------------------------

export interface TelemetryPositionData {
  latitude: number | null
  longitude: number | null
  altitude: number | null
  accuracy: number | null
  gps_mode: string | null
  hdop: number | null
  speed: number | null
  rtk_status: string | null
  satellites: number | null
}

export interface TelemetryNavigationMessage {
  position: TelemetryPositionData
  source: string
  speed_mps?: number
  accuracy_m?: number
  hdop?: number
}

export interface TelemetryPowerBattery {
  voltage?: number
  current?: number
  power?: number
  percentage?: number
  charging_state?: 'charging' | 'discharging' | 'idle'
}

export interface TelemetryPowerSolar {
  voltage: number | null
  current: number | null
  power: number | null
  timestamp?: unknown
}

export interface TelemetryPowerLoad {
  state: 'on' | 'off' | null
  current: number | null
  power: number | null
}

export interface TelemetryPowerMessage {
  source: string
  // Raw power-sensor readings; shape depends on the active driver (hardware or simulated).
  power?: Record<string, unknown>
  battery?: TelemetryPowerBattery
  solar?: TelemetryPowerSolar
  load?: TelemetryPowerLoad
  timestamp?: unknown
}

// imu/environmental/motor_status have a fixed shape in _generate_telemetry
// (backend/src/services/websocket_hub.py, both the hardware and simulated branches
// always populate exactly these keys, defaulting to null when a reading is unavailable).
export interface TelemetryImuData {
  roll: number | null
  pitch: number | null
  yaw: number | null
  gyro_z: number | null
  calibration: number | null
  calibration_status: string | null
}

export interface TelemetrySensorsMessage {
  imu: TelemetryImuData
  source: string
}

export interface TelemetryEnvironmentalData {
  temperature_c: number | null
  humidity_percent: number | null
  pressure_hpa: number | null
  altitude_m: number | null
}

export interface TelemetryEnvironmentalMessage {
  environmental: TelemetryEnvironmentalData
  source: string
}

// tof's shape has drifted across sensor-driver versions (see the multi-candidate field
// lookups in DashboardView's coerceTofReading), so it's left untyped rather than
// asserting a schema that isn't actually guaranteed.
export interface TelemetryTofMessage {
  tof: unknown
  source: string
}

export interface TelemetryMotorsMessage {
  motor_status: string
  source: string
}

export interface TelemetrySystemMessage {
  safety_state: string
  uptime_seconds: number
  source: string
}

export interface TelemetryWeatherMessage {
  temperature_c: number | null
  humidity_percent: number | null
  pressure_hpa: number | null
  altitude_m: number | null
  wind_speed_ms: number | null
  precipitation_mm: number | null
  source: string
}

// jobs.progress is currently a stub on the backend (random values, hardcoded job name),
// not real mission telemetry. Typed as it's actually sent today.
export interface JobsProgressMessage {
  current_job: string
  progress_percent: number
  remaining_time_min: number
  status: 'running' | 'paused' | 'idle'
  source: string
}

export interface SystemSafetyInterlock {
  interlock_id: string
  interlock_type: string
  triggered_at_us: number
  cleared_at_us: number | null
  acknowledged_at_us: number | null
  state: string
  trigger_value: number | null
  description: string
}

export interface SystemSafetyMessage {
  action: 'activate' | 'clear'
  interlock: SystemSafetyInterlock
  timestamp: string
}

// telemetry.tractor — see websocket_hub.py _broadcast_additional_topics(). Unlike
// GET /tractor/state (a bare TractorState), the WS topic wraps it: {tractor, source}.
// `tractor` is null when server-side derivation fails (source: 'unavailable').
export interface TelemetryTractorMessage {
  tractor: TractorState | null
  source: string
}

export interface TelemetryTopicPayloadMap {
  'telemetry.power': TelemetryPowerMessage
  'telemetry.navigation': TelemetryNavigationMessage
  'telemetry.sensors': TelemetrySensorsMessage
  'telemetry.environmental': TelemetryEnvironmentalMessage
  'telemetry.tof': TelemetryTofMessage
  'telemetry.motors': TelemetryMotorsMessage
  'telemetry.system': TelemetrySystemMessage
  'telemetry.weather': TelemetryWeatherMessage
  'jobs.progress': JobsProgressMessage
  'system.safety': SystemSafetyMessage
  'telemetry.tractor': TelemetryTractorMessage
}
