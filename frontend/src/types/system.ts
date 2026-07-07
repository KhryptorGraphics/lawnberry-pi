export type SystemStatus = 'active' | 'warning' | 'error' | 'unknown' | 'maintenance'
export type ConnectionStatus = 'connected' | 'disconnected' | 'connecting' | 'error'

// GET /dashboard/status (response_model=MowerStatus, backend/src/api/routers/sensors.py)
export interface Position {
  latitude?: number | null
  longitude?: number | null
  altitude?: number | null
  accuracy?: number | null
  gps_mode?: string | null
}

export interface SafetyStatus {
  emergency_stop_active: boolean
  tilt_detected: boolean
  obstacle_detected: boolean
  blade_safety_ok: boolean
  safety_interlocks: string[]
}

export interface MowerStatus {
  position: Position | null
  battery_percentage: number
  power_mode: string
  navigation_state: string
  safety_status: SafetyStatus
  blade_active: boolean
  last_updated: string
}

export interface TelemetryData {
  timestamp: string
  sensors: {
    gps?: any
    imu?: any
    tof?: any
    environmental?: any
    power?: any
    health: boolean
  }
  motors: {
    left?: any
    right?: any
    blade?: any
    health: boolean
  }
  navigation: {
    position?: any
    target?: any
    status?: string
    health: boolean
  }
  system: {
    cpu_usage: number
    memory_usage: number
    temperature: number
    uptime: number
  }
}