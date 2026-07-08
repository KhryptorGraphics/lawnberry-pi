// Ground truth: backend/src/api/routers/sensors.py get_rtk_diagnostics()
// (GET /api/v2/sensors/gps/rtk/diagnostics)

// backend/src/services/ntrip_client.py NtripForwarder.get_stats(). When no
// forwarder is running, sensors.py defaults this to just { enabled: false }.
export interface NtripStats {
  enabled: boolean
  connected?: boolean
  host?: string
  port?: number
  mountpoint?: string
  serial_device?: string
  baudrate?: number
  gga_configured?: boolean
  gga_interval_s?: number
  total_bytes_forwarded?: number
  bytes_forwarded_current_window?: number
  current_window_seconds?: number | null
  approx_rate_bps?: number | null
  uptime_s?: number | null
  last_forward_age_s?: number | null
}

// backend/src/models/sensor_data.py GpsReading, as returned by reading.model_dump().
export interface GpsReadingDump {
  latitude: number | null
  longitude: number | null
  altitude: number | null
  accuracy: number | null
  heading: number | null
  speed: number | null
  satellites: number | null
  hdop: number | null
  mode: string
  rtk_status: string | null
  timestamp: string
}

// sensors.py's gps_block dict — always initialized with this fixed key set.
export interface RtkGpsBlock {
  mode: string | null
  reading: GpsReadingDump | null
  last_hdop: number | null
  rtk_status: string | null
  satellites: number | null
  nmea: Record<string, string> | null
}

// sensors.py's hw_block — {} on any lookup failure, otherwise these two fields.
export interface RtkHardwareBlock {
  gps_type?: string
  gps_ntrip_enabled?: boolean
}

export interface RtkDiagnosticsResponse {
  ntrip: NtripStats
  gps: RtkGpsBlock
  hardware: RtkHardwareBlock
}
