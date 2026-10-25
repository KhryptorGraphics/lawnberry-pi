// Control API methods
export async function sendControlCommand(
  command: 'drive' | 'emergency',
  payload?: Record<string, unknown>
): Promise<ControlResponseV2>
export async function sendControlCommand(
  command: 'blade',
  payload?: Record<string, unknown>
): Promise<BladeCommandResponse>
export async function sendControlCommand(
  command: string,
  payload?: Record<string, unknown>
): Promise<ControlResponseV2 | BladeCommandResponse>
export async function sendControlCommand(command: string, payload: Record<string, unknown> = {}) {
  // Map command to endpoint
  let url = ''
  switch (command) {
    case 'drive':
      url = '/api/v2/control/drive'
      break
    case 'blade':
      url = '/api/v2/control/blade'
      break
    case 'emergency':
      url = '/api/v2/control/emergency'
      break
    default:
      throw new Error(`Unknown control command: ${command}`)
  }
  const response = await apiService.post(url, payload)
  return response.data
}

export async function getRoboHATStatus(): Promise<RoboHATStatus> {
  const response = await apiService.get<RoboHATStatus>('/api/v2/hardware/robohat')
  return response.data
}

// Map API methods
export async function getMapConfiguration(configId: string = 'default') {
  const response = await apiService.get(`/api/v2/map/configuration?config_id=${configId}`)
  return response.data
}

export async function saveMapConfiguration(configId: string, config: any) {
  const response = await apiService.put(`/api/v2/map/configuration?config_id=${configId}`, config)
  return response.data
}

// Autonomous navigation control
export async function startAutonomous(zones?: string[]): Promise<AutonomyStartResult> {
  const response = await apiService.post<AutonomyStartResult>(
    '/api/v2/navigation/start',
    zones ? { zones } : {}
  )
  return response.data
}

export async function stopAutonomous(): Promise<AutonomyStopResult> {
  const response = await apiService.post<AutonomyStopResult>('/api/v2/navigation/stop', {})
  return response.data
}

export async function pauseAutonomous(): Promise<AutonomyPauseResult> {
  const response = await apiService.post<AutonomyPauseResult>('/api/v2/navigation/pause', {})
  return response.data
}

export async function resumeAutonomous(): Promise<AutonomyResumeResult> {
  const response = await apiService.post<AutonomyResumeResult>('/api/v2/navigation/resume', {})
  return response.data
}

export async function returnToBase(): Promise<AutonomyReturnResult> {
  const response = await apiService.post<AutonomyReturnResult>('/api/v2/navigation/return', {})
  return response.data
}

export async function getNavigationStatus(): Promise<AutonomyStatusResult> {
  const response = await apiService.get<AutonomyStatusResult>('/api/v2/navigation/status')
  return response.data
}

export async function setControlMode(
  mode: 'manual' | 'autonomous' | 'idle',
  zones?: string[]
): Promise<ControlModeResult> {
  const payload: Record<string, unknown> = { mode }
  if (zones) payload.zones = zones
  const response = await apiService.post<ControlModeResult>('/api/v2/control/mode', payload)
  return response.data
}

// Planning job lifecycle
export async function planningJobAction(
  jobId: string,
  action: 'start' | 'pause' | 'resume' | 'cancel'
) {
  const response = await apiService.post(`/api/v2/planning/jobs/${jobId}/${action}`, {})
  return response.data
}

// --- 50" zero-turn mower: twin-lever hydrostatic drive actuation ---

export async function getTractorState(): Promise<TractorState> {
  return (await apiService.get<TractorState>('/api/v2/tractor/state')).data
}
// A lever/throttle command is only held for the backend's command-hold window
// (500 ms by default: backend tractor_service.py, `command_deadman`), so a caller
// that keeps re-sending one (composables/useTractorHold.ts) passes a per-request
// `timeout`: a stalled link must be noticed inside that window, not after the
// client-wide 10 s default.
export interface TractorCommandOptions {
  /** Per-request timeout in ms (overrides the client-wide default). */
  timeout?: number
}
function commandConfig(options?: TractorCommandOptions): AxiosRequestConfig | undefined {
  return options?.timeout ? { timeout: options.timeout } : undefined
}
export async function tractorLeftLever(
  value: number,
  options?: TractorCommandOptions
): Promise<TractorActuatorResponse> {
  return (
    await apiService.post<TractorActuatorResponse>('/api/v2/tractor/left-lever', { value }, commandConfig(options))
  ).data
}
export async function tractorRightLever(
  value: number,
  options?: TractorCommandOptions
): Promise<TractorActuatorResponse> {
  return (
    await apiService.post<TractorActuatorResponse>('/api/v2/tractor/right-lever', { value }, commandConfig(options))
  ).data
}
export async function tractorThrottle(
  value: number,
  options?: TractorCommandOptions
): Promise<TractorActuatorResponse> {
  return (
    await apiService.post<TractorActuatorResponse>('/api/v2/tractor/throttle', { value }, commandConfig(options))
  ).data
}
export async function tractorBlade(engaged: boolean): Promise<TractorActuatorResponse> {
  return (await apiService.post<TractorActuatorResponse>('/api/v2/tractor/blade', { engaged })).data
}
export async function tractorStart(): Promise<TractorActuatorResponse> {
  return (await apiService.post<TractorActuatorResponse>('/api/v2/tractor/starter', {})).data
}
export async function tractorStopEngine(): Promise<TractorActuatorResponse> {
  return (await apiService.post<TractorActuatorResponse>('/api/v2/tractor/stop-engine', {})).data
}
export async function tractorEmergencyStop(): Promise<TractorSafetyResponse> {
  return (await apiService.post<TractorSafetyResponse>('/api/v2/tractor/emergency-stop', {})).data
}
export async function tractorClearEmergency(): Promise<TractorSafetyResponse> {
  return (await apiService.post<TractorSafetyResponse>('/api/v2/tractor/clear-emergency', {})).data
}
export async function tractorAuthorize(authorized: boolean): Promise<TractorAuthResult> {
  const path = authorized ? '/api/v2/tractor/authorize' : '/api/v2/tractor/revoke'
  return (await apiService.post<TractorAuthResult>(path, {})).data
}
import axios from 'axios'
import type { AxiosInstance, AxiosRequestConfig, AxiosResponse } from 'axios'
import { useAuthStore } from '@/stores/auth'
import type { AuthResponse, RefreshResponse, LoginCredentials, User } from '@/types/auth'
import type { MowerStatus } from '@/types/system'
import type {
  ControlResponseV2,
  BladeCommandResponse,
  EmergencyStopResult,
  RoboHATStatus,
  AutonomyStartResult,
  AutonomyStopResult,
  AutonomyPauseResult,
  AutonomyResumeResult,
  AutonomyReturnResult,
  AutonomyStatusResult,
  ControlModeResult,
  TractorState,
  TractorActuatorResponse,
  TractorAuthResult,
  TractorSafetyResponse,
  IMUCalibrationResult,
  IMUCalibrationStatus,
} from '@/types/control'

const CLIENT_ID_STORAGE_KEY = 'lawnberry-client-id'
const CLIENT_ID_GLOBAL_KEY = '__LAWN_CLIENT_ID__'

function generateClientId(): string {
  const randomId = `web-${Math.random().toString(36).slice(2)}-${Date.now().toString(36)}`

  if (typeof window !== 'undefined') {
    try {
      const existing = window.localStorage?.getItem(CLIENT_ID_STORAGE_KEY)
      if (existing) {
        return existing
      }
      window.localStorage?.setItem(CLIENT_ID_STORAGE_KEY, randomId)
      return randomId
    } catch (error) {
      /* localStorage unavailable, fall through */
    }
  }

  const globalScope = globalThis as Record<string, unknown>
  const existingGlobal = globalScope[CLIENT_ID_GLOBAL_KEY]
  if (typeof existingGlobal === 'string' && existingGlobal.length > 0) {
    return existingGlobal
  }
  globalScope[CLIENT_ID_GLOBAL_KEY] = randomId
  return randomId
}

// A 401 from these calls is final: refreshing the token and replaying the request would be wrong or pointless.
//  - /auth/refresh is the refresh itself (replaying it would loop).
//  - /auth/login and /control/manual-unlock carry a credential the operator just typed. Their 401 means "wrong or
//    missing credential", never "expired token": replaying would send the same wrong password a second time and
//    count it twice against the backend's per-peer lockout (429 after repeated wrong passwords).
//  - /auth/logout: a 401 means the Bearer is already dead, so there is nothing to refresh. Refreshing anyway made
//    authStore.refreshToken() -> logout() -> POST /auth/logout (401) -> refreshToken() -> logout() ... recurse
//    without end: a request storm that never settled the call that started it.
// Matching is by path suffix, so `/control/manual-unlock/status` is deliberately not covered.
const NO_REFRESH_ON_401_PATHS = ['/auth/refresh', '/auth/login', '/auth/logout', '/control/manual-unlock']

function skipsTokenRefresh(url: unknown): boolean {
  if (typeof url !== 'string') return false
  const path = url.split('?')[0].replace(/\/+$/, '')
  return NO_REFRESH_ON_401_PATHS.some((suffix) => path.endsWith(suffix))
}

class ApiService {
  private client: AxiosInstance

  constructor() {
    // Compute default base URL from current origin to use frontend proxy (/api and /api/v2)
    const defaultBase = ''
    const clientId = generateClientId()
    this.client = axios.create({
      baseURL: defaultBase,
      timeout: 10000,
      headers: {
        'Content-Type': 'application/json',
        'X-Client-Id': clientId
      }
    })

    // Request interceptor to add auth token
    this.client.interceptors.request.use(
      (config) => {
        const authStore = useAuthStore()
        if (authStore.token) {
          config.headers.Authorization = `Bearer ${authStore.token}`
        }
        return config
      },
      (error) => {
        return Promise.reject(error)
      }
    )

    // Response interceptor: on 401, refresh once and retry (except for the calls listed above, whose 401 is final)
    this.client.interceptors.response.use(
      (response) => response,
      async (error) => {
        const originalRequest = error.config
        if (
          error.response?.status === 401 &&
          !skipsTokenRefresh(originalRequest?.url) &&
          !originalRequest?._retried
        ) {
          const authStore = useAuthStore()
          const refreshed = await authStore.refreshToken()
          if (refreshed && authStore.token) {
            originalRequest._retried = true
            originalRequest.headers = originalRequest.headers || {}
            originalRequest.headers.Authorization = `Bearer ${authStore.token}`
            return this.client(originalRequest)
          }
        }
        return Promise.reject(error)
      }
    )
  }

  async get<T = any>(url: string, config?: AxiosRequestConfig): Promise<AxiosResponse<T>> {
    return this.client.get<T>(url, config)
  }

  async post<T = any>(url: string, data?: any, config?: AxiosRequestConfig): Promise<AxiosResponse<T>> {
    return this.client.post<T>(url, data, config)
  }

  async put<T = any>(url: string, data?: any): Promise<AxiosResponse<T>> {
    return this.client.put<T>(url, data)
  }

  async delete<T = any>(url: string): Promise<AxiosResponse<T>> {
    return this.client.delete<T>(url)
  }

  async patch<T = any>(url: string, data?: any): Promise<AxiosResponse<T>> {
    return this.client.patch<T>(url, data)
  }

  // Telemetry API methods
  async getTelemetryStream(params?: {
    page?: number
    per_page?: number
    component_id?: string
    start_time?: string
    end_time?: string
  }) {
    const query = new URLSearchParams()
    if (params) {
      Object.entries(params).forEach(([key, value]) => {
        if (value !== undefined && value !== null) {
          query.append(key, String(value))
        }
      })
    }
    const queryString = query.toString()
    return this.get(`/api/v2/telemetry/stream${queryString ? `?${queryString}` : ''}`)
  }

  async exportTelemetryDiagnostic(params?: {
    component_id?: string
    start_time?: string
    end_time?: string
  }) {
    const query = new URLSearchParams()
    if (params) {
      Object.entries(params).forEach(([key, value]) => {
        if (value !== undefined && value !== null) {
          query.append(key, String(value))
        }
      })
    }
    const queryString = query.toString()
    return this.client.get(`/api/v2/telemetry/export${queryString ? `?${queryString}` : ''}`, {
      responseType: 'blob'
    })
  }

  async pingTelemetry(data: {
    component_id: string
    sample_count?: number
  }) {
    return this.post('/api/v2/telemetry/ping', data)
  }
}

// Singleton instance
const apiService = new ApiService()

// Auth
export const authApi = {
  login: async (credentials: LoginCredentials): Promise<AuthResponse> =>
    (await apiService.post<AuthResponse>('/api/v2/auth/login', credentials)).data,
  // These two POSTs carry no data, and axios then sends no Content-Type, so a backend that insists
  // on JSON (the auth middleware did, answering 415) never saw the logout: the session and its
  // WebSocket tickets stayed alive after "sign out" (SEC20-06). `{}` is a real JSON body.
  logout: async (): Promise<void> => {
    await apiService.post('/api/v2/auth/logout', {})
  },
  refresh: async (): Promise<RefreshResponse> =>
    (await apiService.post<RefreshResponse>('/api/v2/auth/refresh', {})).data,
  getProfile: async (): Promise<User> => (await apiService.get<User>('/api/v2/auth/profile')).data,
}

// Dashboard system status
export const systemApi = {
  getStatus: async (): Promise<MowerStatus> =>
    (await apiService.get<MowerStatus>('/api/v2/dashboard/status')).data,
}

// General mower emergency stop (distinct from /tractor/emergency-stop, the ride-on platform's own E-stop)
export const controlApi = {
  emergencyStop: async (): Promise<EmergencyStopResult> =>
    (await apiService.post<EmergencyStopResult>('/api/v2/control/emergency-stop')).data,
}

export const settingsApi = {
  getSettings: async (): Promise<Record<string, unknown>> =>
    (await apiService.get<Record<string, unknown>>('/api/v2/settings')).data,
}

export const telemetryApi = {
  getCurrent: async () =>
    // Hardware response can be slow; override the default 10s timeout
    (await apiService.get('/api/v2/dashboard/telemetry', { timeout: 30000 })).data,
}

export const weatherApi = {
  getCurrent: async (params?: { lat?: number; lon?: number }) => {
    const query = new URLSearchParams()
    if (params?.lat !== undefined) query.append('lat', String(params.lat))
    if (params?.lon !== undefined) query.append('lon', String(params.lon))
    const path = query.toString() ? `/api/v2/weather/current?${query}` : '/api/v2/weather/current'
    const response = await apiService.get(path)
    // Backend contract (GET /api/v2/weather/current): { timestamp, source, temperature_c, humidity_percent, pressure_hpa }
    return response.data as {
      timestamp: string
      source: string
      temperature_c: number
      humidity_percent: number
      pressure_hpa: number
    }
  },
}

export const maintenanceApi = {
  runImuCalibration: async (): Promise<IMUCalibrationResult> => {
    try {
      // Calibration routine takes ~18-20s on hardware; override default 10s timeout
      return (
        await apiService.post<IMUCalibrationResult>(
          '/api/v2/maintenance/imu/calibrate',
          {},
          { timeout: 30000 }
        )
      ).data
    } catch (error: any) {
      if (error?.response?.status === 404) {
        const unsupported = new Error('IMU calibration endpoint not available')
        ;(unsupported as any).unsupported = true
        throw unsupported
      }
      throw error
    }
  },
  getImuCalibrationStatus: async (): Promise<IMUCalibrationStatus> => {
    try {
      return (await apiService.get<IMUCalibrationStatus>('/api/v2/maintenance/imu/calibrate')).data
    } catch (error: any) {
      if (error?.response?.status === 404) {
        return { in_progress: false, last_result: undefined, supported: false }
      }
      throw error
    }
  },
}

// Composable for use in Vue components
export function useApiService() {
  return apiService
}

export default apiService