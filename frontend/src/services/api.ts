// Control API methods
export async function sendControlCommand(command: string, payload: any = {}) {
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

export async function getRoboHATStatus() {
  const response = await apiService.get('/api/v2/hardware/robohat')
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
export async function startAutonomous(zones?: string[]) {
  const response = await apiService.post('/api/v2/navigation/start', zones ? { zones } : {})
  return response.data
}

export async function stopAutonomous() {
  const response = await apiService.post('/api/v2/navigation/stop', {})
  return response.data
}

export async function pauseAutonomous() {
  const response = await apiService.post('/api/v2/navigation/pause', {})
  return response.data
}

export async function resumeAutonomous() {
  const response = await apiService.post('/api/v2/navigation/resume', {})
  return response.data
}

export async function returnToBase() {
  const response = await apiService.post('/api/v2/navigation/return', {})
  return response.data
}

export async function getNavigationStatus() {
  const response = await apiService.get('/api/v2/navigation/status')
  return response.data
}

export async function setControlMode(
  mode: 'manual' | 'autonomous' | 'idle',
  zones?: string[]
) {
  const payload: Record<string, unknown> = { mode }
  if (zones) payload.zones = zones
  const response = await apiService.post('/api/v2/control/mode', payload)
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

// --- Ride-on lawn tractor actuation ---
export type TractorGear = 'forward' | 'neutral' | 'reverse'

export async function getTractorState() {
  return (await apiService.get('/api/v2/tractor/state')).data
}
export async function tractorSteering(value: number) {
  return (await apiService.post('/api/v2/tractor/steering', { value })).data
}
export async function tractorThrottle(value: number) {
  return (await apiService.post('/api/v2/tractor/throttle', { value })).data
}
export async function tractorSpeed(value: number) {
  return (await apiService.post('/api/v2/tractor/speed', { value })).data
}
export async function tractorClutch(value: number) {
  return (await apiService.post('/api/v2/tractor/clutch', { value })).data
}
export async function tractorGear(gear: TractorGear) {
  return (await apiService.post('/api/v2/tractor/gear', { gear })).data
}
export async function tractorBlade(engaged: boolean) {
  return (await apiService.post('/api/v2/tractor/blade', { engaged })).data
}
export async function tractorStart() {
  return (await apiService.post('/api/v2/tractor/starter', {})).data
}
export async function tractorStopEngine() {
  return (await apiService.post('/api/v2/tractor/stop-engine', {})).data
}
export async function tractorEmergencyStop() {
  return (await apiService.post('/api/v2/tractor/emergency-stop', {})).data
}
export async function tractorClearEmergency() {
  return (await apiService.post('/api/v2/tractor/clear-emergency', {})).data
}
export async function tractorAuthorize(authorized: boolean) {
  const path = authorized ? '/api/v2/tractor/authorize' : '/api/v2/tractor/revoke'
  return (await apiService.post(path, {})).data
}
import axios from 'axios'
import type { AxiosInstance, AxiosRequestConfig, AxiosResponse } from 'axios'
import { useAuthStore } from '@/stores/auth'
import type { AuthResponse, RefreshResponse, LoginCredentials, User } from '@/types/auth'

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

    // Response interceptor: on 401, refresh once and retry (skip for the refresh call itself to avoid looping)
    this.client.interceptors.response.use(
      (response) => response,
      async (error) => {
        const originalRequest = error.config
        const isRefreshCall = typeof originalRequest?.url === 'string' && originalRequest.url.includes('/auth/refresh')
        if (error.response?.status === 401 && !isRefreshCall && !originalRequest?._retried) {
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
  logout: async (): Promise<void> => {
    await apiService.post('/api/v2/auth/logout')
  },
  refresh: async (): Promise<RefreshResponse> =>
    (await apiService.post<RefreshResponse>('/api/v2/auth/refresh')).data,
  getProfile: async (): Promise<User> => (await apiService.get<User>('/api/v2/auth/profile')).data,
}

// Dashboard system status
export const systemApi = {
  getStatus: async () => (await apiService.get('/api/v2/dashboard/status')).data,
}

// General mower emergency stop (distinct from /tractor/emergency-stop, the ride-on platform's own E-stop)
export const controlApi = {
  emergencyStop: async () => (await apiService.post('/api/v2/control/emergency-stop')).data,
}

export const settingsApi = {
  getSettings: async () => (await apiService.get('/api/v2/settings')).data,
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
  runImuCalibration: async () => {
    try {
      // Calibration routine takes ~18-20s on hardware; override default 10s timeout
      return (await apiService.post('/api/v2/maintenance/imu/calibrate', {}, { timeout: 30000 })).data
    } catch (error: any) {
      if (error?.response?.status === 404) {
        const unsupported = new Error('IMU calibration endpoint not available')
        ;(unsupported as any).unsupported = true
        throw unsupported
      }
      throw error
    }
  },
  getImuCalibrationStatus: async () => {
    try {
      return (await apiService.get('/api/v2/maintenance/imu/calibrate')).data
    } catch (error: any) {
      if (error?.response?.status === 404) {
        return { in_progress: false, last_result: null, supported: false }
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