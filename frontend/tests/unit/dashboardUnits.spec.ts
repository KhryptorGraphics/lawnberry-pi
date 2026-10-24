import { describe, it, beforeEach, expect, vi } from 'vitest'
import { ref, nextTick } from 'vue'
import { mount, flushPromises } from '@vue/test-utils'
import { setActivePinia, createPinia, storeToRefs } from 'pinia'

// Mock API modules first, before importing components
vi.mock('@/services/api', () => ({
  systemApi: {
    getStatus: vi.fn(),
  },
  controlApi: {
    emergencyStop: vi.fn().mockResolvedValue(undefined),
  },
  telemetryApi: {
    getCurrent: vi.fn(),
  },
  weatherApi: {
    getCurrent: vi.fn(),
  },
  settingsApi: {
    getSettings: vi.fn(),
  },
  maintenanceApi: {
    getImuCalibrationStatus: vi.fn(),
    runImuCalibration: vi.fn(),
  },
}))

vi.mock('@/services/websocket', () => ({
  useWebSocket: () => {
    const connected = ref(false)
    const connecting = ref(false)
    return {
      connected,
      connecting,
      connect: vi.fn(async () => {
        connected.value = true
      }),
      subscribe: vi.fn(),
      unsubscribe: vi.fn(),
      setCadence: vi.fn(),
    }
  },
}))

// Import after mocking
import DashboardView from '@/views/DashboardView.vue'
import { usePreferencesStore } from '@/stores/preferences'
import { systemApi, telemetryApi, weatherApi, settingsApi, maintenanceApi } from '@/services/api'
import { SIGN_IN_REQUIRED_MESSAGE } from '@/utils/apiErrors'

// Create telemetry factory function
const createTelemetryPayload = () => ({
  battery: { percentage: 55.5, voltage: 12.4 },
  position: {
    latitude: 37.123456,
    longitude: -122.123456,
    accuracy: 1.2,
    hdop: 0.8,
    satellites: 12,
    speed: 1.5,
    rtk_status: 'fix',
  },
  environmental: {
    temperature_c: 22,
    humidity_percent: 45,
    pressure_hpa: 1012,
    altitude_m: 10,
  },
  tof: {
    left: { distance_mm: 1200, range_status: 'valid', signal_strength: 200 },
    right: { distance_mm: 1300, range_status: 'valid', signal_strength: 210 },
  },
  imu: {
    calibration: 2,
    calibration_status: 'calibrated',
  },
  motor_status: 'idle',
  source: 'hardware',
})

const defaultSystemStatus = { status: 'Active', uptime: '1h 0m' }

describe('DashboardView unit preferences', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    window.localStorage.clear()
    vi.mocked(systemApi.getStatus).mockResolvedValue(defaultSystemStatus)
    vi.mocked(telemetryApi.getCurrent).mockResolvedValue(createTelemetryPayload())
    vi.mocked(weatherApi.getCurrent).mockResolvedValue({})
    vi.mocked(maintenanceApi.getImuCalibrationStatus).mockResolvedValue({ in_progress: false, last_result: null })
    vi.mocked(maintenanceApi.runImuCalibration).mockResolvedValue({
      calibration_score: 3,
      calibration_status: 'calibrated',
      status: 'completed',
      timestamp: new Date().toISOString(),
      steps: [],
    })
  })

  it('updates unit-dependent computed values when preference changes', async () => {
    vi.mocked(settingsApi.getSettings).mockResolvedValue({ system: { unit_system: 'metric' } })
    const pinia = createPinia()
    setActivePinia(pinia)

    const wrapper = mount(DashboardView, {
      global: {
        plugins: [pinia],
      },
    })

    await flushPromises()

    const preferences = usePreferencesStore()

    expect(wrapper.vm.speedUnit).toBe('m/s')
    expect(wrapper.vm.temperatureUnit).toBe('°C')
    expect(wrapper.vm.altitudeUnit).toBe('m')
    expect(wrapper.vm.pressureUnit).toBe('hPa')
    expect(wrapper.vm.tofUnit).toBe('cm')
    expect(wrapper.vm.speedDisplay).toBe('1.5')
    expect(wrapper.vm.temperatureDisplay).toBe('22.0')
    expect(wrapper.vm.altitudeDisplay).toBe('10.0')
    expect(wrapper.vm.pressureDisplay).toBe('1012.0')
    expect(wrapper.vm.tofLeftDisplay).toBe('120.0')

    preferences.setUnitSystem('imperial')
    await nextTick()
    await nextTick()

    expect(wrapper.vm.speedUnit).toBe('mph')
    expect(wrapper.vm.temperatureUnit).toBe('°F')
    expect(wrapper.vm.altitudeUnit).toBe('ft')
    expect(wrapper.vm.pressureUnit).toBe('inHg')
    expect(wrapper.vm.tofUnit).toBe('in')
    expect(wrapper.vm.speedDisplay).toBe('3.4')
    expect(wrapper.vm.temperatureDisplay).toBe('71.6')
    expect(wrapper.vm.altitudeDisplay).toBe('32.8')
    expect(wrapper.vm.pressureDisplay).toBe('29.88')
    expect(wrapper.vm.tofLeftDisplay).toBe('47.2')

    wrapper.unmount()
  })

  it('applies imperial preference from settings API on load', async () => {
    vi.mocked(settingsApi.getSettings).mockResolvedValue({ system: { unit_system: 'imperial' } })
    const pinia = createPinia()
    setActivePinia(pinia)

    const wrapper = mount(DashboardView, {
      global: {
        plugins: [pinia],
      },
    })

    await flushPromises()

  const preferences = usePreferencesStore()
  const { unitSystem } = storeToRefs(preferences)
  expect(unitSystem.value).toBe('imperial')
    expect(wrapper.vm.speedUnit).toBe('mph')
    expect(wrapper.vm.temperatureUnit).toBe('°F')
    expect(wrapper.vm.altitudeUnit).toBe('ft')
    expect(wrapper.vm.pressureUnit).toBe('inHg')
    expect(wrapper.vm.tofUnit).toBe('in')

    wrapper.unmount()
  })
})

// POST /maintenance/imu/calibrate needs the operator Bearer and answers 401 until someone has signed in.
describe('DashboardView IMU calibration failures', () => {
  async function mountDashboard() {
    vi.mocked(settingsApi.getSettings).mockResolvedValue({ system: { unit_system: 'metric' } })
    const pinia = createPinia()
    setActivePinia(pinia)
    const wrapper = mount(DashboardView, { global: { plugins: [pinia] } })
    await flushPromises()
    return wrapper
  }

  beforeEach(() => {
    vi.clearAllMocks()
    window.localStorage.clear()
    vi.mocked(systemApi.getStatus).mockResolvedValue(defaultSystemStatus)
    vi.mocked(telemetryApi.getCurrent).mockResolvedValue(createTelemetryPayload())
    vi.mocked(weatherApi.getCurrent).mockResolvedValue({})
    vi.mocked(maintenanceApi.getImuCalibrationStatus).mockResolvedValue({ in_progress: false, last_result: null })
  })

  it('a 401 shows the sign-in instruction in the calibration error, the card and the event log', async () => {
    vi.mocked(maintenanceApi.runImuCalibration).mockRejectedValue({
      response: { status: 401, data: { detail: 'Operator authentication required' } },
    })
    const wrapper = await mountDashboard()
    const statusReadsBefore = vi.mocked(maintenanceApi.getImuCalibrationStatus).mock.calls.length

    await wrapper.vm.runImuCalibration()
    await flushPromises()

    // The failed run is followed by a healthy status read, which used to wipe the error one tick after it was set.
    expect(vi.mocked(maintenanceApi.getImuCalibrationStatus).mock.calls.length).toBeGreaterThan(statusReadsBefore)
    expect(wrapper.vm.calibrationError).toBe(SIGN_IN_REQUIRED_MESSAGE)
    expect(wrapper.find('.calibration-error').text()).toContain(SIGN_IN_REQUIRED_MESSAGE)
    expect(wrapper.vm.recentEvents[0].message).toBe(`IMU calibration failed: ${SIGN_IN_REQUIRED_MESSAGE}`)
    expect(wrapper.vm.imuCalibrating).toBe(false)

    wrapper.unmount()
  })

  it('the next attempt clears the error', async () => {
    vi.mocked(maintenanceApi.runImuCalibration).mockRejectedValueOnce({ response: { status: 401 } })
    const wrapper = await mountDashboard()
    await wrapper.vm.runImuCalibration()
    expect(wrapper.vm.calibrationError).toBe(SIGN_IN_REQUIRED_MESSAGE)

    vi.mocked(maintenanceApi.runImuCalibration).mockResolvedValueOnce({
      calibration_score: 3,
      calibration_status: 'calibrated',
      status: 'completed',
      timestamp: new Date().toISOString(),
      steps: [],
    })
    await wrapper.vm.runImuCalibration()
    await flushPromises()

    expect(wrapper.vm.calibrationError).toBeNull()
    expect(wrapper.vm.recentEvents[0].message).toBe('IMU calibration completed successfully')

    wrapper.unmount()
  })

  it.each([
    ['a backend detail', { response: { status: 500, data: { detail: 'IMU not responding' } } }, 'IMU not responding'],
    ['a rate limit', { response: { status: 429, headers: { 'retry-after': '20' } } }, 'Too many attempts: try again in 20 s.'],
    ['no HTTP response', new Error('timeout of 30000ms exceeded'), 'Calibration failed: no response from the backend.'],
    ['a bare HTTP error', { response: { status: 500 } }, 'Calibration failed'],
  ])('%s reads as an operator-facing message', async (_name, error, expected) => {
    vi.mocked(maintenanceApi.runImuCalibration).mockRejectedValue(error)
    const wrapper = await mountDashboard()

    await wrapper.vm.runImuCalibration()
    await flushPromises()

    expect(wrapper.vm.calibrationError).toBe(expected)
    expect(wrapper.vm.recentEvents[0].message).toBe(`IMU calibration failed: ${expected}`)

    wrapper.unmount()
  })

  it('the "endpoint not available" error maintenanceApi raises for a 404 keeps its own message', async () => {
    // maintenanceApi.runImuCalibration() turns a 404 into this response-less Error; it must not read "no response"
    const unsupported = Object.assign(new Error('IMU calibration endpoint not available'), { unsupported: true })
    vi.mocked(maintenanceApi.runImuCalibration).mockRejectedValue(unsupported)
    const wrapper = await mountDashboard()

    await wrapper.vm.runImuCalibration()
    await flushPromises()

    expect(wrapper.vm.calibrationError).toBe('IMU calibration endpoint not available')

    wrapper.unmount()
  })

  it('a 404 response still marks the IMU unsupported and shows no error (unchanged)', async () => {
    vi.mocked(maintenanceApi.runImuCalibration).mockRejectedValue({ response: { status: 404 } })
    const wrapper = await mountDashboard()

    await wrapper.vm.runImuCalibration()
    await flushPromises()

    expect(wrapper.vm.imuSupported).toBe(false)
    expect(wrapper.vm.calibrationError).toBeNull()
    expect(wrapper.vm.imuCalibrating).toBe(false)

    wrapper.unmount()
  })

  it('a successful run shows no error (unchanged)', async () => {
    vi.mocked(maintenanceApi.runImuCalibration).mockResolvedValue({
      calibration_score: 3,
      calibration_status: 'calibrated',
      status: 'completed',
      timestamp: new Date().toISOString(),
      steps: [],
    })
    const wrapper = await mountDashboard()

    await wrapper.vm.runImuCalibration()
    await flushPromises()

    expect(wrapper.vm.calibrationError).toBeNull()
    expect(wrapper.find('.calibration-error').exists()).toBe(false)
    expect(wrapper.vm.recentEvents[0].message).toBe('IMU calibration completed successfully')

    wrapper.unmount()
  })
})
