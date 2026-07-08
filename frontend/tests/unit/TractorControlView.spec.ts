// Model: ControlView.unlock.spec.ts. Exercises TractorControlView.vue wired to
// the real (Pinia-backed) tractor store and toast store, with only the network
// boundary (services/api, services/websocket) mocked — see vitest.setup.ts.
import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest'
import { mount, flushPromises } from '@vue/test-utils'
import { setActivePinia, createPinia } from 'pinia'
import TractorControlView from '@/views/TractorControlView.vue'
import { useToastStore } from '@/stores/toast'
import apiClient from '@/services/api'
import * as api from '@/services/api'

function getTractorWsEntry() {
  const instances = (globalThis as any).__wsMockInstances as Array<{
    type: string
    instance: any
    topicCallbacks: any
  }>
  for (let i = instances.length - 1; i >= 0; i -= 1) {
    if (instances[i].type === 'telemetry') return instances[i]
  }
  throw new Error('Tractor WebSocket mock was not initialized')
}

function baseState(overrides: Record<string, any> = {}) {
  return {
    steering: 0,
    throttle: 0,
    ground_speed: 0,
    gear: 'neutral',
    clutch: 1,
    blade_engaged: false,
    engine: 'off',
    enabled: true,
    emergency_stop_active: false,
    authorized: false,
    interlock_reason: null,
    last_updated: new Date(2026, 0, 1, 12, 0, 0).toISOString(),
    engine_running: false,
    moving: false,
    ...overrides,
  }
}

async function mountView(initialState = baseState()) {
  vi.mocked(api.getTractorState).mockResolvedValue(initialState)
  apiClient.get.mockImplementation((url: string) => {
    if (url.startsWith('/api/v2/camera')) {
      return Promise.resolve({ data: {} })
    }
    return Promise.resolve({ data: {} })
  })
  apiClient.post.mockResolvedValue({ data: {} })

  const wrapper = mount(TractorControlView)
  await flushPromises()
  return wrapper
}

describe('TractorControlView', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
    vi.clearAllMocks()
  })

  afterEach(() => {
    const instances = (globalThis as any).__wsMockInstances as Array<any>
    if (instances) instances.length = 0
  })

  it('fetches state once on mount and syncs the drivetrain sliders', async () => {
    const wrapper = await mountView(baseState({ steering: 0.4, throttle: 0.6, ground_speed: 0.2, clutch: 0.1 }))

    expect(api.getTractorState).toHaveBeenCalledTimes(1)
    const steeringInput = wrapper.findAll('input[type="range"]')[0]
    expect(Number((steeringInput.element as HTMLInputElement).value)).toBeCloseTo(0.4)
  })

  it('toggles authorize/revoke based on current state', async () => {
    const wrapper = await mountView(baseState({ authorized: false }))
    vi.mocked(api.tractorAuthorize).mockResolvedValue({ status: 'authorized' } as any)

    const authorizeButton = wrapper.findAll('button').find((b) => b.text() === 'Authorize')!
    await authorizeButton.trigger('click')
    await flushPromises()

    expect(api.tractorAuthorize).toHaveBeenCalledWith(true)
  })

  it('emergency stop instantly reflects emergency_stop_active through the real store', async () => {
    const wrapper = await mountView(baseState({ authorized: true }))
    vi.mocked(api.tractorEmergencyStop).mockResolvedValue({ status: 'emergency_stop', engine: 'running' } as any)
    vi.mocked(api.getTractorState).mockResolvedValue(
      baseState({ emergency_stop_active: true, authorized: false })
    )

    const estopButton = wrapper.findAll('button').find((b) => b.text().includes('EMERGENCY STOP'))!
    await estopButton.trigger('click')
    await flushPromises()

    expect(wrapper.text()).toContain('E-STOP')
  })

  it('surfaces a toast (does not throw) when an actuator command is rejected', async () => {
    const wrapper = await mountView(baseState({ gear: 'neutral' }))
    vi.mocked(api.tractorGear).mockResolvedValue({
      status: 'rejected',
      reason: 'emergency stop active',
    } as any)

    const forwardButton = wrapper.findAll('.gear-buttons button').find((b) => b.text() === 'forward')!
    await forwardButton.trigger('click')
    await flushPromises()

    const toastStore = useToastStore()
    expect(toastStore.toasts.some((t) => t.message.includes('emergency stop active'))).toBe(true)
  })

  it('updates the GPS display from a telemetry.navigation WS tick', async () => {
    const wrapper = await mountView()
    const wsEntry = getTractorWsEntry()
    const callbacks = wsEntry.topicCallbacks.get('telemetry.navigation') as
      | Array<{ callback: (data: any) => void }>
      | undefined
    expect(callbacks?.length ?? 0).toBeGreaterThan(0)

    callbacks!.forEach((entry) =>
      entry.callback({ position: { latitude: 40.123456, longitude: -74.654321, satellites: 9 } })
    )
    await flushPromises()

    expect(wrapper.text()).toContain('40.123456')
  })
})
