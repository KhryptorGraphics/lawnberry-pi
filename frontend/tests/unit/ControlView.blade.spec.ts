/* eslint-disable @typescript-eslint/ban-ts-comment */
// @ts-nocheck
// HW-03 / BL-01: the backend blade ON is a deadman, so Start Mowing must keep repeating the ON (after reading
// the backend status) until the operator stops, and must never restart a blade the backend dropped.
import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest'
import { mount, flushPromises, RouterLinkStub } from '@vue/test-utils'
import { setActivePinia, createPinia } from 'pinia'
import { ref, defineComponent, h } from 'vue'
import ControlView from '@/views/ControlView.vue'
import apiClient, { getBladeStatus } from '@/services/api'

const controlStoreContainer: { current: object | null } = { current: null }

vi.mock('@/stores/control', () => ({
  useControlStore: () => controlStoreContainer.current,
}))

const toastStore = { show: vi.fn() }
vi.mock('@/stores/toast', () => ({
  useToastStore: () => toastStore,
}))

vi.mock('@/components/ui/VirtualJoystick.vue', () => {
  const stub = defineComponent({
    name: 'VirtualJoystickStub',
    emits: ['change', 'end'],
    methods: {
      reset() {
        this.$emit('change', { x: 0, y: 0, magnitude: 0, active: false })
        this.$emit('end')
      },
    },
    render() {
      return h('div', { class: 'virtual-joystick-stub' })
    },
  })
  return { default: stub }
})

vi.mock('@/stores/preferences', () => {
  const unitSystem = ref<'metric' | 'imperial'>('metric')
  return {
    usePreferencesStore: () => ({
      unitSystem,
      ensureInitialized: vi.fn(),
      setUnitSystem: vi.fn(),
      syncWithServer: vi.fn(),
    }),
  }
})

const ACCEPTED = { accepted: true, result: 'accepted' }
const statusOn = (extra = {}) => ({ commanded: 'on', deadman_ms: 2000, last_drop: null, ...extra })

function createMockControlStore() {
  return {
    lockout: false,
    lockoutReason: '',
    lockoutUntil: null,
    lastEcho: null,
    lastCommandEcho: null,
    lastCommandResult: null,
    remediationLink: '',
    isLoading: false,
    commandInProgress: false,
    robohatStatus: null,
    submitCommand: vi.fn().mockResolvedValue(ACCEPTED),
    fetchRoboHATStatus: vi.fn().mockResolvedValue({ telemetry_source: 'simulated' }),
    initWebSocket: vi.fn(),
    cleanup: vi.fn(),
  }
}

async function mountUnlocked() {
  const wrapper = mount(ControlView, { global: { stubs: { RouterLink: RouterLinkStub } } })
  await flushPromises()
  wrapper.vm.stopCameraFeed()
  const vm = wrapper.vm
  vm.authForm.password = 'secret'
  apiClient.post.mockResolvedValueOnce({ data: { session_id: 's1' } })
  await vm.authenticateControl()
  await flushPromises()
  expect(vm.isControlUnlocked).toBe(true)
  vm.stopCameraFeed()
  await flushPromises()
  return { wrapper, vm }
}

const store = () => controlStoreContainer.current
const bladeCalls = (action?: string, reason?: string) =>
  store().submitCommand.mock.calls.filter(
    ([cmd, body]) => cmd === 'blade' && (!action || body.action === action) && (!reason || body.reason === reason)
  )
const repeats = () => bladeCalls('enable', 'manual-control-repeat')
const tick = (ms: number) => vi.advanceTimersByTimeAsync(ms)

// Mount with real timers (flushPromises needs them), then switch to fake timers and press Start Mowing.
async function startMowing(status = statusOn()) {
  const mounted = await mountUnlocked()
  getBladeStatus.mockResolvedValue(status)
  vi.useFakeTimers()
  await mounted.vm.toggleMowing()
  await tick(0)
  expect(mounted.vm.mowingActive).toBe(true)
  return mounted
}

beforeEach(() => {
  setActivePinia(createPinia())
  controlStoreContainer.current = createMockControlStore()
  toastStore.show.mockReset()
  apiClient.get.mockReset()
  apiClient.post.mockReset()
  apiClient.get.mockImplementation((url: string) => {
    if (url === '/api/v2/settings/security') {
      return Promise.resolve({ data: { security_level: 'password', session_timeout_minutes: 15 } })
    }
    return Promise.resolve({ data: {} })
  })
})

afterEach(() => {
  vi.useRealTimers()
  controlStoreContainer.current = null
})

describe('ControlView blade keep-alive', () => {
  it('repeats the ON every interval with the same session, each preceded by a status read', async () => {
    const { wrapper } = await startMowing()
    expect(bladeCalls('enable', 'manual-control')).toHaveLength(1)
    getBladeStatus.mockClear()

    for (let i = 1; i <= 4; i++) {
      await tick(667)
      expect(repeats()).toHaveLength(i)
      expect(getBladeStatus).toHaveBeenCalledTimes(i)
    }
    for (const [, body] of repeats()) {
      expect(body).toEqual({ session_id: 's1', action: 'enable', reason: 'manual-control-repeat' })
    }
    // status is read before each ON
    const order = [...getBladeStatus.mock.invocationCallOrder]
    const sends = store().submitCommand.mock.invocationCallOrder.slice(1)
    order.forEach((n, i) => expect(n).toBeLessThan(sends[i]))
    wrapper.unmount()
  })

  describe('interval follows deadman_ms', () => {
    it.each([
      [900, 300],
      [6000, 2000],
      [undefined, 667],
      [Number.NaN, 667],
      [0, 667],
    ])('deadman_ms=%s repeats every %s ms', async (deadman, expected) => {
      const { wrapper } = await startMowing(statusOn({ deadman_ms: deadman }))
      await tick(expected - 1)
      expect(repeats()).toHaveLength(0)
      await tick(1)
      expect(repeats()).toHaveLength(1)
      await tick(expected)
      expect(repeats()).toHaveLength(2)
      wrapper.unmount()
    })

    it('uses the default window when the status read fails', async () => {
      const mounted = await mountUnlocked()
      getBladeStatus.mockRejectedValueOnce(new Error('offline'))
      getBladeStatus.mockResolvedValue(statusOn())
      vi.useFakeTimers()
      await mounted.vm.toggleMowing()
      await tick(667)
      expect(repeats()).toHaveLength(1)
      mounted.wrapper.unmount()
    })
  })

  describe('emergency stop', () => {
    it('stops the repeat and clears mowing', async () => {
      const { wrapper, vm } = await startMowing()
      await tick(667)
      expect(repeats()).toHaveLength(1)
      await vm.emergencyStop()
      await tick(0)
      expect(vm.mowingActive).toBe(false)
      await tick(10000)
      expect(repeats()).toHaveLength(1)
      wrapper.unmount()
    })

    it('stops the repeat even when the E-stop request rejects', async () => {
      const { wrapper, vm } = await startMowing()
      store().submitCommand.mockImplementation((cmd) =>
        cmd === 'emergency' ? Promise.reject(new Error('network')) : Promise.resolve(ACCEPTED)
      )
      await vm.emergencyStop()
      await tick(0)
      expect(vm.mowingActive).toBe(false)
      await tick(10000)
      expect(repeats()).toHaveLength(0)
      wrapper.unmount()
    })

    it('ignores a tick whose status read was in flight when E-stop was pressed', async () => {
      const { wrapper, vm } = await startMowing()
      let resolveStatus
      getBladeStatus.mockReturnValueOnce(new Promise((r) => { resolveStatus = r }))
      await tick(667)
      expect(getBladeStatus).toHaveBeenCalled()
      await vm.emergencyStop()
      resolveStatus(statusOn())
      await tick(0)
      expect(repeats()).toHaveLength(0)
      expect(vm.mowingActive).toBe(false)
      wrapper.unmount()
    })
  })

  describe('Stop Mowing', () => {
    it('sends the disable and nothing repeats afterwards', async () => {
      const { wrapper, vm } = await startMowing()
      await tick(667)
      await vm.toggleMowing()
      await tick(0)
      expect(bladeCalls('disable')).toHaveLength(1)
      expect(vm.mowingActive).toBe(false)
      const before = repeats().length
      await tick(10000)
      expect(repeats()).toHaveLength(before)
      wrapper.unmount()
    })

    it('still ends mowing and keeps the repeat stopped when the disable rejects', async () => {
      const { wrapper, vm } = await startMowing()
      store().submitCommand.mockRejectedValueOnce({ response: { status: 500, data: { detail: 'boom' } } })
      await vm.toggleMowing()
      await tick(0)
      expect(vm.mowingActive).toBe(false)
      expect(vm.statusSuccess).toBe(false)
      await tick(10000)
      expect(repeats()).toHaveLength(0)
      wrapper.unmount()
    })

    it('does not let a status read in flight at Stop send another ON', async () => {
      const { wrapper, vm } = await startMowing()
      let resolveStatus
      getBladeStatus.mockReturnValueOnce(new Promise((r) => { resolveStatus = r }))
      await tick(667)
      await vm.toggleMowing()
      resolveStatus(statusOn())
      await tick(0)
      expect(repeats()).toHaveLength(0)
      expect(vm.mowingActive).toBe(false)
      wrapper.unmount()
    })
  })

  it('does not send an ON when the backend no longer has the blade commanded on', async () => {
    const { wrapper, vm } = await startMowing()
    getBladeStatus.mockResolvedValue(statusOn({ commanded: 'off', last_drop: 'deadman' }))
    await tick(667)
    expect(repeats()).toHaveLength(0)
    expect(vm.mowingActive).toBe(false)
    expect(vm.statusMessage).toContain('Blade stopped by the backend (deadman)')
    expect(vm.statusSuccess).toBe(false)
    expect(toastStore.show).toHaveBeenCalledWith(expect.stringContaining('deadman'), 'error', expect.any(Number))
    const reads = getBladeStatus.mock.calls.length
    await tick(10000)
    expect(getBladeStatus).toHaveBeenCalledTimes(reads)
    expect(repeats()).toHaveLength(0)
    wrapper.unmount()
  })

  it('says "no reason given" when the backend reports no last_drop', async () => {
    const { wrapper, vm } = await startMowing()
    getBladeStatus.mockResolvedValue(statusOn({ commanded: 'off', last_drop: null }))
    await tick(667)
    expect(vm.statusMessage).toContain('no reason given')
    wrapper.unmount()
  })

  describe('a failed repeat ends mowing', () => {
    it.each([
      ['409-shaped error', () => Promise.reject({ response: { status: 409, data: { accepted: false, result: 'rejected', reason: 'hold' } } })],
      ['network error', () => Promise.reject(new Error('Network Error'))],
      ['accepted:false reply', () => Promise.resolve({ accepted: false, result: 'rejected', reason: 'obstacle' })],
      ['blocked reply', () => Promise.resolve({ result: 'blocked' })],
    ])('%s', async (_name, failure) => {
      const { wrapper, vm } = await startMowing()
      store().submitCommand.mockImplementation(failure)
      await tick(667)
      expect(repeats()).toHaveLength(1)
      expect(vm.mowingActive).toBe(false)
      expect(vm.statusSuccess).toBe(false)
      expect(vm.statusMessage).not.toBe('')
      await tick(10000)
      expect(repeats()).toHaveLength(1)
      wrapper.unmount()
    })
  })

  it('runs only one status request at a time (in-flight guard)', async () => {
    const { wrapper } = await startMowing()
    getBladeStatus.mockClear()
    getBladeStatus.mockReturnValue(new Promise(() => {}))
    await tick(667 * 10)
    expect(getBladeStatus).toHaveBeenCalledTimes(1)
    expect(repeats()).toHaveLength(0)
    wrapper.unmount()
  })

  it('never takes the E-stop button out of service while a repeat is in flight', async () => {
    const { wrapper, vm } = await startMowing()
    getBladeStatus.mockReturnValue(new Promise(() => {}))
    await tick(667)
    expect(getBladeStatus).toHaveBeenCalled()
    expect(vm.performing).toBe(false)
    const button = wrapper.find('button.btn-emergency')
    expect(button.exists()).toBe(true)
    expect(button.attributes('disabled')).toBeUndefined()
    wrapper.unmount()
  })

  describe('leaving and locking', () => {
    it('unmount stops the repeat and sends one best-effort disable', async () => {
      const { wrapper } = await startMowing()
      await tick(667)
      store().submitCommand.mockRejectedValue(new Error('gone'))
      wrapper.unmount()
      await tick(0)
      expect(bladeCalls('disable', 'manual-control-leave')).toHaveLength(1)
      expect(bladeCalls('disable')[0][1].session_id).toBe('s1')
      const before = repeats().length
      await tick(10000)
      expect(repeats()).toHaveLength(before)
    })

    it('unmount without mowing sends no disable', async () => {
      const { wrapper } = await mountUnlocked()
      wrapper.unmount()
      expect(bladeCalls('disable')).toHaveLength(0)
    })

    it('locking the control (session expiry) stops the repeat and clears mowing', async () => {
      const { wrapper, vm } = await startMowing()
      await tick(667)
      vm.lockControl()
      await tick(0)
      expect(vm.mowingActive).toBe(false)
      const before = repeats().length
      await tick(10000)
      expect(repeats()).toHaveLength(before)
      wrapper.unmount()
    })

    it('a lock that lands while the first ON is in flight wins: no mowing, nothing repeats', async () => {
      const { wrapper, vm } = await mountUnlocked()
      getBladeStatus.mockResolvedValue(statusOn())
      vi.useFakeTimers()
      const request = Promise.withResolvers()
      store().submitCommand.mockReturnValueOnce(request.promise)
      const started = vm.toggleMowing()
      await tick(0)
      vm.lockControl() // the session expires while the request is out
      request.resolve(ACCEPTED)
      await started
      await tick(0)
      expect(vm.mowingActive).toBe(false)
      expect(vm.statusMessage).not.toBe('Mowing started')
      await tick(10000)
      expect(getBladeStatus).not.toHaveBeenCalled()
      expect(repeats()).toHaveLength(0)
      wrapper.unmount()
    })

    it('an answer of "blocked" to the stop request still ends mowing (nothing repeats the ON any more)', async () => {
      const { wrapper, vm } = await startMowing()
      store().submitCommand.mockResolvedValueOnce({ accepted: false, result: 'blocked', status_reason: 'SAFETY_LOCKOUT' })
      await vm.toggleMowing()
      await tick(0)
      expect(vm.mowingActive).toBe(false)
      expect(vm.statusMessage).toBe('Blade action blocked by safety system')
      await tick(10000)
      expect(repeats()).toHaveLength(0)
      wrapper.unmount()
    })
  })
})
