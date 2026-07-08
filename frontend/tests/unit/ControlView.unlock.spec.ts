// @ts-nocheck
// Pins the fail-closed manual-control unlock contract (Tier 1 item #2) so the
// Tier 3.14 view split (DashboardView/ControlView) can't silently regress it:
// isControlUnlocked must only ever flip true from an explicit success
// response, never from a catch/error branch.
import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest'
import { mount, flushPromises } from '@vue/test-utils'
import { ref, defineComponent, h } from 'vue'
import ControlView from '@/views/ControlView.vue'
import apiClient from '@/services/api'

const controlStoreContainer: { current: any } = { current: null }

vi.mock('@/stores/control', () => ({
  useControlStore: () => controlStoreContainer.current,
}))

const toastStore = {
  show: vi.fn(),
}

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
    submitCommand: vi.fn().mockResolvedValue({ result: 'accepted' }),
    fetchRoboHATStatus: vi.fn().mockResolvedValue({ telemetry_source: 'simulated' }),
    initWebSocket: vi.fn(),
    cleanup: vi.fn(),
  }
}

async function mountLocked() {
  const wrapper = mount(ControlView)
  await flushPromises()
  wrapper.vm.stopCameraFeed()
  await flushPromises()
  return wrapper
}

beforeEach(() => {
  controlStoreContainer.current = createMockControlStore()
  toastStore.show.mockReset()
  apiClient.get.mockReset()
  apiClient.post.mockReset()
  apiClient.get.mockImplementation((url: string) => {
    if (url.startsWith('/api/v2/camera')) {
      return Promise.resolve({ data: {} })
    }
    if (url === '/api/v2/settings/security') {
      return Promise.resolve({ data: { security_level: 'password', session_timeout_minutes: 15 } })
    }
    return Promise.resolve({ data: {} })
  })
})

afterEach(() => {
  controlStoreContainer.current = null
})

describe('ControlView manual-unlock fail-closed behavior', () => {
  it('unlocks only on an explicit successful unlock response', async () => {
    const wrapper = await mountLocked()
    const vm = wrapper.vm as any
    vm.authForm.password = 'secret'
    apiClient.post.mockResolvedValueOnce({ data: { session_id: 's1' } })

    await vm.authenticateControl()
    await flushPromises()

    expect(vm.isControlUnlocked).toBe(true)
    wrapper.unmount()
  })

  it('stays locked when the unlock endpoint is missing (404)', async () => {
    const wrapper = await mountLocked()
    const vm = wrapper.vm as any
    vm.authForm.password = 'secret'
    apiClient.post.mockRejectedValueOnce({ response: { status: 404 } })

    await vm.authenticateControl()
    await flushPromises()

    expect(vm.isControlUnlocked).toBe(false)
    expect(toastStore.show).toHaveBeenCalledWith(
      expect.stringContaining('unavailable'),
      'error',
      expect.any(Number)
    )
    wrapper.unmount()
  })

  it('stays locked when the unlock endpoint is not implemented (501)', async () => {
    const wrapper = await mountLocked()
    const vm = wrapper.vm as any
    vm.authForm.password = 'secret'
    apiClient.post.mockRejectedValueOnce({ response: { status: 501 } })

    await vm.authenticateControl()
    await flushPromises()

    expect(vm.isControlUnlocked).toBe(false)
    wrapper.unmount()
  })

  it('stays locked on a generic authentication failure', async () => {
    const wrapper = await mountLocked()
    const vm = wrapper.vm as any
    vm.authForm.password = 'wrong'
    apiClient.post.mockRejectedValueOnce({ response: { status: 401, data: { detail: 'Invalid password' } } })

    await vm.authenticateControl()
    await flushPromises()

    expect(vm.isControlUnlocked).toBe(false)
    expect(vm.authError).toBe('Invalid password')
    wrapper.unmount()
  })

  it('cloudflare verification unlocks only when the backend reports authorized', async () => {
    const wrapper = await mountLocked()
    const vm = wrapper.vm as any
    apiClient.get.mockImplementation((url: string) => {
      if (url === '/api/v2/control/manual-unlock/status') {
        return Promise.resolve({ data: { authorized: true, session_id: 's2' } })
      }
      return Promise.resolve({ data: {} })
    })

    await vm.verifyCloudflareAuth()
    await flushPromises()

    expect(vm.isControlUnlocked).toBe(true)
    wrapper.unmount()
  })

  it('cloudflare verification stays locked when the backend reports not authorized', async () => {
    const wrapper = await mountLocked()
    const vm = wrapper.vm as any
    apiClient.get.mockImplementation((url: string) => {
      if (url === '/api/v2/control/manual-unlock/status') {
        return Promise.resolve({ data: { authorized: false } })
      }
      return Promise.resolve({ data: {} })
    })

    await vm.verifyCloudflareAuth()
    await flushPromises()

    expect(vm.isControlUnlocked).toBe(false)
    wrapper.unmount()
  })

  it('cloudflare verification stays locked when the status check errors (assumed unauthorized)', async () => {
    const wrapper = await mountLocked()
    const vm = wrapper.vm as any
    apiClient.get.mockImplementation((url: string) => {
      if (url === '/api/v2/control/manual-unlock/status') {
        return Promise.reject(new Error('network down'))
      }
      return Promise.resolve({ data: {} })
    })

    await vm.verifyCloudflareAuth()
    await flushPromises()

    expect(vm.isControlUnlocked).toBe(false)
    expect(toastStore.show).toHaveBeenCalledWith(
      expect.stringContaining('remains locked'),
      'error',
      expect.any(Number)
    )
    wrapper.unmount()
  })
})
