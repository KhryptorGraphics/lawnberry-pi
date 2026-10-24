/* eslint-disable @typescript-eslint/ban-ts-comment */
// @ts-nocheck
// Pins the fail-closed manual-control unlock contract (Tier 1 item #2) so the
// Tier 3.14 view split (DashboardView/ControlView) can't silently regress it:
// isControlUnlocked must only ever flip true from an explicit success
// response, never from a catch/error branch.
import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest'
import { mount, flushPromises, RouterLinkStub } from '@vue/test-utils'
import { setActivePinia, createPinia } from 'pinia'
import { ref, defineComponent, h, nextTick } from 'vue'
import ControlView from '@/views/ControlView.vue'
import apiClient from '@/services/api'
import { useAuthStore } from '@/stores/auth'
import { SIGN_IN_REQUIRED_MESSAGE } from '@/utils/apiErrors'

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
  // The gate renders a <router-link> to /login while nobody is signed in.
  const wrapper = mount(ControlView, { global: { stubs: { RouterLink: RouterLinkStub } } })
  await flushPromises()
  wrapper.vm.stopCameraFeed()
  await flushPromises()
  return wrapper
}

async function mountUnlocked() {
  const wrapper = await mountLocked()
  const vm = wrapper.vm as any
  vm.authForm.password = 'secret'
  apiClient.post.mockResolvedValueOnce({ data: { session_id: 's1' } })
  await vm.authenticateControl()
  await flushPromises()
  expect(vm.isControlUnlocked).toBe(true)
  vm.stopCameraFeed()
  await flushPromises()
  return { wrapper, vm }
}

beforeEach(() => {
  setActivePinia(createPinia())
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
    expect(vm.authError).toBe('Manual control unlock is unavailable (backend endpoint missing).')
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
    expect(vm.authError).toBe('Manual control unlock is unavailable (backend endpoint missing).')
    wrapper.unmount()
  })

  it('stays locked on a generic authentication failure', async () => {
    const wrapper = await mountLocked()
    const vm = wrapper.vm as any
    vm.authForm.password = 'wrong'
    // Signed in (the unlock now needs the operator Bearer): a 401 is the wrong password. Signed out it is the
    // sign-in instruction instead, see "ControlView manual-unlock error messages" below.
    useAuthStore().token = 'tok-123'
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
    expect(vm.statusMessage).toBe('Cloudflare verification failed')
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

// POST /control/manual-unlock is now a step-up check: it needs the operator's Bearer as well as the password.
// The same 401 therefore means "sign in first" when signed out and "wrong password" when signed in.
describe('ControlView manual-unlock error messages', () => {
  async function failUnlock(rejection: unknown, token: string | null = 'tok-123') {
    useAuthStore().token = token
    const wrapper = await mountLocked()
    const vm = wrapper.vm as any
    vm.authForm.password = 'secret'
    apiClient.post.mockRejectedValueOnce(rejection)
    await vm.authenticateControl()
    await flushPromises()
    return { wrapper, vm }
  }

  it('signed out: a 401 says to sign in, and the gate shows the sign-in notice', async () => {
    const { wrapper, vm } = await failUnlock(
      { response: { status: 401, data: { detail: 'Operator authentication required' } } },
      null
    )

    expect(vm.isControlUnlocked).toBe(false)
    expect(vm.authError).toBe(SIGN_IN_REQUIRED_MESSAGE)
    expect(vm.statusMessage).toBe(SIGN_IN_REQUIRED_MESSAGE)
    expect(wrapper.find('.sign-in-notice').exists()).toBe(true)
    wrapper.unmount()
  })

  it('signed in: a 401 is a wrong password, shown as the backend says it, with no sign-in notice', async () => {
    const { wrapper, vm } = await failUnlock({ response: { status: 401, data: { detail: 'Invalid password' } } })

    expect(vm.isControlUnlocked).toBe(false)
    expect(vm.authError).toBe('Invalid password')
    expect(wrapper.find('.sign-in-notice').exists()).toBe(false)
    wrapper.unmount()
  })

  it('signed in: a 401 without a detail still reads "Invalid password"', async () => {
    const { wrapper, vm } = await failUnlock({ response: { status: 401 } })

    expect(vm.authError).toBe('Invalid password')
    wrapper.unmount()
  })

  it('the sign-in notice follows the auth store: signing in removes it', async () => {
    useAuthStore().token = null
    const wrapper = await mountLocked()
    expect(wrapper.find('.sign-in-notice').exists()).toBe(true)
    expect(wrapper.findComponent(RouterLinkStub).props('to')).toBe('/login')

    useAuthStore().token = 'tok-123'
    await nextTick()

    expect(wrapper.find('.sign-in-notice').exists()).toBe(false)
    wrapper.unmount()
  })

  it('429 shows how long to wait (Retry-After)', async () => {
    const { wrapper, vm } = await failUnlock({
      response: { status: 429, headers: { 'retry-after': '30' }, data: { detail: 'Too many failed attempts' } },
    })

    expect(vm.isControlUnlocked).toBe(false)
    expect(vm.authError).toBe('Too many attempts: try again in 30 s.')
    wrapper.unmount()
  })

  it('429 without Retry-After shows the backend detail', async () => {
    const { wrapper, vm } = await failUnlock({
      response: { status: 429, data: { detail: 'Too many failed attempts' } },
    })

    expect(vm.authError).toBe('Too many failed attempts')
    wrapper.unmount()
  })

  it('400 (blank password) shows the backend detail', async () => {
    const { wrapper, vm } = await failUnlock({
      response: { status: 400, data: { detail: 'Password required for manual unlock' } },
    })

    expect(vm.isControlUnlocked).toBe(false)
    expect(vm.authError).toBe('Password required for manual unlock')
    wrapper.unmount()
  })

  it('404 shows the endpoint-missing text', async () => {
    const { wrapper, vm } = await failUnlock({ response: { status: 404, data: { detail: 'Not Found' } } })

    expect(vm.isControlUnlocked).toBe(false)
    expect(vm.authError).toBe('Manual control unlock is unavailable (backend endpoint missing).')
    wrapper.unmount()
  })

  it('501 shows the backend detail when it has one (e.g. Google unlock)', async () => {
    const { wrapper, vm } = await failUnlock({
      response: { status: 501, data: { detail: 'Google-based manual unlock is not implemented' } },
    })

    expect(vm.isControlUnlocked).toBe(false)
    expect(vm.authError).toBe('Google-based manual unlock is not implemented')
    expect(toastStore.show).toHaveBeenCalledWith('Google-based manual unlock is not implemented', 'error', expect.any(Number))
    wrapper.unmount()
  })

  it('a transport failure keeps the error message', async () => {
    const { wrapper, vm } = await failUnlock(new Error('Network Error'))

    expect(vm.isControlUnlocked).toBe(false)
    expect(vm.authError).toBe('Network Error')
    wrapper.unmount()
  })
})

describe('ControlView Cloudflare security level', () => {
  const CLOUDFLARE_TEXT =
    'Cloudflare Access cannot be verified by this backend; manual unlock is unavailable with this security level. Choose Password in Settings > Security.'

  it('says Cloudflare Access cannot be verified and stays locked when the backend gives that reason', async () => {
    const wrapper = await mountLocked()
    const vm = wrapper.vm as any
    apiClient.get.mockImplementation((url: string) => {
      if (url === '/api/v2/control/manual-unlock/status') {
        return Promise.resolve({ data: { authorized: false, reason: 'cloudflare_access_not_verified' } })
      }
      return Promise.resolve({ data: {} })
    })

    await vm.verifyCloudflareAuth()
    await flushPromises()

    expect(vm.isControlUnlocked).toBe(false)
    expect(vm.authError).toBe(CLOUDFLARE_TEXT)
    expect(vm.statusMessage).toBe(CLOUDFLARE_TEXT)
    expect(toastStore.show).toHaveBeenCalledWith(CLOUDFLARE_TEXT, 'error', expect.any(Number))
    wrapper.unmount()
  })

  it('the Cloudflare card no longer claims you are already authenticated', async () => {
    apiClient.get.mockImplementation((url: string) => {
      if (url === '/api/v2/settings/security') {
        return Promise.resolve({ data: { security_level: 'cloudflare_tunnel_auth' } })
      }
      if (url === '/api/v2/control/manual-unlock/status') {
        return Promise.resolve({ data: { authorized: false, reason: 'cloudflare_access_not_verified' } })
      }
      return Promise.resolve({ data: {} })
    })
    const wrapper = await mountLocked()

    const card = wrapper.find('.security-gate')
    expect(card.text()).not.toContain('already be authenticated')
    expect(card.text()).toContain('cannot verify Cloudflare Access')
    expect(wrapper.vm.isControlUnlocked).toBe(false)
    wrapper.unmount()
  })
})

// Drive / blade / emergency failures used to read "Failed to ..." whatever went wrong.
describe('ControlView drive, blade and emergency-stop failure messages', () => {
  const unsigned = { response: { status: 401, data: { detail: 'Operator authentication required' } } }

  it('a drive command refused with 401 says to sign in', async () => {
    const { wrapper, vm } = await mountUnlocked()
    controlStoreContainer.current.submitCommand.mockRejectedValueOnce(unsigned)

    await vm.queueDriveCommand({ linear: 0.5, angular: 0 })

    expect(vm.statusMessage).toBe(SIGN_IN_REQUIRED_MESSAGE)
    wrapper.unmount()
  })

  it('a drive command that gets no response says so', async () => {
    const { wrapper, vm } = await mountUnlocked()
    controlStoreContainer.current.submitCommand.mockRejectedValueOnce(new Error('Network Error'))

    await vm.queueDriveCommand({ linear: 0.5, angular: 0 })

    expect(vm.statusMessage).toBe('Failed to send drive command: no response from the backend.')
    wrapper.unmount()
  })

  it('a blade toggle refused with 401 says to sign in and does not flip the mowing state', async () => {
    const { wrapper, vm } = await mountUnlocked()
    controlStoreContainer.current.submitCommand.mockRejectedValueOnce(unsigned)

    await vm.toggleMowing()

    expect(vm.statusMessage).toBe(SIGN_IN_REQUIRED_MESSAGE)
    expect(vm.mowingActive).toBe(false)
    wrapper.unmount()
  })

  it('a 401 about the manual-control session (not the Bearer) returns to the gate instead of saying "sign in"', async () => {
    const { wrapper, vm } = await mountUnlocked()
    controlStoreContainer.current.submitCommand.mockRejectedValueOnce({
      response: { status: 401, data: { detail: 'Manual control session invalid or expired' } },
    })

    await vm.toggleMowing()
    await flushPromises()

    expect(vm.statusMessage).toBe('Manual control session invalid or expired. Unlock manual control again.')
    expect(vm.statusMessage).not.toContain('Sign in')
    expect(vm.isControlUnlocked).toBe(false)
    expect(wrapper.find('.security-gate').exists()).toBe(true)
    wrapper.unmount()
  })

  it('the emergency stop never tells the operator to sign in, even for a 401', async () => {
    const { wrapper, vm } = await mountUnlocked()
    controlStoreContainer.current.submitCommand.mockRejectedValueOnce(unsigned)

    await vm.emergencyStop()

    expect(vm.statusMessage).toBe('Failed to trigger emergency stop')
    expect(vm.statusMessage).not.toMatch(/sign in/i)
    wrapper.unmount()
  })

  it('an emergency stop that gets no response says so', async () => {
    const { wrapper, vm } = await mountUnlocked()
    controlStoreContainer.current.submitCommand.mockRejectedValueOnce(new Error('Network Error'))

    await vm.emergencyStop()

    expect(vm.statusMessage).toBe('Failed to trigger emergency stop: no response from the backend.')
    wrapper.unmount()
  })

  it('the emergency stop needs no sign-in: signed out, it is still sent', async () => {
    useAuthStore().token = null
    const { wrapper, vm } = await mountUnlocked()

    await vm.emergencyStop()

    expect(controlStoreContainer.current.submitCommand).toHaveBeenCalledWith(
      'emergency',
      expect.objectContaining({ session_id: expect.any(String) })
    )
    expect(vm.statusMessage).toBe('Emergency stop activated')
    wrapper.unmount()
  })
})

// POST /control/emergency answers accepted:false, result:"incomplete" when the stop is latched but not confirmed on
// every output. That must never be reported as a clean "Emergency stop activated".
describe('ControlView emergency stop that is not confirmed on every output', () => {
  const NOT_CONFIRMED = 'Emergency stop latched, but NOT every output was confirmed'

  it.each([
    ['accepted:false, result:"incomplete"', { accepted: false, result: 'incomplete', status_reason: 'EMERGENCY_STOP_INCOMPLETE' }],
    ['accepted:false alone', { accepted: false, result: 'rejected', status_reason: 'EMERGENCY_STOP_INCOMPLETE' }],
    ['result:"incomplete" alone', { accepted: true, result: 'incomplete', status_reason: 'EMERGENCY_STOP_INCOMPLETE' }],
  ])('%s: shows an error with the reason, never "Emergency stop activated", and still clears speed and mowing', async (_name, reply) => {
    const { wrapper, vm } = await mountUnlocked()
    controlStoreContainer.current.submitCommand.mockResolvedValueOnce(reply)
    vm.mowingActive = true
    vm.currentSpeed = 0.7

    await vm.emergencyStop()

    expect(vm.statusMessage).toBe(`${NOT_CONFIRMED} (EMERGENCY_STOP_INCOMPLETE)`)
    expect(vm.statusSuccess).toBe(false)
    expect(vm.statusMessage).not.toContain('Emergency stop activated')
    expect(toastStore.show).toHaveBeenCalledWith(`${NOT_CONFIRMED} (EMERGENCY_STOP_INCOMPLETE)`, 'error', expect.any(Number))
    expect(vm.mowingActive).toBe(false)
    expect(vm.currentSpeed).toBe(0)
    wrapper.unmount()
  })

  it('without a status_reason the message is just the plain warning', async () => {
    const { wrapper, vm } = await mountUnlocked()
    controlStoreContainer.current.submitCommand.mockResolvedValueOnce({ accepted: false, result: 'incomplete' })

    await vm.emergencyStop()

    expect(vm.statusMessage).toBe(NOT_CONFIRMED)
    expect(vm.statusSuccess).toBe(false)
    wrapper.unmount()
  })

  it('a confirmed stop (accepted:true, result:"accepted") is still reported as activated, without a toast', async () => {
    const { wrapper, vm } = await mountUnlocked()
    controlStoreContainer.current.submitCommand.mockResolvedValueOnce({
      accepted: true,
      result: 'accepted',
      status_reason: 'EMERGENCY_STOP_TRIGGERED',
    })
    toastStore.show.mockClear()

    await vm.emergencyStop()

    expect(vm.statusMessage).toBe('Emergency stop activated')
    expect(vm.statusSuccess).toBe(true)
    expect(toastStore.show).not.toHaveBeenCalled()
    wrapper.unmount()
  })
})
