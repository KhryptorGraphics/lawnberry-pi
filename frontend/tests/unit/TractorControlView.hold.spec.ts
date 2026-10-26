// SF-08: the backend only holds a lever/throttle command for a short window (500 ms by
// default, echoed as `hold_ms`), so the page has to keep re-sending what the operator is
// holding and has to stop -- sending neutral -- the moment the operator, the page or the link
// stops. These tests drive the real TractorControlView, the real tractor store and the real
// useTractorHold composable on fake timers; only services/api (the network boundary) is mocked.
import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest'
import { mount, flushPromises, type VueWrapper } from '@vue/test-utils'
import { setActivePinia, createPinia } from 'pinia'
import TractorControlView from '@/views/TractorControlView.vue'
import { useToastStore } from '@/stores/toast'
import {
  FALLBACK_REFRESH_MS,
  commandTimeoutFor,
  refreshIntervalFor,
} from '@/composables/useTractorHold'
import { SIGN_IN_REQUIRED_MESSAGE } from '@/utils/apiErrors'
import apiClient from '@/services/api'
import * as api from '@/services/api'
import type { TractorActuatorResponse, TractorSafetyResponse, TractorState } from '@/types/control'

type Control = 'left' | 'right' | 'throttle'
interface Call {
  t: number
  control: Control
  value: number
  options?: api.TractorCommandOptions
}
type Actuator = (value: number, options?: api.TractorCommandOptions) => Promise<TractorActuatorResponse>

const STATE_AT = (second: number) => new Date(2026, 0, 1, 12, 0, second).toISOString()

// Not Promise.withResolvers: CI runs the unit tests on the Playwright image's Node 20, which lacks it.
function deferred() {
  let resolve!: () => void
  const promise = new Promise<void>((done) => {
    resolve = done
  })
  return { promise, resolve }
}

function baseState(overrides: Partial<TractorState> = {}): TractorState {
  return {
    left_lever: 0,
    right_lever: 0,
    throttle: 0,
    blade_engaged: false,
    engine: 'running',
    enabled: true,
    emergency_stop_active: false,
    authorized: true,
    interlock_reason: null,
    last_updated: STATE_AT(0),
    engine_running: true,
    moving: false,
    reversing: false,
    ...overrides,
  }
}

let calls: Call[] = []
let wrappers: VueWrapper[] = []

const FIELD: Record<Control, string> = { left: 'left_lever', right: 'right_lever', throttle: 'throttle' }
const ACTUATORS = {
  left: api.tractorLeftLever,
  right: api.tractorRightLever,
  throttle: api.tractorThrottle,
}

/** A backend that accepts every command and echoes `hold_ms` like tractor_service.py does. */
function installBackend(options: { holdMs?: number | null; delayMs?: number } = {}) {
  const holdMs = options.holdMs === undefined ? 500 : options.holdMs
  const answer =
    (control: Control): Actuator =>
    async (value, requestOptions) => {
      calls.push({ t: Date.now(), control, value, options: requestOptions })
      if (options.delayMs) {
        const { promise, resolve } = deferred()
        setTimeout(resolve, options.delayMs)
        await promise
      }
      return {
        status: 'ok',
        [FIELD[control]]: value,
        ...(holdMs !== null && value !== 0 ? { hold_ms: holdMs } : {}),
      }
    }
  for (const control of ['left', 'right', 'throttle'] as const) {
    vi.mocked(ACTUATORS[control]).mockImplementation(answer(control))
  }
  vi.mocked(api.tractorEmergencyStop).mockResolvedValue({
    status: 'emergency_stop',
    complete: true,
    faults: [],
  })
}

/** Replace one actuator's behaviour, still recording every call it receives. */
function replaceBackend(control: Control, reply: (value: number) => TractorActuatorResponse | Error) {
  vi.mocked(ACTUATORS[control]).mockImplementation(async (value) => {
    calls.push({ t: Date.now(), control, value })
    const result = reply(value)
    if (result instanceof Error) throw result
    return result
  })
}

async function mountView(state = baseState()) {
  vi.mocked(api.getTractorState).mockResolvedValue(state)
  vi.mocked(apiClient.get).mockResolvedValue({ data: {} })
  vi.mocked(apiClient.post).mockResolvedValue({ data: {} })
  const wrapper = mount(TractorControlView)
  wrappers.push(wrapper)
  await flushPromises()
  return wrapper
}

const slider = (wrapper: VueWrapper, testId: string) => wrapper.find(`[data-testid="${testId}"]`)
const sliderValue = (wrapper: VueWrapper, testId: string) =>
  Number((slider(wrapper, testId).element as HTMLInputElement).value)
const text = (wrapper: VueWrapper, testId: string) => slider(wrapper, testId).text()
const callsFor = (control: Control) => calls.filter((c) => c.control === control)
const valuesFor = (control: Control) => callsFor(control).map((c) => c.value)
const lastValue = (control: Control) => valuesFor(control).at(-1)
const gaps = (list: Call[]) => list.slice(1).map((c, i) => c.t - list[i].t)

function button(wrapper: VueWrapper, label: string) {
  const found = wrapper.findAll('button').find((b) => b.text().includes(label))
  if (!found) throw new Error(`no button labelled "${label}"`)
  return found
}

/**
 * Put a pointer on a slider and move it to `value` the way a real slider drag/commit does: pointerdown,
 * input, change. Only elements and events that exist on the pre-SF-08 page are used, so the same
 * interaction can be replayed against the old page to show what it did wrong.
 */
async function dragTo(wrapper: VueWrapper, testId: string, value: number, pointerId = 1) {
  await slider(wrapper, testId).trigger('pointerdown', { pointerId })
  await slider(wrapper, testId).setValue(value)
  await slider(wrapper, testId).trigger('change')
  await flushPromises()
}

/** The pointer goes up somewhere on the page (the page listens on window, not on the slider). */
async function releasePointer() {
  window.dispatchEvent(new Event('pointerup'))
  await flushPromises()
}

function setVisibility(state: 'visible' | 'hidden') {
  Object.defineProperty(document, 'visibilityState', { configurable: true, get: () => state })
  document.dispatchEvent(new Event('visibilitychange'))
}

interface WsMockInstance {
  topicCallbacks: Map<string, Array<{ callback: (data: unknown) => void }>>
}
const wsInstances = () =>
  (globalThis as unknown as { __wsMockInstances: WsMockInstance[] }).__wsMockInstances

function emitTractorTick(state: TractorState) {
  for (const entry of wsInstances()) {
    entry.topicCallbacks
      .get('telemetry.tractor')
      ?.forEach((cb) => cb.callback({ tractor: state, source: 'hardware' }))
  }
}

const toastMessages = () => useToastStore().toasts.map((t) => t.message)

describe('TractorControlView hold-to-run (SF-08)', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
    vi.clearAllMocks()
    vi.useFakeTimers()
    calls = []
    wrappers = []
    installBackend()
  })

  afterEach(() => {
    wrappers.forEach((w) => w.unmount())
    vi.useRealTimers()
    Reflect.deleteProperty(document, 'visibilityState')
    wsInstances().length = 0
  })

  describe('interval maths', () => {
    it('refreshes three times per echoed hold, clamped, with a conservative fallback', () => {
      expect(refreshIntervalFor(500)).toBe(166)
      expect(refreshIntervalFor(300)).toBe(100)
      expect(refreshIntervalFor(60)).toBe(50) // never faster than 20 Hz
      expect(refreshIntervalFor(5000)).toBe(250) // never slower than 4 Hz
      expect(refreshIntervalFor(null)).toBe(FALLBACK_REFRESH_MS)
      expect(refreshIntervalFor(0)).toBe(FALLBACK_REFRESH_MS)
      expect(FALLBACK_REFRESH_MS).toBeLessThanOrEqual(200) // well under the 500 ms default hold
    })

    it('times a command out at the hold: a reply slower than the hold means the hold lapsed', () => {
      expect(commandTimeoutFor(null)).toBe(500)
      expect(commandTimeoutFor(500)).toBe(500)
      expect(commandTimeoutFor(100)).toBe(300)
      expect(commandTimeoutFor(2000)).toBe(1000)
    })
  })

  describe('re-sending a held command', () => {
    it('keeps re-sending a latched lever command inside the backend hold (the old page posted it once)', async () => {
      const wrapper = await mountView()
      await slider(wrapper, 'latch-toggle').setValue(true)

      // What the old page did on a slider release: input, then one change.
      await slider(wrapper, 'left-lever-slider').setValue(0.5)
      await slider(wrapper, 'left-lever-slider').trigger('change')
      await flushPromises()
      expect(valuesFor('left')).toEqual([0.5])

      await vi.advanceTimersByTimeAsync(1000)

      const sent = callsFor('left')
      expect(sent.length).toBeGreaterThanOrEqual(6)
      expect(new Set(valuesFor('left'))).toEqual(new Set([0.5]))
      // Every gap is inside the 500 ms hold the backend echoed, with room for two lost refreshes.
      expect(Math.max(...gaps(sent))).toBeLessThan(250)
    })

    it('keeps re-sending while the operator holds the slider, then neutral on release', async () => {
      const wrapper = await mountView()
      await dragTo(wrapper, 'left-lever-slider', 0.6)

      await vi.advanceTimersByTimeAsync(1000)
      expect(callsFor('left').length).toBeGreaterThanOrEqual(6)
      expect(valuesFor('left').every((v) => v === 0.6)).toBe(true)

      await releasePointer()
      expect(lastValue('left')).toBe(0)
      expect(lastValue('right')).toBe(0)
      expect(sliderValue(wrapper, 'left-lever-slider')).toBe(0)

      const afterRelease = calls.length
      await vi.advanceTimersByTimeAsync(3000)
      expect(calls.length).toBe(afterRelease) // nothing more once released
      expect(text(wrapper, 'hold-status')).toContain('Not commanding')
    })

    it('writes both levers every time, left then right, so the machine does what the page shows', async () => {
      const wrapper = await mountView()
      await dragTo(wrapper, 'left-lever-slider', 0.6)
      await vi.advanceTimersByTimeAsync(400)

      const pairs = calls.filter((c) => c.control !== 'throttle')
      expect(pairs.length).toBeGreaterThanOrEqual(4)
      pairs.forEach((c, i) => expect(c.control).toBe(i % 2 === 0 ? 'left' : 'right'))
      expect(valuesFor('right').every((v) => v === 0)).toBe(true)
    })

    it('derives the refresh period from the hold the backend echoes, not from a constant', async () => {
      installBackend({ holdMs: 300 })
      const wrapper = await mountView()
      await dragTo(wrapper, 'left-lever-slider', 0.5)
      calls.length = 0

      await vi.advanceTimersByTimeAsync(1000)

      const sent = callsFor('left')
      expect(sent.length).toBeGreaterThanOrEqual(9) // 300 ms hold -> every 100 ms
      expect(Math.max(...gaps(sent))).toBeLessThanOrEqual(100)
      expect(text(wrapper, 'hold-status')).toContain('backend hold 300 ms')
      expect(text(wrapper, 'hold-status')).toContain('every 100 ms')
    })

    it('falls back to a conservative constant when the backend echoes no hold', async () => {
      installBackend({ holdMs: null })
      const wrapper = await mountView()
      await dragTo(wrapper, 'left-lever-slider', 0.5)
      calls.length = 0

      await vi.advanceTimersByTimeAsync(1000)

      const sent = callsFor('left')
      expect(sent.length).toBeGreaterThanOrEqual(6)
      expect(Math.max(...gaps(sent))).toBeLessThanOrEqual(FALLBACK_REFRESH_MS)
    })

    it('bounds every command by a timeout tied to the hold', async () => {
      const wrapper = await mountView()
      await dragTo(wrapper, 'left-lever-slider', 0.5)
      await vi.advanceTimersByTimeAsync(300)

      expect(callsFor('left')[0].options).toEqual({ timeout: 500 })
      callsFor('left').forEach((c) => expect(c.options?.timeout).toBeLessThanOrEqual(500))
    })

    it('never has two requests in flight, even on a slow link', async () => {
      installBackend({ delayMs: 400 })
      let inFlight = 0
      let peak = 0
      for (const control of ['left', 'right'] as const) {
        const inner = vi.mocked(ACTUATORS[control]).getMockImplementation()
        if (!inner) throw new Error('backend not installed')
        vi.mocked(ACTUATORS[control]).mockImplementation(async (value, options) => {
          inFlight += 1
          peak = Math.max(peak, inFlight)
          try {
            return await inner(value, options)
          } finally {
            inFlight -= 1
          }
        })
      }
      const wrapper = await mountView()
      await dragTo(wrapper, 'left-lever-slider', 0.5)

      await vi.advanceTimersByTimeAsync(3000)

      expect(callsFor('left').length).toBeGreaterThanOrEqual(3)
      expect(peak).toBe(1)
    })

    it('holds the throttle the same way and idles it on release', async () => {
      const wrapper = await mountView()
      await dragTo(wrapper, 'throttle-slider', 0.7)
      await vi.advanceTimersByTimeAsync(1000)

      expect(callsFor('throttle').length).toBeGreaterThanOrEqual(6)
      expect(new Set(valuesFor('throttle'))).toEqual(new Set([0.7]))
      expect(valuesFor('left')).toEqual([]) // the levers are not this control's business

      await releasePointer()
      expect(lastValue('throttle')).toBe(0)
      expect(sliderValue(wrapper, 'throttle-slider')).toBe(0)
    })

    it('sends neutral and stops when the slider is dragged back into the detent', async () => {
      const wrapper = await mountView()
      await dragTo(wrapper, 'left-lever-slider', 0.6)
      await vi.advanceTimersByTimeAsync(300)

      await slider(wrapper, 'left-lever-slider').setValue(0.05) // inside the backend's 0.05 neutral band
      await flushPromises()
      expect(lastValue('left')).toBe(0) // neutral is sent as exactly 0, not 0.05

      const sentSoFar = calls.length
      await vi.advanceTimersByTimeAsync(2000)
      expect(calls.length).toBe(sentSoFar)
    })
  })

  describe('latch', () => {
    it('keeps commanding after release while latch is on, and neutral the moment it is turned off', async () => {
      const wrapper = await mountView()
      await slider(wrapper, 'latch-toggle').setValue(true)
      await dragTo(wrapper, 'left-lever-slider', 0.5)
      await releasePointer()

      expect(sliderValue(wrapper, 'left-lever-slider')).toBe(0.5)
      await vi.advanceTimersByTimeAsync(1000)
      expect(valuesFor('left').every((v) => v === 0.5)).toBe(true)

      await slider(wrapper, 'latch-toggle').setValue(false)
      await flushPromises()
      expect(lastValue('left')).toBe(0)
      expect(sliderValue(wrapper, 'left-lever-slider')).toBe(0)
      const after = calls.length
      await vi.advanceTimersByTimeAsync(2000)
      expect(calls.length).toBe(after)
    })
  })

  describe('hold-to-run needs a physical grip', () => {
    it('ignores a slider value that arrives without a pointer or key down (assistive tech, wheel, script)', async () => {
      const wrapper = await mountView()

      await slider(wrapper, 'left-lever-slider').setValue(0.8)
      await vi.advanceTimersByTimeAsync(1000)

      expect(calls).toEqual([])
      expect(sliderValue(wrapper, 'left-lever-slider')).toBe(0)
    })

    it('releases when the slider loses focus or a key goes up', async () => {
      const wrapper = await mountView()
      await slider(wrapper, 'right-lever-slider').trigger('keydown', { key: 'ArrowRight' })
      await slider(wrapper, 'right-lever-slider').setValue(0.4)
      await flushPromises()
      expect(lastValue('right')).toBe(0.4)

      await slider(wrapper, 'right-lever-slider').trigger('keyup', { key: 'ArrowRight' })
      await flushPromises()
      expect(lastValue('right')).toBe(0)

      await slider(wrapper, 'left-lever-slider').trigger('keydown', { key: 'ArrowRight' })
      await slider(wrapper, 'left-lever-slider').setValue(0.4)
      await flushPromises()
      await slider(wrapper, 'left-lever-slider').trigger('blur')
      await flushPromises()
      expect(lastValue('left')).toBe(0)
    })

    it('does not release a control when a different pointer goes up', async () => {
      const wrapper = await mountView()
      await dragTo(wrapper, 'left-lever-slider', 0.5, 7)
      const sentSoFar = callsFor('left').length

      const other = new Event('pointerup')
      Object.defineProperty(other, 'pointerId', { value: 99 })
      window.dispatchEvent(other)
      await vi.advanceTimersByTimeAsync(400)

      expect(callsFor('left').length).toBeGreaterThan(sentSoFar) // still being re-sent
      expect(lastValue('left')).toBe(0.5)
    })

    it('keeps a pointer grip when a key goes down on the same slider', async () => {
      const wrapper = await mountView()
      await dragTo(wrapper, 'left-lever-slider', 0.5, 3)
      await slider(wrapper, 'left-lever-slider').trigger('keydown', { key: 'Shift' })

      await releasePointer() // the pointer goes up: the control must still be released

      expect(lastValue('left')).toBe(0)
      expect(sliderValue(wrapper, 'left-lever-slider')).toBe(0)
    })
  })

  describe('stop conditions', () => {
    it('sends neutral and stops when the page is hidden, and says why', async () => {
      const wrapper = await mountView()
      await dragTo(wrapper, 'left-lever-slider', 0.6)
      await vi.advanceTimersByTimeAsync(300)

      setVisibility('hidden')
      await flushPromises()

      expect(lastValue('left')).toBe(0)
      expect(lastValue('right')).toBe(0)
      expect(sliderValue(wrapper, 'left-lever-slider')).toBe(0)
      const after = calls.length
      await vi.advanceTimersByTimeAsync(3000)
      expect(calls.length).toBe(after)
      expect(text(wrapper, 'hold-notice')).toContain('hidden')
    })

    it('does not resume on its own when the page becomes visible again', async () => {
      const wrapper = await mountView()
      await dragTo(wrapper, 'left-lever-slider', 0.6)
      setVisibility('hidden')
      await flushPromises()
      const after = calls.length

      setVisibility('visible')
      await vi.advanceTimersByTimeAsync(2000)

      expect(calls.length).toBe(after)
      expect(sliderValue(wrapper, 'left-lever-slider')).toBe(0)
    })

    it('sends neutral on pagehide and when the browser goes offline', async () => {
      const wrapper = await mountView()
      await dragTo(wrapper, 'left-lever-slider', 0.6)
      await vi.advanceTimersByTimeAsync(200)
      window.dispatchEvent(new Event('pagehide'))
      await flushPromises()
      expect(lastValue('left')).toBe(0)

      await dragTo(wrapper, 'right-lever-slider', 0.6)
      await vi.advanceTimersByTimeAsync(200)
      window.dispatchEvent(new Event('offline'))
      await flushPromises()
      expect(lastValue('right')).toBe(0)
      expect(text(wrapper, 'hold-notice')).toContain('offline')
    })

    it('sends neutral and stops on unmount', async () => {
      const wrapper = await mountView()
      await dragTo(wrapper, 'left-lever-slider', 0.6)
      await vi.advanceTimersByTimeAsync(300)

      wrapper.unmount()
      wrappers = []
      await flushPromises()

      expect(lastValue('left')).toBe(0)
      expect(lastValue('right')).toBe(0)
      const after = calls.length
      await vi.advanceTimersByTimeAsync(3000)
      expect(calls.length).toBe(after)
    })

    it('idles a held throttle when the page is hidden or unmounted', async () => {
      const wrapper = await mountView()
      await dragTo(wrapper, 'throttle-slider', 0.9)
      await vi.advanceTimersByTimeAsync(300)
      setVisibility('hidden')
      await flushPromises()
      expect(lastValue('throttle')).toBe(0)
      expect(sliderValue(wrapper, 'throttle-slider')).toBe(0)
      setVisibility('visible')

      await dragTo(wrapper, 'throttle-slider', 0.5)
      await vi.advanceTimersByTimeAsync(300)
      wrapper.unmount()
      wrappers = []
      await flushPromises()
      expect(lastValue('throttle')).toBe(0)
      const after = calls.length
      await vi.advanceTimersByTimeAsync(3000)
      expect(calls.length).toBe(after)
    })

    it('treats a lost connection as a stop: toast, neutral attempted once, nothing re-sent', async () => {
      const wrapper = await mountView()
      await dragTo(wrapper, 'left-lever-slider', 0.6)
      await vi.advanceTimersByTimeAsync(300)
      const before = calls.length

      // The link dies: every request now fails without an HTTP response (timeout / network error).
      replaceBackend('left', () => new Error('timeout of 500ms exceeded'))
      replaceBackend('right', () => new Error('timeout of 500ms exceeded'))
      await vi.advanceTimersByTimeAsync(200)

      expect(sliderValue(wrapper, 'left-lever-slider')).toBe(0)
      expect(toastMessages().some((m) => m.includes('Left lever') && m.includes('no response'))).toBe(true)

      // The live write that failed, then the stop tries neutral once per lever: no retry loop.
      expect(calls.slice(before).map((c) => `${c.control}:${c.value}`)).toEqual(['left:0.6', 'left:0', 'right:0'])
      expect(toastMessages().filter((m) => m.includes('neutral not confirmed'))).toHaveLength(2)
      const attemptsAfterFailure = calls.length
      await vi.advanceTimersByTimeAsync(5000)
      expect(calls.length).toBe(attemptsAfterFailure) // and nothing re-commands on its own
      expect(text(wrapper, 'hold-notice')).toContain('no response')
      expect(text(wrapper, 'hold-status')).toContain('Not commanding')
    })

    it('stops on a rejected command with the backend reason, and does not retry it', async () => {
      const wrapper = await mountView()
      await dragTo(wrapper, 'left-lever-slider', 0.6)
      await vi.advanceTimersByTimeAsync(300)

      replaceBackend('left', (value) =>
        value === 0 ? { status: 'ok' } : { status: 'rejected', reason: 'actuator fault: PCA9685' }
      )
      await vi.advanceTimersByTimeAsync(200)

      expect(toastMessages()).toContain('Left lever: actuator fault: PCA9685')
      expect(sliderValue(wrapper, 'left-lever-slider')).toBe(0)
      expect(lastValue('left')).toBe(0) // a neutral followed the refusal
      const after = calls.length
      await vi.advanceTimersByTimeAsync(3000)
      expect(calls.length).toBe(after)
    })

    it('shows the sign-in instruction when the backend wants an operator login (401)', async () => {
      const wrapper = await mountView()
      vi.mocked(api.tractorLeftLever).mockRejectedValue({
        response: { status: 401, data: { detail: 'Operator authentication required' } },
      })

      await dragTo(wrapper, 'left-lever-slider', 0.6)
      await flushPromises()

      expect(toastMessages().some((m) => m.includes(SIGN_IN_REQUIRED_MESSAGE))).toBe(true)
      expect(sliderValue(wrapper, 'left-lever-slider')).toBe(0)
    })

    it('a latched E-stop from the backend ends every hold without sending anything', async () => {
      const wrapper = await mountView()
      await dragTo(wrapper, 'left-lever-slider', 0.6)
      await vi.advanceTimersByTimeAsync(300)
      const before = calls.length

      emitTractorTick(baseState({ emergency_stop_active: true, authorized: false, last_updated: STATE_AT(5) }))
      await flushPromises()

      expect(sliderValue(wrapper, 'left-lever-slider')).toBe(0)
      await vi.advanceTimersByTimeAsync(3000)
      expect(calls.length).toBe(before) // no neutral writes (the E-stop drove neutral), no refresh
      expect(text(wrapper, 'hold-notice')).toContain('Emergency stop')
    })

    it('the E-stop button halts every hold first and is clickable while another request is in flight', async () => {
      const wrapper = await mountView()
      await dragTo(wrapper, 'left-lever-slider', 0.6)
      await vi.advanceTimersByTimeAsync(300)

      // A slow authorize request keeps the page `busy`.
      vi.mocked(api.tractorAuthorize).mockReturnValue(new Promise(() => undefined))
      await button(wrapper, 'Revoke').trigger('click')
      await flushPromises()

      const estop = button(wrapper, 'EMERGENCY STOP')
      expect(estop.attributes('disabled')).toBeUndefined()
      const writesBefore = calls.length
      await estop.trigger('click')
      await flushPromises()

      expect(api.tractorEmergencyStop).toHaveBeenCalledTimes(1)
      expect(sliderValue(wrapper, 'left-lever-slider')).toBe(0)
      await vi.advanceTimersByTimeAsync(3000)
      expect(calls.length).toBe(writesBefore) // nothing else sent after the E-stop
    })
  })

  describe('what the page shows', () => {
    it('shows what the backend is holding, and keeps the sliders at what this page commands', async () => {
      const wrapper = await mountView(baseState({ left_lever: 0.4, right_lever: -0.6, throttle: 0.3 }))

      expect(api.getTractorState).toHaveBeenCalledTimes(1)
      expect(text(wrapper, 'left-lever-held')).toContain('0.40 · Forward')
      expect(text(wrapper, 'right-lever-held')).toContain('-0.60 · Reverse')
      expect(text(wrapper, 'throttle-held')).toContain('0.30')
      // A fresh page commands nothing, so its sliders must not claim otherwise.
      expect(sliderValue(wrapper, 'left-lever-slider')).toBe(0)
      expect(sliderValue(wrapper, 'right-lever-slider')).toBe(0)
      expect(calls).toEqual([])
    })

    it('follows the backend state, not the slider, when the backend lets a lever go', async () => {
      const wrapper = await mountView()
      await slider(wrapper, 'latch-toggle').setValue(true)
      await dragTo(wrapper, 'left-lever-slider', 0.5)
      await vi.advanceTimersByTimeAsync(300)
      expect(text(wrapper, 'left-lever-held')).toContain('0.50 · Forward')

      // The backend deadman fired (a tick shows the lever back at neutral) while the slider is up.
      emitTractorTick(baseState({ left_lever: 0, last_updated: STATE_AT(9) }))
      await flushPromises()

      expect(text(wrapper, 'left-lever-command')).toContain('0.50')
      expect(text(wrapper, 'left-lever-held')).toContain('0.00 · Neutral')
    })

    it('shows the effective hold and refresh period while commanding', async () => {
      const wrapper = await mountView()
      expect(text(wrapper, 'hold-status')).toContain('Not commanding')

      await dragTo(wrapper, 'left-lever-slider', 0.5)

      expect(text(wrapper, 'hold-status')).toContain('every 166 ms')
      expect(text(wrapper, 'hold-status')).toContain('backend hold 500 ms')
    })
  })

  describe('both levers to neutral', () => {
    it('writes left, waits for the reply, then right (concurrent writes each read the other lever stale)', async () => {
      const wrapper = await mountView(baseState({ left_lever: 0.5, right_lever: 0.5 }))
      const order: string[] = []
      const leftReply = deferred()
      vi.mocked(api.tractorLeftLever).mockImplementation(async (value) => {
        order.push(`left:start:${value}`)
        await leftReply.promise
        order.push('left:done')
        return { status: 'ok' }
      })
      vi.mocked(api.tractorRightLever).mockImplementation(async (value) => {
        order.push(`right:start:${value}`)
        return { status: 'ok' }
      })

      await slider(wrapper, 'neutral-button').trigger('click')
      await flushPromises()
      expect(order).toEqual(['left:start:0']) // right has not been sent yet

      leftReply.resolve()
      await flushPromises()
      expect(order).toEqual(['left:start:0', 'left:done', 'right:start:0'])
    })

    it('reports a refused neutral to the operator', async () => {
      const wrapper = await mountView()
      replaceBackend('left', () => ({ status: 'rejected', reason: 'actuators not ready' }))
      replaceBackend('right', () => ({ status: 'rejected', reason: 'actuators not ready' }))

      await slider(wrapper, 'neutral-button').trigger('click')
      await flushPromises()

      expect(toastMessages().some((m) => m.includes('neutral refused') && m.includes('actuators not ready'))).toBe(true)
    })

    it('stops a running stream and sends neutral', async () => {
      const wrapper = await mountView()
      await slider(wrapper, 'latch-toggle').setValue(true)
      await dragTo(wrapper, 'left-lever-slider', 0.5)
      await dragTo(wrapper, 'right-lever-slider', 0.5)
      await vi.advanceTimersByTimeAsync(300)

      await slider(wrapper, 'neutral-button').trigger('click')
      await flushPromises()

      expect(lastValue('left')).toBe(0)
      expect(lastValue('right')).toBe(0)
      expect(sliderValue(wrapper, 'left-lever-slider')).toBe(0)
      const after = calls.length
      await vi.advanceTimersByTimeAsync(3000)
      expect(calls.length).toBe(after)
    })
  })

  describe('E-stop and clear replies', () => {
    it('says so when the E-stop latched but an output was not confirmed', async () => {
      const wrapper = await mountView()
      const reply: TractorSafetyResponse = {
        status: 'emergency_stop',
        complete: false,
        faults: ['left_lever: PCA9685 write timed out'],
      }
      vi.mocked(api.tractorEmergencyStop).mockResolvedValue(reply)

      await button(wrapper, 'EMERGENCY STOP').trigger('click')
      await flushPromises()

      expect(toastMessages().some((m) => m.includes('NOT every output was confirmed') && m.includes('PCA9685'))).toBe(true)
    })

    it('says so when a clear leaves a latch behind, and re-reads the state', async () => {
      const wrapper = await mountView(baseState({ emergency_stop_active: true }))
      vi.mocked(api.tractorClearEmergency).mockResolvedValue({
        status: 'partial',
        still_latched: ['tractor_interlock:tilt'],
      })
      vi.mocked(api.getTractorState).mockClear()

      await button(wrapper, 'Clear E-stop').trigger('click')
      await flushPromises()

      expect(toastMessages().some((m) => m.includes('not fully applied') && m.includes('tractor_interlock:tilt'))).toBe(true)
      expect(api.getTractorState).toHaveBeenCalledTimes(1)
    })
  })
})
