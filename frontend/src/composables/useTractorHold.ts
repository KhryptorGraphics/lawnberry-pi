// Hold-to-run command stream for the tractor page (review finding SF-08).
//
// The backend (services/tractor_service.py, `command_deadman`) only holds a
// lever/throttle command for a short window -- 500 ms by default, echoed as
// `hold_ms` in every accepted reply -- and then returns the lever to neutral and
// the throttle to idle unless it is refreshed. A page that POSTs a slider value
// once therefore moved the machine for half a second while the slider kept
// showing the old value. This composable is the client half of that contract:
//
//  * While a control is commanded non-neutral FROM THIS PAGE it re-sends the
//    command every `refreshMs` (a third of the hold the backend echoed, else a
//    conservative 150 ms) -- never overlapping requests, so a slow link
//    stretches the period instead of piling requests up.
//  * A control is commanded only while the operator holds it (pointer or key
//    down). Releasing it returns it to neutral and sends neutral. "Latch" mode
//    keeps a slider where it was left (two sliders cannot both be dragged with
//    one mouse) but every stop condition below still applies.
//  * Everything else ends every hold and sends neutral: page hidden, page
//    unload, browser offline, unmount, a request that fails or times out (a
//    reply slower than the hold means the hold already lapsed), a rejected
//    command, or a latched E-stop (which sends nothing: the E-stop drives
//    neutral itself and the backend refuses further writes).
//  * The page never re-commands on its own after a stop: the operator has to
//    move the control again.
//
// The backend deadman stays the real backstop (a page that dies cannot send
// neutral); this only stops the page from making the machine stutter and from
// showing a value the machine is no longer holding.
import { computed, onMounted, onUnmounted, reactive, ref, watch, type Ref } from 'vue'
import { useTractorStore } from '@/stores/tractor'
import { describeApiError } from '@/utils/apiErrors'
import type { TractorActuatorResponse } from '@/types/control'

export type HoldControl = 'left' | 'right' | 'throttle'
export type HoldStopReason = 'hidden' | 'connection' | 'rejected' | 'estop' | 'unmount'

/** Lever magnitude the backend reads as neutral (LEVER_NEUTRAL_EPS, models/tractor_control.py). */
export const LEVER_NEUTRAL_EPS = 0.05
/** Refresh period until the backend has echoed its hold once: 3 refreshes per 500 ms default hold. */
export const FALLBACK_REFRESH_MS = 150
export const MIN_REFRESH_MS = 50
export const MAX_REFRESH_MS = 250
/** The refresh period is the backend's echoed hold divided by this (3 chances per hold). */
export const REFRESHES_PER_HOLD = 3
const FALLBACK_HOLD_MS = 500
const MIN_COMMAND_TIMEOUT_MS = 300
const MAX_COMMAND_TIMEOUT_MS = 1000

/** Refresh period for a backend hold of `holdMs` (null/invalid: the conservative fallback). */
export function refreshIntervalFor(holdMs: number | null): number {
  if (holdMs === null || !(holdMs > 0)) return FALLBACK_REFRESH_MS
  return Math.min(MAX_REFRESH_MS, Math.max(MIN_REFRESH_MS, Math.floor(holdMs / REFRESHES_PER_HOLD)))
}

/** Per-request timeout: a reply slower than the hold means the hold already lapsed. */
export function commandTimeoutFor(holdMs: number | null): number {
  const hold = holdMs !== null && holdMs > 0 ? holdMs : FALLBACK_HOLD_MS
  return Math.min(MAX_COMMAND_TIMEOUT_MS, Math.max(MIN_COMMAND_TIMEOUT_MS, hold))
}

type Group = 'levers' | 'throttle'
const CONTROLS: HoldControl[] = ['left', 'right', 'throttle']
const GROUPS: Group[] = ['levers', 'throttle']
const LABEL: Record<HoldControl, string> = {
  left: 'Left lever',
  right: 'Right lever',
  throttle: 'Throttle',
}

function isNeutral(control: HoldControl, value: number): boolean {
  if (!Number.isFinite(value)) return true
  // Same cut-offs as the backend: levers inside the detent, throttle at/below idle.
  return control === 'throttle' ? value <= 0 : Math.abs(value) <= LEVER_NEUTRAL_EPS
}

export interface UseTractorHoldOptions {
  /** The slider models; the composable writes 0 back when a hold ends. */
  values: Record<HoldControl, Ref<number>>
  /** Latched E-stop flag (backend state): ends every hold. */
  emergencyStopActive: Ref<boolean>
  /** Operator-facing message sink (the view wires it to the toast store). */
  notify: (message: string, good?: boolean) => void
}

export function useTractorHold(options: UseTractorHoldOptions) {
  const tractor = useTractorStore()
  const { values, notify } = options

  const latch = ref(false)
  const holdMs = ref<number | null>(null)
  const stopNotice = ref('')
  /** Controls this page is commanding right now (not merely controls whose slider is off zero). */
  const engaged = reactive<Record<HoldControl, boolean>>({ left: false, right: false, throttle: false })
  const holding = computed(() => engaged.left || engaged.right || engaged.throttle)
  const refreshMs = computed(() => refreshIntervalFor(holdMs.value))

  // Operator grips (pointer/key down on a slider). Plain objects: nothing renders them.
  const gripped: Record<HoldControl, boolean> = { left: false, right: false, throttle: false }
  const gripSource: Record<HoldControl, 'pointer' | 'key' | null> = {
    left: null,
    right: null,
    throttle: null,
  }
  const gripPointer: Record<HoldControl, number | undefined> = {
    left: undefined,
    right: undefined,
    throttle: undefined,
  }

  // `sent`: a non-neutral command from this page may be in force on the backend.
  // `pending`: a neutral still has to be delivered for that group.
  const sent: Record<Group, boolean> = { levers: false, throttle: false }
  const pending: Record<Group, boolean> = { levers: false, throttle: false }
  // The operator pressed "both levers to neutral": a refusal must be shown, not swallowed.
  const explicitNeutral: Record<Group, boolean> = { levers: false, throttle: false }

  let generation = 0 // bumped by halt(): a round in flight stops after its current request
  let inFlight: Promise<void> | null = null
  let dirty = false // a round was requested while one was running: run again right after it
  let timer: number | undefined
  let timerMs = 0

  // --- planning (always derived from the current state, never from stale events) ---

  const groupEngaged = (group: Group): boolean =>
    group === 'levers' ? engaged.left || engaged.right : engaged.throttle

  function target(control: HoldControl): number {
    const value = Number(values[control].value)
    return engaged[control] && !isNeutral(control, value) ? value : 0
  }

  /** What to write this round for a group: live (engaged, values read at send time) or a neutral. */
  function plan(group: Group): { controls: HoldControl[]; live: boolean } | null {
    const controls: HoldControl[] = group === 'levers' ? ['left', 'right'] : ['throttle']
    if (groupEngaged(group)) return { controls, live: true }
    if (pending[group]) return { controls, live: false }
    return null
  }

  // --- timer ---

  function stopTimer() {
    window.clearInterval(timer)
    timer = undefined
    timerMs = 0
  }

  function startTimer() {
    const ms = refreshMs.value
    if (timer !== undefined && timerMs === ms) return
    stopTimer()
    timerMs = ms
    timer = window.setInterval(() => void requestRound(), ms)
  }

  function learnHold(result: TractorActuatorResponse) {
    if (result.status !== 'ok') return
    const echoed = result.hold_ms
    if (typeof echoed !== 'number' || !(echoed > 0) || echoed === holdMs.value) return
    holdMs.value = echoed
    if (timer !== undefined) startTimer() // re-time to the new refresh period
  }

  // --- sending ---

  function callActuator(control: HoldControl, value: number): Promise<TractorActuatorResponse> {
    const opts = { timeout: commandTimeoutFor(holdMs.value) }
    if (control === 'left') return tractor.setLeftLever(value, opts)
    if (control === 'right') return tractor.setRightLever(value, opts)
    return tractor.setThrottle(value, opts)
  }

  /**
   * Deliver one write. `live`: its group is being commanded non-neutral, so any failure ends every
   * hold (false). A neutral delivery after a stop is best effort: the backend deadman is the
   * backstop, so it is attempted once and never retried.
   */
  async function deliver(control: HoldControl, value: number, live: boolean, report: boolean): Promise<boolean> {
    try {
      const result = await callActuator(control, value)
      if (result.status === 'ok') {
        if (live) learnHold(result)
        return true
      }
      const reason = String(result.reason || 'command refused')
      if (!live) {
        // A refused automatic neutral means nothing is being held (E-stop latched, actuators
        // down); only an explicit request is worth telling the operator about.
        if (report) notify(`${LABEL[control]}: neutral refused: ${reason}`, false)
        return true
      }
      fail('rejected', `${LABEL[control]}: ${reason}`)
      return false
    } catch (error) {
      if (!live) {
        notify(
          `${LABEL[control]}: neutral not confirmed (${describeApiError(error, 'no answer')}). ` +
            'The backend returns it to neutral when its hold runs out.',
          false
        )
        return false
      }
      fail(
        'connection',
        `${LABEL[control]}: ${describeApiError(error, 'Command failed')} ` +
          'Levers and throttle return to neutral when the backend hold runs out.'
      )
      return false
    }
  }

  async function writeGroup(group: Group, controls: HoldControl[], live: boolean, round: number) {
    const report = explicitNeutral[group]
    explicitNeutral[group] = false
    if (live) sent[group] = true // before the request: it may land even if its reply is lost
    for (const control of controls) {
      // Read at send time: the operator may have moved or released the other control meanwhile.
      const ok = await deliver(control, live ? target(control) : 0, live, report)
      // A failed or superseded live write ends the round; the halt's own neutral round follows.
      if (live && (!ok || round !== generation)) return
    }
    if (!live) {
      sent[group] = false
      pending[group] = false
    }
  }

  async function sendRound(round: number) {
    for (const group of GROUPS) {
      const step = plan(group)
      if (step) await writeGroup(group, step.controls, step.live, round)
      if (round !== generation) return
    }
  }

  function requestRound(): Promise<void> {
    if (inFlight) {
      dirty = true
      return inFlight
    }
    inFlight = (async () => {
      try {
        do {
          dirty = false
          await sendRound(generation)
        } while (dirty)
      } catch (error) {
        console.error('Tractor hold round failed:', error)
      } finally {
        inFlight = null
      }
    })()
    return inFlight
  }

  /** Resolves when every request this page has issued has settled (tests, unmount). */
  function idle(): Promise<void> {
    return inFlight ?? Promise.resolve()
  }

  // --- state transitions ---

  /** Re-derive timer/pending state after any change; `now` sends without waiting for the next tick. */
  function reconcile(now = false) {
    for (const group of GROUPS) {
      if (!groupEngaged(group) && sent[group]) pending[group] = true
    }
    if (holding.value) startTimer()
    else stopTimer()
    const fresh = GROUPS.some((group) => groupEngaged(group) && !sent[group])
    if (now || fresh || pending.levers || pending.throttle) void requestRound()
  }

  function grip(control: HoldControl, source: 'pointer' | 'key' = 'key', pointerId?: number) {
    // A key going down on a slider the pointer already holds must not turn it into a key grip
    // (the pointer going up would then no longer release it).
    if (gripped[control] && gripSource[control] === 'pointer' && source === 'key') return
    gripped[control] = true
    gripSource[control] = source
    gripPointer[control] = pointerId
  }

  /** The operator moved a slider. */
  function input(control: HoldControl) {
    const slider = values[control]
    if (options.emergencyStopActive.value || (!latch.value && !gripped[control])) {
      // Hold-to-run needs a physical grip: a value that arrives without one (assistive
      // technology, a wheel event, a script) must not start moving the machine. Use latch.
      slider.value = 0
      return
    }
    if (isNeutral(control, Number(slider.value))) {
      engaged[control] = false // back inside the detent: neutral is sent in place of the raw value
      reconcile(true)
      return
    }
    engaged[control] = true
    stopNotice.value = ''
    reconcile()
  }

  /** Pointer/key released (or the slider lost focus): hold-to-run ends, latch keeps the value. */
  function release(control: HoldControl) {
    gripped[control] = false
    gripSource[control] = null
    gripPointer[control] = undefined
    if (latch.value) return
    if (!engaged[control] && isNeutral(control, Number(values[control].value))) return
    engaged[control] = false
    values[control].value = 0
    reconcile(true)
  }

  /** End every hold from this page and return everything to neutral. */
  function halt(reason: HoldStopReason, message = '') {
    generation += 1
    const wasCommanding = holding.value || sent.levers || sent.throttle
    for (const control of CONTROLS) {
      engaged[control] = false
      gripped[control] = false
      gripSource[control] = null
      gripPointer[control] = undefined
      values[control].value = 0
    }
    stopTimer()
    if (reason === 'estop') {
      // The E-stop request drives neutral itself and the backend now refuses writes.
      for (const group of GROUPS) {
        sent[group] = false
        pending[group] = false
      }
    }
    if (wasCommanding && reason !== 'unmount') {
      stopNotice.value =
        reason === 'hidden'
          ? 'The page was hidden, so levers and throttle were sent to neutral.'
          : reason === 'estop'
            ? 'Emergency stop: every hold from this page was ended.'
            : message
    }
    reconcile(true)
  }

  function fail(reason: 'connection' | 'rejected', message: string) {
    notify(message, false)
    halt(reason, message)
  }

  /** Both levers to neutral, even if this page never moved them (the backend may be holding another source's command). */
  function neutralLevers(): Promise<void> {
    engaged.left = false
    engaged.right = false
    values.left.value = 0
    values.right.value = 0
    pending.levers = true
    explicitNeutral.levers = true
    reconcile(true)
    return idle()
  }

  // --- stop conditions outside the sliders ---

  function onVisibilityChange() {
    if (document.visibilityState === 'hidden') halt('hidden')
  }
  function onPageHide() {
    halt('hidden')
  }
  function onOffline() {
    halt(
      'connection',
      'The browser went offline. Levers and throttle return to neutral when the backend hold runs out.'
    )
  }
  function onPointerEnd(event: PointerEvent) {
    const pointerId = event.pointerId
    for (const control of CONTROLS) {
      if (!gripped[control] || gripSource[control] !== 'pointer') continue
      const ours = gripPointer[control]
      if (pointerId === undefined || ours === undefined || ours === pointerId) release(control)
    }
  }

  // A latched E-stop ends every hold. Sync: the sliders must not stay up for a render.
  watch(
    options.emergencyStopActive,
    (active) => {
      if (active) halt('estop')
    },
    { flush: 'sync' }
  )
  // Turning latch off must not leave a slider commanding: whatever nobody is holding goes neutral.
  watch(
    latch,
    (on) => {
      if (on) return
      let changed = false
      for (const control of CONTROLS) {
        if (gripped[control] || (!engaged[control] && isNeutral(control, Number(values[control].value))))
          continue
        engaged[control] = false
        values[control].value = 0
        changed = true
      }
      if (changed) reconcile(true)
    },
    { flush: 'sync' }
  )

  onMounted(() => {
    document.addEventListener('visibilitychange', onVisibilityChange)
    window.addEventListener('pagehide', onPageHide)
    window.addEventListener('offline', onOffline)
    window.addEventListener('pointerup', onPointerEnd)
    window.addEventListener('pointercancel', onPointerEnd)
  })
  onUnmounted(() => {
    document.removeEventListener('visibilitychange', onVisibilityChange)
    window.removeEventListener('pagehide', onPageHide)
    window.removeEventListener('offline', onOffline)
    window.removeEventListener('pointerup', onPointerEnd)
    window.removeEventListener('pointercancel', onPointerEnd)
    halt('unmount')
  })

  return {
    latch,
    holding,
    engaged,
    holdMs,
    refreshMs,
    stopNotice,
    grip,
    input,
    release,
    halt,
    neutralLevers,
    idle,
  }
}
