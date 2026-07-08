import { defineStore } from 'pinia'
import { ref } from 'vue'
import {
  getTractorState,
  tractorAuthorize,
  tractorBlade,
  tractorClearEmergency,
  tractorClutch,
  tractorEmergencyStop,
  tractorGear,
  tractorSpeed,
  tractorStart,
  tractorStopEngine,
  tractorSteering,
  tractorThrottle,
} from '../services/api'
import { useWebSocket } from '../services/websocket'
import type { TractorActuatorResponse, Transmission, TractorState } from '../types/control'

// How long to distrust incoming telemetry.tractor WS ticks after a local
// emergency-stop/revoke: long enough to bridge an in-flight, pre-action tick
// that hasn't yet been superseded by the server's own next tick, short enough
// that a real subsequent tick (telemetry runs at several Hz elsewhere in this
// app) takes back over quickly.
// ponytail: fixed window, not adaptive to observed WS latency — revisit if
// field testing on the real tractor shows it's too short or too long.
const EMERGENCY_SUPPRESS_WINDOW_MS = 2000

export const useTractorStore = defineStore('tractor', () => {
  const state = ref<TractorState | null>(null)
  const isLoading = ref(false)
  const error = ref<string | null>(null)

  // Ordering guard for ordinary WS ticks: only ever compares two server-issued
  // last_updated timestamps against each other, never against the client clock —
  // mixing clocks would let a stale tick with a server timestamp numerically
  // ahead of a skewed client clock slip through.
  let lastAppliedServerTs = 0

  // Second line of defense, and the primary guard for the local-optimistic
  // emergency write specifically: POST /tractor/emergency-stop and
  // /tractor/revoke's responses don't carry last_updated, so there is no server
  // timestamp to anchor a comparison to for that write. See tractorStore.spec.ts.
  let emergencySuppressUntil = 0

  function parseTimestamp(value: unknown): number {
    if (typeof value !== 'string') return NaN
    return Date.parse(value)
  }

  /** Apply an incoming telemetry.tractor WS tick, fail-closed. */
  function applyIncomingTick(tick: TractorState | null | undefined) {
    if (!tick) return
    const tickTs = parseTimestamp(tick.last_updated)
    if (Number.isNaN(tickTs)) return // can't verify ordering: drop it
    if (tickTs <= lastAppliedServerTs) return // stale/out-of-order: drop it
    if (Date.now() < emergencySuppressUntil) return // just issued a local e-stop/revoke: distrust all incoming ticks
    state.value = tick
    lastAppliedServerTs = tickTs
  }

  async function fetchState() {
    isLoading.value = true
    error.value = null
    try {
      const result = await getTractorState()
      // A direct REST fetch is ground truth, not a "tick" — it always wins,
      // suppress window or not, and re-anchors the ordering guard.
      state.value = result
      const ts = parseTimestamp(result.last_updated)
      if (!Number.isNaN(ts)) lastAppliedServerTs = ts
      return result
    } catch (e: any) {
      error.value = e?.message || 'Failed to load tractor state'
      throw e
    } finally {
      isLoading.value = false
    }
  }

  /** Merge every field an 'ok' actuator response echoes back (e.g. gear changes
   * also echo blade_engaged when ROS auto-disengages it). Never mutates on a
   * 'rejected' response. Loosely typed at this boundary by design — see
   * TractorActuatorResponse. */
  function mergeOkFields(result: TractorActuatorResponse): TractorActuatorResponse {
    if (result.status === 'ok' && state.value) {
      const fields: Record<string, unknown> = { ...result }
      delete fields.status
      Object.assign(state.value, fields as Partial<TractorState>)
    }
    return result
  }

  async function setSteering(value: number) {
    return mergeOkFields(await tractorSteering(value))
  }
  async function setThrottle(value: number) {
    return mergeOkFields(await tractorThrottle(value))
  }
  async function setGroundSpeed(value: number) {
    return mergeOkFields(await tractorSpeed(value))
  }
  async function setClutch(value: number) {
    return mergeOkFields(await tractorClutch(value))
  }
  async function setGear(gear: Transmission) {
    return mergeOkFields(await tractorGear(gear))
  }
  async function setBlade(engaged: boolean) {
    return mergeOkFields(await tractorBlade(engaged))
  }
  async function startEngine() {
    return mergeOkFields(await tractorStart())
  }
  async function stopEngine() {
    return mergeOkFields(await tractorStopEngine())
  }

  async function authorize() {
    const result = await tractorAuthorize(true)
    if (result.status === 'authorized' && state.value) {
      state.value.authorized = true
    }
    return result
  }

  async function revoke() {
    // Deauthorizing is the safe direction: show it immediately, and don't trust
    // an incoming tick that would silently re-authorize for a short window.
    if (state.value) state.value.authorized = false
    emergencySuppressUntil = Date.now() + EMERGENCY_SUPPRESS_WINDOW_MS
    const result = await tractorAuthorize(false)
    await fetchState().catch(() => undefined)
    return result
  }

  async function emergencyStop() {
    // Instant local write in the safe direction (stopped/unauthorized), then
    // reconcile the rest of the fields (blade/gear/clutch/throttle/speed) from
    // a REST fetch once the command lands — see fetchState()'s ground-truth note.
    if (state.value) {
      state.value.emergency_stop_active = true
      state.value.authorized = false
    }
    emergencySuppressUntil = Date.now() + EMERGENCY_SUPPRESS_WINDOW_MS
    const result = await tractorEmergencyStop()
    await fetchState().catch(() => undefined)
    return result
  }

  async function clearEmergency() {
    // Unwinding direction: wait for real confirmation before showing "cleared".
    const result = await tractorClearEmergency()
    if (result.status === 'ok') {
      await fetchState().catch(() => undefined)
    }
    return result
  }

  // WebSocket wiring — eager, mirroring useControlStore's pattern so
  // `const tractor = useTractorStore()` gives live updates for free.
  const ws = useWebSocket('telemetry')
  let unsubscribeFn: (() => void) | null = null

  function initWebSocket() {
    cleanup()
    unsubscribeFn = ws.subscribe('telemetry.tractor', (msg) => {
      applyIncomingTick(msg?.tractor)
    })
    ws.connect?.()
  }

  function cleanup() {
    if (unsubscribeFn) {
      unsubscribeFn()
      unsubscribeFn = null
    }
  }

  initWebSocket()

  return {
    state,
    isLoading,
    error,
    fetchState,
    setSteering,
    setThrottle,
    setGroundSpeed,
    setClutch,
    setGear,
    setBlade,
    startEngine,
    stopEngine,
    authorize,
    revoke,
    emergencyStop,
    clearEmergency,
    initWebSocket,
    cleanup,
  }
})
