// Pins the tractor store's fail-closed WS conflict resolution: an in-flight,
// stale WS tick must never silently undo a just-issued local emergency-stop.
// Two isolated guards are involved and each gets its own test so a broken
// suppress-window can't hide behind a passing ordering-guard test (or vice
// versa) — see stores/tractor.ts's applyIncomingTick()/emergencyStop().
import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest'
import { setActivePinia, createPinia } from 'pinia'
import { useTractorStore } from '@/stores/tractor'
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

function createStoreWithWs() {
  const store = useTractorStore()
  const wsEntry = getTractorWsEntry()
  wsEntry.instance.subscribe.mockClear()
  wsEntry.topicCallbacks.clear()
  store.initWebSocket()
  return { store, wsEntry }
}

function emitTractorTick(wsEntry: ReturnType<typeof getTractorWsEntry>, tractor: any, source = 'hardware') {
  const callbacks = wsEntry.topicCallbacks.get('telemetry.tractor') as
    | Array<{ callback: (data: any) => void }>
    | undefined
  callbacks?.forEach((entry) => entry.callback({ tractor, source }))
}

function makeState(overrides: Record<string, any> = {}) {
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

describe('Tractor Store', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
    vi.clearAllMocks()
  })

  afterEach(() => {
    const instances = (globalThis as any).__wsMockInstances as Array<any>
    instances.length = 0
  })

  describe('fetchState', () => {
    it('loads state via REST', async () => {
      const { store } = createStoreWithWs()
      const s = makeState({ steering: 0.5 })
      vi.mocked(api.getTractorState).mockResolvedValue(s)

      await store.fetchState()

      expect(store.state).toEqual(s)
    })
  })

  describe('actuator actions', () => {
    it('merges every field an ok response echoes back', async () => {
      const { store } = createStoreWithWs()
      store.state = makeState() as any
      vi.mocked(api.tractorGear).mockResolvedValue({
        status: 'ok',
        gear: 'reverse',
        blade_engaged: false,
      } as any)

      await store.setGear('reverse' as any)

      expect(store.state?.gear).toBe('reverse')
      expect(store.state?.blade_engaged).toBe(false)
    })

    it('does not mutate state on a rejected response', async () => {
      const { store } = createStoreWithWs()
      store.state = makeState({ gear: 'neutral' }) as any
      vi.mocked(api.tractorGear).mockResolvedValue({
        status: 'rejected',
        reason: 'emergency stop active',
      } as any)

      await store.setGear('forward' as any)

      expect(store.state?.gear).toBe('neutral')
    })
  })

  describe('fail-closed WS conflict resolution', () => {
    it('drops an out-of-order WS tick older than the last applied tick (ordering guard, isolated — no local command involved)', () => {
      const { store, wsEntry } = createStoreWithWs()
      const newer = makeState({
        steering: 0.9,
        last_updated: new Date(2026, 0, 1, 12, 0, 5).toISOString(),
      })
      const older = makeState({
        steering: 0.1,
        last_updated: new Date(2026, 0, 1, 12, 0, 1).toISOString(),
      })

      emitTractorTick(wsEntry, newer)
      expect(store.state?.steering).toBe(0.9)

      emitTractorTick(wsEntry, older)

      expect(store.state?.steering).toBe(0.9) // stale tick dropped, newer state kept
    })

    it('keeps a local emergency-stop applied against an in-flight tick that lands before the reconciling fetch resolves (suppress-window guard, isolated from timestamp ordering)', async () => {
      const { store, wsEntry } = createStoreWithWs()
      const baseline = makeState({
        emergency_stop_active: false,
        authorized: true,
        last_updated: new Date(2026, 0, 1, 12, 0, 0).toISOString(),
      })
      vi.mocked(api.getTractorState).mockResolvedValueOnce(baseline)
      await store.fetchState()

      vi.mocked(api.tractorEmergencyStop).mockResolvedValue({
        status: 'emergency_stop',
        engine: 'running',
      } as any)
      // Delay the reconciling fetch so a WS tick can land in the gap between the
      // instant local write and the fetch that would otherwise re-anchor the
      // ordering guard.
      let resolveReconcile!: (value: any) => void
      vi.mocked(api.getTractorState).mockReturnValueOnce(
        new Promise((resolve) => {
          resolveReconcile = resolve
        })
      )

      const emergencyPromise = store.emergencyStop()
      expect(store.state?.emergency_stop_active).toBe(true) // instant local write

      // This tick's timestamp (12:00:01) is NEWER than the pre-action baseline
      // (12:00:00) the plain ordering guard is still anchored to — so the ordering
      // guard alone would accept it. Its content is stale (pre-e-stop) regardless.
      // Only the suppress window, not the timestamp comparison, may reject it.
      emitTractorTick(
        wsEntry,
        makeState({
          emergency_stop_active: false,
          authorized: true,
          last_updated: new Date(2026, 0, 1, 12, 0, 1).toISOString(),
        })
      )
      expect(store.state?.emergency_stop_active).toBe(true)

      resolveReconcile(
        makeState({
          emergency_stop_active: true,
          authorized: false,
          last_updated: new Date(2026, 0, 1, 12, 0, 2).toISOString(),
        })
      )
      await emergencyPromise

      expect(store.state?.emergency_stop_active).toBe(true)
    })

    it('drops an incoming tick with a missing or unparseable last_updated', () => {
      const { store, wsEntry } = createStoreWithWs()
      store.state = makeState({ steering: 0.4 }) as any

      emitTractorTick(wsEntry, { ...makeState({ steering: 0.7 }), last_updated: 'not-a-date' })

      expect(store.state?.steering).toBe(0.4)
    })
  })

  describe('cleanup', () => {
    it('unsubscribes from WebSocket on cleanup', () => {
      const { store, wsEntry } = createStoreWithWs()
      const callbacks = wsEntry.topicCallbacks.get('telemetry.tractor') as any[] | undefined
      expect(callbacks?.length ?? 0).toBeGreaterThan(0)
      const unsubscribeSpy = callbacks![0]?.unsubscribe

      store.cleanup()

      expect(unsubscribeSpy).toHaveBeenCalled()
    })
  })
})
