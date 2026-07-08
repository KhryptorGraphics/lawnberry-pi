import { beforeEach, vi } from 'vitest'

// The real plugin is a bare-global IIFE that assumes `window === globalThis`
// (true in a real browser, false under Vitest's jsdom environment, where
// arbitrary properties assigned onto `window` are not mirrored onto the
// bare global scope). It has no unit-testable behavior of its own here -
// components only call `L.gridLayer.googleMutant(...)` behind an online-only,
// real-Google-Maps-API code path - so replace it with a no-op in tests.
vi.mock('leaflet.gridlayer.googlemutant/dist/Leaflet.GoogleMutant.js', () => ({}))

class LocalStorageMock {
  private store = new Map<string, string>()

  clear() {
    this.store.clear()
  }

  getItem(key: string) {
    return this.store.has(key) ? this.store.get(key)! : null
  }

  setItem(key: string, value: string) {
    this.store.set(key, String(value))
  }

  removeItem(key: string) {
    this.store.delete(key)
  }
}

if (typeof globalThis.localStorage === 'undefined') {
  Object.defineProperty(globalThis, 'localStorage', {
    value: new LocalStorageMock(),
    configurable: true,
    enumerable: true,
    writable: true,
  })
}

const apiClient = {
  get: vi.fn(),
  post: vi.fn(),
  put: vi.fn(),
  delete: vi.fn(),
  patch: vi.fn(),
}

const sendControlCommand = vi.fn()
const getRoboHATStatus = vi.fn()
const getMapConfiguration = vi.fn()
const saveMapConfiguration = vi.fn()

// Ride-on tractor actuation — see frontend/src/stores/tractor.ts and
// frontend/src/views/TractorControlView.vue. Any test mounting either without
// these would crash on an undefined import.
const getTractorState = vi.fn()
const tractorSteering = vi.fn()
const tractorThrottle = vi.fn()
const tractorSpeed = vi.fn()
const tractorClutch = vi.fn()
const tractorGear = vi.fn()
const tractorBlade = vi.fn()
const tractorStart = vi.fn()
const tractorStopEngine = vi.fn()
const tractorEmergencyStop = vi.fn()
const tractorClearEmergency = vi.fn()
const tractorAuthorize = vi.fn()

vi.mock('@/services/api', () => ({
  __esModule: true,
  default: apiClient,
  useApiService: () => apiClient,
  sendControlCommand,
  getRoboHATStatus,
  getMapConfiguration,
  saveMapConfiguration,
  getTractorState,
  tractorSteering,
  tractorThrottle,
  tractorSpeed,
  tractorClutch,
  tractorGear,
  tractorBlade,
  tractorStart,
  tractorStopEngine,
  tractorEmergencyStop,
  tractorClearEmergency,
  tractorAuthorize,
}))

type TopicCallbackEntry = {
  callback: (data: any) => void
  unsubscribe: ReturnType<typeof vi.fn>
}

const wsInstances: Array<{
  type: string
  handlers?: { onMessage?: (msg: any) => void }
  instance: any
  topicCallbacks: Map<string, TopicCallbackEntry[]>
}> = []

const useWebSocketMock = vi.fn((type: 'telemetry' | 'control' = 'telemetry', handlers?: { onMessage?: (msg: any) => void }) => {
  const topicCallbacks = new Map<string, TopicCallbackEntry[]>()

  const instance = {
    connected: { value: false },
    connecting: { value: false },
    connect: vi.fn().mockResolvedValue(undefined),
    disconnect: vi.fn(),
    subscribe: vi.fn((topic: string, callback: (data: any) => void) => {
      const existing = topicCallbacks.get(topic) ?? []
      const unsubscribe = vi.fn(() => {
        const arr = topicCallbacks.get(topic)
        if (!arr) return
        const idx = arr.findIndex(entry => entry.callback === callback)
        if (idx >= 0) {
          arr.splice(idx, 1)
        }
        if (arr.length === 0) {
          topicCallbacks.delete(topic)
        }
      })

      existing.push({ callback, unsubscribe })
      topicCallbacks.set(topic, existing)
      return unsubscribe
    }),
    unsubscribe: vi.fn((topic: string, callback?: (data: any) => void) => {
      if (!callback) {
        topicCallbacks.delete(topic)
        return
      }
      const arr = topicCallbacks.get(topic)
      if (!arr) {
        return
      }
      const entry = arr.find(item => item.callback === callback)
      if (entry) {
        entry.unsubscribe()
      }
      if (topicCallbacks.get(topic)?.length === 0) {
        topicCallbacks.delete(topic)
      }
    }),
    setCadence: vi.fn(),
    ping: vi.fn(),
    listTopics: vi.fn(),
    dispatchTestMessage: vi.fn((message: any) => {
      handlers?.onMessage?.(message)
      if (message?.topic) {
        const callbacks = topicCallbacks.get(message.topic)
        callbacks?.forEach(entry => entry.callback(message.data ?? message))
      }
    }),
    __emit(topic: string, payload: any) {
      const callbacks = topicCallbacks.get(topic)
      callbacks?.forEach(entry => entry.callback(payload))
    },
  }

  wsInstances.push({ type, handlers, instance, topicCallbacks })
  return instance
})

vi.mock('@/services/websocket', () => ({
  __esModule: true,
  useWebSocket: useWebSocketMock,
}))

Object.defineProperty(globalThis, '__wsMockInstances', {
  value: wsInstances,
  configurable: true,
  enumerable: false,
  writable: false,
})

beforeEach(() => {
  apiClient.get.mockReset()
  apiClient.post.mockReset()
  apiClient.put.mockReset()
  apiClient.delete.mockReset()
  apiClient.patch.mockReset()
  sendControlCommand.mockReset()
  getRoboHATStatus.mockReset()
  getMapConfiguration.mockReset()
  saveMapConfiguration.mockReset()
  getTractorState.mockReset()
  tractorSteering.mockReset()
  tractorThrottle.mockReset()
  tractorSpeed.mockReset()
  tractorClutch.mockReset()
  tractorGear.mockReset()
  tractorBlade.mockReset()
  tractorStart.mockReset()
  tractorStopEngine.mockReset()
  tractorEmergencyStop.mockReset()
  tractorClearEmergency.mockReset()
  tractorAuthorize.mockReset()
  useWebSocketMock.mockClear()
  localStorage.clear()
})
