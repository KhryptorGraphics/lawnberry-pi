import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest'
import { setActivePinia, createPinia } from 'pinia'
import { useSystemStore } from '@/stores/system'
import * as api from '@/services/api'

function getSystemWsEntry() {
  const instances = (globalThis as any).__wsMockInstances as Array<{
    type: string
    instance: any
  }>
  return instances[instances.length - 1]
}

describe('System Store', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
    vi.clearAllMocks()
  })

  afterEach(() => {
    const instances = (globalThis as any).__wsMockInstances as Array<any>
    instances.length = 0
  })

  it('initializes with unknown status and disconnected connection', () => {
    const store = useSystemStore()

    expect(store.status).toBe('unknown')
    expect(store.connectionStatus).toBe('disconnected')
    expect(store.isOnline).toBe(false)
    expect(store.isSystemHealthy).toBe(false)
  })

  describe('initialize', () => {
    it('connects the websocket and reflects the connected state', async () => {
      const store = useSystemStore()
      const wsEntry = getSystemWsEntry()
      wsEntry.instance.connected.value = true

      await store.initialize()

      expect(wsEntry.instance.connect).toHaveBeenCalled()
      expect(store.connectionStatus).toBe('connected')
      expect(store.isOnline).toBe(true)
      expect(store.isLoading).toBe(false)
      expect(store.error).toBeNull()
    })

    it('marks the connection errored when connect() rejects', async () => {
      const store = useSystemStore()
      const wsEntry = getSystemWsEntry()
      wsEntry.instance.connect.mockRejectedValueOnce(new Error('socket unavailable'))

      await store.initialize()

      expect(store.connectionStatus).toBe('error')
      expect(store.error).toBe('socket unavailable')
      expect(store.isLoading).toBe(false)
    })
  })

  describe('updateSystemStatus', () => {
    it('applies a system_status message', () => {
      const store = useSystemStore()

      store.updateSystemStatus({ type: 'system_status', status: 'active' })

      expect(store.status).toBe('active')
    })

    it('derives status from telemetry health flags', () => {
      const store = useSystemStore()

      store.updateSystemStatus({ type: 'telemetry', sensors: { health: true } })
      expect(store.status).toBe('active')
      expect(store.telemetryData).toMatchObject({ type: 'telemetry' })

      store.updateSystemStatus({ type: 'telemetry', sensors: { health: false }, motors: { health: false }, navigation: { health: false } })
      expect(store.status).toBe('warning')
    })
  })

  describe('fetchPlatform', () => {
    it('initializes with platform unknown', () => {
      const store = useSystemStore()
      expect(store.platform).toBe('unknown')
    })

    it('resolves to tractor when the backend reports enabled: true', async () => {
      const store = useSystemStore()
      vi.mocked(api.getTractorState).mockResolvedValue({ enabled: true } as any)

      await store.fetchPlatform()

      expect(store.platform).toBe('tractor')
    })

    it('resolves to mower when the backend reports enabled: false', async () => {
      const store = useSystemStore()
      vi.mocked(api.getTractorState).mockResolvedValue({ enabled: false } as any)

      await store.fetchPlatform()

      expect(store.platform).toBe('mower')
    })

    it('fails open to unknown (never mower) when the fetch rejects', async () => {
      const store = useSystemStore()
      store.platform = 'tractor' // simulate a previously-resolved value
      vi.mocked(api.getTractorState).mockRejectedValue(new Error('network down'))

      await store.fetchPlatform()

      expect(store.platform).toBe('unknown')
    })
  })

  describe('shutdown', () => {
    it('disconnects the websocket and resets state', async () => {
      const store = useSystemStore()
      const wsEntry = getSystemWsEntry()
      wsEntry.instance.connected.value = true
      await store.initialize()

      await store.shutdown()

      expect(wsEntry.instance.disconnect).toHaveBeenCalled()
      expect(store.connectionStatus).toBe('disconnected')
      expect(store.status).toBe('unknown')
    })
  })
})
