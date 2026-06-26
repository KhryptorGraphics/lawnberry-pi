import { setActivePinia, createPinia } from 'pinia'
import { beforeEach, describe, expect, it, vi } from 'vitest'

vi.mock('@/services/api', () => ({
  startAutonomous: vi.fn(async (zones?: string[]) => ({
    status: 'started',
    mode: 'autonomous',
    mission_id: 'm-1',
    waypoint_count: zones ? zones.length * 10 : 5,
  })),
  stopAutonomous: vi.fn(async () => ({ status: 'stopped', mode: 'idle', mission_id: 'm-1' })),
  pauseAutonomous: vi.fn(async () => ({ status: 'paused', mode: 'paused', mission_id: 'm-1' })),
  resumeAutonomous: vi.fn(async () => ({ status: 'running', mode: 'autonomous', mission_id: 'm-1' })),
  getNavigationStatus: vi.fn(async () => ({
    mode: 'autonomous',
    active: true,
    mission_id: 'm-1',
    mission_status: 'running',
    completion_percentage: 42,
  })),
}))

import { useAutonomyStore } from '@/stores/autonomy'

describe('autonomy store', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
  })

  it('starts an autonomous run and tracks mission state', async () => {
    const store = useAutonomyStore()
    await store.start(['yard'])
    expect(store.mode).toBe('autonomous')
    expect(store.active).toBe(true)
    expect(store.missionId).toBe('m-1')
    store.stopPolling()
  })

  it('refreshes status from the backend', async () => {
    const store = useAutonomyStore()
    await store.refreshStatus()
    expect(store.completion).toBe(42)
    expect(store.active).toBe(true)
  })

  it('stops and returns to idle', async () => {
    const store = useAutonomyStore()
    await store.start()
    await store.stop()
    expect(store.mode).toBe('idle')
    expect(store.active).toBe(false)
    expect(store.missionId).toBeNull()
  })

  it('pause/resume update mode', async () => {
    const store = useAutonomyStore()
    await store.start()
    await store.pause()
    expect(store.mode).toBe('autonomous') // refreshStatus mock reports running
    await store.resume()
    expect(store.mode).toBe('autonomous')
    store.stopPolling()
  })
})
