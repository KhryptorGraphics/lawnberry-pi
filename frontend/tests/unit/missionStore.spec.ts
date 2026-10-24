// POST /missions/create and /missions/{id}/start|pause|resume|abort now need the operator Bearer and answer 401
// until someone has signed in. The store used to swallow every failure (createMission returned void, the others
// only console.error'd); it now reports why through `lastError`.
import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest'
import type { Mock, MockInstance } from 'vitest'
import { setActivePinia, createPinia } from 'pinia'
import apiService from '@/services/api'
import { useMissionStore } from '@/stores/mission'
import { SIGN_IN_REQUIRED_MESSAGE } from '@/utils/apiErrors'

// tests/vitest.setup.ts replaces the default export with this mock
const api = apiService as unknown as { get: Mock; post: Mock }

const unsigned = { response: { status: 401, data: { detail: 'Operator authentication required' } } }
const mission = { id: 'm-1', name: 'north lawn', waypoints: [], created_at: '2026-01-01T00:00:00Z' }

let consoleError: MockInstance

beforeEach(() => {
  setActivePinia(createPinia())
  // start/resume arm a 2 s status poll: fake timers let the tests count and discard it
  vi.useFakeTimers()
  consoleError = vi.spyOn(console, 'error').mockImplementation(() => {})
  api.post.mockResolvedValue({ data: {} })
  api.get.mockResolvedValue({ data: { mission_id: 'm-1', status: 'running', completion_percentage: 10 } })
})

afterEach(() => {
  consoleError.mockRestore()
  vi.clearAllTimers()
  vi.useRealTimers()
})

function storeWithMission(status: string) {
  const store = useMissionStore()
  store.currentMission = { ...mission }
  store.missionStatus = status
  return store
}

describe('createMission', () => {
  it('posts the name and waypoints, loads the mission, returns true and leaves no error', async () => {
    const store = useMissionStore()
    store.addWaypoint(51.5, -0.09)
    api.post.mockResolvedValueOnce({ data: mission })

    await expect(store.createMission('north lawn')).resolves.toBe(true)

    expect(api.post).toHaveBeenCalledWith('/api/v2/missions/create', {
      name: 'north lawn',
      waypoints: store.waypoints,
    })
    expect(store.currentMission).toEqual(mission)
    expect(store.missionStatus).toBe('idle')
    expect(store.lastError).toBe('')
  })

  it('a 401 returns false, keeps the sign-in text in lastError and leaves the store untouched', async () => {
    const store = useMissionStore()
    store.addWaypoint(51.5, -0.09)
    api.post.mockRejectedValueOnce(unsigned)

    await expect(store.createMission('north lawn')).resolves.toBe(false)

    expect(store.lastError).toBe(SIGN_IN_REQUIRED_MESSAGE)
    expect(store.currentMission).toBeNull()
    expect(store.missionStatus).toBe('')
    expect(store.waypoints).toHaveLength(1)
    expect(consoleError).toHaveBeenCalledWith('Error creating mission:', unsigned)
  })

  it('a later success clears lastError', async () => {
    const store = useMissionStore()
    api.post.mockRejectedValueOnce(unsigned)
    await store.createMission('north lawn')
    expect(store.lastError).not.toBe('')

    api.post.mockResolvedValueOnce({ data: mission })
    await expect(store.createMission('north lawn')).resolves.toBe(true)

    expect(store.lastError).toBe('')
  })

  it.each([
    ['a rate limit with Retry-After', { response: { status: 429, headers: { 'retry-after': '12' } } }, 'Too many attempts: try again in 12 s.'],
    ['a backend detail', { response: { status: 422, data: { detail: 'Mission needs at least one waypoint' } } }, 'Mission needs at least one waypoint'],
    ['no HTTP response', new Error('Network Error'), 'Failed to create mission: no response from the backend.'],
    ['a bare HTTP error', { response: { status: 500 } }, 'Failed to create mission'],
  ])('%s reads as an operator-facing message', async (_name, error, text) => {
    const store = useMissionStore()
    api.post.mockRejectedValueOnce(error)

    await expect(store.createMission('x')).resolves.toBe(false)

    expect(store.lastError).toBe(text)
  })
})

// [action, url suffix, status before, status after, fallback when the backend says nothing]
const ACTIONS = [
  ['startCurrentMission', 'start', 'idle', 'running', 'Failed to start mission'],
  ['pauseCurrentMission', 'pause', 'running', 'paused', 'Failed to pause mission'],
  ['resumeCurrentMission', 'resume', 'paused', 'running', 'Failed to resume mission'],
  ['abortCurrentMission', 'abort', 'running', 'aborted', 'Failed to abort mission'],
] as const

describe.each(ACTIONS)('%s', (action, suffix, before, after, fallback) => {
  it(`posts to /missions/{id}/${suffix}, moves the status ${before} -> ${after} and returns true`, async () => {
    const store = storeWithMission(before)

    await expect(store[action]()).resolves.toBe(true)

    expect(api.post).toHaveBeenCalledWith(`/api/v2/missions/m-1/${suffix}`, {})
    expect(store.missionStatus).toBe(after)
    expect(store.lastError).toBe('')
  })

  it('a 401 returns false, reports the sign-in text and changes neither the status nor the mission', async () => {
    const store = storeWithMission(before)
    api.post.mockRejectedValueOnce(unsigned)

    await expect(store[action]()).resolves.toBe(false)

    expect(store.lastError).toBe(SIGN_IN_REQUIRED_MESSAGE)
    expect(store.missionStatus).toBe(before)
    expect(store.currentMission).toEqual(mission)
    expect(consoleError).toHaveBeenCalledTimes(1)
  })

  it('a later success clears lastError', async () => {
    const store = storeWithMission(before)
    api.post.mockRejectedValueOnce(unsigned)
    await store[action]()
    expect(store.lastError).toBe(SIGN_IN_REQUIRED_MESSAGE)

    await expect(store[action]()).resolves.toBe(true)

    expect(store.lastError).toBe('')
  })

  it.each([
    ['a backend detail', { response: { status: 409, data: { detail: 'Mission is not in a runnable state' } } }, 'Mission is not in a runnable state'],
    ['no HTTP response', new Error('timeout of 10000ms exceeded'), `${fallback}: no response from the backend.`],
    ['a bare HTTP error', { response: { status: 500 } }, fallback],
  ])('%s reads as an operator-facing message', async (_name, error, text) => {
    const store = storeWithMission(before)
    api.post.mockRejectedValueOnce(error)

    await expect(store[action]()).resolves.toBe(false)

    expect(store.lastError).toBe(text)
  })

  it('with no mission loaded it returns false without calling the backend and says why', async () => {
    const store = useMissionStore()
    store.lastError = ''

    await expect(store[action]()).resolves.toBe(false)

    expect(api.post).not.toHaveBeenCalled()
    expect(store.lastError).toBe('No mission is loaded: create a mission first.')
  })
})

describe('status polling is tied to the outcome of the call', () => {
  it('start: a 401 arms no poll, a success arms exactly one', async () => {
    const store = storeWithMission('idle')
    api.post.mockRejectedValueOnce(unsigned)
    await store.startCurrentMission()
    expect(vi.getTimerCount()).toBe(0)

    await store.startCurrentMission()

    expect(vi.getTimerCount()).toBe(1)
  })

  it('pause: a 401 leaves the poll running, a success stops it', async () => {
    const store = storeWithMission('idle')
    await store.startCurrentMission()
    expect(vi.getTimerCount()).toBe(1)
    api.post.mockRejectedValueOnce(unsigned)

    await store.pauseCurrentMission()
    expect(vi.getTimerCount()).toBe(1)
    expect(store.missionStatus).toBe('running')

    await store.pauseCurrentMission()
    expect(vi.getTimerCount()).toBe(0)
    expect(store.missionStatus).toBe('paused')
  })

  it('abort: a 401 keeps the mission, a success drops it', async () => {
    const store = storeWithMission('running')
    api.post.mockRejectedValueOnce(unsigned)

    await store.abortCurrentMission()
    expect(store.currentMission).toEqual(mission)

    await store.abortCurrentMission()
    expect(store.currentMission).toBeNull()
    expect(store.missionStatus).toBe('aborted')
  })
})
