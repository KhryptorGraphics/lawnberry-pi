// The ApiService answers a 401 by refreshing the token once and replaying the request.
// That is right for an expired token and wrong for the two calls that carry a credential
// the operator just typed (POST /auth/login, POST /control/manual-unlock): their 401 means
// "wrong password", and replaying would send the same wrong password again and count it
// twice against the backend's per-peer lockout (429 after repeated wrong passwords).
//
// Drives the REAL ApiService (interceptors included) over a recording network adapter.
import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest'
import { setActivePinia, createPinia } from 'pinia'
import { createRecordingAdapter, replaceNetworkAdapter } from './helpers/recordingAdapter'
import type { Responder } from './helpers/recordingAdapter'

// Escape the global @/services/api stub (tests/vitest.setup.ts).
vi.mock('@/services/api', async (importOriginal) => importOriginal())

import { useApiService } from '@/services/api'
import { useAuthStore } from '@/stores/auth'

const REFRESHED = { access_token: 'tok-new', token: 'tok-new', expires_in: 3600 }

let restoreAdapter: (() => void) | null = null

function install(responder: Responder) {
  const { adapter, requests } = createRecordingAdapter(responder)
  restoreAdapter = replaceNetworkAdapter(useApiService(), adapter)
  return requests
}

function wire(requests: Array<{ method: string; url: string; headers: Record<string, string> }>) {
  return requests.map((r) => `${r.method} ${r.url} ${r.headers.authorization ?? '(no bearer)'}`)
}

describe('401 handling in the shared ApiService', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
    // refreshToken() arms a refresh timer; keep it from outliving the test.
    vi.useFakeTimers()
    useAuthStore().token = 'tok-123'
  })

  afterEach(() => {
    restoreAdapter?.()
    restoreAdapter = null
    vi.clearAllTimers()
    vi.useRealTimers()
    vi.restoreAllMocks()
  })

  it.each([
    ['manual-unlock', '/api/v2/control/manual-unlock', { method: 'password', password: 'wrong' }],
    ['v2 login', '/api/v2/auth/login', { username: 'admin', password: 'wrong' }],
    ['v1 login', '/api/v1/auth/login', { username: 'admin', password: 'wrong' }],
    ['manual-unlock with a query string', '/api/v2/control/manual-unlock?probe=1', { method: 'password' }],
  ])('a 401 from %s is a credential failure: no token refresh, no replay', async (_name, url, body) => {
    const requests = install((req) =>
      req.url.endsWith('/auth/refresh') ? { data: REFRESHED } : { status: 401, data: { detail: 'Invalid password' } }
    )

    await expect(useApiService().post(url, body)).rejects.toMatchObject({
      response: { status: 401, data: { detail: 'Invalid password' } },
    })

    // Exactly one request on the wire: not even a /auth/refresh, let alone a replay.
    expect(wire(requests)).toEqual([`POST ${url} Bearer tok-123`])
    expect(useAuthStore().token).toBe('tok-123')
  })

  it('a 401 from any other route still refreshes the token once and replays with the new Bearer', async () => {
    const requests = install((req) => {
      if (req.url.endsWith('/auth/refresh')) return { data: REFRESHED }
      return req.headers.authorization === 'Bearer tok-new'
        ? { data: { id: 'm-1' } }
        : { status: 401, data: { detail: 'Operator authentication required' } }
    })

    const response = await useApiService().post('/api/v2/missions/create', { name: 'north lawn' })

    expect(response.data).toEqual({ id: 'm-1' })
    expect(wire(requests)).toEqual([
      'POST /api/v2/missions/create Bearer tok-123',
      'POST /api/v2/auth/refresh Bearer tok-123',
      'POST /api/v2/missions/create Bearer tok-new',
    ])
    expect(useAuthStore().token).toBe('tok-new')
  })

  it('does not treat /control/manual-unlock/status as a credential check (suffix match, not substring)', async () => {
    const requests = install((req) => {
      if (req.url.endsWith('/auth/refresh')) return { data: REFRESHED }
      return req.headers.authorization === 'Bearer tok-new'
        ? { data: { authorized: false } }
        : { status: 401, data: { detail: 'Operator authentication required' } }
    })

    await useApiService().get('/api/v2/control/manual-unlock/status')

    expect(wire(requests)).toEqual([
      'GET /api/v2/control/manual-unlock/status Bearer tok-123',
      'POST /api/v2/auth/refresh Bearer tok-123',
      'GET /api/v2/control/manual-unlock/status Bearer tok-new',
    ])
  })

  it('replays only once: a 401 that survives the refresh is surfaced', async () => {
    const requests = install((req) =>
      req.url.endsWith('/auth/refresh') ? { data: REFRESHED } : { status: 401, data: { detail: 'Operator authentication required' } }
    )

    await expect(useApiService().post('/api/v2/ai/enable', { mode: 'ai_autonomous' })).rejects.toMatchObject({
      response: { status: 401 },
    })

    expect(wire(requests)).toEqual([
      'POST /api/v2/ai/enable Bearer tok-123',
      'POST /api/v2/auth/refresh Bearer tok-123',
      'POST /api/v2/ai/enable Bearer tok-new',
    ])
  })

  // With a Bearer the backend no longer knows (restart, expired session), /auth/refresh AND /auth/logout both
  // answer 401. Refreshing on the logout's 401 re-entered authStore.refreshToken() -> logout() -> POST /auth/logout
  // -> ... without end: the call that started it never settled and the backend was hammered.
  it('a dead Bearer ends in a bounded request sequence, not a refresh/logout storm', async () => {
    vi.spyOn(console, 'warn').mockImplementation(() => {}) // refreshToken() and logout() warn about their failures
    let sent = 0
    const requests = install(() => {
      // Safety net: a regression fails this test instead of hanging the run.
      sent += 1
      if (sent > 40) throw new Error('request storm: more than 40 requests for one call')
      return { status: 401, data: { detail: 'Unauthorized' } }
    })

    await expect(useApiService().post('/api/v2/missions/create', { name: 'x' })).rejects.toMatchObject({
      response: { status: 401 },
    })

    // gated call -> refresh (401) -> logout (401: the Bearer is already dead, nothing to refresh) -> done
    expect(wire(requests)).toEqual([
      'POST /api/v2/missions/create Bearer tok-123',
      'POST /api/v2/auth/refresh Bearer tok-123',
      'POST /api/v2/auth/logout Bearer tok-123',
    ])
    expect(useAuthStore().token).toBeNull() // logout() dropped the dead token
  })

  it('signed out: a 401 is surfaced at once (nothing to refresh, no Authorization header)', async () => {
    useAuthStore().token = null
    const requests = install(() => ({ status: 401, data: { detail: 'Operator authentication required' } }))

    await expect(useApiService().post('/api/v2/missions/create', { name: 'x' })).rejects.toMatchObject({
      response: { status: 401 },
    })

    expect(wire(requests)).toEqual(['POST /api/v2/missions/create (no bearer)'])
  })
})
