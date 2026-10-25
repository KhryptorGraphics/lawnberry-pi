// SEC20-06: the UI's logout and token refresh are POSTs with nothing to send. axios then sends NO
// Content-Type (xhr adapter: `requestData === undefined && requestHeaders.setContentType(null)`),
// and the backend's auth middleware answered 415, so "sign out" never reached the session: the
// token and every WebSocket ticket minted for it stayed valid for the rest of the 8 h lifetime.
//
// Drives the REAL ApiService/authApi over a recording adapter. The adapter sees the request before
// axios' xhr adapter drops the header, so what is pinned here is the thing that decides it: both
// calls carry a real JSON body (`{}`), which keeps `Content-Type: application/json` on the wire.
import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest'
import { setActivePinia, createPinia } from 'pinia'
import { createRecordingAdapter, replaceNetworkAdapter } from './helpers/recordingAdapter'

// Escape the global @/services/api stub (tests/vitest.setup.ts).
vi.mock('@/services/api', async (importOriginal) => importOriginal())

import { authApi, useApiService } from '@/services/api'

let restoreAdapter: (() => void) | null = null

function install() {
  const { adapter, requests } = createRecordingAdapter((request) =>
    request.url.endsWith('/auth/refresh')
      ? { data: { access_token: 'tok-new', token: 'tok-new', expires_in: 3600 } }
      : { data: { ok: true } }
  )
  restoreAdapter = replaceNetworkAdapter(useApiService(), adapter)
  return requests
}

describe('body-less auth POSTs send an explicit JSON body', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
  })

  afterEach(() => {
    restoreAdapter?.()
    restoreAdapter = null
    vi.restoreAllMocks()
  })

  it('logout posts {} as application/json', async () => {
    const requests = install()
    await authApi.logout()
    expect(requests).toHaveLength(1)
    expect(requests[0].method).toBe('POST')
    expect(requests[0].url).toBe('/api/v2/auth/logout')
    expect(requests[0].body).toEqual({})
    expect(requests[0].headers['content-type']).toContain('application/json')
  })

  it('refresh posts {} as application/json and returns the new token', async () => {
    const requests = install()
    const refreshed = await authApi.refresh()
    expect(refreshed.access_token).toBe('tok-new')
    expect(requests).toHaveLength(1)
    expect(requests[0].method).toBe('POST')
    expect(requests[0].url).toBe('/api/v2/auth/refresh')
    expect(requests[0].body).toEqual({})
    expect(requests[0].headers['content-type']).toContain('application/json')
  })
})
