// The login JWT must not travel in a WebSocket URL (the server logs the query string). A signed-in
// client trades its bearer token for a single-use ticket (POST /api/v2/auth/ws-ticket) before EVERY
// connect attempt. Escape the global WebSocket stub (tests/vitest.setup.ts): this drives the real
// WebSocketService against a fake WebSocket and the mocked api layer.
import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest'
import apiClient from '@/services/api'
import { WebSocketService } from '@/services/websocket'

vi.mock('@/services/websocket', async (importOriginal) => importOriginal())

const JWT = 'eyJhbGciOiJIUzI1NiJ9.login.jwt'
const BASE = 'ws://mower.local:3000/api/v2/ws/telemetry'

class FakeSocket {
  static OPEN = 1
  static instances: FakeSocket[] = []
  readyState = 0
  onopen: (() => void) | null = null
  onclose: (() => void) | null = null
  onerror: ((error: unknown) => void) | null = null
  onmessage: ((event: { data: string }) => void) | null = null
  constructor(public url: string) {
    FakeSocket.instances.push(this)
  }
  send() {}
  close() {}
}

const urls = () => FakeSocket.instances.map((s) => s.url)
const flush = () => vi.advanceTimersByTimeAsync(0)

describe('WebSocket tickets', () => {
  beforeEach(() => {
    vi.useFakeTimers()
    FakeSocket.instances = []
    vi.stubGlobal('WebSocket', FakeSocket)
    localStorage.setItem('auth_token', JWT)
  })

  afterEach(() => {
    vi.useRealTimers()
    vi.unstubAllGlobals()
  })

  it('trades the bearer token for a ticket and connects with ?ticket=, never with the JWT', async () => {
    vi.mocked(apiClient.post).mockResolvedValue({ data: { ticket: 'T-1', expires_in: 30 } })
    const service = new WebSocketService(BASE)

    void service.connect()
    await flush()

    expect(apiClient.post).toHaveBeenCalledWith('/api/v2/auth/ws-ticket', {}, { timeout: 5000 })
    expect(urls()).toEqual([`${BASE}?ticket=T-1`])
    expect(urls()[0]).not.toContain(JWT)
    expect(urls()[0]).not.toContain('access_token')
  })

  it('fetches a fresh ticket for every reconnect (a ticket is single-use)', async () => {
    vi.mocked(apiClient.post)
      .mockResolvedValueOnce({ data: { ticket: 'T-1' } })
      .mockResolvedValueOnce({ data: { ticket: 'T-2' } })
    const service = new WebSocketService(BASE)
    void service.connect()
    await flush()
    FakeSocket.instances[0].onopen?.()

    FakeSocket.instances[0].onclose?.() // the link drops; the service schedules a reconnect
    await vi.advanceTimersByTimeAsync(2000)

    expect(urls()).toEqual([`${BASE}?ticket=T-1`, `${BASE}?ticket=T-2`])
    expect(apiClient.post).toHaveBeenCalledTimes(2)
  })

  it('fetches a fresh ticket for the alternate URL candidate too', async () => {
    vi.mocked(apiClient.post)
      .mockResolvedValueOnce({ data: { ticket: 'T-1' } })
      .mockResolvedValueOnce({ data: { ticket: 'T-2' } })
    const service = new WebSocketService(BASE)
    void service.connect()
    await flush()

    FakeSocket.instances[0].onerror?.(new Error('refused')) // primary path refused: try /ws/telemetry
    await flush()

    expect(urls()).toEqual([`${BASE}?ticket=T-1`, 'ws://mower.local:3000/ws/telemetry?ticket=T-2'])
  })

  it('connects without a ticket (and without calling the backend) when nobody is signed in', async () => {
    localStorage.removeItem('auth_token')
    const service = new WebSocketService(BASE)

    void service.connect()
    await flush()

    expect(apiClient.post).not.toHaveBeenCalled()
    expect(urls()).toEqual([BASE])
  })

  it('still connects, anonymously and without the JWT, when the ticket request fails', async () => {
    vi.mocked(apiClient.post).mockRejectedValue({ response: { status: 401 } })
    const service = new WebSocketService(BASE)

    void service.connect()
    await flush()

    expect(urls()).toEqual([BASE])
    expect(urls()[0]).not.toContain(JWT)
  })
})
