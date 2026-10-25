// @vitest-environment node
// SEC3-03: the E-stop POST reaches the backend through the UI server, so the UI server must always
// have a socket for it. These specs run the real server.mjs against a recording stub backend and
// fill its socket table the ways a client can: sockets that never finish their request (the
// reviewer's scenario e5), sockets that hold a COMPLETE request whose response never finishes, and
// plain silent sockets. The policy itself is described in proxy-trust.mjs (SocketGuard).
import { describe, it, expect, afterEach, vi } from 'vitest'
import {
  parseResponse,
  rawClient,
  request,
  sleep,
  startUiServer,
  waitFor,
  wsHandshake,
  type RawClient,
  type UiServer,
} from './helpers/uiServer'

const E_STOP = '/api/v2/tractor/emergency-stop'
const SLOW_BODY =
  'POST /api/v2/missions/create HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\n' +
  'Content-Length: 1000\r\n\r\n{'
const HOLD = 'GET /api/v2/hang HTTP/1.1\r\nHost: x\r\n\r\n'
const BUSY = '{"detail":"Server busy, retry shortly"}'

// Every wait below has its own deadline; the test and hook timeouts only have to outlast them.
vi.setConfig({ testTimeout: 40_000, hookTimeout: 30_000 })

let ui: UiServer
afterEach(async () => {
  await ui?.stop()
})

const hanging = () => ui.seen.filter((s) => s.url === '/api/v2/hang').length
const isShed = (client: RawClient) => client.statusLine().startsWith('HTTP/1.1 503')
const isHeld = (client: RawClient) => client.received() === '' && client.closedAfter() === null

/**
 * `n` sockets that each send a complete request whose response never finishes. Resolves when every
 * request has either reached the backend (and is held) or been answered 503 and closed.
 */
async function hold(n: number) {
  const holders = Array.from({ length: n }, () => rawClient(ui.port, HOLD))
  await waitFor(
    () => hanging() + holders.filter(isShed).length === n,
    10_000,
    `${n} requests to be proxied or shed`
  )
  await waitFor(
    () => holders.filter(isShed).every((h) => h.closedAfter() !== null),
    5000,
    'the shed sockets to be closed'
  )
  return { holders, held: holders.filter(isHeld), shed: holders.filter(isShed) }
}

describe('sockets that never finish their request (the reviewer scenario e5)', () => {
  it('cannot deny the E-stop, now or after the deadline has passed', async () => {
    ui = await startUiServer({
      UI_MAX_CONNECTIONS: '16', // the minimum
      UI_HEADER_DEADLINE_MS: '1000',
      UI_REQUEST_DEADLINE_MS: '2000',
    })
    expect(await request(ui.port, 'POST', E_STOP, { body: '{}' })).toMatchObject({ status: 202 })

    // 16 sockets send complete headers and `Content-Length: 1000` with one byte of body, then nothing
    const slow = Array.from({ length: 16 }, () => rawClient(ui.port, SLOW_BODY))
    const opened = Date.now()
    await waitFor(
      () => ui.seen.filter((s) => s.url === '/api/v2/missions/create').length === 16,
      10_000,
      'the server to hold all 16 slow-body requests'
    )
    const stops = ui.seen.length

    // right away: the table is full, an E-stop still gets through (evicting a slow socket)
    const first = await request(ui.port, 'POST', E_STOP, { body: '{}' })
    expect(first).toMatchObject({ status: 202 })
    expect(await request(ui.port, 'POST', '/api/v2/control/emergency', { body: '{}' })).toMatchObject({
      status: 202,
    })
    expect(ui.seen.slice(stops)).toEqual([
      { method: 'POST', url: E_STOP },
      { method: 'POST', url: '/api/v2/control/emergency' },
    ])
    expect(slow.filter((c) => c.closedAfter() !== null).length).toBeGreaterThanOrEqual(1)

    // the request deadline closes every one of them (Node's own default would take 120 s)
    await waitFor(
      () => slow.every((c) => c.closedAfter() !== null),
      10_000,
      'the request deadline to close every slow-body socket'
    )

    // 12 s after they opened, past the 10 s the reviewer waited for
    await sleep(Math.max(0, 12_000 - (Date.now() - opened)))
    expect(await request(ui.port, 'POST', E_STOP, { body: '{}' })).toMatchObject({ status: 202 })
    expect(ui.seen.at(-1)).toEqual({ method: 'POST', url: E_STOP })
  }, 60_000)
})

describe('sockets that each hold a complete request whose response never finishes', () => {
  const env = {
    UI_MAX_CONNECTIONS: '16',
    UI_HEADER_DEADLINE_MS: '5000',
    UI_REQUEST_DEADLINE_MS: '8000',
  }

  it('keep at most max - reserve slots: the rest is shed with 503 and the E-stop gets through', async () => {
    ui = await startUiServer(env) // reserve 4: at most 12 productive sockets
    const { holders, held, shed } = await hold(16)
    expect(held).toHaveLength(12)
    expect(shed).toHaveLength(4)
    expect(hanging()).toBe(12) // a shed request was answered here, never proxied
    const answer = parseResponse(shed[0].received())
    expect(answer.statusLine).toBe('HTTP/1.1 503 Service Unavailable')
    expect(answer.headers).toMatchObject({
      'retry-after': '1',
      connection: 'close',
      'content-type': 'application/json',
    })
    expect(answer.body).toBe(BUSY)

    const proxied = ui.seen.length
    // an ordinary request, a GET and a PUT to an E-stop path (they stop nothing): all shed
    const ordinary = await request(ui.port, 'GET', '/api/v2/status')
    expect(ordinary).toMatchObject({ status: 503, body: BUSY })
    expect(ordinary.headers).toMatchObject({ 'retry-after': '1', connection: 'close' })
    expect(await request(ui.port, 'GET', E_STOP)).toMatchObject({ status: 503, body: BUSY })
    expect(
      await request(ui.port, 'PUT', '/api/v2/control/emergency', { body: '{}' })
    ).toMatchObject({ status: 503, body: BUSY })
    expect(ui.seen).toHaveLength(proxied)

    // the E-stop POST is not, in any of its spellings
    expect(await request(ui.port, 'POST', E_STOP, { body: '{}' })).toMatchObject({ status: 202 })
    expect(
      await request(ui.port, 'POST', '/api/control/emergency/?x=1', { body: '{}' })
    ).toMatchObject({ status: 202 })
    expect(ui.seen.slice(proxied)).toEqual([
      { method: 'POST', url: E_STOP },
      { method: 'POST', url: '/api/v2/control/emergency/?x=1' },
    ])

    // nothing that was serving a request was evicted for it
    expect(holders.filter(isHeld)).toHaveLength(12)
    // and the shedding is logged once, not once per request
    expect(ui.output().match(/request limit reached/g)).toHaveLength(1)
  }, 40_000)

  it('leave the E-stop a slot even when silent sockets fill the rest: it evicts the oldest of them', async () => {
    ui = await startUiServer(env)
    const { holders, held } = await hold(16)
    expect(held).toHaveLength(12)
    const silent = Array.from({ length: 4 }, () => rawClient(ui.port)) // 12 + 4: the table is full

    const stop = await request(ui.port, 'POST', E_STOP, { body: '{}' })

    expect(stop).toMatchObject({ status: 202 })
    await waitFor(() => silent[0].closedAfter() !== null, 5000, 'the oldest silent socket to go')
    expect(silent.slice(1).map((c) => c.closedAfter())).toEqual([null, null, null])
    expect(holders.filter(isHeld)).toHaveLength(12) // never a socket that is serving a request
  }, 40_000)

  it('also keep WebSockets within max - reserve: an upgrade over the quota is refused with 503', async () => {
    ui = await startUiServer(env)
    const sockets = Array.from({ length: 14 }, () => rawClient(ui.port, wsHandshake()))
    const answered = (c: RawClient) => c.statusLine() !== ''
    await waitFor(() => sockets.every(answered), 10_000, 'all 14 handshakes to be answered')

    const upgraded = sockets.filter((c) => c.statusLine().startsWith('HTTP/1.1 101'))
    const refused = sockets.filter(isShed)
    expect(upgraded).toHaveLength(12)
    expect(refused).toHaveLength(2)
    const answer = parseResponse(refused[0].received())
    expect(answer.headers).toMatchObject({ 'retry-after': '1', connection: 'close' })
    expect(answer.body).toBe(BUSY)
    expect(ui.seen.filter((s) => s.url.startsWith('/api/v2/ws/'))).toHaveLength(12)
    // the refused sockets are closed and the E-stop still gets through
    await waitFor(() => refused.every((c) => c.closedAfter() !== null), 5000, 'the refused to close')
    expect(await request(ui.port, 'POST', E_STOP, { body: '{}' })).toMatchObject({ status: 202 })
    expect(upgraded.every((c) => c.closedAfter() === null)).toBe(true)
  }, 40_000)
})

describe('UI_CONNECTION_RESERVE', () => {
  const env = {
    UI_MAX_CONNECTIONS: '16',
    UI_HEADER_DEADLINE_MS: '5000',
    UI_REQUEST_DEADLINE_MS: '8000',
  }

  it('sets how many slots are kept for the E-stop', async () => {
    ui = await startUiServer({ ...env, UI_CONNECTION_RESERVE: '8' })
    const { held, shed } = await hold(16)
    expect(held).toHaveLength(8)
    expect(shed).toHaveLength(8)
    expect(await request(ui.port, 'POST', E_STOP, { body: '{}' })).toMatchObject({ status: 202 })
  }, 40_000)

  it.each(['abc', '3', '0', '-2', '5.5', '16', '17'])(
    'falls back to the default (a quarter of the cap) and says so when it is %s',
    async (value) => {
      ui = await startUiServer({ ...env, UI_CONNECTION_RESERVE: value })
      const { held, shed } = await hold(16)
      expect(held).toHaveLength(12) // never a disabled or a widened reserve
      expect(shed).toHaveLength(4)
      expect(ui.output()).toContain(
        `UI_CONNECTION_RESERVE=${JSON.stringify(value)} is not an integer between 4 and 15: using 4`
      )
    },
    40_000
  )
})

describe('a flood of silent sockets', () => {
  it('is closed at the cap and logged once, not once per socket', async () => {
    ui = await startUiServer({ UI_MAX_CONNECTIONS: '16', UI_HEADER_DEADLINE_MS: '60000' })
    const silent = Array.from({ length: 200 }, () => rawClient(ui.port))

    await waitFor(
      () => silent.filter((c) => c.closedAfter() !== null).length >= 184,
      15_000,
      'the 184 sockets over the cap to be evicted'
    )

    expect(silent.filter((c) => c.closedAfter() === null)).toHaveLength(16)
    expect(ui.output().match(/connection limit \(16\) reached/g)).toHaveLength(1)
    expect(await request(ui.port, 'POST', E_STOP, { body: '{}' })).toMatchObject({ status: 202 })
  }, 40_000)
})
