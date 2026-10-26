// @vitest-environment node
// The deadlines of the UI server (server.mjs): the headers of EVERY request on a socket within
// UI_HEADER_DEADLINE_MS (the first request of a socket and each later one on a keep-alive socket),
// the whole request within UI_REQUEST_DEADLINE_MS, and nothing for an upgraded WebSocket or for a
// response that takes long. A socket that misses a deadline is destroyed; one that holds the table
// is what keeps the E-stop out (SEC3-03, serverEstopReserve.spec.ts).
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

// Every wait below has its own deadline; the test and hook timeouts only have to outlast them.
vi.setConfig({ testTimeout: 40_000, hookTimeout: 30_000 })

// Answered by the UI server itself, so the socket stays open for the next request (a proxied API
// request is sent to the backend with `Connection: close` and closes its socket).
const STATIC = 'GET /static-probe HTTP/1.1\r\nHost: x\r\n\r\n'
const SLOW_BODY =
  'POST /api/v2/missions/create HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\n' +
  'Content-Length: 1000\r\n\r\n{'
const HEAD = 'GET /static-probe HTTP/1.1\r\nHost: x\r\nX-Pad: ' + 'a'.repeat(600)

let ui: UiServer
const dribblers: ReturnType<typeof setInterval>[] = []
afterEach(async () => {
  dribblers.splice(0).forEach(clearInterval)
  await ui?.stop()
})

const answered = (client: RawClient) => {
  const { headers, body } = parseResponse(client.received())
  return client.received().includes('\r\n\r\n') && body.length >= Number(headers['content-length'])
}

/** Opens a keep-alive socket and waits for its first response. */
async function keepAlive(): Promise<RawClient> {
  const client = rawClient(ui.port, STATIC)
  await waitFor(() => answered(client), 10_000, 'the first response')
  expect(client.closedAfter()).toBeNull() // it is a keep-alive socket
  return client
}

/** Milliseconds until `client` is closed by the server, measured from now. */
async function closedWithin(client: RawClient, limitMs: number, what: string): Promise<number> {
  const started = Date.now()
  await waitFor(() => client.closedAfter() !== null, limitMs, what)
  return Date.now() - started
}

/** Sends the headers of a request one byte per 200 ms; returns how many bytes it has sent so far. */
function dribble(client: RawClient): () => number {
  let sent = 0
  dribblers.push(
    setInterval(() => {
      if (client.closedAfter() === null && sent < HEAD.length) client.write(HEAD[sent++])
    }, 200)
  )
  return () => sent
}

describe('the headers of every request', () => {
  const env = {
    UI_MAX_CONNECTIONS: '16',
    UI_HEADER_DEADLINE_MS: '1000',
    UI_REQUEST_DEADLINE_MS: '4000',
  }

  it('close a socket that connects and sends nothing', async () => {
    ui = await startUiServer(env)
    const client = rawClient(ui.port)

    expect(await closedWithin(client, 10_000, 'the deadline to close it')).toBeLessThan(6000)
  })

  it('close a socket that dribbles the headers of its first request', async () => {
    ui = await startUiServer(env)
    const client = rawClient(ui.port)
    const sent = dribble(client)

    expect(await closedWithin(client, 10_000, 'the deadline to close it')).toBeLessThan(6000)
    expect(sent()).toBeLessThan(HEAD.length) // one byte per 200 ms did not keep it alive
  })

  it('close a keep-alive socket that says nothing after a response, within the deadline', async () => {
    ui = await startUiServer(env)
    const client = await keepAlive()

    // the HTTP server's own keep-alive timeout would have waited for 5 s
    expect(await closedWithin(client, 10_000, 'the deadline to close it')).toBeLessThan(4000)
  })

  it('close a keep-alive socket that dribbles the headers of its next request', async () => {
    ui = await startUiServer(env)
    const client = await keepAlive()
    const sent = dribble(client)

    expect(await closedWithin(client, 10_000, 'the deadline to close it')).toBeLessThan(6000)
    expect(sent()).toBeLessThan(HEAD.length)
  })

  it('leave a keep-alive socket alone that sends its next request in time, again and again', async () => {
    ui = await startUiServer({ ...env, UI_HEADER_DEADLINE_MS: '3000', UI_REQUEST_DEADLINE_MS: '8000' })
    const client = await keepAlive()
    for (let round = 2; round <= 4; round += 1) {
      await sleep(1000) // each wait is within the 3 s deadline, all three together are not
      client.write(STATIC)
      await waitFor(
        () => (client.received().match(/HTTP\/1\.1 /g) ?? []).length === round,
        5000,
        `response number ${round}`
      )
    }
    expect(client.closedAfter()).toBeNull()
  })
})

describe('the whole request', () => {
  it('is cut off at UI_REQUEST_DEADLINE_MS when its body does not arrive (Node waits 120 s)', async () => {
    ui = await startUiServer({
      UI_MAX_CONNECTIONS: '16',
      UI_HEADER_DEADLINE_MS: '1000',
      UI_REQUEST_DEADLINE_MS: '1000',
    })
    const client = rawClient(ui.port, SLOW_BODY)

    expect(await closedWithin(client, 15_000, 'the request deadline to close it')).toBeLessThan(8000)
    expect(client.statusLine()).toBe('HTTP/1.1 408 Request Timeout')
  })

  it('may take longer than the header deadline when it keeps arriving within its own', async () => {
    ui = await startUiServer({
      UI_MAX_CONNECTIONS: '16',
      UI_HEADER_DEADLINE_MS: '1000',
      UI_REQUEST_DEADLINE_MS: '8000',
    })
    const client = rawClient(
      ui.port,
      'POST /api/v2/missions/create HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\n' +
        'Content-Length: 7\r\n\r\n{"a'
    )
    await sleep(2200) // past the header deadline: the headers were in long ago
    client.write('":1}')

    // a proxied API response is `Connection: close`: the body is complete when the socket closes
    await waitFor(() => client.closedAfter() !== null, 5000, 'the answer')
    expect(client.statusLine()).toBe('HTTP/1.1 200 OK')
    expect(JSON.parse(parseResponse(client.received()).body)).toEqual({
      url: '/api/v2/missions/create',
      method: 'POST',
    })
    expect(ui.seen.at(-1)).toEqual({ method: 'POST', url: '/api/v2/missions/create' })
  })

  it('is not cut off because the response is slow: the deadlines are about receiving', async () => {
    ui = await startUiServer({
      UI_MAX_CONNECTIONS: '16',
      UI_HEADER_DEADLINE_MS: '1000',
      UI_REQUEST_DEADLINE_MS: '1000',
    })

    const outcome = await request(ui.port, 'GET', '/api/v2/slow?ms=3500')

    expect(outcome).toMatchObject({ status: 200 })
    expect(JSON.parse(outcome.body ?? '{}')).toEqual({ url: '/api/v2/slow?ms=3500', method: 'GET' })
  })
})

describe('an upgraded WebSocket', () => {
  it('is out of reach of both deadlines', async () => {
    ui = await startUiServer({
      UI_MAX_CONNECTIONS: '16',
      UI_HEADER_DEADLINE_MS: '1000',
      UI_REQUEST_DEADLINE_MS: '1000',
    })
    const client = rawClient(ui.port, wsHandshake())
    await waitFor(() => client.statusLine() === 'HTTP/1.1 101 Switching Protocols', 5000, 'the 101')

    await sleep(3500) // three times the deadlines, and the connection checks have run

    expect(client.closedAfter()).toBeNull()
    client.write('still-here')
    await waitFor(() => client.received().endsWith('still-here'), 5000, 'the echo through the proxy')
  })
})

describe('the settings', () => {
  // Node swaps the two timeouts when the header one is longer: left alone, UI_HEADER_DEADLINE_MS=60000
  // would silently turn the request deadline into 60 s
  it('cap the header deadline at the request deadline, which includes it, and say so', async () => {
    ui = await startUiServer({
      UI_MAX_CONNECTIONS: '16',
      UI_HEADER_DEADLINE_MS: '60000',
      UI_REQUEST_DEADLINE_MS: '2000',
    })
    const silent = rawClient(ui.port)
    const slowBody = rawClient(ui.port, SLOW_BODY)

    expect(await closedWithin(silent, 15_000, 'the capped deadline to close it')).toBeLessThan(8000)
    expect(await closedWithin(slowBody, 15_000, 'the request deadline to close it')).toBeLessThan(8000)
    expect(slowBody.statusLine()).toBe('HTTP/1.1 408 Request Timeout')
    expect(ui.output()).toContain(
      'UI_HEADER_DEADLINE_MS=60000 is longer than UI_REQUEST_DEADLINE_MS=2000'
    )
  })

  it('refuse a request deadline under one second and use 15 s instead, saying so', async () => {
    ui = await startUiServer({
      UI_MAX_CONNECTIONS: '16',
      UI_HEADER_DEADLINE_MS: '1000',
      UI_REQUEST_DEADLINE_MS: '500',
    })
    const client = rawClient(ui.port, SLOW_BODY)

    await sleep(3500) // 500 ms would have closed it three seconds ago

    expect(client.closedAfter()).toBeNull()
    expect(ui.output()).toContain(
      'UI_REQUEST_DEADLINE_MS="500" is not an integer >= 1000: using 15000'
    )
  })

  it.each(['abc', '-1', '1000.5', '0'])('refuse a request deadline of %s', async (value) => {
    ui = await startUiServer({ UI_REQUEST_DEADLINE_MS: value })

    expect(ui.output()).toContain(
      `UI_REQUEST_DEADLINE_MS=${JSON.stringify(value)} is not an integer >= 1000: using 15000`
    )
  })
})
