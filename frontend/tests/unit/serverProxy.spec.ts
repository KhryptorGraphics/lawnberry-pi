// @vitest-environment node
// frontend/server.mjs is the only hop between a browser and the backend. The backend takes the
// client's address from the LAST X-Forwarded-For hop and trusts it only from this proxy, so the
// proxy must (a) drop every forwarding header a client sends, on plain HTTP AND on WebSocket
// upgrades (Express middleware never sees an upgrade), and (b) add the real socket address. These
// tests run the real server.mjs against a recording backend.
import { describe, it, expect, beforeAll, afterAll } from 'vitest'
import http from 'node:http'
import net from 'node:net'
import crypto from 'node:crypto'
import { spawn, type ChildProcess } from 'node:child_process'
import { fileURLToPath } from 'node:url'

const SERVER = fileURLToPath(new URL('../../server.mjs', import.meta.url))
const SPOOF = { 'X-Forwarded-For': '6.6.6.6', Forwarded: 'for=6.6.6.6', 'X-Real-IP': '6.6.6.6' }

interface Seen {
  url?: string
  headers: http.IncomingHttpHeaders
}

let backend: http.Server
let proxy: ChildProcess
let proxyPort = 0
const upgrades: Seen[] = []

function freePort(): Promise<number> {
  return new Promise((resolve) => {
    const probe = net.createServer()
    probe.listen(0, '127.0.0.1', () => {
      const { port } = probe.address() as net.AddressInfo
      probe.close(() => resolve(port))
    })
  })
}

function get(path: string, headers: Record<string, string> = {}): Promise<Seen> {
  return new Promise((resolve, reject) => {
    const req = http.request({ host: '127.0.0.1', port: proxyPort, path, headers }, (res) => {
      let body = ''
      res.on('data', (chunk) => (body += chunk))
      res.on('end', () => resolve(JSON.parse(body)))
    })
    req.on('error', reject)
    req.end()
  })
}

/** Raw WebSocket handshake through the proxy; resolves with the status line outcome. */
function handshake(path: string, headers: Record<string, string> = {}): Promise<string> {
  return new Promise((resolve) => {
    const req = http.request({
      host: '127.0.0.1',
      port: proxyPort,
      path,
      headers: {
        Connection: 'Upgrade',
        Upgrade: 'websocket',
        'Sec-WebSocket-Version': '13',
        'Sec-WebSocket-Key': crypto.randomBytes(16).toString('base64'),
        ...headers,
      },
    })
    const timer = setTimeout(() => {
      req.destroy()
      resolve('no response')
    }, 3000)
    req.on('upgrade', (res, socket) => {
      clearTimeout(timer)
      socket.destroy()
      resolve(`upgrade ${res.statusCode}`)
    })
    req.on('response', (res) => {
      clearTimeout(timer)
      resolve(`plain ${res.statusCode}`)
    })
    req.on('error', (e: NodeJS.ErrnoException) => {
      clearTimeout(timer)
      resolve(`error ${e.code}`)
    })
    req.end()
  })
}

beforeAll(async () => {
  const backendPort = await freePort()
  proxyPort = await freePort()
  backend = http.createServer((req, res) => {
    res.setHeader('content-type', 'application/json')
    res.end(JSON.stringify({ url: req.url, headers: req.headers } satisfies Seen))
  })
  backend.on('upgrade', (req, socket) => {
    upgrades.push({ url: req.url, headers: req.headers })
    const accept = crypto
      .createHash('sha1')
      .update(`${req.headers['sec-websocket-key']}258EAFA5-E914-47DA-95CA-C5AB0DC85B11`)
      .digest('base64')
    socket.write(
      'HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n' +
        `Sec-WebSocket-Accept: ${accept}\r\n\r\n`
    )
    setTimeout(() => socket.destroy(), 100)
  })
  await new Promise<void>((resolve) => backend.listen(backendPort, '127.0.0.1', resolve))

  proxy = spawn('node', [SERVER], {
    env: { ...process.env, PORT: String(proxyPort), BACKEND_URL: `http://127.0.0.1:${backendPort}` },
    stdio: ['ignore', 'pipe', 'inherit'],
  })
  await new Promise<void>((resolve, reject) => {
    const timer = setTimeout(() => reject(new Error('server.mjs did not start')), 15000)
    proxy.stdout?.on('data', (chunk: Buffer) => {
      if (chunk.toString().includes('is running')) {
        clearTimeout(timer)
        resolve()
      }
    })
  })
}, 20000)

afterAll(() => {
  proxy?.kill('SIGTERM')
  backend?.close()
})

describe('frontend/server.mjs forwarding headers', () => {
  it('drops spoofed forwarding headers on plain HTTP and forwards only the real socket address', async () => {
    const seen = await get('/api/v2/ping?x=1', SPOOF)

    expect(seen.url).toBe('/api/v2/ping?x=1')
    expect(seen.headers['x-forwarded-for']).toBe('127.0.0.1')
    expect(seen.headers['forwarded']).toBeUndefined()
    expect(seen.headers['x-real-ip']).toBeUndefined()
  })

  it('does the same for the /api (legacy mount) proxy', async () => {
    const seen = await get('/api/ping', SPOOF)

    expect(seen.headers['x-forwarded-for']).toBe('127.0.0.1')
    expect(seen.headers['forwarded']).toBeUndefined()
    expect(seen.headers['x-real-ip']).toBeUndefined()
  })

  it('drops spoofed forwarding headers on a WebSocket upgrade, and keeps the ticket query', async () => {
    upgrades.length = 0

    expect(await handshake('/api/v2/ws/telemetry?ticket=T-1', SPOOF)).toBe('upgrade 101')

    expect(upgrades).toHaveLength(1)
    expect(upgrades[0].url).toBe('/api/v2/ws/telemetry?ticket=T-1')
    expect(upgrades[0].headers['x-forwarded-for']).toBe('127.0.0.1')
    expect(upgrades[0].headers['forwarded']).toBeUndefined()
    expect(upgrades[0].headers['x-real-ip']).toBeUndefined()
  })

  it('maps the legacy /ws/<channel> path onto /api/v2/ws/<channel> without losing the ticket', async () => {
    upgrades.length = 0

    expect(await handshake('/ws/telemetry?ticket=T-2', SPOOF)).toBe('upgrade 101')
    expect(await handshake('/ws?ticket=T-3', SPOOF)).toBe('upgrade 101')

    expect(upgrades.map((u) => u.url)).toEqual([
      '/api/v2/ws/telemetry?ticket=T-2',
      '/api/v2/ws/telemetry?ticket=T-3',
    ])
    upgrades.forEach((u) => expect(u.headers['x-forwarded-for']).toBe('127.0.0.1'))
  })

  it('refuses an upgrade for a path that is not a WebSocket endpoint of this app', async () => {
    upgrades.length = 0

    const outcome = await handshake('/not-a-socket')

    expect(outcome).not.toBe('upgrade 101')
    expect(upgrades).toHaveLength(0)
  })
})
