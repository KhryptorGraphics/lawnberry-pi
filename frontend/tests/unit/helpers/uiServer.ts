// Harness for the specs that run the real frontend/server.mjs (a child process) in front of a
// recording stub backend, and talk to it over raw sockets. Every wait has its own deadline: a spec
// that waits on a socket must never hang the run (there is no test timeout plugin).
import crypto from 'node:crypto'
import http from 'node:http'
import net from 'node:net'
import { spawn, type ChildProcess } from 'node:child_process'
import { fileURLToPath } from 'node:url'

const SERVER = fileURLToPath(new URL('../../../server.mjs', import.meta.url))

export const sleep = (ms: number) => new Promise<void>((resolve) => setTimeout(resolve, ms))

/** Polls `predicate` every 25 ms; throws naming `what` when it still does not hold after `ms`. */
export async function waitFor(predicate: () => boolean, ms: number, what: string): Promise<void> {
  const deadline = Date.now() + ms
  while (!predicate()) {
    if (Date.now() > deadline) throw new Error(`gave up after ${ms} ms waiting for ${what}`)
    await sleep(25)
  }
}

/** A port nobody listens on right now (it can be taken before the caller binds it: see below). */
function freePort(): Promise<number> {
  return new Promise((resolve) => {
    const probe = net.createServer()
    probe.listen(0, '127.0.0.1', () => {
      const { port } = probe.address() as net.AddressInfo
      probe.close(() => resolve(port))
    })
  })
}

/** What the stub backend saw: the request line, as the backend would route it. */
export interface Seen {
  method: string
  url: string
}

export interface UiServer {
  port: number
  /** Requests that reached the stub backend, in arrival order. */
  seen: Seen[]
  /** Everything the server printed (stdout and stderr). */
  output(): string
  stop(): Promise<void>
}

const E_STOP_PATH = /^\/api\/v2\/(control\/emergency|control\/emergency-stop|tractor\/emergency-stop)$/

/**
 * The stub backend: `/api/v2/hang...` is never answered (its socket stays busy for good),
 * `/api/v2/slow?ms=N` answers after N ms, a POST to an E-stop path answers 202, anything else 200.
 * Request bodies are drained, but a request whose body never completes is never answered. A
 * WebSocket upgrade is accepted and echoes what it is sent, until the client goes away.
 */
function stubBackend(seen: Seen[]): http.Server {
  const backend = http.createServer((req, res) => {
    const url = req.url ?? ''
    seen.push({ method: req.method ?? '', url })
    const pathname = url.split('?')[0].replace(/\/+$/, '')
    if (pathname === '/api/v2/hang') return
    req.resume()
    req.on('end', () => {
      const slow = /^\/api\/v2\/slow\?ms=(\d+)$/.exec(url)
      const status = req.method === 'POST' && E_STOP_PATH.test(pathname) ? 202 : 200
      setTimeout(() => {
        const text = JSON.stringify({ url, method: req.method })
        res.writeHead(status, {
          'content-type': 'application/json',
          'content-length': Buffer.byteLength(text),
        })
        res.end(text)
      }, slow ? Number(slow[1]) : 0)
    })
  })
  backend.on('upgrade', (req, socket) => {
    seen.push({ method: req.method ?? '', url: req.url ?? '' })
    const accept = crypto
      .createHash('sha1')
      .update(`${req.headers['sec-websocket-key']}258EAFA5-E914-47DA-95CA-C5AB0DC85B11`)
      .digest('base64')
    socket.write(
      'HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n' +
        `Sec-WebSocket-Accept: ${accept}\r\n\r\n`
    )
    socket.on('data', (chunk) => socket.write(chunk))
    socket.on('error', () => undefined)
  })
  return backend
}

/** The request line and headers of a WebSocket handshake (the server answers 101 or 503). */
export const wsHandshake = (path = '/api/v2/ws/telemetry?ticket=T') =>
  `GET ${path} HTTP/1.1\r\nHost: x\r\nConnection: Upgrade\r\nUpgrade: websocket\r\n` +
  `Sec-WebSocket-Version: 13\r\nSec-WebSocket-Key: ${crypto.randomBytes(16).toString('base64')}\r\n\r\n`

// Settings of the developer's shell (UI_*, BACKEND_*, PORT) must not leak into the server under test.
function inheritedEnv(): Record<string, string> {
  const env: Record<string, string> = {}
  for (const [name, text] of Object.entries(process.env)) {
    if (text !== undefined && !/^(UI_|BACKEND_|PORT$)/.test(name)) env[name] = text
  }
  return env
}

const clients = new Set<net.Socket>()

async function startOnce(env: Record<string, string>): Promise<UiServer> {
  const seen: Seen[] = []
  const backend = stubBackend(seen)
  // closeAllConnections() does not know the sockets that upgraded: destroy every one ourselves
  const backendSockets = new Set<net.Socket>()
  backend.on('connection', (socket) => {
    backendSockets.add(socket)
    socket.once('close', () => backendSockets.delete(socket))
  })
  await new Promise<void>((resolve) => backend.listen(0, '127.0.0.1', resolve)) // no race: port 0
  const backendPort = (backend.address() as net.AddressInfo).port
  const port = await freePort()
  let printed = ''
  const child: ChildProcess = spawn(process.execPath, [SERVER], {
    env: {
      ...inheritedEnv(),
      PORT: String(port),
      BACKEND_URL: `http://127.0.0.1:${backendPort}`,
      ...env,
    },
    stdio: ['ignore', 'pipe', 'pipe'],
  })
  const collect = (chunk: Buffer) => {
    if (printed.length < 200_000) printed += chunk.toString()
  }
  child.stdout?.on('data', collect)
  child.stderr?.on('data', collect)

  const stop = async () => {
    for (const socket of clients) socket.destroy()
    clients.clear()
    if (child.exitCode === null) {
      const exited = new Promise<void>((resolve) => child.once('exit', () => resolve()))
      child.kill('SIGTERM')
      await Promise.race([exited, sleep(3000)])
      if (child.exitCode === null) child.kill('SIGKILL')
    }
    for (const socket of backendSockets) socket.destroy()
    await new Promise<void>((resolve) => backend.close(() => resolve()))
  }

  try {
    await new Promise<void>((resolve, reject) => {
      const timer = setTimeout(() => reject(new Error(`server.mjs did not start:\n${printed}`)), 20_000)
      child.once('exit', () => {
        clearTimeout(timer)
        reject(new Error(`server.mjs exited while starting:\n${printed}`))
      })
      child.stdout?.on('data', () => {
        if (!printed.includes('is running')) return
        clearTimeout(timer)
        resolve()
      })
    })
  } catch (error) {
    await stop()
    throw error
  }
  return { port, seen, output: () => printed, stop }
}

/**
 * Starts server.mjs with `env` on a free port. Always `stop()` it: that also destroys every raw
 * client opened with `rawClient`. A port found free can be taken before the server binds it (the
 * box runs other things): the start is then repeated on another one.
 */
export async function startUiServer(env: Record<string, string> = {}): Promise<UiServer> {
  for (let attempt = 1; ; attempt += 1) {
    try {
      return await startOnce(env)
    } catch (error) {
      if (attempt >= 5 || !String(error).includes('EADDRINUSE')) throw error
    }
  }
}

export interface RawClient {
  socket: net.Socket
  /** Everything the server has sent so far. */
  received(): string
  /** The status line of the first response, or '' while there is none. */
  statusLine(): string
  /** Milliseconds from connecting to the server closing the socket, or null while it is open. */
  closedAfter(): number | null
  write(text: string): void
}

/**
 * A raw TCP client of the UI server, optionally sending `text` at once. The socket is read as it
 * comes (an unread socket never notices that the server closed it).
 */
export function rawClient(port: number, text?: string): RawClient {
  const socket = net.connect(port, '127.0.0.1')
  clients.add(socket)
  const started = Date.now()
  let got = ''
  let closedAt: number | null = null
  socket.on('data', (chunk) => (got += chunk.toString('latin1')))
  socket.on('error', () => undefined) // the server resets sockets it evicts
  socket.on('close', () => {
    closedAt = Date.now()
    clients.delete(socket)
  })
  if (text !== undefined) socket.write(text)
  return {
    socket,
    received: () => got,
    statusLine: () => got.split('\r\n')[0],
    closedAfter: () => (closedAt === null ? null : closedAt - started),
    write: (more) => void socket.write(more),
  }
}

export interface Outcome {
  status?: number
  headers?: http.IncomingHttpHeaders
  body?: string
  error?: string
}

/** One request on a fresh connection (`Connection: close`) with its own client-side deadline. */
export function request(
  port: number,
  method: string,
  path: string,
  options: { body?: string; timeoutMs?: number } = {}
): Promise<Outcome> {
  const { body, timeoutMs = 8000 } = options
  return new Promise((resolve) => {
    const headers: Record<string, string | number> = { connection: 'close' }
    if (body !== undefined) {
      headers['content-type'] = 'application/json'
      headers['content-length'] = Buffer.byteLength(body)
    }
    const req = http.request({ host: '127.0.0.1', port, method, path, headers, agent: false }, (res) => {
      let text = ''
      res.on('data', (chunk) => (text += chunk))
      res.on('end', () => resolve({ status: res.statusCode, headers: res.headers, body: text }))
      res.on('error', (e: NodeJS.ErrnoException) => resolve({ error: e.code ?? e.message }))
    })
    req.setTimeout(timeoutMs, () => {
      req.destroy()
      resolve({ error: `no answer within ${timeoutMs} ms` })
    })
    req.on('error', (e: NodeJS.ErrnoException) => resolve({ error: e.code ?? e.message }))
    req.end(body)
  })
}

/** The first response in `text` (a raw socket's input): status line, lower-cased headers, body. */
export function parseResponse(text: string) {
  const [head, ...rest] = text.split('\r\n\r\n')
  const [statusLine, ...lines] = head.split('\r\n')
  const headers: Record<string, string> = {}
  for (const line of lines) {
    const at = line.indexOf(':')
    headers[line.slice(0, at).toLowerCase()] = line.slice(at + 1).trim()
  }
  return { statusLine, headers, body: rest.join('\r\n\r\n') }
}
