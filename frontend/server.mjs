// Simple production server for LawnBerry frontend
// - Serves static files from dist/
// - Proxies /api -> backend /api/v2
// - Proxies /ws (WebSocket) -> backend /api/v2/ws/telemetry
//
// Client address handling. The backend counts login lockouts and rate limits per client address
// and believes an X-Forwarded-For hop only from a forwarder that was declared to it
// (LAWN_BERRY_TRUSTED_PROXIES, see backend/src/core/peer_address.py). This server is that
// forwarder: it throws away every forwarding header a client sent, works out the client itself
// (the socket address, or, ONLY behind a forwarder listed in UI_TRUSTED_PROXIES, the address that
// forwarder recorded: see proxy-trust.mjs) and sends the backend exactly ONE hop. It connects to
// the backend from BACKEND_SOURCE_ADDRESS (the shipped unit: 127.0.0.2) so the backend can tell
// this server from every other process on the host, which all connect from 127.0.0.1. Plain HTTP
// and WebSocket upgrades alike.
//
// Connection handling. A socket that connects and sends nothing never ends on its own, and with
// every descriptor held the E-stop POST cannot connect. Sockets get a deadline for their first
// request and for their headers, and when UI_MAX_CONNECTIONS are open the socket that has been
// idle the longest makes room for the new one (see SocketGuard in proxy-trust.mjs).

import express from 'express'
import http from 'http'
import path from 'path'
import { fileURLToPath } from 'url'
import compression from 'compression'
import morgan from 'morgan'
import { createProxyMiddleware } from 'http-proxy-middleware'
import {
  SocketGuard,
  canBind,
  describeTrustedProxies,
  parseTrustedProxies,
  resolveClient,
} from './proxy-trust.mjs'

const __filename = fileURLToPath(import.meta.url)
const __dirname = path.dirname(__filename)

const app = express()
const PORT = process.env.PORT || 3000
const BACKEND_URL = process.env.BACKEND_URL || 'http://127.0.0.1:8081'

// ---- who forwards client addresses to us -------------------------------------------------------
const trust = parseTrustedProxies(process.env.UI_TRUSTED_PROXIES)
if (trust.problems.length) {
  console.error(
    `UI_TRUSTED_PROXIES: refusing the whole list, NO forwarder is trusted until it is fixed ` +
      `(every client behind a forwarder then shares the forwarder's address): ${trust.problems.join('; ')}`
  )
}
console.log(`Trusted forwarders (UI_TRUSTED_PROXIES): ${describeTrustedProxies(trust.proxies)}`)

// The address we connect to the backend FROM. Only used when this process can bind it (an alias of
// the loopback on Linux; a bind that fails would make every proxied request fail).
let backendSourceAddress
const requestedSource = (process.env.BACKEND_SOURCE_ADDRESS || '').trim()
if (requestedSource) {
  if (await canBind(requestedSource)) {
    backendSourceAddress = requestedSource
  } else {
    console.error(
      `BACKEND_SOURCE_ADDRESS=${requestedSource} cannot be bound here: connecting to the backend ` +
        `from the default address (the backend then cannot tell this server from other local ` +
        `processes and counts every browser as one client)`
    )
  }
}
console.log(`Backend source address: ${backendSourceAddress ?? '(default)'}`)

function intFromEnv(name, fallback, min) {
  const raw = process.env[name]
  if (raw === undefined || raw.trim() === '') return fallback
  const value = Number(raw)
  if (!Number.isInteger(value) || value < min) {
    console.error(`${name}=${JSON.stringify(raw)} is not an integer >= ${min}: using ${fallback}`)
    return fallback
  }
  return value
}
const MAX_CONNECTIONS = intFromEnv('UI_MAX_CONNECTIONS', 512, 16)
const HEADER_DEADLINE_MS = intFromEnv('UI_HEADER_DEADLINE_MS', 10_000, 1000)

// One warning per message per minute: a flood of odd headers must not be a flood of log lines.
const lastWarned = new Map()
function warnLimited(message) {
  const now = Date.now()
  if (now - (lastWarned.get(message) ?? 0) < 60_000) return
  if (lastWarned.size >= 64) lastWarned.delete(lastWarned.keys().next().value)
  lastWarned.set(message, now)
  console.warn(message)
}

// Headers a client could use to claim another address or origin. Header names are lower-case in Node.
const CLIENT_WRITTEN_FORWARDING_HEADERS = [
  'x-forwarded-for',
  'forwarded',
  'x-real-ip',
  'x-forwarded-proto',
  'x-forwarded-host',
  'x-forwarded-port',
]

/**
 * Replace every forwarding header of the request with what THIS server knows: one
 * X-Forwarded-For hop (the resolved client), the Host the client asked for, and the scheme.
 * Runs first on plain HTTP and again on WebSocket upgrades (which never pass Express middleware).
 */
function applyForwardingHeaders(req) {
  const peer = req.socket.remoteAddress
  const { address, forwarded, warning } = resolveClient(
    peer,
    req.headers['x-forwarded-for'],
    trust.proxies
  )
  if (warning) warnLimited(warning)
  const inboundProto = String(req.headers['x-forwarded-proto'] ?? '').split(',').pop().trim()
  const host = req.headers.host
  for (const name of CLIENT_WRITTEN_FORWARDING_HEADERS) delete req.headers[name]
  req.lbClient = address
  req.headers['x-forwarded-for'] = address
  if (host) req.headers['x-forwarded-host'] = host
  req.headers['x-forwarded-proto'] =
    forwarded && (inboundProto === 'http' || inboundProto === 'https')
      ? inboundProto
      : req.socket.encrypted
        ? 'https'
        : 'http'
}

// MUST stay the first middleware: nothing below may see a client-supplied forwarding header.
app.use((req, _res, next) => {
  applyForwardingHeaders(req)
  next()
})

// Logging & compression: the log shows the resolved client, not the forwarder's socket
morgan.token('client', (req) => req.lbClient ?? req.socket.remoteAddress)
app.use(
  morgan(
    ':client - :remote-user [:date[clf]] ":method :url HTTP/:http-version" :status ' +
      ':res[content-length] ":referrer" ":user-agent"'
  )
)
app.use(compression())

// Serve branding assets (e.g., LawnBerryPi_Pin.png) so they are available to the UI
const brandingDir = path.resolve(__dirname, '../branding')
app.use('/branding', express.static(brandingDir, { maxAge: '30d' }))
// Back-compat: expose the primary mower pin at the root path expected by the editor
app.get('/LawnBerryPi_Pin.png', (_req, res) => {
  res.sendFile(path.join(brandingDir, 'LawnBerryPi_Pin.png'))
})

// WebSocket upgrades never pass through Express middleware, so they are dispatched explicitly by
// the 'upgrade' listener at the bottom (`ws: false` here keeps http-proxy-middleware from also
// subscribing to 'upgrade' lazily, and only after a first plain request, which is what used to
// make the first WebSocket attempt after a restart fail).
//
// '/ws/<channel>?query' (legacy path) -> '/api/v2/ws/<channel>?query', default channel telemetry.
// The query string (the single-use `ticket`) must survive. Plain HTTP requests reach the rewrite
// with Express's mount prefix already stripped ('/<channel>?query'), upgrades with the full URL.
function legacyWsRewrite(url) {
  const queryAt = url.indexOf('?')
  const pathname = queryAt === -1 ? url : url.slice(0, queryAt)
  const query = queryAt === -1 ? '' : url.slice(queryAt)
  const channel = pathname.replace(/^\/ws(?=\/|$)/, '').replace(/^\//, '')
  return `/api/v2/ws/${channel || 'telemetry'}${query}`
}

// `xfwd` stays false: the forwarding headers are set above, from the resolved client. http-proxy's
// own xfwd would append the SOCKET address to them (a second hop, and the wrong one behind a
// forwarder). `localAddress` makes the connection to the backend come from BACKEND_SOURCE_ADDRESS.
const proxyBase = {
  target: BACKEND_URL,
  changeOrigin: true,
  ws: false,
  xfwd: false,
  ...(backendSourceAddress ? { localAddress: backendSourceAddress } : {}),
}

// IMPORTANT: Register more specific proxies BEFORE generic ones to avoid mismatches

// Proxy WebSocket for /api/v2/ws/telemetry as-is (frontend connects here by default)
const apiWsProxy = createProxyMiddleware(proxyBase)
app.use('/api/v2/ws/telemetry', apiWsProxy)

// Direct /api/v2 passthrough (must be before generic '/api')
app.use('/api/v2', (req, _res, next) => {
  console.log(`[proxy] match /api/v2 -> ${req.method} ${req.originalUrl}`)
  next()
})
app.use(
  '/api/v2',
  createProxyMiddleware({
    ...proxyBase,
    logLevel: 'debug',
    pathRewrite: (path) => '/api/v2' + path,
  })
)

// Proxy API to backend. Express strips the mount prefix '/api', so inside
// this middleware, `path` will start with '/v2/...' or '/something'.
// - If it's '/v2/...', prepend '/api' -> '/api/v2/...'
// - Else, prepend '/api/v2' -> '/api/v2/...'
app.use('/api', (req, _res, next) => {
  console.log(`[proxy] match /api -> ${req.method} ${req.originalUrl}`)
  next()
})
app.use(
  '/api',
  createProxyMiddleware({
    ...proxyBase,
    logLevel: 'debug',
    pathRewrite: (path) => {
      if (path.startsWith('/v2/')) return '/api' + path
      return '/api/v2' + path
    },
  })
)

// Proxy WebSocket path to backend telemetry endpoint
const legacyWsProxy = createProxyMiddleware({
  ...proxyBase,
  pathRewrite: legacyWsRewrite,
})
app.use('/ws', legacyWsProxy)

// (kept earlier and more specific above)

// Serve static built files
const distDir = path.resolve(__dirname, 'dist')
app.use(express.static(distDir, { maxAge: '1d', index: 'index.html' }))

// SPA fallback: serve index.html for any non-API route
app.get('*', (req, res) => {
  // Avoid intercepting proxied routes
  if (req.path.startsWith('/api') || req.path.startsWith('/ws')) {
    return res.status(404).send('Not Found')
  }
  res.sendFile(path.join(distDir, 'index.html'))
})

// ---- the server ----------------------------------------------------------------------------------
const server = http.createServer({ connectionsCheckingInterval: 1000 }, app)
// Slow or stalled requests: the headers within the deadline, the whole request within two minutes
// (Node's defaults are 60 s and 300 s); an idle keep-alive connection is closed after 5 s.
server.headersTimeout = HEADER_DEADLINE_MS
server.requestTimeout = 120_000
server.keepAliveTimeout = 5_000

const guard = new SocketGuard(MAX_CONNECTIONS)
server.on('connection', (socket) => {
  const destroyed = guard.register(socket)
  if (destroyed.length) {
    warnLimited(
      `connection limit (${MAX_CONNECTIONS}) reached: closed ${destroyed.length} idle socket(s) ` +
        `(evicted ${guard.evicted}, refused ${guard.refused} so far)`
    )
  }
  if (!socket.destroyed) {
    // A socket that never sends a request line is not covered by headersTimeout: give it a deadline.
    socket.setTimeout(HEADER_DEADLINE_MS, () => socket.destroy())
  }
})
server.on('request', (req, res) => {
  const socket = req.socket
  socket.setTimeout(0) // it has spoken: keep-alive and request timeouts take over
  guard.begin(socket)
  res.once('close', () => guard.end(socket))
})

server.listen(PORT, '0.0.0.0', () => {
  console.log(`LawnBerry frontend is running at http://0.0.0.0:${PORT}`)
  console.log(`Proxying API to ${BACKEND_URL}`)
})

// The one 'upgrade' listener: replace client-supplied forwarding headers with ours, then hand the
// upgrade to the proxy that owns the path. Anything else is not a WebSocket endpoint of this app.
server.on('upgrade', (req, socket, head) => {
  socket.setTimeout(0)
  guard.begin(socket) // an upgraded socket is busy for good
  applyForwardingHeaders(req)
  const url = req.url || ''
  if (url.startsWith('/api/v2/ws/')) {
    apiWsProxy.upgrade(req, socket, head)
  } else if (url === '/ws' || /^\/ws[/?]/.test(url)) {
    legacyWsProxy.upgrade(req, socket, head)
  } else {
    socket.destroy()
  }
})
