// Simple production server for LawnBerry frontend
// - Serves static files from dist/
// - Proxies /api -> backend /api/v2
// - Proxies /ws (WebSocket) -> backend /api/v2/ws/telemetry
//
// Client address handling: the backend trusts X-Forwarded-For only from this proxy and uses its
// LAST hop as the client's address (LAWN_BERRY_TRUSTED_PROXIES; login lockout and rate limits are
// keyed on it). So whatever forwarding headers a client sends are discarded here, and http-proxy
// (`xfwd: true`) then adds the real socket address as the only hop, for plain HTTP and for
// WebSocket upgrades alike.

import express from 'express'
import path from 'path'
import { fileURLToPath } from 'url'
import compression from 'compression'
import morgan from 'morgan'
import { createProxyMiddleware } from 'http-proxy-middleware'

const __filename = fileURLToPath(import.meta.url)
const __dirname = path.dirname(__filename)

const app = express()
const PORT = process.env.PORT || 3000
const BACKEND_URL = process.env.BACKEND_URL || 'http://127.0.0.1:8081'

// Headers a client could use to claim another address. Header names are lower-case in Node.
const SPOOFABLE_FORWARDING_HEADERS = ['x-forwarded-for', 'forwarded', 'x-real-ip']
function stripForwardingHeaders(req) {
  for (const name of SPOOFABLE_FORWARDING_HEADERS) delete req.headers[name]
}

// MUST stay the first middleware: nothing below may see a client-supplied forwarding header.
app.use((req, _res, next) => {
  stripForwardingHeaders(req)
  next()
})

// Logging & compression
app.use(morgan('combined'))
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

// IMPORTANT: Register more specific proxies BEFORE generic ones to avoid mismatches

// Proxy WebSocket for /api/v2/ws/telemetry as-is (frontend connects here by default)
const apiWsProxy = createProxyMiddleware({
  target: BACKEND_URL,
  changeOrigin: true,
  ws: false,
  xfwd: true,
})
app.use('/api/v2/ws/telemetry', apiWsProxy)

// Direct /api/v2 passthrough (must be before generic '/api')
app.use('/api/v2', (req, _res, next) => {
  console.log(`[proxy] match /api/v2 -> ${req.method} ${req.originalUrl}`)
  next()
})
app.use(
  '/api/v2',
  createProxyMiddleware({
    target: BACKEND_URL,
    changeOrigin: true,
    ws: false,
    xfwd: true,
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
    target: BACKEND_URL,
    changeOrigin: true,
    ws: false,
    xfwd: true,
    logLevel: 'debug',
    pathRewrite: (path) => {
      if (path.startsWith('/v2/')) return '/api' + path
      return '/api/v2' + path
    },
  })
)

// Proxy WebSocket path to backend telemetry endpoint
const legacyWsProxy = createProxyMiddleware({
  target: BACKEND_URL,
  changeOrigin: true,
  ws: false,
  xfwd: true,
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

const server = app.listen(PORT, '0.0.0.0', () => {
  console.log(`LawnBerry frontend is running at http://0.0.0.0:${PORT}`)
  console.log(`Proxying API to ${BACKEND_URL}`)
})

// The one 'upgrade' listener: discard client-supplied forwarding headers, then hand the upgrade to
// the proxy that owns the path. Anything else is not a WebSocket endpoint of this app.
server.on('upgrade', (req, socket, head) => {
  stripForwardingHeaders(req)
  const url = req.url || ''
  if (url.startsWith('/api/v2/ws/')) {
    apiWsProxy.upgrade(req, socket, head)
  } else if (url === '/ws' || /^\/ws[/?]/.test(url)) {
    legacyWsProxy.upgrade(req, socket, head)
  } else {
    socket.destroy()
  }
})
