// Who may tell the UI server (frontend/server.mjs) the address of a client, and how; and which
// sockets the UI server keeps open (SocketGuard, at the bottom) so the E-stop can always connect.
//
// server.mjs sends the backend ONE X-Forwarded-For hop: the address of the client it resolved.
// The backend (backend/src/core/peer_address.py, LAWN_BERRY_TRUSTED_PROXIES) believes that hop
// only from this server's own source address, so the rate limits and the login lockout count each
// browser separately. This module decides what that address is. It is the same model as the
// backend's, written once more in JavaScript on purpose (keep the two in step):
//
//   UI_TRUSTED_PROXIES = comma-separated ADDRESS[/PREFIX][=SHAPE], default NOBODY.
//
//   single  the forwarder REPLACES X-Forwarded-For with exactly one hop (nginx with
//           `proxy_set_header X-Forwarded-For $remote_addr`)
//   last    the forwarder APPENDS the address it saw (nginx `$proxy_add_x_forwarded_for`, a
//           Cloudflare or ngrok tunnel edge): only the LAST hop, the one it wrote, is believed;
//           trailing hops that are themselves listed forwarders (a chain) are skipped
//
// From any peer that is not listed the header is never read: the client is the socket address.
// An entry names a forwarder HOST: networks wider than one LAN (IPv4 shorter than /24, IPv6
// shorter than /64), /0, host bits set (192.168.1.64/24), unknown shapes and non-addresses are
// errors, and one bad entry makes the whole list empty (fail closed, never a widened list).
//
// Why this matters: all of 127.0.0.0/8 is the loopback and any process on the host can use any
// address in it, so "127.0.0.1" cannot tell a tunnel agent that appends the client's address from
// an `ssh -R` or `socat` that passes a client-written header through. List only forwarders that
// append (or replace) the header, and only the address THEY connect from.
//
// The last third of the file is the socket policy of the UI server (SocketGuard) with what it needs
// to know about the E-stop: the three backend E-stop endpoints and the rewrite the proxies apply
// to a request path. It is the JavaScript twin of backend/src/core/connection_guard.py (keep the
// two in step).

import net from 'node:net'

export const SHAPE_SINGLE = 'single'
export const SHAPE_LAST = 'last'
const SHAPES = [SHAPE_SINGLE, SHAPE_LAST]
const WIDEST_PREFIX = { 4: 24, 6: 64 }

function ipv4ToBigInt(text) {
  const parts = text.split('.')
  if (parts.length !== 4) return null
  let value = 0n
  for (const part of parts) {
    if (!/^\d{1,3}$/.test(part) || Number(part) > 255) return null
    value = (value << 8n) | BigInt(Number(part))
  }
  return value
}

function bigIntToIpv4(value) {
  return [24n, 16n, 8n, 0n].map((shift) => String((value >> shift) & 255n)).join('.')
}

function ipv6ToBigInt(text) {
  let rest = text.toLowerCase()
  const lastColon = rest.lastIndexOf(':')
  if (rest.slice(lastColon + 1).includes('.')) {
    // an embedded IPv4 tail (::ffff:1.2.3.4) is two more 16-bit groups
    const v4 = ipv4ToBigInt(rest.slice(lastColon + 1))
    if (v4 === null) return null
    rest = `${rest.slice(0, lastColon + 1)}${(v4 >> 16n).toString(16)}:${(v4 & 0xffffn).toString(16)}`
  }
  const halves = rest.split('::')
  if (halves.length > 2) return null
  const head = halves[0] === '' ? [] : halves[0].split(':')
  const back = halves.length === 2 && halves[1] !== '' ? halves[1].split(':') : []
  let groups = head
  if (halves.length === 2) {
    const fill = 8 - head.length - back.length
    if (fill < 1) return null
    groups = [...head, ...Array(fill).fill('0'), ...back]
  }
  if (groups.length !== 8) return null
  let value = 0n
  for (const group of groups) {
    if (!/^[0-9a-f]{1,4}$/.test(group)) return null
    value = (value << 16n) | BigInt(parseInt(group, 16))
  }
  return value
}

/** The numeric value of an IP address text of the given family (4 or 6), or null. */
function toBigInt(text, family) {
  return family === 4 ? ipv4ToBigInt(text) : ipv6ToBigInt(text)
}

/**
 * The canonical address text of a peer or forwarded hop (`a.b.c.d`, `a.b.c.d:port`, `[v6]:port`;
 * a zone id and an IPv4-mapped IPv6 address are accepted, the latter counts as its IPv4
 * address), or null when it is not an address.
 */
export function normalizeAddress(text) {
  if (typeof text !== 'string') return null
  let value = text.trim()
  if (value.startsWith('[')) {
    const end = value.indexOf(']')
    if (end === -1) return null
    value = value.slice(1, end)
  } else if (value.split(':').length === 2) {
    value = value.split(':')[0]
  }
  value = value.split('%')[0]
  const family = net.isIP(value)
  if (family === 0) return null
  if (family === 6) {
    const number = ipv6ToBigInt(value)
    if (number !== null && number >> 32n === 0xffffn) return bigIntToIpv4(number & 0xffffffffn)
    return value.toLowerCase()
  }
  return value
}

function parseEntry(entry) {
  const quoted = JSON.stringify(entry)
  const text = entry.trim()
  const eq = text.lastIndexOf('=')
  const networkText = (eq === -1 ? text : text.slice(0, eq)).trim()
  const shape = eq === -1 ? SHAPE_SINGLE : text.slice(eq + 1).trim().toLowerCase()
  if (!SHAPES.includes(shape)) {
    return { problem: `${quoted} has the unknown shape ${JSON.stringify(shape)} (use one of: ${SHAPES.join(', ')})` }
  }
  if (networkText.includes('%')) {
    return { problem: `${quoted} has an IPv6 zone id: list the address without it` }
  }
  const slash = networkText.indexOf('/')
  const addressText = slash === -1 ? networkText : networkText.slice(0, slash)
  const family = net.isIP(addressText)
  const notNetwork = { problem: `${quoted} is not an IP address or CIDR network` }
  if (family === 0) return notNetwork
  const bits = family === 4 ? 32 : 128
  let prefix = bits
  if (slash !== -1) {
    const prefixText = networkText.slice(slash + 1)
    if (!/^\d{1,3}$/.test(prefixText) || Number(prefixText) > bits) return notNetwork
    prefix = Number(prefixText)
  }
  const value = toBigInt(addressText, family)
  if (value === null) return notNetwork
  const hostMask = (1n << BigInt(bits - prefix)) - 1n
  if ((value & hostMask) !== 0n) {
    const widened = family === 4 ? ` ${bigIntToIpv4(value & ~hostMask)}/${prefix}` : ''
    return {
      problem:
        `${quoted} has host bits set, so it would silently become the whole network${widened}: ` +
        `write the network address, or the one host`,
    }
  }
  if (prefix === 0) {
    return { problem: `${quoted} matches every address: name the forwarder's own address` }
  }
  if (prefix < WIDEST_PREFIX[family]) {
    return {
      problem:
        `${quoted} names more than one LAN: a forwarder is a host, list its own address ` +
        `(a repo nginx connects from 127.0.0.3)`,
    }
  }
  const list = new net.BlockList()
  const kind = family === 4 ? 'ipv4' : 'ipv6'
  list.addSubnet(addressText, prefix, kind)
  return { proxy: { list, family: kind, shape, text: `${networkText}=${shape}` } }
}

/**
 * Parse UI_TRUSTED_PROXIES strictly. Returns `{ proxies, problems }`; when `problems` is not
 * empty `proxies` is empty too: one bad entry spoils the whole list (never a widened one).
 */
export function parseTrustedProxies(raw) {
  const proxies = []
  const problems = []
  for (const item of String(raw ?? '').split(',')) {
    if (!item.trim()) continue
    const { proxy, problem } = parseEntry(item)
    if (problem) problems.push(problem)
    else proxies.push(proxy)
  }
  return problems.length ? { proxies: [], problems } : { proxies, problems }
}

export function describeTrustedProxies(proxies) {
  return proxies.map((proxy) => proxy.text).join(', ') || 'none'
}

function matchProxy(address, proxies) {
  const family = net.isIPv4(address) ? 'ipv4' : 'ipv6'
  return proxies.find((proxy) => proxy.family === family && proxy.list.check(address, family)) ?? null
}

/**
 * The client of a request: `{ address, forwarded, warning }`. `peerRaw` is the socket's remote
 * address, `forwardedFor` the request's X-Forwarded-For (Node joins repeated header lines with
 * ", "). The header is read only when the peer is a listed forwarder, in the shape declared for
 * it; anything that does not fit is never guessed at (the forwarder's own address is returned and
 * `warning` says why).
 */
export function resolveClient(peerRaw, forwardedFor, proxies) {
  const peer = normalizeAddress(peerRaw)
  if (peer === null) return { address: peerRaw ?? '', forwarded: false }
  const proxy = matchProxy(peer, proxies)
  if (proxy === null) return { address: peer, forwarded: false }
  const hops = String(forwardedFor ?? '')
    .split(',')
    .map((hop) => hop.trim())
    .filter(Boolean)
  if (hops.length === 0) return { address: peer, forwarded: false }
  if (proxy.shape === SHAPE_SINGLE) {
    const address = hops.length === 1 ? normalizeAddress(hops[0]) : null
    if (address === null) {
      return {
        address: peer,
        forwarded: false,
        warning:
          `forwarder ${peer} is declared 'single' (one X-Forwarded-For hop) but sent ` +
          `${hops.length} hop(s) that do not fit: ignoring the header`,
      }
    }
    return { address, forwarded: true }
  }
  for (let i = hops.length - 1; i >= 0; i -= 1) {
    const address = normalizeAddress(hops[i])
    if (address === null) {
      return {
        address: peer,
        forwarded: false,
        warning:
          `forwarder ${peer} is declared 'last' but its last X-Forwarded-For hop is not an ` +
          `address: ignoring the header`,
      }
    }
    if (matchProxy(address, proxies) === null) return { address, forwarded: true }
  }
  return { address: peer, forwarded: false }
}

/** Can this process bind a socket to `address` (an alias of the loopback on Linux)? */
export function canBind(address) {
  return new Promise((resolve) => {
    const probe = net.createServer()
    probe.once('error', () => resolve(false))
    probe.listen(0, address, () => probe.close(() => resolve(true)))
  })
}

// ---- the E-stop, as the UI server has to recognise it -----------------------------------------

/**
 * The three E-stop endpoints of the backend (backend/src/middleware/emergency_paths.py: keep the
 * two in step). Only a POST to one of them stops the mower.
 */
export const BACKEND_EMERGENCY_STOP_PATHS = Object.freeze([
  '/api/v2/control/emergency',
  '/api/v2/control/emergency-stop',
  '/api/v2/tractor/emergency-stop',
])
const EMERGENCY_STOP_SET = new Set(BACKEND_EMERGENCY_STOP_PATHS)

// The two proxies of server.mjs. Express strips the mount prefix before a proxy sees the path
// ('/<rest>', query included); these are the rewrites they apply to it. They are defined once:
// server.mjs registers these very functions and `backendPathFor` replays them, so what is judged
// an E-stop below cannot drift from what the proxies send the backend.

/** The `/api/v2` mount: the path reaches the backend as it is. */
export function rewriteApiV2Path(strippedPath) {
  return '/api/v2' + strippedPath
}

/** The `/api` mount: '/v2/...' keeps its version, anything else is a v2 route. */
export function rewriteApiPath(strippedPath) {
  return strippedPath.startsWith('/v2/') ? '/api' + strippedPath : '/api/v2' + strippedPath
}

// In the order server.mjs registers them (the first mount that matches serves the request).
const API_MOUNTS = [
  ['/api/v2', rewriteApiV2Path],
  ['/api', rewriteApiPath],
]

/**
 * The path (query dropped) the proxies of server.mjs send the backend for the request target
 * `url`, or null when no proxy forwards it (static files, the SPA, the legacy /ws paths).
 */
export function backendPathFor(url) {
  const queryAt = url.indexOf('?')
  const pathname = queryAt === -1 ? url : url.slice(0, queryAt)
  for (const [mount, rewrite] of API_MOUNTS) {
    if (pathname === mount || pathname.startsWith(mount + '/')) {
      return rewrite(pathname.slice(mount.length) || '/')
    }
  }
  return null
}

/**
 * Is this request an E-stop: a POST that, after the rewrite the proxies apply, addresses one of the
 * three E-stop endpoints of the backend? The query and trailing slashes are ignored, as the backend
 * ignores them. Judged on the method AND the path: a GET, HEAD or PUT to those paths answers 405 and
 * stops nothing, so it earns no priority.
 *
 * Only the exact spellings count: nothing is percent-decoded, case-folded or de-duplicated. The
 * proxy itself normalises a few more spellings onto an E-stop route (a doubled slash, another case
 * in the `/api` or `/api/v2` prefix, a #fragment, a backslash): they still stop the mower, but they
 * are ordinary requests to the guard (they can be shed, and they get no priority). The reverse
 * never happens: nothing that is not an E-stop for the backend is called one, so nobody can claim
 * the reserve with a spelling of their own (serverProxy.spec.ts checks both against the real proxy).
 */
export function isEmergencyStopRequest(method, url) {
  if (method !== 'POST' || typeof url !== 'string') return false
  const path = backendPathFor(url)
  if (path === null) return false
  let end = path.length
  while (end > 0 && path.charCodeAt(end - 1) === 47) end -= 1 // the backend does rstrip('/')
  return EMERGENCY_STOP_SET.has(path.slice(0, end))
}

// ---- the socket table ------------------------------------------------------------------------

/**
 * The reserve for the E-stop when nothing says otherwise: a quarter of the cap, at least 4 and
 * always below the cap (the backend's LAWN_BERRY_CONNECTION_RESERVE default is the same quarter).
 */
export function defaultConnectionReserve(max) {
  return Math.min(Math.max(4, Math.floor(max / 4)), max - 1)
}

// setTimeout fires at once (with a warning) for a longer delay.
const MAX_TIMER_MS = 2 ** 31 - 1

/**
 * Bounded, E-stop-first admission of sockets (descriptor-exhaustion defence).
 *
 * A socket that connects and sends nothing, sends a request one byte at a time or announces a body
 * it never delivers holds a descriptor for as long as it likes, and with every descriptor held the
 * E-stop POST cannot even connect. The policy that keeps room for it (O(1) per event):
 *
 * - A socket is PRODUCTIVE only from the moment its request is completely received (headers and
 *   body) until its response is finished: it is doing work and is never evicted. A socket whose
 *   request is still arriving (RECEIVING) and one that waits for a request (IDLE: it never spoke,
 *   or it is a keep-alive socket between requests) are evictable. A request that is still arriving
 *   is deliberately not protected, E-stop requests included: otherwise `POST .../emergency` with a
 *   Content-Length and no body, repeated, would pin every slot against the real E-stop. An upgraded
 *   WebSocket is productive for good (the backend caps their number).
 * - At the cap a NEW socket evicts the evictable socket that has been evictable the LONGEST (since
 *   it entered that state, whatever it did since: a legitimate client sends its request within a
 *   round trip, so the junk that has waited longest goes first). Only if every socket is
 *   productive is the newcomer refused.
 * - `reserve` slots are kept for the E-stop. A request that is not an E-stop may make its socket
 *   productive only while fewer than `softCap = max - reserve` sockets are productive; the HTTP
 *   layer answers it 503 itself instead (`begin` returns null). So at least `reserve` slots are
 *   always free or evictable, and the E-stop, which the reserve never refuses, can take one.
 * - The header deadline (`headerDeadlineMs`, 0 = none) is armed when a socket is registered and
 *   again whenever a response finishes and nothing else is in flight on the socket: one that has
 *   not delivered the headers of its next request by then is destroyed, however slowly the bytes
 *   dribble (a total deadline, not an inactivity timeout). The deadline for a whole request is
 *   Node's own `server.requestTimeout`.
 *
 * A request on a socket: `begin(socket, estop)` when its headers are complete (a ticket, or null
 * when it must be shed), `received(ticket)` when it has been completely received, `end(ticket)`
 * when its response is finished or the request is gone. `upgrade(socket)` for a WebSocket.
 * Counters: `evicted`, `refused`, `shed`, `overQuota`, `deadlineClosed` (see `snapshot`).
 */
export class SocketGuard {
  constructor(max, reserve = defaultConnectionReserve(max), { headerDeadlineMs = 0 } = {}) {
    if (!Number.isInteger(max) || max < 2) {
      throw new RangeError(`max must be an integer >= 2, got ${max}`)
    }
    if (!Number.isInteger(reserve) || reserve < 1 || reserve >= max) {
      throw new RangeError(`reserve must be an integer in [1, max), got ${reserve} (max ${max})`)
    }
    if (!Number.isFinite(headerDeadlineMs) || headerDeadlineMs < 0) {
      throw new RangeError(`headerDeadlineMs must be a number >= 0, got ${headerDeadlineMs}`)
    }
    this.max = max
    this.reserve = reserve
    this.softCap = max - reserve
    this.headerDeadlineMs = Math.min(headerDeadlineMs, MAX_TIMER_MS)
    this.live = new Set() // every open socket
    this.evictable = new Set() // the socket that has been evictable the longest first
    this.productive = new Map() // socket -> completely received requests in flight (Infinity: WS)
    this.requests = new Map() // socket -> requests begun and not ended
    this.timers = new Map() // socket -> its header deadline
    this.evicted = 0
    this.refused = 0
    this.shed = 0
    this.overQuota = 0
    this.deadlineClosed = 0
  }

  /** A new connection. Returns the sockets this destroyed (the new one too, when refused). */
  register(socket) {
    this.live.add(socket)
    this.evictable.add(socket)
    socket.once('close', () => this.forget(socket))
    this.armHeaderDeadline(socket)
    const destroyed = []
    while (this.live.size > this.max) {
      const victim = this.oldestEvictable(socket)
      if (victim === null) {
        this.refused += 1
        destroyed.push(socket)
        this.forget(socket)
        socket.destroy()
        break
      }
      this.evicted += 1
      destroyed.push(victim)
      this.forget(victim)
      victim.destroy()
    }
    return destroyed
  }

  forget(socket) {
    this.live.delete(socket)
    this.evictable.delete(socket)
    this.productive.delete(socket)
    this.requests.delete(socket)
    this.disarm(socket)
  }

  oldestEvictable(except) {
    for (const socket of this.evictable) if (socket !== except) return socket
    return null
  }

  /**
   * The headers of a request are complete on `socket`. Returns its ticket, or null when the request
   * must be shed (answered 503, not served): it is not an E-stop, its socket is not productive yet
   * and `softCap` sockets already are.
   */
  begin(socket, estop = false) {
    if (this.live.has(socket)) {
      if (!estop && !this.productive.has(socket) && this.productive.size >= this.softCap) {
        this.shed += 1
        return null
      }
      this.disarm(socket)
      this.requests.set(socket, (this.requests.get(socket) ?? 0) + 1)
    }
    return { socket, estop, received: false, promoted: false, done: false }
  }

  /**
   * The request of `ticket` has been completely received: its socket becomes productive and
   * `true` is returned, unless the request is not an E-stop and `softCap` sockets already are
   * productive. It then stays evictable (counted in `overQuota`) and the answer is `false`.
   */
  received(ticket) {
    const { socket } = ticket
    if (ticket.received) return ticket.promoted
    ticket.received = true
    if (ticket.done || !this.live.has(socket)) return false
    if (!this.productive.has(socket)) {
      if (!ticket.estop && this.productive.size >= this.softCap) {
        this.overQuota += 1
        return false
      }
      this.evictable.delete(socket)
    }
    this.productive.set(socket, (this.productive.get(socket) ?? 0) + 1)
    ticket.promoted = true
    return true
  }

  /** The response of `ticket` is finished, or its request is gone. Idempotent. */
  end(ticket) {
    if (ticket.done) return
    ticket.done = true
    const { socket } = ticket
    if (!this.live.has(socket)) return
    const requests = (this.requests.get(socket) ?? 1) - 1
    if (requests > 0) this.requests.set(socket, requests)
    else this.requests.delete(socket)
    if (ticket.promoted) {
      const promoted = (this.productive.get(socket) ?? 1) - 1
      if (promoted > 0) {
        this.productive.set(socket, promoted)
      } else {
        this.productive.delete(socket)
        this.evictable.add(socket) // evictable again, and the most recent one to be
      }
    }
    if (requests === 0 && !this.productive.has(socket)) this.armHeaderDeadline(socket)
  }

  /**
   * `socket` asks to become a WebSocket: productive for good and out of reach of the deadlines,
   * but subject to the reserve like every request that is not an E-stop. Returns false (counted
   * in `shed`) when `softCap` sockets are productive already: the caller must refuse the upgrade.
   */
  upgrade(socket) {
    if (!this.live.has(socket)) return false
    if (!this.productive.has(socket)) {
      if (this.productive.size >= this.softCap) {
        this.shed += 1
        return false
      }
      this.evictable.delete(socket)
    }
    this.productive.set(socket, Infinity)
    this.disarm(socket)
    return true
  }

  isProductive(socket) {
    return this.productive.has(socket)
  }

  /** Counters and gauges, plain values. */
  snapshot() {
    return {
      max: this.max,
      reserve: this.reserve,
      softCap: this.softCap,
      connections: this.live.size,
      productive: this.productive.size,
      evictable: this.evictable.size,
      evicted: this.evicted,
      refused: this.refused,
      shed: this.shed,
      overQuota: this.overQuota,
      deadlineClosed: this.deadlineClosed,
    }
  }

  armHeaderDeadline(socket) {
    if (this.headerDeadlineMs === 0) return
    this.disarm(socket)
    const timer = setTimeout(() => {
      this.timers.delete(socket)
      this.deadlineClosed += 1
      socket.destroy()
    }, this.headerDeadlineMs)
    timer.unref?.()
    this.timers.set(socket, timer)
  }

  disarm(socket) {
    const timer = this.timers.get(socket)
    if (timer === undefined) return
    clearTimeout(timer)
    this.timers.delete(socket)
  }
}
