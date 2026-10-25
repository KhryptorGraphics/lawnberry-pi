// Who may tell the UI server (frontend/server.mjs) the address of a client, and how.
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

/**
 * Keeps sockets that do no work out of the way of new ones.
 *
 * A socket that connects and sends nothing holds a descriptor for ever (Node reaps it only when
 * `server.timeout` is set, and then only after that whole time). With every descriptor held, the
 * E-stop POST cannot even connect. The guard counts the open sockets; when more than `max` are
 * open, the one that has been idle the LONGEST (no request in flight: never served, or a
 * keep-alive connection waiting for its next request) is destroyed to make room, and the new
 * socket itself only if every other one is busy answering a request. `begin` / `end` bracket a
 * request on a socket. Counters: `evicted`, `refused`.
 */
export class SocketGuard {
  constructor(max) {
    this.max = max
    this.live = new Set()
    this.idle = new Set() // no request in flight; insertion order = longest idle first
    this.busy = new Map() // socket -> requests in flight
    this.evicted = 0
    this.refused = 0
  }

  /** A new connection. Returns the sockets this destroyed (the new one too, when refused). */
  register(socket) {
    this.live.add(socket)
    this.idle.add(socket)
    socket.once('close', () => this.forget(socket))
    const destroyed = []
    while (this.live.size > this.max) {
      const victim = this.longestIdle(socket)
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
    this.idle.delete(socket)
    this.busy.delete(socket)
  }

  longestIdle(except) {
    for (const socket of this.idle) if (socket !== except) return socket
    return null
  }

  /** A request began on `socket`: it is no longer a candidate. */
  begin(socket) {
    if (!this.live.has(socket)) return
    this.busy.set(socket, (this.busy.get(socket) ?? 0) + 1)
    this.idle.delete(socket)
  }

  /** The request on `socket` is finished: it is idle again (the most recently active). */
  end(socket) {
    if (!this.live.has(socket)) return
    const left = (this.busy.get(socket) ?? 1) - 1
    if (left > 0) {
      this.busy.set(socket, left)
      return
    }
    this.busy.delete(socket)
    this.idle.add(socket)
  }
}
