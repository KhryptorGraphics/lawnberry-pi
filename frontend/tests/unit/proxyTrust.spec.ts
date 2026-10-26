// @vitest-environment node
// frontend/proxy-trust.mjs decides which forwarders may name a client address for server.mjs. It is
// the JavaScript twin of backend/src/core/peer_address.py (tests/unit/test_peer_address_trust.py):
// the same declaration grammar, the same strictness, the same shapes. It also holds the socket
// policy of the UI server (SocketGuard) and the E-stop classification that policy needs, the twin
// of backend/src/core/connection_guard.py.
import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest'
import { EventEmitter } from 'node:events'
// @ts-expect-error plain ES module without type declarations
import * as trust from '../../proxy-trust.mjs'

const parse = (raw: string | undefined) => trust.parseTrustedProxies(raw)
const texts = (raw: string) => parse(raw).proxies.map((p: { text: string }) => p.text)

describe('UI_TRUSTED_PROXIES parsing', () => {
  it('trusts nobody by default', () => {
    for (const raw of [undefined, '', '   ', ',,']) {
      expect(parse(raw)).toEqual({ proxies: [], problems: [] })
    }
    expect(trust.describeTrustedProxies([])).toBe('none')
  })

  it('accepts hosts and ordinary networks with an explicit shape', () => {
    expect(texts('127.0.0.3=last')).toEqual(['127.0.0.3=last'])
    expect(texts('127.0.0.4')).toEqual(['127.0.0.4=single']) // the strict shape is the default
    expect(texts(' 127.0.0.3 = LAST , 10.1.2.0/24=single,')).toEqual([
      '127.0.0.3=last',
      '10.1.2.0/24=single',
    ])
    expect(texts('::1')).toEqual(['::1=single'])
    expect(texts('2001:db8:1:2::/64=last')).toEqual(['2001:db8:1:2::/64=last'])
  })

  it.each([
    ['192.168.1.64/24', 'host bits'],
    ['10.1.2.3/8', 'host bits'],
    ['fd00::1/64', 'host bits'],
    ['0.0.0.0/0', 'matches every address'],
    ['::/0', 'matches every address'],
    ['10.0.0.0/8', 'more than one LAN'],
    ['127.0.0.0/8', 'more than one LAN'],
    ['fe80::/10', 'more than one LAN'],
    ['nonsense', 'not an IP address'],
    ['localhost', 'not an IP address'],
    ['127.0.0.2/33', 'not an IP address'],
    ['127.0.0.2=weird', 'unknown shape'],
    ['fe80::1%eth0', 'zone id'],
  ])('refuses %s (%s) and trusts nobody', (raw, needle) => {
    const { proxies, problems } = parse(raw)
    expect(proxies).toEqual([])
    expect(problems.join(' ')).toContain(needle)
  })

  it('lets one bad entry spoil the whole list', () => {
    const { proxies, problems } = parse('127.0.0.3=last, 0.0.0.0/0, 192.168.1.64/24')
    expect(proxies).toEqual([])
    expect(problems).toHaveLength(2)
  })
})

describe('client resolution', () => {
  const proxies = parse('127.0.0.3=last,127.0.0.4=single,127.0.0.5=last').proxies
  const resolve = (peer: string | undefined, header: string | undefined) =>
    trust.resolveClient(peer, header, proxies).address

  it.each([
    // last: the hop the forwarder appended; nothing to its left is read
    ['127.0.0.3', '203.0.113.9', '203.0.113.9'],
    ['127.0.0.3', '6.6.6.6, 203.0.113.9', '203.0.113.9'],
    ['127.0.0.3', '6.6.6.6, 203.0.113.9, 127.0.0.5', '203.0.113.9'], // a chain: skip forwarders
    ['127.0.0.3', '203.0.113.9:51234', '203.0.113.9'],
    ['127.0.0.3', '[2001:db8::7]:51234', '2001:db8::7'],
    ['127.0.0.3', '203.0.113.9, unknown', '127.0.0.3'],
    ['127.0.0.3', '127.0.0.5', '127.0.0.3'],
    ['127.0.0.3', undefined, '127.0.0.3'],
    // single: exactly one valid hop
    ['127.0.0.4', '203.0.113.9', '203.0.113.9'],
    ['127.0.0.4', '::ffff:203.0.113.9', '203.0.113.9'],
    ['127.0.0.4', '6.6.6.6, 203.0.113.9', '127.0.0.4'],
    ['127.0.0.4', 'garbage', '127.0.0.4'],
    ['127.0.0.4', undefined, '127.0.0.4'],
    // not listed: the header is never read (an ssh -R / socat on the same host included)
    ['127.0.0.1', '203.0.113.9', '127.0.0.1'],
    ['::1', '203.0.113.9', '::1'],
    ['198.51.100.9', '127.0.0.3', '198.51.100.9'],
    ['::ffff:127.0.0.4', '203.0.113.9', '203.0.113.9'], // a dual-stack socket reports mapped v4
    [undefined, '203.0.113.9', ''],
  ])('peer %s, header %s -> %s', (peer, header, expected) => {
    expect(resolve(peer, header)).toBe(expected)
  })

  it('says why a header that does not fit the declared shape is ignored', () => {
    expect(trust.resolveClient('127.0.0.4', '6.6.6.6, 203.0.113.9', proxies).warning).toContain(
      "declared 'single'"
    )
    expect(trust.resolveClient('127.0.0.3', 'x, junk', proxies).warning).toContain("declared 'last'")
    expect(trust.resolveClient('127.0.0.3', '203.0.113.9', proxies).warning).toBeUndefined()
  })

  it('cannot be made to pick an identity by a header transparent loopback forwarder', () => {
    const none = parse('').proxies
    const seen = new Set(
      Array.from({ length: 12 }, (_, n) =>
        trust.resolveClient('127.0.0.1', `198.51.100.${n}`, none).address
      )
    )
    expect([...seen]).toEqual(['127.0.0.1'])
  })
})

describe('isEmergencyStopRequest', () => {
  // The spellings a client can use: the proxies send /api/<x> and /api/v2/<x> to the same route.
  const SPELLINGS = [
    '/api/v2/control/emergency',
    '/api/control/emergency',
    '/api/v2/control/emergency-stop',
    '/api/control/emergency-stop',
    '/api/v2/tractor/emergency-stop',
    '/api/tractor/emergency-stop',
  ]
  const variants = (path: string) => [
    path,
    `${path}/`,
    `${path}///`,
    `${path}?x=1`,
    `${path}/?x=1&next=/a/b`,
  ]

  it.each(SPELLINGS.flatMap(variants))('POST %s is an E-stop', (url) => {
    expect(trust.isEmergencyStopRequest('POST', url)).toBe(true)
  })

  it.each(['GET', 'HEAD', 'PUT', 'PATCH', 'DELETE', 'OPTIONS', 'post'])(
    'a %s to the E-stop paths is an ordinary request: it answers 405 and stops nothing',
    (method) => {
      for (const url of SPELLINGS.flatMap(variants)) {
        expect(trust.isEmergencyStopRequest(method, url), `${method} ${url}`).toBe(false)
      }
    }
  )

  it.each([
    // the authenticated neighbours of the E-stop routes must stay ordinary requests
    '/api/v2/control/emergency_clear',
    '/api/control/emergency_clear',
    '/api/v2/tractor/clear-emergency',
    '/api/v2/control/emergency-stop-all',
    '/api/v2/control/emergency/stop',
    '/api/v2/control/emergencyx',
    '/api/v2/tractor/emergency',
    '/api/tractor/emergency',
    '/api/v2/control',
    // other roots, other versions, other spellings: nothing is decoded, folded or normalised
    '/api/v2/v2/control/emergency',
    '/api/v1/control/emergency',
    '/x/api/v2/control/emergency',
    '/api/v2//control/emergency',
    '//api/v2/control/emergency',
    '/API/V2/control/emergency',
    '/api/v2/control/%65mergency',
    '/api/v2/control/emergency%2F',
    '/api/v2/control/emergency%00',
    '/api/v2/control/./emergency',
    'http://mower.local/api/v2/control/emergency',
    '/ws/control/emergency',
    '/api/v2/ws/telemetry',
    '/control/emergency',
    '/emergency',
    '/',
    '*',
    '',
  ])('POST %s is an ordinary request', (url) => {
    expect(trust.isEmergencyStopRequest('POST', url)).toBe(false)
  })

  it('is not fooled by an E-stop path in the query string', () => {
    expect(trust.isEmergencyStopRequest('POST', '/api/v2/status?next=/api/v2/control/emergency')).toBe(
      false
    )
  })
})

describe('backendPathFor: what the proxies send the backend', () => {
  it.each([
    ['/api/v2/control/emergency', '/api/v2/control/emergency'],
    ['/api/control/emergency', '/api/v2/control/emergency'],
    ['/api/v2/status?x=1', '/api/v2/status'],
    ['/api/status', '/api/v2/status'],
    ['/api/v2', '/api/v2/'],
    ['/api', '/api/v2/'],
    ['/api/v2/', '/api/v2/'],
    ['/api/v2/ws/telemetry', '/api/v2/ws/telemetry'],
  ])('%s -> %s', (url, expected) => {
    expect(trust.backendPathFor(url)).toBe(expected)
  })

  it.each(['/', '/index.html', '/apix', '/api2/status', '/ws/telemetry', '/assets/api/v2/x', ''])(
    '%s is not forwarded to the backend',
    (url) => {
      expect(trust.backendPathFor(url)).toBeNull()
    }
  )
})

// proxy-trust.mjs ships no type declarations: this is the part of SocketGuard the tests use
interface Ticket {
  socket: FakeSocket
  estop: boolean
  promoted: boolean
}

interface Snapshot {
  connections: number
  productive: number
  evictable: number
  evicted: number
  refused: number
  shed: number
  overQuota: number
  deadlineClosed: number
}

interface GuardLike {
  max: number
  reserve: number
  softCap: number
  live: Set<FakeSocket>
  evicted: number
  refused: number
  register(socket: FakeSocket): FakeSocket[]
  begin(socket: FakeSocket, estop?: boolean): Ticket | null
  received(ticket: Ticket): boolean
  end(ticket: Ticket): void
  upgrade(socket: FakeSocket): boolean
  isProductive(socket: FakeSocket): boolean
  snapshot(): Snapshot
}

class FakeSocket extends EventEmitter {
  destroyed = false
  destroy() {
    if (this.destroyed) return
    this.destroyed = true
    this.emit('close')
  }
}

const newGuard = (max: number, reserve?: number, headerDeadlineMs?: number): GuardLike =>
  new trust.SocketGuard(max, reserve, { headerDeadlineMs })

const open = (guard: GuardLike, n: number) => {
  const sockets = Array.from({ length: n }, () => new FakeSocket())
  sockets.forEach((s) => guard.register(s))
  return sockets
}

/** A request on `socket` whose headers are in (its body may still be arriving). */
const begun = (guard: GuardLike, socket: FakeSocket, estop = false): Ticket => {
  const ticket = guard.begin(socket, estop)
  if (ticket === null) throw new Error('the request was shed')
  return ticket
}

/** A request on `socket` that has been completely received and is not answered yet. */
const serving = (guard: GuardLike, socket: FakeSocket, estop = false): Ticket => {
  const ticket = begun(guard, socket, estop)
  guard.received(ticket)
  return ticket
}

describe('SocketGuard: who is evicted', () => {
  it('destroys the socket that has been idle the longest to make room for a new one', () => {
    const guard = newGuard(3)
    const [a, b, c] = open(guard, 3)
    const d = new FakeSocket()
    expect(guard.register(d)).toEqual([a])
    expect([a.destroyed, b.destroyed, c.destroyed, d.destroyed]).toEqual([true, false, false, false])
    expect(guard.live.size).toBe(3)
    expect(guard.evicted).toBe(1)
  })

  it('stays within its cap under a flood of idle sockets', () => {
    const guard = newGuard(20)
    open(guard, 500)
    expect(guard.live.size).toBe(20)
    expect(guard.evicted).toBe(480)
  })

  // Changed from "never evicts a socket that is answering a request": a request whose headers are
  // in but whose body is still arriving is the shape of the slow-body attack (SEC3-03), so it is
  // NOT protected. Only a completely received request is (next describe).
  it('does not protect a socket whose request is still arriving, nor move it up the queue', () => {
    const guard = newGuard(3)
    const [a, b, c] = open(guard, 3)
    expect(guard.begin(a)).not.toBeNull() // headers are in, the body is not
    const d = new FakeSocket()
    expect(guard.register(d)).toEqual([a]) // still the one that has been evictable the longest
    expect([b.destroyed, c.destroyed]).toEqual([false, false])
  })

  // Changed from the old "begin; end": the policy protects a socket only from the moment its
  // request is completely received, so the socket has to be served for it to come back as newest.
  it('makes a keep-alive socket a candidate again once its response is finished, as the newest', () => {
    const guard = newGuard(3)
    const [a, b, c] = open(guard, 3)
    guard.end(serving(guard, a)) // evictable again, but more recently active than b and c
    const d = new FakeSocket()
    expect(guard.register(d)).toEqual([b])
    const e = new FakeSocket()
    expect(guard.register(e)).toEqual([c])
    expect(a.destroyed).toBe(false)
  })

  it('orders by the time a socket became evictable, not by when it connected', () => {
    const guard = newGuard(3)
    const [a, b, c] = open(guard, 3)
    const served = serving(guard, a) // a is protected: the queue is b, c
    const d = new FakeSocket()
    expect(guard.register(d)).toEqual([b]) // c, d
    guard.end(served) // a is evictable again and the newest: c, d, a
    const e = new FakeSocket()
    const f = new FakeSocket()
    expect(guard.register(e)).toEqual([c])
    expect(guard.register(f)).toEqual([d])
    expect(a.destroyed).toBe(false) // it connected first and became evictable last
  })

  it('leaves a socket where it was when its request is answered before it was received', () => {
    const guard = newGuard(3)
    const [a, b, c] = open(guard, 3)
    const early = begun(guard, a) // a 404 or 413 for a POST whose body is still arriving
    guard.end(early)
    const d = new FakeSocket()
    expect(guard.register(d)).toEqual([a]) // it never was protected, so it did not become recent
    expect([b.destroyed, c.destroyed]).toEqual([false, false])
  })
})

describe('SocketGuard: productive sockets', () => {
  it('never evicts a socket whose request is completely received, until its response is done', () => {
    const guard = newGuard(4, 1)
    const [a, b, c, d] = open(guard, 4)
    const servingA = serving(guard, a)
    serving(guard, b)
    const e = new FakeSocket()
    const f = new FakeSocket()
    expect(guard.register(e)).toEqual([c])
    expect(guard.register(f)).toEqual([d])
    expect([a.destroyed, b.destroyed]).toEqual([false, false])
    guard.end(servingA) // now a can go: the queue is e, f, a
    const g = new FakeSocket()
    expect(guard.register(g)).toEqual([e])
    expect(b.destroyed).toBe(false)
  })

  it('refuses the newcomer only when every other socket is productive', () => {
    const guard = newGuard(3, 1) // E-stop requests may exceed the quota of max - reserve
    const [a, b, c] = open(guard, 3)
    serving(guard, a, true)
    serving(guard, b, true)
    const servingC = serving(guard, c, true)
    const d = new FakeSocket()
    expect(guard.register(d)).toEqual([d]) // nothing to evict: refused
    expect(d.destroyed).toBe(true)
    expect(guard.snapshot()).toMatchObject({ refused: 1, evicted: 0, connections: 3 })
    guard.end(servingC) // one evictable socket is enough to admit the newcomer
    const e = new FakeSocket()
    expect(guard.register(e)).toEqual([c])
    expect(e.destroyed).toBe(false)
    expect(guard.snapshot()).toMatchObject({ refused: 1, evicted: 1 })
  })

  it('counts pipelined requests: the socket is evictable again after the last response', () => {
    const guard = newGuard(3)
    const [a] = open(guard, 1)
    const first = serving(guard, a)
    const second = serving(guard, a) // on a socket that is productive already
    expect(guard.snapshot().productive).toBe(1)
    guard.end(first)
    expect(guard.isProductive(a)).toBe(true) // one request still in flight
    guard.end(second)
    expect(guard.isProductive(a)).toBe(false)
    guard.end(second) // idempotent
    expect(guard.snapshot()).toMatchObject({ productive: 0, evictable: 1 })
  })

  it('does not protect a request whose response is finished already', () => {
    const guard = newGuard(3)
    const [a] = open(guard, 1)
    const ticket = begun(guard, a)
    guard.end(ticket) // answered early; the rest of its body is still read off the wire
    expect(guard.received(ticket)).toBe(false) // ... and when that ends nothing may pin the slot
    expect(guard.isProductive(a)).toBe(false)
    expect(guard.snapshot()).toMatchObject({ productive: 0, evictable: 1 })
  })

  it('forgets a closed socket and ignores what is reported about it afterwards', () => {
    const guard = newGuard(16)
    const [a, b] = open(guard, 2)
    const served = serving(guard, a)
    const receiving = begun(guard, b) // its body is still arriving when the socket goes away
    a.destroy()
    b.destroy()
    expect(guard.snapshot()).toMatchObject({ connections: 0, productive: 0, evictable: 0 })
    guard.end(served)
    expect(guard.received(receiving)).toBe(false) // a late 'end' cannot resurrect a dead socket
    expect(guard.begin(a)).not.toBeNull() // nor is a late 'request' for it an error
    expect(guard.snapshot()).toMatchObject({ connections: 0, productive: 0, evictable: 0 })
  })
})

describe('SocketGuard: the reserve for the E-stop', () => {
  it('derives the reserve and the quota from the cap', () => {
    expect(trust.defaultConnectionReserve(16)).toBe(4)
    expect(trust.defaultConnectionReserve(100)).toBe(25)
    expect(trust.defaultConnectionReserve(512)).toBe(128)
    expect(trust.defaultConnectionReserve(3)).toBe(2) // always below the cap
    expect(trust.defaultConnectionReserve(2)).toBe(1)
    const guard = newGuard(16)
    expect([guard.max, guard.reserve, guard.softCap]).toEqual([16, 4, 12])
    const custom = newGuard(16, 6)
    expect([custom.reserve, custom.softCap]).toEqual([6, 10])
  })

  it.each([
    [1, undefined],
    [16.5, undefined],
    [16, 0],
    [16, 16],
    [16, 17],
    [16, 3.5],
  ])('refuses max %s with reserve %s', (max, reserve) => {
    expect(() => newGuard(max, reserve)).toThrow(RangeError)
  })

  it('answers a request that is not an E-stop with 503 once max - reserve sockets are productive', () => {
    const guard = newGuard(16, 4)
    const sockets = open(guard, 16)
    sockets.slice(0, 12).forEach((s) => serving(guard, s))
    expect(guard.snapshot().productive).toBe(12)
    expect(guard.begin(sockets[12])).toBeNull()
    expect(guard.snapshot().shed).toBe(1)
    const stop = guard.begin(sockets[13], true)
    expect(stop).not.toBeNull() // the E-stop is never refused by the reserve
    expect(guard.received(stop as Ticket)).toBe(true)
    expect(guard.snapshot()).toMatchObject({ productive: 13, shed: 1 })
  })

  it('does not shed a request on a socket that is productive already', () => {
    const guard = newGuard(16, 4)
    const sockets = open(guard, 16)
    sockets.slice(0, 12).forEach((s) => serving(guard, s))
    expect(guard.begin(sockets[0])).not.toBeNull() // a second request on a counted socket
    expect(guard.snapshot().shed).toBe(0)
  })

  it('does not protect a request that completes after the quota filled in the meantime', () => {
    const guard = newGuard(16, 4)
    const sockets = open(guard, 16)
    const tickets = sockets.map((s) => begun(guard, s)) // all admitted while nothing is productive
    expect(tickets.every(Boolean)).toBe(true)
    const protectedNow = tickets.map((t) => guard.received(t))
    expect(protectedNow.filter(Boolean)).toHaveLength(12)
    expect(protectedNow.slice(12)).toEqual([false, false, false, false])
    expect(guard.received(tickets[12])).toBe(false) // asking again changes nothing
    expect(guard.snapshot()).toMatchObject({ productive: 12, evictable: 4, overQuota: 4 })
    const newcomer = new FakeSocket()
    expect(guard.register(newcomer)).toEqual([sockets[12]]) // the unprotected one is evictable
  })

  it('keeps room for the E-stop under a flood of requests whose responses never finish', () => {
    const guard = newGuard(16, 4)
    for (let i = 0; i < 200; i += 1) {
      const socket = new FakeSocket()
      guard.register(socket)
      if (socket.destroyed) continue
      const ticket = guard.begin(socket)
      if (ticket === null) {
        socket.destroy() // the 503 closes it
        continue
      }
      guard.received(ticket)
    }
    expect(guard.snapshot()).toMatchObject({ productive: 12, connections: 12, shed: 188 })
    // the free slots: the E-stop is admitted and, once received, protected like any other
    const stop = new FakeSocket()
    expect(guard.register(stop)).toEqual([])
    expect(guard.received(begun(guard, stop, true))).toBe(true)
    // the slots taken by silent sockets: the E-stop evicts one of them, never a productive one
    const junk = open(guard, 3)
    expect(guard.snapshot().connections).toBe(16)
    const second = new FakeSocket()
    expect(guard.register(second)).toEqual([junk[0]])
    expect(guard.received(begun(guard, second, true))).toBe(true)
    expect(guard.snapshot().refused).toBe(0)
  })

  it('makes an upgraded WebSocket productive for good, and refuses one over the quota', () => {
    const guard = newGuard(16, 4)
    const sockets = open(guard, 16)
    sockets.slice(0, 12).forEach((s) => expect(guard.upgrade(s)).toBe(true))
    expect(guard.upgrade(sockets[12])).toBe(false)
    expect(guard.snapshot()).toMatchObject({ productive: 12, shed: 1 })
    const newcomer = new FakeSocket()
    expect(guard.register(newcomer)).toEqual([sockets[12]]) // never a WebSocket
    expect(guard.upgrade(sockets[0])).toBe(true) // idempotent for a socket that is up already
    expect(guard.snapshot().productive).toBe(12)
  })

  it('leaves no trace of an upgrade asked for by a socket that is gone', () => {
    const guard = newGuard(16, 4)
    const [a] = open(guard, 1)
    a.destroy()

    expect(guard.upgrade(a)).toBe(false)

    // a ghost that counted as productive for good would hold a slot of the quota for ever
    expect(guard.snapshot()).toMatchObject({ connections: 0, productive: 0, evictable: 0, shed: 0 })
  })
})

describe('SocketGuard: the header deadline', () => {
  beforeEach(() => vi.useFakeTimers())
  afterEach(() => vi.useRealTimers())

  it('destroys a socket that has not delivered the headers of a request in time', () => {
    const guard = newGuard(16, 4, 10_000)
    const [a] = open(guard, 1)
    vi.advanceTimersByTime(9_999)
    expect(a.destroyed).toBe(false)
    vi.advanceTimersByTime(1)
    expect(a.destroyed).toBe(true)
    expect(guard.snapshot()).toMatchObject({ deadlineClosed: 1, connections: 0 })
  })

  it('stops waiting once the headers of a request are complete', () => {
    const guard = newGuard(16, 4, 10_000)
    const [a] = open(guard, 1)
    vi.advanceTimersByTime(5_000)
    guard.begin(a)
    vi.advanceTimersByTime(3_600_000)
    expect(a.destroyed).toBe(false)
  })

  it('waits again after a response: the second request of a keep-alive socket has a deadline too', () => {
    const guard = newGuard(16, 4, 10_000)
    const [a] = open(guard, 1)
    vi.advanceTimersByTime(5_000)
    guard.end(serving(guard, a)) // the first request is answered 5 s after the connect
    vi.advanceTimersByTime(9_999)
    expect(a.destroyed).toBe(false)
    vi.advanceTimersByTime(1)
    expect(a.destroyed).toBe(true) // 10 s after the response, not 10 s after the connect
    expect(guard.snapshot().deadlineClosed).toBe(1)
  })

  it('does not wait again while another request is in flight on the socket', () => {
    const guard = newGuard(16, 4, 10_000)
    const [a] = open(guard, 1)
    const first = serving(guard, a)
    const second = serving(guard, a) // pipelined
    guard.end(first)
    vi.advanceTimersByTime(3_600_000)
    expect(a.destroyed).toBe(false)
    guard.end(second)
    vi.advanceTimersByTime(10_000)
    expect(a.destroyed).toBe(true)
  })

  it('starts the wait over for each response, not once for the socket', () => {
    const guard = newGuard(16, 4, 10_000)
    const [a] = open(guard, 1)
    for (let i = 0; i < 3; i += 1) {
      vi.advanceTimersByTime(8_000)
      expect(a.destroyed).toBe(false)
      guard.end(serving(guard, a))
    }
    vi.advanceTimersByTime(10_000)
    expect(a.destroyed).toBe(true)
  })

  it('leaves an upgraded WebSocket alone for good', () => {
    const guard = newGuard(16, 4, 10_000)
    const [a] = open(guard, 1)
    expect(guard.upgrade(a)).toBe(true)
    vi.advanceTimersByTime(3_600_000)
    expect(a.destroyed).toBe(false)
  })

  it('has no timer left for a socket that closed', () => {
    const guard = newGuard(16, 4, 10_000)
    const [a, b] = open(guard, 2)
    a.destroy()
    guard.end(serving(guard, b))
    b.destroy()
    expect(vi.getTimerCount()).toBe(0)
    expect(guard.snapshot().deadlineClosed).toBe(0)
  })

  it('does not time a socket out when no deadline is configured', () => {
    const guard = newGuard(16, 4)
    const [a] = open(guard, 1)
    vi.advanceTimersByTime(3_600_000)
    expect(a.destroyed).toBe(false)
  })
})
