// @vitest-environment node
// frontend/proxy-trust.mjs decides which forwarders may name a client address for server.mjs. It is
// the JavaScript twin of backend/src/core/peer_address.py (tests/unit/test_peer_address_trust.py):
// the same declaration grammar, the same strictness, the same shapes.
import { describe, it, expect } from 'vitest'
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

// proxy-trust.mjs ships no type declarations: this is the part of SocketGuard the tests use
interface GuardLike {
  register(socket: FakeSocket): FakeSocket[]
}

class FakeSocket extends EventEmitter {
  destroyed = false
  destroy() {
    this.destroyed = true
    this.emit('close')
  }
}

describe('SocketGuard', () => {
  const open = (guard: GuardLike, n: number) => {
    const sockets = Array.from({ length: n }, () => new FakeSocket())
    sockets.forEach((s) => guard.register(s))
    return sockets
  }

  it('destroys the socket that has been idle the longest to make room for a new one', () => {
    const guard = new trust.SocketGuard(3)
    const [a, b, c] = open(guard, 3)
    const d = new FakeSocket()
    expect(guard.register(d)).toEqual([a])
    expect([a.destroyed, b.destroyed, c.destroyed, d.destroyed]).toEqual([true, false, false, false])
    expect(guard.live.size).toBe(3)
    expect(guard.evicted).toBe(1)
  })

  it('never evicts a socket that is answering a request', () => {
    const guard = new trust.SocketGuard(3)
    const [a, b, c] = open(guard, 3)
    guard.begin(a)
    const d = new FakeSocket()
    expect(guard.register(d)).toEqual([b])
    expect(a.destroyed).toBe(false)
    guard.begin(c)
    guard.begin(d)
    const e = new FakeSocket()
    expect(guard.register(e)).toEqual([e]) // everything else is busy: the newcomer is refused
    expect(e.destroyed).toBe(true)
    expect(guard.refused).toBe(1)
  })

  it('makes a keep-alive socket a candidate again once its request is done, as the newest', () => {
    const guard = new trust.SocketGuard(3)
    const [a, b, c] = open(guard, 3)
    guard.begin(a)
    guard.end(a) // idle again, but more recently active than b and c
    const d = new FakeSocket()
    expect(guard.register(d)).toEqual([b])
    const e = new FakeSocket()
    expect(guard.register(e)).toEqual([c])
    expect(a.destroyed).toBe(false)
  })

  it('counts pipelined requests and forgets closed sockets', () => {
    const guard = new trust.SocketGuard(5)
    const [a] = open(guard, 1)
    guard.begin(a)
    guard.begin(a)
    guard.end(a)
    expect(guard.idle.has(a)).toBe(false) // one request still in flight
    guard.end(a)
    expect(guard.idle.has(a)).toBe(true)
    a.destroy()
    expect(guard.live.size).toBe(0)
    expect(guard.idle.size).toBe(0)
    expect(guard.busy.size).toBe(0)
  })

  it('stays within its cap under a flood of idle sockets', () => {
    const guard = new trust.SocketGuard(20)
    open(guard, 500)
    expect(guard.live.size).toBe(20)
    expect(guard.evicted).toBe(480)
  })
})
