import { describe, it, expect, vi, beforeEach } from 'vitest'
import { weatherApi } from '@/services/api'

// Escape the global @/services/api stub (tests/vitest.setup.ts) — this test drives the
// real weatherApi implementation against a mocked axios instance instead.
vi.mock('@/services/api', async (importOriginal) => importOriginal())

vi.mock('axios', () => {
  const get = vi.fn(async (url: string) => {
    if (url.includes('/weather/current')) {
      return {
        data: {
          temperature_c: 21.2,
          humidity_percent: 55,
          wind_speed_mps: 2.5,
          condition: 'clear',
          source: 'offline-default',
          ts: new Date().toISOString(),
        },
      }
    }
    return { data: {} }
  })
  const interceptors = {
    request: { use: vi.fn(() => {}) },
    response: { use: vi.fn(() => {}) },
  }
  const instance = { get, interceptors }
  return { default: { create: () => instance }, get }
})

describe('weatherApi', () => {
  beforeEach(() => {
    // Reset mocks between tests
    vi.clearAllMocks()
  })

  it('fetches current weather', async () => {
    const res = await weatherApi.getCurrent()
    expect(res.temperature_c).toBeTypeOf('number')
    expect(res.source).toBeDefined()
  })
})
