import { describe, it, expect, vi, beforeEach } from 'vitest'

// Escape the global @/services/api stub (tests/vitest.setup.ts) — this test
// drives the real ApiService (base URL, auth-header injection) against a
// mocked axios instance instead. Same pattern as weatherApi.spec.ts.
vi.mock('@/services/api', async (importOriginal) => importOriginal())

const authStoreState: { token: string | null } = { token: null }
vi.mock('@/stores/auth', () => ({
  useAuthStore: () => authStoreState,
}))

// vi.mock factories are hoisted above the rest of the file, so any state they
// need to write into must be created via vi.hoisted() rather than a plain
// module-scope `let` (which would still be in its TDZ at hoist time).
const captured = vi.hoisted(() => ({
  createConfig: null as any,
  requestInterceptor: undefined as ((config: any) => any) | undefined,
}))

vi.mock('axios', () => {
  const instance = {
    get: vi.fn(),
    post: vi.fn(),
    put: vi.fn(),
    delete: vi.fn(),
    patch: vi.fn(),
    interceptors: {
      request: {
        use: vi.fn((onFulfilled: (config: any) => any) => {
          captured.requestInterceptor = onFulfilled
        }),
      },
      response: { use: vi.fn(() => {}) },
    },
  }
  return {
    default: {
      create: vi.fn((config: any) => {
        captured.createConfig = config
        return instance
      }),
    },
  }
})

// Triggers module evaluation (ApiService construction) so the mocked
// axios.create()/interceptors.request.use() calls above get captured.
import apiClient from '@/services/api'

describe('consolidated API client', () => {
  beforeEach(() => {
    authStoreState.token = null
  })

  it('exists and uses a relative base URL so requests route through the frontend proxy', () => {
    expect(typeof apiClient.get).toBe('function')
    expect(captured.createConfig).toMatchObject({ baseURL: '' })
  })

  it('injects a Bearer auth header from the auth store token', () => {
    authStoreState.token = 'abc123'

    const config = captured.requestInterceptor!({ headers: {} })

    expect(config.headers.Authorization).toBe('Bearer abc123')
  })

  it('does not set an auth header when there is no token', () => {
    const config = captured.requestInterceptor!({ headers: {} })

    expect(config.headers.Authorization).toBeUndefined()
  })
})
