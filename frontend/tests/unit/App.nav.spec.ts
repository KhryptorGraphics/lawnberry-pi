// Pins the fail-open platform-gating contract: an operator must never lose
// their one control link (mower's /control or the tractor's /tractor) to a
// slow or failed platform-detection fetch. See useSystemStore.fetchPlatform()
// and App.vue's nav v-if guards.
import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest'
import { mount, flushPromises, type VueWrapper } from '@vue/test-utils'
import { createRouter, createMemoryHistory } from 'vue-router'
import { setActivePinia, createPinia } from 'pinia'
import App from '@/App.vue'
import { useSystemStore } from '@/stores/system'
import * as api from '@/services/api'

vi.mock('@/stores/auth', () => ({
  useAuthStore: () => ({
    user: null,
    token: null,
    validateSession: vi.fn(),
    updateActivity: vi.fn(),
  }),
}))

vi.mock('@/stores/preferences', () => ({
  usePreferencesStore: () => ({
    ensureInitialized: vi.fn(),
    syncWithServer: vi.fn().mockResolvedValue(undefined),
  }),
}))

const NAV_PATHS = [
  '/', '/control', '/tractor', '/maps', '/planning', '/mission-planner',
  '/ai', '/telemetry', '/rtk', '/docs', '/settings', '/login',
]

function makeRouter() {
  return createRouter({
    history: createMemoryHistory(),
    routes: NAV_PATHS.map((path) => ({ path, component: { template: '<div/>' } })),
  })
}

async function mountApp() {
  const router = makeRouter()
  const wrapper = mount(App, {
    global: {
      plugins: [router],
      stubs: {
        UserMenu: true,
        ToastHost: true,
        ConfirmDialog: true,
        CommandPalette: true,
        TopProgress: true,
      },
    },
  })
  await router.isReady()
  await flushPromises()
  return wrapper
}

function hasLink(wrapper: VueWrapper, to: string) {
  return wrapper.findAll('a.nav-link').some((a) => a.attributes('href') === to)
}

describe('App nav fail-open platform gating', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
    vi.clearAllMocks()
  })

  afterEach(() => {
    const instances = (globalThis as any).__wsMockInstances as Array<any>
    if (instances) instances.length = 0
  })

  it('shows both Control and Tractor links while the boot fetch is still in flight', async () => {
    vi.mocked(api.getTractorState).mockReturnValue(new Promise(() => {
      /* never resolves during this test */
    }))

    const wrapper = await mountApp()

    expect(hasLink(wrapper, '/control')).toBe(true)
    expect(hasLink(wrapper, '/tractor')).toBe(true)
  })

  it('shows both links when the platform fetch rejects, never defaulting to mower', async () => {
    vi.mocked(api.getTractorState).mockRejectedValue(new Error('network down'))

    const wrapper = await mountApp()

    expect(useSystemStore().platform).toBe('unknown')
    expect(hasLink(wrapper, '/control')).toBe(true)
    expect(hasLink(wrapper, '/tractor')).toBe(true)
  })

  it('narrows to only Control once the platform resolves to mower', async () => {
    vi.mocked(api.getTractorState).mockResolvedValue({ enabled: false } as any)

    const wrapper = await mountApp()

    expect(hasLink(wrapper, '/control')).toBe(true)
    expect(hasLink(wrapper, '/tractor')).toBe(false)
  })

  it('narrows to only Tractor once the platform resolves to tractor', async () => {
    vi.mocked(api.getTractorState).mockResolvedValue({ enabled: true } as any)

    const wrapper = await mountApp()

    expect(hasLink(wrapper, '/control')).toBe(false)
    expect(hasLink(wrapper, '/tractor')).toBe(true)
  })
})
