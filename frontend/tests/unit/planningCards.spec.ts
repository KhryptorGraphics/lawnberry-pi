import { describe, it, expect, beforeEach, vi } from 'vitest'
import { mount, flushPromises, type DOMWrapper } from '@vue/test-utils'
import { setActivePinia, createPinia } from 'pinia'

// Shared mock handles — hoisted so the vi.mock factories (also hoisted) can
// reference them. api.get rejects so the seeded jobs/schedules survive
// onMounted's refresh (mirrors the no-backend case); post/put/delete resolve.
const { apiMock, routerPush, mapConfig } = vi.hoisted(() => ({
  apiMock: {
    get: vi.fn(),
    post: vi.fn(),
    put: vi.fn(),
    delete: vi.fn(),
  },
  routerPush: vi.fn(),
  mapConfig: { current: null as Record<string, unknown> | null },
}))

vi.mock('@/services/api', () => ({
  useApiService: () => apiMock,
  startAutonomous: vi.fn().mockResolvedValue({}),
  default: apiMock,
}))

vi.mock('@/services/websocket', () => ({
  useWebSocket: () => ({
    connect: vi.fn(async () => { /* no-op */ }),
    subscribe: vi.fn(),
  }),
}))

vi.mock('vue-router', () => ({
  useRouter: () => ({ push: routerPush }),
}))

// Controllable map store — the zones computed reads mapStore.configuration.
vi.mock('@/stores/map', () => ({
  useMapStore: () => ({
    get configuration() { return mapConfig.current },
    loadConfiguration: vi.fn(async () => { /* no-op */ }),
  }),
}))

import PlanningView from '@/views/PlanningView.vue'
import JobsCard from '@/components/planning/JobsCard.vue'
import SchedulesCard from '@/components/planning/SchedulesCard.vue'
import ZonesCard from '@/components/planning/ZonesCard.vue'
import PatternsCard from '@/components/planning/PatternsCard.vue'

type Wrapper = ReturnType<typeof mount>

const mountView = async (): Promise<Wrapper> => {
  const pinia = createPinia()
  setActivePinia(pinia)
  const wrapper = mount(PlanningView, { global: { plugins: [pinia] } })
  await flushPromises()
  return wrapper
}

const clickButtonByText = async (wrapper: Wrapper, text: string) => {
  const btn = wrapper.findAll('button').find((b) => b.text().includes(text))
  if (!btn) throw new Error(`button containing "${text}" not found`)
  await btn.trigger('click')
  await flushPromises()
}

// Tab labels overlap quick-action labels (e.g. "Zones" vs "Manage Zones"),
// so switch tabs via the tab bar specifically.
const clickTab = async (wrapper: Wrapper, label: string) => {
  const tab = wrapper.findAll('.tab-button').find((b) => b.text().trim() === label)
  if (!tab) throw new Error(`tab "${label}" not found`)
  await tab.trigger('click')
  await flushPromises()
}

describe('PlanningView card split', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    apiMock.get.mockRejectedValue(new Error('no backend'))
    apiMock.post.mockResolvedValue({ data: {} })
    apiMock.put.mockResolvedValue({ data: {} })
    apiMock.delete.mockResolvedValue({ data: {} })
    mapConfig.current = null
  })

  it('renders the Current Jobs tab with seeded jobs via JobsCard', async () => {
    const wrapper = await mountView()
    const jobs = wrapper.findComponent(JobsCard)
    expect(jobs.exists()).toBe(true)
    expect(jobs.text()).toContain('Front Lawn Weekly')
    expect(jobs.text()).toContain('Back Lawn Maintenance')
    // Other tab cards are not mounted until their tab is active.
    expect(wrapper.findComponent(ZonesCard).exists()).toBe(false)
    wrapper.unmount()
  })

  it('starting a scheduled job emits up and posts to the start endpoint', async () => {
    const wrapper = await mountView()
    // Seeded job id '2' (Back Lawn Maintenance) is the only "scheduled" one.
    await clickButtonByText(wrapper, 'Start')
    expect(apiMock.post).toHaveBeenCalledWith('/api/v2/planning/jobs/2/start')
    wrapper.unmount()
  })

  it('switching to the Zones tab renders map zones via ZonesCard', async () => {
    mapConfig.current = {
      mowing_zones: [
        {
          id: 'z1',
          name: 'Front Lawn',
          zone_type: 'mowing',
          priority: 9,
          polygon: [
            { latitude: 0, longitude: 0 },
            { latitude: 0, longitude: 0.001 },
            { latitude: 0.001, longitude: 0.001 },
          ],
        },
      ],
    }
    const wrapper = await mountView()
    await clickTab(wrapper, 'Zones')
    const zonesCard = wrapper.findComponent(ZonesCard)
    expect(zonesCard.exists()).toBe(true)
    expect(zonesCard.text()).toContain('Front Lawn')
    wrapper.unmount()
  })

  it('Patterns tab: clicking a pattern updates the v-model selection', async () => {
    const wrapper = await mountView()
    await clickTab(wrapper, 'Patterns')
    const patterns = wrapper.findComponent(PatternsCard)
    expect(patterns.exists()).toBe(true)
    // Default selection is 'parallel'; click the Spiral card.
    const spiral = patterns.findAll('.pattern-card').find((c: DOMWrapper<Element>) => c.text().includes('Spiral'))
    if (!spiral) throw new Error('Spiral pattern card not found')
    await spiral.trigger('click')
    await flushPromises()
    expect(spiral.classes()).toContain('selected')
    wrapper.unmount()
  })

  it('Scheduling tab renders seeded schedules via SchedulesCard', async () => {
    const wrapper = await mountView()
    await clickTab(wrapper, 'Scheduling')
    const schedules = wrapper.findComponent(SchedulesCard)
    expect(schedules.exists()).toBe(true)
    expect(schedules.text()).toContain('Weekly Full Property')
    wrapper.unmount()
  })
})
