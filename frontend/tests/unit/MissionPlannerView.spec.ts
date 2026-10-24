import { flushPromises, mount } from '@vue/test-utils';
import type { VueWrapper } from '@vue/test-utils';
import { createPinia, setActivePinia } from 'pinia';
import MissionPlannerView from '@/views/MissionPlannerView.vue';
import { useMissionStore } from '@/stores/mission';
import { useMapStore } from '@/stores/map';
import apiService from '@/services/api';
import { useToastStore } from '@/stores/toast';
import { SIGN_IN_REQUIRED_MESSAGE } from '@/utils/apiErrors';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import type { MockInstance } from 'vitest';

const mapPaneStub = { _leaflet_pos: { x: 0, y: 0 } }

const layerGroupInstance = {
  addTo: vi.fn(() => layerGroupInstance),
  clearLayers: vi.fn(),
  removeLayer: vi.fn(),
}

const tileLayerInstance = {
  addTo: vi.fn(() => tileLayerInstance),
}

const markerInstance = {
  addTo: vi.fn(() => markerInstance),
  on: vi.fn(),
  setLatLng: vi.fn(),
}

const polylineInstance = {
  addTo: vi.fn(() => polylineInstance),
}

const circleInstance = {
  addTo: vi.fn(() => circleInstance),
}

const circleMarkerInstance = {
  addTo: vi.fn(() => circleMarkerInstance),
}

const mapInstance = {
  setView: vi.fn(),
  on: vi.fn((event: string, handler: () => void) => {
    if (event === 'load') {
      handler()
    }
  }),
  off: vi.fn(),
  remove: vi.fn(),
  eachLayer: vi.fn(),
  whenReady: vi.fn((handler: () => void) => handler()),
  getZoom: vi.fn(() => 18),
  _mapPane: mapPaneStub,
}

// Mock Leaflet with required stubs
vi.mock('leaflet', () => ({
  default: {
    map: vi.fn(() => mapInstance),
    layerGroup: vi.fn(() => layerGroupInstance),
    tileLayer: vi.fn(() => tileLayerInstance),
    marker: vi.fn(() => markerInstance),
    polyline: vi.fn(() => polylineInstance),
    circle: vi.fn(() => circleInstance),
    circleMarker: vi.fn(() => circleMarkerInstance),
    divIcon: vi.fn(() => ({})),
    Icon: {
      Default: {
        mergeOptions: vi.fn(),
      },
    },
  },
}));

vi.mock('@/components/MissionWaypointList.vue', () => ({
  default: {
    name: 'MissionWaypointList',
    template: '<div class="mission-waypoint-list" />',
  },
}))

const mockedApi = apiService as unknown as {
  get: ReturnType<typeof vi.fn>
  post: ReturnType<typeof vi.fn>
}

function primeMapStore() {
  const mapStore = useMapStore()
  mapStore.configuration = {
    config_id: 'default',
    config_version: 1,
    provider: 'osm',
    provider_metadata: {},
    boundary_zone: null,
    exclusion_zones: [],
    mowing_zones: [],
    markers: [],
    center_point: null,
    zoom_level: 18,
    map_rotation_deg: 0,
    validation_errors: [],
    last_modified: '2025-10-25T15:28:33.451Z',
    created_at: '2025-10-25T15:28:33.451Z',
  } as any
  mapStore.loadConfiguration = vi.fn().mockResolvedValue(undefined)
}

describe('MissionPlannerView.vue', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
    vi.clearAllMocks()
    primeMapStore()

    mockedApi.get.mockImplementation((url: string) => {
      if (url.startsWith('/api/v2/settings/maps')) {
        return Promise.resolve({ data: { provider: 'osm', style: 'standard', google_api_key: '' } })
      }
      if (url.startsWith('/api/v2/map/configuration')) {
        return Promise.resolve({ data: { provider: 'osm', zones: [], markers: [], updated_at: '2025-10-25T15:28:33.451Z' } })
      }
      return Promise.resolve({ data: {} })
    })

    mockedApi.post.mockResolvedValue({ data: {} })
  })

  it('renders the mission planner view', async () => {
    const wrapper = mount(MissionPlannerView)
    await flushPromises()
    expect(wrapper.find('h1').text()).toBe('Mission Planner')
  })

  it('allows creating a mission', async () => {
    const wrapper = mount(MissionPlannerView)
    const missionStore = useMissionStore()
    missionStore.addWaypoint(51.5, -0.09)
    // createMission now resolves to a boolean (true = created); the view toasts on the result
    const createMissionSpy = vi.spyOn(missionStore, 'createMission').mockResolvedValue(true)

    await flushPromises()
    const nameInput = wrapper.find('input[placeholder="Mission Name"]')
    await nameInput.setValue('Test Mission')
    const createButton = wrapper.findAll('button').find(btn => btn.text() === 'Create Mission')!
    await createButton.trigger('click')

    expect(createMissionSpy).toHaveBeenCalledWith('Test Mission')
  })

  // POST /missions/create and /missions/{id}/start|pause|resume|abort need the operator Bearer: unsigned, they answer
  // 401. The planner used to toast a bare "Failed to ..." (and, for create, nothing at all).
  describe('operator-auth failures', () => {
    const unsigned = { response: { status: 401, data: { detail: 'Operator authentication required' } } }

    let consoleError: MockInstance

    beforeEach(() => {
      // start/resume arm a 2 s status poll; keep it from outliving the test
      vi.useFakeTimers()
      // the store logs every failed call (kept on purpose); not interesting here
      consoleError = vi.spyOn(console, 'error').mockImplementation(() => {})
    })

    afterEach(() => {
      consoleError.mockRestore()
      vi.clearAllTimers()
      vi.useRealTimers()
    })

    async function mountWithMission(status: string) {
      const wrapper = mount(MissionPlannerView)
      const missionStore = useMissionStore()
      missionStore.currentMission = { id: 'm-1', name: 'north lawn', waypoints: [], created_at: '2026-01-01T00:00:00Z' }
      missionStore.missionStatus = status
      await flushPromises()
      return { wrapper, missionStore }
    }

    function control(wrapper: VueWrapper, label: string) {
      return wrapper.findAll('button').find((b) => b.text() === label)!
    }

    function toasts() {
      return useToastStore().toasts.map((t) => [t.type, t.message])
    }

    it('Create Mission with a 401 toasts the sign-in instruction and leaves no mission behind', async () => {
      const wrapper = mount(MissionPlannerView)
      const missionStore = useMissionStore()
      missionStore.addWaypoint(51.5, -0.09)
      mockedApi.post.mockRejectedValueOnce(unsigned)
      await flushPromises()
      await wrapper.find('input[placeholder="Mission Name"]').setValue('North lawn')

      await control(wrapper, 'Create Mission').trigger('click')
      await flushPromises()

      expect(mockedApi.post).toHaveBeenCalledWith(
        '/api/v2/missions/create',
        expect.objectContaining({ name: 'North lawn' })
      )
      expect(toasts()).toEqual([['error', SIGN_IN_REQUIRED_MESSAGE]])
      expect(missionStore.currentMission).toBeNull()
      expect(missionStore.missionStatus).toBe('')
    })

    it('Create Mission that succeeds toasts "Mission created" and loads the mission', async () => {
      const wrapper = mount(MissionPlannerView)
      const missionStore = useMissionStore()
      missionStore.addWaypoint(51.5, -0.09)
      mockedApi.post.mockResolvedValueOnce({
        data: { id: 'm-9', name: 'North lawn', waypoints: [], created_at: '2026-01-01T00:00:00Z' },
      })
      await flushPromises()
      await wrapper.find('input[placeholder="Mission Name"]').setValue('North lawn')

      await control(wrapper, 'Create Mission').trigger('click')
      await flushPromises()

      expect(toasts()).toEqual([['success', 'Mission created']])
      expect(missionStore.currentMission?.id).toBe('m-9')
      expect(missionStore.missionStatus).toBe('idle')
    })

    it.each([
      ['Start Mission', 'idle', 'start'],
      ['Pause', 'running', 'pause'],
      ['Resume', 'paused', 'resume'],
      ['Abort', 'running', 'abort'],
    ])('%s with a 401 toasts the sign-in instruction and changes nothing', async (label, status, action) => {
      const { wrapper, missionStore } = await mountWithMission(status)
      mockedApi.post.mockRejectedValueOnce(unsigned)

      await control(wrapper, label).trigger('click')
      await flushPromises()

      expect(mockedApi.post).toHaveBeenCalledWith(`/api/v2/missions/m-1/${action}`, {})
      expect(toasts()).toEqual([['error', SIGN_IN_REQUIRED_MESSAGE]])
      expect(missionStore.missionStatus).toBe(status)
      expect(missionStore.currentMission?.id).toBe('m-1')
      // the busy guard is released, so the operator can retry once signed in
      expect(control(wrapper, label).attributes('disabled')).toBeUndefined()
    })

    it.each([
      ['Start Mission', 'Mission started', 'idle', 'running'],
      ['Pause', 'Mission paused', 'running', 'paused'],
      ['Resume', 'Mission resumed', 'paused', 'running'],
      ['Abort', 'Mission aborted', 'running', 'aborted'],
    ])('%s that succeeds toasts "%s" and moves the status on', async (label, message, before, after) => {
      const { wrapper, missionStore } = await mountWithMission(before)

      await control(wrapper, label).trigger('click')
      await flushPromises()

      expect(toasts()).toEqual([['success', message]])
      expect(missionStore.missionStatus).toBe(after)
    })

    it('shows the backend detail for any other refusal', async () => {
      const { wrapper } = await mountWithMission('idle')
      mockedApi.post.mockRejectedValueOnce({ response: { status: 409, data: { detail: 'Mission already running' } } })

      await control(wrapper, 'Start Mission').trigger('click')
      await flushPromises()

      expect(toasts()).toEqual([['error', 'Mission already running']])
    })
  })
})
