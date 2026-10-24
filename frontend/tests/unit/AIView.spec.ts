// POST /ai/enable, /ai/disable, /ai/model, /ai/metrics/reset and /ai/datasets/{id}/export need the operator Bearer
// and answer 401 until someone has signed in. AIView used to show the bare "Failed to ..." (reset metrics) or the
// raw backend detail (the others); a 401 now reads as the sign-in instruction. The network boundary is the global
// @/services/api stub (tests/vitest.setup.ts); the toast store is the real one.
import { describe, it, expect, beforeEach, afterEach } from 'vitest'
import { mount, flushPromises } from '@vue/test-utils'
import type { VueWrapper } from '@vue/test-utils'
import { setActivePinia, createPinia } from 'pinia'
import AIView from '@/views/AIView.vue'
import apiClient from '@/services/api'
import { useToastStore } from '@/stores/toast'
import { SIGN_IN_REQUIRED_MESSAGE } from '@/utils/apiErrors'

const MODEL_PATH = '/apps/lawnberry-pi/models/lawn_vla.hef'

function stubReads(enabled: boolean) {
  apiClient.get.mockImplementation((url: string) => {
    switch (url) {
      case '/api/v2/ai/status':
        return Promise.resolve({
          data: {
            enabled,
            mode: 'ai_autonomous',
            model_loaded: true,
            hailo_available: false,
            using_hardware: false,
            safety_engaged: false,
            success_rate: 1,
            avg_latency_ms: 10,
            current_fps: 5,
          },
        })
      case '/api/v2/ai/datasets':
        return Promise.resolve({ data: [{ id: 'ds-1', name: 'Lawn A', label_count: 12 }] })
      case '/api/v2/ai/health':
        return Promise.resolve({ data: { healthy: true, message: 'ok' } })
      default:
        return Promise.resolve({ data: {} })
    }
  })
}

let mounted: VueWrapper | null = null

async function mountView(enabled: boolean) {
  stubReads(enabled)
  mounted = mount(AIView)
  await flushPromises()
  await mounted.find('#model-path').setValue(MODEL_PATH)
  return mounted
}

function toasts() {
  return useToastStore().toasts.map((t) => [t.type, t.message])
}

function press(wrapper: VueWrapper, label: string) {
  return wrapper
    .findAll('button')
    .find((b) => b.text() === label)!
    .trigger('click')
}

// The exact call each button makes (URL and body are the ones AIView sends).
const ACTIONS = [
  {
    name: 'Enable AI',
    enabled: false,
    button: 'Enable AI',
    post: ['/api/v2/ai/enable', { mode: 'ai_autonomous' }],
    okResponse: { data: {} },
    okToast: 'AI autonomous control enabled',
    fallback: 'Failed to enable AI',
  },
  {
    name: 'Disable AI',
    enabled: true,
    button: 'Disable AI',
    post: ['/api/v2/ai/disable', {}],
    okResponse: { data: {} },
    okToast: 'AI control disabled',
    fallback: 'Failed to disable AI',
  },
  {
    name: 'Deploy model',
    enabled: false,
    button: 'Deploy',
    post: ['/api/v2/ai/model', { model_path: MODEL_PATH }],
    okResponse: { data: { success: true, model_name: 'lawn_vla', model_version: '1.2' } },
    okToast: 'Deployed lawn_vla v1.2',
    fallback: 'Failed to load model',
  },
  {
    name: 'Reset metrics',
    enabled: false,
    button: 'Reset',
    post: ['/api/v2/ai/metrics/reset', {}],
    okResponse: { data: {} },
    okToast: 'Metrics reset',
    fallback: 'Failed to reset metrics',
  },
  {
    name: 'Export dataset',
    enabled: false,
    button: 'Export',
    post: ['/api/v2/ai/datasets/ds-1/export', { format: 'YOLO' }],
    okResponse: { data: { format: 'YOLO' } },
    okToast: 'Export started for Lawn A (YOLO)',
    fallback: 'Failed to export Lawn A',
  },
]

describe('AIView actions that need the operator Bearer', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
    apiClient.post.mockResolvedValue({ data: {} })
  })

  afterEach(() => {
    mounted?.unmount()
    mounted = null
  })

  describe.each(ACTIONS)('$name', (action) => {
    it('a 401 toasts the sign-in instruction, and the button is usable again', async () => {
      const wrapper = await mountView(action.enabled)
      apiClient.post.mockRejectedValueOnce({
        response: { status: 401, data: { detail: 'Operator authentication required' } },
      })

      await press(wrapper, action.button)
      await flushPromises()

      expect(apiClient.post).toHaveBeenCalledWith(...action.post)
      expect(toasts()).toEqual([['error', SIGN_IN_REQUIRED_MESSAGE]])
      expect(wrapper.findAll('button').find((b) => b.text() === action.button)!.attributes('disabled')).toBeUndefined()
    })

    it('success keeps its own toast', async () => {
      const wrapper = await mountView(action.enabled)
      apiClient.post.mockResolvedValueOnce(action.okResponse)

      await press(wrapper, action.button)
      await flushPromises()

      expect(apiClient.post).toHaveBeenCalledWith(...action.post)
      expect(toasts()).toEqual([['success', action.okToast]])
    })

    it.each([
      ['a backend detail', { response: { status: 409, data: { detail: 'Hailo busy' } } }, () => 'Hailo busy'],
      ['a rate limit', { response: { status: 429, headers: { 'retry-after': '8' } } }, () => 'Too many attempts: try again in 8 s.'],
      ['no HTTP response', new Error('Network Error'), (f: string) => `${f}: no response from the backend.`],
      ['a bare HTTP error', { response: { status: 500 } }, (f: string) => f],
    ])('%s reads as an operator-facing message', async (_name, error, expected) => {
      const wrapper = await mountView(action.enabled)
      apiClient.post.mockRejectedValueOnce(error)

      await press(wrapper, action.button)
      await flushPromises()

      expect(toasts()).toEqual([['error', expected(action.fallback)]])
    })
  })

  it('a model load the backend declines (success: false) still shows its message, unchanged', async () => {
    const wrapper = await mountView(false)
    apiClient.post.mockResolvedValueOnce({ data: { success: false, message: 'HEF file not found' } })

    await press(wrapper, 'Deploy')
    await flushPromises()

    expect(toasts()).toEqual([['error', 'HEF file not found']])
  })
})
