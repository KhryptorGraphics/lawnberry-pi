// Call-site proof for the operator-auth contract.
//
// The backend (OPERATOR_AUTH_REQUIRED=1) now wants `Authorization: Bearer <token from /auth/login>` on every
// actuating write. This drives the REAL ApiService (interceptors, base headers, auth store) over a recording
// network adapter and walks every gated call site the frontend has, the way the app calls it:
//   - signed in : the request carries `Authorization: Bearer tok-123` and `X-Client-Id`
//   - signed out: it carries no Authorization header, goes out exactly once (nothing to refresh, no replay) and the
//                 401 reaches the operator as the sign-in instruction (and never trips the safety lockout)
// The emergency stops are the opposite: they must work with no token at all. A static guard then fails if any
// source file writes to the API around the shared ApiService.
import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest'
import type { MockInstance } from 'vitest'
import { mount, flushPromises, RouterLinkStub } from '@vue/test-utils'
import type { VueWrapper } from '@vue/test-utils'
import { setActivePinia, createPinia } from 'pinia'
import { readdirSync, readFileSync } from 'node:fs'
import { join, relative } from 'node:path'
import { createRecordingAdapter, replaceNetworkAdapter } from './helpers/recordingAdapter'
import type { CannedResponse, RecordedRequest, Responder } from './helpers/recordingAdapter'

// Escape the global @/services/api stub (tests/vitest.setup.ts): the REAL ApiService is the point here.
vi.mock('@/services/api', async (importOriginal) => importOriginal())

import {
  controlApi,
  maintenanceApi,
  sendControlCommand,
  tractorAuthorize,
  tractorBlade,
  tractorClearEmergency,
  tractorEmergencyStop,
  tractorLeftLever,
  tractorRightLever,
  tractorStart,
  tractorStopEngine,
  tractorThrottle,
  useApiService,
} from '@/services/api'
import { useAuthStore } from '@/stores/auth'
import { useControlStore } from '@/stores/control'
import { useMissionStore } from '@/stores/mission'
import { useToastStore } from '@/stores/toast'
import AIView from '@/views/AIView.vue'
import ControlView from '@/views/ControlView.vue'
import { SIGN_IN_REQUIRED_MESSAGE, describeApiError } from '@/utils/apiErrors'

// ---------------------------------------------------------------------------------------------------------------
// Harness
// ---------------------------------------------------------------------------------------------------------------

const TOKEN = 'tok-123'
const UNSIGNED: CannedResponse = { status: 401, data: { detail: 'Operator authentication required' } }
const MISSION = { id: 'm-1', name: 'north lawn', waypoints: [], created_at: '2026-01-01T00:00:00Z' }

let restoreAdapter: (() => void) | null = null
let thrown: unknown = null
let aiEnabled = false
let consoleError: MockInstance | null = null
const cleanups: Array<() => void> = []

function install(responder: Responder): RecordedRequest[] {
  const { adapter, requests } = createRecordingAdapter(responder)
  restoreAdapter = replaceNetworkAdapter(useApiService(), adapter)
  return requests
}

/** What the backend answers to the reads the mounted views make on their own. */
function readResponse(url: string): CannedResponse {
  switch (url) {
    case '/api/v2/ai/status':
      return {
        data: {
          enabled: aiEnabled,
          mode: 'ai_autonomous',
          model_loaded: true,
          hailo_available: false,
          using_hardware: false,
          safety_engaged: false,
          success_rate: 1,
          avg_latency_ms: 10,
          current_fps: 5,
        },
      }
    case '/api/v2/ai/datasets':
      return { data: [{ id: 'ds-1', name: 'Lawn A', label_count: 12 }] }
    case '/api/v2/settings/security':
      return { data: { security_level: 'password', session_timeout_minutes: 15 } }
    case '/api/v2/hardware/robohat':
      return { data: { telemetry_source: 'simulated', safety_state: 'nominal', serial_connected: true } }
    default:
      return { data: {} }
  }
}

/** A backend that has a valid session for the Bearer: every write succeeds. */
function signedInResponder(request: RecordedRequest): CannedResponse {
  if (request.method === 'GET') return readResponse(request.url)
  if (request.url === '/api/v2/control/manual-unlock') {
    // ControlView first tries a silent unlock without a password (mount), then the real one
    const typed = (request.body as { password?: string } | undefined)?.password
    return typed
      ? { data: { authorized: true, session_id: 's-1', source: 'password' } }
      : { status: 400, data: { detail: 'Password required for manual unlock' } }
  }
  if (request.url === '/api/v2/missions/create') return { data: MISSION }
  if (request.url === '/api/v2/ai/model') return { data: { success: true, model_name: 'lawn_vla', model_version: '1.2' } }
  return { data: {} }
}

/** A backend enforcing operator auth against a client with no token: every write is a 401. */
function signedOutResponder(request: RecordedRequest): CannedResponse {
  return request.method === 'GET' ? readResponse(request.url) : UNSIGNED
}

function toastTexts(): string[] {
  return useToastStore().toasts.map((toast) => toast.message)
}

function press(wrapper: VueWrapper, label: string) {
  return wrapper
    .findAll('button')
    .find((button) => button.text() === label)!
    .trigger('click')
}

function lastCommandReason(): string {
  const result = useControlStore().lastCommandResult
  return result && 'status_reason' in result && typeof result.status_reason === 'string' ? result.status_reason : ''
}

beforeEach(() => {
  setActivePinia(createPinia())
  // the mission store polls and refreshToken() arms a timer; none may outlive a test
  vi.useFakeTimers()
  // the mission store logs every failed call on purpose; the 401 runs below would bury the output in axios dumps
  consoleError = vi.spyOn(console, 'error').mockImplementation(() => {})
  thrown = null
  aiEnabled = false
})

afterEach(() => {
  for (const cleanup of cleanups.splice(0)) cleanup()
  restoreAdapter?.()
  restoreAdapter = null
  consoleError?.mockRestore()
  vi.clearAllTimers()
  vi.useRealTimers()
})

// ---------------------------------------------------------------------------------------------------------------
// Every gated call site
// ---------------------------------------------------------------------------------------------------------------

interface CallSite {
  name: string
  method: 'POST'
  url: string
  /** The body the call site sends, when it is worth pinning. */
  body?: unknown
  /** Requests to this URL one run legitimately makes (ControlView tries a silent unlock on mount, then unlocks). */
  attempts?: number
  /** Performs the call the way the app does; resolves once it and the UI's reaction to it have settled. */
  run: () => Promise<void>
  /** What the operator was told about the failure `run` just hit. */
  failureText: () => string
}

/** A call made straight through the exported API: the error that reaches the caller is what gets described. */
function apiSite(name: string, url: string, call: () => Promise<unknown>, body?: unknown): CallSite {
  return {
    name,
    method: 'POST',
    url,
    body,
    run: async () => {
      try {
        await call()
      } catch (error) {
        thrown = error
      }
    },
    failureText: () => describeApiError(thrown, 'Failed'),
  }
}

function controlStoreSite(command: 'drive' | 'blade', url: string, payload: Record<string, unknown>): CallSite {
  return {
    name: `control store: ${command}`,
    method: 'POST',
    url,
    body: payload,
    run: async () => {
      await useControlStore()
        .submitCommand(command, payload)
        .catch(() => undefined)
    },
    failureText: lastCommandReason,
  }
}

function missionSite(name: string, url: string, act: () => Promise<unknown>): CallSite {
  return {
    name: `mission store: ${name}`,
    method: 'POST',
    url,
    run: async () => {
      const store = useMissionStore()
      store.addWaypoint(51.5, -0.09)
      store.currentMission = { ...MISSION }
      await act()
    },
    failureText: () => useMissionStore().lastError,
  }
}

function aiSite(
  name: string,
  button: string,
  url: string,
  body: unknown,
  { enabled = false, modelPath = '' }: { enabled?: boolean; modelPath?: string } = {}
): CallSite {
  return {
    name: `AIView: ${name}`,
    method: 'POST',
    url,
    body,
    run: async () => {
      aiEnabled = enabled
      const wrapper = mount(AIView)
      cleanups.push(() => wrapper.unmount())
      await flushPromises()
      if (modelPath) await wrapper.find('#model-path').setValue(modelPath)
      await press(wrapper, button)
      await flushPromises()
    },
    failureText: () => toastTexts()[0] ?? '',
  }
}

// ControlView.vue's own bindings, reached the way the view specs reach them
interface ControlViewVm {
  authForm: { password: string }
  authError: string
  authenticateControl: () => Promise<void>
  stopCameraFeed: () => void
}

let manualUnlockView: ControlViewVm | null = null

const manualUnlockSite: CallSite = {
  name: 'ControlView: manual-unlock (password step-up)',
  method: 'POST',
  url: '/api/v2/control/manual-unlock',
  body: { method: 'password', password: 'operator-pw' },
  attempts: 2,
  run: async () => {
    const wrapper = mount(ControlView, { global: { stubs: { VirtualJoystick: true, RouterLink: RouterLinkStub } } })
    cleanups.push(() => wrapper.unmount())
    const view = wrapper.vm as unknown as ControlViewVm // <script setup> bindings are reachable on vm in tests
    await flushPromises()
    view.stopCameraFeed()
    view.authForm.password = 'operator-pw'
    await view.authenticateControl()
    await flushPromises()
    view.stopCameraFeed() // a successful unlock starts the camera feed
    manualUnlockView = view
  },
  failureText: () => manualUnlockView?.authError ?? '',
}

const DRIVE = { session_id: 's-1', vector: { linear: 0.4, angular: 0 }, duration_ms: 160, reason: 'manual-joystick', max_speed_limit: 0.5 }
const BLADE = { session_id: 's-1', action: 'enable', reason: 'manual-control' }

const GATED_CALL_SITES: CallSite[] = [
  // control (src/services/api.ts sendControlCommand <- src/stores/control.ts submitCommand <- ControlView.vue)
  apiSite('sendControlCommand(drive)', '/api/v2/control/drive', () => sendControlCommand('drive', DRIVE), DRIVE),
  apiSite('sendControlCommand(blade)', '/api/v2/control/blade', () => sendControlCommand('blade', BLADE), BLADE),
  controlStoreSite('drive', '/api/v2/control/drive', DRIVE),
  controlStoreSite('blade', '/api/v2/control/blade', BLADE),
  manualUnlockSite,
  // maintenance (src/services/api.ts maintenanceApi <- DashboardView.vue runImuCalibration)
  apiSite('maintenanceApi.runImuCalibration', '/api/v2/maintenance/imu/calibrate', () => maintenanceApi.runImuCalibration(), {}),
  // missions (src/stores/mission.ts <- MissionPlannerView.vue)
  missionSite('createMission', '/api/v2/missions/create', () => useMissionStore().createMission('north lawn')),
  missionSite('startCurrentMission', '/api/v2/missions/m-1/start', () => useMissionStore().startCurrentMission()),
  missionSite('pauseCurrentMission', '/api/v2/missions/m-1/pause', () => useMissionStore().pauseCurrentMission()),
  missionSite('resumeCurrentMission', '/api/v2/missions/m-1/resume', () => useMissionStore().resumeCurrentMission()),
  missionSite('abortCurrentMission', '/api/v2/missions/m-1/abort', () => useMissionStore().abortCurrentMission()),
  // AI (src/views/AIView.vue)
  aiSite('Enable AI', 'Enable AI', '/api/v2/ai/enable', { mode: 'ai_autonomous' }),
  aiSite('Disable AI', 'Disable AI', '/api/v2/ai/disable', {}, { enabled: true }),
  aiSite('Deploy model', 'Deploy', '/api/v2/ai/model', { model_path: '/models/lawn_vla.hef' }, { modelPath: '/models/lawn_vla.hef' }),
  aiSite('Reset metrics', 'Reset', '/api/v2/ai/metrics/reset', {}),
  aiSite('Export dataset', 'Export', '/api/v2/ai/datasets/ds-1/export', { format: 'YOLO' }),
  // tractor writes that were already gated (src/services/api.ts <- TractorControlView.vue / stores/tractor.ts)
  apiSite('tractorLeftLever', '/api/v2/tractor/left-lever', () => tractorLeftLever(0.5), { value: 0.5 }),
  apiSite('tractorRightLever', '/api/v2/tractor/right-lever', () => tractorRightLever(-0.5), { value: -0.5 }),
  apiSite('tractorThrottle', '/api/v2/tractor/throttle', () => tractorThrottle(0.3), { value: 0.3 }),
  apiSite('tractorBlade', '/api/v2/tractor/blade', () => tractorBlade(true), { engaged: true }),
  apiSite('tractorStart', '/api/v2/tractor/starter', () => tractorStart(), {}),
  apiSite('tractorStopEngine', '/api/v2/tractor/stop-engine', () => tractorStopEngine(), {}),
  apiSite('tractorClearEmergency', '/api/v2/tractor/clear-emergency', () => tractorClearEmergency(), {}),
  apiSite('tractorAuthorize(true)', '/api/v2/tractor/authorize', () => tractorAuthorize(true), {}),
  apiSite('tractorAuthorize(false) = revoke', '/api/v2/tractor/revoke', () => tractorAuthorize(false), {}),
]

describe.each(GATED_CALL_SITES)('gated call site: $name  [POST $url]', (site) => {
  it('signed in: every request to it carries the Bearer token and the client id', async () => {
    useAuthStore().token = TOKEN
    const requests = install(signedInResponder)

    await site.run()

    const sent = requests.filter((request) => request.method === site.method && request.url === site.url)
    expect(sent).toHaveLength(site.attempts ?? 1)
    for (const request of sent) {
      expect(request.headers.authorization).toBe(`Bearer ${TOKEN}`)
      expect(request.headers['x-client-id']).toMatch(/^web-/)
    }
    if (site.body !== undefined) expect(sent.at(-1)?.body).toEqual(site.body)
    expect(thrown).toBeNull()
  })

  it('signed out: no Authorization header, one attempt, and the 401 reads as the sign-in text', async () => {
    useAuthStore().token = null
    const requests = install(signedOutResponder)

    await site.run()

    const sent = requests.filter((request) => request.method === site.method && request.url === site.url)
    expect(sent).toHaveLength(site.attempts ?? 1) // nothing replays it
    for (const request of sent) {
      expect(request.headers.authorization).toBeUndefined()
      expect(request.headers['x-client-id']).toMatch(/^web-/)
    }
    expect(requests.filter((request) => request.url.includes('/auth/'))).toEqual([]) // no refresh or login detour
    expect(site.failureText()).toBe(SIGN_IN_REQUIRED_MESSAGE)
  })
})

describe('a 401 on drive or blade is not a safety interlock', () => {
  it.each([
    ['drive', DRIVE],
    ['blade', BLADE],
  ] as const)('%s while signed out: the sign-in text, and the control store is not locked out', async (command, payload) => {
    useAuthStore().token = null
    install(signedOutResponder)
    const control = useControlStore()

    await expect(control.submitCommand(command, payload)).rejects.toMatchObject({ response: { status: 401 } })

    expect(control.lockout).toBe(false)
    expect(control.lockoutActive).toBe(false)
    expect(control.lockoutReason).toBe('')
    expect(control.lastCommandResult).toEqual({ result: 'error', status_reason: SIGN_IN_REQUIRED_MESSAGE })
    expect(control.canSubmitCommand).toBe(true)
  })
})

// ---------------------------------------------------------------------------------------------------------------
// The emergency stops stay open
// ---------------------------------------------------------------------------------------------------------------

describe('the emergency stops need no sign-in', () => {
  const ESTOPS: Array<[string, string, () => Promise<unknown>]> = [
    ['sendControlCommand(emergency)', '/api/v2/control/emergency', () => sendControlCommand('emergency', { session_id: 's-1' })],
    ['controlApi.emergencyStop', '/api/v2/control/emergency-stop', () => controlApi.emergencyStop()],
    ['tractorEmergencyStop', '/api/v2/tractor/emergency-stop', () => tractorEmergencyStop()],
  ]

  it.each(ESTOPS)('%s works with no token: still sent, no Authorization header, resolves', async (_name, url, call) => {
    useAuthStore().token = null
    const requests = install(() => ({ status: 202, data: { accepted: true, result: 'accepted' } }))

    await expect(call()).resolves.toMatchObject({ accepted: true })

    expect(requests).toHaveLength(1)
    expect(requests[0]).toMatchObject({ method: 'POST', url })
    expect(requests[0].headers.authorization).toBeUndefined()
    expect(requests[0].headers['x-client-id']).toMatch(/^web-/)
  })

  it('through the control store, signed out: sent, and the store stays usable', async () => {
    useAuthStore().token = null
    const requests = install(() => ({ status: 202, data: { accepted: true, result: 'accepted' } }))
    const control = useControlStore()

    await control.submitCommand('emergency', { session_id: 's-1' })

    expect(requests.map((request) => `${request.method} ${request.url}`)).toEqual(['POST /api/v2/control/emergency'])
    expect(requests[0].headers.authorization).toBeUndefined()
    expect(control.lockoutActive).toBe(false)
  })
})

// ---------------------------------------------------------------------------------------------------------------
// Static guard: nothing writes to the API around the shared ApiService
// ---------------------------------------------------------------------------------------------------------------

const NON_GET_METHOD = /\bmethod\s*:\s*['"`](?:post|put|patch|delete)['"`]/i

/** The text between the parentheses of the call whose "(" is at `open`; string literals are skipped. */
function callArguments(source: string, open: number): string {
  let depth = 0
  let quote = ''
  for (let i = open; i < source.length; i += 1) {
    const char = source[i]
    if (quote) {
      if (char === '\\') i += 1
      else if (char === quote) quote = ''
    } else if (char === '"' || char === "'" || char === '`') {
      quote = char
    } else if (char === '(') {
      depth += 1
    } else if (char === ')') {
      depth -= 1
      if (depth === 0) return source.slice(open + 1, i)
    }
  }
  return source.slice(open + 1) // unbalanced: judge everything that follows (fail closed)
}

/** Every call matching `callee` (a global regex ending in "("), with its argument text. */
function callsOf(source: string, callee: RegExp): Array<{ text: string; args: string }> {
  return [...source.matchAll(callee)].map((match) => {
    const args = callArguments(source, (match.index ?? 0) + match[0].length - 1)
    return { text: `${match[0]}${args})`, args }
  })
}

/**
 * Ways a source file can write to the backend without going through ApiService (src/services/api.ts), which is the
 * only place that attaches the Bearer and refreshes it on 401. Textual on purpose: a mention in a comment is
 * flagged too, and the fix is to reword it.
 */
function findApiBypasses(source: string): string[] {
  const brief = (text: string) => text.replace(/\s+/g, ' ').slice(0, 90)
  const found: string[] = []

  for (const call of callsOf(source, /\bfetch\s*\(/g)) {
    if (!NON_GET_METHOD.test(call.args)) continue
    // a literal absolute URL to some other host is not the API
    if (/^\s*['"`]https?:\/\//.test(call.args) && !call.args.includes('/api/')) continue
    found.push(`fetch with a write method: ${brief(call.text)}`)
  }
  for (const match of source.matchAll(/\baxios\s*\.\s*(?:post|put|patch|delete)\s*\(/g)) {
    found.push(`axios used directly for a write: ${brief(match[0])}`)
  }
  for (const call of callsOf(source, /\baxios(?:\s*\.\s*request)?\s*\(/g)) {
    if (NON_GET_METHOD.test(call.args)) found.push(`axios request with a write method: ${brief(call.text)}`)
  }
  for (const match of source.matchAll(/\baxios\s*\.\s*create\s*\(/g)) {
    found.push(`a second axios instance (no interceptors): ${brief(match[0])}`)
  }
  for (const match of source.matchAll(/\bnew\s+XMLHttpRequest\b/g)) {
    found.push(`XMLHttpRequest: ${brief(match[0])}`)
  }
  for (const match of source.matchAll(/\bsendBeacon\s*\(/g)) {
    found.push(`sendBeacon (always a POST): ${brief(match[0])}`)
  }
  return found
}

describe('static guard: findApiBypasses (the scanner itself)', () => {
  it.each([
    ['fetch POST', `await fetch('/api/v2/control/drive', { method: 'POST', body: JSON.stringify(x) })`],
    ['fetch PUT, lower case, multi-line', `fetch('/api/v2/map/zones', {\n  headers: { a: 'b' },\n  method: "put",\n})`],
    ['fetch PATCH with a template-literal URL', 'fetch(`/api/v2/missions/${id}/start`, { method: `PATCH` })'],
    ['fetch DELETE to a variable URL', `fetch(url, { method: 'DELETE' })`],
    ['fetch POST whose URL contains a closing paren', `fetch('/api/v2/x)y', { method: 'POST' })`],
    ['axios.post', `axios.post('/api/v2/missions/create', body)`],
    ['axios.put', `await axios . put(url, body)`],
    ['axios.patch', `axios.patch(url)`],
    ['axios.delete', `axios.delete('/api/v2/recording/sessions/1')`],
    ['axios.request with a write method', `axios.request({ url: '/api/v2/x', method: 'post' })`],
    ['axios(config) with a write method', `axios({ method: 'delete', url: '/api/v2/x' })`],
    ['a second axios instance', `const raw = axios.create({ baseURL: '' })`],
    ['XMLHttpRequest', `const xhr = new XMLHttpRequest()`],
    ['sendBeacon', `navigator.sendBeacon('/api/v2/telemetry/ping', data)`],
    ['a write inside a .vue <script>', `<template><div/></template>\n<script setup lang="ts">\nfetch('/api/v2/ai/enable', { method: 'POST' })\n</script>`],
  ])('flags %s', (_name, snippet) => {
    expect(findApiBypasses(snippet)).not.toEqual([])
  })

  it.each([
    ['a GET fetch with no options', `fetch('/api/v2/dashboard/telemetry')`],
    ['a GET fetch with headers', `fetch('/api/v2/settings/maps', { headers: { 'Cache-Control': 'no-cache' } })`],
    ['an explicit GET fetch', `fetch('/api/v2/nav/coverage-plan', { method: 'GET' })`],
    ['a write fetch to another host', `fetch('https://tiles.example.com/upload', { method: 'POST' })`],
    ['the shared ApiService', `await apiService.post('/api/v2/missions/create', body)`],
    ['the injected ApiService', `await api.post('/api/v2/ai/enable', {})\nawait api.put(url, body)`],
    ['axios type helpers', `import axios from 'axios'\nif (axios.isAxiosError(error)) {}`],
    ['axios GET', `axios.get('/x')`],
    ['an unrelated method property', `const options = { method: 'POST' } // never passed to fetch`],
  ])('does not flag %s', (_name, snippet) => {
    expect(findApiBypasses(snippet)).toEqual([])
  })

  it('names the offending call so a failure is actionable', () => {
    const [finding] = findApiBypasses(`await fetch('/api/v2/control/drive', {\n  method: 'POST',\n})`)

    expect(finding).toContain("fetch('/api/v2/control/drive'")
    expect(finding).toContain("method: 'POST'")
  })
})

describe('static guard: every write to the API in src/ goes through the shared ApiService', () => {
  // vite-node injects __dirname into test files (import.meta.url is not a file: URL under jsdom)
  const SRC_ROOT = join(__dirname, '..', '..', 'src')
  const SHARED_SERVICE = join(SRC_ROOT, 'services', 'api.ts')

  const entries: string[] = readdirSync(SRC_ROOT, { recursive: true })
  const sources = entries
    .map((entry) => join(SRC_ROOT, entry))
    .filter((file) => /\.(ts|vue)$/.test(file) && file !== SHARED_SERVICE)

  it('scans the real source tree (so an empty result cannot mean nothing was read)', () => {
    expect(sources.length).toBeGreaterThan(30)
    expect(sources.some((file) => file.endsWith(join('stores', 'mission.ts')))).toBe(true)
    expect(sources.some((file) => file.endsWith('AIView.vue'))).toBe(true)
  })

  it('finds no fetch/XMLHttpRequest/axios/sendBeacon write outside services/api.ts', () => {
    const offenders = sources.flatMap((file) =>
      findApiBypasses(readFileSync(file, 'utf8')).map((finding) => `${relative(SRC_ROOT, file)}: ${finding}`)
    )

    expect(offenders).toEqual([])
  })
})
