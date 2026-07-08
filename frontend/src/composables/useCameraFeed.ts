// Camera feed (MJPEG stream + snapshot-fallback) state and control, extracted
// verbatim from ControlView.vue so TractorControlView.vue can reuse it without
// duplicating ~350 lines of stream/retry/snapshot-fallback logic. Zero behavior
// change from the original ControlView.vue implementation — see
// ControlView.unlock.spec.ts / ControlView.movement.spec.ts, which pin this
// contract and must stay green.
import { computed, reactive, ref } from 'vue'
import axios from 'axios'
import { useApiService } from '@/services/api'

const CAMERA_RETRY_COOLDOWN_MS = 5000

interface CameraStatusSummary {
  active: boolean
  mode: string
  fps: number | null
  client_count: number | null
}

export interface UseCameraFeedOptions {
  /** Manual-control auth session id, if the consuming view has one (optional). */
  sessionId?: () => string | undefined
}

export function useCameraFeed(options: UseCameraFeedOptions = {}) {
  const api = useApiService()
  const getSessionId = options.sessionId ?? (() => undefined)

  const cameraInfo = reactive<CameraStatusSummary>({
    active: false,
    mode: 'offline',
    fps: null,
    client_count: null
  })
  const cameraFrameUrl = ref<string | null>(null)
  const cameraStreamUrl = ref<string | null>(null)
  const cameraStreamUnavailable = ref(false)
  const cameraStatusMessage = ref('Initializing camera…')
  const cameraError = ref<string | null>(null)
  const cameraLastFrame = ref<string | null>(null)
  const cameraFetchInFlight = ref(false)
  const cameraStreamFailureCount = ref(0)
  const cameraStreamClientId = (() => {
    try {
      if (typeof crypto !== 'undefined' && typeof crypto.randomUUID === 'function') {
        return crypto.randomUUID()
      }
    } catch (error) {
      /* noop */
    }
    return `client-${Math.random().toString(36).slice(2)}`
  })()
  let cameraFrameTimer: number | undefined
  let cameraStatusTimer: number | undefined
  let cameraRetryTimer: number | undefined
  let cameraStartRequested = false

  const cameraDisplaySource = computed(() => cameraStreamUrl.value ?? cameraFrameUrl.value)
  const cameraIsStreaming = computed(() => Boolean(cameraStreamUrl.value))

  function formatCameraFps(value?: number | null) {
    if (typeof value !== 'number' || Number.isNaN(value) || value <= 0) {
      return '—'
    }
    return value.toFixed(1)
  }

  function formatCameraTimestamp(timestamp: string | null | undefined) {
    if (!timestamp) {
      return 'No frames yet'
    }
    const parsed = new Date(timestamp)
    if (Number.isNaN(parsed.getTime())) {
      return 'No frames yet'
    }
    return parsed.toLocaleTimeString()
  }

  function resetCameraState() {
    cameraStreamUrl.value = null
    cameraFrameUrl.value = null
    cameraStatusMessage.value = 'Initializing camera…'
    cameraError.value = null
    cameraLastFrame.value = null
    cameraStreamFailureCount.value = 0
    cameraStreamUnavailable.value = false
    clearCameraRetryTimer()
    Object.assign(cameraInfo, {
      active: false,
      mode: 'offline',
      fps: null,
      client_count: null
    })
  }

  function buildCameraStreamUrl(forceRefresh = false) {
    if (cameraStreamUnavailable.value) {
      return null
    }
    const params = new URLSearchParams()
    params.set('client', cameraStreamClientId)
    const sessionId = getSessionId()
    if (sessionId) {
      params.set('session_id', sessionId)
    }
    if (forceRefresh) {
      params.set('ts', Date.now().toString(36))
    }
    return `/api/v2/camera/stream.mjpeg?${params.toString()}`
  }

  function refreshCameraStream(forceRefresh = false, resetFailures = false) {
    const nextUrl = buildCameraStreamUrl(forceRefresh)
    if (!nextUrl) {
      return
    }
    if (resetFailures) {
      cameraStreamFailureCount.value = 0
    }
    cameraStreamUrl.value = nextUrl
    cameraStatusMessage.value = 'Connecting to stream…'
  }

  function clearSnapshotTimer() {
    if (cameraFrameTimer) {
      window.clearInterval(cameraFrameTimer)
      cameraFrameTimer = undefined
    }
  }

  function clearCameraRetryTimer() {
    if (cameraRetryTimer) {
      window.clearTimeout(cameraRetryTimer)
      cameraRetryTimer = undefined
    }
  }

  async function attemptCameraStreamRecovery() {
    clearCameraRetryTimer()
    if (!cameraStreamUnavailable.value) {
      return
    }
    try {
      const streaming = await ensureCameraStreaming()
      if (streaming) {
        cameraStreamUnavailable.value = false
        refreshCameraStream(true, true)
        return
      }
    } catch (error) {
      /* noop – retry will be scheduled */
    }
    scheduleCameraStreamRetry()
  }

  function scheduleCameraStreamRetry(delayMs = CAMERA_RETRY_COOLDOWN_MS) {
    if (!cameraStreamUnavailable.value) {
      clearCameraRetryTimer()
      return
    }
    clearCameraRetryTimer()
    cameraRetryTimer = window.setTimeout(() => {
      void attemptCameraStreamRecovery()
    }, delayMs)
  }

  function startSnapshotFallback(message?: string) {
    cameraStreamUrl.value = null
    if (message) {
      cameraStatusMessage.value = message
    }
    clearSnapshotTimer()
    void fetchCameraFrame()
    cameraFrameTimer = window.setInterval(fetchCameraFrame, 2000)
  }

  function handleCameraStreamLoad() {
    cameraError.value = null
    cameraStatusMessage.value = 'Streaming…'
    cameraStreamFailureCount.value = 0
    cameraStreamUnavailable.value = false
    clearCameraRetryTimer()
  }

  function handleCameraStreamError() {
    cameraStreamFailureCount.value += 1
    if (cameraStreamFailureCount.value <= 1) {
      refreshCameraStream(true)
      return
    }
    cameraStreamUnavailable.value = true
    cameraError.value = 'Camera stream unavailable'
    startSnapshotFallback('Camera stream unavailable – using snapshots…')
    scheduleCameraStreamRetry()
  }

  async function fetchCameraStatus() {
    try {
      const response = await api.get('/api/v2/camera/status')
      const payload = response.data
      if (payload?.status === 'success' && payload.data) {
        const data = payload.data
        Object.assign(cameraInfo, {
          active: Boolean(data.is_active),
          mode: data.mode || 'offline',
          fps: typeof data.statistics?.current_fps === 'number'
            ? Number(data.statistics.current_fps)
            : null,
          client_count: typeof data.client_count === 'number'
            ? Number(data.client_count)
            : null
        })
        if (!cameraInfo.active && cameraStreamUrl.value) {
          startSnapshotFallback('Camera idle')
        } else if (
          cameraInfo.active &&
          !cameraStreamUrl.value &&
          !cameraStartRequested &&
          !cameraStreamUnavailable.value
        ) {
          clearSnapshotTimer()
          refreshCameraStream(true, true)
        }
        if (data.last_frame_time && !cameraLastFrame.value) {
          cameraLastFrame.value = data.last_frame_time
        }
        cameraError.value = null
        if (cameraInfo.active && !cameraFrameUrl.value) {
          cameraStatusMessage.value = 'Waiting for frames…'
        }
        return data
      }
      if (payload?.error) {
        cameraError.value = payload.error
        cameraStatusMessage.value = payload.error
      }
    } catch (error) {
      if (axios.isAxiosError(error) && error.response?.status === 429) {
        cameraError.value = 'Camera service temporarily busy'
        cameraStatusMessage.value = 'Camera requests throttled – retrying…'
      } else {
        cameraError.value = 'Unable to reach camera service'
        cameraStatusMessage.value = 'Camera offline'
      }
    }
    return null
  }

  async function ensureCameraStreaming() {
    const status = await fetchCameraStatus()
    if (status?.is_active) {
      if (cameraStreamUnavailable.value) {
        return false
      }
      return true
    }

    if (cameraStartRequested) {
      return cameraInfo.active
    }

    cameraStartRequested = true
    try {
      const response = await api.post('/api/v2/camera/start')
      const payload = response.data
      if (payload?.status === 'error' && payload?.error) {
        cameraError.value = payload.error
        cameraStatusMessage.value = payload.error
      }
    } catch (error) {
      cameraError.value = 'Failed to start camera stream'
      cameraStatusMessage.value = 'Camera offline'
    } finally {
      await fetchCameraStatus()
      cameraStartRequested = false
    }

    return cameraInfo.active
  }

  async function fetchCameraFrame() {
    if (cameraFetchInFlight.value) {
      return
    }

    cameraFetchInFlight.value = true
    try {
      const response = await api.get('/api/v2/camera/frame')
      const payload = response.data
      if (payload?.status === 'success' && payload.data) {
        const frame = payload.data
        const format = typeof frame?.metadata?.format === 'string'
          ? String(frame.metadata.format).toLowerCase()
          : 'jpeg'
        if (frame?.data) {
          cameraFrameUrl.value = `data:image/${format};base64,${frame.data}`
          cameraStatusMessage.value = 'Snapshots…'
          cameraError.value = null
        } else {
          cameraStatusMessage.value = 'Waiting for frame data…'
        }
        if (frame?.metadata?.timestamp) {
          cameraLastFrame.value = frame.metadata.timestamp
        }
      } else if (payload?.error === 'No frame available') {
        cameraStatusMessage.value = 'Camera warming up…'
      } else if (payload?.error) {
        cameraError.value = payload.error
        cameraStatusMessage.value = payload.error
      }
    } catch (error) {
      if (axios.isAxiosError(error) && error.response?.status === 429) {
        cameraError.value = 'Camera frames throttled'
        cameraStatusMessage.value = 'Cooling down camera requests…'
      } else {
        cameraError.value = 'Camera frame request failed'
        cameraStatusMessage.value = 'Camera offline'
      }
    } finally {
      cameraFetchInFlight.value = false
    }
  }

  async function startCameraFeed(forceReconnect = false) {
    if (!forceReconnect && (cameraIsStreaming.value || cameraFrameTimer || cameraStatusTimer)) {
      return
    }

    if (cameraStatusTimer) {
      window.clearInterval(cameraStatusTimer)
      cameraStatusTimer = undefined
    }
    clearSnapshotTimer()
    resetCameraState()

    const streaming = await ensureCameraStreaming()
    if (streaming && !cameraStreamUnavailable.value) {
      refreshCameraStream(true, true)
      clearSnapshotTimer()
    } else {
      startSnapshotFallback(cameraError.value || 'Camera warming up…')
      cameraStreamUnavailable.value = true
      scheduleCameraStreamRetry()
    }

    if (!cameraStatusTimer) {
      cameraStatusTimer = window.setInterval(fetchCameraStatus, 6000)
    }
  }

  function stopCameraFeed() {
    clearSnapshotTimer()
    clearCameraRetryTimer()
    if (cameraStatusTimer) {
      window.clearInterval(cameraStatusTimer)
      cameraStatusTimer = undefined
    }
    cameraStartRequested = false
    cameraFetchInFlight.value = false
    cameraStreamUrl.value = null
    cameraStreamFailureCount.value = 0
    cameraFrameUrl.value = null
    cameraLastFrame.value = null
    cameraError.value = null
    cameraStatusMessage.value = 'Camera paused'
    cameraStreamUnavailable.value = false
    Object.assign(cameraInfo, {
      active: false,
      mode: 'offline',
      fps: null,
      client_count: null
    })
  }

  async function retryCameraFeed() {
    stopCameraFeed()
    cameraStreamFailureCount.value = 0
    cameraStreamUnavailable.value = false
    await startCameraFeed(true)
  }

  return {
    // State
    cameraInfo,
    cameraFrameUrl,
    cameraStreamUrl,
    cameraStreamUnavailable,
    cameraStatusMessage,
    cameraError,
    cameraLastFrame,
    cameraFetchInFlight,
    cameraStreamFailureCount,
    // Computed
    cameraDisplaySource,
    cameraIsStreaming,
    // Formatting helpers
    formatCameraFps,
    formatCameraTimestamp,
    // Actions
    resetCameraState,
    refreshCameraStream,
    attemptCameraStreamRecovery,
    scheduleCameraStreamRetry,
    handleCameraStreamLoad,
    handleCameraStreamError,
    fetchCameraStatus,
    ensureCameraStreaming,
    fetchCameraFrame,
    startCameraFeed,
    stopCameraFeed,
    retryCameraFeed,
  }
}
