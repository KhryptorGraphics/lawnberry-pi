<template>
  <div class="control-view">
    <div class="page-header">
      <h1>Manual Control</h1>
      <p class="text-muted">Direct manual operation and emergency controls</p>
    </div>

    <!-- Security Gate -->
    <SecurityGateCard
      v-if="!isControlUnlocked"
      v-model:password="authForm.password"
      v-model:totp-code="authForm.totpCode"
      :security-config="securityConfig"
      :authenticating="authenticating"
      :can-authenticate="canAuthenticate"
      :auth-error="authError"
      :format-security-level="formatSecurityLevel"
      @authenticate="authenticateControl"
      @authenticate-google="authenticateWithGoogle"
      @verify-cloudflare="verifyCloudflareAuth"
    />

    <!-- Main Control Interface (shown when unlocked) -->
    <div v-else class="control-interface">
      <!-- Control Status -->
      <div class="control-status">
        <div class="status-indicator" :class="systemStatus">
          <div class="status-light" />
          <span>{{ formatSystemStatus(systemStatus) }}</span>
        </div>
        
        <div class="session-info">
          <small>Control session expires in {{ formatTimeRemaining(sessionTimeRemaining) }}</small>
          <button class="btn btn-sm btn-secondary" @click="lockControl">
            🔒 Lock Control
          </button>
        </div>
      </div>

      <div class="controller-health">
        <div class="controller-chip" :class="`controller-chip--${motorControllerState.severity}`">
          <span class="controller-chip__label">Motor controller</span>
          <span class="controller-chip__value">{{ motorControllerState.label }}</span>
        </div>
        <p v-if="motorControllerState.message" class="controller-chip__message">
          {{ motorControllerState.message }}
        </p>
      </div>

      <!-- Emergency Stop -->
      <div class="emergency-section">
        <button 
          class="btn btn-emergency" 
          :disabled="performing"
          @click="emergencyStop"
        >
          🛑 EMERGENCY STOP
        </button>
      </div>

      <!-- Live Camera Feed -->
      <CameraFeedCard
        :camera-error="cameraError"
        :camera-display-source="cameraDisplaySource"
        :camera-is-streaming="cameraIsStreaming"
        :camera-status-message="cameraStatusMessage"
        :camera-info="cameraInfo"
        :camera-last-frame="cameraLastFrame"
        :format-camera-fps="formatCameraFps"
        :format-camera-timestamp="formatCameraTimestamp"
        @stream-load="handleCameraStreamLoad"
        @stream-error="handleCameraStreamError"
        @retry="retryCameraFeed"
      />

      <!-- Movement Controls -->
      <div class="card">
        <div class="card-header">
          <h3>Movement Controls</h3>
        </div>
        <div class="card-body">
          <div class="movement-layout">
            <div class="joystick-column">
              <VirtualJoystick
                ref="joystickRef"
                class="joystick-component"
                :disabled="!canMove"
                :dead-zone="0.12"
                @change="handleJoystickChange"
                @end="handleJoystickEnd"
              />
              <small class="joystick-hint">Drag to drive • Release or tap stop to halt</small>
            </div>
            <div class="movement-actions">
              <button
                class="btn btn-danger stop-button"
                :disabled="!isControlUnlocked"
                @click="handleStopButton"
              >
                🛑 Stop Motors
              </button>
              <div class="movement-readout">
                <span>Linear: {{ formatCommandValue(activeDriveVector.linear) }}</span>
                <span>Angular: {{ formatCommandValue(activeDriveVector.angular) }}</span>
              </div>
              <div v-if="joystickEngaged" class="movement-status active">Joystick engaged</div>
              <div v-else class="movement-status">Joystick idle</div>
            </div>
          </div>

          <div class="speed-control">
            <label>Speed: {{ speedLevel }}%</label>
            <input 
              v-model.number="speedLevel"
              type="range" 
              min="10" 
              max="100" 
              step="10"
              class="speed-slider"
            >
          </div>
        </div>
      </div>

      <!-- Mowing Controls -->
      <div class="card">
        <div class="card-header">
          <h3>Mowing Controls</h3>
        </div>
        <div class="card-body">
          <div class="mowing-controls">
            <button 
              class="btn" 
              :class="mowingActive ? 'btn-warning' : 'btn-success'"
              :disabled="performing"
              @click="toggleMowing"
            >
              {{ mowingActive ? '⏹️ Stop Mowing' : '▶️ Start Mowing' }}
            </button>
          </div>
        </div>
      </div>

      <!-- System Controls -->
      <div class="card">
        <div class="card-header">
          <h3>System Controls</h3>
        </div>
        <div class="card-body">
          <div class="system-controls">
            <button class="btn btn-info" :disabled="performing" @click="returnToBase">
              🏠 Return to Base
            </button>
            
            <button class="btn btn-warning" :disabled="performing" @click="pauseSystem">
              ⏸️ Pause System
            </button>
            
            <button class="btn btn-success" :disabled="performing" @click="resumeSystem">
              ▶️ Resume System
            </button>
          </div>
        </div>
      </div>

      <!-- Live Telemetry -->
      <LiveStatusCard :telemetry="telemetry" :display-speed="displaySpeed" :speed-unit="speedUnit" />
    </div>

    <!-- Status Messages -->
    <div v-if="lockout" class="alert alert-danger">
      {{ lockoutReason }}
    </div>
    <div v-if="statusMessage" class="alert" :class="statusSuccess ? 'alert-success' : 'alert-danger'">
      {{ statusMessage }}
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed, onMounted, onUnmounted, reactive, ref, watch } from 'vue'
import { type AxiosError } from 'axios'
import { storeToRefs } from 'pinia'
import { useControlStore } from '@/stores/control'
import {
  useApiService,
  pauseAutonomous,
  resumeAutonomous,
  returnToBase as returnToBaseApi,
} from '@/services/api'
import { useToastStore } from '@/stores/toast'
import { usePreferencesStore } from '@/stores/preferences'
import type { RoboHATStatus } from '@/types/control'
import VirtualJoystick from '@/components/ui/VirtualJoystick.vue'
import SecurityGateCard from '@/components/control/SecurityGateCard.vue'
import CameraFeedCard from '@/components/control/CameraFeedCard.vue'
import LiveStatusCard from '@/components/control/LiveStatusCard.vue'
import { useCameraFeed } from '@/composables/useCameraFeed'

interface ManualControlSecurityConfig {
  auth_level: 'password' | 'totp' | 'google' | 'cloudflare'
  session_timeout_minutes: number
  require_https: boolean
  auto_lock_manual_control: boolean
}

interface ManualControlSession {
  session_id: string
  expires_at?: string
}

interface ControlTelemetry {
  battery?: { percentage?: number; voltage?: number | null }
  position?: { latitude?: number | null; longitude?: number | null }
  safety_state?: string
  velocity?: { linear?: { x?: number | null } }
  telemetry_source?: 'hardware' | 'simulated' | 'unknown'
  camera?: RoboHATStatus['camera']
}

interface CameraStatusSummary {
  active: boolean
  mode: string
  fps: number | null
  client_count: number | null
}

interface MotorControllerState {
  ready: boolean
  serialConnected: boolean
  severity: 'success' | 'warning' | 'danger' | 'info'
  label: string
  message: string | null
}

const MOVEMENT_DURATION_MS = 160
const MOVEMENT_REPEAT_INTERVAL_MS = 120
const JOYSTICK_REASON = 'manual-joystick'

const control = useControlStore()
const api = useApiService()
const toast = useToastStore()
const preferences = usePreferencesStore()

preferences.ensureInitialized()
const { unitSystem } = storeToRefs(preferences)

// Store-backed state
const lockout = computed(() => control.lockout)
const lockoutReason = computed(() => control.lockoutReason)
const lastEcho = computed(() => control.lastEcho)
const lastCommandResult = computed(() => control.lastCommandResult)

// Manual control security configuration and authentication state
const securityConfig = ref<ManualControlSecurityConfig>({
  auth_level: 'password',
  session_timeout_minutes: 15,
  require_https: false,
  auto_lock_manual_control: true
})

const isControlUnlocked = ref(false)
const authenticating = ref(false)
const authError = ref('')
const authForm = reactive({
  password: '',
  totpCode: ''
})

const session = ref<ManualControlSession | null>(null)
const sessionTimeRemaining = ref(0)
let sessionTimer: number | undefined
let cloudflareAutoVerificationAttempted = false

// UI state
const statusMessage = ref('')
const statusSuccess = ref(false)
const performing = ref(false)
const systemStatus = ref('unknown')
const telemetry = ref<ControlTelemetry>({ safety_state: 'unknown', telemetry_source: 'unknown' })
const currentSpeed = ref(0)
const displaySpeed = computed(() => {
  const value = Number(currentSpeed.value)
  if (!Number.isFinite(value)) {
    return '0.0'
  }
  const converted = unitSystem.value === 'imperial' ? value * 2.23694 : value
  return converted.toFixed(1)
})
const speedUnit = computed(() => (unitSystem.value === 'imperial' ? 'mph' : 'm/s'))
const mowingActive = ref(false)
const speedLevel = ref(50)

// Not all of these are read by name below — ControlView.unlock.spec.ts and
// ControlView.movement.spec.ts reach into several (e.g. cameraStreamUrl,
// attemptCameraStreamRecovery) via wrapper.vm, which Vue's <script setup>
// compiler exposes for every top-level binding regardless of local usage;
// eslint's static analysis can't see that reflective access.
/* eslint-disable @typescript-eslint/no-unused-vars */
const {
  cameraInfo,
  cameraFrameUrl,
  cameraStreamUrl,
  cameraStreamUnavailable,
  cameraStatusMessage,
  cameraError,
  cameraLastFrame,
  cameraFetchInFlight,
  cameraStreamFailureCount,
  cameraDisplaySource,
  cameraIsStreaming,
  formatCameraFps,
  formatCameraTimestamp,
  attemptCameraStreamRecovery,
  handleCameraStreamLoad,
  handleCameraStreamError,
  startCameraFeed,
  stopCameraFeed,
  retryCameraFeed,
} = useCameraFeed({ sessionId: () => session.value?.session_id })
/* eslint-enable @typescript-eslint/no-unused-vars */

interface JoystickHandle {
  reset: () => void
  setVector: (vector: { x: number; y: number }) => void
}

interface DriveCommandPayload {
  [key: string]: unknown
  session_id: string
  vector: { linear: number; angular: number }
  duration_ms: number
  reason: string
  max_speed_limit: number
}

interface QueueDriveCommandOptions {
  immediate?: boolean
}

const joystickRef = ref<JoystickHandle | null>(null)
const lastJoystickVector = ref({ x: 0, y: 0 })
const joystickEngaged = ref(false)
const activeDriveVector = ref({ linear: 0, angular: 0 })
let movementRepeatTimer: number | undefined
let currentDriveReason = JOYSTICK_REASON
let driveDispatchPromise: Promise<void> | null = null
let driveCommandActive = false
let pendingDrivePayload: DriveCommandPayload | null = null

const previousMotorReady = ref<boolean | null>(null)
const previousSerialConnected = ref<boolean | null>(null)

// Derived state
const canAuthenticate = computed(() => {
  switch (securityConfig.value.auth_level) {
    case 'password':
      return authForm.password.trim().length > 0
    case 'totp':
      return authForm.password.trim().length > 0 && authForm.totpCode.trim().length === 6
    case 'google':
      return true
    case 'cloudflare':
      return true
    default:
      return false
  }
})

const robohatStatus = computed(() => control.robohatStatus as Record<string, any> | null)

const motorControllerState = computed<MotorControllerState>(() => describeMotorController(robohatStatus.value))

const canMove = computed(() =>
  isControlUnlocked.value && !performing.value && !lockout.value && motorControllerState.value.serialConnected
)

const canSubmitBlade = computed(() =>
  isControlUnlocked.value && !performing.value && !lockout.value
)

// Helpers
function mapSecurityLevel(value: unknown): ManualControlSecurityConfig['auth_level'] {
  if (typeof value === 'string') {
    switch (value) {
      case 'password_only':
      case 'password':
        return 'password'
      case 'password_totp':
      case 'totp':
        return 'totp'
      case 'google_auth':
      case 'google':
        return 'google'
      case 'cloudflare_tunnel_auth':
      case 'tunnel':
      case 'cloudflare':
        return 'cloudflare'
      default:
        return 'password'
    }
  }

  if (typeof value === 'number') {
    if (value >= 4) return 'cloudflare'
    if (value === 3) return 'google'
    if (value === 2) return 'totp'
    return 'password'
  }

  return 'password'
}

function formatSecurityLevel(level: ManualControlSecurityConfig['auth_level']) {
  switch (level) {
    case 'password':
      return 'Password'
    case 'totp':
      return 'Password + TOTP'
    case 'google':
      return 'Google OAuth'
    case 'cloudflare':
      return 'Cloudflare Access'
    default:
      return 'Unknown'
  }
}

function formatSystemStatus(status: string) {
  switch (status?.toLowerCase()) {
    case 'nominal':
    case 'ok':
    case 'ready':
      return 'Ready'
    case 'active':
    case 'running':
      return 'Active'
    case 'caution':
    case 'warning':
      return 'Caution'
    case 'emergency':
    case 'fault':
      return 'Emergency Stop'
    default:
      return 'Unknown'
  }
}

function formatTimeRemaining(seconds: number) {
  if (!seconds || seconds <= 0) {
    return 'expired'
  }
  const mins = Math.floor(seconds / 60)
  const secs = seconds % 60
  if (mins === 0) {
    return `${secs}s`
  }
  return `${mins}m ${secs.toString().padStart(2, '0')}s`
}

function describeMotorError(code: string | null | undefined): string | null {
  if (!code) return null
  switch (code) {
    case 'usb_control_unavailable':
      return 'Waiting for RoboHAT to hand over USB control. Ensure the RC override is off.'
    case 'pwm_send_failed':
      return 'Motor PWM command failed to reach the controller. Check USB and power.'
    case 'blade_command_failed':
      return 'Blade command failed to reach the controller.'
    case 'emergency_stop_failed':
      return 'Emergency stop command did not acknowledge. Reissue if mower keeps moving.'
    case 'clear_emergency_failed':
      return 'Emergency clear command did not acknowledge. Verify controller connection.'
    default:
      return code.replace(/_/g, ' ')
  }
}

function describeMotorController(status: Record<string, any> | null | undefined): MotorControllerState {
  if (!status) {
    return {
      ready: false,
      serialConnected: true,
      severity: 'info',
      label: 'Awaiting status',
      message: 'No RoboHAT telemetry received yet.'
    }
  }

  const serialConnected = status.serial_connected === undefined ? true : Boolean(status.serial_connected)
  if (!serialConnected) {
    return {
      ready: false,
      serialConnected,
      severity: 'danger',
      label: 'Disconnected',
      message: 'Motor controller USB link not detected. Check cabling and power.'
    }
  }

  const errorMessage = describeMotorError(status.last_error)
  if (status.motor_controller_ok) {
    return {
      ready: true,
      serialConnected,
      severity: 'success',
      label: 'Ready',
      message: status.last_watchdog_echo ? `Echo: ${status.last_watchdog_echo}` : null
    }
  }

  if (errorMessage) {
    const severity = status.last_error === 'usb_control_unavailable' ? 'warning' : 'danger'
    const label = status.last_error === 'usb_control_unavailable' ? 'Handshake pending' : 'Action required'
    return {
      ready: false,
      serialConnected,
      severity,
      label,
      message: errorMessage
    }
  }

  return {
    ready: false,
    serialConnected,
    severity: 'info',
    label: 'Standby',
    message: 'Waiting for first motor command acknowledgement.'
  }
}

function showStatus(message: string, success: boolean, timeout = 4000) {
  statusMessage.value = message
  statusSuccess.value = success
  if (timeout > 0) {
    window.setTimeout(() => {
      statusMessage.value = ''
    }, timeout)
  }
}

function updateSessionTimer(expiresAt?: string) {
  if (sessionTimer) {
    window.clearInterval(sessionTimer)
    sessionTimer = undefined
  }

  const expiration = expiresAt ? new Date(expiresAt).getTime() : null

  if (!expiration) {
    sessionTimeRemaining.value = securityConfig.value.session_timeout_minutes * 60
    return
  }

  const tick = () => {
    const remaining = Math.max(0, Math.floor((expiration - Date.now()) / 1000))
    sessionTimeRemaining.value = remaining
    if (remaining === 0) {
      lockControl()
      showStatus('Manual control session expired', false)
      toast.show('Manual control session expired', 'warning', 4000)
    }
  }

  tick()
  sessionTimer = window.setInterval(tick, 1000)
}

async function loadSecurityConfig() {
  try {
    const response = await api.get('/api/v2/settings/security')
    const data = response.data ?? {}
    securityConfig.value = {
      auth_level: mapSecurityLevel(data.security_level ?? data.level),
      session_timeout_minutes: data.session_timeout_minutes ?? securityConfig.value.session_timeout_minutes,
      require_https: Boolean(data.require_https ?? securityConfig.value.require_https),
      auto_lock_manual_control: Boolean(data.auto_lock_manual_control ?? securityConfig.value.auto_lock_manual_control)
    }

    if (securityConfig.value.auth_level === 'cloudflare' && !cloudflareAutoVerificationAttempted) {
      cloudflareAutoVerificationAttempted = true
      await verifyCloudflareAuth()
    }
  } catch (error) {
    console.warn('Failed to load security configuration, using defaults.', error)
  }
}

async function refreshTelemetry() {
  try {
    const snapshot = await control.fetchRoboHATStatus()
    const source = (snapshot?.telemetry_source as ControlTelemetry['telemetry_source']) ?? telemetry.value.telemetry_source ?? 'unknown'

    if (source === 'hardware') {
      telemetry.value = {
        ...telemetry.value,
        ...snapshot,
        telemetry_source: 'hardware'
      }
      if (snapshot?.velocity?.linear?.x !== undefined && snapshot.velocity.linear.x !== null) {
        currentSpeed.value = Math.abs(Number(snapshot.velocity.linear.x))
      }
      const cameraSnapshot = snapshot?.camera as Partial<CameraStatusSummary> & { last_frame?: string | null } | undefined
      if (cameraSnapshot) {
        cameraInfo.active = Boolean(cameraSnapshot.active)
        cameraInfo.mode = cameraSnapshot.mode ?? cameraInfo.mode
        cameraInfo.fps = cameraSnapshot.fps ?? cameraInfo.fps
        cameraInfo.client_count = cameraSnapshot.client_count ?? cameraInfo.client_count
        if (cameraSnapshot.last_frame !== undefined) {
          cameraLastFrame.value = cameraSnapshot.last_frame ?? null
        }
      }
    } else {
      telemetry.value = {
        ...telemetry.value,
        safety_state: snapshot?.safety_state ?? telemetry.value.safety_state,
        telemetry_source: source,
        battery: undefined,
        position: undefined,
        velocity: undefined,
        camera: telemetry.value.camera
      }
      currentSpeed.value = 0
    }

    if (snapshot?.safety_state) {
      systemStatus.value = snapshot.safety_state
    }
  } catch (error) {
    // Non-fatal: keep previous telemetry
  }
}

function ensureSession() {
  if (!session.value) {
    session.value = {
      session_id: `local-${Date.now().toString(36)}`
    }
  }
  return session.value
}

async function attemptUnlock(payload: Record<string, unknown>): Promise<boolean> {
  const response = await api.post('/api/v2/control/manual-unlock', payload)
  const data = response.data ?? {}
  session.value = {
    session_id: data.session_id || `session-${Date.now().toString(36)}`,
    expires_at: data.expires_at
  }
  updateSessionTimer(data.expires_at)
  isControlUnlocked.value = true
  return true
}

async function autoUnlockIfPossible() {
  // Auth is not required for this deployment -- the backend auto-approves
  // manual-unlock regardless of method/credential when disabled. Try
  // silently on load so the security gate never shows in that case; any
  // failure just leaves the normal gate in place.
  try {
    await attemptUnlock({ method: securityConfig.value.auth_level })
  } catch {
    // Auth is actually required (or the backend rejected it) -- fall
    // through to the normal manual gate.
  }
}

async function authenticateControl() {
  if (!canAuthenticate.value || authenticating.value) return

  authenticating.value = true
  authError.value = ''

  const payload: Record<string, unknown> = {
    method: securityConfig.value.auth_level,
    password: authForm.password || undefined,
    totp_code: authForm.totpCode || undefined
  }

  try {
    await attemptUnlock(payload)
    toast.show('Manual control unlocked', 'success', 2500)
    showStatus('Manual control unlocked', true)
  } catch (error) {
    const axiosError = error as AxiosError
    const status = axiosError.response?.status
    if (status === 404 || status === 501) {
      // Backend endpoint unavailable – fail closed, do not grant control.
      const message = 'Manual control unlock is unavailable (backend endpoint missing).'
      authError.value = message
      toast.show(message, 'error', 5000)
      showStatus(message, false)
    } else {
      const message = (axiosError.response?.data as any)?.detail || axiosError.message || 'Authentication failed'
      authError.value = message
      showStatus(message, false)
    }
  } finally {
    authenticating.value = false
  }
}

function lockControl() {
  void stopMovement(true)
  isControlUnlocked.value = false
  joystickRef.value?.reset()
  session.value = null
  sessionTimeRemaining.value = 0
  if (sessionTimer) {
    window.clearInterval(sessionTimer)
    sessionTimer = undefined
  }
  currentSpeed.value = 0
  stopCameraFeed()
}

async function verifyCloudflareAuth() {
  try {
    const response = await api.get('/api/v2/control/manual-unlock/status')
    const data = response.data ?? {}
    if (data?.authorized) {
      session.value = {
        session_id: data.session_id || `session-${Date.now().toString(36)}`,
        expires_at: data.expires_at
      }
      updateSessionTimer(data.expires_at)
      isControlUnlocked.value = true
      showStatus('Cloudflare Access verified', true)
      toast.show('Cloudflare Access verified', 'success', 2500)
    } else {
      showStatus('Cloudflare verification failed', false)
    }
  } catch (error) {
    console.warn('Cloudflare verification failed.', error)
    const message = 'Cloudflare Access verification failed. Manual control remains locked.'
    toast.show(message, 'error', 5000)
    showStatus(message, false)
  }
}

function authenticateWithGoogle() {
  toast.show('Google authentication flow is not configured in this environment.', 'info', 4000)
}

function setPerforming(flag: boolean) {
  performing.value = flag
}

function clearMovementTimer() {
  if (movementRepeatTimer) {
    window.clearTimeout(movementRepeatTimer)
    movementRepeatTimer = undefined
  }
}

function scheduleMovementTick() {
  clearMovementTimer()
  if (!joystickEngaged.value) {
    return
  }
  movementRepeatTimer = window.setTimeout(() => {
    if (!joystickEngaged.value) {
      return
    }
    const vector = { ...activeDriveVector.value }
    queueDriveCommand(vector, currentDriveReason)
    scheduleMovementTick()
  }, MOVEMENT_REPEAT_INTERVAL_MS)
}

function clamp(value: number, min: number, max: number) {
  return Math.min(max, Math.max(min, value))
}

function buildDrivePayload(vector: { linear: number; angular: number }, reason = JOYSTICK_REASON, durationMs = MOVEMENT_DURATION_MS): DriveCommandPayload {
  const speedLimit = clamp(speedLevel.value / 100, 0, 1)
  return {
    session_id: ensureSession().session_id,
    vector: { ...vector },
    duration_ms: durationMs,
    reason,
    max_speed_limit: speedLimit
  }
}

function queueDriveCommand(vector: { linear: number; angular: number }, reason = JOYSTICK_REASON, durationMs = MOVEMENT_DURATION_MS, options: QueueDriveCommandOptions = {}) {
  if (!isControlUnlocked.value || lockout.value) {
    return driveDispatchPromise ?? Promise.resolve()
  }

  if (!motorControllerState.value.serialConnected) {
    showStatus('Motor controller offline – command not sent', false, 6000)
    return driveDispatchPromise ?? Promise.resolve()
  }

  if (
    !motorControllerState.value.ready &&
    motorControllerState.value.message &&
    statusMessage.value !== motorControllerState.value.message
  ) {
    showStatus(motorControllerState.value.message, false, 5000)
  }

  pendingDrivePayload = buildDrivePayload(vector, reason, durationMs)

  const immediate = options.immediate === true
  if (!driveCommandActive || immediate) {
    driveDispatchPromise = dispatchDriveCommands()
  }

  return driveDispatchPromise ?? Promise.resolve()
}

async function dispatchDriveCommands(): Promise<void> {
  if (driveCommandActive) {
    return driveDispatchPromise ?? Promise.resolve()
  }

  driveCommandActive = true
  try {
    while (pendingDrivePayload) {
      const payload = pendingDrivePayload
      pendingDrivePayload = null
      if (!payload) {
        continue
      }
      try {
        await control.submitCommand('drive', payload)
        const linearMag = Math.abs(payload.vector.linear ?? 0)
        const angularMag = Math.abs(payload.vector.angular ?? 0)
        currentSpeed.value = Math.max(linearMag, angularMag)
      } catch (error) {
        showStatus('Failed to send drive command', false)
      }
    }
  } finally {
    driveCommandActive = false
    driveDispatchPromise = null
  }
}

function computeDriveVectorFromJoystick(vector: { x: number; y: number }) {
  const speedFactor = clamp(speedLevel.value / 100, 0, 1)
  const linear = clamp(vector.y * speedFactor, -1, 1)
  const angular = clamp(vector.x * speedFactor, -1, 1)
  return { linear, angular }
}

function handleJoystickChange(vector: { x: number; y: number; magnitude: number; active: boolean }) {
  lastJoystickVector.value = { x: vector.x, y: vector.y }
  if (!isControlUnlocked.value) {
    joystickEngaged.value = false
    activeDriveVector.value = { linear: 0, angular: 0 }
    currentSpeed.value = 0
    return
  }

  const driveVector = computeDriveVectorFromJoystick(vector)
  activeDriveVector.value = driveVector
  const engaged = vector.active && (Math.abs(driveVector.linear) > 0.01 || Math.abs(driveVector.angular) > 0.01)

  if (engaged) {
    joystickEngaged.value = true
    currentDriveReason = JOYSTICK_REASON
    clearMovementTimer()
    queueDriveCommand({ ...driveVector }, currentDriveReason)
    scheduleMovementTick()
  } else if (joystickEngaged.value) {
    joystickEngaged.value = false
    void stopMovement(true)
  } else {
    currentSpeed.value = 0
  }
}

function handleJoystickEnd() {
  if (!joystickEngaged.value) {
    return
  }
  joystickEngaged.value = false
  void stopMovement(true)
}

function handleStopButton() {
  joystickRef.value?.reset()
  lastJoystickVector.value = { x: 0, y: 0 }
  joystickEngaged.value = false
  activeDriveVector.value = { linear: 0, angular: 0 }
  void stopMovement(true)
}

function formatCommandValue(value: number) {
  if (!Number.isFinite(value)) {
    return '0.00'
  }
  return value.toFixed(2)
}

async function stopMovement(sendStopCommand = true) {
  clearMovementTimer()
  joystickEngaged.value = false
  activeDriveVector.value = { linear: 0, angular: 0 }
  if (sendStopCommand && isControlUnlocked.value) {
    await queueDriveCommand({ linear: 0, angular: 0 }, 'manual-stop', 0, { immediate: true })
  }
  currentSpeed.value = 0
}

async function emergencyStop() {
  setPerforming(true)
  try {
    await control.submitCommand('emergency', { session_id: ensureSession().session_id })
    currentSpeed.value = 0
    mowingActive.value = false
    showStatus('Emergency stop activated', true)
  } catch (error) {
    showStatus('Failed to trigger emergency stop', false)
  } finally {
    setPerforming(false)
  }
}

async function toggleMowing() {
  if (!canSubmitBlade.value) return
  setPerforming(true)
  try {
    const action = mowingActive.value ? 'disable' : 'enable'
    const response = await control.submitCommand('blade', {
      session_id: ensureSession().session_id,
      action,
      reason: 'manual-control'
    })
    if (response?.result === 'blocked') {
      showStatus('Blade action blocked by safety system', false)
    } else {
      mowingActive.value = !mowingActive.value
      showStatus(mowingActive.value ? 'Mowing started' : 'Mowing stopped', true)
    }
  } catch (error) {
    showStatus('Failed to toggle mowing', false)
  } finally {
    setPerforming(false)
  }
}

async function returnToBase() {
  setPerforming(true)
  try {
    const result = await returnToBaseApi()
    if (result?.status === 'error') {
      showStatus(result.detail || 'Return to base unavailable (no home position set)', false)
    } else {
      showStatus('Return to base command sent', true)
    }
  } catch (error) {
    showStatus('Failed to send return-to-base command', false)
  } finally {
    setPerforming(false)
  }
}

async function pauseSystem() {
  setPerforming(true)
  try {
    await pauseAutonomous()
    showStatus('System paused', true)
  } catch (error) {
    showStatus('Failed to pause', false)
  } finally {
    setPerforming(false)
  }
}

async function resumeSystem() {
  setPerforming(true)
  try {
    await resumeAutonomous()
    showStatus('System resumed', true)
  } catch (error) {
    showStatus('Failed to resume', false)
  } finally {
    setPerforming(false)
  }
}

// Reactive updates from store events
watch(motorControllerState, (state) => {
  if (previousSerialConnected.value !== null && state.serialConnected !== previousSerialConnected.value) {
    if (!state.serialConnected) {
      toast.show('Motor controller disconnected', 'error', 4000)
      showStatus(state.message || 'Motor controller disconnected', false, 6000)
    } else {
      toast.show('Motor controller connected', 'success', 2500)
      showStatus('Motor controller connected', true, 3000)
    }
  }

  if (previousMotorReady.value !== null && state.ready !== previousMotorReady.value) {
    if (state.ready) {
      toast.show('Motor controller ready', 'success', 2500)
      showStatus('Motor controller ready', true, 2500)
    } else if (state.serialConnected) {
      toast.show(state.message || 'Motor controller not ready', 'warning', 4000)
      if (state.message) {
        showStatus(state.message, false, 6000)
      }
    }
  }

  previousSerialConnected.value = state.serialConnected
  previousMotorReady.value = state.ready
}, { immediate: true })

watch(lockout, (value) => {
  if (value) {
    lockControl()
    const reason = lockoutReason.value || 'Safety lockout active'
    showStatus(reason, false, 6000)
  }
})

watch(isControlUnlocked, (unlocked) => {
  if (unlocked) {
    startCameraFeed(true).catch(() => {
      /* errors handled inside startCameraFeed */
    })
  } else {
    stopCameraFeed()
    joystickRef.value?.reset()
    lastJoystickVector.value = { x: 0, y: 0 }
    joystickEngaged.value = false
    activeDriveVector.value = { linear: 0, angular: 0 }
  }
})

watch(lastCommandResult, (result) => {
  if (!result) return
  if (result.result === 'blocked' || result.result === 'error') {
    showStatus(result.status_reason || 'Command blocked', false)
  }
})

watch(lastEcho, (payload) => {
  if (!payload) return
  if (payload.telemetry) {
    telemetry.value = {
      ...telemetry.value,
      ...payload.telemetry
    }
  }
  if (payload.system_status) {
    systemStatus.value = payload.system_status
  }
})

watch(speedLevel, () => {
  if (!isControlUnlocked.value) {
    currentSpeed.value = 0
    return
  }
  if (!joystickEngaged.value) {
    currentSpeed.value = 0
    return
  }
  const driveVector = computeDriveVectorFromJoystick(lastJoystickVector.value)
  activeDriveVector.value = driveVector
  currentSpeed.value = Math.abs(driveVector.linear)
  clearMovementTimer()
  queueDriveCommand({ ...driveVector }, JOYSTICK_REASON)
  scheduleMovementTick()
})

let telemetryInterval: number | undefined

onMounted(async () => {
  await loadSecurityConfig()
  await autoUnlockIfPossible()
  await refreshTelemetry()
  telemetryInterval = window.setInterval(refreshTelemetry, 5000)
})

onUnmounted(() => {
  if (telemetryInterval) {
    window.clearInterval(telemetryInterval)
    telemetryInterval = undefined
  }
  if (sessionTimer) {
    window.clearInterval(sessionTimer)
    sessionTimer = undefined
  }
  void stopMovement(true)
  stopCameraFeed()
})
</script>

<style scoped>
.control-view {
  padding: 0;
}

.page-header {
  margin-bottom: 2rem;
}

.page-header h1 {
  margin-bottom: 0.5rem;
}

.security-gate {
  display: flex;
  justify-content: center;
  align-items: center;
  min-height: 60vh;
}

.card {
  background: var(--secondary-dark);
  border: 1px solid var(--primary-light);
  border-radius: 8px;
  margin-bottom: 2rem;
}

.card-header {
  background: var(--primary-dark);
  padding: 1rem 1.5rem;
  border-bottom: 1px solid var(--primary-light);
  border-radius: 8px 8px 0 0;
}

.card-header h3 {
  margin: 0;
  color: var(--accent-green);
  font-size: 1.25rem;
}

.card-body {
  padding: 1.5rem;
}

.controller-health {
  margin-top: 1rem;
  display: flex;
  flex-direction: column;
  gap: 0.4rem;
}

.controller-chip {
  display: inline-flex;
  align-items: center;
  gap: 0.75rem;
  padding: 0.4rem 0.9rem;
  border-radius: 999px;
  border: 1px solid var(--primary-light);
  background: rgba(255, 255, 255, 0.04);
  font-weight: 600;
  letter-spacing: 0.04em;
  text-transform: uppercase;
}

.controller-chip__label {
  font-size: 0.7rem;
  opacity: 0.75;
}

.controller-chip__value {
  font-size: 0.9rem;
}

.controller-chip--success {
  border-color: var(--accent-green);
  color: var(--accent-green);
}

.controller-chip--warning {
  border-color: #f6c75f;
  color: #f6c75f;
}

.controller-chip--danger {
  border-color: #ff6b6b;
  color: #ff6b6b;
}

.controller-chip--info {
  border-color: #7bc4ff;
  color: #7bc4ff;
}

.controller-chip__message {
  margin: 0;
  font-size: 0.85rem;
  color: rgba(255, 255, 255, 0.75);
}

.btn {
  padding: 0.75rem 1.5rem;
  border: none;
  border-radius: 4px;
  font-weight: 500;
  cursor: pointer;
  transition: all 0.3s ease;
}

.btn-primary {
  background: var(--accent-green);
  color: var(--primary-dark);
}

.btn-secondary {
  background: var(--primary-light);
  color: var(--text-color);
}

.btn-success {
  background: #28a745;
  color: white;
}

.btn-warning {
  background: #ffc107;
  color: #000;
}

.btn-info {
  background: #17a2b8;
  color: white;
}

.btn-emergency {
  background: #ff0000;
  color: white;
  font-size: 1.25rem;
  padding: 1rem 2rem;
  width: 100%;
  margin-bottom: 2rem;
  animation: pulse 2s infinite;
}

@keyframes pulse {
  0% { box-shadow: 0 0 0 0 rgba(255, 0, 0, 0.7); }
  70% { box-shadow: 0 0 0 10px rgba(255, 0, 0, 0); }
  100% { box-shadow: 0 0 0 0 rgba(255, 0, 0, 0); }
}

.btn:hover:not(:disabled) {
  transform: translateY(-2px);
}

.btn:disabled {
  opacity: 0.6;
  cursor: not-allowed;
}

.btn-sm {
  padding: 0.5rem 1rem;
  font-size: 0.875rem;
}

.control-status {
  display: flex;
  justify-content: space-between;
  align-items: center;
  background: var(--primary-dark);
  padding: 1rem;
  border-radius: 8px;
  margin-bottom: 2rem;
}

.status-indicator {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  font-weight: 600;
}

.status-light {
  width: 12px;
  height: 12px;
  border-radius: 50%;
  background: var(--accent-green);
}

.status-indicator.emergency .status-light {
  background: #ff0000;
  animation: blink 1s infinite;
}

@keyframes blink {
  0%, 50% { opacity: 1; }
  51%, 100% { opacity: 0.3; }
}

.session-info {
  display: flex;
  align-items: center;
  gap: 1rem;
  color: var(--text-muted);
}

.movement-layout {
  display: flex;
  gap: 2rem;
  align-items: stretch;
  flex-wrap: wrap;
}

.joystick-column {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 0.75rem;
}

.joystick-component {
  width: 220px;
  height: 220px;
  touch-action: none;
}

.joystick-hint {
  font-size: 0.85rem;
  color: var(--text-muted);
}

.movement-actions {
  flex: 1;
  display: flex;
  flex-direction: column;
  gap: 1rem;
  min-width: 200px;
}

.stop-button {
  align-self: flex-start;
  padding: 0.85rem 1.5rem;
}

.movement-readout {
  display: flex;
  gap: 1.5rem;
  font-family: 'Roboto Mono', 'Fira Code', monospace;
  font-size: 1rem;
  color: var(--text-muted);
}

.movement-status {
  font-weight: 600;
  color: var(--text-muted);
}

.movement-status.active {
  color: var(--accent-green);
}

.speed-control {
  margin-top: 1rem;
}

.speed-slider {
  width: 100%;
  margin-top: 0.5rem;
}

.mowing-controls, .system-controls {
  display: flex;
  gap: 1rem;
  flex-wrap: wrap;
}

.alert {
  padding: 1rem;
  border-radius: 4px;
  margin-top: 2rem;
}

.alert-success {
  background: rgba(0, 255, 146, 0.1);
  border: 1px solid var(--accent-green);
  color: var(--accent-green);
}

.alert-danger {
  background: rgba(255, 67, 67, 0.1);
  border: 1px solid #ff4343;
  color: #ff4343;
}

@media (max-width: 768px) {
  .control-status {
    flex-direction: column;
    gap: 1rem;
    text-align: center;
  }
  
  .movement-layout {
    flex-direction: column;
    align-items: center;
  }
  
  .mowing-controls, .system-controls {
    flex-direction: column;
  }
}
</style>