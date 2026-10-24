<template>
  <div class="tractor-view">
    <div class="page-header">
      <h1>Tractor Control</h1>
      <p class="text-muted">Manual actuation of the zero-turn mower</p>
    </div>

    <div class="tractor-grid">
      <!-- Status + engine/authorization -->
      <div class="card">
        <div class="card-header">
          <h3>Status</h3>
          <span class="badge" :class="state?.emergency_stop_active ? 'badge-estop' : (state?.engine === 'running' ? 'badge-on' : 'badge-off')">
            {{ state?.emergency_stop_active ? 'E-STOP' : String(state?.engine || 'off').toUpperCase() }}
          </span>
        </div>
        <div class="card-body">
          <div class="status-rows">
            <div class="row"><span>Engine</span><strong>{{ state?.engine ?? '—' }}</strong></div>
            <div class="row"><span>Blade (PTO)</span><strong>{{ state?.blade_engaged ? 'engaged' : 'off' }}</strong></div>
            <div class="row"><span>Moving</span><strong>{{ state?.moving ? 'yes' : 'no' }}</strong></div>
            <div class="row"><span>Authorized</span><strong>{{ state?.authorized ? 'yes' : 'no' }}</strong></div>
            <div v-if="state?.interlock_reason" class="row warn"><span>Interlock</span><strong>{{ state.interlock_reason }}</strong></div>
          </div>
          <div class="actions">
            <button
              class="btn"
              :class="state?.authorized ? 'btn-secondary' : 'btn-success'"
              :disabled="busy"
              @click="toggleAuthorize"
            >
              {{ state?.authorized ? 'Revoke' : 'Authorize' }}
            </button>
            <button class="btn btn-primary" :disabled="busy || state?.engine === 'running'" @click="startEngine">Start engine</button>
            <button class="btn btn-secondary" :disabled="busy || state?.engine !== 'running'" @click="stopEngine">Stop engine</button>
          </div>
          <div class="estop-row">
            <!-- Never disabled: the E-stop must stay clickable while any other request is in flight. -->
            <button class="btn btn-emergency" @click="emergencyStop">🛑 EMERGENCY STOP</button>
            <button
              v-if="state?.emergency_stop_active"
              class="btn btn-warning"
              :disabled="busy"
              @click="clearEmergency"
            >
              Clear E-stop
            </button>
          </div>
        </div>
      </div>

      <!-- Drivetrain: twin-lever hydrostatic drive, one servo per lever -->
      <div class="card">
        <div class="card-header"><h3>Drivetrain</h3></div>
        <div class="card-body">
          <label class="ctl">Left Lever <span data-testid="left-lever-command">{{ leftLever.toFixed(2) }} · {{ leverLabel(leftLever) }}</span></label>
          <input
            v-model.number="leftLever"
            type="range"
            min="-1"
            max="1"
            step="0.05"
            data-testid="left-lever-slider"
            :disabled="locked"
            @pointerdown="hold.grip('left', 'pointer', $event.pointerId)"
            @keydown="hold.grip('left')"
            @input="hold.input('left')"
            @keyup="hold.release('left')"
            @blur="hold.release('left')"
          >
          <div class="ctl-ends"><span>reverse</span><span>forward</span></div>
          <div class="held" data-testid="left-lever-held">Backend is holding {{ leverHeld(state?.left_lever) }}</div>

          <label class="ctl">Right Lever <span data-testid="right-lever-command">{{ rightLever.toFixed(2) }} · {{ leverLabel(rightLever) }}</span></label>
          <input
            v-model.number="rightLever"
            type="range"
            min="-1"
            max="1"
            step="0.05"
            data-testid="right-lever-slider"
            :disabled="locked"
            @pointerdown="hold.grip('right', 'pointer', $event.pointerId)"
            @keydown="hold.grip('right')"
            @input="hold.input('right')"
            @keyup="hold.release('right')"
            @blur="hold.release('right')"
          >
          <div class="ctl-ends"><span>reverse</span><span>forward</span></div>
          <div class="held" data-testid="right-lever-held">Backend is holding {{ leverHeld(state?.right_lever) }}</div>

          <button
            class="btn btn-secondary neutral-btn"
            :disabled="!!state?.emergency_stop_active"
            data-testid="neutral-button"
            @click="bothToNeutral"
          >
            Both levers to neutral
          </button>

          <div class="hold-status" :class="{ active: holding }" data-testid="hold-status">
            <template v-if="holding">
              Commanding: re-sent every {{ refreshMs }} ms<template v-if="holdMs"> (backend hold {{ holdMs }} ms)</template>
            </template>
            <template v-else>
              Not commanding anything from this page
            </template>
          </div>
          <div
            v-if="stopNotice"
            class="hold-notice"
            role="alert"
            data-testid="hold-notice"
          >
            {{ stopNotice }}
          </div>
          <label class="latch">
            <input
              v-model="latch"
              type="checkbox"
              data-testid="latch-toggle"
            >
            Latch sliders (keep commanding after release)
          </label>
          <p class="hint">Hold a slider to drive; releasing it sends neutral. While a control is held the page re-sends its command, and the backend returns the machine to neutral on its own if the commands stop.</p>
        </div>
      </div>

      <!-- Engine + implement -->
      <div class="card">
        <div class="card-header"><h3>Engine &amp; Implement</h3></div>
        <div class="card-body">
          <label class="ctl">Throttle (engine RPM) <span data-testid="throttle-command">{{ throttle.toFixed(2) }}</span></label>
          <input
            v-model.number="throttle"
            type="range"
            min="0"
            max="1"
            step="0.05"
            data-testid="throttle-slider"
            :disabled="locked"
            @pointerdown="hold.grip('throttle', 'pointer', $event.pointerId)"
            @keydown="hold.grip('throttle')"
            @input="hold.input('throttle')"
            @keyup="hold.release('throttle')"
            @blur="hold.release('throttle')"
          >
          <div class="ctl-ends"><span>idle</span><span>full</span></div>
          <div class="held" data-testid="throttle-held">Backend is holding {{ throttleHeld(state?.throttle) }}</div>

          <label class="ctl">Blade (PTO)</label>
          <button
            class="btn blade-btn"
            :class="state?.blade_engaged ? 'btn-danger' : 'btn-success'"
            :disabled="locked"
            @click="toggleBlade"
          >
            {{ state?.blade_engaged ? 'Disengage blade' : 'Engage blade' }}
          </button>
          <p class="hint">Blade engages only with the engine running and not reversing (both levers pulled back — a single lever back is just a pivot turn).</p>
        </div>
      </div>

      <!-- GPS -->
      <div class="card">
        <div class="card-header"><h3>GPS</h3></div>
        <div class="card-body">
          <div class="status-rows">
            <div class="row"><span>Status</span><strong>{{ gpsAccuracySummary }}</strong></div>
            <div class="row"><span>Latitude</span><strong>{{ gpsLatitude ?? '—' }}</strong></div>
            <div class="row"><span>Longitude</span><strong>{{ gpsLongitude ?? '—' }}</strong></div>
            <div class="row"><span>Satellites</span><strong>{{ gpsSatellitesDisplay }}</strong></div>
            <div class="row"><span>HDOP</span><strong>{{ gpsHdopDisplay }}</strong></div>
            <div class="row"><span>RTK</span><strong>{{ gpsRtkStatus ?? '—' }}</strong></div>
          </div>
        </div>
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
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed, onMounted, onUnmounted, ref } from 'vue'
import { useTractorStore } from '@/stores/tractor'
import { useCameraFeed } from '@/composables/useCameraFeed'
import { useTractorHold } from '@/composables/useTractorHold'
import { useWebSocket } from '@/services/websocket'
import { useToastStore } from '@/stores/toast'
import { describeApiError } from '@/utils/apiErrors'
import type { TractorSafetyResponse } from '@/types/control'
import CameraFeedCard from '@/components/control/CameraFeedCard.vue'

const tractor = useTractorStore()
const toast = useToastStore()

const state = computed(() => tractor.state)

// The sliders are what THIS page is commanding right now: they sit at 0 whenever it is not
// commanding anything. What the machine is actually holding is read from the backend state
// (the "Backend is holding" lines), so a lever the backend let go of can never keep showing
// the old slider value.
const leftLever = ref(0)
const rightLever = ref(0)
const throttle = ref(0)
const busy = ref(false)

const locked = computed(() => busy.value || !!state.value?.emergency_stop_active)

// Per-lever Forward/Neutral/Reverse label — matches the backend's 0.05 deadband
// for `moving` (a lever isn't "moving" until it clears +/-0.05). Note this is
// distinct from state.reversing, which requires BOTH levers negative.
function leverLabel(value: number): string {
  if (value > 0.05) return 'Forward'
  if (value < -0.05) return 'Reverse'
  return 'Neutral'
}

function leverHeld(value: number | undefined): string {
  return value === undefined ? '—' : `${value.toFixed(2)} · ${leverLabel(value)}`
}

function throttleHeld(value: number | undefined): string {
  return value === undefined ? '—' : value.toFixed(2)
}

function notify(msg: string, good = true) {
  toast.show(msg, good ? 'success' : 'error')
}

// Hold-to-run: while a slider is held the page re-sends its command inside the backend's
// command hold; see composables/useTractorHold.ts for every stop condition.
const hold = useTractorHold({
  values: { left: leftLever, right: rightLever, throttle },
  emergencyStopActive: computed(() => !!state.value?.emergency_stop_active),
  notify,
})
const { latch, holding, refreshMs, holdMs, stopNotice } = hold

function handle(result: TractorSafetyResponse | undefined, label: string) {
  if (!result) return
  if (result.status === 'rejected') {
    notify(`${label}: ${result.reason}`, false)
  } else if (result.status === 'partial') {
    const latched = result.still_latched?.length ? ` (still latched: ${result.still_latched.join(', ')})` : ''
    notify(`${label}: not fully applied${latched}`, false)
  } else if (result.status === 'not_enabled') {
    notify(`${label}: the tractor is not enabled on this backend`, false)
  } else if (result.complete === false) {
    // The latch is set but the backend could not confirm every output went safe.
    notify(`${label}: latched, but NOT every output was confirmed (${result.faults?.join('; ') || 'see the backend log'})`, false)
  }
}

async function call(fn: () => Promise<any>, label: string) {
  busy.value = true
  try {
    handle(await fn(), label)
  } catch (e: unknown) {
    notify(describeApiError(e, `${label} failed`), false)
  } finally {
    busy.value = false
  }
}

const toggleBlade = () => call(() => tractor.setBlade(!state.value?.blade_engaged), 'Blade')
const startEngine = () => call(() => tractor.startEngine(), 'Start engine')
const stopEngine = () => call(() => tractor.stopEngine(), 'Stop engine')
const clearEmergency = () => call(() => tractor.clearEmergency(), 'Clear E-stop')
const toggleAuthorize = () =>
  call(() => (state.value?.authorized ? tractor.revoke() : tractor.authorize()), 'Authorize')

// Every hold ends first, with no neutral writes of its own: the E-stop request drives
// neutral itself and the backend refuses further lever writes once it is latched.
const emergencyStop = () => {
  hold.halt('estop')
  return call(() => tractor.emergencyStop(), 'Emergency stop')
}

// Snap both drive levers to neutral in one click. Goes through the hold so the two writes
// are sequential (left, then right: concurrent writes each read the other lever's old
// state) and so a stream in progress stops.
const bothToNeutral = () => hold.neutralLevers()

// GPS display — telemetry.navigation is generic Pi-GPS telemetry, already
// flowing regardless of drivetrain. ponytail: same ~25-line glue as
// DashboardView.vue's gpsHdopDisplay/gpsSatellitesDisplay/gpsAccuracySummary,
// duplicated directly here rather than extracted; extract if a third
// consumer appears.
const gpsLatitude = ref<string | null>(null)
const gpsLongitude = ref<string | null>(null)
const gpsAccuracy = ref<number | null>(null)
const gpsHdop = ref<number | null>(null)
const gpsSatellites = ref<number | null>(null)
const gpsRtkStatus = ref<string | null>(null)

const hasGpsFix = computed(() => gpsLatitude.value !== null && gpsLongitude.value !== null)
const gpsHdopDisplay = computed(() => (gpsHdop.value === null ? '--' : gpsHdop.value.toFixed(2)))
const gpsSatellitesDisplay = computed(() => (gpsSatellites.value === null ? '--' : gpsSatellites.value.toString()))
const gpsAccuracySummary = computed(() => {
  if (!hasGpsFix.value) return 'NO SIGNAL'
  if (gpsAccuracy.value === null) return 'SIGNAL ACQUIRED'
  return `Accuracy ±${gpsAccuracy.value.toFixed(2)} m`
})

function applyNavigation(data: any) {
  const lat = data?.position?.latitude
  const lon = data?.position?.longitude
  if (typeof lat === 'number' && typeof lon === 'number') {
    gpsLatitude.value = lat.toFixed(6)
    gpsLongitude.value = lon.toFixed(6)
    gpsAccuracy.value = typeof data.position?.accuracy === 'number' ? data.position.accuracy : null
    gpsHdop.value =
      typeof (data.hdop ?? data.position?.hdop) === 'number' ? (data.hdop ?? data.position?.hdop) : null
    gpsSatellites.value = typeof data.position?.satellites === 'number' ? data.position.satellites : null
    gpsRtkStatus.value = data.position?.rtk_status ?? null
  } else {
    gpsLatitude.value = null
    gpsLongitude.value = null
    gpsAccuracy.value = null
    gpsHdop.value = null
    gpsSatellites.value = null
    gpsRtkStatus.value = null
  }
}

const { connect, subscribe, unsubscribe } = useWebSocket()

// Camera feed — extracted composable (see useCameraFeed.ts); this view has no
// manual-control auth session, so the MJPEG stream URL omits session_id.
const {
  cameraInfo,
  cameraError,
  cameraStatusMessage,
  cameraLastFrame,
  cameraDisplaySource,
  cameraIsStreaming,
  formatCameraFps,
  formatCameraTimestamp,
  handleCameraStreamLoad,
  handleCameraStreamError,
  startCameraFeed,
  stopCameraFeed,
  retryCameraFeed,
} = useCameraFeed()

onMounted(async () => {
  // One-time initial REST fetch for fast first paint; the store's WS
  // subscription (wired eagerly at store creation) drives everything after.
  // A failed fetch must not keep the telemetry and camera feeds from starting.
  await tractor.fetchState().catch(() => undefined)

  await connect()
  subscribe('telemetry.navigation', applyNavigation)
  startCameraFeed(true).catch(() => {
    /* errors surfaced via cameraError */
  })
})
onUnmounted(() => {
  unsubscribe('telemetry.navigation')
  stopCameraFeed()
})
</script>

<style scoped>
.tractor-view { padding: 0; }
.page-header { margin-bottom: 1.5rem; }
.page-header h1 { margin-bottom: 0.25rem; }
.tractor-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 1.25rem; }
.card { background: var(--secondary-dark); border: 1px solid var(--primary-light); border-radius: 8px; }
.card-header { display: flex; align-items: center; justify-content: space-between; padding: 1rem 1.25rem; background: var(--primary-dark); border-bottom: 1px solid var(--primary-light); border-radius: 8px 8px 0 0; }
.card-header h3 { margin: 0; font-size: 1.05rem; color: var(--accent-green); }
.card-body { padding: 1.25rem; }
.status-rows { display: flex; flex-direction: column; gap: 0.5rem; margin-bottom: 1rem; }
.status-rows .row { display: flex; justify-content: space-between; }
.status-rows .row span { color: var(--text-muted, #6c757d); }
.status-rows .row.warn strong { color: var(--danger); }
.text-muted { color: var(--text-muted); }
.actions { display: flex; flex-wrap: wrap; gap: 0.5rem; margin-bottom: 0.75rem; }
.estop-row { display: flex; gap: 0.5rem; align-items: center; }
.btn-emergency {
  background: linear-gradient(135deg, #2a0a12, #1a0a0a, #0f0505);
  border: 2px solid #ff0040;
  color: #ff0040;
  font-family: 'Orbitron', 'Courier New', monospace;
  font-weight: 700;
  font-size: 1.1rem;
  letter-spacing: 2px;
  text-transform: uppercase;
  text-shadow: 0 0 10px rgba(255, 0, 64, 0.7);
  padding: 1rem 1.5rem;
  flex: 1;
  border-radius: 6px;
  backdrop-filter: blur(10px);
  transition: background 0.2s ease, color 0.2s ease;
  animation: emergencyFlash 1.5s ease-in-out infinite;
}
.btn-emergency:hover:not(:disabled),
.btn-emergency:focus-visible:not(:disabled) {
  background: linear-gradient(135deg, #ff0040, #0a0a0a);
  color: #000;
  text-shadow: none;
  animation-duration: 0.5s;
}
.btn-emergency:disabled { animation: none; box-shadow: none; }
@keyframes emergencyFlash {
  0%, 100% { box-shadow: 0 0 20px rgba(255, 0, 64, 0.5); }
  50% { box-shadow: 0 0 40px rgba(255, 0, 64, 0.9); }
}
.ctl { display: flex; justify-content: space-between; font-size: 0.9rem; color: var(--text-muted, #6c757d); margin: 0.75rem 0 0.25rem; font-weight: 600; }
.ctl-ends { display: flex; justify-content: space-between; font-size: 0.75rem; color: var(--text-muted, #6c757d); }
.held { margin-top: 0.25rem; font-size: 0.8rem; color: var(--text-muted, #6c757d); }
input[type="range"] {
  -webkit-appearance: none;
  appearance: none;
  width: 100%;
  height: 24px;
  background: transparent;
  /* A drag must stay a drag: a touch scroll would cancel the pointer and release the control. */
  touch-action: none;
}
input[type="range"]::-webkit-slider-runnable-track,
input[type="range"]::-moz-range-track {
  height: 4px;
  background: var(--primary-light);
  border-radius: 2px;
}
input[type="range"]::-webkit-slider-thumb {
  -webkit-appearance: none;
  appearance: none;
  width: 16px;
  height: 16px;
  border-radius: 50%;
  background: var(--accent-green);
  cursor: pointer;
  margin-top: -6px;
}
input[type="range"]::-moz-range-thumb {
  width: 16px;
  height: 16px;
  border-radius: 50%;
  background: var(--accent-green);
  cursor: pointer;
  border: none;
}
input[type="range"]:hover:not(:disabled)::-webkit-slider-thumb,
input[type="range"]:hover:not(:disabled)::-moz-range-thumb { background: var(--accent-green-hover); }
input[type="range"]:focus-visible { outline: 2px solid var(--accent-green); outline-offset: 2px; }
input[type="range"]:disabled { opacity: 0.5; }
.blade-btn { width: 100%; }
.neutral-btn { width: 100%; margin-top: 0.75rem; }
.hold-status { margin-top: 0.75rem; padding: 0.4rem 0.6rem; border: 1px solid var(--primary-light); border-radius: 4px; font-size: 0.85rem; color: var(--text-muted, #6c757d); }
.hold-status.active { border-color: var(--accent-green); color: var(--accent-green); }
.hold-notice { margin-top: 0.5rem; padding: 0.5rem 0.75rem; border: 1px solid var(--danger); border-radius: 4px; background: rgba(255,67,67,0.15); color: var(--danger); font-size: 0.85rem; }
.latch { display: flex; align-items: center; gap: 0.5rem; margin-top: 0.75rem; font-size: 0.85rem; }
.hint { margin-top: 0.5rem; font-size: 0.85rem; color: var(--text-muted, #6c757d); }
.badge { padding: 0.25rem 0.6rem; border-radius: 4px; font-size: 0.75rem; font-weight: 700; }
.badge-on { background: rgba(0,255,146,0.2); color: var(--accent-green); border: 1px solid var(--accent-green); }
.badge-off { background: rgba(45,55,72,0.4); color: var(--text-muted); border: 1px solid var(--primary-light); }
.badge-estop { background: rgba(255,67,67,0.2); color: var(--danger); border: 1px solid var(--danger); }
.btn { padding: 0.75rem 1.5rem; border: none; border-radius: 4px; font-weight: 500; cursor: pointer; transition: all 0.3s ease; }
.btn-primary { background: var(--accent-green); color: var(--primary-dark); }
.btn-secondary { background: var(--primary-light); color: var(--text-color); }
.btn-success { background: var(--accent-green); color: var(--primary-dark); }
.btn-warning { background: var(--warning); color: #000; }
.btn-danger { background: var(--danger); color: white; }
.btn:hover:not(:disabled) { transform: translateY(-2px); }
.btn:disabled { opacity: 0.6; cursor: not-allowed; }
</style>
