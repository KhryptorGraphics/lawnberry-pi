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
            <button class="btn btn-emergency" :disabled="busy" @click="emergencyStop">🛑 EMERGENCY STOP</button>
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
          <label class="ctl">Left Lever <span>{{ leftLever.toFixed(2) }} · {{ leverLabel(leftLever) }}</span></label>
          <input
            v-model.number="leftLever"
            type="range"
            min="-1"
            max="1"
            step="0.05"
            data-testid="left-lever-slider"
            :disabled="locked"
            @change="onLeftLever"
          >
          <div class="ctl-ends"><span>reverse</span><span>forward</span></div>

          <label class="ctl">Right Lever <span>{{ rightLever.toFixed(2) }} · {{ leverLabel(rightLever) }}</span></label>
          <input
            v-model.number="rightLever"
            type="range"
            min="-1"
            max="1"
            step="0.05"
            data-testid="right-lever-slider"
            :disabled="locked"
            @change="onRightLever"
          >
          <div class="ctl-ends"><span>reverse</span><span>forward</span></div>

          <button class="btn btn-secondary neutral-btn" :disabled="locked" @click="bothToNeutral">
            Both levers to neutral
          </button>
        </div>
      </div>

      <!-- Engine + implement -->
      <div class="card">
        <div class="card-header"><h3>Engine &amp; Implement</h3></div>
        <div class="card-body">
          <label class="ctl">Throttle (engine RPM) <span>{{ throttle.toFixed(2) }}</span></label>
          <input
            v-model.number="throttle"
            type="range"
            min="0"
            max="1"
            step="0.05"
            :disabled="locked"
            @change="onThrottle"
          >
          <div class="ctl-ends"><span>idle</span><span>full</span></div>

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
import { useWebSocket } from '@/services/websocket'
import { useToastStore } from '@/stores/toast'
import CameraFeedCard from '@/components/control/CameraFeedCard.vue'

const tractor = useTractorStore()
const toast = useToastStore()

const state = computed(() => tractor.state)

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

function notify(msg: string, good = true) {
  toast.show(msg, good ? 'success' : 'error')
}

function syncFromState() {
  const s = state.value
  if (!s) return
  leftLever.value = s.left_lever
  rightLever.value = s.right_lever
  throttle.value = s.throttle
}

function handle(result: { status?: string; reason?: string }, label: string) {
  if (result?.status === 'rejected') notify(`${label}: ${result.reason}`, false)
}

async function call(fn: () => Promise<any>, label: string) {
  busy.value = true
  try {
    handle(await fn(), label)
  } catch (e: unknown) {
    const detail = (e as { response?: { data?: { detail?: string } } })?.response?.data?.detail
    notify(detail || `${label} failed`, false)
  } finally {
    busy.value = false
  }
}

const onLeftLever = () => call(() => tractor.setLeftLever(leftLever.value), 'Left lever')
const onRightLever = () => call(() => tractor.setRightLever(rightLever.value), 'Right lever')
const onThrottle = () => call(() => tractor.setThrottle(throttle.value), 'Throttle')
const toggleBlade = () => call(() => tractor.setBlade(!state.value?.blade_engaged), 'Blade')
const startEngine = () => call(() => tractor.startEngine(), 'Start engine')
const stopEngine = () => call(() => tractor.stopEngine(), 'Stop engine')
const emergencyStop = () => call(() => tractor.emergencyStop(), 'Emergency stop')
const clearEmergency = () => call(() => tractor.clearEmergency(), 'Clear E-stop')
const toggleAuthorize = () =>
  call(() => (state.value?.authorized ? tractor.revoke() : tractor.authorize()), 'Authorize')

// Convenience: snap both drive levers to neutral in one click. Instant local
// feedback (matches slider-drag UX) then a single busy-flag cycle covering
// both requests; surfaces whichever leg is rejected, if either is.
const bothToNeutral = () => {
  leftLever.value = 0
  rightLever.value = 0
  return call(async () => {
    const [left, right] = await Promise.all([tractor.setLeftLever(0), tractor.setRightLever(0)])
    return left.status === 'rejected' ? left : right
  }, 'Neutral')
}

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
  await tractor.fetchState()
  syncFromState()

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
input[type="range"] {
  -webkit-appearance: none;
  appearance: none;
  width: 100%;
  height: 24px;
  background: transparent;
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
