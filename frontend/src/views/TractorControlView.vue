<template>
  <div class="tractor-view">
    <div class="page-header">
      <h1>Tractor Control</h1>
      <p class="text-muted">Manual actuation of the ride-on lawn tractor</p>
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
            <div class="row"><span>Gear</span><strong>{{ state?.gear ?? '—' }}</strong></div>
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

      <!-- Drivetrain -->
      <div class="card">
        <div class="card-header"><h3>Drivetrain</h3></div>
        <div class="card-body">
          <label class="ctl">Steering <span>{{ steering.toFixed(2) }}</span></label>
          <input
            v-model.number="steering"
            type="range"
            min="-1"
            max="1"
            step="0.05"
            :disabled="locked"
            @change="onSteering"
          >
          <div class="ctl-ends"><span>left</span><span>right</span></div>

          <label class="ctl">Gear</label>
          <div class="gear-buttons">
            <button
              v-for="g in gears"
              :key="g"
              class="btn"
              :class="state?.gear === g ? 'btn-primary' : 'btn-secondary'"
              :disabled="locked"
              @click="setGear(g)"
            >
              {{ g }}
            </button>
          </div>

          <label class="ctl">Ground speed (gas pedal) <span>{{ groundSpeed.toFixed(2) }}</span></label>
          <input
            v-model.number="groundSpeed"
            type="range"
            min="0"
            max="1"
            step="0.05"
            :disabled="locked"
            @change="onSpeed"
          >

          <label class="ctl">Clutch / brake <span>{{ clutch.toFixed(2) }}</span></label>
          <input
            v-model.number="clutch"
            type="range"
            min="0"
            max="1"
            step="0.05"
            :disabled="locked"
            @change="onClutch"
          >
          <div class="ctl-ends"><span>released</span><span>pressed</span></div>
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
          <p class="hint">Blade engages only with the engine running and not in reverse.</p>
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
import type { Transmission } from '@/types/control'

const gears: Transmission[] = ['reverse', 'neutral', 'forward']

const tractor = useTractorStore()
const toast = useToastStore()

const state = computed(() => tractor.state)

const steering = ref(0)
const throttle = ref(0)
const groundSpeed = ref(0)
const clutch = ref(1)
const busy = ref(false)

const locked = computed(() => busy.value || !!state.value?.emergency_stop_active)

function notify(msg: string, good = true) {
  toast.show(msg, good ? 'success' : 'error')
}

function syncFromState() {
  const s = state.value
  if (!s) return
  steering.value = s.steering
  throttle.value = s.throttle
  groundSpeed.value = s.ground_speed
  clutch.value = s.clutch
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

const onSteering = () => call(() => tractor.setSteering(steering.value), 'Steering')
const onThrottle = () => call(() => tractor.setThrottle(throttle.value), 'Throttle')
const onSpeed = () => call(() => tractor.setGroundSpeed(groundSpeed.value), 'Ground speed')
const onClutch = () => call(() => tractor.setClutch(clutch.value), 'Clutch')
const setGear = (g: Transmission) => call(() => tractor.setGear(g), 'Gear')
const toggleBlade = () => call(() => tractor.setBlade(!state.value?.blade_engaged), 'Blade')
const startEngine = () => call(() => tractor.startEngine(), 'Start engine')
const stopEngine = () => call(() => tractor.stopEngine(), 'Stop engine')
const emergencyStop = () => call(() => tractor.emergencyStop(), 'Emergency stop')
const clearEmergency = () => call(() => tractor.clearEmergency(), 'Clear E-stop')
const toggleAuthorize = () =>
  call(() => (state.value?.authorized ? tractor.revoke() : tractor.authorize()), 'Authorize')

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
.card { background: var(--card-bg, #fff); border-radius: 12px; box-shadow: 0 1px 3px rgba(0,0,0,0.08); }
.card-header { display: flex; align-items: center; justify-content: space-between; padding: 1rem 1.25rem; border-bottom: 1px solid var(--border-color, #eee); }
.card-header h3 { margin: 0; font-size: 1.05rem; }
.card-body { padding: 1.25rem; }
.status-rows { display: flex; flex-direction: column; gap: 0.5rem; margin-bottom: 1rem; }
.status-rows .row { display: flex; justify-content: space-between; }
.status-rows .row span { color: var(--text-muted, #6c757d); }
.status-rows .row.warn strong { color: #b42318; }
.actions { display: flex; flex-wrap: wrap; gap: 0.5rem; margin-bottom: 0.75rem; }
.estop-row { display: flex; gap: 0.5rem; align-items: center; }
.btn-emergency { background: #b42318; color: #fff; font-weight: 700; flex: 1; padding: 0.75rem; border: none; border-radius: 8px; }
.ctl { display: flex; justify-content: space-between; font-size: 0.9rem; color: var(--text-muted, #6c757d); margin: 0.75rem 0 0.25rem; font-weight: 600; }
.ctl-ends { display: flex; justify-content: space-between; font-size: 0.75rem; color: var(--text-muted, #6c757d); }
input[type="range"] { width: 100%; }
.gear-buttons { display: flex; gap: 0.5rem; }
.gear-buttons .btn { flex: 1; text-transform: capitalize; }
.blade-btn { width: 100%; }
.hint { margin-top: 0.5rem; font-size: 0.85rem; color: var(--text-muted, #6c757d); }
.badge { padding: 0.25rem 0.6rem; border-radius: 999px; font-size: 0.75rem; font-weight: 700; }
.badge-on { background: #d1f7e0; color: #0c7a43; }
.badge-off { background: #eceff1; color: #607d8b; }
.badge-estop { background: #fde2e1; color: #b42318; }
</style>
