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
          <span class="badge" :class="state.emergency_stop_active ? 'badge-estop' : (state.engine === 'running' ? 'badge-on' : 'badge-off')">
            {{ state.emergency_stop_active ? 'E-STOP' : String(state.engine || 'off').toUpperCase() }}
          </span>
        </div>
        <div class="card-body">
          <div class="status-rows">
            <div class="row"><span>Engine</span><strong>{{ state.engine ?? '—' }}</strong></div>
            <div class="row"><span>Gear</span><strong>{{ state.gear ?? '—' }}</strong></div>
            <div class="row"><span>Blade (PTO)</span><strong>{{ state.blade_engaged ? 'engaged' : 'off' }}</strong></div>
            <div class="row"><span>Moving</span><strong>{{ state.moving ? 'yes' : 'no' }}</strong></div>
            <div class="row"><span>Authorized</span><strong>{{ state.authorized ? 'yes' : 'no' }}</strong></div>
            <div v-if="state.interlock_reason" class="row warn"><span>Interlock</span><strong>{{ state.interlock_reason }}</strong></div>
          </div>
          <div class="actions">
            <button
              class="btn"
              :class="state.authorized ? 'btn-secondary' : 'btn-success'"
              :disabled="busy"
              @click="toggleAuthorize"
            >
              {{ state.authorized ? 'Revoke' : 'Authorize' }}
            </button>
            <button class="btn btn-primary" :disabled="busy || state.engine === 'running'" @click="startEngine">Start engine</button>
            <button class="btn btn-secondary" :disabled="busy || state.engine !== 'running'" @click="stopEngine">Stop engine</button>
          </div>
          <div class="estop-row">
            <button class="btn btn-emergency" :disabled="busy" @click="emergencyStop">🛑 EMERGENCY STOP</button>
            <button
              v-if="state.emergency_stop_active"
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
              :class="state.gear === g ? 'btn-primary' : 'btn-secondary'"
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
            :class="state.blade_engaged ? 'btn-danger' : 'btn-success'"
            :disabled="locked"
            @click="toggleBlade"
          >
            {{ state.blade_engaged ? 'Disengage blade' : 'Engage blade' }}
          </button>
          <p class="hint">Blade engages only with the engine running and not in reverse.</p>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed, onMounted, onUnmounted, ref } from 'vue'
import {
  getTractorState,
  tractorAuthorize,
  tractorBlade,
  tractorClearEmergency,
  tractorClutch,
  tractorEmergencyStop,
  tractorGear,
  tractorSpeed,
  tractorStart,
  tractorSteering,
  tractorStopEngine,
  tractorThrottle,
  type TractorGear,
} from '@/services/api'
import { useToastStore } from '@/stores/toast'

interface TractorState {
  steering: number
  throttle: number
  ground_speed: number
  gear: TractorGear
  clutch: number
  blade_engaged: boolean
  engine: 'off' | 'starting' | 'running'
  emergency_stop_active: boolean
  authorized: boolean
  interlock_reason: string | null
  moving?: boolean
}

const gears: TractorGear[] = ['reverse', 'neutral', 'forward']

const state = ref<Partial<TractorState>>({})
const steering = ref(0)
const throttle = ref(0)
const groundSpeed = ref(0)
const clutch = ref(1)
const busy = ref(false)
const toast = useToastStore()
let timer: ReturnType<typeof setInterval> | null = null

const locked = computed(() => busy.value || !!state.value.emergency_stop_active)

function notify(msg: string, good = true) {
  toast.show(msg, good ? 'success' : 'error')
}

function syncFromState() {
  if (typeof state.value.steering === 'number') steering.value = state.value.steering
  if (typeof state.value.throttle === 'number') throttle.value = state.value.throttle
  if (typeof state.value.ground_speed === 'number') groundSpeed.value = state.value.ground_speed
  if (typeof state.value.clutch === 'number') clutch.value = state.value.clutch
}

async function refresh() {
  try {
    const s = await getTractorState()
    s.moving =
      s.engine === 'running' && s.gear !== 'neutral' && s.clutch < 0.5 && s.ground_speed > 0
    state.value = s
  } catch {
    /* keep last */
  }
}

function handle(result: { status?: string; reason?: string }, label: string) {
  if (result?.status === 'rejected') notify(`${label}: ${result.reason}`, false)
}

async function call(fn: () => Promise<any>, label: string, refreshAfter = true) {
  busy.value = true
  try {
    handle(await fn(), label)
  } catch (e: unknown) {
    const detail = (e as { response?: { data?: { detail?: string } } })?.response?.data?.detail
    notify(detail || `${label} failed`, false)
  } finally {
    busy.value = false
    if (refreshAfter) await refresh()
  }
}

const onSteering = () => call(() => tractorSteering(steering.value), 'Steering', false)
const onThrottle = () => call(() => tractorThrottle(throttle.value), 'Throttle', false)
const onSpeed = () => call(() => tractorSpeed(groundSpeed.value), 'Ground speed', false)
const onClutch = () => call(() => tractorClutch(clutch.value), 'Clutch', false)
const setGear = (g: TractorGear) => call(() => tractorGear(g), 'Gear')
const toggleBlade = () => call(() => tractorBlade(!state.value.blade_engaged), 'Blade')
const startEngine = () => call(() => tractorStart(), 'Start engine')
const stopEngine = () => call(() => tractorStopEngine(), 'Stop engine')
const emergencyStop = () => call(() => tractorEmergencyStop(), 'Emergency stop')
const clearEmergency = () => call(() => tractorClearEmergency(), 'Clear E-stop')
const toggleAuthorize = () => call(() => tractorAuthorize(!state.value.authorized), 'Authorize')

onMounted(async () => {
  await refresh()
  syncFromState()
  timer = setInterval(refresh, 1500)
})
onUnmounted(() => {
  if (timer) clearInterval(timer)
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
