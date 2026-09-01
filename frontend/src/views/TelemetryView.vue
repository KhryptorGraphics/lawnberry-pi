<template>
  <div class="telemetry-view">
    <div class="page-header">
      <h1>Telemetry</h1>
      <p class="text-muted">Real-time sensor data, diagnostics, and export</p>
    </div>

    <div class="toolbar">
      <span class="badge" :class="connClass">{{ connectionLabel }}</span>
      <span v-if="latencyBadge" class="badge" :class="`badge-${latencyBadge.status}`">
        Latency {{ Math.round(latencyBadge.latency_ms) }} ms · target {{ latencyBadge.target_ms }} ms ({{ latencyBadge.device }})
      </span>
      <span class="spacer" />
      <button class="btn btn-secondary" :disabled="isLoading" @click="refresh">
        {{ isLoading ? 'Refreshing…' : 'Refresh' }}
      </button>
      <button class="btn btn-primary" :disabled="isLoading" @click="exportDiagnostic">Export</button>
    </div>

    <p v-if="error" class="error-banner">{{ error }}</p>

    <div class="metric-grid">
      <MetricWidget
        label="RTK Fix"
        :value="rtkFixLabel"
        icon="🛰️"
        :variant="rtkVariant"
      />
      <MetricWidget label="Satellites" :value="rtkStatus?.satellites ?? '—'" icon="📡" />
      <MetricWidget
        label="HDOP"
        :value="hdopDisplay"
        icon="🎯"
        :variant="hdopVariant"
      />
      <MetricWidget
        label="Battery"
        :value="batterySoc"
        unit="%"
        icon="🔋"
        :variant="batteryVariant"
        :show-progress="batterySoc !== '—'"
        :max-value="100"
        progress-label="charge"
      />
      <MetricWidget
        label="Battery Voltage"
        :value="fmt(powerMetrics?.battery?.voltage)"
        unit="V"
        icon="⚡"
      />
      <MetricWidget
        label="Solar Power"
        :value="fmt(powerMetrics?.solar?.power)"
        unit="W"
        icon="☀️"
      />
      <MetricWidget
        label="Roll"
        :value="fmt(imuOrientation?.roll_deg)"
        unit="°"
        icon="🧭"
      />
      <MetricWidget
        label="Pitch"
        :value="fmt(imuOrientation?.pitch_deg)"
        unit="°"
        icon="🧭"
      />
      <MetricWidget
        label="Yaw"
        :value="fmt(imuOrientation?.yaw_deg)"
        unit="°"
        icon="🧭"
      />
    </div>

    <section class="panel-section">
      <h2>RTK / GPS</h2>
      <RtkDiagnosticsPanel />
    </section>

    <section v-if="imuOrientation" class="panel-section">
      <h2>IMU calibration</h2>
      <div class="metric-grid">
        <MetricWidget
          label="System"
          :value="imuOrientation.calibration_sys"
          :max-value="3"
          show-progress
          :variant="calVariant(imuOrientation.calibration_sys)"
        />
        <MetricWidget
          label="Gyro"
          :value="imuOrientation.calibration_gyro"
          :max-value="3"
          show-progress
          :variant="calVariant(imuOrientation.calibration_gyro)"
        />
        <MetricWidget
          label="Accel"
          :value="imuOrientation.calibration_accel"
          :max-value="3"
          show-progress
          :variant="calVariant(imuOrientation.calibration_accel)"
        />
        <MetricWidget
          label="Mag"
          :value="imuOrientation.calibration_mag"
          :max-value="3"
          show-progress
          :variant="calVariant(imuOrientation.calibration_mag)"
        />
      </div>
    </section>

    <section v-if="hardwareStreams.length" class="panel-section">
      <h2>Hardware streams</h2>
      <div class="card">
        <table class="stream-table">
          <thead>
            <tr><th>Component</th><th>Status</th><th>Latency</th><th>Updated</th></tr>
          </thead>
          <tbody>
            <tr v-for="s in hardwareStreams" :key="s.component_id">
              <td>{{ s.component_id }}</td>
              <td><span class="badge" :class="`badge-${streamBadge(s.status)}`">{{ s.status }}</span></td>
              <td>{{ Math.round(s.latency_ms) }} ms</td>
              <td>{{ shortTime(s.timestamp) }}</td>
            </tr>
          </tbody>
        </table>
      </div>
    </section>

    <p v-else-if="!isLoading" class="text-muted empty">
      No hardware streams reported yet. Telemetry populates once sensors are publishing.
    </p>
  </div>
</template>

<script setup lang="ts">
import { computed, onMounted, onUnmounted } from 'vue'
import { storeToRefs } from 'pinia'
import { useSystemStore } from '@/stores/system'
import MetricWidget from '@/components/MetricWidget.vue'
import RtkDiagnosticsPanel from '@/components/RtkDiagnosticsPanel.vue'

const system = useSystemStore()
const {
  connectionStatus,
  isLoading,
  error,
  latencyBadge,
  rtkStatus,
  imuOrientation,
  powerMetrics,
  hardwareStreams,
} = storeToRefs(system)

let timer: ReturnType<typeof setInterval> | null = null

async function refresh() {
  await system.loadTelemetryStream()
}

async function exportDiagnostic() {
  try {
    await system.exportTelemetryDiagnostic()
  } catch {
    /* error surfaced via store.error */
  }
}

const connectionLabel = computed(() => connectionStatus.value)
const connClass = computed(() =>
  connectionStatus.value === 'connected'
    ? 'badge-healthy'
    : connectionStatus.value === 'error'
      ? 'badge-critical'
      : 'badge-warning'
)

const RTK_LABELS: Record<string, string> = {
  no_fix: 'No fix',
  gps_fix: 'GPS',
  dgps_fix: 'DGPS',
  rtk_float: 'RTK float',
  rtk_fixed: 'RTK fixed',
}
const rtkFixLabel = computed(() =>
  rtkStatus.value ? (RTK_LABELS[rtkStatus.value.fix_type] ?? rtkStatus.value.fix_type) : '—'
)
const rtkVariant = computed(() => {
  const f = rtkStatus.value?.fix_type
  if (f === 'rtk_fixed') return 'success'
  if (f === 'rtk_float' || f === 'dgps_fix') return 'warning'
  return 'danger'
})
const hdopDisplay = computed(() => (rtkStatus.value ? rtkStatus.value.hdop.toFixed(2) : '—'))
const hdopVariant = computed(() => {
  const h = rtkStatus.value?.hdop
  if (h == null) return 'info'
  if (h <= 1.0) return 'success'
  if (h <= 2.0) return 'warning'
  return 'danger'
})

const batterySoc = computed<number | string>(() => {
  const soc = powerMetrics.value?.battery?.soc_percent
  return soc == null ? '—' : Math.round(soc)
})
const batteryVariant = computed(() => {
  const soc = powerMetrics.value?.battery?.soc_percent
  if (soc == null) return 'info'
  if (soc > 50) return 'success'
  if (soc > 20) return 'warning'
  return 'danger'
})

function fmt(v: number | null | undefined): string {
  return v == null ? '—' : v.toFixed(1)
}
function calVariant(level: number): 'success' | 'warning' | 'danger' {
  return level >= 3 ? 'success' : level >= 1 ? 'warning' : 'danger'
}
function streamBadge(status: string): string {
  return status === 'healthy' ? 'healthy' : status === 'warning' ? 'warning' : 'critical'
}
function shortTime(ts: string): string {
  const d = new Date(ts)
  return isNaN(d.getTime()) ? ts : d.toLocaleTimeString()
}

onMounted(async () => {
  await refresh()
  timer = setInterval(refresh, 3000)
})
onUnmounted(() => {
  if (timer) clearInterval(timer)
})
</script>

<style scoped>
.telemetry-view { padding: 0; }
.page-header { margin-bottom: 1.5rem; }
.page-header h1 { margin-bottom: 0.25rem; }
.toolbar { display: flex; align-items: center; gap: 0.75rem; margin-bottom: 1rem; flex-wrap: wrap; }
.spacer { flex: 1; }
.metric-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
  gap: 1rem;
  margin-bottom: 1.5rem;
}
.panel-section { margin-bottom: 1.5rem; }
.panel-section h2 { font-size: 1.1rem; margin-bottom: 0.75rem; }
.text-muted { color: var(--text-muted); }
.badge { padding: 0.25rem 0.6rem; border-radius: 999px; font-size: 0.8rem; font-weight: 600; }
.badge-healthy { background: rgba(0,255,0,0.15); color: #00ff00; border: 1px solid #00ff00; text-shadow: 0 0 10px rgba(0,255,0,0.7); }
.badge-warning { background: rgba(255,255,0,0.15); color: #ffff00; border: 1px solid #ffff00; text-shadow: 0 0 10px rgba(255,255,0,0.7); }
.badge-critical { background: rgba(255,0,64,0.15); color: #ff0040; border: 1px solid #ff0040; text-shadow: 0 0 10px rgba(255,0,64,0.7); }
.error-banner { background: rgba(255,0,64,0.12); color: #ff0040; border: 1px solid rgba(255,0,64,0.35); padding: 0.75rem 1rem; border-radius: 8px; margin-bottom: 1rem; }
.empty { padding: 1rem 0; }
.card {
  background: linear-gradient(135deg, #0a0a0a 0%, #1a1a2e 30%, #16213e 70%, #0a0a0a 100%);
  border: 2px solid #00ffff;
  border-radius: 8px;
  backdrop-filter: blur(10px);
  box-shadow: 0 8px 32px rgba(0,255,255,0.3), 0 0 20px rgba(0,255,255,0.2), inset 0 1px 0 rgba(255,255,255,0.1), inset 0 0 30px rgba(0,255,255,0.05);
  padding: 1.5rem;
}
.stream-table { width: 100%; border-collapse: collapse; }
.stream-table th, .stream-table td { text-align: left; padding: 0.6rem 0.75rem; border-bottom: 1px solid rgba(0,255,255,0.15); }
.stream-table th { font-weight: 600; color: var(--text-muted, #6c757d); }
.btn {
  padding: 0.75rem 1.5rem;
  border-radius: 6px;
  font-weight: 700;
  cursor: pointer;
  background: linear-gradient(135deg, #1a1a2e, #16213e, #0f0f23);
  border: 2px solid #00ffff;
  color: #00ffff;
  font-family: inherit;
  text-transform: uppercase;
  letter-spacing: 2px;
  backdrop-filter: blur(10px);
  box-shadow: 0 4px 15px rgba(0,255,255,0.2), inset 0 1px 0 rgba(255,255,255,0.1);
  transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}
.btn:hover:not(:disabled) {
  background: linear-gradient(135deg, #00ffff, #0a0a0a);
  color: #000;
  box-shadow: 0 0 20px rgba(0,255,255,0.8);
}
.btn:focus-visible { outline: 2px solid #00ff92; outline-offset: 2px; }
.btn:disabled { opacity: 0.5; cursor: not-allowed; }
.btn-secondary { border-color: var(--text-muted); color: var(--text-muted); box-shadow: none; }
.btn-secondary:hover:not(:disabled) {
  background: linear-gradient(135deg, var(--text-muted), #0a0a0a);
  color: #000;
  box-shadow: 0 0 15px rgba(157,176,198,0.5);
}
</style>
