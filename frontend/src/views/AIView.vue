<template>
  <div class="ai-view">
    <div class="page-header">
      <h1>AI &amp; Model Control</h1>
      <p class="text-muted">Autonomous inference control, model deployment, and live metrics</p>
    </div>

    <div class="ai-grid">
      <!-- Autonomous AI control -->
      <div class="card">
        <div class="card-header">
          <h3>Autonomous AI Control</h3>
          <span class="badge" :class="status?.enabled ? 'badge-on' : 'badge-off'">
            {{ status?.enabled ? 'ENABLED' : 'DISABLED' }}
          </span>
        </div>
        <div class="card-body">
          <div class="status-rows">
            <div class="row"><span>Mode</span><strong>{{ status?.mode ?? '—' }}</strong></div>
            <div class="row"><span>Model loaded</span><strong>{{ yn(status?.model_loaded) }}</strong></div>
            <div class="row">
              <span>Hailo accelerator</span>
              <strong>{{ yn(status?.hailo_available) }}<template v-if="status?.using_hardware"> (active)</template></strong>
            </div>
            <div v-if="status?.hailo_temperature != null" class="row">
              <span>Hailo temp</span><strong>{{ status?.hailo_temperature?.toFixed(1) }} °C</strong>
            </div>
            <div class="row" :class="{ warn: status?.safety_engaged }">
              <span>Safety</span>
              <strong>{{ status?.safety_engaged ? (status?.safety_reason || 'engaged') : 'clear' }}</strong>
            </div>
          </div>
          <div class="actions">
            <button class="btn btn-success" :disabled="busy || status?.enabled" @click="enableAI">Enable AI</button>
            <button class="btn btn-secondary" :disabled="busy || !status?.enabled" @click="disableAI">Disable AI</button>
          </div>
          <p v-if="!status?.model_loaded" class="hint">
            No model is loaded — deploy a <code>.hef</code> below before autonomous mowing.
          </p>
          <p v-if="health && !health.healthy" class="health-warn">⚠️ {{ health.message }}</p>
        </div>
      </div>

      <!-- Deployed model -->
      <div class="card">
        <div class="card-header"><h3>Deployed Model</h3></div>
        <div class="card-body">
          <div class="status-rows">
            <div class="row"><span>Name</span><strong>{{ metrics?.model_name ?? '—' }}</strong></div>
            <div class="row"><span>Version</span><strong>{{ metrics?.model_version ?? '—' }}</strong></div>
            <div class="row">
              <span>Acceleration</span>
              <strong>{{ metrics?.hardware_accelerated ? 'Hailo INT8' : 'CPU / simulation' }}</strong>
            </div>
          </div>
          <div class="load-model">
            <label for="model-path">Load model (.hef path on the Pi)</label>
            <input
              id="model-path"
              v-model="modelPath"
              type="text"
              class="form-control"
              placeholder="/apps/lawnberry-pi/models/lawn_vla.hef"
            >
            <button class="btn btn-primary" :disabled="busy || !modelPath" @click="loadModel">Deploy</button>
          </div>
        </div>
      </div>

      <!-- Inference metrics -->
      <div class="card metrics-card">
        <div class="card-header">
          <h3>Inference Metrics</h3>
          <button class="btn btn-xs btn-secondary" :disabled="busy" @click="resetMetrics">Reset</button>
        </div>
        <div class="card-body">
          <div class="metric-grid">
            <MetricWidget label="Inferences" :value="metrics?.total_inferences ?? 0" icon="🔁" />
            <MetricWidget
              label="Success rate"
              :value="pct(metrics?.success_rate)"
              unit="%"
              icon="✅"
              :variant="rateVariant"
            />
            <MetricWidget
              label="Avg latency"
              :value="fmt(metrics?.avg_inference_time_ms)"
              unit="ms"
              icon="⏱️"
              :variant="latVariant"
            />
            <MetricWidget
              label="FPS"
              :value="fmt(metrics?.inferences_per_second)"
              icon="🎞️"
              :variant="metrics?.meets_target_fps ? 'success' : 'warning'"
            />
            <MetricWidget
              label="Avg confidence"
              :value="pct(metrics?.avg_confidence)"
              unit="%"
              icon="🎯"
            />
            <MetricWidget
              label="Safety overrides"
              :value="metrics?.safety_overrides ?? 0"
              icon="🛡️"
              :variant="(metrics?.safety_overrides ?? 0) > 0 ? 'warning' : 'info'"
            />
          </div>
        </div>
      </div>

      <!-- Datasets -->
      <div class="card">
        <div class="card-header">
          <h3>Datasets</h3>
          <button class="btn btn-xs btn-secondary" :disabled="busy" @click="loadDatasets">Refresh</button>
        </div>
        <div class="card-body">
          <table v-if="datasets.length" class="ds-table">
            <thead><tr><th>Dataset</th><th>Labels</th><th /></tr></thead>
            <tbody>
              <tr v-for="d in datasets" :key="d.id">
                <td>{{ d.name }}</td>
                <td>{{ d.label_count }}</td>
                <td><button class="btn btn-xs btn-info" :disabled="busy" @click="exportDataset(d)">Export</button></td>
              </tr>
            </tbody>
          </table>
          <p v-else class="text-muted">No datasets available.</p>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed, onMounted, onUnmounted, ref } from 'vue'
import { useApiService } from '@/services/api'
import { useToastStore } from '@/stores/toast'
import MetricWidget from '@/components/MetricWidget.vue'

interface AIStatus {
  enabled: boolean
  mode: string
  model_loaded: boolean
  hailo_available: boolean
  hailo_temperature?: number | null
  using_hardware: boolean
  safety_engaged: boolean
  safety_reason?: string | null
  success_rate: number
  avg_latency_ms: number
  current_fps: number
}
interface AIMetrics {
  total_inferences: number
  safety_overrides: number
  avg_inference_time_ms: number
  inferences_per_second: number
  meets_target_fps: boolean
  success_rate: number
  avg_confidence: number
  model_name: string
  model_version: string
  hardware_accelerated: boolean
}
interface AIHealth {
  healthy: boolean
  message: string
}
interface Dataset {
  id: string
  name: string
  label_count: number
}

const api = useApiService()
const status = ref<AIStatus | null>(null)
const metrics = ref<AIMetrics | null>(null)
const health = ref<AIHealth | null>(null)
const datasets = ref<Dataset[]>([])
const modelPath = ref('')
const busy = ref(false)
const toast = useToastStore()
let timer: ReturnType<typeof setInterval> | null = null

function notify(msg: string, ok = true) {
  toast.show(msg, ok ? 'success' : 'error')
}
function yn(v: boolean | undefined) {
  return v == null ? '—' : v ? 'Yes' : 'No'
}
function fmt(v: number | undefined | null) {
  return v == null ? '—' : v.toFixed(1)
}
function pct(v: number | undefined | null) {
  return v == null ? '—' : Math.round(v * 100)
}
function detail(e: unknown): string {
  return (e as { response?: { data?: { detail?: string } } })?.response?.data?.detail || ''
}

const rateVariant = computed(() => {
  const r = metrics.value?.success_rate
  if (r == null) return 'info'
  return r >= 0.9 ? 'success' : r >= 0.7 ? 'warning' : 'danger'
})
const latVariant = computed(() => {
  const l = metrics.value?.avg_inference_time_ms
  if (l == null) return 'info'
  return l <= 100 ? 'success' : l <= 200 ? 'warning' : 'danger'
})

async function loadStatus() {
  try {
    status.value = (await api.get('/api/v2/ai/status')).data
  } catch {
    /* keep last known status */
  }
}
async function loadMetrics() {
  try {
    metrics.value = (await api.get('/api/v2/ai/metrics')).data
  } catch {
    /* keep last known metrics */
  }
}
async function loadHealth() {
  try {
    health.value = (await api.get('/api/v2/ai/health')).data
  } catch {
    /* ignore */
  }
}
async function loadDatasets() {
  try {
    datasets.value = (await api.get('/api/v2/ai/datasets')).data || []
  } catch {
    datasets.value = []
  }
}
async function refreshAll() {
  await Promise.all([loadStatus(), loadMetrics(), loadHealth()])
}

async function enableAI() {
  busy.value = true
  try {
    await api.post('/api/v2/ai/enable', { mode: 'ai_autonomous' })
    notify('AI autonomous control enabled')
    await refreshAll()
  } catch (e) {
    notify(detail(e) || 'Failed to enable AI', false)
  } finally {
    busy.value = false
  }
}
async function disableAI() {
  busy.value = true
  try {
    await api.post('/api/v2/ai/disable', {})
    notify('AI control disabled')
    await refreshAll()
  } catch (e) {
    notify(detail(e) || 'Failed to disable AI', false)
  } finally {
    busy.value = false
  }
}
async function loadModel() {
  busy.value = true
  try {
    const r = (await api.post('/api/v2/ai/model', { model_path: modelPath.value })).data
    notify(
      r?.success ? `Deployed ${r.model_name} v${r.model_version}` : r?.message || 'Model load failed',
      !!r?.success
    )
    await refreshAll()
  } catch (e) {
    notify(detail(e) || 'Failed to load model', false)
  } finally {
    busy.value = false
  }
}
async function resetMetrics() {
  busy.value = true
  try {
    await api.post('/api/v2/ai/metrics/reset', {})
    notify('Metrics reset')
    await loadMetrics()
  } catch {
    notify('Failed to reset metrics', false)
  } finally {
    busy.value = false
  }
}
async function exportDataset(d: Dataset) {
  busy.value = true
  try {
    const r = (await api.post(`/api/v2/ai/datasets/${d.id}/export`, { format: 'YOLO' })).data
    notify(`Export started for ${d.name} (${r?.format || 'YOLO'})`)
  } catch (e) {
    notify(detail(e) || `Failed to export ${d.name}`, false)
  } finally {
    busy.value = false
  }
}

onMounted(async () => {
  await Promise.all([refreshAll(), loadDatasets()])
  timer = setInterval(refreshAll, 3000)
})
onUnmounted(() => {
  if (timer) clearInterval(timer)
})
</script>

<style scoped>
.ai-view { padding: 0; }
.page-header { margin-bottom: 1.5rem; }
.page-header h1 { margin-bottom: 0.25rem; }
.ai-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
  gap: 1.25rem;
}
.metrics-card { grid-column: 1 / -1; }
.card { background: var(--card-bg, #fff); border-radius: 12px; box-shadow: 0 1px 3px rgba(0,0,0,0.08); }
.card-header { display: flex; align-items: center; justify-content: space-between; padding: 1rem 1.25rem; border-bottom: 1px solid var(--border-color, #eee); }
.card-header h3 { margin: 0; font-size: 1.05rem; }
.card-body { padding: 1.25rem; }
.status-rows { display: flex; flex-direction: column; gap: 0.5rem; margin-bottom: 1rem; }
.status-rows .row { display: flex; justify-content: space-between; gap: 1rem; }
.status-rows .row span { color: var(--text-muted, #6c757d); }
.status-rows .row.warn strong { color: #b42318; }
.actions { display: flex; gap: 0.75rem; }
.hint { margin-top: 0.75rem; color: var(--text-muted, #6c757d); font-size: 0.9rem; }
.hint code { background: #f1f1f1; padding: 0.05rem 0.3rem; border-radius: 4px; }
.health-warn { margin-top: 0.75rem; color: #b42318; }
.load-model { display: flex; flex-direction: column; gap: 0.5rem; }
.load-model label { font-size: 0.9rem; color: var(--text-muted, #6c757d); }
.load-model .form-control { padding: 0.5rem 0.75rem; border: 1px solid var(--border-color, #ccc); border-radius: 8px; }
.metric-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(170px, 1fr)); gap: 1rem; }
.badge { padding: 0.25rem 0.6rem; border-radius: 999px; font-size: 0.75rem; font-weight: 700; }
.badge-on { background: #d1f7e0; color: #0c7a43; }
.badge-off { background: #eceff1; color: #607d8b; }
.ds-table { width: 100%; border-collapse: collapse; }
.ds-table th, .ds-table td { text-align: left; padding: 0.6rem 0.5rem; border-bottom: 1px solid var(--border-color, #eee); }
.ds-table th { color: var(--text-muted, #6c757d); font-weight: 600; }
</style>
