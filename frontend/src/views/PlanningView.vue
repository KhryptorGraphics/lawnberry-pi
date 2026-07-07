<template>
  <div class="planning-view">
    <div class="page-header">
      <h1>Mow Planning</h1>
      <p class="text-muted">Schedule mowing jobs, manage zones, and optimize operations</p>
    </div>

    <!-- Quick Start Actions -->
    <div class="quick-actions">
      <button class="btn btn-primary quick-btn" @click="startQuickMow">
        ⚡ Quick Mow
      </button>
      <button class="btn btn-success quick-btn" @click="openScheduleModal">
        📅 Schedule Job
      </button>
      <button class="btn btn-info quick-btn" @click="activeTab = 'zones'">
        🗺️ Manage Zones
      </button>
    </div>

    <!-- Planning Tabs -->
    <div class="planning-tabs">
      <button
        v-for="tab in tabs"
        :key="tab.id"
        :class="{ active: activeTab === tab.id }"
        class="tab-button"
        @click="activeTab = tab.id"
      >
        {{ tab.label }}
      </button>
    </div>

    <!-- Current Jobs -->
    <div v-if="activeTab === 'current'" class="tab-content">
      <JobsCard
        :jobs="jobs"
        :completed-jobs="completedJobs"
        :area-unit="areaUnit"
        :format-job-status="formatJobStatus"
        :format-date-time="formatDateTime"
        :format-area="formatArea"
        @refresh="refreshJobs"
        @schedule="openScheduleModal"
        @start="startJob"
        @pause="pauseJob"
        @resume="resumeJob"
        @cancel="cancelJob"
      />
    </div>

    <!-- Scheduling -->
    <div v-if="activeTab === 'schedule'" class="tab-content">
      <SchedulesCard
        :schedules="schedules"
        :current-weather="currentWeather"
        :weather-class="weatherClass"
        :weather-temperature-display="weatherTemperatureDisplay"
        :temperature-unit="temperatureUnit"
        :recommendation="recommendation"
        :recommendation-class="recommendationClass"
        :ground-class="groundClass"
        :ground-condition="groundCondition"
        :last-rain="lastRain"
        :format-frequency="formatFrequency"
        :format-date-time="formatDateTime"
        @add="openScheduleModal"
        @toggle="toggleSchedule"
        @edit="editSchedule"
        @delete="deleteSchedule"
      />
    </div>

    <!-- Zone Management -->
    <div v-if="activeTab === 'zones'" class="tab-content">
      <ZonesCard
        :zones="zones"
        :selected-zone-id="selectedZone?.id ?? null"
        :area-unit="areaUnit"
        :cutting-height-unit="cuttingHeightUnit"
        :format-priority="formatPriority"
        :format-area="formatArea"
        :format-cutting-height="formatCuttingHeight"
        :format-relative-time="formatRelativeTime"
        @add="openZoneModal"
        @select="selectZone"
        @mow="mowZone"
        @edit="editZone"
        @go-maps="goToMaps"
      />
    </div>

    <!-- Patterns -->
    <div v-if="activeTab === 'patterns'" class="tab-content">
      <PatternsCard
        v-model:selected-pattern="selectedPattern"
        :patterns="patterns"
      />
    </div>

    <!-- Schedule Job Modal -->
    <div v-if="showScheduleModal" class="modal-overlay" @click="closeScheduleModal">
      <div class="modal-content" @click.stop>
        <div class="modal-header">
          <h3>{{ editingSchedule ? 'Edit Schedule' : 'Schedule Mowing Job' }}</h3>
          <button class="btn btn-sm btn-secondary" @click="closeScheduleModal">✖️</button>
        </div>
        <div class="modal-body">
          <div class="form-group">
            <label>Job Name</label>
            <input v-model="scheduleForm.name" type="text" class="form-control">
          </div>

          <div class="form-group">
            <label>Zones to Mow</label>
            <div class="zone-checkboxes">
              <label v-for="zone in zones" :key="zone.id" class="checkbox-label">
                <input
                  v-model="scheduleForm.zones"
                  type="checkbox"
                  :value="zone.id"
                >
                {{ zone.name }}
              </label>
            </div>
          </div>

          <div class="form-group">
            <label>Mowing Pattern</label>
            <select v-model="scheduleForm.pattern" class="form-control">
              <option v-for="pattern in patterns" :key="pattern.id" :value="pattern.id">
                {{ pattern.name }}
              </option>
            </select>
          </div>

          <div class="form-group">
            <label>Schedule Type</label>
            <select v-model="scheduleForm.type" class="form-control">
              <option value="once">One-time job</option>
              <option value="recurring">Recurring schedule</option>
            </select>
          </div>

          <div v-if="scheduleForm.type === 'once'" class="form-group">
            <label>Start Time</label>
            <input
              v-model="scheduleForm.startTime"
              type="datetime-local"
              class="form-control"
            >
          </div>

          <div v-if="scheduleForm.type === 'recurring'" class="recurring-options">
            <div class="form-group">
              <label>Frequency</label>
              <select v-model="scheduleForm.frequency" class="form-control">
                <option value="daily">Daily</option>
                <option value="weekly">Weekly</option>
                <option value="biweekly">Every 2 weeks</option>
                <option value="monthly">Monthly</option>
              </select>
            </div>

            <div class="form-group">
              <label>Time of Day</label>
              <input
                v-model="scheduleForm.timeOfDay"
                type="time"
                class="form-control"
              >
            </div>
          </div>
        </div>
        <div class="modal-footer">
          <button class="btn btn-secondary" @click="closeScheduleModal">Cancel</button>
          <button class="btn btn-primary" @click="saveSchedule">
            {{ editingSchedule ? 'Update' : 'Schedule' }}
          </button>
        </div>
      </div>
    </div>

    <!-- Status Messages -->
    <div v-if="statusMessage" class="alert" :class="statusSuccess ? 'alert-success' : 'alert-danger'">
      {{ statusMessage }}
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted } from 'vue'
import { storeToRefs } from 'pinia'
import { useApiService, startAutonomous } from '@/services/api'
import { useWebSocket } from '@/services/websocket'
import { usePreferencesStore } from '@/stores/preferences'
import { useMapStore } from '@/stores/map'
import type { Zone } from '@/stores/map'
import { useConfirmStore } from '@/stores/confirm'
import { useRouter } from 'vue-router'
import JobsCard from '@/components/planning/JobsCard.vue'
import SchedulesCard from '@/components/planning/SchedulesCard.vue'
import ZonesCard from '@/components/planning/ZonesCard.vue'
import PatternsCard from '@/components/planning/PatternsCard.vue'

// Backend contract: GET/POST/DELETE /api/v2/planning/jobs
// (backend/src/api/routers/planning.py). On create, the backend only ever
// writes id/name/zones/schedule/priority/enabled/created_at — `pattern` and
// `scheduled_start`, sent in the create payload by mowZone()/saveSchedule()
// below, are silently dropped (never read back). `status`/`mission_id` are
// added only once a lifecycle action (start/pause/resume/cancel) has run —
// a freshly created job has NO `status` at all, so the template's
// `job.status === 'scheduled'` Start-button check never matches it against
// real data (flagged, not fixed — out of scope for this typing pass).
// `progress`/`estimated_remaining` are never sent by the REST endpoints;
// they arrive only via the `jobs.progress` WS topic subscribed in onMounted.
interface MowJob {
  id: string
  name: string
  zones: string[]
  status?: 'scheduled' | 'running' | 'paused' | 'completed' | 'cancelled' | 'failed'
  pattern?: string
  scheduled_start?: string
  progress?: number
  estimated_remaining?: number | null
}

// Job History card. Nothing ever refreshes this from the backend — it stays
// fixed at its seed value. UI-only mock data, not a backend contract.
interface CompletedJob {
  id: number
  name: string
  completed_at: string
  actual_duration: number
  area_covered: number
}

// NOTE: no `/api/v2/schedules` route exists on the backend (confirmed via
// `grep -rn schedules backend/src/` — only an internal
// `_update_recurring_schedules()` job-service method, not an HTTP route).
// refreshSchedules/saveSchedule/toggleSchedule/deleteSchedule below all call
// this nonexistent endpoint and will always fail into their catch blocks;
// this tab runs on mock seed data only. Typed to that seed/mock shape, not a
// real backend contract.
interface MowSchedule {
  id: number
  name: string
  frequency: string
  zones: string[]
  pattern: string
  enabled: boolean
  next_run: string
}

interface ZoneCard {
  id: string
  name: string
  area_m2: number
  cutting_height: number | null
  priority: 'high' | 'medium' | 'low'
  last_mowed: string | null
}

// Local-only shape for the schedule-job modal (not a backend contract —
// see saveSchedule/MowSchedule above).
interface ScheduleForm {
  name: string
  zones: string[]
  pattern: string
  type: string
  startTime: string
  frequency: string
  timeOfDay: string
}

const api = useApiService()
const router = useRouter()
const mapStore = useMapStore()
const confirmStore = useConfirmStore()
const { connect, subscribe } = useWebSocket()
const preferences = usePreferencesStore()

preferences.ensureInitialized()
const { unitSystem } = storeToRefs(preferences)

// State
const activeTab = ref('current')
const showScheduleModal = ref(false)
const editingSchedule = ref<MowSchedule | null>(null)
const selectedZone = ref<ZoneCard | null>(null)
const selectedPattern = ref('parallel')
const statusMessage = ref('')
const statusSuccess = ref(false)

// Tabs
const tabs = [
  { id: 'current', label: 'Current Jobs' },
  { id: 'schedule', label: 'Scheduling' },
  { id: 'zones', label: 'Zones' },
  { id: 'patterns', label: 'Patterns' }
]

// Data
const jobs = ref<MowJob[]>([
  {
    id: '1',
    name: 'Front Lawn Weekly',
    status: 'running',
    zones: ['Front Lawn', 'Side Lawn'],
    pattern: 'parallel',
    progress: 45,
    estimated_remaining: 25,
    scheduled_start: '2024-09-28T10:00:00'
  },
  {
    id: '2',
    name: 'Back Lawn Maintenance',
    status: 'scheduled',
    zones: ['Back Lawn'],
    pattern: 'spiral',
    progress: 0,
    estimated_remaining: null,
    scheduled_start: '2024-09-28T14:00:00'
  }
])

const completedJobs = ref<CompletedJob[]>([
  {
    id: 3,
    name: 'Full Property Mow',
    completed_at: '2024-09-27T16:30:00',
    actual_duration: 180,
    area_covered: 450
  },
  {
    id: 4,
    name: 'Front Lawn Touch-up',
    completed_at: '2024-09-26T11:15:00',
    actual_duration: 45,
    area_covered: 150
  }
])

const schedules = ref<MowSchedule[]>([
  {
    id: 1,
    name: 'Weekly Full Property',
    frequency: 'weekly',
    zones: ['Front Lawn', 'Back Lawn', 'Side Lawn'],
    pattern: 'parallel',
    enabled: true,
    next_run: '2024-09-30T10:00:00'
  },
  {
    id: 2,
    name: 'Bi-weekly Edge Trim',
    frequency: 'biweekly',
    zones: ['Perimeter'],
    pattern: 'edge',
    enabled: false,
    next_run: '2024-10-05T09:00:00'
  }
])

// Real mowing zones come from the saved map configuration (the polygon editor
// lives in the Maps view). Boundary is shown as the implicit full-yard zone
// when no explicit mowing zones are defined.
function polygonAreaM2(poly: Array<{ latitude: number; longitude: number }> | undefined): number {
  if (!poly || poly.length < 3) return 0
  const R = 6378137
  const lat0 = (poly[0].latitude * Math.PI) / 180
  const pts = poly.map((p) => ({
    x: ((p.longitude * Math.PI) / 180) * R * Math.cos(lat0),
    y: ((p.latitude * Math.PI) / 180) * R,
  }))
  let area = 0
  for (let i = 0; i < pts.length; i++) {
    const j = (i + 1) % pts.length
    area += pts[i].x * pts[j].y - pts[j].x * pts[i].y
  }
  return Math.abs(area) / 2
}

function priorityLabel(p: unknown): 'high' | 'medium' | 'low' {
  if (typeof p === 'number') return p >= 8 ? 'high' : p >= 4 ? 'medium' : 'low'
  const s = String(p ?? '').toLowerCase()
  return s === 'high' || s === 'low' ? s : 'medium'
}

function mapZoneToCard(z: Zone, defaultPriority: 'high' | 'medium' | 'low'): ZoneCard {
  return {
    id: z.id,
    name: z.name || z.id,
    area_m2: polygonAreaM2(z.polygon),
    cutting_height: null,
    priority: z.priority != null ? priorityLabel(z.priority) : defaultPriority,
    last_mowed: null,
  }
}

const zones = computed<ZoneCard[]>(() => {
  const cfg = mapStore.configuration
  if (!cfg) return []
  const mow = (cfg.mowing_zones || []).filter((z) => z.polygon?.length)
  if (mow.length) return mow.map((z) => mapZoneToCard(z, 'medium'))
  if (cfg.boundary_zone?.polygon?.length) return [mapZoneToCard(cfg.boundary_zone, 'high')]
  return []
})

const patterns = ref([
  {
    id: 'parallel',
    name: 'Parallel Lines',
    description: 'Straight parallel lines across the area',
    efficiency: 95,
    coverage: 98
  },
  {
    id: 'spiral',
    name: 'Spiral',
    description: 'Spiral pattern from outside to center',
    efficiency: 85,
    coverage: 95
  },
  {
    id: 'random',
    name: 'Random',
    description: 'Random movement pattern',
    efficiency: 70,
    coverage: 92
  },
  {
    id: 'edge',
    name: 'Edge First',
    description: 'Cut edges first, then fill interior',
    efficiency: 90,
    coverage: 99
  }
])

const currentWeather = ref({
  temperature_c: 22,
  humidity_percent: 65,
  condition: 'Partly Cloudy'
})

const temperatureUnit = computed(() => (unitSystem.value === 'imperial' ? '°F' : '°C'))
const weatherTemperatureDisplay = computed(() => {
  const raw = Number(currentWeather.value.temperature_c)
  if (!Number.isFinite(raw)) {
    return '--'
  }
  const converted = unitSystem.value === 'imperial' ? (raw * 9) / 5 + 32 : raw
  const formatter = new Intl.NumberFormat(undefined, {
    minimumFractionDigits: 1,
    maximumFractionDigits: 1,
  })
  return formatter.format(converted)
})

const areaUnit = computed(() => (unitSystem.value === 'imperial' ? 'ft²' : 'm²'))
const cuttingHeightUnit = computed(() => (unitSystem.value === 'imperial' ? 'in' : 'mm'))

const formatArea = (value: unknown): string => {
  const numeric = typeof value === 'number' ? value : Number(value)
  if (!Number.isFinite(numeric)) {
    return 'N/A'
  }
  const converted = unitSystem.value === 'imperial' ? numeric * 10.7639104167 : numeric
  const formatter = new Intl.NumberFormat(undefined, {
    minimumFractionDigits: converted >= 100 ? 0 : 1,
    maximumFractionDigits: converted >= 100 ? 0 : 1,
  })
  return formatter.format(converted)
}

const formatCuttingHeight = (value: unknown): string => {
  const numeric = typeof value === 'number' ? value : Number(value)
  if (!Number.isFinite(numeric)) {
    return 'N/A'
  }
  if (unitSystem.value === 'imperial') {
    const inches = numeric * 0.0393700787
    const formatter = new Intl.NumberFormat(undefined, {
      minimumFractionDigits: 1,
      maximumFractionDigits: 1,
    })
    return formatter.format(inches)
  }
  const formatter = new Intl.NumberFormat(undefined, {
    minimumFractionDigits: 0,
    maximumFractionDigits: 0,
  })
  return formatter.format(numeric)
}

const recommendation = ref({
  advice: 'Proceed',
  reason: 'Good weather conditions for mowing'
})

const scheduleForm = ref<ScheduleForm>({
  name: '',
  zones: [],
  pattern: 'parallel',
  type: 'once',
  startTime: '',
  frequency: 'weekly',
  timeOfDay: '10:00'
})

// Computed
const weatherClass = computed(() => {
  const condition = currentWeather.value.condition?.toLowerCase()
  if (condition?.includes('rain') || condition?.includes('storm')) return 'condition-bad'
  if (condition?.includes('cloud') || condition?.includes('overcast')) return 'condition-warn'
  return 'condition-good'
})

const recommendationClass = computed(() => {
  const advice = recommendation.value.advice?.toLowerCase()
  if (advice === 'proceed') return 'condition-good'
  if (advice === 'caution') return 'condition-warn'
  return 'condition-bad'
})

const groundClass = computed(() => 'condition-good')
const groundCondition = computed(() => 'Dry')
const lastRain = computed(() => '2 days ago')

// Methods
async function startQuickMow() {
  try {
    // Begin an autonomous mowing run over all boundary zones immediately.
    await startAutonomous()
    showStatus('Autonomous mowing started!', true)
    await refreshJobs()
  } catch (error) {
    showStatus('Failed to start autonomous mowing', false)
  }
}

function openScheduleModal() {
  editingSchedule.value = null
  scheduleForm.value = {
    name: '',
    zones: [],
    pattern: 'parallel',
    type: 'once',
    startTime: '',
    frequency: 'weekly',
    timeOfDay: '10:00'
  }
  showScheduleModal.value = true
}

function closeScheduleModal() {
  showScheduleModal.value = false
  editingSchedule.value = null
}

async function saveSchedule() {
  try {
    const endpoint = editingSchedule.value
      ? `/api/v2/schedules/${editingSchedule.value.id}`
      : '/api/v2/schedules'

    const method = editingSchedule.value ? 'put' : 'post'

    await api[method](endpoint, scheduleForm.value)

    showStatus(
      editingSchedule.value ? 'Schedule updated!' : 'Schedule created!',
      true
    )

    closeScheduleModal()
    await refreshSchedules()
  } catch (error) {
    showStatus('Failed to save schedule', false)
  }
}

async function refreshJobs() {
  try {
    // Backend contract: GET /api/v2/planning/jobs -> plain list of jobs
    const response = await api.get<MowJob[] | { active?: MowJob[] }>('/api/v2/planning/jobs')
    jobs.value = Array.isArray(response.data) ? response.data : (response.data?.active || [])
  } catch (error) {
    console.error('Failed to refresh jobs:', error)
  }
}

async function refreshSchedules() {
  try {
    const response = await api.get<MowSchedule[]>('/api/v2/schedules')
    schedules.value = response.data || []
  } catch (error) {
    console.error('Failed to refresh schedules:', error)
  }
}

async function startJob(job: MowJob) {
  try {
    await api.post(`/api/v2/planning/jobs/${job.id}/start`)
    job.status = 'running'
    showStatus('Job started!', true)
  } catch (error) {
    showStatus('Failed to start job', false)
  }
}

async function pauseJob(job: MowJob) {
  try {
    await api.post(`/api/v2/planning/jobs/${job.id}/pause`)
    job.status = 'paused'
    showStatus('Job paused', true)
  } catch (error) {
    showStatus('Failed to pause job', false)
  }
}

async function resumeJob(job: MowJob) {
  try {
    await api.post(`/api/v2/planning/jobs/${job.id}/resume`)
    job.status = 'running'
    showStatus('Job resumed!', true)
  } catch (error) {
    showStatus('Failed to resume job', false)
  }
}

async function cancelJob(job: MowJob) {
  if (!(await confirmStore.ask(`Cancel job "${job.name}"?`))) return

  try {
    await api.delete(`/api/v2/planning/jobs/${job.id}`)
    const index = jobs.value.findIndex(j => j.id === job.id)
    if (index > -1) jobs.value.splice(index, 1)
    showStatus('Job cancelled', true)
  } catch (error) {
    showStatus('Failed to cancel job', false)
  }
}

async function toggleSchedule(schedule: MowSchedule) {
  try {
    await api.put(`/api/v2/schedules/${schedule.id}`, {
      ...schedule,
      enabled: !schedule.enabled
    })
    schedule.enabled = !schedule.enabled
    showStatus(
      schedule.enabled ? 'Schedule enabled' : 'Schedule disabled',
      true
    )
  } catch (error) {
    showStatus('Failed to toggle schedule', false)
  }
}

function editSchedule(schedule: MowSchedule) {
  editingSchedule.value = schedule
  // MowSchedule has no type/startTime/timeOfDay (schedule-form-only fields) —
  // merge over the existing form rather than dropping them to undefined.
  scheduleForm.value = { ...scheduleForm.value, ...schedule }
  showScheduleModal.value = true
}

async function deleteSchedule(schedule: MowSchedule) {
  if (!(await confirmStore.ask(`Delete schedule "${schedule.name}"?`))) return

  try {
    await api.delete(`/api/v2/schedules/${schedule.id}`)
    const index = schedules.value.findIndex(s => s.id === schedule.id)
    if (index > -1) schedules.value.splice(index, 1)
    showStatus('Schedule deleted', true)
  } catch (error) {
    showStatus('Failed to delete schedule', false)
  }
}

function selectZone(zone: ZoneCard) {
  selectedZone.value = selectedZone.value?.id === zone.id ? null : zone
}

async function mowZone(zone: ZoneCard) {
  try {
    await api.post('/api/v2/planning/jobs', {
      name: `${zone.name} - Quick Mow`,
      zones: [zone.id],
      pattern: 'parallel',
      start_immediately: true
    })
    showStatus(`Started mowing ${zone.name}`, true)
    await refreshJobs()
  } catch (error) {
    showStatus(`Failed to start mowing ${zone.name}`, false)
  }
}

function goToMaps() {
  router.push('/maps')
}

function openZoneModal() {
  // Zone geometry is drawn/edited in the Maps view (the polygon editor).
  showStatus('Opening the map editor to add a zone…', true)
  goToMaps()
}

function editZone(_zone: ZoneCard) {
  showStatus('Opening the map editor to edit zones…', true)
  goToMaps()
}

function formatJobStatus(status: string | undefined): string {
  const statusMap = {
    scheduled: 'Scheduled',
    running: 'Running',
    paused: 'Paused',
    completed: 'Completed',
    cancelled: 'Cancelled',
    failed: 'Failed'
  }
  // A job with no status yet (real backend, before any lifecycle action has
  // run — see MowJob's status comment) is effectively still "scheduled".
  if (!status) return statusMap.scheduled
  return statusMap[status as keyof typeof statusMap] || status
}

function formatFrequency(frequency: string): string {
  const freqMap = {
    daily: 'Daily',
    weekly: 'Weekly',
    biweekly: 'Every 2 weeks',
    monthly: 'Monthly'
  }
  return freqMap[frequency as keyof typeof freqMap] || frequency
}

function formatPriority(priority: string): string {
  return priority.charAt(0).toUpperCase() + priority.slice(1)
}

function formatDateTime(dateString: string): string {
  try {
    return new Date(dateString).toLocaleString()
  } catch {
    return dateString
  }
}

function formatRelativeTime(dateString: string | null | undefined): string {
  if (!dateString) return 'Never'
  try {
    const date = new Date(dateString)
    const now = new Date()
    const diffMs = now.getTime() - date.getTime()
    const diffDays = Math.floor(diffMs / (1000 * 60 * 60 * 24))

    if (diffDays === 0) return 'Today'
    if (diffDays === 1) return 'Yesterday'
    if (diffDays < 7) return `${diffDays} days ago`
    if (diffDays < 30) return `${Math.floor(diffDays / 7)} weeks ago`
    return `${Math.floor(diffDays / 30)} months ago`
  } catch {
    return 'Unknown'
  }
}

function showStatus(message: string, success: boolean) {
  statusMessage.value = message
  statusSuccess.value = success
  setTimeout(() => {
    statusMessage.value = ''
  }, 3000)
}

onMounted(async () => {
  await connect()

  // Load real mowing zones from the saved map configuration (best-effort).
  try {
    await mapStore.loadConfiguration()
  } catch {
    /* zones tab will show an empty state */
  }

  // Subscribe to job progress updates
  subscribe('jobs.progress', (data) => {
    const job = jobs.value.find(j => j.name === data.current_job)
    if (job) {
      job.progress = data.progress_percent
      job.estimated_remaining = data.remaining_time_min
    }
  })

  // Subscribe to weather updates
  subscribe('telemetry.weather', (data) => {
    currentWeather.value = {
      ...currentWeather.value,
      temperature_c: data.temperature_c ?? currentWeather.value.temperature_c,
      humidity_percent: data.humidity_percent ?? currentWeather.value.humidity_percent
    }
  })

  await refreshJobs()
  await refreshSchedules()
})
</script>

<style scoped>
.planning-view {
  padding: 0;
}

.page-header {
  margin-bottom: 2rem;
}

.page-header h1 {
  margin-bottom: 0.5rem;
}

.quick-actions {
  display: flex;
  gap: 1rem;
  margin-bottom: 2rem;
  flex-wrap: wrap;
}

.quick-btn {
  padding: 1rem 1.5rem;
  font-size: 1.1rem;
  border-radius: 8px;
}

.planning-tabs {
  display: flex;
  border-bottom: 2px solid var(--primary-dark);
  margin-bottom: 2rem;
  overflow-x: auto;
}

.tab-button {
  background: none;
  border: none;
  padding: 1rem 2rem;
  color: var(--text-color);
  font-weight: 500;
  cursor: pointer;
  border-bottom: 3px solid transparent;
  transition: all 0.3s ease;
  white-space: nowrap;
}

.tab-button:hover {
  background-color: var(--primary-dark);
  color: var(--primary-light);
}

.tab-button.active {
  border-bottom-color: var(--accent-green);
  color: var(--accent-green);
  background-color: var(--primary-dark);
}

.tab-content {
  animation: slideIn 0.3s ease;
}

@keyframes slideIn {
  from {
    opacity: 0;
    transform: translateX(20px);
  }
  to {
    opacity: 1;
    transform: translateX(0);
  }
}

/* Button primitives — still used by the quick-actions bar and the schedule
 * modal below. The planning cards carry their own copies (Vue scoped CSS
 * doesn't reach into child components); keep those in sync with these. */
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

.btn-info {
  background: #17a2b8;
  color: white;
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

.modal-overlay {
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: rgba(0, 0, 0, 0.7);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 1000;
}

.modal-content {
  background: var(--secondary-dark);
  border: 1px solid var(--primary-light);
  border-radius: 8px;
  width: 90%;
  max-width: 600px;
  max-height: 90vh;
  overflow-y: auto;
}

.modal-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 1rem;
  border-bottom: 1px solid var(--primary-light);
}

.modal-header h3 {
  margin: 0;
  color: var(--accent-green);
}

.modal-body {
  padding: 1.5rem;
}

.modal-footer {
  display: flex;
  justify-content: flex-end;
  gap: 1rem;
  padding: 1rem;
  border-top: 1px solid var(--primary-light);
}

.form-group {
  margin-bottom: 1.5rem;
}

.form-group label {
  display: block;
  margin-bottom: 0.5rem;
  color: var(--text-color);
  font-weight: 500;
}

.form-control {
  width: 100%;
  padding: 0.75rem;
  background: var(--primary-dark);
  border: 1px solid var(--primary-light);
  border-radius: 4px;
  color: var(--text-color);
  font-size: 1rem;
}

.form-control:focus {
  outline: none;
  border-color: var(--accent-green);
  box-shadow: 0 0 0 2px rgba(0, 255, 146, 0.2);
}

.zone-checkboxes {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
  gap: 0.5rem;
  margin-top: 0.5rem;
}

.checkbox-label {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  padding: 0.5rem;
  background: var(--primary-dark);
  border-radius: 4px;
  cursor: pointer;
  transition: background-color 0.3s ease;
}

.checkbox-label:hover {
  background: var(--secondary-dark);
}

.recurring-options {
  margin-left: 1rem;
  padding-left: 1rem;
  border-left: 3px solid var(--accent-green);
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
  .quick-actions {
    flex-direction: column;
  }

  .planning-tabs {
    flex-direction: column;
  }

  .tab-button {
    padding: 0.75rem 1rem;
    text-align: left;
  }

  .zone-checkboxes {
    grid-template-columns: 1fr;
  }
}
</style>
