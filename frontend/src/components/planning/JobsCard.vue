<template>
  <div class="jobs-card">
    <!-- Current & Queued Jobs -->
    <div class="card">
      <div class="card-header">
        <h3>Current &amp; Queued Jobs</h3>
        <button class="btn btn-sm btn-secondary" @click="$emit('refresh')">
          🔄 Refresh
        </button>
      </div>
      <div class="card-body">
        <div v-if="jobs.length === 0" class="empty-state">
          <p>No active or queued jobs</p>
          <button class="btn btn-primary" @click="$emit('schedule')">
            Schedule First Job
          </button>
        </div>

        <div v-else class="jobs-list">
          <div
            v-for="job in jobs"
            :key="job.id"
            class="job-item"
            :class="`status-${job.status}`"
          >
            <div class="job-header">
              <div class="job-title">
                <h4>{{ job.name }}</h4>
                <span class="status-badge" :class="`status-${job.status}`">
                  {{ formatJobStatus(job.status) }}
                </span>
              </div>
              <div class="job-actions">
                <button
                  v-if="job.status === 'scheduled'"
                  class="btn btn-xs btn-success"
                  @click="$emit('start', job)"
                >
                  ▶️ Start
                </button>
                <button
                  v-if="job.status === 'running'"
                  class="btn btn-xs btn-warning"
                  @click="$emit('pause', job)"
                >
                  ⏸️ Pause
                </button>
                <button
                  v-if="job.status === 'paused'"
                  class="btn btn-xs btn-success"
                  @click="$emit('resume', job)"
                >
                  ▶️ Resume
                </button>
                <button
                  class="btn btn-xs btn-danger"
                  :disabled="job.status === 'completed'"
                  @click="$emit('cancel', job)"
                >
                  ❌ Cancel
                </button>
              </div>
            </div>

            <div class="job-details">
              <div class="job-info">
                <span>Zones: {{ job.zones.join(', ') }}</span>
                <span>Pattern: {{ job.pattern }}</span>
                <span v-if="job.scheduled_start">Start: {{ formatDateTime(job.scheduled_start) }}</span>
              </div>

              <div v-if="job.status === 'running'" class="job-progress">
                <div class="progress-bar">
                  <div class="progress-fill" :style="{ width: `${job.progress}%` }" />
                </div>
                <span class="progress-text">{{ job.progress }}% complete</span>
                <span v-if="job.estimated_remaining" class="time-remaining">
                  ~{{ job.estimated_remaining }} min remaining
                </span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- Job History -->
    <div class="card">
      <div class="card-header">
        <h3>Recent Job History</h3>
      </div>
      <div class="card-body">
        <div class="history-list">
          <div
            v-for="job in completedJobs"
            :key="job.id"
            class="history-item"
          >
            <div class="history-header">
              <span class="job-name">{{ job.name }}</span>
              <span class="completion-time">{{ formatDateTime(job.completed_at) }}</span>
            </div>
            <div class="history-details">
              <span>Duration: {{ job.actual_duration }} min</span>
              <span>Area: {{ formatArea(job.area_covered) }} {{ areaUnit }}</span>
              <span class="success-indicator">✅ Completed</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
// Presentational card for the "Current Jobs" tab. All state/logic lives in
// PlanningView.vue — this only renders props and emits lifecycle actions up.
// Interfaces mirror the props contract (structural match with PlanningView's
// richly-commented MowJob/CompletedJob); the parent stays authoritative.
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

interface CompletedJob {
  id: number
  name: string
  completed_at: string
  actual_duration: number
  area_covered: number
}

defineProps<{
  jobs: MowJob[]
  completedJobs: CompletedJob[]
  areaUnit: string
  formatJobStatus: (status: string | undefined) => string
  formatDateTime: (dateString: string) => string
  formatArea: (value: unknown) => string
}>()

defineEmits<{
  refresh: []
  schedule: []
  start: [job: MowJob]
  pause: [job: MowJob]
  resume: [job: MowJob]
  cancel: [job: MowJob]
}>()
</script>

<style scoped>
/*
 * Shared card-shell/button primitives are duplicated here from
 * PlanningView.vue — Vue's scoped CSS only stamps the parent's scope-id onto
 * a child component's ROOT element, so nested elements need their own copy of
 * these rules to render identically. Keep in sync with PlanningView.vue (and
 * sibling planning cards) if any of these change.
 */
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
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.card-header h3 {
  margin: 0;
  color: var(--accent-green);
  font-size: 1.25rem;
}

.card-body {
  padding: 1.5rem;
}

.empty-state {
  text-align: center;
  padding: 3rem 1rem;
  color: var(--text-muted);
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

.btn-danger {
  background: #ff4343;
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

.btn-xs {
  padding: 0.25rem 0.5rem;
  font-size: 0.75rem;
}

/* Jobs-card-specific rules (moved verbatim from PlanningView.vue). */
.jobs-list {
  display: flex;
  flex-direction: column;
  gap: 1rem;
}

.job-item {
  background: var(--primary-dark);
  border: 1px solid var(--primary-light);
  border-radius: 8px;
  padding: 1rem;
  transition: all 0.3s ease;
}

.job-item:hover {
  border-color: var(--accent-green);
}

.job-header {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  margin-bottom: 1rem;
}

.job-title {
  flex: 1;
}

.job-title h4 {
  margin: 0 0 0.5rem 0;
  color: var(--text-color);
}

.status-badge {
  padding: 0.25rem 0.75rem;
  border-radius: 4px;
  font-size: 0.75rem;
  font-weight: 600;
  text-transform: uppercase;
}

.status-scheduled {
  background: rgba(0, 123, 255, 0.2);
  color: #007bff;
  border: 1px solid #007bff;
}

.status-running {
  background: rgba(0, 255, 146, 0.2);
  color: var(--accent-green);
  border: 1px solid var(--accent-green);
}

.status-paused {
  background: rgba(255, 193, 7, 0.2);
  color: #ffc107;
  border: 1px solid #ffc107;
}

.status-completed {
  background: rgba(40, 167, 69, 0.2);
  color: #28a745;
  border: 1px solid #28a745;
}

.job-actions {
  display: flex;
  gap: 0.5rem;
  flex-wrap: wrap;
}

.job-details {
  color: var(--text-muted);
  font-size: 0.875rem;
}

.job-info {
  display: flex;
  gap: 1rem;
  flex-wrap: wrap;
  margin-bottom: 1rem;
}

.job-progress {
  margin-top: 1rem;
}

.progress-bar {
  width: 100%;
  height: 8px;
  background: var(--primary-light);
  border-radius: 4px;
  overflow: hidden;
  margin-bottom: 0.5rem;
}

.progress-fill {
  height: 100%;
  background: var(--accent-green);
  transition: width 0.3s ease;
}

.progress-text, .time-remaining {
  font-size: 0.875rem;
  color: var(--text-muted);
  margin-right: 1rem;
}

.history-list {
  display: flex;
  flex-direction: column;
  gap: 0.75rem;
}

.history-item {
  background: var(--primary-dark);
  padding: 0.75rem;
  border-radius: 4px;
  border-left: 3px solid var(--accent-green);
}

.history-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 0.5rem;
}

.job-name {
  font-weight: 500;
  color: var(--text-color);
}

.completion-time {
  font-size: 0.875rem;
  color: var(--text-muted);
}

.history-details {
  display: flex;
  gap: 1rem;
  font-size: 0.875rem;
  color: var(--text-muted);
  flex-wrap: wrap;
}

.success-indicator {
  color: var(--accent-green);
}

@media (max-width: 768px) {
  .job-header {
    flex-direction: column;
    gap: 1rem;
    align-items: stretch;
  }

  .job-info {
    flex-direction: column;
    gap: 0.5rem;
  }
}
</style>
