<template>
  <div class="schedules-card">
    <!-- Recurring Schedules -->
    <div class="card">
      <div class="card-header">
        <h3>Recurring Schedules</h3>
        <button class="btn btn-sm btn-primary" @click="$emit('add')">
          ➕ Add Schedule
        </button>
      </div>
      <div class="card-body">
        <div v-if="schedules.length === 0" class="empty-state">
          <p>No recurring schedules configured</p>
        </div>

        <div v-else class="schedules-list">
          <div
            v-for="schedule in schedules"
            :key="schedule.id"
            class="schedule-item"
          >
            <div class="schedule-header">
              <div class="schedule-info">
                <h4>{{ schedule.name }}</h4>
                <span class="schedule-frequency">{{ formatFrequency(schedule.frequency) }}</span>
              </div>
              <div class="schedule-actions">
                <button
                  class="btn btn-xs"
                  :class="schedule.enabled ? 'btn-warning' : 'btn-success'"
                  @click="$emit('toggle', schedule)"
                >
                  {{ schedule.enabled ? '⏸️ Disable' : '▶️ Enable' }}
                </button>
                <button class="btn btn-xs btn-secondary" @click="$emit('edit', schedule)">
                  ✏️ Edit
                </button>
                <button class="btn btn-xs btn-danger" @click="$emit('delete', schedule)">
                  🗑️ Delete
                </button>
              </div>
            </div>

            <div class="schedule-details">
              <span>Zones: {{ schedule.zones.join(', ') }}</span>
              <span>Pattern: {{ schedule.pattern }}</span>
              <span>Next run: {{ formatDateTime(schedule.next_run) }}</span>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- Weather & Conditions -->
    <div class="card">
      <div class="card-header">
        <h3>Current Conditions</h3>
      </div>
      <div class="card-body">
        <div class="conditions-grid">
          <div class="condition-item">
            <div class="condition-label">Weather</div>
            <div class="condition-value" :class="weatherClass">
              {{ currentWeather.condition || 'Loading...' }}
            </div>
            <div class="condition-detail">
              {{ weatherTemperatureDisplay }}{{ temperatureUnit }}, {{ currentWeather.humidity_percent }}% humidity
            </div>
          </div>

          <div class="condition-item">
            <div class="condition-label">Mowing Recommendation</div>
            <div class="condition-value" :class="recommendationClass">
              {{ recommendation.advice }}
            </div>
            <div class="condition-detail">{{ recommendation.reason }}</div>
          </div>

          <div class="condition-item">
            <div class="condition-label">Ground Conditions</div>
            <div class="condition-value" :class="groundClass">
              {{ groundCondition }}
            </div>
            <div class="condition-detail">Last rain: {{ lastRain }}</div>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
// Presentational card for the "Scheduling" tab (recurring schedules + current
// conditions). State/logic lives in PlanningView.vue. Interface mirrors the
// props contract (structural match with PlanningView's MowSchedule).
interface MowSchedule {
  id: number
  name: string
  frequency: string
  zones: string[]
  pattern: string
  enabled: boolean
  next_run: string
}

interface WeatherState {
  temperature_c: number
  humidity_percent: number
  condition: string
}

defineProps<{
  schedules: MowSchedule[]
  currentWeather: WeatherState
  weatherClass: string
  weatherTemperatureDisplay: string
  temperatureUnit: string
  recommendation: { advice: string; reason: string }
  recommendationClass: string
  groundClass: string
  groundCondition: string
  lastRain: string
  formatFrequency: (frequency: string) => string
  formatDateTime: (dateString: string) => string
}>()

defineEmits<{
  add: []
  toggle: [schedule: MowSchedule]
  edit: [schedule: MowSchedule]
  delete: [schedule: MowSchedule]
}>()
</script>

<style scoped>
/*
 * Shared card-shell/button primitives duplicated from PlanningView.vue — see
 * JobsCard.vue for why (Vue scoped CSS only stamps the parent scope-id onto a
 * child's ROOT element). Keep in sync with PlanningView.vue and sibling cards.
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

/* Schedules-card-specific rules (moved verbatim from PlanningView.vue). */
.schedules-list {
  display: flex;
  flex-direction: column;
  gap: 1rem;
}

.schedule-item {
  background: var(--primary-dark);
  border: 1px solid var(--primary-light);
  border-radius: 8px;
  padding: 1rem;
  transition: all 0.3s ease;
}

.schedule-item:hover {
  border-color: var(--accent-green);
}

.schedule-header {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  margin-bottom: 1rem;
}

.schedule-info {
  flex: 1;
}

.schedule-info h4 {
  margin: 0 0 0.5rem 0;
  color: var(--text-color);
}

.schedule-actions {
  display: flex;
  gap: 0.5rem;
  flex-wrap: wrap;
}

.schedule-details {
  color: var(--text-muted);
  font-size: 0.875rem;
  display: flex;
  gap: 1rem;
  flex-wrap: wrap;
  margin-bottom: 1rem;
}

.schedule-frequency {
  font-size: 0.875rem;
  color: var(--text-muted);
}

.conditions-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 1rem;
}

.condition-item {
  background: var(--primary-dark);
  padding: 1rem;
  border-radius: 8px;
  text-align: center;
}

.condition-label {
  font-size: 0.875rem;
  color: var(--text-muted);
  margin-bottom: 0.5rem;
}

.condition-value {
  font-size: 1.25rem;
  font-weight: 600;
  margin-bottom: 0.5rem;
}

.condition-good {
  color: var(--accent-green);
}

.condition-warn {
  color: #ffc107;
}

.condition-bad {
  color: #ff4343;
}

.condition-detail {
  font-size: 0.875rem;
  color: var(--text-muted);
}

@media (max-width: 768px) {
  .schedule-header {
    flex-direction: column;
    gap: 1rem;
    align-items: stretch;
  }

  .schedule-details {
    flex-direction: column;
    gap: 0.5rem;
  }

  .conditions-grid {
    grid-template-columns: 1fr;
  }
}
</style>
