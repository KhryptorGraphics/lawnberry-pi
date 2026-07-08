<template>
  <div class="zones-card">
    <div class="card">
      <div class="card-header">
        <h3>Mowing Zones</h3>
        <button class="btn btn-sm btn-primary" @click="$emit('add')">
          ➕ Add Zone
        </button>
      </div>
      <div class="card-body">
        <div class="zones-grid">
          <div
            v-for="zone in zones"
            :key="zone.id"
            class="zone-card"
            :class="{ active: selectedZoneId === zone.id }"
            @click="$emit('select', zone)"
          >
            <div class="zone-header">
              <h4>{{ zone.name }}</h4>
              <span class="zone-priority" :class="`priority-${zone.priority}`">
                {{ formatPriority(zone.priority) }}
              </span>
            </div>

            <div class="zone-stats">
              <div class="stat">
                <span class="stat-label">Area</span>
                <span class="stat-value">{{ formatArea(zone.area_m2) }} {{ areaUnit }}</span>
              </div>
              <div class="stat">
                <span class="stat-label">Height</span>
                <span class="stat-value">{{ formatCuttingHeight(zone.cutting_height) }} {{ cuttingHeightUnit }}</span>
              </div>
              <div class="stat">
                <span class="stat-label">Last Cut</span>
                <span class="stat-value">{{ formatRelativeTime(zone.last_mowed) }}</span>
              </div>
            </div>

            <div class="zone-actions">
              <button class="btn btn-xs btn-success" @click.stop="$emit('mow', zone)">
                🌱 Mow Now
              </button>
              <button class="btn btn-xs btn-secondary" @click.stop="$emit('edit', zone)">
                ✏️ Edit
              </button>
            </div>
          </div>
        </div>
        <p v-if="!zones.length" class="empty-zones text-muted">
          No mowing zones defined yet. Draw your boundary and zones in the
          <button class="link-btn" type="button" @click="$emit('go-maps')">Maps editor</button>.
        </p>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
// Presentational card for the "Zones" tab. State/logic lives in
// PlanningView.vue. Interface mirrors the props contract (structural match
// with PlanningView's ZoneCard).
interface ZoneCard {
  id: string
  name: string
  area_m2: number
  cutting_height: number | null
  priority: 'high' | 'medium' | 'low'
  last_mowed: string | null
}

defineProps<{
  zones: ZoneCard[]
  selectedZoneId: string | null
  areaUnit: string
  cuttingHeightUnit: string
  formatPriority: (priority: string) => string
  formatArea: (value: unknown) => string
  formatCuttingHeight: (value: unknown) => string
  formatRelativeTime: (dateString: string | null | undefined) => string
}>()

defineEmits<{
  add: []
  select: [zone: ZoneCard]
  mow: [zone: ZoneCard]
  edit: [zone: ZoneCard]
  'go-maps': []
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

/* Zones-card-specific rules (moved verbatim from PlanningView.vue). */
.zones-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
  gap: 1rem;
}

.empty-zones {
  padding: 1rem 0;
}

.link-btn {
  background: none;
  border: none;
  padding: 0;
  color: var(--primary-color, #2563eb);
  cursor: pointer;
  text-decoration: underline;
  font: inherit;
}

.zone-card {
  background: var(--primary-dark);
  border: 1px solid var(--primary-light);
  border-radius: 8px;
  padding: 1rem;
  cursor: pointer;
  transition: all 0.3s ease;
}

.zone-card:hover {
  border-color: var(--accent-green);
  transform: translateY(-2px);
}

.zone-card.active {
  border-color: var(--accent-green);
  background: rgba(0, 255, 146, 0.1);
}

.zone-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 1rem;
}

.zone-header h4 {
  margin: 0;
  color: var(--text-color);
}

.zone-priority {
  padding: 0.25rem 0.5rem;
  border-radius: 4px;
  font-size: 0.75rem;
  font-weight: 600;
}

.priority-high {
  background: rgba(255, 67, 67, 0.2);
  color: #ff4343;
}

.priority-medium {
  background: rgba(255, 193, 7, 0.2);
  color: #ffc107;
}

.priority-low {
  background: rgba(0, 255, 146, 0.2);
  color: var(--accent-green);
}

.zone-stats {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 0.5rem;
  margin-bottom: 1rem;
}

.stat {
  text-align: center;
}

.stat-label {
  display: block;
  font-size: 0.75rem;
  color: var(--text-muted);
  margin-bottom: 0.25rem;
}

.stat-value {
  font-weight: 600;
  color: var(--text-color);
}

.zone-actions {
  display: flex;
  gap: 0.5rem;
}

@media (max-width: 768px) {
  .zones-grid {
    grid-template-columns: 1fr;
  }
}
</style>
