<template>
  <div class="card">
    <div class="card-header">
      <h3>Live Status</h3>
    </div>
    <div class="card-body">
      <div class="telemetry-grid">
        <div class="telemetry-item">
          <label>Battery</label>
          <div class="value">{{ telemetry.battery?.percentage?.toFixed(1) || 'N/A' }}%</div>
        </div>
        <div class="telemetry-item">
          <label>GPS</label>
          <div class="value">{{ telemetry.position?.latitude ? 'LOCKED' : 'SEARCHING' }}</div>
        </div>
        <div class="telemetry-item">
          <label>Speed</label>
          <div class="value">{{ displaySpeed }} {{ speedUnit }}</div>
        </div>
        <div class="telemetry-item">
          <label>Safety</label>
          <div class="value" :class="`safety-${telemetry.safety_state}`">
            {{ telemetry.safety_state?.toUpperCase() || 'UNKNOWN' }}
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
interface ControlTelemetry {
  battery?: { percentage?: number; voltage?: number | null }
  position?: { latitude?: number | null; longitude?: number | null }
  safety_state?: string
  velocity?: { linear?: { x?: number | null } }
  telemetry_source?: 'hardware' | 'simulated' | 'unknown'
}

defineProps<{
  telemetry: ControlTelemetry
  displaySpeed: string
  speedUnit: string
}>()
</script>

<style scoped>
/*
 * Shared card-shell primitives duplicated from ControlView.vue — see
 * SecurityGateCard.vue for why (Vue scoped CSS only stamps the parent
 * scope-id onto a child's ROOT element, not nested elements). Also kept in
 * ControlView.vue itself since its own remaining template still uses them.
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
}

.card-header h3 {
  margin: 0;
  color: var(--accent-green);
  font-size: 1.25rem;
}

.card-body {
  padding: 1.5rem;
}

/* Live-status-specific rules (moved verbatim from ControlView.vue). */
.telemetry-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
  gap: 1rem;
}

.telemetry-item {
  text-align: center;
  padding: 1rem;
  background: var(--primary-dark);
  border-radius: 4px;
}

.telemetry-item label {
  display: block;
  font-size: 0.875rem;
  color: var(--text-muted);
  margin-bottom: 0.5rem;
}

.telemetry-item .value {
  font-size: 1.5rem;
  font-weight: 600;
  color: var(--text-color);
}

.safety-safe {
  color: var(--accent-green);
}

.safety-warning {
  color: #ffc107;
}

.safety-danger {
  color: #ff4343;
}

@media (max-width: 768px) {
  .telemetry-grid {
    grid-template-columns: 1fr 1fr;
  }
}
</style>
