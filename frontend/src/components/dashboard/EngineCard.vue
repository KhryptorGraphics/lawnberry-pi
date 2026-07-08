<template>
  <div class="retro-card engine-card">
    <div class="card-header">
      <h4>ENGINE</h4>
      <span class="badge" :class="badgeClass">{{ badgeLabel }}</span>
    </div>
    <div class="card-content engine-content">
      <div class="metric-line">
        <span class="metric-label">Gear</span>
        <span class="metric-value">{{ gear }}</span>
      </div>
      <div class="metric-line">
        <span class="metric-label">Moving</span>
        <span class="metric-value">{{ moving ? 'YES' : 'NO' }}</span>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'

// Presentational only — engine/gear/moving is the confirmed minimal telemetry
// scope for the tractor platform (no fuel/RPM/battery fields). Badge classes
// match TractorControlView.vue's existing .badge-on/.badge-off/.badge-estop.
const props = defineProps<{
  engine: 'off' | 'starting' | 'running'
  emergencyStopActive: boolean
  gear: 'forward' | 'neutral' | 'reverse'
  moving: boolean
}>()

const badgeClass = computed(() =>
  props.emergencyStopActive ? 'badge-estop' : props.engine === 'running' ? 'badge-on' : 'badge-off'
)
const badgeLabel = computed(() => (props.emergencyStopActive ? 'E-STOP' : props.engine.toUpperCase()))
</script>

<style scoped>
/*
 * Shared card-shell primitives (.card-header, .card-content, .metric-*)
 * duplicated here from PowerCard.vue/DashboardView.vue — Vue's scoped CSS
 * only stamps the parent's scope-id onto a child component's ROOT element,
 * not nested elements inside this template, so each card keeps its own copy.
 * .retro-card itself is styled once in DashboardView.vue and applies here
 * because this component's root element also carries that parent scope-id.
 */
.card-header {
  background: rgba(0, 255, 255, 0.1);
  padding: 1rem 1.5rem;
  border-bottom: 1px solid #00ffff;
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.card-header h4 {
  margin: 0;
  font-size: 1.1rem;
  font-weight: 700;
  letter-spacing: 2px;
  color: #00ffff;
  text-shadow: 0 0 10px rgba(0, 255, 255, 0.7);
}

.card-content {
  padding: 1.5rem;
}

.engine-content {
  display: grid;
  gap: 0.6rem;
}

.metric-line {
  display: flex;
  justify-content: space-between;
  align-items: center;
  font-size: 0.9rem;
  letter-spacing: 1px;
}

.metric-label {
  color: rgba(0, 255, 255, 0.7);
  text-transform: uppercase;
}

.metric-value {
  color: #ffff00;
  font-weight: 600;
  font-size: 1.25rem;
  text-transform: capitalize;
}

.badge {
  padding: 0.25rem 0.6rem;
  border-radius: 999px;
  font-size: 0.75rem;
  font-weight: 700;
  text-transform: uppercase;
}

.badge-on {
  background: #d1f7e0;
  color: #0c7a43;
}

.badge-off {
  background: #eceff1;
  color: #607d8b;
}

.badge-estop {
  background: #fde2e1;
  color: #b42318;
}
</style>
