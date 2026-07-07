<template>
  <div class="retro-card telemetry-card tof-card">
    <div class="card-header">
      <h4>TOF RANGE</h4>
      <div class="tof-icon">🛰️</div>
    </div>
    <div class="card-content tof-grid">
      <div class="tof-column">
        <div class="metric-label">LEFT</div>
        <div class="metric-value">{{ tofLeftDisplay }}<span class="unit">{{ tofUnit }}</span></div>
        <div class="metric-status" :class="tofStatusClass(tofLeft.status)">{{ formatTofStatus(tofLeft.status) }}</div>
      </div>
      <div class="tof-column">
        <div class="metric-label">RIGHT</div>
        <div class="metric-value">{{ tofRightDisplay }}<span class="unit">{{ tofUnit }}</span></div>
        <div class="metric-status" :class="tofStatusClass(tofRight.status)">{{ formatTofStatus(tofRight.status) }}</div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
interface TofState {
  distance: number | null
  status: string | null
  signal: number | null
}

defineProps<{
  tofLeft: TofState
  tofRight: TofState
  tofLeftDisplay: string
  tofRightDisplay: string
  tofUnit: string
  tofStatusClass: (status: string | null) => string
  formatTofStatus: (status: string | null) => string
}>()
</script>

<style scoped>
/*
 * Shared card-shell primitives duplicated from DashboardView.vue — see
 * PowerCard.vue for why (Vue scoped CSS only stamps the parent scope-id
 * onto a child's ROOT element, not nested elements). Keep in sync.
 */
.card-header {
  background: rgba(0, 255, 255, 0.1);
  padding: 1rem 1.5rem;
  border-bottom: 1px solid #00ffff;
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.card-header h3, .card-header h4 {
  margin: 0;
  font-size: 1.1rem;
  font-weight: 700;
  letter-spacing: 2px;
  color: #00ffff;
  text-shadow: 0 0 10px rgba(0, 255, 255, 0.7);
}

.card-content {
  padding: 1.5rem;
  text-align: center;
}

.metric-value {
  font-size: 2rem;
  font-weight: 900;
  color: #00ffff;
  font-family: 'Orbitron', 'Courier New', monospace;
  text-shadow:
    0 0 20px rgba(0, 255, 255, 0.8),
    0 0 40px rgba(0, 255, 255, 0.4),
    0 2px 4px rgba(0, 0, 0, 0.8);
  margin-bottom: 0.8rem;
  letter-spacing: 3px;
  position: relative;
  text-transform: uppercase;
}

.metric-value::after {
  content: '';
  position: absolute;
  bottom: -4px;
  left: 0;
  right: 0;
  height: 1px;
  background: linear-gradient(90deg, transparent, #00ffff, transparent);
  opacity: 0.6;
  animation: metricUnderline 2s ease-in-out infinite;
}

@keyframes metricUnderline {
  0%, 100% { opacity: 0.4; transform: scaleX(0.8); }
  50% { opacity: 0.8; transform: scaleX(1.2); }
}

.metric-value .unit {
  font-size: 1.5rem;
  color: #ffff00;
}

.metric-status {
  font-size: 1rem;
  font-weight: 700;
  letter-spacing: 1px;
  text-transform: uppercase;
}

/* ToF-card-specific rules (moved verbatim from DashboardView.vue; these
   override the generic .metric-value/.metric-status/.metric-label above
   via higher specificity, same as in the original single-file styles). */
.tof-card .tof-grid {
  display: flex;
  gap: 1.5rem;
  flex-wrap: wrap;
  justify-content: space-between;
}

.tof-card .tof-column {
  flex: 1 1 150px;
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
}

.tof-card .metric-label {
  font-size: 0.8rem;
  letter-spacing: 2px;
  color: #00ffff;
}

.tof-card .metric-value {
  font-size: 1.3rem;
  font-weight: 700;
  color: #ffff00;
}

.tof-card .metric-status {
  font-size: 0.8rem;
  letter-spacing: 1px;
  text-transform: uppercase;
}

@media (max-width: 768px) {
  .metric-value {
    font-size: 1.6rem;
  }

  .card-content {
    padding: 1rem;
  }
}

@media (max-width: 480px) {
  .metric-value {
    font-size: 1.4rem;
  }
}
</style>
