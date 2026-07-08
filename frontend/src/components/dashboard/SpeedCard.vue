<template>
  <div class="retro-card telemetry-card">
    <div class="card-header">
      <h4>VELOCITY</h4>
      <div class="speed-icon">🚀</div>
    </div>
    <div class="card-content">
      <div class="metric-value" data-testid="speed-value">{{ speedDisplay }} <span class="unit">{{ speedUnit }}</span></div>
      <div class="speed-trend" :class="speedTrendClass">
        {{ speedTrend > 0 ? '▲' : speedTrend < 0 ? '▼' : '▬' }} {{ Math.abs(speedTrend) }}%
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
defineProps<{
  speedDisplay: string
  speedUnit: string
  speedTrendClass: string
  speedTrend: number
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

.speed-icon {
  font-size: 1.5rem;
  filter: drop-shadow(0 0 10px currentColor);
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

/* Speed-card-specific rules (moved verbatim from DashboardView.vue). */
.speed-trend {
  font-size: 1.2rem;
  font-weight: 700;
  letter-spacing: 1px;
}

.speed-trend.trend-up {
  color: #00ff00;
  text-shadow: 0 0 10px rgba(0, 255, 0, 0.7);
}

.speed-trend.trend-down {
  color: #ff0040;
  text-shadow: 0 0 10px rgba(255, 0, 64, 0.7);
}

.speed-trend.trend-stable {
  color: #ffff00;
  text-shadow: 0 0 10px rgba(255, 255, 0, 0.7);
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
