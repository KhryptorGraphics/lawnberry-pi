<template>
  <div class="retro-card telemetry-card environmental-card">
    <div class="card-header">
      <h4>ENVIRONMENT</h4>
      <div class="temp-icon">🌡️</div>
    </div>
    <div class="card-content">
      <div class="env-grid">
        <div class="env-metric">
          <span class="metric-label">Temp</span>
          <span class="metric-value" data-testid="temperature-value">{{ temperatureDisplay }}<span class="unit">{{ temperatureUnit }}</span></span>
          <span class="metric-status" :class="tempStatusClass">{{ tempStatus }}</span>
        </div>
        <div class="env-metric">
          <span class="metric-label">Humidity</span>
          <span class="metric-value" data-testid="humidity-value">{{ humidityDisplay }}<span class="unit">%</span></span>
        </div>
        <div class="env-metric">
          <span class="metric-label">Pressure</span>
          <span class="metric-value" data-testid="pressure-value">{{ pressureDisplay }}<span class="unit">{{ pressureUnit }}</span></span>
        </div>
        <div class="env-metric">
          <span class="metric-label">Altitude</span>
          <span class="metric-value" data-testid="altitude-value">{{ altitudeDisplay }}<span class="unit">{{ altitudeUnit }}</span></span>
        </div>
      </div>
      <div class="env-source">Source: {{ environmentalSourceLabel }}</div>
    </div>
  </div>
</template>

<script setup lang="ts">
defineProps<{
  temperatureDisplay: string
  temperatureUnit: string
  tempStatusClass: string
  tempStatus: string
  humidityDisplay: string
  pressureDisplay: string
  pressureUnit: string
  altitudeDisplay: string
  altitudeUnit: string
  environmentalSourceLabel: string
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

.temp-icon {
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

.metric-status {
  font-size: 1rem;
  font-weight: 700;
  letter-spacing: 1px;
  text-transform: uppercase;
}

/* Environmental-card-specific rules (moved verbatim from DashboardView.vue). */
.environmental-card .env-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
  gap: 1rem;
}

.environmental-card .env-metric .metric-label {
  display: block;
  font-size: 0.8rem;
  letter-spacing: 2px;
  color: #00ffff;
  margin-bottom: 0.25rem;
}

.environmental-card .env-metric .metric-value {
  font-size: 1.25rem;
  font-weight: 700;
  color: #ffff00;
}

.environmental-card .env-metric .metric-status {
  font-size: 0.75rem;
  letter-spacing: 1px;
  display: inline-block;
  margin-top: 0.35rem;
}

.environmental-card .env-source {
  margin-top: 1rem;
  font-size: 0.75rem;
  letter-spacing: 2px;
  color: rgba(0, 255, 255, 0.7);
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
