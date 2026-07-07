<template>
  <div class="retro-card telemetry-card gps-card">
    <div class="card-header">
      <h4>GPS NAVIGATION</h4>
      <div class="gps-icon">🧭</div>
    </div>
    <div class="card-content gps-content">
      <div class="gps-status-line" data-testid="gps-status">{{ gpsStatus }}</div>
      <div class="gps-grid">
        <div class="gps-metric">
          <span class="metric-label">Latitude</span>
          <span class="metric-value">{{ gpsLatitude ?? '--' }}</span>
        </div>
        <div class="gps-metric">
          <span class="metric-label">Longitude</span>
          <span class="metric-value">{{ gpsLongitude ?? '--' }}</span>
        </div>
        <div class="gps-metric">
          <span class="metric-label">Accuracy</span>
          <span class="metric-value">{{ gpsAccuracySummary }}</span>
        </div>
        <div class="gps-metric">
          <span class="metric-label">Satellites</span>
          <span class="metric-value">{{ gpsSatellitesDisplay }}</span>
        </div>
        <div class="gps-metric">
          <span class="metric-label">HDOP</span>
          <span class="metric-value">{{ gpsHdopDisplay }}</span>
        </div>
        <div class="gps-metric">
          <span class="metric-label">RTK</span>
          <span class="metric-value">{{ gpsRtkStatus ?? 'N/A' }}</span>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
defineProps<{
  gpsStatus: string
  gpsLatitude: string | null
  gpsLongitude: string | null
  gpsAccuracySummary: string
  gpsSatellitesDisplay: string
  gpsHdopDisplay: string
  gpsRtkStatus: string | null
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

.gps-icon {
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

/* GPS-card-specific rules (moved verbatim from DashboardView.vue). */
.gps-card .gps-content {
  display: flex;
  flex-direction: column;
  gap: 1rem;
}

.gps-status-line {
  font-size: 1rem;
  font-weight: 700;
  color: #00ff00;
  text-transform: uppercase;
  text-shadow: 0 0 12px rgba(0, 255, 0, 0.7);
}

.gps-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
  gap: 0.75rem;
}

.gps-metric {
  display: flex;
  flex-direction: column;
  gap: 0.2rem;
  font-size: 0.9rem;
  letter-spacing: 1px;
}

.gps-metric .metric-label {
  color: rgba(0, 255, 255, 0.7);
  text-transform: uppercase;
}

.gps-metric .metric-value {
  color: #ffff00;
  font-weight: 600;
  font-size: 1.1rem;
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
