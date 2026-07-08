<template>
  <div class="retro-card power-card">
    <div class="card-header">
      <h4>POWER SYSTEM</h4>
      <div class="power-indicator" :class="batteryIconClass" />
    </div>
    <div class="card-content power-content">
      <div class="battery-panel">
        <div class="battery-shell" :class="batteryBarClass">
          <span class="battery-percentage" data-testid="battery-percentage">{{ batteryLevelDisplay }}%</span>
        </div>
        <div class="battery-terminal" />
      </div>
      <div class="power-metrics">
        <div class="metric-line">
          <span class="metric-label">Battery Voltage</span>
          <span class="metric-value" data-testid="battery-voltage">{{ batteryVoltageDisplay }}V</span>
        </div>
        <div class="metric-line">
          <span class="metric-label">Battery Current</span>
          <span class="metric-value">{{ batteryCurrentDisplay }}A</span>
        </div>
        <div class="metric-line">
          <span class="metric-label">Battery Power</span>
          <span class="metric-value">{{ batteryPowerDisplay }}W</span>
        </div>
        <div class="metric-line">
          <span class="metric-label">Battery State</span>
          <span class="metric-value">{{ batteryChargeStateDisplay }}</span>
        </div>
        <div class="metric-line">
          <span class="metric-label">Solar Voltage</span>
          <span class="metric-value">{{ solarVoltageDisplay }}V</span>
        </div>
        <div class="metric-line">
          <span class="metric-label">Solar Current</span>
          <span class="metric-value">{{ solarCurrentDisplay }}A</span>
        </div>
        <div class="metric-line">
          <span class="metric-label">Solar Output</span>
          <span class="metric-value">{{ solarPowerDisplay }}W</span>
        </div>
        <div class="metric-line">
          <span class="metric-label">Solar Yield (Today)</span>
          <span class="metric-value">{{ solarYieldTodayDisplay }}Wh</span>
        </div>
        <div class="metric-line">
          <span class="metric-label">Load State</span>
          <span class="metric-value">{{ loadStateDisplay }}</span>
        </div>
        <div class="metric-line">
          <span class="metric-label">Load Current</span>
          <span class="metric-value">{{ loadCurrentDisplay }}A</span>
        </div>
        <div class="metric-line">
          <span class="metric-label">Load Power</span>
          <span class="metric-value">{{ loadPowerDisplay }}W</span>
        </div>
      </div>
    </div>
    <div class="metric-status solar-status" :class="solarStatusClass">{{ solarStatus }}</div>
  </div>
</template>

<script setup lang="ts">
defineProps<{
  batteryIconClass: string
  batteryBarClass: string
  batteryLevelDisplay: string
  batteryVoltageDisplay: string
  batteryCurrentDisplay: string
  batteryPowerDisplay: string
  batteryChargeStateDisplay: string
  solarVoltageDisplay: string
  solarCurrentDisplay: string
  solarPowerDisplay: string
  solarYieldTodayDisplay: string
  loadStateDisplay: string
  loadCurrentDisplay: string
  loadPowerDisplay: string
  solarStatusClass: string
  solarStatus: string
}>()
</script>

<style scoped>
/*
 * Shared card-shell primitives (.card-header, .card-content, indicators,
 * .metric-value/.metric-status) are duplicated here from DashboardView.vue
 * rather than left there, since Vue's scoped CSS only stamps the parent's
 * scope-id onto a child component's ROOT element — nested elements inside
 * this template (card-header, card-content, metric-value spans) need their
 * own copy of these rules to render identically. Keep in sync with
 * DashboardView.vue if either changes.
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
}

.status-indicator, .power-indicator, .activity-pulse, .log-indicator {
  width: 16px;
  height: 16px;
  border-radius: 50%;
  background: #666;
  box-shadow: 0 0 15px currentColor;
  animation: pulse 2s ease-in-out infinite;
}

@keyframes pulse {
  0%, 100% { transform: scale(1); opacity: 0.8; }
  50% { transform: scale(1.3); opacity: 1; }
}

.status-indicator.active, .power-indicator.active {
  background: #00ff00;
  color: #00ff00;
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

/* Power-card-specific rules (moved verbatim from DashboardView.vue). */
.power-card .power-content {
  display: flex;
  align-items: stretch;
  gap: 1.5rem;
  flex-wrap: wrap;
}

.battery-panel {
  position: relative;
  width: 160px;
  height: 70px;
  border: 2px solid rgba(0, 255, 255, 0.6);
  border-radius: 8px;
  display: flex;
  align-items: center;
  justify-content: center;
  background: rgba(0, 20, 40, 0.6);
  box-shadow: inset 0 0 20px rgba(0, 255, 255, 0.2);
}

.battery-shell {
  position: relative;
  width: 100%;
  height: 100%;
  display: flex;
  align-items: center;
  justify-content: center;
  border-radius: 6px;
  background: linear-gradient(135deg, rgba(0, 255, 255, 0.1), rgba(0, 100, 150, 0.3));
}

.battery-shell::after {
  content: '';
  position: absolute;
  inset: 6px;
  border-radius: 4px;
  background: rgba(0, 255, 255, 0.08);
}

.battery-percentage {
  position: relative;
  font-size: 1.8rem;
  font-weight: 700;
  color: #00ffff;
  z-index: 1;
  text-shadow: 0 0 12px rgba(0, 255, 255, 0.7);
}

.battery-terminal {
  position: absolute;
  right: -14px;
  top: 50%;
  transform: translateY(-50%);
  width: 12px;
  height: 24px;
  background: rgba(0, 255, 255, 0.6);
  border-radius: 3px;
  box-shadow: 0 0 12px rgba(0, 255, 255, 0.5);
}

.power-metrics {
  flex: 1;
  min-width: 200px;
  display: grid;
  gap: 0.6rem;
}

.power-metrics .metric-line {
  display: flex;
  justify-content: space-between;
  align-items: center;
  font-size: 0.9rem;
  letter-spacing: 1px;
}

.power-metrics .metric-label {
  color: rgba(0, 255, 255, 0.7);
  text-transform: uppercase;
}

.power-metrics .metric-value {
  color: #ffff00;
  font-weight: 600;
  font-size: 1.25rem;
}

.solar-status {
  margin-top: 0.25rem;
}

.solar-status.status-active {
  color: #00ff00;
  text-shadow: 0 0 10px rgba(0, 255, 0, 0.7);
}

.solar-status.status-warning {
  color: #ffff00;
  text-shadow: 0 0 10px rgba(255, 255, 0, 0.7);
}

.solar-status.status-error {
  color: #ff6600;
  text-shadow: 0 0 10px rgba(255, 102, 0, 0.7);
}

.solar-status.status-unknown {
  color: #888;
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
