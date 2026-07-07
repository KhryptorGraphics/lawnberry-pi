<template>
  <div class="retro-card calibration-card">
    <div class="card-header">
      <h3>◢ IMU CALIBRATION ◣</h3>
      <div class="calibration-indicator" :class="calibrationStatusClass" />
    </div>
    <div class="card-content">
      <div class="calibration-row">
        <span class="metric-label">Status</span>
        <span class="metric-value">{{ imuCalibrationLabel }}</span>
      </div>
      <div class="calibration-row">
        <span class="metric-label">Score</span>
        <span class="metric-value">{{ imuCalibrationScore }} / 3</span>
      </div>
      <div class="calibration-row">
        <span class="metric-label">Last Run</span>
        <span class="metric-value">{{ lastCalibrationSummary }}</span>
      </div>
      <button class="retro-btn calibrate-btn" :disabled="imuCalibrating || !imuSupported" @click="$emit('calibrate')">
        <span class="btn-icon">♻</span>
        {{ !imuSupported ? 'NOT SUPPORTED' : imuCalibrating ? 'CALIBRATING…' : 'RUN CALIBRATION' }}
      </button>
      <p v-if="calibrationError" class="calibration-error">⚠ {{ calibrationError }}</p>
      <p v-else-if="!imuSupported" class="calibration-note unsupported">IMU calibration not supported on this hardware.</p>
      <p v-else-if="lastCalibration?.notes" class="calibration-note">{{ lastCalibration?.notes }}</p>
    </div>
  </div>
</template>

<script setup lang="ts">
defineProps<{
  calibrationStatusClass: string
  imuCalibrationLabel: string
  imuCalibrationScore: number
  lastCalibrationSummary: string
  imuCalibrating: boolean
  imuSupported: boolean
  calibrationError: string | null
  lastCalibration: { notes?: string | null } | null
}>()

defineEmits<{
  calibrate: []
}>()
</script>

<style scoped>
/*
 * Shared card-shell primitives duplicated from DashboardView.vue — see
 * PowerCard.vue for why (Vue scoped CSS only stamps the parent scope-id
 * onto a child's ROOT element, not nested elements). Keep in sync.
 * Note: .calibration-card itself (on the root) still gets its
 * margin-bottom from DashboardView's own scoped rule via that root-leak
 * behavior — only the nested .card-header override needs duplicating.
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

.calibration-card .card-header {
  align-items: center;
}

/* Calibration-card-specific rules (moved verbatim from DashboardView.vue). */
.calibration-indicator {
  width: 16px;
  height: 16px;
  border-radius: 50%;
  background: #666;
  box-shadow: 0 0 15px currentColor;
}

.calibration-indicator.status-active {
  background: #00ff00;
  color: #00ff00;
}

.calibration-indicator.status-warning {
  background: #ffff00;
  color: #ffff00;
}

.calibration-indicator.status-error {
  background: #ff0040;
  color: #ff0040;
}

.calibration-indicator.status-unknown {
  background: #888;
  color: #888;
}

.calibration-row {
  display: flex;
  justify-content: space-between;
  align-items: center;
  font-family: 'Courier New', monospace;
  letter-spacing: 1px;
  margin-bottom: 0.75rem;
  text-transform: uppercase;
}

.calibration-row .metric-value {
  color: #ffff00;
  font-weight: 700;
}

.retro-btn {
  background: linear-gradient(135deg, #1a1a2e, #16213e, #0f0f23);
  border: 2px solid #00ffff;
  color: #00ffff;
  padding: 1.2rem 1.5rem;
  font-family: 'Orbitron', 'Courier New', monospace;
  font-weight: 700;
  font-size: 0.9rem;
  text-transform: uppercase;
  letter-spacing: 2px;
  cursor: pointer;
  transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
  position: relative;
  border-radius: 6px;
  overflow: hidden;
  backdrop-filter: blur(10px);
  box-shadow:
    0 4px 15px rgba(0, 255, 255, 0.2),
    inset 0 1px 0 rgba(255, 255, 255, 0.1);
}

.retro-btn::before {
  content: '';
  position: absolute;
  top: 0;
  left: -100%;
  width: 100%;
  height: 100%;
  background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.2), transparent);
  transition: left 0.5s ease;
}

.retro-btn:hover::before {
  left: 100%;
}

.retro-btn:hover:not(:disabled) {
  background: linear-gradient(135deg, #00ffff, #0a0a0a);
  color: #000;
  box-shadow: 0 0 20px rgba(0, 255, 255, 0.8);
  text-shadow: none;
}

.retro-btn:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.btn-icon {
  margin-right: 0.5rem;
  font-size: 1.2rem;
}

.calibrate-btn {
  margin-top: 1rem;
  width: 100%;
  background: rgba(0, 255, 255, 0.1);
  border: 1px solid #00ffff;
}

.calibrate-btn:hover:not(:disabled) {
  background: rgba(0, 255, 255, 0.2);
}

.calibration-error {
  margin-top: 0.75rem;
  color: #ff4343;
  font-size: 0.85rem;
  letter-spacing: 1px;
}

.calibration-note {
  margin-top: 0.75rem;
  font-size: 0.85rem;
  color: rgba(0, 255, 255, 0.7);
  letter-spacing: 1px;
}

.calibration-note.unsupported {
  color: rgba(255, 255, 0, 0.9);
  text-transform: uppercase;
}
</style>
