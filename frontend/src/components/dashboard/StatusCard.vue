<template>
  <div class="retro-card status-card">
    <div class="card-header">
      <h3>◢ SYSTEM STATUS ◣</h3>
      <div class="status-indicator" :class="systemStatusClass" />
    </div>
    <div class="card-content">
      <div class="status-row">
        <span class="status-label">UPTIME:</span>
        <span class="status-value uptime">{{ uptime }}</span>
      </div>
      <div class="status-row">
        <span class="status-label">CONNECTION:</span>
        <span class="status-value" :class="connectionStatusClass">{{ connectionStatus }}</span>
      </div>
      <div class="status-row">
        <span class="status-label">STATUS:</span>
        <span class="status-value" :class="systemStatusClass">{{ systemStatus }}</span>
      </div>
      <div class="status-row">
        <span class="status-label">MODE:</span>
        <span class="status-value">{{ currentMode }}</span>
      </div>
      <div class="status-row">
        <span class="status-label">PROGRESS:</span>
        <span class="status-value">{{ progress.toFixed(0) }}%</span>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
defineProps<{
  systemStatusClass: string
  uptime: string
  connectionStatusClass: string
  connectionStatus: string
  systemStatus: string
  currentMode: string
  progress: number
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

.status-indicator.warning {
  background: #ffff00;
  color: #ffff00;
}

.status-indicator.error {
  background: #ff0040;
  color: #ff0040;
}

/* Status-card-specific rules (moved verbatim from DashboardView.vue). */
.status-row {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 1rem;
  padding: 0.5rem 0;
  border-bottom: 1px solid rgba(0, 255, 255, 0.2);
}

.status-row:last-child {
  border-bottom: none;
}

.status-label {
  font-weight: 700;
  letter-spacing: 1px;
  color: #ffff00;
  text-shadow: 0 0 5px rgba(255, 255, 0, 0.5);
}

.status-value {
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 1px;
}

.status-value.status-active, .status-value.uptime {
  color: #00ff00;
  text-shadow: 0 0 10px rgba(0, 255, 0, 0.7);
}

.status-value.status-warning {
  color: #ffff00;
  text-shadow: 0 0 10px rgba(255, 255, 0, 0.7);
}

.status-value.status-error {
  color: #ff0040;
  text-shadow: 0 0 10px rgba(255, 0, 64, 0.7);
}

.status-value.status-unknown {
  color: #666;
}
</style>
