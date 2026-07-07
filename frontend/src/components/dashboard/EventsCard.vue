<template>
  <div class="retro-card events-card">
    <div class="card-header">
      <h3>◢ SYSTEM LOG ◣</h3>
      <div class="log-indicator" />
    </div>
    <div class="card-content">
      <div class="events-terminal">
        <div
          v-for="event in recentEvents"
          :key="event.id"
          class="log-entry"
          :class="event.level"
        >
          <span class="log-time">[{{ formatTime(event.timestamp) }}]</span>
          <span class="log-level">{{ event.level.toUpperCase() }}:</span>
          <span class="log-message">{{ event.message }}</span>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
interface DashboardEvent {
  id: number
  timestamp: Date
  message: string
  level: 'info' | 'success' | 'warning' | 'error'
}

defineProps<{
  recentEvents: DashboardEvent[]
  formatTime: (timestamp: Date) => string
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

/* Event-card-specific rules (moved verbatim from DashboardView.vue). */
.events-terminal {
  background: #000;
  border: 2px solid #00ffff;
  padding: 1rem;
  max-height: 300px;
  overflow-y: auto;
  font-family: 'Courier New', monospace;
}

.log-entry {
  display: flex;
  margin-bottom: 0.5rem;
  font-size: 0.9rem;
  line-height: 1.4;
}

.log-time {
  color: #666;
  margin-right: 1rem;
  min-width: 80px;
}

.log-level {
  margin-right: 1rem;
  min-width: 60px;
  font-weight: 700;
}

.log-entry.info .log-level {
  color: #00ffff;
}

.log-entry.success .log-level {
  color: #00ff00;
}

.log-entry.warning .log-level {
  color: #ffff00;
}

.log-entry.error .log-level {
  color: #ff0040;
  animation: errorBlink 2s ease-in-out infinite;
}

@keyframes errorBlink {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.6; }
}

.log-message {
  flex: 1;
  color: #00ffff;
}
</style>
