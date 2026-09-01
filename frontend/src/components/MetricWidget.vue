<template>
  <div class="metric-widget" :class="`metric-${variant}`">
    <div class="metric-header">
      <div v-if="icon" class="metric-icon">
        <slot name="icon">{{ icon }}</slot>
      </div>
      <div class="metric-info">
        <h4 class="metric-label">{{ label }}</h4>
        <div class="metric-value">
          {{ formattedValue }}
          <span v-if="unit" class="metric-unit">{{ unit }}</span>
        </div>
      </div>
    </div>
    
    <div v-if="showProgress && typeof value === 'number'" class="metric-progress">
      <div class="progress-bar">
        <div
          class="progress-fill"
          :style="{ transform: `scaleX(${progressPercentage / 100})` }"
        />
      </div>
      <div class="progress-text">
        {{ progressPercentage }}% {{ progressLabel }}
      </div>
    </div>
    
    <div v-if="trend !== undefined" class="metric-trend" :class="trendClass">
      <span class="trend-icon">{{ trendIcon }}</span>
      <span class="trend-text">{{ trendText }}</span>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'

interface Props {
  label: string
  value: number | string
  unit?: string
  icon?: string
  showProgress?: boolean
  maxValue?: number
  trend?: number
  variant?: 'primary' | 'success' | 'warning' | 'danger' | 'info'
  progressLabel?: string
}

const props = withDefaults(defineProps<Props>(), {
  unit: '',
  icon: '',
  showProgress: false,
  maxValue: 100,
  trend: undefined,
  variant: 'primary',
  progressLabel: 'complete'
})

const formattedValue = computed(() => {
  if (typeof props.value === 'number') {
    return props.value.toLocaleString()
  }
  return props.value
})

const progressPercentage = computed(() => {
  if (typeof props.value === 'number' && props.maxValue > 0) {
    return Math.min(Math.max((props.value / props.maxValue) * 100, 0), 100)
  }
  return 0
})

const trendClass = computed(() => {
  if (props.trend === undefined) return ''
  if (props.trend > 0) return 'trend-up'
  if (props.trend < 0) return 'trend-down'
  return 'trend-neutral'
})

const trendIcon = computed(() => {
  if (props.trend === undefined) return ''
  if (props.trend > 0) return '↗'
  if (props.trend < 0) return '↘'
  return '→'
})

const trendText = computed(() => {
  if (props.trend === undefined) return ''
  const absValue = Math.abs(props.trend)
  if (props.trend > 0) return `+${absValue}%`
  if (props.trend < 0) return `-${absValue}%`
  return '0%'
})
</script>

<style scoped>
/*
 * Rebuilt on DESIGN.md's documented Telemetry Card (Neon Field Terminal
 * system) — was previously a plain white admin-panel card hardcoding
 * #3498db, unrelated to the rest of the app. This component feeds
 * TelemetryView and AIView, so this one file is most of that drift.
 */
.metric-widget {
  background: linear-gradient(135deg, #0a0a0a 0%, #1a1a2e 30%, #16213e 70%, #0a0a0a 100%);
  border: 2px solid #00ffff;
  border-left: 4px solid #00ffff;
  border-radius: 8px;
  padding: 1.5rem;
  position: relative;
  overflow: hidden;
  backdrop-filter: blur(10px);
  box-shadow:
    0 8px 32px rgba(0, 255, 255, 0.3),
    0 0 20px rgba(0, 255, 255, 0.2),
    inset 0 1px 0 rgba(255, 255, 255, 0.1),
    inset 0 0 30px rgba(0, 255, 255, 0.05);
  transition: transform 0.3s ease, box-shadow 0.3s ease;
}

.metric-widget:hover {
  transform: translateY(-4px);
  box-shadow:
    0 12px 40px rgba(0, 255, 255, 0.4),
    0 0 30px rgba(0, 255, 255, 0.3),
    inset 0 1px 0 rgba(255, 255, 255, 0.2),
    inset 0 0 40px rgba(0, 255, 255, 0.1);
}

.metric-primary {
  border-left-color: #00ffff;
}

.metric-success {
  border-left-color: #00ff00;
}

.metric-warning {
  border-left-color: #ffff00;
}

.metric-danger {
  border-left-color: #ff0040;
}

.metric-info {
  border-left-color: #00ff92;
}

.metric-header {
  display: flex;
  align-items: center;
  gap: 1rem;
  margin-bottom: 1rem;
}

.metric-icon {
  font-size: 1.5rem;
  color: #00ffff;
  min-width: 2rem;
  text-align: center;
  filter: drop-shadow(0 0 8px currentColor);
}

.metric-info {
  flex: 1;
}

.metric-label {
  margin: 0 0 0.25rem;
  font-size: 0.875rem;
  font-weight: 700;
  color: #00ffff;
  text-transform: uppercase;
  letter-spacing: 2px;
  font-family: 'Courier New', 'Consolas', monospace;
  text-shadow: 0 0 10px rgba(0, 255, 255, 0.5);
}

.metric-value {
  margin: 0;
  font-size: 2rem;
  font-weight: 900;
  color: #00ffff;
  line-height: 1;
  font-family: 'Orbitron', 'Courier New', monospace;
  letter-spacing: 1px;
  text-shadow: 0 0 20px rgba(0, 255, 255, 0.8), 0 0 40px rgba(0, 255, 255, 0.4);
}

.metric-unit {
  font-size: 1rem;
  font-weight: 400;
  color: #ffff00;
  margin-left: 0.25rem;
}

.metric-progress {
  margin-top: 1rem;
}

.progress-bar {
  height: 8px;
  background: rgba(0, 255, 255, 0.1);
  border: 1px solid rgba(0, 255, 255, 0.2);
  border-radius: 4px;
  overflow: hidden;
  margin-bottom: 0.5rem;
}

.progress-fill {
  height: 100%;
  width: 100%;
  background-color: #00ffff;
  border-radius: 4px;
  transform-origin: left;
  transition: transform 0.3s ease;
}

.metric-primary .progress-fill {
  background-color: #00ffff;
}

.metric-success .progress-fill {
  background-color: #00ff00;
}

.metric-warning .progress-fill {
  background-color: #ffff00;
}

.metric-danger .progress-fill {
  background-color: #ff0040;
}

.metric-info .progress-fill {
  background-color: #00ff92;
}

.progress-text {
  font-size: 0.75rem;
  color: rgba(0, 255, 255, 0.6);
  text-align: center;
  font-family: 'Courier New', 'Consolas', monospace;
  text-transform: uppercase;
  letter-spacing: 0.5px;
}

.metric-trend {
  display: flex;
  align-items: center;
  gap: 0.25rem;
  margin-top: 0.75rem;
  font-size: 0.875rem;
  font-weight: 700;
  font-family: 'Courier New', 'Consolas', monospace;
}

.trend-up {
  color: #00ff00;
  text-shadow: 0 0 10px rgba(0, 255, 0, 0.7);
}

.trend-down {
  color: #ff0040;
  text-shadow: 0 0 10px rgba(255, 0, 64, 0.7);
}

.trend-neutral {
  color: #ffff00;
  text-shadow: 0 0 10px rgba(255, 255, 0, 0.7);
}

.trend-icon {
  font-size: 1rem;
}

/* Mobile-first responsive design */
@media (max-width: 480px) {
  .metric-widget {
    padding: 1rem;
    margin-bottom: 1rem;
  }
  
  .metric-header {
    gap: 0.75rem;
    align-items: center;
  }
  
  .metric-icon {
    font-size: 1.25rem;
    min-width: 1.5rem;
  }
  
  .metric-value {
    font-size: 1.75rem;
    line-height: 1.1;
  }
  
  .metric-unit {
    font-size: 0.875rem;
  }
  
  .metric-label {
    font-size: 0.75rem;
    margin-bottom: 0.5rem;
  }
  
  .progress-text {
    font-size: 0.8rem;
  }
  
  .metric-trend {
    margin-top: 0.5rem;
    font-size: 0.8rem;
  }
}

@media (min-width: 481px) and (max-width: 768px) {
  .metric-widget {
    padding: 1.25rem;
  }
  
  .metric-value {
    font-size: 1.75rem;
  }
  
  .metric-header {
    gap: 1rem;
  }
}
</style>