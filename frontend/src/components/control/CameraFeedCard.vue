<template>
  <div class="card">
    <div class="card-header">
      <h3>Live Camera Feed</h3>
    </div>
    <div class="card-body">
      <div class="camera-feed" :class="{ 'camera-feed-error': cameraError }">
        <img
          v-if="cameraDisplaySource"
          :src="cameraDisplaySource"
          alt="Live mower camera feed"
          class="camera-frame"
          :class="{ 'camera-frame--stream': cameraIsStreaming }"
          @load="$emit('stream-load')"
          @error="$emit('stream-error')"
        >
        <div v-else class="camera-placeholder">
          <p>{{ cameraStatusMessage }}</p>
          <button
            v-if="cameraError"
            class="btn btn-sm btn-secondary"
            @click="$emit('retry')"
          >
            Retry
          </button>
        </div>
        <div class="camera-badge">
          {{ cameraInfo.mode ? cameraInfo.mode.toUpperCase() : 'OFFLINE' }}
        </div>
      </div>
      <div class="camera-meta">
        <span :class="{ 'camera-meta-active': cameraInfo.active }">
          {{ cameraIsStreaming ? 'Streaming' : (cameraInfo.active ? 'Snapshots' : 'Idle') }}
        </span>
        <span>FPS: {{ formatCameraFps(cameraInfo.fps) }}</span>
        <span>Last frame: {{ formatCameraTimestamp(cameraLastFrame) }}</span>
        <span>Clients: {{ cameraInfo.client_count ?? '0' }}</span>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
interface CameraStatusSummary {
  active: boolean
  mode: string
  fps: number | null
  client_count: number | null
}

defineProps<{
  cameraError: string | null
  cameraDisplaySource: string | null
  cameraIsStreaming: boolean
  cameraStatusMessage: string
  cameraInfo: CameraStatusSummary
  cameraLastFrame: string | null
  formatCameraFps: (value?: number | null) => string
  formatCameraTimestamp: (timestamp?: string | null) => string
}>()

defineEmits<{
  'stream-load': []
  'stream-error': []
  retry: []
}>()
</script>

<style scoped>
/*
 * Shared card-shell primitives duplicated from ControlView.vue — see
 * SecurityGateCard.vue for why (Vue scoped CSS only stamps the parent
 * scope-id onto a child's ROOT element, not nested elements). Also kept in
 * ControlView.vue itself since its own remaining template still uses them.
 */
.card {
  background: var(--secondary-dark);
  border: 1px solid var(--primary-light);
  border-radius: 8px;
  margin-bottom: 2rem;
}

.card-header {
  background: var(--primary-dark);
  padding: 1rem 1.5rem;
  border-bottom: 1px solid var(--primary-light);
  border-radius: 8px 8px 0 0;
}

.card-header h3 {
  margin: 0;
  color: var(--accent-green);
  font-size: 1.25rem;
}

.card-body {
  padding: 1.5rem;
}

.btn {
  padding: 0.75rem 1.5rem;
  border: none;
  border-radius: 4px;
  font-weight: 500;
  cursor: pointer;
  transition: all 0.3s ease;
}

.btn:hover:not(:disabled) {
  transform: translateY(-2px);
}

.btn:disabled {
  opacity: 0.6;
  cursor: not-allowed;
}

.btn-sm {
  padding: 0.5rem 1rem;
  font-size: 0.875rem;
}

.btn-secondary {
  background: var(--primary-light);
  color: var(--text-color);
}

/* Camera-feed-specific rules (moved verbatim from ControlView.vue). */
.camera-feed {
  position: relative;
  display: flex;
  align-items: center;
  justify-content: center;
  min-height: 240px;
  background: #000;
  border-radius: 6px;
  overflow: hidden;
  border: 1px solid var(--primary-light);
}

.camera-feed-error {
  border-color: #ff4343;
}

.camera-frame {
  width: 100%;
  height: 100%;
  object-fit: cover;
}

.camera-frame--stream {
  image-rendering: auto;
}

.camera-placeholder {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  text-align: center;
  gap: 0.75rem;
  padding: 2rem;
  color: var(--text-muted);
}

.camera-badge {
  position: absolute;
  top: 0.75rem;
  left: 0.75rem;
  background: rgba(0, 0, 0, 0.65);
  color: #fff;
  padding: 0.3rem 0.75rem;
  border-radius: 4px;
  font-size: 0.75rem;
  letter-spacing: 0.05em;
}

.camera-meta {
  display: flex;
  flex-wrap: wrap;
  gap: 1rem;
  margin-top: 1rem;
  font-size: 0.875rem;
  color: var(--text-muted);
}

.camera-meta-active {
  color: var(--accent-green);
}
</style>
