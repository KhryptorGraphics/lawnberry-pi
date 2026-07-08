<template>
  <div
    v-if="confirmStore.pending"
    class="confirm-overlay"
    @click.self="confirmStore.cancel()"
  >
    <div
      ref="dialogRef"
      class="confirm-dialog"
      role="alertdialog"
      aria-modal="true"
      :aria-label="confirmStore.pending.title || 'Confirm'"
      aria-describedby="confirm-dialog-message"
    >
      <h3 v-if="confirmStore.pending.title" class="confirm-title">{{ confirmStore.pending.title }}</h3>
      <p id="confirm-dialog-message" class="confirm-message">{{ confirmStore.pending.message }}</p>
      <div class="confirm-actions">
        <button class="btn" @click="confirmStore.cancel()">
          {{ confirmStore.pending.cancelLabel || 'Cancel' }}
        </button>
        <button
          class="btn"
          :class="confirmStore.pending.danger ? 'btn-danger' : 'btn-primary'"
          @click="confirmStore.confirm()"
        >
          {{ confirmStore.pending.confirmLabel || 'OK' }}
        </button>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed, ref } from 'vue'
import { useConfirmStore } from '@/stores/confirm'
import { useFocusTrap } from '@/composables/useFocusTrap'

const confirmStore = useConfirmStore()
const dialogRef = ref<HTMLElement | null>(null)
const isOpen = computed(() => !!confirmStore.pending)

useFocusTrap(dialogRef, isOpen, () => confirmStore.cancel())
</script>

<style scoped>
.confirm-overlay {
  position: fixed;
  inset: 0;
  background: rgba(0, 0, 0, 0.55);
  z-index: 2600;
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 1rem;
}
.confirm-dialog {
  width: min(420px, 100%);
  background: #121212;
  color: #e0e0e0;
  border-radius: 12px;
  border: 1px solid rgba(0, 255, 255, 0.25);
  box-shadow: 0 30px 80px rgba(0, 0, 0, 0.5);
  padding: 1.25rem;
}
.confirm-title {
  margin: 0 0 0.5rem;
}
.confirm-message {
  margin: 0 0 1.25rem;
  line-height: 1.4;
}
.confirm-actions {
  display: flex;
  justify-content: flex-end;
  gap: 0.5rem;
}
.btn {
  padding: 0.5rem 1rem;
  border-radius: 6px;
  border: 1px solid rgba(255, 255, 255, 0.2);
  background: rgba(255, 255, 255, 0.06);
  color: inherit;
  cursor: pointer;
}
.btn-primary {
  border-color: rgba(0, 255, 255, 0.5);
  background: rgba(0, 200, 255, 0.15);
}
.btn-danger {
  border-color: rgba(255, 67, 67, 0.6);
  background: rgba(255, 67, 67, 0.15);
  color: #ff6b6b;
}
</style>
