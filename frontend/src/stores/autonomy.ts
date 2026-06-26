import { defineStore } from 'pinia'
import { ref } from 'vue'
import {
  getNavigationStatus,
  pauseAutonomous,
  resumeAutonomous,
  startAutonomous,
  stopAutonomous,
} from '@/services/api'

export type AutonomyMode = 'idle' | 'autonomous' | 'paused'

export const useAutonomyStore = defineStore('autonomy', () => {
  const mode = ref<AutonomyMode>('idle')
  const active = ref(false)
  const completion = ref(0)
  const missionId = ref<string | null>(null)
  const error = ref<string | null>(null)
  let pollTimer: ReturnType<typeof setInterval> | null = null

  async function refreshStatus() {
    try {
      const s = await getNavigationStatus()
      mode.value = (s.mode as AutonomyMode) ?? 'idle'
      active.value = !!s.active
      completion.value = Number(s.completion_percentage ?? 0)
      missionId.value = s.mission_id ?? null
    } catch (e: any) {
      error.value = e?.message ?? 'Failed to fetch navigation status'
    }
  }

  async function start(zones?: string[]) {
    error.value = null
    try {
      const r = await startAutonomous(zones)
      missionId.value = r.mission_id ?? null
      mode.value = 'autonomous'
      active.value = true
      startPolling()
    } catch (e: any) {
      error.value = e?.message ?? 'Failed to start autonomous mode'
    }
  }

  async function stop() {
    try {
      await stopAutonomous()
    } finally {
      mode.value = 'idle'
      active.value = false
      completion.value = 0
      missionId.value = null
      stopPolling()
    }
  }

  async function pause() {
    await pauseAutonomous()
    mode.value = 'paused'
    await refreshStatus()
  }

  async function resume() {
    await resumeAutonomous()
    mode.value = 'autonomous'
    await refreshStatus()
  }

  function startPolling(intervalMs = 2000) {
    if (pollTimer) return
    pollTimer = setInterval(refreshStatus, intervalMs)
  }

  function stopPolling() {
    if (pollTimer) {
      clearInterval(pollTimer)
      pollTimer = null
    }
  }

  return {
    mode,
    active,
    completion,
    missionId,
    error,
    refreshStatus,
    start,
    stop,
    pause,
    resume,
    startPolling,
    stopPolling,
  }
})
