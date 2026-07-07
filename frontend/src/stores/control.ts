import { defineStore } from 'pinia';
import { ref, computed } from 'vue';
import { sendControlCommand, getRoboHATStatus } from '../services/api';
import { useWebSocket } from '../services/websocket';

export const useControlStore = defineStore('control', () => {
  // State
  const lockoutActive = ref(false);
  const lockout = ref(false);
  const lockoutReason = ref('');
  const lockoutUntil = ref(null as string | null);
  const lastEcho = ref(null as null | Record<string, any>);
  const lastCommandEcho = ref(null as null | Record<string, any>);
  const lastCommandResult = ref(null as null | { result: string; status_reason?: string });
  const remediationLink = ref('');
  const isLoading = ref(false);
  const commandInProgress = ref(false);
  const robohatStatus = ref(null as null | Record<string, any>);

  // Safety-lockout push: the backend broadcasts interlock activate/clear events on the
  // "system.safety" topic over the (hub-wired) /ws/telemetry socket — /ws/control is a
  // separate, isolated echo-only connection that never emits this data. Track the set of
  // currently-active interlocks (keyed by interlock_type) since activate/clear are
  // per-interlock and multiple can be active at once; only fully clear the lockout when
  // none remain, otherwise a single cleared interlock would wrongly unlock the machine.
  const activeInterlocks = new Map<string, string>();

  function describeInterlock(interlock: any): string {
    return interlock?.description || interlock?.interlock_type || 'Safety interlock active';
  }

  function handleSafetyEvent(data: any) {
    const interlockType = data?.interlock?.interlock_type;
    if (!interlockType) return;
    if (data.action === 'activate') {
      activeInterlocks.set(interlockType, describeInterlock(data.interlock));
      lockout.value = true;
      lockoutActive.value = true;
      lockoutReason.value = describeInterlock(data.interlock);
    } else if (data.action === 'clear') {
      activeInterlocks.delete(interlockType);
      if (activeInterlocks.size === 0) {
        lockout.value = false;
        lockoutActive.value = false;
        lockoutReason.value = '';
      } else {
        lockoutReason.value = Array.from(activeInterlocks.values())[0];
      }
    }
  }

  const ws = useWebSocket('telemetry', {});

  // Actions
  async function submitCommand(command: string, payload: any = {}) {
    if (lockoutActive.value) {
      throw new Error(`Control locked out: ${lockoutReason.value}`);
    }
    
    commandInProgress.value = true;
    isLoading.value = true;
    try {
      const result = await sendControlCommand(command, payload);
      lastCommandResult.value = result;
      lastCommandEcho.value = result;
      if (result.result === 'blocked') {
        lockout.value = true;
        lockoutActive.value = true;
        lockoutReason.value = result.status_reason || 'SAFETY_LOCKOUT';
        remediationLink.value = result.remediation_link || '';
      }
      return result;
    } catch (e: any) {
      lastCommandResult.value = { result: 'error', status_reason: e?.message || 'Unknown error' };
      lockout.value = true;
      lockoutActive.value = true;
      lockoutReason.value = e?.message || 'Unknown error';
      remediationLink.value = e?.remediation_link || '';
      throw e;
    } finally {
      commandInProgress.value = false;
      isLoading.value = false;
    }
  }

  async function fetchRoboHATStatus() {
    try {
      const result = await getRoboHATStatus();
      robohatStatus.value = result;
      return result;
    } catch (e) {
      robohatStatus.value = null;
      throw e;
    }
  }

  // Computed properties
  const canSubmitCommand = computed(() => {
    return !lockoutActive.value && !commandInProgress.value;
  });

  const lockoutTimeRemaining = computed(() => {
    if (!lockoutUntil.value) return 0;
    const until = new Date(lockoutUntil.value).getTime();
    const now = Date.now();
    return Math.max(0, until - now);
  });

  // WebSocket management functions
  let unsubscribeFunction: (() => void) | null = null;

  function initWebSocket() {
    cleanup();
    unsubscribeFunction = ws.subscribe ? ws.subscribe('system.safety', handleSafetyEvent) : null;
    ws.connect?.();
  }

  function cleanup() {
    if (unsubscribeFunction) {
      unsubscribeFunction();
      unsubscribeFunction = null;
    }
  }

  initWebSocket();

  return {
    // State
    lockout,
    lockoutActive,
    lockoutReason,
    lockoutUntil,
    lastEcho,
    lastCommandEcho,
    lastCommandResult,
    remediationLink,
    isLoading,
    commandInProgress,
    robohatStatus,
    
    // Computed
    canSubmitCommand,
    lockoutTimeRemaining,
    
    // Actions
    submitCommand,
    fetchRoboHATStatus,
    initWebSocket,
    cleanup
  };
});
