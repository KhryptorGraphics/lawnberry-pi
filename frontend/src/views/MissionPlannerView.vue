<template>
  <div class="mission-planner-view">
    <h1>Mission Planner</h1>
    <div class="map-toolbar">
      <label class="follow-toggle"><input v-model="followMower" type="checkbox"> Follow mower</label>
      <button class="btn" :disabled="!mowerLatLng" @click="recenterToMower">🎯 Recenter</button>
      <button class="btn" :disabled="missionStore.waypoints.length === 0" @click="undoLastWaypoint">↩️ Undo last</button>
      <button class="btn btn-danger" :disabled="missionStore.waypoints.length === 0" @click="clearAllWaypoints">🗑️ Clear all</button>
    </div>
    <div class="map-container">
      <MissionMap
        ref="missionMapRef"
        :waypoints="missionStore.waypoints"
        :mower-position="mowerPosition"
        :follow-mower="followMower"
        @add-waypoint="handleAddWaypoint"
        @update-waypoint="handleUpdateWaypoint"
        @remove-waypoint="handleRemoveWaypoint"
      />
    </div>
    <MissionWaypointList />
    <div class="mission-controls">
      <input v-model="missionName" placeholder="Mission Name">
      <button :disabled="!missionName || missionStore.waypoints.length === 0" @click="createMission">Create Mission</button>
      <button :disabled="!missionStore.currentMission" @click="startMission">Start Mission</button>
      <button :disabled="missionStore.missionStatus !== 'running'" @click="pauseMission">Pause</button>
      <button :disabled="missionStore.missionStatus !== 'paused'" @click="resumeMission">Resume</button>
      <button :disabled="!missionStore.currentMission" @click="abortMission">Abort</button>
    </div>
    <div v-if="missionStore.currentMission">
      <h2>Mission Status: {{ missionStore.missionStatus }}</h2>
      <p>Progress: {{ missionStore.progress.toFixed(2) }}%</p>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted, onUnmounted, computed } from 'vue';
import MissionWaypointList from '@/components/MissionWaypointList.vue';
import MissionMap from '@/components/mission/MissionMap.vue';
import { useMissionStore } from '@/stores/mission';
import { useMapStore } from '@/stores/map';
import { useToastStore } from '@/stores/toast';
import { useWebSocket } from '@/services/websocket';

const missionStore = useMissionStore();
const mapStore = useMapStore();
const toast = useToastStore();
const telemetrySocket = useWebSocket('telemetry');

const missionMapRef = ref<any>(null);
const followMower = ref(true);
const mowerLatLng = ref<[number, number] | null>(null);
const gpsAccuracyMeters = ref<number | null>(null);
const missionName = ref('');
const restPollTimer = ref<number | null>(null);
const lastWsUpdateAt = ref<number>(0);

let navigationHandler: ((payload: any) => void) | null = null;
let componentDestroyed = false;

onMounted(async () => {
  componentDestroyed = false;

  // Ensure configuration for initial center
  if (!mapStore.configuration) {
    try {
      await mapStore.loadConfiguration('default');
    } catch (error) {
      console.error('Failed to load map configuration on mount:', error);
    }
  }

  // Telemetry subscription for live mower position
  try {
    await telemetrySocket.connect();
    navigationHandler = (payload: any) => {
      if (componentDestroyed) return;
      const pos = payload?.position;
      const lat = Number(pos?.latitude);
      const lon = Number(pos?.longitude);
      if (Number.isFinite(lat) && Number.isFinite(lon)) {
        mowerLatLng.value = [lat, lon];
        const accuracy = Number(pos?.accuracy);
        gpsAccuracyMeters.value = Number.isFinite(accuracy) ? accuracy : null;
        lastWsUpdateAt.value = Date.now();
      }
    };
    telemetrySocket.subscribe('telemetry.navigation', navigationHandler);
  } catch (error) {
    console.warn('Failed to initialize telemetry socket for mission planner:', error);
  }

  // REST fallback polling for environments that block WebSockets (e.g., some Cloudflare setups)
  restPollTimer.value = window.setInterval(async () => {
    try {
      // If we have not received a websocket update within 5 seconds, poll REST
      if (Date.now() - lastWsUpdateAt.value < 5000) return;
      const res = await fetch('/api/v2/dashboard/telemetry', { headers: { 'Cache-Control': 'no-cache' } })
      if (!res.ok) return
      const data = await res.json()
      const lat = Number(data?.position?.latitude)
      const lon = Number(data?.position?.longitude)
      if (Number.isFinite(lat) && Number.isFinite(lon)) {
        mowerLatLng.value = [lat, lon]
        const acc = Number(data?.position?.accuracy)
        gpsAccuracyMeters.value = Number.isFinite(acc) ? acc : null
      }
    } catch {/* ignore */}
  }, 2000)
});

onUnmounted(() => {
  componentDestroyed = true;

  if (navigationHandler) {
    telemetrySocket.unsubscribe('telemetry.navigation', navigationHandler);
    navigationHandler = null;
  }
  telemetrySocket.disconnect();

  if (restPollTimer.value) {
    clearInterval(restPollTimer.value)
    restPollTimer.value = null
  }
});

const mowerPosition = computed(() => {
  if (!mowerLatLng.value) return null;
  return {
    lat: mowerLatLng.value[0],
    lon: mowerLatLng.value[1],
    accuracy: gpsAccuracyMeters.value || 0,
  };
});

function handleAddWaypoint(lat: number, lon: number) {
  missionStore.addWaypoint(lat, lon);
}

function handleUpdateWaypoint(waypoint: any) {
  missionStore.updateWaypoint(waypoint);
}

function handleRemoveWaypoint(id: string) {
  missionStore.removeWaypoint(id);
}

function recenterToMower() {
  if (mowerLatLng.value && missionMapRef.value) {
    missionMapRef.value.recenter(mowerLatLng.value[0], mowerLatLng.value[1], 18);
  }
}

const createMission = () => {
  if (missionName.value) {
    missionStore.createMission(missionName.value);
  }
};

function clearAllWaypoints() {
  if (missionStore.waypoints.length && confirm('Clear all waypoints from this mission plan?')) {
    missionStore.clearWaypoints();
  }
}

function undoLastWaypoint() {
  missionStore.removeLastWaypoint();
}

const startMission = async () => {
  const ok = await missionStore.startCurrentMission();
  toast.show(ok ? 'Mission started' : 'Failed to start mission', ok ? 'success' : 'error');
};
const pauseMission = async () => {
  const ok = await missionStore.pauseCurrentMission();
  toast.show(ok ? 'Mission paused' : 'Failed to pause mission', ok ? 'success' : 'error');
};
const resumeMission = async () => {
  const ok = await missionStore.resumeCurrentMission();
  toast.show(ok ? 'Mission resumed' : 'Failed to resume mission', ok ? 'success' : 'error');
};
const abortMission = async () => {
  const ok = await missionStore.abortCurrentMission();
  toast.show(ok ? 'Mission aborted' : 'Failed to abort mission', ok ? 'success' : 'error');
};

</script>

<style scoped>
.mission-planner-view {
  display: flex;
  flex-direction: column;
  gap: 1rem;
}
.map-container {
  height: 500px;
  width: 100%;
  background: #1a1a1a; /* Dark background for loading state */
}
.mission-controls {
  display: flex;
  gap: 1rem;
  align-items: center;
}
.map-toolbar { display:flex; gap:1rem; align-items:center; }
.provider-badge { font-size:.85rem; opacity:.75; }
.follow-toggle { display:flex; align-items:center; gap:.4rem; }

/* Numbered waypoint dot */
.wp-pin-wrap { background: transparent; border: none; }
.wp-pin {
  width: 22px; height: 22px; border-radius: 50%;
  background: #00ffff; color: #001018; font-weight: 800;
  display: flex; align-items: center; justify-content: center;
  box-shadow: 0 0 8px rgba(0,255,255,0.6);
  border: 2px solid #001018;
}
.wp-pin span { font-size: 12px; line-height: 1; }
</style>
