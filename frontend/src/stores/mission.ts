import { defineStore } from 'pinia';
import { ref } from 'vue';
import apiService from '@/services/api';
import { describeApiError } from '@/utils/apiErrors';

// Define the types right in the store file for simplicity
export interface Waypoint {
  id: string;
  lat: number;
  lon: number;
  blade_on: boolean;
  speed: number; // 0-100%
}

export interface Mission {
  id: string;
  name: string;
  waypoints: Waypoint[];
  created_at: string;
}

// GET /api/v2/missions/{id}/status (response_model=MissionStatus, backend/src/models/mission.py)
export interface MissionStatus {
  mission_id: string;
  status: string;
  current_waypoint_index?: number | null;
  completion_percentage: number;
}

export const useMissionStore = defineStore('mission', () => {
  const waypoints = ref<Waypoint[]>([]);
  const currentMission = ref<Mission | null>(null);
  const missionStatus = ref(''); // e.g., 'idle', 'running', 'paused'
  const progress = ref(0); // 0-100%
  // Why the last create/start/pause/resume/abort failed, as operator-facing text; '' after any success. The POST
  // routes answer 401 until an operator has signed in, which the planner used to report as a bare "Failed to ...".
  const lastError = ref('');
  let statusPollInterval: ReturnType<typeof setInterval> | null = null;

  const addWaypoint = (lat: number, lon: number) => {
    const newWaypoint: Waypoint = {
      id: new Date().toISOString(), // simple unique id
      lat,
      lon,
      blade_on: false,
      speed: 50,
    };
    waypoints.value.push(newWaypoint);
  };

  const removeWaypoint = (id: string) => {
    waypoints.value = waypoints.value.filter(wp => wp.id !== id);
  };
  
  const updateWaypoint = (updatedWaypoint: Waypoint) => {
    const index = waypoints.value.findIndex(wp => wp.id === updatedWaypoint.id);
    if (index !== -1) {
      waypoints.value[index] = updatedWaypoint;
    }
  };

  const reorderWaypoints = (newOrder: Waypoint[]) => {
    waypoints.value = newOrder;
  };

  const removeLastWaypoint = () => {
    if (waypoints.value.length > 0) {
      waypoints.value = waypoints.value.slice(0, -1);
    }
  };

  const clearWaypoints = () => {
    waypoints.value = [];
  };

  const createMission = async (name: string): Promise<boolean> => {
    try {
      const response = await apiService.post<Mission>('/api/v2/missions/create', {
        name,
        waypoints: waypoints.value,
      });
      lastError.value = '';
      currentMission.value = response.data;
      missionStatus.value = 'idle';
      return true;
    } catch (error) {
      console.error('Error creating mission:', error);
      lastError.value = describeApiError(error, 'Failed to create mission');
      return false;
    }
  };

  const startCurrentMission = async () => {
    if (!currentMission.value) return failNoMission();
    try {
      await apiService.post(`/api/v2/missions/${currentMission.value.id}/start`, {});
      lastError.value = '';
      missionStatus.value = 'running';
      startStatusPolling();
      return true;
    } catch (error) {
      console.error('Error starting mission:', error);
      lastError.value = describeApiError(error, 'Failed to start mission');
      return false;
    }
  };

  const pauseCurrentMission = async () => {
    if (!currentMission.value) return failNoMission();
    try {
      await apiService.post(`/api/v2/missions/${currentMission.value.id}/pause`, {});
      lastError.value = '';
      missionStatus.value = 'paused';
      stopStatusPolling();
      return true;
    } catch (error) {
      console.error('Error pausing mission:', error);
      lastError.value = describeApiError(error, 'Failed to pause mission');
      return false;
    }
  };

  const resumeCurrentMission = async () => {
    if (!currentMission.value) return failNoMission();
    try {
      await apiService.post(`/api/v2/missions/${currentMission.value.id}/resume`, {});
      lastError.value = '';
      missionStatus.value = 'running';
      startStatusPolling();
      return true;
    } catch (error) {
      console.error('Error resuming mission:', error);
      lastError.value = describeApiError(error, 'Failed to resume mission');
      return false;
    }
  };

  const abortCurrentMission = async () => {
    if (!currentMission.value) return failNoMission();
    try {
      await apiService.post(`/api/v2/missions/${currentMission.value.id}/abort`, {});
      lastError.value = '';
      missionStatus.value = 'aborted';
      stopStatusPolling();
      currentMission.value = null;
      return true;
    } catch (error) {
      console.error('Error aborting mission:', error);
      lastError.value = describeApiError(error, 'Failed to abort mission');
      return false;
    }
  };

  // start/pause/resume/abort with nothing loaded: report it like any other failure so lastError is never stale.
  const failNoMission = () => {
    lastError.value = 'No mission is loaded: create a mission first.';
    return false;
  };
  
  const pollMissionStatus = async () => {
    if (!currentMission.value) return;
    try {
      const response = await apiService.get<MissionStatus>(`/api/v2/missions/${currentMission.value.id}/status`);
      const status = response.data;
      missionStatus.value = status.status;
      progress.value = status.completion_percentage;
      if (status.status === 'completed' || status.status === 'aborted' || status.status === 'failed') {
        stopStatusPolling();
      }
    } catch (error) {
      console.error('Error polling mission status:', error);
    }
  };

  const startStatusPolling = () => {
    if (statusPollInterval) return;
    statusPollInterval = setInterval(pollMissionStatus, 2000);
  };

  const stopStatusPolling = () => {
    if (statusPollInterval) {
      clearInterval(statusPollInterval);
      statusPollInterval = null;
    }
  };

  return {
    waypoints,
    currentMission,
    missionStatus,
    progress,
    lastError,
    addWaypoint,
    removeWaypoint,
    updateWaypoint,
    reorderWaypoints,
    removeLastWaypoint,
    clearWaypoints,
    createMission,
    startCurrentMission,
    pauseCurrentMission,
    resumeCurrentMission,
    abortCurrentMission,
  };
});
