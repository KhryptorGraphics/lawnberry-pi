# LawnBerry Pi v2 🌱🤖

**Professional Autonomous Mowing System with AI-Powered Navigation**

Target platform: Raspberry Pi OS Lite Bookworm (64-bit) on Raspberry Pi 5 (16 GB) with Hailo 8L AI accelerator, Python 3.11.x, with graceful degradation validated on Raspberry Pi 4B (4–8 GB).

## 🚀 Quick Start

The LawnBerry Pi v2 system is now fully operational with hardware integration, AI-powered autonomous navigation, and real-time telemetry streaming.

### System Architecture
- **Backend**: FastAPI with hardware sensor integration (`backend/`)
- **Frontend**: Vue.js 3 with professional 1980s cyberpunk theme (`frontend/`)
- **Real-time**: WebSocket streaming at 5Hz for live telemetry
- **AI Engine**: Vision-Language-Action (VLA) model on Hailo 8L (13 TOPS)
- **Hardware**: Raspberry Pi 5, stereo cameras, GPS RTK, ultrasonic sensors, IMU

### Getting Started
```bash
# Backend (Terminal 1)
cd backend
python -m uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload

# Frontend (Terminal 2)  
cd frontend
npm run dev -- --host 0.0.0.0 --port 3001
```

### System Status
- ✅ **Hardware Integration**: Real sensor data streaming from Pi hardware
- ✅ **AI Autonomous Mode**: Vision-Language-Action model for intelligent navigation
- ✅ **Edge Inference**: Hailo 8L accelerator for 10+ FPS real-time control
- ✅ **Professional UI**: 1980s cyberpunk design with Orbitron fonts and neon effects
- ✅ **Real-time Telemetry**: Live GPS, battery, IMU data at 5Hz via WebSocket
- ✅ **Safety Systems**: Multi-layer obstacle detection and emergency stop
- ✅ **Production Ready**: Complete system validated on Raspberry Pi hardware

### Mission Planner
- ✅ Interactive Mission Planner UI is available under the "Mission Planner" navigation item.
- Click on the map to add waypoints, reorder them in the sidebar, set blade and speed per waypoint, then Create and Start the mission.
- Mission status and completion percentage are shown live; you can Pause, Resume, or Abort at any time.

### Documentation
- Setup Guide: `docs/installation-setup-guide.md`
- **AI Architecture**: `docs/ai-architecture.md` (VLA model, inference service)
- **Edge Deployment**: `docs/deployment-guide.md` (Hailo setup, production ops)
- GPS RTK Configuration: `docs/gps-ntrip-setup.md` (centimeter-level accuracy)
- Hardware Overview: `docs/hardware-overview.md` (sensors, accelerators)
- Hardware Integration: `docs/hardware-integration.md`
- Operations Guide: `docs/OPERATIONS.md`
- Contributing Guide: `CONTRIBUTING.md` (includes TODO policy)
- Feature Specifications: `specs/004-lawnberry-pi-v2/`
- Testing: `tests/` (unit, integration, contract tests)

### Access Points
- **Frontend**: http://192.168.50.215:3001 (or your Pi's IP)
- **Backend API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs (Swagger UI)
- **WebSocket**: ws://localhost:8000/api/v2/ws/telemetry

### AI Autonomous Control

Enable AI-powered autonomous mowing via the REST API:

```bash
# Enable AI autonomous mode
curl -X POST http://localhost:8000/api/v2/ai/enable \
  -H "Content-Type: application/json" \
  -d '{"mode": "ai_autonomous"}'

# Check AI status
curl http://localhost:8000/api/v2/ai/status

# View inference metrics
curl http://localhost:8000/api/v2/ai/metrics
```

The AI system uses a Vision-Language-Action model to process multi-modal sensor inputs (stereo depth, RGB vision, GPS, IMU) and output steering, throttle, and blade control commands at 10+ FPS.

The system provides a complete real-time dashboard for autonomous lawn mowing operations with professional-grade user experience, AI-powered navigation, and full hardware integration.