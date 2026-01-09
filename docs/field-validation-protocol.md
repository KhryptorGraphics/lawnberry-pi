# Field Validation Protocol - LawnBerry Pi Autonomous Mower

This document defines the procedures for validating the autonomous mowing system in real-world conditions.

## Overview

Field validation ensures the AI-powered mowing system operates safely and effectively on actual lawns before unsupervised deployment. All tests require a safety observer present.

## Pre-Validation Checklist

### Hardware Verification

| Check | Requirement | How to Verify |
|-------|-------------|---------------|
| Emergency Stop | Physical button accessible | Press and release, verify motor stop |
| Battery Level | > 80% charge | `curl localhost:8000/api/v2/sensors/status \| jq .battery` |
| GPS Fix | RTK Fixed or Float | `curl localhost:8000/api/v2/sensors/status \| jq .gps` |
| All Sensors | Healthy status | `curl localhost:8000/api/v2/health` |
| Blade Guard | Installed and secure | Visual inspection |
| Wheels | Properly inflated, no damage | Visual inspection |

### Software Verification

```bash
# Run system health check
curl http://localhost:8000/api/v2/health

# Verify AI service ready
curl http://localhost:8000/api/v2/ai/health

# Check all sensors streaming
curl http://localhost:8000/api/v2/sensors/status
```

### Environmental Conditions

| Condition | Acceptable Range | Notes |
|-----------|-----------------|-------|
| Weather | Dry, no rain | Wet grass affects traction |
| Wind | < 20 mph | High wind affects GPS accuracy |
| Temperature | 40-95°F | Battery and motor limits |
| Visibility | Daylight preferred | Camera performance |
| Grass Height | 2-6 inches | Optimal cutting range |

## Test Lawn Configurations

### Configuration A: Simple Rectangular

```
+---------------------------+
|                           |
|     TEST AREA             |
|     20m x 15m             |
|     No obstacles          |
|                           |
+---------------------------+
```

**Purpose**: Baseline coverage efficiency test
**Success Criteria**:
- 95%+ coverage in single pass
- No boundary violations
- Consistent mowing pattern

### Configuration B: With Obstacles

```
+---------------------------+
|        [TREE]             |
|                           |
|    [ROCK]    TEST AREA    |
|              20m x 15m    |
|                  [BUSH]   |
+---------------------------+
```

**Purpose**: Obstacle avoidance validation
**Success Criteria**:
- All obstacles avoided (30cm+ clearance)
- Coverage around obstacles > 90%
- No collisions or near-misses

### Configuration C: Complex Boundary

```
    +-------+
    |       |
+---+       +---+
|               |
|   L-SHAPED    |
|   LAWN        |
+-------+       |
        |       |
        +-------+
```

**Purpose**: Complex boundary navigation
**Success Criteria**:
- Correct boundary following
- All corners covered
- No repeated missed areas

### Configuration D: Sloped Terrain

```
        TOP (10% grade)
       /               \
      /                 \
     /    SLOPED LAWN    \
    /      15m x 10m      \
   /                       \
  +-------------------------+
         BOTTOM
```

**Purpose**: Tilt safety and traction
**Success Criteria**:
- Stable operation up to 15° slope
- Auto-stop at 25°+ tilt
- No wheel slip or loss of control

## Test Procedures

### Test 1: Manual Override Verification

**Duration**: 5 minutes
**Observer**: Required

1. Start system in manual mode
2. Enable AI autonomous mode via API:
   ```bash
   curl -X POST http://localhost:8000/api/v2/ai/enable \
     -H "Content-Type: application/json" \
     -d '{"mode": "ai_autonomous"}'
   ```
3. While mowing, press physical e-stop
4. **Expected**: Immediate full stop within 0.5 seconds
5. Release e-stop, verify system requires manual restart
6. Test web interface stop button
7. Test API emergency stop:
   ```bash
   curl -X POST http://localhost:8000/api/v2/control/emergency_stop
   ```

**Pass Criteria**: All stop methods halt mower within 1 second

### Test 2: Obstacle Avoidance

**Duration**: 15 minutes
**Observer**: Required

1. Place test obstacles in lawn:
   - 1x large obstacle (person-sized cardboard)
   - 1x small obstacle (bucket)
   - 1x moving obstacle (push across path)
2. Start AI autonomous mode
3. Observe mower approach each obstacle
4. Record stopping distance and avoidance behavior

**Pass Criteria**:
- Stops or avoids all static obstacles at 30cm+
- Stops for moving obstacles within 50cm
- No contact with any obstacle
- Resumes operation after obstacle removed

### Test 3: Boundary Compliance

**Duration**: 20 minutes
**Observer**: Required

1. Define geofence boundary via web interface
2. Start AI autonomous mode
3. Let mower operate for full test duration
4. Monitor for boundary violations

**Pass Criteria**:
- Zero boundary violations
- Turns occur within boundary with 50cm+ margin
- GPS loss triggers safe stop within 30 seconds

### Test 4: Coverage Efficiency

**Duration**: 30-60 minutes (depends on lawn size)
**Observer**: Required for first 10 minutes, then periodic checks

1. Mark lawn with visible grid (chalk lines or flags)
2. Start AI autonomous mode
3. Record coverage progression at 10-minute intervals
4. Stop when mower signals completion or timeout

**Pass Criteria**:
- 90%+ coverage achieved
- Overlap < 20%
- No large missed areas (> 1m²)
- Battery consumption within expected range

### Test 5: Extended Operation (Soak Test)

**Duration**: 2 hours
**Observer**: Periodic checks every 15 minutes

1. Fully charge battery
2. Start AI autonomous mode
3. Monitor via web dashboard remotely
4. Check for:
   - Consistent inference times
   - No memory leaks
   - Stable GPS fix
   - No unexpected stops

**Pass Criteria**:
- Completes 2 hours without failure
- Inference time stable (< 50ms average)
- No safety interventions (unless obstacle present)
- Battery depletes linearly as expected

## Safety Observer Requirements

### Qualifications
- Familiar with mower operation
- Knows location of e-stop button
- Has mobile phone for emergencies
- Can run to mower within 10 seconds

### Responsibilities
1. Maintain visual contact with mower at all times
2. Keep unauthorized persons and pets away from test area
3. Ready to press e-stop immediately if:
   - Mower approaches person/animal
   - Mower leaves boundary
   - Mower tips or becomes unstable
   - Any unexpected behavior
4. Log all observations and anomalies

### Emergency Procedures

1. **Press E-Stop** - First action for any safety concern
2. **Clear Area** - Move people/pets away
3. **Assess Situation** - Determine cause of issue
4. **Document** - Record what happened for analysis
5. **Report** - Log incident in test documentation

## Data Collection During Tests

### Automatic Logging

The system automatically logs:
- All sensor readings at 10Hz
- AI inference decisions
- Motor commands
- Safety system activations
- GPS trajectory

Access logs:
```bash
# View real-time telemetry
curl http://localhost:8000/api/v2/ai/metrics

# Export session data
curl http://localhost:8000/api/v2/recording/sessions
```

### Manual Observations

Record in field notebook:
- Start/end times
- Weather conditions
- Any anomalies observed
- Obstacle encounters
- Coverage quality assessment
- Battery consumption

## Success Criteria Summary

| Test | Minimum Pass | Target |
|------|--------------|--------|
| Manual Override | 100% stop success | < 0.5s response |
| Obstacle Avoidance | 100% avoidance | 30cm+ clearance |
| Boundary Compliance | 0 violations | 50cm+ margin |
| Coverage Efficiency | 90% coverage | 95%+ coverage |
| Extended Operation | 2 hours stable | 4+ hours |

## Failure Handling

### If Test Fails

1. Document the failure mode completely
2. Collect logs from the session
3. Create GitHub issue with:
   - Test name and configuration
   - Expected vs actual behavior
   - Logs and sensor data
   - Environmental conditions
4. Do not proceed to next test until fixed

### Common Issues and Remediation

| Issue | Likely Cause | Remediation |
|-------|--------------|-------------|
| Slow obstacle response | Inference latency | Check Hailo status, reduce input resolution |
| Boundary violations | GPS drift | Verify RTK fix, check NTRIP connection |
| Missed coverage areas | Path planning issue | Adjust pattern parameters |
| Unexpected stops | Safety threshold too sensitive | Review ultrasonic/IMU calibration |
| Battery drains fast | Motor inefficiency | Check wheel alignment, blade balance |

## Sign-Off

Each validation session requires sign-off:

```
Test Date: _______________
Configuration: _______________
Observer Name: _______________
Tests Completed: [ ] 1  [ ] 2  [ ] 3  [ ] 4  [ ] 5
All Tests Passed: [ ] Yes  [ ] No
Notes: _______________________________________________
Signature: _______________
```

## Revision History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2026-01-09 | Claude | Initial protocol |
