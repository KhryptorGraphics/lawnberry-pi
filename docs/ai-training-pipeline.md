# AI Training & Deployment Pipeline (Pi ↔ Thor)

How a manually-driven mow becomes an autonomous, location-aware policy on the
Pi. The Pi only records data and runs INT8 inference; training/distillation runs
on the NVIDIA Thor server.

## Data flow

```
Pi records a mow (perimeter_recorder, manual drive, blade on)
  └─ auto-uploads on stop      (THOR_UPLOAD_ENABLED=1, THOR_BASE_URL → Thor receiver)
       └─ Thor upload receiver  :15454  → data/real/uploaded/<session>/<task>/session.msgpack
            └─ session collector        → data/real/sessions/<id>.msgpack
                 └─ train_vla.py  → distill_hailo.py → Hailo .hef
                      └─ deploy .hef to Pi → autonomous AI navigation
```

## Pi side (this repo)

- **Recording → upload.** Finishing a recording queues it for upload when
  `THOR_UPLOAD_ENABLED=1`; point `THOR_BASE_URL` at Thor's receiver
  (`http://<thor-host>:15454`). See `.env.example` and
  `backend/src/services/thor_uploader.py`.
- **Location features (runtime).** `backend/src/nav/location_features.py` turns
  RTK GPS into an 8-d local-ENU feature vector (position east/north relative to
  the yard datum, heading sin/cos, speed, RTK fix/HDOP/satellite confidence) and
  maintains a causal 64×64 **coverage map** of already-mowed cells.
- **Inference inputs.** `ai_inference_service._preprocess` emits the deployed
  (distilled) model's inputs: `image (1,3,224,224)`, `sensors (1,20 =
  gps8+imu9+ultra3)`, `coverage_map (1,1,64,64)`.
- **Autonomous loop.** `navigation_service` builds a live frame, infers with the
  current location features + coverage snapshot, applies the prediction through
  the existing safety-gated motor path, then causally marks mowed cells.
  **Autonomy will not start on hardware until a real model is loaded.**

## Thor side (separate `~/thordrive/mower` project)

Teacher (`train_vla.py`) and the Hailo student (`distill_hailo.py`) consume the
same location + coverage features. Run under the `gr00t-thor` env (the only one
with `msgpack_numpy` + torch + torchvision). Train on `data/real/sessions`,
distil to a 3-input ONNX (`image`, `sensors`, `coverage_map`), compile to `.hef`
with the Hailo DFC on the Pi, then deploy.

## Yard-datum contract (required for location features to work)

Training and runtime must share one local coordinate frame. Set the Pi's
home/dock (geofence origin) and copy that latitude/longitude into the Thor
training config `config/training.yaml: data.yard_datum`. If they differ, the
model sees positions from a different distribution than it trained on.

## Prerequisites before autonomy

1. A real `.hef` trained on actual mow data (the first mow is manual — no model
   needed to collect it).
2. The Hailo SDK on the Pi to compile the ONNX → HEF.
3. RTK fixed solution (NTRIP configured) so location features are accurate.
4. On-device hardware-in-the-loop safety validation.
