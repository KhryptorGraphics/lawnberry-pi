# Autonomy toolchain compatibility matrix (W0)

Assigns every pipeline stage of the autonomy goal to a host and records what
has actually been verified. A row is `verified` only with a tested command
and observed result. W0 is not complete until every row is verified or has a
recorded, accepted blocker.

Last run: 2026-10-06.

## Hosts

| Host | Access | Identity | Status |
|---|---|---|---|
| Mower Pi `pi5` | `ssh -J orangepi@192.168.1.140 kp@192.168.50.20` | Pi 5 8 GB, Ubuntu 24.04.4, kernel 6.8.0-raspi, Python 3.12.3. **Hailo-8L** (part `HM21LB1C2LAE`) on PCIe Gen3, firmware + HailoRT **4.24.0** (built from source: `scripts/pi_build_hailort.sh`, installed: `scripts/pi_install_hailort.sh`; apt 4.20 pkgs held). Backend on port **8081**. GNSS: **Quectel LC29H-DA** on `/dev/ttyAMA0` (RTK-fixed reached per `~/PI-SETUP-README.md`), not ZED-F9P. Cameras `/dev/video0-2`. | **verified** |
| x86 workshop `utrainn` | `ssh kp@192.168.1.53` | Ubuntu, 96 cores **without AVX2**, 377 GB RAM, driver 580.178.04, Docker 29.8.2 + `nvidia` runtime, xrdp running. V100 32 GB (reports `Tesla PG500-216`, sm_70, PCI 21:00), P100 16 GB (sm_60, PCI 04:00), 3080 Ti 12 GB (sm_86, PCI 07:00). | **verified**. 3080 Ti: CUDA works (correct 4096² FP16 matmul), but the power sensor reads ~416 W at idle, so `SW Power Cap` pins it at 1200 MHz in P2 (5.2 FP16 TFLOPS). Link is PCIe Gen3 **x4**. The GPUs are shared with other user jobs (`longcat-video` ~26 GB on the V100, `phosphene-nvidia` ~6 GB on the 3080 Ti). |
| HaLow link | Pi `halow0` 192.168.50.20 → gateway 192.168.50.1 → LAN | Morse Micro MM6108 (Seeed WM6108) | **measured at one location** (below) |

## Hardware deviations from the goal's fixed-hardware list

- Accelerator is **Hailo-8L**, not Hailo-8. `hailortcli fw-control identify` reports
  `Device Architecture: HAILO8L`, part `HM21LB1C2LAE` ("HAILO-8L AI ACC M.2 B+M KEY").
  HEFs must be compiled for `hailo8l`; the zoo's `hailo8` builds are rejected
  (`HAILO_HEF_NOT_COMPATIBLE_WITH_DEVICE`). Measured on HailoRT 4.24.0: yolo26s 41.7 FPS,
  yolo26m 20.6 FPS (hw_only, batch 1) — below the zoo's Hailo-8 spec figures (97.8/48.7),
  consistent with the halved MAC count.
- GNSS is **LC29H-DA**, not ZED-F9P. `backend` GPS drivers and the RTK-fix
  integrity metric must use its NMEA GGA quality field.

## HaLow link (Pi → Thor, 2026-10-06, single location, not yet across the yard)

| Test | Result |
|---|---|
| `iperf3` TCP Pi→Thor, 10 s | **4.79 Mbit/s** (~0.60 MB/s) |
| `iperf3 -R` TCP Thor→Pi, 10 s | **4.61 Mbit/s** |
| `ping` ×50 @ 5 Hz | 0% loss, RTT min/avg/max/mdev 7.3/18.0/73.2/12.0 ms |
| `iperf3 -u -b 200k -l 1200`, 10 s | 0% loss, jitter 4.12 ms |

The reported "1.2–1.5 MB/s" does not match this location: measured TCP is
~0.6 MB/s (4.8 Mbit/s). Worst-case protocol load is 1,200 B × 10 Hz = 12 kB/s =
0.096 Mbit/s per direction, about **2% of measured capacity** (50× margin).
Budget stays fixed at 1,200 B/datagram, 10 Hz. **Still required:** repeat at
the far and obstructed points of the yard, and record the worst result here.

## Stage assignments

| Stage | Host | Status | Evidence / blocker |
|---|---|---|---|
| Teleop capture logging (W1) | Pi | **built; unit-tested; Pi session blocked** | `backend/src/services/capture_service.py`, `backend/src/api/routers/capture.py`, `scripts/teleop.py`, `tests/unit/test_capture_service.py` (7 tests: clean session passes the gate, mid-stream hole rejected + renamed `.rejected`, tail loss ≤ flush buffer is undetectable by design, teleop rows carry `source`). Pi bench session (2026-10-06): the Pi dropped off HaLow because `systemctl stop lawnberry-backend` also stops `lawnberry-camera` (PartOf=), and ~10 s later the Morse SPI driver failed (`morse_spi cmd53_read ret:-71`). It stayed dark until power-cycled. **Do not stop or restart the backend without physical access.** Even when reachable, an indoor session **cannot pass**. The LC29H GGA shows 0 satellites (`...,0,00,99.99`) despite NTRIP flowing. There is also no IMU: `config.txt` sets `dtparam=i2c_arm=off`, so I2C bus 1 does not exist, and `BNO085Driver` is a stub that returns constant 0/0/0 on hardware. Do **not** repoint the probe at buses 13/14: they ACK every address with no device behind them, which would report a fake level IMU as ONLINE. `imu.jsonl` rows are null. |
| Capture integrity check (W1) | workshop / any | **verified (software)** | `backend/src/services/capture_integrity.py`, `tests/unit/test_capture_integrity.py` |
| Isaac Sim rendering on Thor | Thor | **not usable; root cause found** | NVIDIA lists aarch64 Isaac Sim as DGX-Spark-only. On the Thor, the `nvcr.io/nvidia/isaac-sim:6.0.0` arm64 container runs physics (`ISAAC_SMOKE z=0.250`), and Vulkan raster plus readback work (a viewport capture shows the grid). **Every RTX frame is black with zero boxes.** Vulkan validation (`--/renderer/debug/validation/enabled=true`) shows 25 RTX compute pipelines declaring `SPIR-V Capability RayQueryKHR`, while the JetPack 7.2.1 driver (595.78) does not expose `VK_KHR_ray_query`. It exposes `VK_KHR_ray_tracing_pipeline`/`acceleration_structure` but no ray query, opacity micromap or cluster AS. Results of fixes tried: the desktop arm64 595.80 user-space ICD reports the identical extension set. `--/rtx/supportRtInline=false` gives no change. A Vulkan layer faking `VK_KHR_ray_query` makes RTX take another path, but the driver rejects it (`vkCreateRayTracingPipelinesKHR` → `ERROR_INITIALIZATION_FAILED`, then a hang). The 5.1.0 builds (source and pip) segfault in `librtx.scenedb`. **Not fixable from user space; needs an NVIDIA Thor driver with ray-query support.** |
| Isaac Sim + synthetic data (W3/W4) | **workshop: RTX 3080 Ti (default, `ISAACSIM_GPU=1`)**; V100 fallback | **verified on both** | **3080 Ti:** full RTX path, despite the power cap. DLSS denoised (noise metric 0.19); native `bounding_box_2d_tight` and `semantic_segmentation` correct; **0.25 s/frame** warm (first run 13.6 s/frame while the shader cache built). **V100 fallback** (`ISAACSIM_GPU=2`): RTX skips it as "unsupported non-RTX GPU" unless run with `--/renderer/gpuEnumeration/rtxRequired=false`. No DLSS (noise 8.0 per frame, 2.4 averaged over 16). Isaac's synthetic-data CUDA kernels are sm_75–sm_90 only, so labels come from the exact emissive ID pass (`workshop/idpass.py`), used on both GPUs. V100 smoke (12 frames, 960×540, accum 8): 12/12 labelled, ~26 s/frame. Path tracing crashes on the V100; use Real-Time 2.0. |
| Isaac Sim desktop in RDP | workshop: RTX 3080 Ti | **verified** | `scripts/isaacsim-rdp.sh` (run on the workshop inside an RDP session) finds the caller's xrdp display (xorgxrdp or Xvnc backend; `:10` here), exports its X cookie for the container's uid 1234, and opens Isaac Sim Full 6.0.0 there with `--init` (no zombie containers). On the 3080 Ti: RTX Real-Time 2.0, asset browser connected; first start 379 s (cold shader cache, window blank until `app ready`). The viewport is ~3 FPS through the software Xvnc display and the x4 link. |
| SfM: COLMAP (W3) | workshop (CPU) | **installed** | `apt install colmap`: COLMAP 3.9.1, **built without CUDA**. Feature matching runs on CPU (96 cores). |
| Gaussian-splat reconstruction (W3) | workshop: RTX 3080 Ti (to test) | unverified | NuRec/3DGUT needs capture data first; the 3080 Ti is the RTX candidate (12 GB, power-capped). |
| Detector training (W5) | workshop: V100 (P100 also works) | **verified** | `workshop/train_detector.py`: YOLO26s, 3-epoch smoke on 1448 SDG train / 162 SDG val frames (`--smoke-sdg-val`; recall gates need real val, W2) on the P100 at ~2 it/s, batch 16. Needs `polars[rtcompat]` (no AVX2). |
| ONNX export (W5) | workshop | **verified** | opset 17, onnxslim; `onnx.checker` clean: `$LB/data/models/yolo26s_smoke/weights/best.onnx` |
| Hailo DFC compile to HEF (W5) | workshop (DFC is x86-only) | **blocked on wheel only** | `scripts/hailo_compile.sh --tag smoke --dry-run` exits 0; sole WARN is the missing DFC 3.34 wheel (Hailo developer-zone login). Target **`hailo8l`**; DFC 3.34 pairs with HailoRT 4.24 (installed on the Pi). |
| HailoRT inference (W5) | Pi | **verified** | HailoRT 4.24.0; `hailortcli benchmark` on zoo v2.19.0 hailo8l HEFs: yolo26s **41.7 FPS**, yolo26m **20.6 FPS** (hw_only, batch 1, `-t 10`) |
| Detector training fallback, mower parked only | Thor | verified (capability) | torch 2.14.0+cu130, 22.9 FP16 TFLOPS |
| Thor inference (ONNX → TensorRT) | Thor | **verified** | `trtexec` TensorRT 10.16.2 FP16 engine `PASSED` |
| ONNX Runtime GPU on Thor | Thor | blocked (not needed) | `onnxruntime 1.30.0` is CPU-only; use TensorRT |
| Coverage planner (W6) | Thor | **verified (software)** | `backend/src/nav/thor_strategist.py`, `tests/unit/test_thor_strategist.py` |
| Pi ↔ Thor link (W7) | Pi + Thor | protocol verified in software; link measured | `tests/unit/test_thor_link.py`, HaLow table above |

## Storage

`LB=/home/kp/nvme2/lawnberrypiserver` on the workshop (`nvme2`, 1.8 TB) is the single data root for
the autonomy pipeline. Nothing lives under `~/lawnberry-workshop` or `~/isaacsim-cache` any more.

```
$LB/
  env/                    conda env (python 3.11, torch 2.7.1+cu126)
  repo/                   rsync mirror of the lawnberry-pi repo (workshop/, scripts/; no git, no data/)
  cache/isaacsim/         Isaac Sim shader/asset caches (uid 1234 dirs)
  cache/huggingface/      HF_HOME for workshop scripts (Grounding DINO weights)
  cache/objaverse/        Objaverse metadata (*.json.gz)
  data/captures/          W1 teleop sessions, offloaded from the Pi
  data/datasets/          W2 reviewed real-image datasets, versioned dirs
  data/sdg/               W4 synthetic datasets, versioned dirs
  data/assets/objaverse/  curated glb/ + usd/ + manifest.json
  data/models/            W5 float training runs
  data/hailo/             model-zoo HEFs (mz-v2.19.0/) and our compiled HEFs (compiled/)
  data/smoke/             W0 smoke artifacts (w0/, sdg-isaac/)
  logs/                   long-job stdout
```

The Isaac launchers mount `$LB/data` at `/data` in the container (and the repo at `/workspace`), so
container paths are `/data/...`. `$LB/data` is world-writable because the image runs as uid 1234.
The workshop user's general HF cache (`HF_HOME=~/nvme2/hf-cache` in `~/.bashrc`, plus the legacy
`~/.cache/huggingface`) holds unrelated models and is not part of this layout; set
`HF_HOME=$LB/cache/huggingface` when running `workshop/autolabel.py`.

## Workshop environment (reproducible)

```bash
ssh kp@192.168.1.53
LB=~/nvme2/lawnberrypiserver
~/anaconda3/bin/conda create -y -p $LB/env python=3.11
P="$LB/env/bin/python -m pip"
$P install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu126
$P install -r workshop/requirements.txt "polars[rtcompat]"
# V100 only:
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2
```

Pins and why: torch cu128+ wheels drop sm_60/sm_70. transformers 5.18 returns
corrupted MM Grounding DINO boxes; 4.57.1 is correct. The base anaconda env on
the server has a broken `psutil` and is not used. The `$LB/env` conda env was
relocated from `~/lawnberry-workshop/env`: console-script shebangs still point
at the old prefix, so invoke everything as `$LB/env/bin/python -m <tool>` and
`$LB/env/bin/python script.py` (bare `$LB/env/bin/pip` etc. break).

## Decisions

- **Simulation, synthetic data and the Isaac desktop: workshop RTX 3080 Ti** (`ISAACSIM_GPU=1`,
  launcher default) despite its power-capped 1200 MHz clock: 0.25 s/frame with DLSS and native
  annotators. V100 (`ISAACSIM_GPU=2`, `rtxRequired=false`, frame averaging, ID-pass labels) is the
  fallback. Thor cannot render RTX (no `VK_KHR_ray_query` in JetPack 7.2.1). Repairing the 3080 Ti's
  power sensing would lift the clock cap.
- **Detector: YOLO26s** (Hailo Model Zoo v2.19 hailo8l: 41.7 FPS measured on the Pi;
  the zoo's Hailo-8 spec 97.8 FPS does not apply to this module),
  with **YOLO26m** (20.6 FPS measured) if hard-stop recall needs it. Train on the V100 with
  Ultralytics; compile with DFC **3.34.0** for `hailo8l`; run on HailoRT **4.24.0** (built from
  public sources on the Pi: `scripts/pi_build_hailort.sh`, installed by
  `scripts/pi_install_hailort.sh`). Zoo HEFs (hailo8 + hailo8l) and pretrained weights are
  in `$LB/data/hailo/mz-v2.19.0` (`SHA256SUMS` covers the hailo8 pair; the hailo8l
  downloads are recorded in `SHA256SUMS.hailo8l`).

## Open W0 actions

1. HaLow: repeat the measurements at far and obstructed yard points.
2. Hailo DFC 3.34.0: the only source is Hailo's developer zone (free login). It is not on PyPI,
   GitHub or Hugging Face; Hugging Face has only third-party HEFs for other tasks. Download the x86
   wheel to the workshop, then compile YOLO26s to a `hailo8` HEF and benchmark it on the Pi.
3. W4 asset gaps closed from Hugging Face `allenai/objaverse` (CC-BY only; reviewed via contact
   sheets; `workshop/objaverse_curation.json`): 10 pet, 13 wildlife, 19 root/stump, 2 sprinkler,
   2 hose models, converted to USD in `$LB/data/assets/objaverse`. Procedural hose coils and
   pop-up sprinkler heads (`isaac_sdg.PROCEDURAL`) supply half of those classes' instances.

## Synthetic data (W4)

Run on the workshop from the repo mirror (`$LB/repo`), after `docker login nvcr.io`:

```bash
scripts/isaacsim-headless.sh workshop/isaac_sdg.py --assets workshop/sdg_assets.json \
    --out /data/sdg/v1 --frames 500 --seed 1 --accum 2       # 3080 Ti: DLSS, little averaging needed
ISAACSIM_GPU=2 scripts/isaacsim-headless.sh ... --accum 16  # V100 fallback (no DLSS)
scripts/isaacsim-rdp.sh                                          # Isaac desktop in your RDP session
```

Output: `images/`, YOLO `labels/` (class index = `workshop/classes.py`), `rejected/` (frames whose ID
pass was not exact; never labelled), `manifest.json` (seed, asset-config hash, per-class counts).
The asset library is fetched over HTTPS from NVIDIA's S3 at run time; pin it by keeping
`assets_root` and `assets_sha256` from the manifest. Run
`$LB/env/bin/python -m workshop.build_assets_json --manifest $LB/data/assets/objaverse/manifest.json`
after every repo rsync (the generated config is untracked, so `rsync --delete` removes it).

ID pass on the 3080 Ti (measured, `workshop/idpass.py`): ~0.3% of pixels are 1-px colour blends at
silhouettes (largest island 65 px over 13 frames, unchanged with DLSS/AA off), tolerated as
background up to 200 px; and parts of objects intermittently render at exactly 2x brightness
(150/1751 frames of the first v1 attempt), so object ids use only levels {0, 85, 255} and a
doubled colour folds back to its unique source id.

**Dataset v1 status (2026-10-06): no usable v1 yet.** The run left in `$LB/data/sdg/v1-blown`
passed the 2% rejection gate (40/2000) but its RGB is blown out: frame 0 is normal (mean 134),
then 37 of 40 sampled frames are pure white. The same seed in `v1-wide-fold` rendered normally,
and only label code changed between the runs, so suspect render state left behind by the
per-frame ID pass. The 3080 Ti was shared with an unrelated LTX job at the time. **The
rejection gate cannot see this.** Spot-check image brightness, not just `rejected_frames`.
`v1-wide-fold` has good images but labels from the pre-fix fold; the ID-to-class map is not
saved, so it cannot be relabelled offline. Re-render v1 on an idle 3080 Ti.
