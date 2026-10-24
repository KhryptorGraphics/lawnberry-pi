# Autonomy toolchain compatibility matrix (W0)

Assigns every pipeline stage of the autonomy goal to a host and records what
has actually been verified. A row is `verified` only with a tested command
and observed result. W0 is not complete until every row is verified or has a
recorded, accepted blocker.

Last run: 2026-10-06.

## Hosts

| Host | Access | Identity | Status |
|---|---|---|---|
| Mower Pi `pi5` | From the Thor: `ssh kp@192.168.50.20` (needs the Thor's `192.168.50.0/24 via 192.168.1.140` route). From other LAN hosts: `ssh -J orangepi@192.168.1.140 kp@192.168.50.20` | Pi 5 8 GB, Ubuntu 24.04.4, kernel 6.8.0-raspi, Python 3.12.3. **Hailo-8L** (part `HM21LB1C2LAE`) on PCIe Gen3, firmware + HailoRT **4.24.0** (built from source: `scripts/pi_build_hailort.sh`, installed: `scripts/pi_install_hailort.sh`; apt 4.20 pkgs held). Backend on port **8081**. GNSS: **Quectel LC29H-DA** on `/dev/ttyAMA0` (RTK-fixed reached per `~/PI-SETUP-README.md`), not ZED-F9P. Cameras `/dev/video0-2`. | **verified** |
| x86 workshop `utrainn` | `ssh kp@192.168.1.53` | Ubuntu, 96 cores **without AVX2**, 377 GB RAM, driver 580.178.04, Docker 29.8.2 + `nvidia` runtime, xrdp running. V100 32 GB (reports `Tesla PG500-216`, sm_70, PCI 21:00), P100 16 GB (sm_60, PCI 04:00), 3080 Ti 12 GB (sm_86, PCI 07:00). | **verified**. 3080 Ti: CUDA works (correct 4096² FP16 matmul), but the power sensor reads ~416 W at idle, so `SW Power Cap` pins it at 1200 MHz in P2 (5.2 FP16 TFLOPS). Link is PCIe Gen3 **x4**. The GPUs are shared with other user jobs (`longcat-video` ~26 GB on the V100, `phosphene-nvidia` ~6 GB on the 3080 Ti). |
| Thor `agx-thor` | local shell on the Thor itself | aarch64, Ubuntu 24.04.5, 131.86 GB RAM, 1213 GB free, driver 595.78, Vulkan 1.4 with RT extensions. Isaac Sim **6.1.0-rc.26 (arm64, native)** at `~/isaacsim` plus the `nvcr.io/nvidia/isaac-sim:6.0.0` **arm64** container. torch 2.14.0+cu130 works; no ultralytics. | **launches, does not render** (2026-10-23) |
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

### HaLow outage and re-measurement (2026-10-13, Pi ↔ AP, shed position)

The AP radio (Orange Pi 4 Pro, HT-HC01P on SPI3) vanished on 2026-10-08 01:39 UTC, about 48 h
after its TX cap was removed (28 dBm). It was restored by a cold power cycle with
`tx_max_power_mbm=2000` (20 dBm). The driver is never reloaded live; the AP's SPI-error guard only stops hostapd. Measured between the Pi 5 and
the AP (192.168.50.1). These numbers are not Pi → Thor. 4 MHz, channel 40; Pi signal at the AP is −56 dBm.

| Test | Result |
|---|---|
| `iperf3` TCP Pi→AP, 30 s | **4.84 Mbit/s** |
| `iperf3 -R` TCP AP→Pi, 30 s | **5.17 Mbit/s** |
| UDP 500 kbit/s, 1,200 B, 70 s (video-like), with `ping` 10 Hz ×600 alongside | UDP 0/3,646 lost, jitter 5.9 ms; ping 0% loss, RTT max 48.9 ms |
| `ping` 1,200 B, 1 Hz ×1,800 (30 min) | 0% loss, RTT avg 18.5 ms, max 256 ms |
| Guard trips / SPI failures during the tests | 0 / 0 |

A 500 kbit/s monitoring stream plus 10 Hz control fits with RTT far below the 0.5 s safe-hold
timeout. The Thor reaches the Pi directly (`ssh kp@192.168.50.20`, backend `http://192.168.50.20:8081`)
through the route `192.168.50.0/24 via 192.168.1.140` in its NetworkManager profile "Wired connection 2".
That `/24` overrides the unused Docker network `app-network` (`192.168.48.0/20`, which contains the HaLow
subnet). Other LAN hosts use the jump host `orangepi@192.168.1.140` or the AP's forwards
(`192.168.1.140:2222` → Pi ssh, `:1337` → Pi web UI). The Pi's home WiFi is near-dead in the shed
(−83 dBm, ~43% loss), so on the Pi `halow-lan-routes.timer` (every 15 s, station mode only) keeps
`192.168.1.0/24` and `192.168.1.1` routed via HaLow (metric 500 vs wlan0's 600). It restored a
manually deleted route in 12 s. A real HaLow drop and a Pi reboot were not exercised. Failure
evidence is kept on the AP in `/var/lib/halow-diag/`. The 2026-10-13 section of its README-STATE.md
has the details.

## Stage assignments

| Stage | Host | Status | Evidence / blocker |
|---|---|---|---|
| Teleop capture logging (W1) | Pi | **built; unit-tested; Pi session blocked** | `backend/src/services/capture_service.py`, `backend/src/api/routers/capture.py`, `scripts/teleop.py`, `tests/unit/test_capture_service.py` (7 tests: clean session passes the gate, mid-stream hole rejected + renamed `.rejected`, tail loss ≤ flush buffer is undetectable by design, teleop rows carry `source`). Pi bench session (2026-10-06): the Pi dropped off HaLow because `systemctl stop lawnberry-backend` also stops `lawnberry-camera` (PartOf=), and ~10 s later the Morse SPI driver failed (`morse_spi cmd53_read ret:-71`). It stayed dark until power-cycled. **Do not stop or restart the backend without physical access.** Even when reachable, an indoor session **cannot pass**. The LC29H GGA shows 0 satellites (`...,0,00,99.99`) despite NTRIP flowing. There is also no IMU: `config.txt` sets `dtparam=i2c_arm=off`, so I2C bus 1 does not exist, and `BNO085Driver` is a stub that returns constant 0/0/0 on hardware. Do **not** repoint the probe at buses 13/14: they ACK every address with no device behind them, which would report a fake level IMU as ONLINE. `imu.jsonl` rows are null. |
| Capture integrity check (W1) | workshop / any | **verified (software)** | `backend/src/services/capture_integrity.py`, `tests/unit/test_capture_integrity.py` |
| Isaac Sim rendering on Thor | Thor | **not usable; root cause found** | NVIDIA lists aarch64 Isaac Sim as DGX-Spark-only. On the Thor, the `nvcr.io/nvidia/isaac-sim:6.0.0` arm64 container runs physics (`ISAAC_SMOKE z=0.250`), and Vulkan raster plus readback work (a viewport capture shows the grid). **Every RTX frame is black with zero boxes.** Vulkan validation (`--/renderer/debug/validation/enabled=true`) shows 25 RTX compute pipelines declaring `SPIR-V Capability RayQueryKHR`, while the JetPack 7.2.1 driver (595.78) does not expose `VK_KHR_ray_query`. It exposes `VK_KHR_ray_tracing_pipeline`/`acceleration_structure` but no ray query, opacity micromap or cluster AS. Results of fixes tried: the desktop arm64 595.80 user-space ICD reports the identical extension set. `--/rtx/supportRtInline=false` gives no change. A Vulkan layer faking `VK_KHR_ray_query` makes RTX take another path, but the driver rejects it (`vkCreateRayTracingPipelinesKHR` → `ERROR_INITIALIZATION_FAILED`, then a hang). The 5.1.0 builds (source and pip) segfault in `librtx.scenedb`. **Not fixable from user space; needs an NVIDIA Thor driver with ray-query support.** |
| Isaac Sim + synthetic data (W3/W4) | **workshop: RTX 3080 Ti (`ISAACSIM_GPU=1`)**; V100 fallback | **Verified on both; native labels preferred on 3080 Ti** | 3080 Ti: full RTX path; native Replicator `bounding_box_2d_tight` + `instance_id_segmentation` labels verified. Warm RT 2.0 is 0.25 s/frame after shader cache. Path-traced v2 at 256 spp completed 2,000 frames (14.5 s/frame benchmark). **Do not use the 3080 Ti ID pass for production labels:** visual comparison found cross-object box mismatches. V100 (`ISAACSIM_GPU=2`) needs `--/renderer/gpuEnumeration/rtxRequired=false`; no DLSS, average 16 frames, use the emissive ID pass. V100 smoke: 12/12 labelled at ~26 s/frame. PathTracing crashes on V100. |
| Isaac Sim desktop in RDP | workshop: RTX 3080 Ti | **verified** | `scripts/isaacsim-rdp.sh` (run on the workshop inside an RDP session) finds the caller's xrdp display (xorgxrdp or Xvnc backend; `:10` here), exports its X cookie for the container's uid 1234, and opens Isaac Sim Full 6.0.0 there with `--init` (no zombie containers). On the 3080 Ti: RTX Real-Time 2.0, asset browser connected; first start 379 s (cold shader cache, window blank until `app ready`). The viewport is ~3 FPS through the software Xvnc display and the x4 link. |
| SfM: COLMAP (W3) | workshop (CPU) | **installed** | `apt install colmap`: COLMAP 3.9.1, **built without CUDA** (`colmap -h`: "without CUDA"; no CUDA libs in `ldd`). Feature matching runs on CPU (96 cores). GPU SfM: pyCuSFM (below). |
| Gaussian-splat reconstruction (W3) | workshop: RTX 3080 Ti | **installed; tracer builds and loads; not yet trained on data** | 3DGRUT (`envs/3dgrut`, sm_86): the 3DGUT CUDA tracer JIT-builds and loads on the 3080 Ti. Poses must come from COLMAP (`apps/colmap_3dgut`): pyCuSFM's binaries need AVX2, which this CPU lacks. NuRec `nre-ga` needs ≥24 GB. Details in "Models and tools" below. Still needs capture data. |
| Gaussian-splat reconstruction + labelled splat render (W3) | Thor (sm_110) | **verified on a public scene (Mip-NeRF 360 garden); Isaac Sim ParticleField check pending on the workshop** | 3DGRUT `e6f4552` (pyproject 2.0.0; no v2.0.0 tag), **3DGUT** (Thor has no OptiX/RT cores). Conda env `~/thordrive/nvidia-sim/envs/3dgrut`: Python 3.11, `openusd 26.08` from conda-forge (PyPI `usd-core` has no aarch64 wheel), `torch 2.13.0+cu130` (arch list includes sm_110). `fused-ssim`, `ppisp` and the JIT `lib3dgut_cc` were built with `TORCH_CUDA_ARCH_LIST=11.0` and contain sm_110 SASS only. Builds use **`CUDA_HOME=/usr/local/cuda-13.2`**: on this host `cuda-13.0/.../include/cuda` symlinks into 13.2's CCCL, so 13.0 thrust fails a `static_assert`. Garden, `images_4`, 15k iterations (densify/LR schedule halved): 3,059 s at 4.90 it/s, 2,420,270 Gaussians, test **PSNR 26.78 / SSIM 0.836 / LPIPS 0.188**, peak CUDA 5.0 GB reserved. Export: `export_last_lightfield.usdz` 545 MB (`UsdVol.ParticleField3DGaussianSplat`, SH3, root normalizing xform, upAxis Y, OpenUSD validation passes) and `export_last.ply` 572 MB (raw COLMAP frame). Newton 1.6.1 `SensorTiledCamera` with the splat (`add_shape_gaussian`) and a mesh box: 1297×840 FAST 1.30–1.43 s/frame, QUALITY 3.6–5.7 s/frame. Box shape id matches the splat-vs-box depth order on 100% of box-footprint pixels in 2 views (vase correctly occludes the box). Gotchas: `Gaussian.create_from_ply` + open3d 0.20 silently drops scale/rotation/SH (use the numpy loader in `tools/thor/newton_render_splat.py`); never share one `Gaussian` between models (CUDA error 700). Runbook: `~/thordrive/nvidia-sim/src/3dgrut/THOR_RUNBOOK.md`. Assets copied to the workshop `$LB/data/assets/w3-garden/` (sha256 match). Dataset has no stated licence (Google Research benchmark, CVPR 2022): evaluation only. |
| Detector training (W5) | workshop: V100 (P100 also works) | **verified** | `workshop/train_detector.py`: YOLO26s, 3-epoch smoke on 1448 SDG train / 162 SDG val frames (`--smoke-sdg-val`; recall gates need real val, W2) on the P100 at ~2 it/s, batch 16. Needs `polars[rtcompat]` (no AVX2). |
| ONNX export (W5) | workshop | **verified** | opset 17, onnxslim; `onnx.checker` clean: `$LB/data/models/yolo26s_smoke/weights/best.onnx` |
| Hailo DFC compile to HEF (W5) | workshop (DFC is x86-only) | **wheel downloaded; not installed** | Wheel in `~/Downloads` on the workshop (2026-10-13). `scripts/hailo_compile.sh --tag smoke --dry-run` exits 0 with no WARN. Open risk: the workshop CPU (Xeon E5-4657L v2) has AVX but no AVX2/FMA, and the DFC pins tensorflow 2.18 and jaxlib 0.5.3 prebuilt wheels; an import test in `env-dfc` decides whether this host can compile. Target **`hailo8l`**; DFC 3.34 pairs with HailoRT 4.24 (installed on the Pi). |
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
  cache/huggingface/      HF_HOME for workshop scripts (MM-GDINO, GDINO base, OWLv2, SAM 2.1)
  cache/objaverse/        Objaverse metadata (*.json.gz)
  data/captures/          W1 teleop sessions, offloaded from the Pi
  data/datasets/          W2 reviewed real-image datasets, versioned dirs
  data/sdg/               W4 synthetic datasets, versioned dirs
  data/assets/objaverse/  curated glb/ + usd/ + manifest.json
  data/isaac-assets/6.0/  Isaac Sim 6.0.0 local asset pack (Isaac/, NVIDIA/); zips/MD5SUMS
  data/models/            W5 float training runs
  data/models/pretrained/ YOLO26 s/m, YOLO-World v2 x, CLIP ViT-B/32 (weights/clip/)
  data/hailo/             model-zoo HEFs (mz-v2.19.0/) and our compiled HEFs (compiled/)
  data/smoke/             W0 smoke artifacts (w0/, sdg-isaac/)
  envs/3dgrut/            conda env for 3DGRUT (cuda-toolkit 12.8.1, torch 2.8.0+cu128)
  tools/                  3dgrut/, pyCuSFM/ (git checkouts); bin/ = git-lfs, uv, gcc/g++ -> gcc-11
  cache/toolchain-setup/  download, verify and smoke scripts for the models below
  logs/                   long-job stdout
```

Docker storage migration completed 2026-10-07. `/etc/docker/daemon.json` sets
`data-root=/home/kp/nvme2/docker-data/docker`, and `docker info` reports the same path. The
Isaac Sim image and other Docker images now reside on the workshop NVMe rather than the ZFS
root pool. Migration restarted Docker; all four containers returned. `snitch`, Neo4j, and
`ai-memory` were healthy, and `edgelauncher` was Up. Coordinate future Docker restarts because
they interrupt these other users' containers. Post-migration bulk-pull performance is unknown.

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
# YOLO-World text encoder; pins keep torch/transformers untouched
$P freeze | grep -iE "^(torch|torchvision|transformers|numpy|pillow|tqdm|regex)==" > pins.txt
$P install -c pins.txt "git+https://github.com/ultralytics/CLIP.git"
# V100 only:
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2
```

Pins and why: torch cu128+ wheels drop sm_60/sm_70. transformers 5.18 returns
corrupted MM Grounding DINO boxes; 4.57.1 is correct. The base anaconda env on
the server has a broken `psutil` and is not used. The `$LB/env` conda env was
relocated from `~/lawnberry-workshop/env`: console-script shebangs still point
at the old prefix, so invoke everything as `$LB/env/bin/python -m <tool>` and
`$LB/env/bin/python script.py` (bare `$LB/env/bin/pip` etc. break).
The relocation also left OpenSSL's default CA file pointing at the old prefix
(`/home/kp/lawnberry-workshop/env/ssl/cert.pem`), so stdlib `urllib` HTTPS downloads from
`$LB/env` fail with `CERTIFICATE_VERIFY_FAILED` (CLIP's weight fetch hit this). `requests`
and `huggingface_hub` use certifi and work. Workaround:
`SSL_CERT_FILE=$($LB/env/bin/python -m certifi)`, or fetch with `curl`.

## Models and tools (downloaded 2026-10-06)

All on the workshop. GPU smoke tests ran on the 3080 Ti (`CUDA_DEVICE_ORDER=PCI_BUS_ID
CUDA_VISIBLE_DEVICES=1`, device name asserted), except SAM 3, which ran on the P100 (see its row). Scripts: `$LB/cache/toolchain-setup/`; logs:
`$LB/logs/{verify-hf,smoke-det,smoke-yolo,isaac-assets-6.0.0,isaac-assets-probe,build-3dgrut-conda,smoke-3dgut,build-pycusfm,smoke-cusfm}.log`.
Sanity images: Ultralytics `bus.jpg` (4 people + bus) and SDG `v1-wide-fold/images/000795.jpg`
(read-only).

| Item | Version / revision | License | Location | Size | Verification | Blockers |
|---|---|---|---|---|---|---|
| MM Grounding DINO large (`openmmlab-community/mm_grounding_dino_large_all`), primary W2 labeller | HF `0445d0a0ae5f` | Apache-2.0 | HF cache | 1.38 GB | Already present; the weights sit in the cache's hash-sharded store (`hub/blobs/52/…`), so the model dir looks ~1 MB. sha256 = Hub LFS oid. `bus.jpg`: bus + 4 people; frame 000795: person box within 2 px of the ground-truth label. | — |
| Grounding DINO base (`IDEA-Research/grounding-dino-base`) | HF `12bdfa3120f3` | Apache-2.0 | HF cache | 0.93 GB | Already present; sha256 OK. `bus.jpg`: bus + 4 people. | — |
| OWLv2 large ensemble (`google/owlv2-large-patch14-ensemble`) | HF `95e26936e865` | Apache-2.0 | HF cache (safetensors only, no `.bin`) | 1.75 GB | sha256 OK. Scores run far lower than DINO's. At 0.2 it finds only the bus. At 0.1, `bus.jpg` gives the bus + 4 people (+1 false person, 3 low "fence"), and on 000795 the person (box within 3 px of ground truth) plus rocks matching MM-GDINO's. | — |
| SAM 2.1 large (`facebook/sam2.1-hiera-large`) | HF `665f8e2ad61c` | Apache-2.0 | HF cache: `model.safetensors` (transformers) + `sam2.1_hiera_large.pt` (Meta `sam2` package) | 1.80 GB | Both sha256 OK. `transformers.Sam2Model` + box prompt from a DINO detection: mask returned, predicted IoU 0.99. | — |
| SAM 3.1 (`facebook/sam3.1`, `sam3.1_multiplex.pt`), chosen; SAM 3 (`facebook/sam3`, `sam3.pt` + `model.safetensors`) kept as fallback | HF `daa63191845a` (2026-03-27) / `3c879f39826c` (2025-11-20). Code: `facebookresearch/sam3` `0570b3a5` (2026-10-06) in `tools/sam3`. | Meta "SAM License" (2025-11-19): royalty-free use, modification and redistribution (licence copy must travel with it); no ITAR, military or other trade-controlled end uses. | `data/models/pretrained/sam3/{sam3.1,sam3}/` with `SHA256SUMS` and `REVISIONS.txt`. Env `envs/sam3` (uv venv, Python 3.12.3): torch 2.7.1+cu126, torchvision 0.22.1, triton 3.3.1, timm 1.0.30, numpy 1.26.4, `sam3` editable; freeze in `envs/sam3-freeze.txt`. | 3.50 GB / 6.89 GB; env 5.2 GB | All three sha256 = Hub LFS oids. Downloaded on Thor (gated access) and rsynced; no HF token on the server. **P100** (`CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0`, name asserted), `HF_HUB_OFFLINE=1`, fp32, `build_sam3_image_model` + `Sam3Processor`, threshold 0.5. `bus.jpg`: "person" 4 (0.94–0.97), "bus" 1. SDG `000000.jpg`: "snake" 1, box IoU 0.994 vs its `wildlife` label. 1.27 s/image (0.88 s encode + 0.20 s per prompt), 4.0 GiB peak. bf16 autocast (Meta's notebook recipe) also runs on sm_60, with boxes within 1 px, but takes 1.93 s/image and 8.3 GiB. SAM 3 gives the same boxes, with scores 0.002–0.010 higher. Script `cache/toolchain-setup/smoke_sam3.py`; logs `logs/smoke-sam3{.1-fp32,.1-bf16,.0-fp32}.log`. | SAM 3.1 has no transformers integration and no image builder. Its `detector.*` weights load into `build_sam3_image_model` with only the 4th neck level (`convs.3`) missing, which the image model discards (`scalp=1`); the script asserts this. Its detector weights differ from 3.0's (1093 of 1130 shared tensors), but Meta reports gains for video only. fp32 needs a patch, because the ViT MLP's fused `addmm_act` hard-casts to bf16; the script swaps in fp32 Linear + tanh-GELU. The package also needs `einops`, `pycocotools` and `psutil`, which it does not declare. Prompts must be concrete nouns: "wildlife" finds nothing, so `classes.py`'s `" . "` lists must be split into one prompt per noun. Triton kernels (EDT, connected components, NMS fallback) are not on the image text-prompt path; triton cannot target sm_60. The 3.0 `model.safetensors` needs transformers ≥ 5 in another env, which is not set up. |
| YOLO-World v2 x (`yolov8x-worldv2.pt`) | Ultralytics assets v8.4.0 | AGPL-3.0 | `data/models/pretrained/` | 146 MB | sha256 = GitHub release digest `41e771bf…`. `set_classes` + predict: `bus.jpg` bus + 4 people. | Needs the Ultralytics CLIP fork (installed into `$LB/env`: `clip 1.0`, `ftfy 6.3.1`, `wcwidth 0.9.2`; torch/transformers unchanged) and CLIP ViT-B/32 (`pretrained/weights/clip/ViT-B-32.pt`, 338 MB, sha256 = the hash in its URL). Ultralytics' `weights_dir` is the relative path `weights`, so run from `data/models/pretrained/`. |
| YOLO26 s / m | Ultralytics v8.4.0 | AGPL-3.0 | `data/models/pretrained/yolo26{s,m}.pt` | 20 / 44 MB | Extracted from the zoo's `yolo26{s,m}_pretrained.zip` (no new download); sha256 equals the v8.4.0 release digests (`646f8bc3…`, `401cea9a…`). Both detect bus + 4 people. | — |
| Isaac Sim 6.0.0 local asset pack | `isaac-sim-assets-complete-6.0.0.00{1..5}.zip` (2026-06-04) | No pack-level license file. Per-asset LICENSE files are under `Isaac/Robots/*` and in `Isaac/Environments/environment-supplement-LICENSE.txt`. The download is offered on the Isaac Sim docs page. | `data/isaac-assets/6.0/{Isaac,NVIDIA}` | 80.2 GB download, 98 GB extracted | All five MD5s match the docs. Extracted with the documented `cat … > one.zip; unzip`. Zips deleted after verification; `zips/MD5SUMS` kept (re-download ≈20 min at the ~60 MB/s seen). All 10 `/Isaac/` paths in `sdg_assets.generated.json` exist under `6.0/Isaac`. Isaac Sim 6.0.0 container probe on the 3080 Ti, run with `--/persistent/isaac/asset_root/default=/data/isaac-assets/6.0`: `get_assets_root_path()` returns `/data/isaac-assets/6.0`, and `/Isaac/People/Characters/{F_Business_02,M_Medical_01}` stat OK and open as USD stages. | **Covers only `/Isaac/` paths.** The SDG's other S3 sources are missing: `/ArchVis/*` 0/21 and `/Vegetation/*` 0/9 are absent. `/Characters/*` 1/2 sits under `6.0/NVIDIA/Assets/`, not where `resolve()` looks. 31 of 42 non-Objaverse paths therefore still need S3 (the unversioned `…/Assets/{ArchVis,Vegetation,Characters}` tree, not part of the Isaac pack). |
| 3DGRUT (3DGUT / 3DGRT, NVIDIA's Gaussian reconstruction used by the NuRec robotics workflow) | `nv-tlabs/3dgrut` `e6f4552b` (2026-09-22) | Apache-2.0 | `tools/3dgrut`; conda env `envs/3dgrut` (cuda-toolkit 12.8.1, torch 2.8.0+cu128, tiny-cuda-nn, kaolin, slangc); build script `cache/toolchain-setup/build_3dgrut_conda.sh` | 0.36 GB checkout + 14 GB env (+2.4 GB conda pkgs, 7.8 GB uv cache in `cache/`) | `train.py --help` lists the `cusfm_3dgut`/`colmap_3dgut` apps. The 3DGUT CUDA tracer (`lib3dgut_cc`) JIT-builds and loads on the 3080 Ti from `apps/cusfm_3dgut` (5 min cold, cached in `cache/torch-ext`). Not yet trained on data. The activation recipe is `cache/toolchain-setup/smoke_3dgut.sh`. The env persists `TORCH_EXTENSIONS_DIR`/`UV_CACHE_DIR`/`PIP_CACHE_DIR` on nvme2, so JIT builds never land in `~` on `rpool`. | Built for **sm_86 only** (`TORCH_CUDA_ARCH_LIST=8.6`): run with `CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1`. On the P100 tiny-cuda-nn refuses to load (sm_60), and torch cu128 drops sm_60. The Docker route was abandoned because it stalled `rpool` (see Storage). `tools/bin/gcc`/`g++` point at gcc-11, because Ubuntu has no plain `g++` and conda defaults has no `gcc_linux-64=13`. The repo's `create_conda.sh` dies under `set -u` in conda activate hooks; its env vars are set in the build script instead. Skipped: the playground's mesa GL headers. |
| pyCuSFM (cuSFM) | `nvidia-isaac/pyCuSFM` `0c97a670` (2026-02-09), `x86_cuda13` binaries | Repo LICENSE Apache-2.0; scripts are tagged `LicenseRef-NVIDIA-License`; third-party table in README | `tools/pyCuSFM` (`setup.bash cuda13`), image `pycusfm:cuda13` (base `nvcr.io/nvidia/tensorrt:25.09-py3`) | 1.6 GB checkout (LFS: 0.54 GB binaries, 0.16 GB models, 71 MB `r2b_galileo` sample) + 7 GB image | `git lfs fsck` OK. Image builds; inside it `nvidia-smi -L` shows the 3080 Ti and `cusfm_cli --help` runs. **The bundled sample crashes with SIGILL** in `keyframe_metadata_to_edex_main` (`ExtractStereoEdex`). | **Not runnable on this CPU.** The prebuilt, closed binaries use AVX2 (`objdump`: 62–78 AVX2-integer ops per binary, cuda12 and cuda13 builds alike). The Xeon E5-4657L v2 (Ivy Bridge) has AVX and F16C but no AVX2/FMA. They cannot be rebuilt, and the repo ships x86 binaries only, so no Thor build. Needs an AVX2 x86 host. SfM here stays COLMAP (CPU) → `apps/colmap_3dgut`. No host install either (`install_in_host.sh` uses `sudo apt`). `git-lfs` 3.8.0 installed user-space to `tools/bin` (sha256 = release digest). |
| NuRec NRE container (`nvcr.io/nvidia/nre/nre-ga`) | 26.04.01 | NVIDIA Software License + Omniverse product terms | **not pulled** (deliberate) | 14.3 GB compressed | — | The autonomous-vehicle product: NCore V3/V4 camera+lidar input, **≥24 GB VRAM** on Ampere or newer. The 3080 Ti (12 GB) is below the minimum; the V100 is Volta and shared. For the mower (cameras, no lidar) the robotics path is pyCuSFM → nvblox → 3DGRUT. `docker pull` reverses this. |
| Cosmos-Transfer2.5-2B (`nvidia/Cosmos-Transfer2.5-2B`) | HF `ce8440327c63`; code `nvidia-cosmos/cosmos-transfer2.5` `2ff49d0` (v1.5.0) | NVIDIA Open Model License (weights); Apache-2.0 (code) | **Thor:** repo `~/thordrive/nvidia-sim/src/cosmos-transfer2.5`, uv venv `~/thordrive/nvidia-sim/envs/cosmos-transfer2.5` (Python 3.13.11, torch 2.9.1+cu130, `uv sync --frozen --extra cu130`), runbook `THOR_RUNBOOK.md` in the repo. Workshop: not installed | ~40 GB cached in `~/thordrive/nvidia-sim/models/hf-hub`: seg + edge checkpoints 5.5 GB each, tokenizer 0.5 GB, Reason1-7B text encoder 16.6 GB, SigLIP2 4.3 GB, Guardrail1 3.5 GB + SigLIP 3.3 GB + Qwen3Guard 1.5 GB | **Thor: seg-controlled inference verified** (2026-10-07, repo sample `robot_input.mp4`; one 93-frame chunk at 720p (960×704); 35 steps; guardrails on and passed). Exit 0. **Wall 41:45**: denoising 33:46 at 57.9 s/step, ~5 min VAE decode. **Peak ~64 GB** system-wide (MemAvailable 95.4 → 31.3 GB with `brandy-llm` stopped); 49 GB GPU in `nvidia-smi`. Output `~/thordrive/nvidia-sim/outputs/ct25-seg-smoke-20261007-074905/robot_seg.mp4`. The GPU was 97% busy, but the DiT ran torch's flash SDPA at 46 TFLOPS: the repo puts cuDNN first only for cc 90/100. cuDNN measures 111 TFLOPS at the real shape, so ~46 of the 58 s/step was avoidable attention time. The checkout now adds `110` to that list; projected ~31 s/step, not yet re-run. Env checks (cuobjdump on all 27 CUDA `.so` + tiny GPU calls): torch has `sm_110`, and SDPA flash/efficient/cuDNN bf16 runs (the DiT's attention path on cc 11.0). natten 0.21.0 and xformers 0.0.33 have sm_110 SASS and run. transformer-engine 2.8 has no sm_110 SASS, but its compute_100 PTX JITs (RMSNorm, Linear, fused RoPE, DotProductAttention OK; 6.9 s first call, then cached). The cosmos flash-attn 2.7.4.post1 wheel has no sm_110 (`no kernel image`), so it was **rebuilt from source for sm_110** (`wheels/sm110/`, 62 min, CUDA 13.2: the 13.0 tree has a stray CCCL symlink that breaks thrust). The rebuild has sm_110 SASS; fwd, varlen, bwd, Triton rotary and the Reason1 text encoder's `flash_attention_2` import pass. `import cosmos_transfer2` and the inference CLI load. | Thor's shell sets `HF_HUB_DISABLE_IMPLICIT_TOKEN=1`, which makes HF requests anonymous, so gated repos look inaccessible. Unset it for downloads and runs (`giggahost` has access to all three gated repos). `HF_HUB_OFFLINE=1` breaks the run: Reason1/Guardrail1 resolve via `hf download --include '*'`, which needs the online tree listing. Inference needs ~64 GB of the 122 GB unified memory, so `brandy-llm.service` must be stopped first (≥70 GB free; `scripts/ct25-smoke-seg.sh` always restarts it). The workshop GPUs cannot run it (3080 Ti 12 GB; V100 32 GB, no bf16). Never `uv sync` the env: it reinstalls the non-sm_110 flash-attn. |
| COLMAP | 3.9.1 (apt) | BSD | `/usr/bin/colmap` | — | `colmap -h`: built **without CUDA** | CPU only |
| Hailo DFC | 3.34.0 (`hailo_dataflow_compiler-3.34.0-py3-none-linux_x86_64.whl`) | Hailo EULA | workshop `~/Downloads/` (not installed) | 524 MB | sha256 `f539ebb5997149ec68ca443a547196a03d28c624fbb072fdcd22a7d37fad9fb1`; zip test OK. 47 `.so` with no CPython tag; no `Requires-Python`; pins tensorflow 2.18.0, jax/jaxlib 0.5.3, scipy 1.12.0 (all have cp311 wheels). Hailo's catalogue labels the row Python 3.10. | Not yet imported; AVX2-less CPU (see W5 row) |

**SAM 3 (2026-10-07).** Access to both gated repos was granted to the Thor HF login. The weights were
downloaded there and rsynced, so the workshop needs no token. Run it as
`CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 $LB/envs/sam3/bin/python
$LB/cache/toolchain-setup/smoke_sam3.py cuda $LB/data/models/pretrained/sam3/sam3.1/sam3.1_multiplex.pt IMG=prompt[,prompt]`.
`$LB/env` is unchanged (transformers 4.57.1).

**Isaac Sim local assets.** The pack is mounted at `/data/isaac-assets/6.0`. The generator's
`--local-assets` option sets that root at `SimulationApp` startup; `resolve()` uses local
`Isaac/` and `NVIDIA/Assets/` files when present and falls back to NVIDIA S3 for missing
assets. `/data` and `/workspace` asset paths pass through unchanged. SDG v2 runs with
`--local-assets`; its manifest records distinct local/S3 asset-file counts. The pack covers
the `/Isaac/` assets in the generated configuration, while missing ArchVis/Vegetation assets
and the other uncovered paths continue to resolve from S3. The asset-config hash records
the JSON configuration, not the contents of each referenced file.

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
2. Hailo DFC 3.34.0: wheel downloaded to the workshop (2026-10-13). Install it into `$LB/env-dfc`
   and import-test it on the AVX2-less workshop CPU, then compile YOLO26s to a `hailo8l` HEF and
   benchmark it on the Pi.
3. Integrate SAM 3.1 into `workshop/autolabel.py`; the model and environment are verified on
   the P100 but not yet wired into the labelling workflow.

## Synthetic data (W4)

Run on the workshop from the repo mirror (`$LB/repo`), after `docker login nvcr.io`:

```bash
$LB/env/bin/python -m workshop.build_assets_json \
    --manifest $LB/data/assets/objaverse/manifest.json \
    --out workshop/sdg_assets.generated.json
scripts/isaacsim-headless.sh workshop/isaac_sdg.py \
    --assets workshop/sdg_assets.generated.json --out /data/sdg/v1 \
    --frames 500 --seed 1 --accum 2 --labels native --local-assets
ISAACSIM_GPU=2 scripts/isaacsim-headless.sh workshop/isaac_sdg.py \
    --assets workshop/sdg_assets.generated.json --out /data/sdg/v1-v100 \
    --frames 500 --seed 1 --accum 16 --labels idpass --local-assets
scripts/isaacsim-rdp.sh  # Isaac desktop in your RDP session
```

Output: `images/`, YOLO `labels/` (class index = `workshop/classes.py`), `rejected/`, and
`manifest.json` (seed, asset-config hash, per-class counts). `rejected_exposure` counts
unusable black/blown images; `rejected_frames` counts ID-pass failures and is meaningful only
when `--labels idpass|both` is used. Native mode is preferred on the 3080 Ti; the V100 needs
ID-pass labels because its annotator kernels are unsupported. `--local-assets` uses files in
the local Isaac pack when present; missing assets fall back to S3. The manifest records the
resolved asset root and distinct local/S3 file counts. Run `build_assets_json` after every
repo rsync: the generated config is untracked, so `rsync --delete` removes it.

ID pass on the 3080 Ti (measured, `workshop/idpass.py`): ~0.3% of pixels are 1-px colour blends
at silhouettes (largest island 65 px over 13 frames); the size-gated fold treats components
≤200 pixels as background. Brightness doubling was observed in 150/1751 frames. Although the
palette maps a doubled colour back to a unique source id, v1 visual QA found large cross-object
box mismatches that this blend fix does not prevent. Use native labels for 3080 Ti datasets;
keep ID-pass for the V100 fallback.

**Dataset v1 status (2026-10-07): complete — native-label RT 2.0 with `--local-assets`.**
`v1-blown` failed visual QA (37/40 sampled frames pure white). The `v1-wide-fold` attempt had
38/2000 ID-pass rejects (1.9%) but folded small level-2 blends into unrelated boxes.
`fold_doubled()` now folds only 8-connected components larger than 200 px; tests cover the
small-blend/large-surface distinction. The completed 2,000-frame ID-pass rerun rejected 30 ID
passes (1.5%) and 67 exposure frames (3.35%), leaving 1,903 image/label pairs. A 10-frame
overlay compared with native labels showed major box mismatches in at least two samples
(wildlife IoU 0.221; tree IoU 0.174); it is preserved as `$LB/data/sdg/v1-idpass-qa`, not
accepted for training.

**Native-label rerun (2026-10-07): complete.**
- Run: `$LB/data/sdg/v1` with `--seed 1 --frames 2000 --accum 2 --labels native --local-assets` on the 3080 Ti.
- Manifest: 2,000 attempted; 1,933 image/label pairs; 67 exposure rejects (3.35%); zero native-label rejects.
- Runtime: 4,774 s total (2.4 s/frame average). Manifest `asset_sources`: 11 local, 31 S3 asset files; the asset-config hash is `77b14549…`.
- Per-class counts in `manifest.json`; classes without assets have zero. All 46 converted Objaverse USDs exist and are referenced by the merged config.
- Exposure and ID-pass counters remain separate in the manifest. The v1-idpass-qa dataset is preserved at `$LB/data/sdg/v1-idpass-qa` for diagnostic comparison.

- At frame 205, a 12-sample 1 Hz check averaged 23% GPU utilization (0–61%); only Isaac used compute memory (2.4–2.8 GiB). The card was P2 at 1,200 MHz with SW Power Cap active and reported 418–427 W. Python used 376% CPU (container 491%); no competing GPU process. Host-side/render synchronization plus power capping are the likely utilization limits.

**Dataset v2 (2026-10-07): complete; exposure attrition noted.**
- Run: `$LB/data/sdg/v2` with `--seed 1 --frames 2000 --renderer PathTracing --spp 256 --labels native --local-assets` on the 3080 Ti.
- Manifest: 2,000 attempted; 1,933 image/label pairs; 67 exposure rejects (3.35%). Native mode was used, so the ID-pass rejection counter was not exercised.
- Runtime: 26,616 s total (13.3 s/frame average). Manifest `asset_sources`: 11 local, 31 S3 asset files; the asset-config hash is `77b14549…`.
- Per-class counts are in `manifest.json`; classes without assets have zero. All 46 converted Objaverse USDs exist and are referenced by the merged config. Hose and sprinkler labels were also visible in the procedural-class overlay spot-check.
- An 8-frame label overlay, including hose/sprinkler examples, showed native boxes tracking visible objects; no suspiciously oversized boxes appeared. Native-vs-ID-pass check on 27 frames: median corner error 0 px, p95 1.2 px; IoU ≥ 0.9 on 44/46 boxes with short side ≥ 30 px. Every large disagreement was an ID-pass error.

- The ID pass stays as `--labels idpass|both`.
- PT spp scan, RMS against PT@512:

  | spp | s/frame | RMS |
  |---|---|---|
  | 128 | 9.4 | 1.5 |
  | 256 | 14.5 | 1.3 |
  | 384 | 23.0 | 0.9 |
  | RT 2.0 | 2.3 | 3.4 |

- Isaac 6.0 gotchas:
  - Set the renderer at `SimulationApp` launch; a runtime switch was once silently ignored.
  - The first PT frame is black, so warm up with 1 spp.
  - Semantics must use `rep.functional.modify.semantics`; `UsdSemantics` labels go stale after frame 0.
  - Rebuild each frame under a new prim path.
- Known issues:
  - The Kit process can crash at shutdown after `SDG done`, once the manifest is written.
  - Some assets float, from the spawn code.

## Isaac Sim hosts (measured 2026-10-23)

The Thor runs Kit but produces no usable frame, so it cannot replace the 3080 Ti for SDG:

| Check | Result |
|---|---|
| `isaac-sim.compatibility_check.sh` | `System checking result: PASSED` (RAM 131.86 GB, storage 1213 GB, driver supported) |
| `SimulationApp({"headless": True})`, native 6.1.0 | app ready in ~22 s, `ISAAC_OK` |
| `rep.orchestrator.run()` + BasicWriter, full experience, native 6.1.0 | 1000 PNGs written, **every one fully black** (320x240, max pixel 0) — including a rerun with a Dome light and `rt_subframes=8` |
| Canonical `rep.create.camera(look_at=…)` scene, native 6.1.0 | **0 frames** in 35 s |
| `nvcr.io/nvidia/isaac-sim:6.0.0` arm64 container, same script | **0 frames** |
| Logs, both paths | `[omni.rtx] Failed to create an Optix adaptor for device 0`, `NGX isn't enabled`, `Failed to initialize OptixDenoiserContext` |

`~/isaacsim_assets` is empty, so an SDG run there would also need the Isaac asset library
(the workshop keeps it under `$LB/data/isaac-assets/6.0`).

The workshop box's RTX 3080 Ti remains the only host that has produced SDG output: `data/sdg/v1` =
1,933 labelled frames from a 2,000-frame run (`manifest.json`: Isaac 6.0.0, RealTimePathTracing,
seed 1, 1280x720). Its `nvcr.io/nvidia/isaac-sim:6.0.0` image has since been removed; the box keeps
an `nvcr.io` credential in `~/.docker/config.json`, so re-pulling the 17.6 GB image is one command.

## Detector data on hand (2026-10-23)

| Item | State |
|---|---|
| `$LB/data/sdg/v1` | 1,933 images + labels (synthetic; 24 classes, `workshop/classes.py`) |
| `$LB/data/models/yolo26s_smoke` | 3-epoch recipe run on 1,610 images; `--smoke-sdg-val`, so its metrics are not a recall gate |
| `$LB/data/hailo/compiled` | `yolo26s_smoke.hef` + sidecar (opt level 2; failed its on-device gate on 2026-10-13) |
| `$LB/data/captures`, `$LB/data/datasets` | **empty** — no real imagery has been collected |
| `workshop.autolabel` | deps present in `$LB/env` (`transformers==4.57.1`, imagehash, PIL); MM-Grounding-DINO weights are **not** cached; huggingface.co reachable from the box |
| `workshop.train_detector` | requires `--real` for a valid validation split; `--smoke-sdg-val` is documented as "metrics are NOT a recall gate" |
| `$LB/repo` | a 32 MB snapshot dated 2026-10-13, **not a git checkout** (no `.git`) |

## Workshop `$LB/env` re-verified on the P100 (2026-10-24)

`$LB/env` (conda, python 3.11.17) already had everything requested except `onnxsim`, now installed.
`$LB/cache/toolchain-setup/verify_aux_software.py` passes 12/12 (exit 0; log
`$LB/logs/aux-software-2-verify-final.log`) with `CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0
YOLO_AUTOINSTALL=False HF_HUB_OFFLINE=1`. GPU 0 is the Tesla P100-PCIE-16GB
(`GPU-2314d735-8adb-672e-23d8-7af00db59caa`, sm_60): 63 MiB idle before and after, and the script
asserts its own CUDA context sits on that UUID only (3080 Ti and V100 untouched).

| Package | Version | Verified by |
|---|---|---|
| torch | 2.7.1+cu126 | `get_arch_list()` = `sm_50 sm_60 sm_70 sm_75 sm_80 sm_86 sm_90`; `cuda:0` = P100 cc 6.0; fp32 + fp16 matmul, conv backward ok |
| torchvision | 0.22.1+cu126 | `nms` and `roi_align` on CUDA equal the CPU result |
| ultralytics | 8.4.173 | `YOLO("yolo26s.pt").predict(bus.jpg, device=0)` → bus + 4 person; `export(format="onnx", simplify=True, device=0)` (opset 18) → ONNX Runtime CPU gives the same class ids |
| transformers | 4.57.1 | tiny `BertModel` forward on cuda:0; `IDEA-Research/grounding-dino-base` from the offline HF cache → bus + 4 person on cuda:0 |
| imagehash | 4.3.2 | a/p/d/w/color hash: Hamming distance bus vs half-size blurred bus 0–1, bus vs zidane 32–38 |
| onnx | 1.23.2 | `check_model(full_check)`, `version_converter` 13→21, strict `infer_shapes`, `ReferenceEvaluator` (IR 14, max opset 28) |
| onnxsim | **0.8.1** (installed today) | `simplify()` of a torch-exported net (9→6 nodes, max abs error vs torch 4.8e-7) and of the YOLO26s ONNX (ORT class-id parity; 368→368 nodes, nothing left after onnxslim); `python -m onnxsim` rc 0 |
| onnxslim, onnxruntime | 0.1.97, 1.30.0 | already present; ultralytics' `simplify=True` uses **onnxslim**, not onnxsim; ORT is the CPU build |

Install: `$LB/env/bin/python -m pip install -c $LB/cache/toolchain-setup/pin-constraints.txt onnxsim==0.8.1`.
`--dry-run` listed exactly one distribution and `pip freeze` before/after differs by one line
(`onnxsim==0.8.1`), so torch, torchvision, numpy and transformers are untouched; `pip check` is clean.
onnxsim 0.4.36, 0.7.3 and 0.8.1 all passed a torch-2.7.1 export → simplify → ORT probe against onnx 1.23.2;
0.8.1 (2026-10-09) is the newest. The verified set is
`$LB/cache/toolchain-setup/pin-constraints-verified-20261024.txt` (use with `pip install -c`).
Ultralytics installs missing export dependencies into the active env unless `YOLO_AUTOINSTALL=False`.

| Host check | Result |
|---|---|
| `docker run --rm --gpus device=0 nvidia/cuda:12.2.0-runtime-ubuntu22.04 nvidia-smi` | exit 0; exactly one GPU row, Tesla P100-PCIE-16GB (`nvidia-smi -L` in the container: `GPU-2314d735-8adb-672e-23d8-7af00db59caa`). Docker 29.9.2, default runtime `nvidia`, NVIDIA Container Toolkit 1.20.1, driver 580.196.07 |
| NGC CLI | 4.36.6, x86-64, `~/bin/ngc` → `~/bin/ngc-cli/ngc`; already present (symlink mtime 00:39 on 2026-10-24, binary dated 2026-09-04), so not reinstalled. `find ngc-cli/ -type f -exec md5sum {} + \| LC_ALL=C sort \| md5sum -c ngc-cli.md5` → `-: OK` |

`~/bin` is on the login-shell PATH only; a non-login `ssh host cmd` needs `~/bin/ngc`. The CLI holds only a
guest token (no API key); the box's `~/.docker/config.json` has an `nvcr.io` credential.

## Isaac GR00T N1.7 on Thor (2026-10-24)

**Install verified. The official demo is blocked by a gated backbone; inference runs on a clearly labelled stand-in.**
Runbook `~/thordrive/nvidia-sim/GROOT_N17_THOR_RUNBOOK.md`; logs `~/thordrive/nvidia-sim/logs/groot-n17/`.

| Item | Result |
|---|---|
| Repo / install | `NVIDIA/Isaac-GR00T` main `944fe70` (2026-10-21, N1.7) in `~/thordrive/nvidia-sim/Isaac-GR00T-n17`. The repo's bare-metal Thor path `scripts/deployment/thor/install_deps.sh` (JetPack 7.2, Python 3.12.3), run through `scripts/groot-n17-install.sh` in an `env -i` shell: exit 0, `.venv` 8.8 GB. torch 2.13.0+cu132 (arch list has `sm_110`), torchvision 0.28.0, torchcodec 0.15.0, transformers 4.57.6, flash-attn 2.8.3 (repo LFS wheel; `flash_attn_func` vs SDPA max abs diff 0.00098 on Thor), TensorRT 10.16.1.11, ModelOpt 0.39.0. 16/16 imports OK; bf16 4096² matmul 140 TFLOPS. |
| Base checkpoint | `nvidia/GR00T-N1.7-3B` @ `2fc962b` (not gated; 27 files, 6.9 GB, 3.144 B params) in `~/thordrive/nvidia-sim/models/GR00T-N1.7-3B`. Sizes and all 6 LFS sha256 equal the Hub's; a `hf download` re-run fetched 27/27 files already up to date. |
| **Blocker (official path)** | The README zero-shot demo dies loading the VLM backbone: `RuntimeError: Cannot download the VLM backbone 'nvidia/Cosmos-Reason2-2B', which is a gated Hugging Face repo` (`403 Client Error ... Access to model nvidia/Cosmos-Reason2-2B is restricted and you are not in the authorized list`). Every GR00T N1.7 checkpoint names it; `gated=auto`, NVIDIA Open Model License. HF user `giggahost` (token `superagi`) has not accepted it. **One click by the account owner:** <https://huggingface.co/nvidia/Cosmos-Reason2-2B> → "Agree and access repository", then `~/thordrive/nvidia-sim/scripts/groot-n17-official.sh` (checks access first, exits 2 while gated). |
| Stand-in behind the numbers below | `models/GR00T-N1.7-3B-qwen3vl-standin`: symlinks to the checkpoint plus 2 edited JSON files (`model_name` → the local public `Qwen/Qwen3-VL-2B-Instruct`, Apache-2.0, sha256 verified; Cosmos-Reason2-2B is post-trained from it with the same architecture). No gated file was fetched. Loaded model: 494/494 backbone and 537/537 action-head tensors bit-identical to the GR00T shards, 0 missing/unexpected/mismatched keys. Valid for shapes and latency; action values are not guaranteed identical to the official path (the Cosmos chat template and tokenizer files are gated and were not compared). |
| Action output | `Gr00tPolicy.get_action`, batch 1, DROID embodiment: `eef_9d` (1,40,9) + `gripper_position` (1,40,1) + `joint_position` (1,40,7), float32, finite, i.e. a (1, 40, 17) chunk. Observation: 2 cameras × 2 frames `(1,2,180,320,3)` uint8, state (1,1,9/1/7), 1 instruction. |
| Latency, PyTorch eager | Idle GPU, MAXN, clocks at max (GPU 1574, CPU 2601, EMC 4266 MHz; Tj ≤ 56 °C). README demo `standalone_inference_script.py --traj-ids 1 2 --execution-horizon 8`: 48 timed steps avg **186.7 ms**, min 174.4, P90 196.3, max 253.0; Avg MSE 0.0202, MAE 0.0768. Per `get_action` call (n=50): median **176 ms**, p90 189 (5.7 Hz) = preprocess 7 + collate 16 + model 142 + decode 6. Official `build_trt_pipeline.py --steps benchmark --flat-e2e` eager median 167.4 ms. Server/client (`run_gr00t_server.py` + ZMQ client) median 169.5 ms, p90 174.8 (n=30). GPU peak 6.5 GB, process RSS 17 GB, model load 9-26 s. |
| Latency, TensorRT BF16 full pipeline | Export 1 m + build 7 m 41 s (7 engines, 6.1 GB), verify cosine 0.999958 PASS. `--flat-e2e` median **168.5 ms**, no faster than eager; component benchmark eager 161.3 → TRT 148.9 ms (backbone 77.2 → 84.0, action head 66.8 → 48.6). The README's Thor 76.8 ms is LIBERO with 2 images; DROID feeds 4, i.e. 1792 ViT patches, and the baseline ViT engine is FP32. |
| Latency, TensorRT optimized + mixed NVFP4 (README Thor recipe) | `scripts/groot-n17-nvfp4.sh` = `build_trt_pipeline.py --execution-profile optimized --quantization mixed_nvfp4 --calib-dataset-path demo_data/droid_sample --calib-size 10 --flat-e2e`: export 2 m 20 s + build 4 m 09 s (8 engines, 2.5 GB: ViT FP8, LLM and DiT NVFP4, action head as a CUDA graph), exit 0. Idle GPU, n=20: median **72.4 ms = 13.8 Hz** (min 69.1, max 84.6) against eager 174.7 ms in the same run, 2.4× faster. Verify flags "minor drift": final-action cosine 0.9952 (ViT embeds 0.977, backbone 0.998). README demo task-space error is unchanged: Avg MSE 0.0209 / MAE 0.0776 vs eager 0.0202 / 0.0768. The README's Thor 35.6 ms (28 Hz) is again the 2-image LIBERO case. |
| Noise sources | One early run, made while other agents' Cosmos jobs were starting and stopping on the Thor, measured 527 ms median; idle-GPU runs gave 161-176 ms. On Thor's unified memory a large `cudaMalloc` with a full page cache can hold the NVIDIA RM lock for minutes (CUDA processes and `nvidia-smi` sit in D state); `sudo sh -c 'sync; echo 1 > /proc/sys/vm/drop_caches'` clears it. |

Run everything with `~/thordrive/nvidia-sim/scripts/groot-n17-run.sh` (clean `env -i`, venv, `activate_thor.sh`) and plain `python`: `uv run` re-syncs against the x86 lock and destroys the Thor env.

## Cosmos on Thor (2026-10-24)

**Cosmos 3 (the current family) is installed and verified on the Thor: Reasoner (image and video to text), Generator (image to video) and edge-controlled Transfer each produced real output; the older Predict2.5 also ran. Two Hugging Face licence clicks remain (below).**
Runbook `~/thordrive/nvidia-sim/cosmos3/README.md`; scripts in `~/thordrive/nvidia-sim/cosmos3/`; outputs `~/thordrive/nvidia-sim/outputs/cosmos*`. One job at a time, idle GPU, MAXN.

| Item | Result |
|---|---|
| Which Cosmos | `github.com/NVIDIA/Cosmos` (HF collection `nvidia/cosmos3`) is the current family: Cosmos3-Edge 4B, Nano 16B, Super 64B (bf16 128 GB, does not fit Thor). The `nvidia-cosmos` org's Predict2.5, Transfer2.5 and Reason1/2 READMEs say "no longer under active development ... migrate to Cosmos 3". |
| Env A (HF Transformers/Diffusers) | uv venv `~/thordrive/nvidia-sim/envs/cosmos3` (Python 3.13.11, 5.8 GB): torch 2.14.1+cu130 (arch list has `sm_110`), transformers 5.21.0.dev0 (`50e35cc`), diffusers 0.42.0.dev0 (`aaf57d0`), torchcodec 0.17.0, cosmos_guardrail 0.3.2. `source ~/thordrive/nvidia-sim/cosmos3/env.sh`. No source build. |
| Env B (official `NVIDIA/cosmos-framework`) | `9214f43` (2026-10-23) in `~/thordrive/nvidia-sim/src/cosmos-framework`; venv `/home/kp/cosmos-work/envs/cosmos-framework` (8.8 GB) from `uv sync --frozen --all-extras --group cu130-torch213` with the repo-pinned uv 0.11.29 (system uv 0.11.28 is refused): torch 2.13.0+cu130 (`sm_110`), natten 0.21.6 sm_110 wheel; the repo's attention dispatcher has an arch-110 path (cuDNN, flash2, natten). `source ~/thordrive/nvidia-sim/cosmos3/env-framework.sh`. |
| Weights | `HF_HUB_CACHE=~/thordrive/nvidia-sim/models/hf-hub`: Cosmos3-Edge 9.2 GB; Cosmos3-Nano 35 GB (physically on `/home/kp/cosmos-work/hf-store`, symlinked in, because `~/thordrive` is 97% full); Cosmos-Predict2.5-2B `diffusers/base/post-trained` 21 GB; Wan2.2 VAE. |
| Reasoner, image (acceptance test) | **Edge, HF Transformers:** `outputs/cosmos3-edge-reasoner-20261024-005730/reasoner_text.txt`: the card's robot-flower planning prompt, 740 new tokens in 14.4 s (51 tok/s), load 10 s, GPU 4.7 GiB. A lawn photo (Mip-NeRF 360 garden) with an "autonomous mower vision system" prompt (`...-garden-20261024-010507`; Nano `cosmos3-nano-reasoner-20261024-010936/garden`): both name the round table as the obstacle and call the grass safe to mow. Video input (the generated clip, 4,006 prompt tokens) answered in 10.3 s. Nano reasoner: GPU 16.4 GiB, answers directly; Edge emits `<think>...</think>` first. Framework Edge reasoner: `cosmosfw-edge-reasoner-20261024-014240`, 78.5 ms/token. |
| Generator, image to video (Diffusers) | Edge: `outputs/cosmos3-edge-i2v-20261024-010125/edge_i2v.mp4`, 121 frames 832x480 @ 24 fps, 20 steps, **148.5 s** (4.2 s/step), GPU peak 12.4 GiB, safety checker on. Nano: `outputs/cosmos3-nano-i2v-20261024-012716/nano_i2v_480p.mp4`, same settings, **303 s** (12 s/step), load 32 s, GPU peak 35.3 GiB. Both clips show the prompted rockslide. The Nano card's default (720p, 189 frames, 35 steps) was not run. |
| Transfer (Cosmos3-Nano, framework) | `outputs/cosmosfw-nano-transfer-20261024-014456/transfer_edge_480p/vision.mp4`: Canny-edge control video to 121 frames 832x480 @ 30 fps, 20 steps, sampling 4 m 46 s, 6.5 min end to end; a control-vs-output contact sheet shows the output following the control edges. The older Cosmos-Transfer2.5-2B seg result (2026-10-07, row above) is unchanged: `envs/cosmos-transfer2.5` still imports clean. |
| Predict2.5-2B (older generation, Diffusers) | `outputs/cosmos-predict25-i2w-20261024-015307/predict25_garden_i2w.mp4`: image2world from a 16:9 crop of the garden photo with a robotic-mower prompt, 93 frames 832x480, 28 steps, **718 s** (24 s/step), load 24 s, GPU peak 25.7 GiB; a black and orange mower drives in from the right. |
| **Blockers (licence clicks by the owner of HF user `giggahost`)** | (1) `nvidia/Cosmos-1.0-Guardrail` (gated=auto, 403) is the default of `cosmos_guardrail`, Diffusers and the framework's `--guardrails`: accept at <https://huggingface.co/nvidia/Cosmos-1.0-Guardrail>. Workaround in use: `CosmosSafetyChecker(checkpoint_id="nvidia/Cosmos-Guardrail1")` in the Diffusers scripts (safety checker stays on); the framework runs use `--no-guardrails`. (2) `nvidia/Cosmos-Reason2-2B` and `-8B` (gated=auto, 403): accept at <https://huggingface.co/nvidia/Cosmos-Reason2-2B> and <https://huggingface.co/nvidia/Cosmos-Reason2-8B> (the same blocker as GR00T N1.7 above; the Cosmos 3 Reasoner needs neither). |
| Thor pitfalls | (a) transformers/diffusers `caching_allocator_warmup` makes one model-sized `cudaMalloc`; with a ~45 GB page cache it held the NVIDIA RM lock for ~4 min (`nvidia-smi` and the GR00T jobs sat in D state). `cosmos3/thor_patches.py` skips it; `sudo sh -c 'sync; echo 1 > /proc/sys/vm/drop_caches'` clears a live stall. (b) nltk refuses group-writable data dirs (umask 0002): `chmod -R go-w ~/.cache/huggingface/cosmos_guardrail`. (c) The login shell's `HF_HUB_DISABLE_IMPLICIT_TOKEN=1` turns gated repos into 403s; `env.sh` unsets it. (d) huggingface_hub 2.x shared blob store uses relative symlinks that break for a symlinked model dir: `HF_HUB_DISABLE_SHARED_BLOBS=1`. (e) Video input needs `torchcodec`. (f) `Cosmos2_5_PredictBasePipeline._execution_device` raises when a `CosmosSafetyChecker` is attached (guardrail sub-models have no `.device`): patched in `run_predict25.py`. |

## Hailo compile with DALI (workshop box, 2026-10-24)

DFC 3.34 only uses DALI for adaround when `nvidia.dali.plugin.tf` imports
(`hailo_model_optimization/algorithms/dali_utils/__init__.py` falls back to a TF dataset otherwise, logging
"DALI is not installed"). Measured on the P100 (adaround block 2): **15 -> 21 batches/s** with DALI, GPU use 22 -> 50 %.
The GPU is not the bottleneck without it: the V100 managed only 34 min/block vs the P100's 33.5.

Env: `/home/kp/repossata1/env-dfc-dali` (ext4 copy of `$LB/env-dfc`; `$LB/env-dfc` is left without DALI). Recipe:

| Step | Why |
|---|---|
| `cp -a $LB/env-dfc $D`, then rewrite `#!$LB/env-dfc/` shebangs in `$D/bin/*` to `#!$D/` | otherwise `hailomz` and `pip` silently run the original env |
| `$D/bin/python -m pip install --force-reinstall --no-deps "setuptools<81"` | the copied setuptools lacked `pkg_resources` |
| `$D/bin/python -m pip install nvidia-dali-cuda120==2.3.0` | DALI core |
| `PATH=~/bin/cxx:$PATH` (symlinks to `x86_64-linux-gnu-g++-13`/`gcc-13`), then `pip install --no-build-isolation --extra-index-url https://pypi.nvidia.com nvidia-dali-tf-plugin-cuda120==2.3.0` | the plugin builds against the env's TF 2.18; `/usr/bin/g++` is missing on this box |

Check: `$D/bin/python -c "from hailo_model_optimization.algorithms.dali_utils import dali_train_dataset as d; print(d.__module__)"`
must print `...dali_dataset`, not `...mock_dali_dataset`. Run: `DFC_ENV=$D scripts/hailo_compile.sh --tag T --gpu 2`.
