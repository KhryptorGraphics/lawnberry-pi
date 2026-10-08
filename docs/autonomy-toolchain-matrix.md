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
| Isaac Sim + synthetic data (W3/W4) | **workshop: RTX 3080 Ti (`ISAACSIM_GPU=1`)**; V100 fallback | **Verified on both; native labels preferred on 3080 Ti** | 3080 Ti: full RTX path; native Replicator `bounding_box_2d_tight` + `instance_id_segmentation` labels verified. Warm RT 2.0 is 0.25 s/frame after shader cache. Path-traced v2 at 256 spp completed 2,000 frames (14.5 s/frame benchmark). **Do not use the 3080 Ti ID pass for production labels:** visual comparison found cross-object box mismatches. V100 (`ISAACSIM_GPU=2`) needs `--/renderer/gpuEnumeration/rtxRequired=false`; no DLSS, average 16 frames, use the emissive ID pass. V100 smoke: 12/12 labelled at ~26 s/frame. PathTracing crashes on V100. |
| Isaac Sim desktop in RDP | workshop: RTX 3080 Ti | **verified** | `scripts/isaacsim-rdp.sh` (run on the workshop inside an RDP session) finds the caller's xrdp display (xorgxrdp or Xvnc backend; `:10` here), exports its X cookie for the container's uid 1234, and opens Isaac Sim Full 6.0.0 there with `--init` (no zombie containers). On the 3080 Ti: RTX Real-Time 2.0, asset browser connected; first start 379 s (cold shader cache, window blank until `app ready`). The viewport is ~3 FPS through the software Xvnc display and the x4 link. |
| SfM: COLMAP (W3) | workshop (CPU) | **installed** | `apt install colmap`: COLMAP 3.9.1, **built without CUDA** (`colmap -h`: "without CUDA"; no CUDA libs in `ldd`). Feature matching runs on CPU (96 cores). GPU SfM: pyCuSFM (below). |
| Gaussian-splat reconstruction (W3) | workshop: RTX 3080 Ti | **installed; tracer builds and loads; not yet trained on data** | 3DGRUT (`envs/3dgrut`, sm_86): the 3DGUT CUDA tracer JIT-builds and loads on the 3080 Ti. Poses must come from COLMAP (`apps/colmap_3dgut`): pyCuSFM's binaries need AVX2, which this CPU lacks. NuRec `nre-ga` needs ≥24 GB. Details in "Models and tools" below. Still needs capture data. |
| Gaussian-splat reconstruction + labelled splat render (W3) | Thor (sm_110) | **verified on a public scene (Mip-NeRF 360 garden); Isaac Sim ParticleField check pending on the workshop** | 3DGRUT `e6f4552` (pyproject 2.0.0; no v2.0.0 tag), **3DGUT** (Thor has no OptiX/RT cores). Conda env `~/thordrive/nvidia-sim/envs/3dgrut`: Python 3.11, `openusd 26.08` from conda-forge (PyPI `usd-core` has no aarch64 wheel), `torch 2.13.0+cu130` (arch list includes sm_110). `fused-ssim`, `ppisp` and the JIT `lib3dgut_cc` were built with `TORCH_CUDA_ARCH_LIST=11.0` and contain sm_110 SASS only. Builds use **`CUDA_HOME=/usr/local/cuda-13.2`**: on this host `cuda-13.0/.../include/cuda` symlinks into 13.2's CCCL, so 13.0 thrust fails a `static_assert`. Garden, `images_4`, 15k iterations (densify/LR schedule halved): 3,059 s at 4.90 it/s, 2,420,270 Gaussians, test **PSNR 26.78 / SSIM 0.836 / LPIPS 0.188**, peak CUDA 5.0 GB reserved. Export: `export_last_lightfield.usdz` 545 MB (`UsdVol.ParticleField3DGaussianSplat`, SH3, root normalizing xform, upAxis Y, OpenUSD validation passes) and `export_last.ply` 572 MB (raw COLMAP frame). Newton 1.6.1 `SensorTiledCamera` with the splat (`add_shape_gaussian`) and a mesh box: 1297×840 FAST 1.30–1.43 s/frame, QUALITY 3.6–5.7 s/frame. Box shape id matches the splat-vs-box depth order on 100% of box-footprint pixels in 2 views (vase correctly occludes the box). Gotchas: `Gaussian.create_from_ply` + open3d 0.20 silently drops scale/rotation/SH (use the numpy loader in `tools/thor/newton_render_splat.py`); never share one `Gaussian` between models (CUDA error 700). Runbook: `~/thordrive/nvidia-sim/src/3dgrut/THOR_RUNBOOK.md`. Assets copied to the workshop `$LB/data/assets/w3-garden/` (sha256 match). Dataset has no stated licence (Google Research benchmark, CVPR 2022): evaluation only. |
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
| Hailo DFC | 3.34.0 needed | Hailo EULA | **absent** | — | `pip show hailo_dataflow_compiler`: not found | Developer-zone login (open action 2) |

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
2. Hailo DFC 3.34.0: the only source is Hailo's developer zone (free login). It is not on PyPI,
   GitHub or Hugging Face; Hugging Face has only third-party HEFs for other tasks. Download the x86
   wheel to the workshop, then compile YOLO26s to a `hailo8l` HEF and benchmark it on the Pi.
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
