# Best-quality simulation and synthetic-data pipeline: decision and build plan

Status: decided 2026-10-07. Supersedes the "build Isaac Sim from source for Thor" idea.

## Decision

| Stage | Where | What | Why |
|---|---|---|---|
| W4 RGB rendering (quality tier) | Workshop **RTX 3080 Ti** | Isaac Sim RTX `PathTracing` with accumulated samples | Highest-fidelity renderer available on any machine we own |
| W4 RGB rendering (throughput tier) | Workshop RTX 3080 Ti | Isaac Sim RTX Real-Time 2.0 (DLSS), current `workshop/isaac_sdg.py` | 0.25 s/frame warm |
| W4 labels | Workshop RTX 3080 Ti | **Native Replicator annotators** (`bounding_box_2d_tight`, `instance_segmentation`) | Verified correct on the 3080 Ti; the emissive ID pass exists only for the V100 (sm_70) and stays behind a flag |
| W4 sim-to-real augmentation | **Thor** (122 GB unified memory) | Cosmos-Transfer2.5-2B, seg/depth/edge control from the Isaac renders | Needs 65.4 GB of GPU memory; Thor is the only machine with that much. Control inputs keep labels exact |
| W3 yard reconstruction | Workshop RTX 3080 Ti | COLMAP, then 3DGRUT (3DGUT), then USD `ParticleField3DGaussianSplat` | 3DGRUT is already built there; Kit/RTX 110.1 (Isaac Sim 6.0.1+) renders ParticleField splats natively |
| W6 RL / camera-in-the-loop | Thor (or the 3080 Ti) | **Newton** `SensorTiledCamera` (Warp CUDA ray caster) | Verified on Thor: RGB, depth, normals and exact per-shape ids without RTX |
| W5 detector training | Workshop V100 (P100 fallback); Thor if parked | Ultralytics YOLO26 | Unchanged |

## Why Isaac Sim RTX cannot run on Thor (verified 2026-10-07; no source build fixes it)

1. **The hardware has no RT cores.** An NVIDIA moderator on the developer forum wrote (2025-09-01): "There is no RT cores in AGX T5000"; the "Ray Tracing" line on the module data sheet is a known error. NVIDIA (2025-07-23): "there is no plan to support Optix currently" on Thor. Sources: https://forums.developer.nvidia.com/t/thor-rt-cores/343693 and https://forums.developer.nvidia.com/t/nvidia-optix-on-jetson-thor/339693.
2. **The driver (595.78, JetPack 7.2.1) exposes only a compute-emulated `VK_KHR_ray_tracing_pipeline`**, with no `VK_KHR_ray_query`, the same pattern as GTX 16-series cards. RTX's Real-Time 2.0, DLSS (NGX), OptiX denoising and the synthetic-data kernels all need what is missing.
3. **Isaac Sim's renderer is closed.** `isaac-sim/IsaacSim` is open source, but it fetches the Kit kernel, `librtx.*`, NGX and OptiX as prebuilt binaries. Rebuilding the open part changes nothing for an unsupported GPU.
4. **NVIDIA support statement:** Isaac Sim 6.0 and 6.1 list DGX Spark as the *only* supported aarch64 system. Staff answers on Thor threads: "Jetson-based platforms, including Thor AGX, are not supported" (https://forums.developer.nvidia.com/t/isaac-sim-report-error-on-thor-agx-t5000-error-rtx-denoising-plugin-failed-to-compile-compute-shader-rtx-nrd-packfornrd-cs-hlsl/358400, https://github.com/isaac-sim/IsaacSim/issues/269).
5. **Measured on this Thor** with `nvcr.io/nvidia/isaac-sim:6.0.0`, a lit scene of plane, cube, sphere, distant light and dome, at 640×480 over 20 steps:

   | `renderer` | RGB mean | Instance segmentation | 2D boxes | s/step |
   |---|---|---|---|---|
   | RealTimePathTracing | 0.0 | background only | 0 | 10.76 (first, cold) |
   | PathTracing | 0.0 | background only | 0 | 0.08 |
   | RaytracedLighting (maps to RT 2.0 in 6.0) | 0.0 | background only | 0 | 0.17 |
   | MinimalRendering, shading 2 | 0.0 | background only | 0 | 0.12 |
   | MinimalRendering, shading 1 | 0.0 | background only | 0 | 0.07 |

   Every mode is black, and the RTX annotators return nothing.

Corroborating facts (research agent, 2026-10-07):
- **No newer driver is coming soon.** The JetPack archive stops at 7.2.1. The apt repos `r39.3` and `r39.4` exist but are empty, and their Release files carry the same date as r39.2 (2026-09-14). No announcement mentions ray query or OptiX for Thor.
- **`libnvoptix` is absent system-wide;** `libnvidia-rtcore` is present.
- **The Kit kernel is a prebuilt `non-redist` package.** Isaac Sim `develop` (6.1.0-rc → v6.1.0, released 2026-09-10; v7.0.0a1 on 2026-09-18) pulls `kit-kernel 110.3.0` and `omni_physics 110.3.2` as binaries. `omni.hydra.rtx`, `omni.replicator.core`, `omni.syntheticdata` and `omni.mdl` also come from the extension registry; none of them is source.
- **NVIDIA staff (Richard3D, 2026-03-16):** "Omniverse will not run on a Jetson THOR. That is not RTX capable." Their answer for an IGX Thor plus RTX PRO 6000 dGPU on 6.0.0 was also "not supported".
- **ovrtx (standalone RTX library) requires an RTX-capable GPU**, so it is blocked the same way.
- **OpenUSD Storm (OpenGL) supports `primId`/`instanceId` AOVs** and could serve as an exact-ID fallback renderer. Catch: there is no aarch64 `usd-core` wheel (it must be built from source), and the 6.0 container ships `hdStorm.so` without `omni.hydra.pxr`.
- **Isaac Lab 3.0.0-EA (2026-09-16)** runs kit-less on Newton physics and the Newton Warp renderer (Python 3.12, PyTorch 2.11, Warp 1.16, Newton 1.5.2). aarch64 is listed as "help us test", not validated.

## The Thor-native renderer that does work: Newton (verified)

- `pip install newton` in a plain venv gives newton 1.6.1 and warp-lang 1.18.0; Warp sees `NVIDIA Thor arch 110`. A sample of the Warp test suite (array, codegen, mesh ray query, BVH, tape, grad) passes on Thor: 467 tests, 0 failures, 6 skipped.
- `SensorTiledCamera` renders 16 worlds × 640×480 in **3.4 ms per update (~4,700 frames/s)**. Outputs: RGB with shadows, depth, normals, albedo, and **exact per-pixel shape ids** (box 0, sphere 1, capsule 2, ground 48, sky 0xFFFFFFFF).
- It also renders Gaussian splats (`ModelBuilder.add_shape_gaussian`). Isaac Lab 3.0 uses it as the kit-less "Newton Warp" renderer; Isaac Lab's integration exposes only rgb and depth, while Newton itself exposes shape ids.
- Quality is simple direct shading, not path tracing. It is the right tool for high-throughput RL cameras, not for photoreal SDG, unless Cosmos-Transfer post-processes its output.

## What is worth building from source for sm_110 (and what is not)

| Component | Action | Notes |
|---|---|---|
| Isaac Sim / Kit / RTX | **Do not build.** | Closed renderer, and the hardware lacks RT cores; see above |
| Cosmos-Transfer2.5 dependency chain | Build or install cu130 wheels into a **fresh** env | Existing `cosmos` env is broken: editable installs point at a deleted `~/rosrepos/cosmos`; flash-attn 2.7.4 has no sm_110 kernels (`cudaErrorNoKernelImageForDevice`); transformer-engine is the empty meta-package. Chain: torch 2.9.1 cu130, flash-attn (2.7.4+cu130 from the cosmos-dependencies v1.2.0 index, or 2.8.4 from the Jetson AI Lab sbsa/cu130 index; the `gr00t` env's wheel already has sm_110), natten 0.21, transformer-engine 2.8 (from source, sm_110), xformers 0.0.33, decord, megatron-core ≥0.14. `TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas` |
| 3DGRUT CUDA extension | Optional on Thor | Already built on the 3080 Ti; the repo supports arm64 builds with `CUDA_VERSION=13.0.2` |
| COLMAP with CUDA | Optional | The workshop's COLMAP 3.9.1 is CPU-only; a CUDA build there (sm_86) speeds up W3 more than a Thor build would |
| pyCuSFM | **Blocked** | x86 binaries only, and they need AVX2, which the workshop Xeon lacks |
| onnxruntime-gpu | Not needed | Thor inference uses TensorRT 10.16 (verified) |

Jetson package sources for Thor: the pip index `https://pypi.jetson-ai-lab.io/sbsa/cu130` (cp312 torch 2.9–2.11, flash-attn 2.8.4, vllm 0.20, onnxruntime-gpu 1.24, natten, xformers) and `jetson-containers` recipes (transformer-engine 2.15, flash-attn 3, apex, 3dgrut 2.0.0, gsplat, warp, isaac-lab).

Cosmos facts (2026-10-07):
- Transfer2.5-2B needs 65.4 GB. On B200 it takes 92 s per 93-frame 720p seg-control clip; Thor is estimated at 5–10× slower than an H100.
- The repo has been maintenance-only since 2026-06, in favour of Cosmos 3. Cosmos 3 transfer covers only Nano (16B) and Super; neither is benchmarked on Thor.
- Licences: Transfer2.5 code Apache-2.0, weights under the NVIDIA Open Model License with a click-through gate. Cosmos 3 is OpenMDW-1.1 and not gated.

## Status (2026-10-07)

- **Cosmos-Transfer2.5 env on Thor: built and verified.**
  - Location: `~/thordrive/nvidia-sim/envs/cosmos-transfer2.5` (Python 3.13, torch 2.9.1+cu130).
  - flash-attn 2.7.4 was rebuilt from source for sm_110 (wheel in `~/thordrive/nvidia-sim/wheels/sm110/`). The other extensions run: transformer-engine 2.8 via JIT from compute_100 PTX; natten, xformers and torch with native sm_110.
  - **Never run `uv sync` in this env** — it reinstalls the broken flash-attn.
  - Runbook: `~/thordrive/nvidia-sim/src/cosmos-transfer2.5/THOR_RUNBOOK.md`.
  - **Blocked on the user:** accept the HF gates for `nvidia/Cosmos-Transfer2.5-2B`, `nvidia/Cosmos-Predict2.5-2B` and `nvidia/Cosmos-Guardrail1` as `giggahost`. The ungated dependencies (22 GB) are already downloaded.
  - Inference procedure: stop `brandy-llm.service`, confirm ≥70 GB free, run, then restart brandy-llm.
- **Thor machine quirks found while building:**
  - `/usr/local/cuda-13.0/targets/sbsa-linux/include/cuda` is an orphan symlink into the 13.2 headers, so builds against CUDA 13.0 fail; use 13.2.
  - Triton needs `TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas`, because its bundled ptxas rejects sm_110a.
- **SAM 3.1 (W2): ready on the workshop.**
  - Env `$LB/envs/sam3`; weights in `$LB/data/models/pretrained/sam3/{sam3.1,sam3}`; tested on the P100 at 1.27 s per image.
  - Prompts must be single concrete nouns ("snake" works, "wildlife" finds nothing).
- **Workshop Docker move to nvme2: staged.**
  - A first cold copy kept other users' containers down about 75 minutes and was aborted; Docker was restarted on the original storage.
  - A live `ionice -c3` pre-copy now runs into `/home/kp/nvme2/docker-data/{containerd,docker}`; log at `/var/log/claude-docker-precopy.log`.
  - The final switch — stop, delta `rsync --delete`, `daemon.json` `data-root`, containerd `root`, `RequiresMountsFor` drop-ins, start — waits until the SDG v2 production render finishes. Config backups are in `/root/claude-backups/docker-20261007004236`.
- **Hailo DFC:** not found anywhere on the workshop, including the 24 TB `/home/kp/repodisk`. It needs a Hailo developer-zone download.
- **SDG v2 (quality tier) is rendering on the 3080 Ti.**
  - Settings: PathTracing at 256 spp, native labels, local assets, 2000 frames at seed 1; 14.5 s/frame, about 8 h, started 03:36 −05:00.
  - Native labels were verified against the ID pass, and the ID pass was wrong in every large disagreement.
  - Details are in `docs/autonomy-toolchain-matrix.md` (Dataset v2).
- **Docker final switch is automated.** `/root/claude-docker-final-switch.sh` on the workshop (log `/var/log/claude-docker-final-switch.log`):
  1. It waits for `SDG done` with no Isaac container running.
  2. It stops Docker and containerd and runs a delta `rsync --delete` to nvme2.
  3. It switches the configs and adds the mount dependency.
  4. It verifies root, the image count (≥60) and the isaac-sim image, and **rolls back automatically** if any check fails.
  5. It pulls `nvcr.io/nvidia/isaac-sim:6.1.0`.

  The old dirs are kept as `/var/lib/{containerd,docker}.old-*` until deleted by hand.
- **Next, once the switch is done:** render `$LB/data/assets/w3-garden/export_last_lightfield.usdz` (ParticleField, Y-up with a normalizing root transform) in Isaac Sim 6.1 on the 3080 Ti, with an inserted hazard mesh and native labels.
- **W3 validated end to end on Thor with a public outdoor scene.**
  - Build: 3DGRUT `e6f4552`, 3DGUT method, built for sm_110 with `CUDA_HOME=/usr/local/cuda-13.2`. Env `~/thordrive/nvidia-sim/envs/3dgrut` (torch 2.13+cu130, OpenUSD 26.08 from conda-forge).
  - Training: Mip-NeRF 360 *garden*, images_4, 15k iterations. 53.6 min, 2.42 M Gaussians, test PSNR 26.78 / SSIM 0.836 / LPIPS 0.188, 5 GB GPU memory.
  - Exports:
    - `ParticleField3DGaussianSplat` USDZ (545 MB, passes OpenUSD validation; Y-up, with a normalizing root transform)
    - PLY (572 MB, raw COLMAP frame)
    - Both copied to the workshop at `$LB/data/assets/w3-garden/` for the Isaac Sim 6.x render check.
  - Newton render on Thor:
    - The splat plus an inserted mesh box composites with **100% correct per-pixel shape ids** against depth order.
    - Speed: FAST 38 ms at 162×105, 374 ms at 648×420, about 1.3 s at 1297×840. QUALITY is 3–4× slower and has no specks on semi-transparent regions.
    - Script: `workshop/newton_render_splat.py`. It parses the 3DGS PLY itself, because open3d 0.20 on aarch64 drops scales, rotations and SH.
  - This real-scene splat plus inserted hazard meshes path is the **highest-realism labelled-data path that runs on Thor**. The inserted meshes are simply shaded; Cosmos-Transfer can harmonise them.
  - The dataset has no stated licence (Google Research benchmark): evaluation only, do not redistribute.

## Build order

1. **3080 Ti SDG v2 (no gate).** In `workshop/isaac_sdg.py`:
   - add `--labels native|idpass` (default native) and `--renderer RealTimePathTracing|PathTracing` with `--spp`;
   - add an optional local Isaac asset root (`/data/isaac-assets/6.0`; the 31 uncovered asset paths stay on S3).

   Before the 2000-frame run, verify on 30 frames of seed 1 that native boxes match ID-pass boxes within a few px, and measure PathTracing s/frame.
2. **Gate A — workshop Docker data-root** onto nvme2. Restarting Docker kills other users' containers. Needed before pulling the Isaac Sim 6.1 image, which ParticleField splats require.
3. **Gate B — free about 70 GB on Thor** (llama-server holds 23 GB, about 63 GB is free), then build the Cosmos-Transfer2.5 env and run one seg-controlled clip on an SDG v2 sequence.
4. **W3:** COLMAP → 3DGUT on the 3080 Ti → ParticleField USD into Isaac Sim 6.1. **Blocked on footage**: the mower capture needs outdoor RTK. A phone-video walk of the yard can validate the path sooner.
5. **W6:** Newton-based training environment on Thor once a policy design exists.
