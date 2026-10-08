"""W4 synthetic data generator (Isaac Sim 6.0 container; RTX 3080 Ti, V100 fallback).

    scripts/isaacsim-headless.sh workshop/isaac_sdg.py \
        --assets workshop/sdg_assets.generated.json --out /data/sdg/<version> --frames 500 --seed 1
    # quality tier (3080 Ti, ~14.5 s/frame): --renderer PathTracing --spp 256 --local-assets
    # V100 (no native annotators): --labels idpass --accum 16

``--assets`` is ``workshop/sdg_assets.json`` or the Objaverse-merged config from
``workshop.build_assets_json``; ``/data`` is the data root mounted by the launcher.
Hose and sprinkler-head instances are also generated procedurally (``PROCEDURAL``).

Per frame:
1. Randomise: sun elevation/azimuth (time of day), sky intensity, ground tint,
   which classes appear, their count, distance, bearing, yaw and scale, and the
   mower camera's yaw and pitch.
2. Beauty pass: RT 2.0 RGB averaged over ``--accum`` frames (on the V100 there is
   no DLSS/NGX, so averaging is the denoiser), or ``--renderer PathTracing`` at ``--spp``.
3. Labels. ``native`` (default): Replicator ``bounding_box_2d_tight`` and
   ``instance_id_segmentation`` of the beauty render (``idpass.native_boxes``). ``idpass``
   (see ``workshop/idpass.py``): every labelled instance gets a flat emissive ID
   material; lights, GI, mesh lights, AA, auto-exposure and tonemapping are off.
   Both give exact visible (occlusion-aware) boxes.
4. Write ``images/NNNNNN.jpg``, YOLO ``labels/NNNNNN.txt`` (class index =
   ``workshop.classes``), and ``manifest.json`` with per-class instance counts.

Camera geometry approximates the mower's front camera (height ``--cam-height``,
looking ahead and slightly down). Replace with the measured mount when W3 lands.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
import time

ap = argparse.ArgumentParser()
ap.add_argument("--assets", required=True, help="JSON: class name -> list of USD paths/URLs")
ap.add_argument("--out", required=True)
ap.add_argument("--frames", type=int, default=100)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--width", type=int, default=1280)
ap.add_argument("--height", type=int, default=720)
ap.add_argument("--accum", type=int, default=16, help="beauty frames averaged per image")
ap.add_argument("--max-objects", type=int, default=8)
ap.add_argument("--cam-height", type=float, default=0.8)
ap.add_argument("--save-id", action="store_true", help="also write raw ID-pass images (debug)")
ap.add_argument(
    "--labels",
    choices=("native", "idpass", "both"),
    default="native",
    help="native: Replicator annotators (sm_75+, i.e. not the V100); idpass: emissive ID pass; "
    "both: native labels/ plus ID-pass labels_idpass/ for comparison",
)
ap.add_argument(
    "--renderer", choices=("RealTimePathTracing", "PathTracing"), default="RealTimePathTracing"
)
ap.add_argument("--spp", type=int, default=64, help="PathTracing samples per pixel")
ap.add_argument(
    "--local-assets",
    action="store_true",
    help="use the local Isaac asset pack under /data; files it lacks still come from S3",
)
args = ap.parse_args()

LOCAL_ASSETS = "/data/isaac-assets/6.0"  # $LB/data/isaac-assets/6.0; must not contain /Isaac/
# Isaac Sim 6.0.0's default (S3) asset root, the fallback for files the local pack lacks.
S3_ASSETS = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/6.0"
# PathTracing: Replicator renders totalSpp // spp subframes per captured step (measured ~20 ms
# overhead each), and RTX clamps spp per subframe to /rtx/pathtracing/clampSpp (default 64).
PT_SUBFRAME_SPP = min(args.spp, 64)
if args.renderer == "PathTracing" and args.spp % PT_SUBFRAME_SPP:
    ap.error(f"--spp must be a multiple of {PT_SUBFRAME_SPP}")

from isaacsim import SimulationApp  # noqa: E402

app = SimulationApp(
    {
        "headless": True,
        "width": args.width,
        "height": args.height,
        # Set at launch: a runtime /rtx/rendermode switch was once silently ignored (the setting
        # read PathTracing, the frames were RT 2.0).
        "renderer": args.renderer,
        "samples_per_pixel_per_frame": PT_SUBFRAME_SPP,
        # V100 has no RT cores: RTX skips "non-RTX" GPUs unless told otherwise.
        "extra_args": ["--/renderer/gpuEnumeration/rtxRequired=false"]
        + ([f"--/persistent/isaac/asset_root/default={LOCAL_ASSETS}"] if args.local_assets else []),
    }
)

import carb.settings  # noqa: E402
import numpy as np  # noqa: E402
import omni.replicator.core as rep  # noqa: E402
import omni.usd  # noqa: E402
import warp as wp  # noqa: E402
from isaacsim.storage.native import get_assets_root_path  # noqa: E402
from PIL import Image  # noqa: E402
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdShade  # noqa: E402

sys.path.insert(0, os.getcwd())
from workshop import idpass  # noqa: E402
from workshop.classes import NAMES  # noqa: E402

rng = random.Random(args.seed)
settings = carb.settings.get_settings()
stage = omni.usd.get_context().get_stage()
assets_root = get_assets_root_path()
asset_cfg: dict[str, list[str]] = json.load(open(args.assets))
unknown = sorted(set(asset_cfg) - set(NAMES))
if unknown:
    raise SystemExit(f"asset config has classes not in workshop.classes: {unknown}")
asset_cfg = {k: v for k, v in asset_cfg.items() if v}


remote_root = S3_ASSETS if args.local_assets else assets_root
ASSETS_BASE = remote_root.rsplit("/Isaac/", 1)[0]  # .../Assets
_asset_frame: dict[str, tuple[float, str]] = {}
_resolved: dict[str, str] = {}
asset_sources = {"local": 0, "s3": 0}  # distinct asset files used, by where they resolved


def resolve(path: str) -> str:
    """``/Isaac/...`` is relative to the Isaac assets root; other ``/...`` to Assets/.

    URLs, ``/workspace`` (repo mount) and ``/data`` (data-root mount) pass through. With
    ``--local-assets`` a file present in the local pack (``Isaac/``, or ``NVIDIA/Assets/`` for
    the rest) is used from there; anything it lacks (ArchVis, Vegetation) falls back to S3.
    """
    if "://" in path or path.startswith(("/workspace/", "/data/")):
        return path
    if path not in _resolved:
        isaac = path.startswith("/Isaac/")
        local = LOCAL_ASSETS + (path if isaac else "/NVIDIA/Assets" + path)
        if args.local_assets and os.path.exists(local):
            _resolved[path] = local
            asset_sources["local"] += 1
        else:
            _resolved[path] = remote_root + path if isaac else ASSETS_BASE + path
            asset_sources["s3"] += 1
    return _resolved[path]


def asset_frame(url: str) -> tuple[float, str]:
    """(metersPerUnit, upAxis) of an asset; ArchVis/Vegetation assets are cm and often Y-up."""
    if url not in _asset_frame:
        s = Usd.Stage.Open(url, load=Usd.Stage.LoadNone)
        _asset_frame[url] = (UsdGeom.GetStageMetersPerUnit(s), UsdGeom.GetStageUpAxis(s))
    return _asset_frame[url]


# ---- static scene ---------------------------------------------------------
UsdGeom.SetStageMetersPerUnit(stage, 1.0)
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
bbox = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
UsdGeom.Xform.Define(stage, "/World")
ground = UsdGeom.Mesh.Define(stage, "/World/Ground")
S = 60.0
ground.CreatePointsAttr([(-S, -S, 0), (S, -S, 0), (S, S, 0), (-S, S, 0)])
ground.CreateFaceVertexCountsAttr([4])
ground.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
sky = UsdLux.DomeLight.Define(stage, "/World/Sky")
sun = UsdLux.DistantLight.Define(stage, "/World/Sun")
sun.CreateAngleAttr(0.53)
sun_xf = UsdGeom.Xformable(sun).AddRotateXYZOp()

cam = UsdGeom.Camera.Define(stage, "/World/MowerCam")
cam.CreateFocalLengthAttr(18.0)  # wide FOV, roughly a 120-degree mower camera on 36 mm aperture
cam.CreateClippingRangeAttr((0.05, 500.0))
cam_xf = UsdGeom.Xformable(cam).AddTransformOp()
rp = rep.create.render_product(str(cam.GetPath()), (args.width, args.height))
rgb = rep.AnnotatorRegistry.get_annotator("rgb")
rgb.attach(rp)
if args.labels != "idpass":
    # Measured in Isaac Sim 6.0 with prims rebuilt every frame: instance_segmentation keeps
    # only frame 0's instances, so visible-pixel counts come from the per-mesh ids instead.
    tight = rep.AnnotatorRegistry.get_annotator("bounding_box_2d_tight")
    inst = rep.AnnotatorRegistry.get_annotator("instance_id_segmentation")
    tight.attach(rp)
    inst.attach(rp)


def pt_spp(per_subframe: int, total: int) -> None:
    settings.set_int("/rtx/pathtracing/spp", per_subframe)
    settings.set_int("/rtx/pathtracing/totalSpp", total)


if args.renderer == "PathTracing":
    pt_spp(1, 1)
    for _ in range(10):  # measured: without this the first PT frame comes out black
        rep.orchestrator.step(rt_subframes=1)

F3, C3, B = Sdf.ValueTypeNames.Float, Sdf.ValueTypeNames.Color3f, Sdf.ValueTypeNames.Bool


def omnipbr(mpath: str, inputs: dict) -> tuple[UsdShade.Material, UsdShade.Shader]:
    mat = UsdShade.Material.Define(stage, mpath)
    sh = UsdShade.Shader.Define(stage, mpath + "/Shader")
    sh.CreateImplementationSourceAttr(UsdShade.Tokens.sourceAsset)
    sh.SetSourceAsset("OmniPBR.mdl", "mdl")
    sh.SetSourceAssetSubIdentifier("OmniPBR", "mdl")
    for name, (vtype, value) in inputs.items():
        sh.CreateInput(name, vtype).Set(value)
    mat.CreateSurfaceOutput("mdl").ConnectToSource(sh.ConnectableAPI(), "out")
    return mat, sh


# Rough grass-coloured ground; colour randomised per frame.
ground_material, ground_shader = omnipbr(
    "/World/Looks/Ground",
    {
        "diffuse_color_constant": (C3, (0.2, 0.35, 0.1)),
        "reflection_roughness_constant": (F3, 0.95),
        "specular_level": (F3, 0.1),
    },
)
UsdShade.MaterialBindingAPI.Apply(ground.GetPrim()).Bind(ground_material)

# ID-pass materials are created once per palette id and reused across frames.
_id_materials: dict[int, UsdShade.Material] = {}


def id_material(iid: int) -> UsdShade.Material:
    if iid not in _id_materials:
        _id_materials[iid], _ = omnipbr(
            f"/World/IdLooks/id_{iid}",
            {
                "diffuse_color_constant": (C3, (0.0, 0.0, 0.0)),
                "specular_level": (F3, 0.0),
                "reflection_roughness_constant": (F3, 1.0),
                "metallic_constant": (F3, 0.0),
                "enable_emission": (B, True),
                "emissive_color": (C3, idpass.id_to_emissive(iid)),
                "emissive_intensity": (F3, idpass.EMISSIVE_INTENSITY),
            },
        )
    return _id_materials[iid]


ID_SETTINGS = {
    "/rtx/post/tonemap/op": 0,
    "/rtx/post/histogram/enabled": False,
    "/rtx/post/aa/op": 0,
    "/rtx/indirectDiffuse/enabled": False,
    "/rtx/reflections/enabled": False,
    "/rtx/translucency/enabled": False,
    "/rtx/ambientOcclusion/enabled": False,
    "/rtx/directLighting/enabled": False,
    "/rtx/sceneDb/ambientLightIntensity": 0.0,
    "/rtx-transient/meshlights/forceDisable": True,
    "/rtx/post/lensFlares/enabled": False,
    "/rtx/post/motionblur/enabled": False,
    "/rtx/post/dof/enabled": False,
}
if args.renderer == "PathTracing":
    ID_SETTINGS["/rtx/rendermode"] = "RealTimePathTracing"  # the ID pass is calibrated in RT 2.0
BEAUTY_SETTINGS = {k: settings.get(k) for k in ID_SETTINGS}


def apply(values: dict) -> None:
    for k, v in values.items():
        if v is not None:
            settings.set(k, v)


# ---- procedural classes ------------------------------------------------------
# Objaverse has only two hoses and two sprinklers; these build randomised USD
# geometry (metres, Z-up) under ``<obj>/asset`` instead of referencing a file.
def _torus_mesh(path: str, major: float, minor: float, nu: int = 48, nv: int = 12) -> UsdGeom.Mesh:
    """Torus lying in the XY plane, centred on the origin, as a quad mesh."""
    pts, idx = [], []
    for i in range(nu):
        u = 2 * math.pi * i / nu
        for j in range(nv):
            v = 2 * math.pi * j / nv
            ring = major + minor * math.cos(v)
            pts.append((ring * math.cos(u), ring * math.sin(u), minor * math.sin(v)))
            a, b = i * nv + j, ((i + 1) % nu) * nv + j
            idx += [a, b, ((i + 1) % nu) * nv + (j + 1) % nv, i * nv + (j + 1) % nv]
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr(pts)
    mesh.CreateFaceVertexCountsAttr([4] * (nu * nv))
    mesh.CreateFaceVertexIndicesAttr(idx)
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    return mesh


def _cylinder(path: str, radius: float, height: float) -> UsdGeom.Cylinder:
    cyl = UsdGeom.Cylinder.Define(stage, path)
    cyl.CreateRadiusAttr(radius)
    cyl.CreateHeightAttr(height)
    cyl.CreateAxisAttr(UsdGeom.Tokens.z)
    return cyl


def _bind_colour(prim_path: str, colour: tuple[float, float, float]) -> None:
    mat, _ = omnipbr(
        prim_path + "/Looks/Mat",
        {
            "diffuse_color_constant": (C3, colour),
            "reflection_roughness_constant": (F3, 0.6),
        },
    )
    UsdShade.MaterialBindingAPI.Apply(stage.GetPrimAtPath(prim_path)).Bind(mat)


def _spawn_hose(root: str, r: random.Random) -> None:
    """Loose garden-hose coil: 5-9 tori jittered in XY plus a tipped nozzle."""
    UsdGeom.Xform.Define(stage, root)
    for n in range(r.randint(5, 9)):
        minor = r.uniform(0.008, 0.011)
        coil = _torus_mesh(f"{root}/coil_{n}", r.uniform(0.06, 0.12), minor)
        xf = UsdGeom.Xformable(coil)
        xf.AddTranslateOp().Set(Gf.Vec3d(r.uniform(-0.05, 0.05), r.uniform(-0.05, 0.05), minor))
        xf.AddRotateXYZOp().Set(Gf.Vec3f(r.uniform(-10, 10), r.uniform(-10, 10), 0.0))
    nozzle = UsdGeom.Xformable(_cylinder(f"{root}/nozzle", 0.012, 0.08))
    nozzle.AddTranslateOp().Set(Gf.Vec3d(r.uniform(0.1, 0.16), 0.0, 0.012))
    nozzle.AddRotateXYZOp().Set(Gf.Vec3f(0.0, 90.0, r.uniform(-30, 30)))  # lying on its side
    _bind_colour(root, r.choice([(0.05, 0.3, 0.08), (0.6, 0.05, 0.05), (0.1, 0.1, 0.12)]))


def _spawn_sprinkler(root: str, r: random.Random) -> None:
    """Pop-up sprinkler head; riser height varies with how far it has popped up."""
    UsdGeom.Xform.Define(stage, root)
    h = r.uniform(0.05, 0.14)
    UsdGeom.Xformable(_cylinder(f"{root}/riser", 0.03, h)).AddTranslateOp().Set(
        Gf.Vec3d(0.0, 0.0, h / 2)
    )
    UsdGeom.Xformable(_cylinder(f"{root}/cap", 0.045, 0.02)).AddTranslateOp().Set(
        Gf.Vec3d(0.0, 0.0, h + 0.01)
    )
    _bind_colour(root, r.choice([(0.03, 0.12, 0.04), (0.02, 0.02, 0.02), (0.05, 0.08, 0.05)]))


PROCEDURAL = {"hose": _spawn_hose, "sprinkler_head": _spawn_sprinkler}


# ---- per-frame randomisation ----------------------------------------------
frame_assets: dict[str, str] = {}  # prim path -> asset file, for diagnostics


def randomise(frame_root: str) -> dict[str, int]:
    """Spawn instances under a new ``frame_root``; return prim path -> class index.

    The previous frame's ``/World/Frame_*`` is removed. Roots are unique per frame because the
    native annotators kept stale instances when labelled prims were re-created at the same path.
    """
    for old in stage.GetPrimAtPath("/World").GetChildren():
        if old.GetName().startswith("Frame_"):
            stage.RemovePrim(old.GetPath())
    UsdGeom.Xform.Define(stage, frame_root)
    elev, azim = rng.uniform(8, 75), rng.uniform(0, 360)
    sun_xf.Set(Gf.Vec3f(-elev, 0.0, azim))
    sun.CreateIntensityAttr(rng.uniform(1500, 5000) * math.sin(math.radians(elev)) + 200)
    sun.CreateColorAttr(Gf.Vec3f(1.0, 0.85 + 0.15 * elev / 75, 0.7 + 0.3 * elev / 75))
    sky.CreateIntensityAttr(rng.uniform(300, 1500))
    tint = (rng.uniform(0.12, 0.3), rng.uniform(0.25, 0.45), rng.uniform(0.05, 0.18))
    ground_shader.GetInput("diffuse_color_constant").Set(Gf.Vec3f(*tint))

    yaw = rng.uniform(0, 360)
    pitch = rng.uniform(5, 20)
    eye = Gf.Vec3d(0, 0, args.cam_height)
    fwd = Gf.Vec3d(
        math.cos(math.radians(yaw)), math.sin(math.radians(yaw)), -math.tan(math.radians(pitch))
    )
    cam_xf.Set(Gf.Matrix4d().SetLookAt(eye, eye + fwd, Gf.Vec3d(0, 0, 1)).GetInverse())

    spawned: dict[str, int] = {}
    frame_assets.clear()
    classes = sorted(set(asset_cfg) | set(PROCEDURAL))
    for k in range(rng.randint(1, args.max_objects)):
        name = rng.choice(classes)
        dist = rng.uniform(0.8, 15.0)
        bearing = math.radians(yaw + rng.uniform(-50, 50))
        path = f"{frame_root}/obj_{k:02d}"
        prim = UsdGeom.Xform.Define(stage, path).GetPrim()  # our transform, no asset ops
        # Classes with a generator use it for half their instances (all, without files).
        if name in PROCEDURAL and (name not in asset_cfg or rng.random() < 0.5):
            PROCEDURAL[name](path + "/asset", rng)
            frame_assets[path] = f"procedural:{name}"
            mpu, up = 1.0, UsdGeom.Tokens.z
        else:
            url = resolve(rng.choice(asset_cfg[name]))
            stage.DefinePrim(path + "/asset").GetReferences().AddReference(url)
            frame_assets[path] = url.rsplit("/", 1)[-1]
            mpu, up = asset_frame(url)
        xf = UsdGeom.Xformable(prim)
        move = xf.AddTranslateOp()
        move.Set(Gf.Vec3d(dist * math.cos(bearing), dist * math.sin(bearing), 0.0))
        xf.AddRotateZOp().Set(rng.uniform(0, 360))
        xf.AddScaleOp().Set(Gf.Vec3f(mpu * rng.uniform(0.85, 1.15)))
        if up == UsdGeom.Tokens.y:
            xf.AddRotateXOp().Set(90.0)
        # Rest the asset on the ground whatever its pivot.
        lo = bbox.ComputeWorldBound(prim).ComputeAlignedRange().GetMin()
        p = move.Get()
        move.Set(Gf.Vec3d(p[0], p[1], -lo[2]))
        bbox.Clear()
        spawned[path] = NAMES.index(name)
        if args.labels != "idpass":  # one semantic instance per spawned object
            # Not UsdSemantics.LabelsAPI directly: measured, boxes then come back empty for
            # every frame after the first. Replicator's own writer on a fresh path works.
            rep.functional.modify.semantics(prim, {"class": name}, mode="replace")
    return spawned


def beauty() -> np.ndarray:
    apply(BEAUTY_SETTINGS)
    pt = args.renderer == "PathTracing"
    if pt:  # every PT step path-traces totalSpp samples; warm-ups need only 1 spp
        pt_spp(1, 1)
    for _ in range(3):  # let referenced assets, textures and new MDL materials load
        rep.orchestrator.step(rt_subframes=1)
    if pt:  # one capture accumulating --spp samples (OptiX denoiser on, the RTX default)
        pt_spp(PT_SUBFRAME_SPP, args.spp)
        rep.orchestrator.step(rt_subframes=1)
        return np.asarray(rgb.get_data())[..., :3].copy()
    acc = None
    for _ in range(args.accum):
        rep.orchestrator.step(rt_subframes=1)
        f = np.asarray(rgb.get_data())[..., :3].astype(np.float32)
        acc = f if acc is None else acc + f
    return (acc / args.accum).clip(0, 255).astype(np.uint8)


def native_labels(spawned: dict[str, int]) -> list[tuple[int, int, int, int, int, int]]:
    """Visible boxes of the last rendered step, from Replicator's own annotators."""
    t, s = tight.get_data(), inst.get_data()
    return idpass.native_boxes(
        t["data"], t["info"]["primPaths"], np.asarray(s["data"]), s["info"]["idToLabels"], spawned
    )


def bind_id(prim: Usd.Prim, iid: int) -> None:
    """Override every material on ``prim``'s subtree with the ID material.

    Bound for both all-purpose and ``full`` purposes: RTX prefers ``full``, and
    assets that ship a ``material:binding:full`` would otherwise keep their own look.
    """
    api = UsdShade.MaterialBindingAPI.Apply(prim)
    for purpose in (UsdShade.Tokens.allPurpose, UsdShade.Tokens.full):
        api.Bind(id_material(iid), UsdShade.Tokens.strongerThanDescendants, purpose)


def id_pass(spawned: dict[str, int]) -> tuple[np.ndarray, dict[int, int]]:
    """Render the ID pass; return (raw RGB, assigned id -> class, -1 for ground).

    Authored directly on the stage: spawned prims are rebuilt every frame, so
    their ID bindings need no undo. Lights and the ground binding are restored.
    """
    assigned: dict[int, int] = {}
    for iid, (path, cls) in zip(idpass.OBJECT_IDS, spawned.items(), strict=False):
        bind_id(stage.GetPrimAtPath(path), iid)
        assigned[iid] = cls
    sky_i, sun_i = sky.GetIntensityAttr().Get(), sun.GetIntensityAttr().Get()
    sky.GetIntensityAttr().Set(0.0)
    sun.GetIntensityAttr().Set(0.0)
    bind_id(ground.GetPrim(), idpass.MAX_ID)
    assigned[idpass.MAX_ID] = -1  # ground: decoded, then dropped
    apply(ID_SETTINGS)
    for _ in range(6):  # flush real-time RTX temporal history from the beauty pass
        rep.orchestrator.step(rt_subframes=1)
    raw = np.asarray(rgb.get_data())[..., :3].copy()
    sky.GetIntensityAttr().Set(sky_i)
    sun.GetIntensityAttr().Set(sun_i)
    api = UsdShade.MaterialBindingAPI(ground.GetPrim())
    api.UnbindDirectBinding(UsdShade.Tokens.full)
    api.Bind(ground_material)
    return raw, assigned


# ---- main loop -------------------------------------------------------------
subdirs = ["images", "labels", "rejected"]
if args.save_id:
    subdirs.append("idpass")
if args.labels == "both":
    subdirs.append("labels_idpass")
for sub in subdirs:
    os.makedirs(os.path.join(args.out, sub), exist_ok=True)
counts = {n: 0 for n in NAMES}
rejected = rejected_exposure = 0
t0 = time.time()
for i in range(args.frames):
    stem = f"{i:06d}"
    tick = [time.perf_counter()]
    spawned = randomise(f"/World/Frame_{stem}")
    tick.append(time.perf_counter())
    img = beauty()
    tick.append(time.perf_counter())
    # ponytail: ~3% of frames put the camera inside/under an object and render black; they
    # are dropped here. Add a mesh-occupancy spawn test if that rate starts to matter.
    problem = idpass.exposure_problem(img)
    if problem:
        rejected_exposure += 1
        Image.fromarray(img).save(os.path.join(args.out, "rejected", stem + "_exposure.jpg"))
        print(f"SDG frame {i + 1}/{args.frames} REJECTED exposure={problem}", flush=True)
        # A shared GPU once blew out a whole 1.5 h run; stop early instead.
        if i + 1 >= 30 and rejected_exposure / (i + 1) > 0.10:
            raise SystemExit(f"aborting: {rejected_exposure}/{i + 1} frames rejected for exposure")
        continue
    # Native labels come from the beauty render's own annotators (read before the ID pass).
    native = native_labels(spawned) if args.labels != "idpass" else None
    id_labels = None
    if args.labels != "native":
        raw, assigned = id_pass(spawned)
        if args.save_id:
            Image.fromarray(raw).save(os.path.join(args.out, "idpass", stem + ".png"))
        try:
            id_labels = [b for b in idpass.boxes(idpass.decode(raw), assigned) if b[0] >= 0]
        except idpass.IdPassError as exc:
            # Never label a frame whose ID pass is not exact; keep it for diagnosis.
            rejected += 1
            Image.fromarray(raw).save(os.path.join(args.out, "rejected", stem + "_id.png"))
            Image.fromarray(img).save(os.path.join(args.out, "rejected", stem + ".jpg"), quality=90)
            print(
                f"SDG frame {i + 1}/{args.frames} REJECTED {exc} "
                f"assets={list(frame_assets.values())}",
                flush=True,
            )
            if native is None:
                continue
    tick.append(time.perf_counter())
    labels = id_labels if native is None else native
    Image.fromarray(img).save(os.path.join(args.out, "images", stem + ".jpg"), quality=95)
    label_sets = [("labels", labels)]
    if args.labels == "both" and id_labels is not None:
        label_sets.append(("labels_idpass", id_labels))
    for sub, boxes in label_sets:
        with open(os.path.join(args.out, sub, stem + ".txt"), "w") as fh:
            for cls, x0, y0, x1, y1, _px in boxes:
                fh.write(idpass.yolo_line(cls, x0, y0, x1, y1, args.width, args.height) + "\n")
    for cls, *_ in labels:
        counts[NAMES[cls]] += 1
    tick.append(time.perf_counter())
    spawn_s, beauty_s, id_s, post_s = (b - a for a, b in zip(tick, tick[1:], strict=False))
    print(
        f"SDG frame {i + 1}/{args.frames} objects={len(spawned)} labelled={len(labels)} "
        f"elapsed={time.time() - t0:.0f}s spawn={spawn_s:.2f} beauty={beauty_s:.2f} "
        f"id={id_s:.2f} post={post_s:.2f}",
        flush=True,
    )

json.dump(
    {
        "stage": "isaac_sdg",
        "isaac_sim": "6.0.0",
        "gpu": wp.get_cuda_device(0).name,
        "seed": args.seed,
        "frames": args.frames,
        "width": args.width,
        "height": args.height,
        "accum": args.accum,
        "renderer": args.renderer,
        "spp": args.spp if args.renderer == "PathTracing" else None,
        "labels": args.labels,
        "assets_root": assets_root,
        "asset_sources": asset_sources,
        "assets_sha256": hashlib.sha256(open(args.assets, "rb").read()).hexdigest(),
        "classes": NAMES,
        "instance_counts": counts,
        "rejected_frames": rejected,
        "rejected_exposure": rejected_exposure,
        "seconds": round(time.time() - t0, 1),
    },
    open(os.path.join(args.out, "manifest.json"), "w"),
    indent=2,
)
print("SDG done", json.dumps({k: v for k, v in counts.items() if v}), flush=True)
app.close()
