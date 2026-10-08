"""W3 check: render a 3DGRUT .ply splat + a mesh box with Newton SensorTiledCamera on Thor.

Runs in the Newton venv (newton 1.6.1 / warp 1.18 + open3d for create_from_ply), not the 3dgrut env.
Camera poses come from the Mip-NeRF 360 poses_bounds.npy (LLFF: same world frame as COLMAP, which is
the frame 3DGRUT writes the PLY in) and intrinsics from sparse/0/cameras.bin (PINHOLE).

Compositing check: model A = splat + box, B = box only, C = splat only. Where the box covers a pixel
(B), A must show the box id when the box is nearer than the splat (C depth) and the Gaussian id
otherwise.
"""

import argparse
import json
import struct
import time
from pathlib import Path

import newton
import numpy as np
import warp as wp
from newton.sensors import SensorTiledCamera
from PIL import Image

MISS = np.uint32(0xFFFFFFFF)


def read_pinhole(cameras_bin: Path) -> tuple[int, int, float, float, float, float]:
    with open(cameras_bin, "rb") as f:
        f.read(8)  # num cameras (garden has one shared camera)
        _, model_id, w, h = struct.unpack("<iiQQ", f.read(24))
        if model_id != 1:
            raise ValueError(f"expected COLMAP PINHOLE (model 1), got {model_id}")
        return (w, h, *struct.unpack("<4d", f.read(32)))


def gaussian_from_3dgs_ply(path: str, min_response: float) -> "newton.Gaussian":
    """Same conversion as newton.Gaussian.create_from_ply, but parses the binary PLY with numpy.

    Newton 1.6.1 reads via open3d. open3d 0.20 (the only cp313 aarch64 wheel) groups 3DGS fields
    into scale/rot/f_dc/f_rest, so create_from_ply silently falls back to unit scale, identity
    rotation and white SH (only positions and opacity load) -> huge, white Gaussians and minutes
    per frame.
    """
    with open(path, "rb") as f:
        if f.readline().strip() != b"ply" or b"binary_little_endian" not in f.readline():
            raise ValueError(f"{path}: expected binary_little_endian PLY")
        names, count = [], None
        while (line := f.readline().decode().strip()) != "end_header":
            tok = line.split()
            if tok[:2] == ["element", "vertex"]:
                count = int(tok[2])
            elif tok[0] == "property":
                if tok[1] != "float":
                    raise ValueError(f"{path}: unsupported property type {tok[1]}")
                names.append(tok[2])
        data = np.fromfile(f, dtype=[(n, "<f4") for n in names], count=count)

    def cols(*keys):
        return np.stack([data[k] for k in keys], axis=1).astype(np.float32)

    rest = sorted((n for n in names if n.startswith("f_rest_")), key=lambda n: int(n[7:]))
    rot = cols("rot_1", "rot_2", "rot_3", "rot_0")  # PLY wxyz -> Newton xyzw
    rot /= np.maximum(np.linalg.norm(rot, axis=1, keepdims=True), 1e-12)
    return newton.Gaussian(
        positions=cols("x", "y", "z"),
        rotations=rot,
        scales=np.exp(cols("scale_0", "scale_1", "scale_2")),
        opacities=1.0 / (1.0 + np.exp(-data["opacity"].astype(np.float32))),
        sh_coeffs=np.concatenate([cols("f_dc_0", "f_dc_1", "f_dc_2"), cols(*rest)], axis=1),
        min_response=min_response,
    )


def llff_c2w_opengl(poses_bounds: Path) -> tuple[np.ndarray, np.ndarray]:
    """LLFF columns are [down, right, back, t]; Newton cameras look along local -Z with +Y up."""
    p = np.load(poses_bounds)[:, :15].reshape(-1, 3, 5)
    rot = np.stack([p[:, :, 1], -p[:, :, 0], p[:, :, 2]], axis=2)
    return rot, p[:, :, 3]


def scene_center(rot: np.ndarray, pos: np.ndarray) -> np.ndarray:
    """Least-squares point nearest to all optical axes (the garden table)."""
    a, b = np.zeros((3, 3)), np.zeros(3)
    for r, c in zip(rot, pos, strict=True):
        d = -r[:, 2]
        m = np.eye(3) - np.outer(d, d)
        a += m
        b += m @ c
    return np.linalg.solve(a, b)


def build(gaussian, box_xform, box_half, with_splat: bool, with_box: bool):
    builder = newton.ModelBuilder()
    ids = {}
    if with_splat:
        ids["gaussian"] = builder.add_shape_gaussian(
            body=-1, gaussian=gaussian, label="garden_splat"
        )
    if with_box:
        ids["box"] = builder.add_shape_box(
            body=-1,
            xform=box_xform,
            hx=box_half,
            hy=box_half,
            hz=box_half,
            color=(0.9, 0.1, 0.1),
            label="box",
        )
    model = builder.finalize()
    sensor = SensorTiledCamera(model=model)
    sensor.utils.create_default_light(enable_shadows=False)
    return model, model.state(), sensor, ids


def render(sensor, state, cam_tf, rays, w, h, mode, n_timed=0):
    sensor.default_render_config.gaussians_mode = mode
    color = sensor.utils.create_color_image_output(w, h, 1)
    depth = sensor.utils.create_depth_image_output(w, h, 1)
    shape = sensor.utils.create_shape_index_image_output(w, h, 1)
    kw = dict(color_image=color, depth_image=depth, shape_index_image=shape)
    sensor.update(state, cam_tf, rays, **kw)  # warm-up (kernel load / BVH)
    wp.synchronize_device()
    ms = None
    if n_timed:
        t0 = time.perf_counter()
        for _ in range(n_timed):
            sensor.update(state, cam_tf, rays, **kw)
        wp.synchronize_device()
        ms = (time.perf_counter() - t0) * 1e3 / n_timed
    rgba = color.numpy()[0, 0].view(np.uint8).reshape(h, w, 4)
    return rgba, depth.numpy()[0, 0], shape.numpy()[0, 0], ms


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ply", required=True)
    ap.add_argument(
        "--scene",
        required=True,
        help="mipnerf360/garden dir (poses_bounds.npy, sparse/0, images_4)",
    )
    ap.add_argument("--out", required=True)
    ap.add_argument("--views", type=int, nargs="+", default=[0, 92])
    ap.add_argument("--downsample", type=int, default=4)
    ap.add_argument(
        "--min-response", type=float, default=0.0113, help="3DGUT particle_kernel_min_response"
    )
    ap.add_argument("--timed", type=int, default=20)
    ap.add_argument(
        "--modes",
        nargs="+",
        default=["FAST"],
        choices=["FAST", "QUALITY"],
        help="Gaussian render modes; the compositing check uses the first one",
    )
    args = ap.parse_args()

    scene, out = Path(args.scene), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    wp.init()

    cw, ch, fx, fy, cx, cy = read_pinhole(scene / "sparse/0/cameras.bin")
    images = sorted((scene / f"images_{args.downsample}").iterdir())
    w, h = Image.open(images[0]).size
    rot, pos = llff_c2w_opengl(scene / "poses_bounds.npy")
    if len(rot) != len(images):
        raise ValueError("poses_bounds / image count mismatch")

    center = scene_center(rot, pos)
    up = np.mean(rot[:, :, 1], axis=0)  # mean camera +Y (OpenGL up) ~ scene up
    up /= np.linalg.norm(up)
    radius = float(np.median(np.linalg.norm(pos - center, axis=1)))
    box_half = 0.06 * radius
    box_pos = center + up * (0.12 * radius)
    x = np.cross(up, [1.0, 0.0, 0.0])
    x /= np.linalg.norm(x)
    box_rot = np.stack([x, np.cross(up, x), up], axis=1)  # box z-axis = scene up
    box_xform = wp.transform(wp.vec3(*box_pos), wp.quat_from_matrix(wp.mat33(*box_rot.flatten())))

    t0 = time.perf_counter()
    gaussian = gaussian_from_3dgs_ply(args.ply, args.min_response)
    t_load = time.perf_counter() - t0
    # Self-check against Newton's own loader on the fields it does read (positions, opacity),
    # and record what it drops (scales) so the open3d incompatibility stays visible in the report.
    ref = newton.Gaussian.create_from_ply(args.ply, args.min_response)
    assert np.allclose(ref.positions, gaussian.positions) and np.allclose(
        ref.opacities, gaussian.opacities, atol=1e-6
    )
    create_from_ply_scales_all_one = bool(np.all(np.asarray(ref.scales) == 1.0))
    del ref
    t0 = time.perf_counter()
    # Newton 1.6.1: one Gaussian instance shared by two finalized models -> CUDA error 700 (illegal
    # address) when rendering the first model. Give each model its own instance.
    models = {
        "A_splat_box": build(gaussian, box_xform, box_half, True, True),
        "B_box": build(None, box_xform, box_half, False, True),
        "C_splat": build(
            gaussian_from_3dgs_ply(args.ply, args.min_response), box_xform, box_half, True, False
        ),
    }
    t_build = time.perf_counter() - t0

    sensor_a = models["A_splat_box"][2]
    rays = sensor_a.utils.compute_camera_rays_pinhole_opencv(
        w, h, fx, fy, cx, cy, image_width=float(cw), image_height=float(ch)
    )
    modes = SensorTiledCamera.GaussianRenderMode
    report = {
        "ply": args.ply,
        "num_gaussians": len(gaussian.positions),
        "create_from_ply_scales_all_one": create_from_ply_scales_all_one,
        "resolution": [w, h],
        "ply_load_s": round(t_load, 2),
        "build_3_models_s": round(t_build, 2),
        "shape_ids": models["A_splat_box"][3],
        "box_center": box_pos.tolist(),
        "box_half": box_half,
        "views": [],
    }
    ids = models["A_splat_box"][3]

    for v in args.views:
        q = wp.quat_from_matrix(wp.mat33(*rot[v].flatten()))
        cam_tf = wp.array([[wp.transformf(wp.vec3f(*pos[v]), q)]], dtype=wp.transformf)
        entry = {"view": v, "image": images[v].name, "ms_per_frame": {}}
        res = {}
        for mode in (modes[m] for m in args.modes):
            for name, (_, state, sensor, _) in models.items():
                n = args.timed if name == "A_splat_box" else 0
                res[(name, mode.name)] = render(sensor, state, cam_tf, rays, w, h, mode, n)
            entry["ms_per_frame"][mode.name] = round(res[("A_splat_box", mode.name)][3], 2)
            rgba = res[("A_splat_box", mode.name)][0]
            Image.fromarray(rgba[..., :3]).save(out / f"view{v:03d}_{mode.name.lower()}_color.png")

        # compositing check on the first requested mode
        m0 = args.modes[0]
        rgba_a, _, s_a, _ = res[("A_splat_box", m0)]
        _, d_b, s_b, _ = res[("B_box", m0)]
        _, d_c, s_c, _ = res[("C_splat", m0)]
        box_px = s_b == models["B_box"][3]["box"]  # shape ids are per model (box is 0 in B, 1 in A)
        box_front = box_px & ((s_c == MISS) | (d_b < d_c))
        expect = np.where(box_front, ids["box"], ids["gaussian"])
        entry.update(
            box_pixels_if_unoccluded=int(box_px.sum()),
            box_id_pixels_in_composite=int((s_a == ids["box"]).sum()),
            gaussian_id_pixels_in_composite=int((s_a == ids["gaussian"]).sum()),
            miss_pixels=int((s_a == MISS).sum()),
            box_region_id_agreement=round(float((s_a[box_px] == expect[box_px]).mean()), 4)
            if box_px.any()
            else None,
            box_id_outside_box_region=int(((s_a == ids["box"]) & ~box_px).sum()),
        )
        np.save(out / f"view{v:03d}_shape_index.npy", s_a)
        vis = np.zeros((h, w, 3), np.uint8)
        vis[s_a == ids["gaussian"]] = (40, 160, 60)
        vis[s_a == ids["box"]] = (230, 30, 30)
        Image.fromarray(vis).save(out / f"view{v:03d}_shape_index.png")
        gt = np.asarray(Image.open(images[v]).convert("RGB"))
        Image.fromarray(np.concatenate([gt, rgba_a[..., :3], vis], axis=1)).save(
            out / f"view{v:03d}_gt_render_ids.png"
        )
        report["views"].append(entry)

    (out / "newton_render_report.json").write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
