"""Render every printable and prove mesh topology plus negative-volume fit checks.

Requires OpenSCAD 2021.01+ and xvfb-run for PNG previews; Python standard library only.
Run from any directory: python hardware/mounts/build_printables.py
Outputs are published only after all mesh/clearance checks pass. No strength/IP claim.

Each published STL also gets a `3mf/<part>.3mf` repackaging of that same mesh, so a slicer
can be fed either file. Hand-made slicer projects left in this directory are only AUDITED
against the published meshes - never rewritten, because they also carry the operator's
print settings - and any mismatch is reported as a warning.
"""

from __future__ import annotations

import concurrent.futures
import hashlib
import json
import math
import re
import shutil
import struct
import subprocess
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
# Sleeve set size: the bar sizes the clamp takes. MUST match clamp_bar_od_mm in common.scad;
# an out-of-range index fails loudly there rather than silently building a wrong part.
CLAMP_INSERT_COUNT = 5
PARTS = [
    ("arm_layer_center", "hitch_arm_layer", {"arm_part": "center"}),
    ("arm_layer_left", "hitch_arm_layer", {"arm_part": "left"}),
    ("arm_layer_right", "hitch_arm_layer", {"arm_part": "right"}),
    ("arm_head_left", "hitch_arm_layer", {"arm_part": "head_left"}),
    ("arm_head_right", "hitch_arm_layer", {"arm_part": "head_right"}),
    ("arm_layer_gauge", "hitch_arm_layer", {"arm_part": "gauge"}),
    *[
        (
            f"arm_brace_bar_{side}_{index}",
            "arm_support",
            {"arm_brace_part": f"bar_{index}", "arm_brace_side": side},
        )
        for side in ("left", "right")
        for index in range(3)
    ],
    ("lapbar_clamp_anchor", "lapbar_four_bolt_clamp", {"clamp_part": "anchor"}),
    ("lapbar_clamp_cap", "lapbar_four_bolt_clamp", {"clamp_part": "cap"}),
    *[
        (
            f"lapbar_clamp_insert_{index}",
            "lapbar_four_bolt_clamp",
            {"clamp_part": "insert", "clamp_insert_index": index},
        )
        for index in range(CLAMP_INSERT_COUNT)
    ],
    *[
        (f"pushrod_{piece}", "pushrod", {"rod_part": piece})
        for piece in ("servo_end", "middle_a", "middle_b", "sleeve", "inner")
    ],
    *[
        (f"pushrod_gauge_{index}", "pushrod", {"rod_part": "gauge", "rod_gauge_index": index})
        for index in range(3)
    ],
    ("throttle_servo_mount", "throttle_servo_mount", {}),
    ("estop_bracket", "estop_bracket", {}),
    ("estop_contact_cover", "estop_contact_cover", {}),
    ("enclosure_body", "enclosure_body", {}),
    ("hitch_gauge", "enclosure_body", {"body_part": "hitch_gauge"}),
    ("camera_tower_base", "camera_tower", {"tower_part": "base"}),
    ("camera_tower_segment", "camera_tower", {"tower_part": "segment"}),
    ("camera_tower_extension", "camera_tower", {"tower_part": "extension"}),
    ("camera_tower_cap", "camera_tower", {"tower_part": "cap"}),
    ("camera_tower_backing", "camera_tower", {"tower_part": "backing"}),
    ("enclosure_lid", "enclosure_lid", {}),
    ("electronics_tray", "electronics_tray", {}),
    ("relay_board_sled", "enclosure_board_sled", {"sled_type": "relay"}),
    ("utility_board_sled", "enclosure_board_sled", {"sled_type": "utility"}),
    *[
        (f"camera_{dispatch}", "camera_mount", {"camera_part": dispatch})
        for dispatch in (
            "pi_housing",
            "pi_bottom",
            "pi_carrier",
            "pi_retainer",
            "pi_lens_insert",
            "pi_face_gasket",
            "pi_floor_gasket",
            "pi_insert_gasket",
            "pi_rim_gasket",
            "pi_lens_gauge",
            "pi_tile_fit_kit",
            "pi_groove_pad_kit",
            "stereo_housing",
            "stereo_bottom",
            "stereo_carrier",
            "stereo_retainer",
            "stereo_lens_insert_left",
            "stereo_lens_insert_right",
            "stereo_face_gasket",
            "stereo_floor_gasket",
            "stereo_insert_gasket_left",
            "stereo_insert_gasket_right",
            "stereo_rim_gasket_left",
            "stereo_rim_gasket_right",
            "stereo_lens_gauge",
            "stereo_tile_fit_kit",
            "stereo_groove_pad_kit",
            "foot",
        )
    ],
    *[
        (f"servo_adapter_{piece}", "servo_pushrod_adapter", {"adapter_part": piece})
        for piece in ("right", "left", "gauge")
    ],
    ("fit_gauge_1", "actuation_fit_gauge", {"gauge": 1}),
    ("fit_gauge_2", "actuation_fit_gauge", {"gauge": 2}),
    ("sensor_carrier", "sensor_carrier", {}),
    ("harness_saddle", "accessory_mounts", {"accessory_part": "harness_saddle"}),
    ("equipment_saddle", "accessory_mounts", {"accessory_part": "equipment_saddle"}),
    ("antenna_deck", "accessory_mounts", {"accessory_part": "antenna_deck"}),
]
# Coloured OpenCSG previews of assemblies.scad scenes for INSTALL.md.
# (name, assembly, full OpenSCAD camera, automatic fit).
# Large CSG clipping solids distort --viewall for the arm scenes; their default
# dimension illustrations use explicit cameras to keep the hardware legible.
VIEWS = [
    ("overview", "overview", "0,560,450,60,0,42,5400", False),
    ("steering_arms", "steering_arms", "0,560,330,62,0,42,3900", False),
    ("arm_layer", "arm_layer", "0,120,220,70,0,35,2200", False),
    ("lapbar_connector", "lapbar_connector", "-355.6,1075,485.23,65,0,35,700", False),
    ("rod_adjustment", "rod_adjustment", "0,0,0,70,0,35,0", True),
    ("estop_station", "estop_station", "0,0,0,60,0,-35,0", True),
    ("enclosure_mounted", "enclosure_mounted", "0,560,450,70,0,42,5400", False),
    ("enclosure_tongue", "enclosure_mounted", "0,450,450,60,0,150,5600", False),
    ("tower_base", "tower_base", "0,0,0,65,0,150,0", True),
    ("tower_full", "tower", "0,0,0,70,0,150,0", True),
    ("enclosure_cutaway", "enclosure", "0,0,0,60,0,30,0", True),
    ("estop_exploded", "estop", "0,0,0,60,0,-40,0", True),
    ("camera_stack", "camera_stack", "0,0,0,70,0,-35,0", True),
    ("camera_pi_exploded", "camera_pi_exploded", "0,0,0,65,0,-35,0", True),
    ("camera_stereo_exploded", "camera_stereo_exploded", "0,0,0,65,0,-35,0", True),
    ("sensor_tilt", "sensor", "0,0,0,60,0,-35,0", True),
    ("servo_adapter", "servo_adapter", "0,0,0,70,0,35,0", True),
]
CHECKS = [
    "body_lid",
    "body_tray",
    "sled_clearance",
    "pi_bores",
    "tray_fixings",
    "pi_keepout",
    "stack_columns",
    "stack_payloads",
    "lid_nut_access",
    "throttle_bores",
    "frame_bores",
    "frame_nut_access",
    "arm_layer_hitch_bore",
    "arm_layer_gauge_bores",
    "arm_servo_passages",
    "arm_servo_hardware_access",
    "arm_layer_aux_bores",
    "arm_tower_clearance",
    "arm_joint_bores",
    "arm_layer_root_clearance",
    "arm_plate_clearance",
    "arm_brace_stations",
    "arm_brace_pair_clearance",
    "arm_brace_insertion",
    "arm_brace_bar_fit",
    "arm_brace_bolt_bores",
    "clamp_four_bolts",
    "arm_layer_gland_paths",
    "arm_layer_box_fasteners",
    "arm_layer_tray_clearance",
    "clamp_tube_fit",
    "clamp_insert_clear",
    "clamp_pin_bore",
    "rod_lock_bores",
    "rod_splice_bores",
    "rod_end_bores",
    "rod_lock_bores_max",
    "rod_clamp_interface",
    "rod_clamp_sweep",
    "rod_servo_sweep",
    "rod_servo_sweep_band",
    "tongue_bore",
    "tongue_aux_bores",
    "tower_wall_bores",
    "tower_brace_bores",
    "tower_foot_bores",
    "tower_port",
    "tower_body_clearance",
    "tower_lid_clearance",
    "tower_joint",
    "tower_cable_path",
    "tower_cap_bores",
    "tower_cap_socket",
    "estop_cover",
    "estop_contacts",
    "estop_bore",
    "camera_bottom_clearance",
    "camera_bottom_service",
    "camera_carrier_fixings",
    "camera_carrier_travel",
    "camera_cassette_insertion",
    "camera_pcb_groove_capacity",
    "camera_bottom_fixings",
    "camera_flange_bores",
    "camera_flange_access",
    "camera_stack_joints",
    "camera_usb_housing_bends",
    "camera_usb_base_bend",
    "camera_fov",
    "camera_material_gauges",
    "sensor_pitch",
    "arm_servo_registration",
    "arm_head_joint_clearance",
    "arm_head_joint_seating_material",
    "servo_adapter_bolt_passages",
    "servo_adapter_hardware_access",
    "servo_adapter_sweep_clearance",
    "servo_adapter_eye_fit",
]

# Contact checks are inverse clearances and MUST succeed for every station
# and both root seats, not merely for any one joint in the assembly.
CONTACT_CHECKS = [
    *(("arm_brace_bar_contact", index) for index in range(3)),
    *(("arm_root_seat_contact", index) for index in range(2)),
    *(("arm_head_joint_seat_contact", index) for index in range(8)),
    *(("clamp_insert_grip", index) for index in range(CLAMP_INSERT_COUNT)),
    *(("camera_bottom_nut_land_contact", index) for index in range(8)),
    *(("camera_carrier_nut_land_contact", index) for index in range(4)),
    *(("camera_flange_land_contact", index) for index in range(16)),
    *(("camera_floor_gasket_contact", index) for index in range(4)),
    *(("camera_face_gasket_contact", index) for index in range(4)),
    *(("camera_insert_gasket_contact", index) for index in range(6)),
    *(("camera_rim_gasket_contact", index) for index in range(3)),
    *(("arm_servo_washer_land_contact", index) for index in range(12)),
    *(("servo_adapter_seat_contact", index) for index in range(2)),
    *(("servo_adapter_washer_land_contact", index) for index in range(12)),
]


# Per-command cap for every openscad invocation. 240 s was tuned when the check set was smaller;
# the heavier probes (arm_brace_insertion's Minkowski sweep, rod_servo_sweep_band) run concurrently
# on a 4-core Pi and their wall time inflates with contention, so a tight cap fails a good build.
COMMAND_TIMEOUT_S = 600


def run(command: list[str], *, empty: bool = False) -> str:
    result = subprocess.run(
        command, cwd=ROOT, capture_output=True, text=True, timeout=COMMAND_TIMEOUT_S, check=False
    )
    text = result.stdout + result.stderr
    if empty and result.returncode == 1 and "Current top level object is empty" in text:
        if "WARNING:" not in text and "ERROR:" not in text:
            return text
    if result.returncode or "WARNING:" in text or "ERROR:" in text:
        raise RuntimeError(f"Command failed: {command}\n{text}")
    return text


def mesh_report(path: Path) -> dict:
    data = path.read_bytes()
    if len(data) < 84:
        raise ValueError(f"Missing binary STL: {path}")
    count = struct.unpack_from("<I", data, 80)[0]
    if len(data) != 84 + count * 50 or count == 0:
        raise ValueError(f"Malformed or empty binary STL: {path}")
    vertices: dict[tuple, int] = {}
    edges: dict[tuple[int, int], list] = {}
    parent = list(range(count))
    volume = 0.0

    def root(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    for index in range(count):
        row = struct.unpack_from("<12fH", data, 84 + index * 50)
        points = [tuple(row[start : start + 3]) for start in (3, 6, 9)]
        if not all(math.isfinite(value) for point in points for value in point):
            raise ValueError(f"Nonfinite vertex: {path}")
        ids = [vertices.setdefault(point, len(vertices)) for point in points]
        a, b, c = points
        ab = [b[k] - a[k] for k in range(3)]
        ac = [c[k] - a[k] for k in range(3)]
        cross = (
            ab[1] * ac[2] - ab[2] * ac[1],
            ab[2] * ac[0] - ab[0] * ac[2],
            ab[0] * ac[1] - ab[1] * ac[0],
        )
        if sum(value * value for value in cross) < 1e-16:
            raise ValueError(f"Degenerate triangle: {path}, {index}")
        volume += (
            a[0] * (b[1] * c[2] - b[2] * c[1])
            + a[1] * (b[2] * c[0] - b[0] * c[2])
            + a[2] * (b[0] * c[1] - b[1] * c[0])
        ) / 6
        for u, v in zip(ids, ids[1:] + ids[:1], strict=True):
            edge = (min(u, v), max(u, v))
            state = edges.setdefault(edge, [0, 0, index])
            state[0] += 1
            state[1] += 1 if u < v else -1
            parent[root(index)] = root(state[2])
    components = len({root(index) for index in range(count)})
    if any(uses != 2 or balance != 0 for uses, balance, _ in edges.values()):
        raise ValueError(f"Nonmanifold edge or inconsistent winding: {path}")
    if components != 1 or volume <= 0:
        raise ValueError(f"Expected one positive connected solid: {path}, {components=}, {volume=}")
    low = [min(point[k] for point in vertices) for k in range(3)]
    high = [max(point[k] for point in vertices) for k in range(3)]
    size = [high[k] - low[k] for k in range(3)]
    if abs(low[2]) > 0.001 or max(size[:2]) > 420.001 or size[2] > 500.001:
        raise ValueError(f"Not bed-zero or exceeds 420x420x500 printer: {path}, {low=}, {size=}")
    return {
        "triangles": count,
        "connected_solids": components,
        "watertight": True,
        "consistent_winding": True,
        "bounds_mm": [low, high],
        "size_mm": [round(value, 3) for value in size],
        "volume_mm3": round(volume, 3),
        # Raw STL bytes can vary between equivalent exports. This canonical
        # digest ignores triangle/vertex ordering, not changes in triangulation.
        "stl_sha256": hashlib.sha256(data).hexdigest(),
        "mesh_sha256": canonical_mesh_digest(path),
    }


def read_stl_triangles(path: Path) -> list[tuple]:
    """The binary STL's triangles as corner triples, in file order."""
    data = path.read_bytes()
    if len(data) < 84:
        raise ValueError(f"Missing binary STL: {path}")
    count = struct.unpack_from("<I", data, 80)[0]
    if len(data) != 84 + count * 50 or count == 0:
        raise ValueError(f"Malformed or empty binary STL: {path}")
    return [
        tuple(
            tuple(round(value, 5) for value in struct.unpack_from("<12fH", data, 84 + index * 50)[
                start : start + 3
            ])
            for start in (3, 6, 9)
        )
        for index in range(count)
    ]


def canonical_mesh_digest(path: Path) -> str:
    """Digest rounded triangles independent of vertex/triangle ordering.

    Re-tessellating the same surface can change this hash; it is not a
    geometric-equivalence proof. A raw STL hash can also vary for identical
    triangles because the export's byte stream is not reliably stable.
    """
    faces = sorted(
        tuple(sorted(face)) for face in read_stl_triangles(path)
    )
    payload = ";".join(
        ",".join(f"{x:.4f},{y:.4f},{z:.4f}" for x, y, z in face) for face in faces
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def write_3mf(source: Path, target: Path) -> None:
    """Repackage a published STL as a 3MF so a slicer opens exactly the shipped mesh.

    3MF addresses the positive octant, so the mesh is shifted by its own negative minimum in
    X and Y; the parts are already bed-zero in Z. Triangle winding and vertex order are kept,
    which makes the output reproducible.
    """
    triangles = read_stl_triangles(source)
    low = [min(corner[axis] for face in triangles for corner in face) for axis in range(3)]
    shift = [-value if value < 0 else 0.0 for value in low]
    index: dict[tuple[float, float, float], int] = {}
    points: list[tuple[float, float, float]] = []
    faces: list[tuple[int, int, int]] = []
    for face in triangles:
        corners = []
        for axis in range(3):
            corner = tuple(round(face[axis][i] + shift[i], 5) for i in range(3))
            if corner not in index:
                index[corner] = len(points)
                points.append(corner)
            corners.append(index[corner])
        faces.append(tuple(corners))
    name = re.sub(r"[^A-Za-z0-9_.-]", "_", source.stem)
    vertices = "".join(f'<vertex x="{x}" y="{y}" z="{z}"/>' for x, y, z in points)
    mesh_faces = "".join(f'<triangle v1="{a}" v2="{b}" v3="{c}"/>' for a, b, c in faces)
    target.parent.mkdir(parents=True, exist_ok=True)

    def member(filename: str) -> zipfile.ZipInfo:
        # A fixed timestamp keeps the package byte-identical across runs, unlike the STL
        # export, so validation.json's hash of it is meaningful.
        info = zipfile.ZipInfo(filename, date_time=(1980, 1, 1, 0, 0, 0))
        info.compress_type = zipfile.ZIP_DEFLATED
        return info

    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as package:
        package.writestr(
            member("[Content_Types].xml"),
            '<?xml version="1.0" encoding="UTF-8"?>'
            '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
            '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package'
            '.relationships+xml"/>'
            '<Default Extension="model" ContentType="application/vnd.ms-package'
            '.3dmanufacturing-3dmodel+xml"/>'
            "</Types>",
        )
        package.writestr(
            member("_rels/.rels"),
            '<?xml version="1.0" encoding="UTF-8"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Target="/3D/3dmodel.model" Id="rel0" Type="http://schemas'
            '.microsoft.com/3dmanufacturing/2013/01/3dmodel"/>'
            "</Relationships>",
        )
        package.writestr(
            member("3D/3dmodel.model"),
            '<?xml version="1.0" encoding="UTF-8"?>'
            '<model unit="millimeter" xml:lang="en-US" xmlns="http://schemas.microsoft.com'
            '/3dmanufacturing/core/2015/02">'
            f'<metadata name="Title">{name}</metadata>'
            '<resources>'
            f'<object id="1" type="model" name="{name}"><mesh><vertices>{vertices}</vertices>'
            f"<triangles>{mesh_faces}</triangles></mesh></object>"
            "</resources>"
            '<build><item objectid="1"/></build>'
            "</model>",
        )


def mesh_extent_3mf(path: Path) -> list[float] | None:
    """The per-axis extent of a 3MF's mesh, or None when it holds no vertices."""
    points = []
    with zipfile.ZipFile(path) as package:
        for member in package.namelist():
            if not member.lower().endswith(".model"):
                continue
            text = package.read(member).decode("utf8", "ignore")
            points.extend(
                (float(match.group(1)), float(match.group(2)), float(match.group(3)))
                for match in re.finditer(
                    r'<vertex\s+x="([-\d.eE+]+)"\s+y="([-\d.eE+]+)"\s+z="([-\d.eE+]+)"', text
                )
            )
    if not points:
        return None
    return [
        round(max(point[axis] for point in points) - min(point[axis] for point in points), 3)
        for axis in range(3)
    ]


def audit_print_projects(published: dict[str, list[float]]) -> list[dict]:
    """Compare slicer projects in this directory with the published meshes, read-only."""
    audit = []
    for project in sorted(ROOT.glob("*.3mf")):
        extent = mesh_extent_3mf(project)
        current = published.get(project.stem)
        if extent is None:
            status, detail = "unreadable", "no mesh vertices found"
        elif current is None:
            status, detail = "no_source", "no part of this name is generated by this script"
        elif [round(value, 3) for value in current] == extent:
            status, detail = "current", "matches the published STL"
        else:
            status, detail = "stale", f"project {extent} vs published {current} mm"
        audit.append({"file": project.name, "status": status, "detail": detail})
    return audit


def defines(parameters: dict) -> list[str]:
    return [
        arg for name, value in parameters.items() for arg in ("-D", f"{name}={json.dumps(value)}")
    ]


def annotate_svg(path: Path, title: str, notes: list[str], watermark: str) -> None:
    """Add a text overlay to an OpenSCAD 2D projection.

    OpenSCAD emits geometry only. A drawing that carries no dimension values, no part labels
    and no qualification statement is a silhouette, not an assembly drawing, so this appends a
    notes band below the projected geometry. The band extends the viewBox rather than
    overlapping the section, and the transform is deterministic: the same input gives the same
    bytes.
    """
    text = path.read_text(encoding="utf-8")
    match = re.search(r'viewBox="([-\d.]+) ([-\d.]+) ([-\d.]+) ([-\d.]+)"', text)
    if not match:
        raise ValueError(f"SVG without a viewBox: {path}")
    vx, vy, vw, vh = (float(value) for value in match.groups())

    band = 4.5 * (len(notes) + 3)
    new_h = vh + band
    text = text.replace(match.group(0), f'viewBox="{vx} {vy} {vw} {new_h}"')
    text = re.sub(r'height="[\d.]+mm"', f'height="{new_h}mm"', text, count=1)

    # White sheet FIRST (behind the geometry). Appending it with the text put it on top and
    # hid the section entirely.
    sheet = (
        f'\n<rect x="{vx:.2f}" y="{vy:.2f}" width="{vw:.2f}" height="{new_h:.2f}" '
        'fill="white"/>'
    )
    text = re.sub(r'(<svg\b[^>]*>)', r'\1' + sheet.replace("\\", "\\\\"), text, count=1)

    base = vy + vh
    rows = [
        (base + 5.0, title, 2.6, "bold"),
        (base + 9.5, watermark, 2.1, "bold"),
    ]
    for index, note in enumerate(notes):
        rows.append((base + 13.5 + index * 4.5, note, 1.9, "normal"))

    parts = ['<g font-family="Helvetica, Arial, sans-serif" fill="black">']
    for y, label, size, weight in rows:
        safe = (
            label.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        )
        parts.append(
            f'<text x="{vx + 2:.2f}" y="{y:.2f}" font-size="{size}" '
            f'font-weight="{weight}">{safe}</text>'
        )
    parts.append("</g>")
    text = text.replace("</svg>", "\n".join(parts) + "\n</svg>")
    path.write_text(text, encoding="utf-8")


def build_hardware_77603(output_dir: Path | None = None) -> int:
    """Render the Toro 77603 profile into its own output tree.

    Additive: this never writes a legacy root export, view or ``validation.json``. Geometry
    rules are the same ones the legacy build uses (``mesh_report``), plus the usable-envelope
    margin check and CSG probes that must be empty or non-empty as declared in the catalog.
    """
    from hardware_77603_catalog import (  # local import: legacy build must not require it
        BUILD_VOLUME_MM,
        COMPONENT_INVENTORY,
        DRAWING_WATERMARK,
        HORN_RADIUS_MAX_MM,
        HORN_RADIUS_MIN_MM,
        MATERIAL_ASSUMPTIONS,
        PARTS_77603,
        PROBES_77603,
        USABLE_VOLUME_MM,
        VIEWS_77603,
    )

    out = (output_dir or ROOT / "generated_77603").resolve()
    staging_root = Path(tempfile.mkdtemp(prefix=".cad-build-77603-", dir=ROOT))
    if out.exists():
        raise SystemExit(f"refusing to overwrite existing output tree: {out}")

    def render_piece(piece) -> tuple:
        target = staging_root / f"{piece.name}.stl"
        run(
            [
                "openscad",
                "--export-format",
                "binstl",
                "-o",
                str(target),
                *defines(piece.parameters),
                str(ROOT / piece.source),
            ]
        )
        report = mesh_report(target)
        size = report["size_mm"]
        if (
            size[0] > USABLE_VOLUME_MM[0]
            or size[1] > USABLE_VOLUME_MM[1]
            or size[2] > USABLE_VOLUME_MM[2]
        ):
            raise ValueError(
                f"{piece.name} exceeds the usable {USABLE_VOLUME_MM} mm envelope: {size}"
            )
        report.update(
            source=f"{piece.source}.scad",
            parameters=piece.parameters,
            quantity=piece.quantity,
            print_orientation=piece.orientation,
            material_hypothesis=piece.material,
            note=piece.note,
            build_envelope_mm=list(BUILD_VOLUME_MM),
            usable_envelope_mm=list(USABLE_VOLUME_MM),
        )
        print(f"PASS mesh {piece.name}: {size}", flush=True)
        return piece.name, report

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        reports = dict(pool.map(render_piece, PARTS_77603))

    def run_probe(probe) -> dict:
        results = []
        for params in probe.parameter_sets:
            target = staging_root / f"probe_{probe.check}.stl"
            text = run(
                [
                    "openscad",
                    "--export-format",
                    "binstl",
                    "-o",
                    str(target),
                    # The parameter set carries the implemented check name; Probe.check is
                    # only a human-facing label, so the parameter set must win the merge.
                    *defines({"check": probe.check, **params}),
                    str(ROOT / "throttle_77603" / "fit_checks.scad"),
                ],
                empty=True,
            )
            empty = "Current top level object is empty" in text
            if probe.expect == "empty" and not empty:
                raise ValueError(f"interference in check {probe.check}: {text}")
            if probe.expect == "contact" and empty:
                raise ValueError(f"no retention/land in check {probe.check}")
            results.append({"check": probe.check, "expect": probe.expect, "empty": empty})
            print(f"PASS probe {probe.check} ({probe.expect})", flush=True)
        return results

    probes = []
    for probe in PROBES_77603:
        probes.extend(run_probe(probe))

    def render_view(scene) -> str:
        folder = "drawings" if scene.kind == "drawing" else "views"
        if scene.kind == "drawing":
            target = staging_root / f"{scene.name}.svg"
            run(
                [
                    "openscad",
                    "-o",
                    str(target),
                    *defines(scene.parameters),
                    str(ROOT / scene.source),
                ]
            )
            if not target.read_bytes().lstrip().startswith(b"<"):
                raise ValueError(f"Missing SVG drawing {scene.name}")
            variant = str(scene.parameters.get("horn_variant", "?"))
            annotate_svg(
                target,
                title=f"Throttle horn interface, variant {variant} - section through pin plane",
                notes=[
                    "Units: millimetres. Section cut through the rod-pin plane.",
                    f"Horn radius band: {HORN_RADIUS_MIN_MM}-{HORN_RADIUS_MAX_MM} mm (variant C adjusts "
                    "continuously in this range)",
                    "Lever capture, inside opening: 31.75 mm left-right x 12.70 mm "
                    "front-back (owner-selected)",
                    "All fits are PROTOTYPE geometry: horn holes, shaft position and lever "
                    "cross-section are unmeasured",
                    "Scale is not 1:1 unless printed at the stated sheet size",
                ],
                watermark=DRAWING_WATERMARK,
            )
        else:
            target = staging_root / f"{scene.name}.png"
            run(
                [
                    "xvfb-run",
                    "-a",
                    "openscad",
                    "-o",
                    str(target),
                    "--imgsize=1200,900",
                    "--projection=p",
                    f"--camera={scene.camera}",
                    "--colorscheme=Tomorrow",
                    *defines(scene.parameters),
                    str(ROOT / scene.source),
                ]
            )
            if not target.read_bytes().startswith(b"\x89PNG"):
                raise ValueError(f"Missing view {scene.name}")
        print(f"PASS {scene.kind} {scene.name}", flush=True)
        return f"{folder}/{target.name}"

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        artifacts = list(pool.map(render_view, VIEWS_77603))

    # Publish only after every mesh and probe passed.
    for sub in ("stl", "3mf", "previews", "views", "drawings"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    print_files = {}
    for name in sorted(reports):
        shutil.move(str(staging_root / f"{name}.stl"), str(out / "stl" / f"{name}.stl"))
        target = out / "3mf" / f"{name}.3mf"
        write_3mf(out / "stl" / f"{name}.stl", target)
        print_files[name] = {
            "path": f"3mf/{name}.3mf",
            "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
        }
    for artifact in artifacts:
        folder, filename = artifact.split("/", 1)
        shutil.move(str(staging_root / filename), str(out / folder / filename))

    sources = sorted(
        {
            *(p.source for p in PARTS_77603),
            *(v.source for v in VIEWS_77603),
            "throttle_77603/fit_checks.scad",
            "throttle_77603/parameters.scad",
            "throttle_77603/printed_joints.scad",
            "control_77603/parameters.scad",
            "control_77603/board_carriers.scad",
            "enclosure_common.scad",
            "common.scad",
        }
    )
    manifest = {
        "profile": "hardware_77603",
        "model": "Toro TimeCutter MAX Havoc MyRIDE 60 in, model 77603 (serial unknown)",
        "status": "geometry_generated_not_physically_qualified",
        "watermark": DRAWING_WATERMARK,
        "tool": run(["openscad", "--version"]).strip(),
        "build_volume_mm": list(BUILD_VOLUME_MM),
        "usable_envelope_mm": list(USABLE_VOLUME_MM),
        "step_status": "UNAVAILABLE: OpenSCAD 2021.01 has no STEP writer; no BREP exported",
        "sources_sha256": {
            source: hashlib.sha256((ROOT / source).read_bytes()).hexdigest()
            for source in sources
        },
        "parts": reports,
        "print_files": print_files,
        "probes": probes,
        "artifacts": sorted(artifacts),
        "component_inventory": [
            {"name": n, "quantity": q, "status": s, "role": r}
            for n, q, s, r in COMPONENT_INVENTORY
        ],
        "material_assumptions": MATERIAL_ASSUMPTIONS,
    }
    (out / "validation.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        f"Published {len(reports)} meshes, {len(print_files)} 3MFs, "
        f"{len(artifacts)} views/drawings and {len(probes)} probes to {out}"
    )
    return 0


def main() -> None:
    with tempfile.TemporaryDirectory(prefix=".cad-build-", dir=ROOT) as temp:
        staging = Path(temp)

        def render(part: tuple) -> tuple:
            name, source, parameters = part
            target = staging / f"{name}.stl"
            run(
                [
                    "openscad",
                    "--export-format",
                    "binstl",
                    "-o",
                    str(target),
                    *defines(parameters),
                    str(ROOT / f"{source}.scad"),
                ]
            )
            report = mesh_report(target)
            report.update(source=f"{source}.scad", parameters=parameters)
            print(f"PASS mesh {name}: {report['size_mm']}", flush=True)
            return name, report

        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
            reports = dict(pool.map(render, PARTS))

        def clearance(name: str) -> str:
            target = staging / f"check_{name}.stl"
            text = run(
                [
                    "openscad",
                    "--export-format",
                    "binstl",
                    "-o",
                    str(target),
                    "-D",
                    f"check={json.dumps(name)}",
                    str(ROOT / "fit_checks.scad"),
                ],
                empty=True,
            )
            if "Current top level object is empty" not in text:
                raise ValueError(f"Clearance interference in {name}: {text}")
            print(f"PASS clearance {name}", flush=True)
            return name

        # Check each actual pad independently; one contacting bar must not mask
        # another station's gap.
        def contact(item: tuple[str, int]) -> str:
            name, index = item
            label = f"{name}_{index}"
            target = staging / f"check_{label}.stl"
            text = run(
                [
                    "openscad",
                    "--export-format",
                    "binstl",
                    "-o",
                    str(target),
                    "-D",
                    f"check={json.dumps(name)}",
                    "-D",
                    f"contact_station={index}",
                    str(ROOT / "fit_checks.scad"),
                ],
                empty=True,
            )
            if "Current top level object is empty" in text:
                raise ValueError(f"NO CONTACT in {label}: bar does not reach its pad")
            print(f"PASS contact {label}", flush=True)
            return label

        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
            cleared = list(pool.map(clearance, CHECKS))
            contacted = list(pool.map(contact, CONTACT_CHECKS))

        def preview(part: tuple) -> None:
            name, source, parameters = part
            target = staging / f"prev_{name}.png"
            run(
                [
                    "xvfb-run",
                    "-a",
                    "openscad",
                    "-o",
                    str(target),
                    "--imgsize=1000,800",
                    "--autocenter",
                    "--viewall",
                    "--render",
                    "--colorscheme=Tomorrow",
                    *defines(parameters),
                    str(ROOT / f"{source}.scad"),
                ]
            )
            if not target.read_bytes().startswith(b"\x89PNG"):
                raise ValueError(f"Missing preview {name}")

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            list(pool.map(preview, PARTS))

        def view(item: tuple) -> None:
            name, scene, camera, automatic_fit = item
            target = staging / f"view_{name}.png"
            run(
                [
                    "xvfb-run",
                    "-a",
                    "openscad",
                    "-o",
                    str(target),
                    "--imgsize=1200,900",
                    "--projection=p",
                    f"--camera={camera}",
                    *(["--autocenter", "--viewall"] if automatic_fit else []),
                    "--colorscheme=Tomorrow",
                    "-D",
                    f"assembly={json.dumps(scene)}",
                    str(ROOT / "assemblies.scad"),
                ]
            )
            if not target.read_bytes().startswith(b"\x89PNG"):
                raise ValueError(f"Missing view {name}")

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            list(pool.map(view, VIEWS))
        for name in reports:
            for file in (f"{name}.stl", f"prev_{name}.png"):
                (staging / file).replace(ROOT / file)
        for name, _, _, _ in VIEWS:
            (staging / f"view_{name}.png").replace(ROOT / f"view_{name}.png")
        print_files = {}
        for name in sorted(reports):
            target = ROOT / "3mf" / f"{name}.3mf"
            write_3mf(ROOT / f"{name}.stl", target)
            print_files[name] = {
                "path": f"3mf/{name}.3mf",
                "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
            }
        projects = audit_print_projects(
            {name: report["size_mm"] for name, report in reports.items()}
        )
        for project in projects:
            if project["status"] != "current":
                print(
                    f"WARN print project {project['file']}: "
                    f"{project['status']} - {project['detail']}",
                    flush=True,
                )
        manifest = {
            "tool": run(["openscad", "--version"]).strip(),
            "status": "mesh_and_named_geometry_checks_passed_not_physical_qualification",
            "sources_sha256": {
                path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                for path in sorted(ROOT.glob("*.scad"))
            },
            "parts": reports,
            "print_files": print_files,
            "print_projects": projects,
            "empty_interference_checks": cleared,
            "nonempty_contact_checks": contacted,
        }
        (ROOT / "validation.json").write_text(json.dumps(manifest, indent=2) + "\n")
        stale = sum(1 for project in projects if project["status"] != "current")
        print(
            f"Published {len(reports)} STL/PNG pairs, {len(print_files)} 3MFs, "
            f"{len(VIEWS)} views; {len(cleared)} clearances and {len(contacted)} contacts "
            f"passed." + (f" {stale} print project(s) need attention." if stale else "")
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",
        choices=("legacy", "hardware_77603"),
        default="legacy",
        help="legacy rebuilds every existing part in place; hardware_77603 writes its own tree",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="hardware_77603 only: publish here instead of hardware/mounts/generated_77603",
    )
    args = parser.parse_args()
    if args.profile == "legacy":
        main()
    else:
        raise SystemExit(build_hardware_77603(args.output_dir))
