"""Check the published 77603 files against the CAD geometry delivered.

These checks detect stale inputs and different STL/3MF meshes. They do not
establish physical fit, strength or safety.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import struct
import subprocess
import xml.etree.ElementTree as ET
import zipfile
from collections import Counter
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
MOUNTS = REPO / "hardware" / "mounts"
MANIFEST = MOUNTS / "generated_77603" / "validation.json"


@pytest.fixture(scope="module")
def manifest():
    return json.loads(MANIFEST.read_text())


def test_manifest_inputs_are_current(manifest):
    """An edited head or shared steering datum invalidates an old printable."""
    hashes = manifest["sources_sha256"]
    required = {part["source"] for part in manifest["parts"].values()}
    required.update({"hitch_arm_layer.scad", "enclosure_common.scad", "pushrod.scad"})
    missing = required - hashes.keys()
    assert not missing, f"unrecorded CAD inputs: {missing}"
    for source, expected in hashes.items():
        actual = hashlib.sha256((MOUNTS / source).read_bytes()).hexdigest()
        assert actual == expected, source


def test_published_stls_match_their_manifest(manifest):
    root = REPO / manifest["artifact_root"]
    for name, part in manifest["parts"].items():
        actual = hashlib.sha256((root / f"{name}.stl").read_bytes()).hexdigest()
        assert actual == part["stl_sha256"], name


def _stl_triangles(path):
    data = path.read_bytes()
    count = struct.unpack_from("<I", data, 80)[0]
    assert len(data) == 84 + 50 * count
    triangles = []
    for i in range(count):
        record = struct.unpack_from("<12fH", data, 84 + 50 * i)
        triangles.append(
            tuple(tuple(round(value, 5) for value in record[j : j + 3]) for j in (3, 6, 9))
        )
    low = [min(vertex[axis] for face in triangles for vertex in face) for axis in range(3)]
    shift = [-value if value < 0 else 0.0 for value in low]
    return Counter(
        tuple(sorted(tuple(round(vertex[i] + shift[i], 5) for i in range(3)) for vertex in face))
        for face in triangles
    )


def _three_mf_triangles(path):
    with zipfile.ZipFile(path) as package:
        root = ET.fromstring(package.read("3D/3dmodel.model"))
    triangles = Counter()
    for mesh in root.findall(".//{*}mesh"):
        vertices = [
            tuple(float(vertex.attrib[axis]) for axis in ("x", "y", "z"))
            for vertex in mesh.findall("./{*}vertices/{*}vertex")
        ]
        for face in mesh.findall("./{*}triangles/{*}triangle"):
            corners = (vertices[int(face.attrib[key])] for key in ("v1", "v2", "v3"))
            triangles[tuple(sorted(corners))] += 1
    return triangles


def test_3mf_contains_the_same_mesh_as_the_stl(manifest):
    root = REPO / manifest["artifact_root"]
    for name, entry in manifest["print_files"].items():
        path = root / entry["path"]
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        assert actual == entry["sha256"], name
        assert _three_mf_triangles(path) == _stl_triangles(root / f"{name}.stl"), name


@pytest.mark.skipif(shutil.which("openscad") is None, reason="OpenSCAD is not installed")
def test_derivative_does_not_remove_legacy_steering_geometry(tmp_path):
    """Subtract the derivative from the actual legacy head."""
    src = tmp_path / "preserved.scad"
    src.write_text(
        f"include <{MOUNTS / 'throttle_77603/parameters.scad'}>\n"
        f"use <{MOUNTS / 'hitch_arm_layer.scad'}>\n"
        f"use <{MOUNTS / 'throttle_77603/throttle_arm_head.scad'}>\n"
        "difference() { arm_head(); throttle_arm_head(); }\n"
    )
    result = subprocess.run(
        ["openscad", "-o", str(tmp_path / "missing.stl"), str(src)],
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    output = result.stdout + result.stderr
    assert "WARNING:" not in output and "ERROR:" not in output, output
    assert result.returncode == 1 and "Current top level object is empty" in output, output


@pytest.mark.skipif(shutil.which("openscad") is None, reason="OpenSCAD is not installed")
def test_derivative_keeps_servo_passages_open(tmp_path):
    """Added material must not fill the existing six holes or centre opening."""
    src = tmp_path / "servo_passages.scad"
    src.write_text(
        f"include <{MOUNTS / 'throttle_77603/parameters.scad'}>\n"
        f"use <{MOUNTS / 'hitch_arm_layer.scad'}>\n"
        f"use <{MOUNTS / 'throttle_77603/throttle_arm_head.scad'}>\n"
        "intersection() {\n"
        "  throttle_arm_head();\n"
        "  translate([-1, 0, arm_head_rise()]) {\n"
        "    servo_rear_hole_cut(servo_face_dx() + 20);\n"
        "    servo_rear_opening_cut(servo_face_dx() + 20);\n"
        "  }\n"
        "}\n"
    )
    result = subprocess.run(
        ["openscad", "-o", str(tmp_path / "blocked.stl"), str(src)],
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    output = result.stdout + result.stderr
    assert "WARNING:" not in output and "ERROR:" not in output, output
    assert result.returncode == 1 and "Current top level object is empty" in output, output


@pytest.mark.skipif(shutil.which("openscad") is None, reason="OpenSCAD is not installed")
def test_printable_u_fitting_has_open_nominal_cavity(tmp_path):
    """Test the exported STL: an included joint used to block its cavity."""
    src = tmp_path / "u_cavity.scad"
    src.write_text(
        "intersection() {\n"
        f'    import("{MOUNTS / "throttle_u_body.stl"}");\n'
        "    translate([-15.865, -6.34, 0.1]) cube([31.73, 12.68, 17.8]);\n"
        "}\n"
    )
    result = subprocess.run(
        ["openscad", "-o", str(tmp_path / "blocked.stl"), str(src)],
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    output = result.stdout + result.stderr
    assert "WARNING:" not in output and "ERROR:" not in output, output
    assert result.returncode == 1 and "Current top level object is empty" in output, output


@pytest.mark.skipif(shutil.which("openscad") is None, reason="OpenSCAD is not installed")
def test_printed_u_body_has_an_integral_hinge_eye(tmp_path):
    """The cavity alone is not enough: the neck must have material to pin to."""
    src = tmp_path / "u_eye.scad"
    src.write_text(
        f"include <{MOUNTS / 'throttle_77603/parameters.scad'}>\n"
        "intersection() {\n"
        f'    import("{MOUNTS / "throttle_u_body.stl"}");\n'
        "    translate([-6, -3, throttle_u_engagement_mm + throttle_u_wall_mm + 1])\n"
        "        cube([12, 6, 2]);\n"
        "}\n"
    )
    result = subprocess.run(
        ["openscad", "-o", str(tmp_path / "eye.stl"), str(src)],
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    assert "WARNING:" not in output and "ERROR:" not in output, output


@pytest.mark.skipif(shutil.which("openscad") is None, reason="OpenSCAD is not installed")
@pytest.mark.parametrize("check", ["body_bore", "neck_lower", "mating", "lower_pin"])
def test_printed_u_joint_clearances(tmp_path, check):
    """Actual exported parts must have aligned, open pin paths and fit together."""
    body = MOUNTS / "throttle_u_body.stl"
    neck = MOUNTS / "throttle_u_neck.stl"
    pin = MOUNTS / "throttle_u_pin.stl"
    cases = {
        "body_bore": (
            f'import("{body}"); '
            "translate([0, 20, first_pin]) rotate([90, 0, 0]) "
            "cylinder(d=pin_d_mm, h=40, $fn=32);"
        ),
        "neck_lower": (
            f'import("{neck}"); '
            "translate([0, 20, clevis_ear_t_mm + clevis_gap_mm / 2]) "
            "rotate([90, 0, 0]) cylinder(d=pin_d_mm, h=40, $fn=32);"
        ),
        "mating": (f'import("{body}"); translate([0, 0, neck_base]) import("{neck}");'),
        # The pin's head is designed to bear on the neck ear's outer face; at the nominal
        # station the two faces are coplanar and OpenSCAD reports a zero-volume, non-manifold
        # intersection. Back the pin 0.1 mm off its seat to test only the shaft's path;
        # test_u_pin_head_seats_on_the_neck_ear covers the seat itself.
        "lower_pin": (
            f'union() {{ import("{body}"); translate([0, 0, neck_base]) import("{neck}"); }} '
            "translate([0, (clevis_gap_mm + 2 * clevis_ear_t_mm) / 2 + keeper_t_mm - 0.1, "
            "first_pin]) rotate([90, 0, 0]) "
            f'import("{pin}");'
        ),
    }
    src = tmp_path / f"{check}.scad"
    src.write_text(
        f"include <{MOUNTS / 'throttle_77603/parameters.scad'}>\n"
        f"use <{MOUNTS / 'throttle_77603/throttle_u_fitting.scad'}>\n"
        "neck_base = throttle_u_engagement_mm + throttle_u_wall_mm + joint_clearance_mm;\n"
        "first_pin = neck_base + clevis_ear_t_mm + clevis_gap_mm / 2;\n"
        "neck_height = 2 * clevis_ear_t_mm + clevis_gap_mm + 6;\n"
        f"intersection() {{ {cases[check]} }}\n"
    )
    result = subprocess.run(
        ["openscad", "-o", str(tmp_path / "interference.stl"), str(src)],
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    output = result.stdout + result.stderr
    assert "WARNING:" not in output and "ERROR:" not in output, output
    assert result.returncode == 1 and "Current top level object is empty" in output, output


@pytest.mark.skipif(shutil.which("openscad") is None, reason="OpenSCAD is not installed")
def test_u_pin_head_seats_on_the_neck_ear(tmp_path):
    """Driving the exported pin 0.1 mm past its station must meet the neck ear: the head
    reaches its bearing face rather than floating clear of the joint."""
    src = tmp_path / "seat.scad"
    src.write_text(
        f"include <{MOUNTS / 'throttle_77603/parameters.scad'}>\n"
        "neck_base = throttle_u_engagement_mm + throttle_u_wall_mm + joint_clearance_mm;\n"
        "first_pin = neck_base + clevis_ear_t_mm + clevis_gap_mm / 2;\n"
        "intersection() {\n"
        f'  translate([0, 0, neck_base]) import("{MOUNTS / "throttle_u_neck.stl"}");\n'
        "  translate([0, (clevis_gap_mm + 2 * clevis_ear_t_mm) / 2 + keeper_t_mm + 0.1,\n"
        "             first_pin])\n"
        f'    rotate([90, 0, 0]) import("{MOUNTS / "throttle_u_pin.stl"}");\n'
        "}\n"
    )
    result = subprocess.run(
        ["openscad", "-o", str(tmp_path / "seat.stl"), str(src)],
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    output = result.stdout + result.stderr
    assert "ERROR:" not in output, output
    assert result.returncode == 0 and "Current top level object is empty" not in output, output


@pytest.mark.skipif(shutil.which("openscad") is None, reason="OpenSCAD is not installed")
@pytest.mark.parametrize("side", [-1, 1])
def test_printed_clevis_has_two_bored_ears(tmp_path, side):
    """Each actual exported ear must carry material surrounding an open X-axis bore."""
    header = (
        f"include <{MOUNTS / 'throttle_77603/parameters.scad'}>\n"
        f'part = "{MOUNTS / "clevis.stl"}";\n'
        "pin_z = clevis_ear_t_mm + clevis_gap_mm / 2;\n"
    )
    probes = {
        "bore": (
            "intersection() { import(part); "
            "translate([-30, 0, pin_z]) rotate([0, 90, 0]) "
            "cylinder(d=pin_d_mm, h=60, $fn=32); }\n"
        ),
        "ear": (
            "intersection() { import(part); "
            f"translate([{side} * (clevis_gap_mm / 2 + clevis_ear_t_mm / 2) - 1,"
            " -2, pin_z + pin_d_mm / 2 + 0.6]) cube([2, 4, 1]); }\n"
        ),
    }
    for label, probe in probes.items():
        src = tmp_path / f"{label}.scad"
        src.write_text(header + probe)
        result = subprocess.run(
            ["openscad", "-o", str(tmp_path / f"{label}.stl"), str(src)],
            capture_output=True,
            text=True,
            timeout=180,
            check=False,
        )
        output = result.stdout + result.stderr
        assert "WARNING:" not in output and "ERROR:" not in output, output
        if label == "bore":
            assert result.returncode == 1 and "Current top level object is empty" in output, output
        else:
            assert result.returncode == 0, output


@pytest.mark.skipif(shutil.which("openscad") is None, reason="OpenSCAD is not installed")
def test_leaned_fitting_keeps_print_flat_and_tilts_the_cavity(tmp_path):
    """A nonzero lever lean applies to the assembled fitting, not its print bed."""
    body = tmp_path / "body.stl"
    result = subprocess.run(
        [
            "openscad",
            "--export-format",
            "binstl",
            "-o",
            str(body),
            "-D",
            'u_part="body"',
            "-D",
            "throttle_lean_right_deg=12",
            str(MOUNTS / "throttle_77603/throttle_u_fitting.scad"),
        ],
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    assert "WARNING:" not in output and "ERROR:" not in output, output
    data = body.read_bytes()
    count = struct.unpack_from("<I", data, 80)[0]
    assert count > 0
    z_values = [
        struct.unpack_from("<12fH", data, 84 + i * 50)[j] for i in range(count) for j in (5, 8, 11)
    ]
    assert min(z_values) == pytest.approx(0, abs=0.001)

    src = tmp_path / "leaned.scad"
    src.write_text(
        f"include <{MOUNTS / 'throttle_77603/parameters.scad'}>\n"
        f"use <{MOUNTS / 'throttle_77603/throttle_u_fitting.scad'}>\n"
        "intersection() {\n"
        '  throttle_u_fitting("assembly");\n'
        "  rotate([0, 12, 0]) translate([-15.865, -6.34, 0.1])\n"
        "    cube([31.73, 12.68, 17.8]);\n"
        "}\n"
    )
    result = subprocess.run(
        [
            "openscad",
            "-o",
            str(tmp_path / "collision.stl"),
            "-D",
            "throttle_lean_right_deg=12",
            str(src),
        ],
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    output = result.stdout + result.stderr
    assert "WARNING:" not in output and "ERROR:" not in output, output
    assert result.returncode == 1 and "Current top level object is empty" in output, output


@pytest.mark.skipif(shutil.which("openscad") is None, reason="OpenSCAD is not installed")
@pytest.mark.parametrize("misalignment", [0, 2])
def test_tilted_u_post_datum_matches_the_actual_post(tmp_path, misalignment):
    """The rod eye's swept annulus at the published datum clears the tilted post; a
    2 mm displacement must hit the actual spigot."""
    src = tmp_path / "tilted_post.scad"
    src.write_text(
        f"include <{MOUNTS / 'throttle_77603/parameters.scad'}>\n"
        f"use <{MOUNTS / 'throttle_77603/throttle_u_fitting.scad'}>\n"
        "intersection() {\n"
        "  u_oriented() translate([0, 0, u_neck_base_z()]) u_neck();\n"
        "  translate(u_post_offset_mm())\n"
        "    rotate([throttle_pitch_deg, 0, 0])\n"
        "    rotate([0, 0, throttle_yaw_deg])\n"
        "    rotate([0, throttle_lean_right_deg, 0])\n"
        f"    translate([0, {misalignment}, -pivot_spigot_len_mm / 2 + 0.2])\n"
        "      difference() {\n"
        "        cylinder(d=12, h=pivot_spigot_len_mm - 0.4, $fn=48);\n"
        "        translate([0, 0, -1])\n"
        "          cylinder(d=pivot_spigot_d_mm + 0.4, h=pivot_spigot_len_mm + 2, $fn=48);\n"
        "      }\n"
        "}\n"
    )
    result = subprocess.run(
        [
            "openscad",
            "-o",
            str(tmp_path / "blocked.stl"),
            "-D",
            "throttle_lean_right_deg=12",
            "-D",
            "throttle_yaw_deg=-7",
            "-D",
            "throttle_pitch_deg=5",
            str(src),
        ],
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    output = result.stdout + result.stderr
    assert "WARNING:" not in output and "ERROR:" not in output, output
    if misalignment:
        assert result.returncode == 0, output
    else:
        assert result.returncode == 1 and "Current top level object is empty" in output, output


@pytest.mark.skipif(shutil.which("openscad") is None, reason="OpenSCAD is not installed")
@pytest.mark.parametrize("region,expect_empty", [("band_outer", True), ("band_core", False)])
def test_printed_pin_has_captured_keeper_band(tmp_path, region, expect_empty):
    """The keeper band must have a solid core and a reduced outside diameter."""
    src = tmp_path / "keeper_band.scad"
    probe = (
        "difference() { cylinder(d=5.2,h=2,$fn=48); cylinder(d=4.7,h=2,$fn=48); }"
        if region == "band_outer"
        else "cylinder(d=4,h=2,$fn=48);"
    )
    src.write_text(
        f"use <{MOUNTS / 'throttle_77603/printed_joints.scad'}>\n"
        "intersection() { printed_pin(24); translate([0,0,1]) "
        f"{probe} }}\n"
    )
    result = subprocess.run(
        ["openscad", "-o", str(tmp_path / "band.stl"), str(src)],
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    output = result.stdout + result.stderr
    assert "WARNING:" not in output and "ERROR:" not in output, output
    if expect_empty:
        assert result.returncode == 1 and "Current top level object is empty" in output, output
    else:
        assert result.returncode == 0, output
