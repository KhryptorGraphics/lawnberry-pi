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
    """Test the exported print: neither the body nor its printed-in-place yoke may block
    the lever cavity."""
    src = tmp_path / "u_cavity.scad"
    src.write_text(
        "intersection() {\n"
        f'    import("{MOUNTS / "throttle_u_fitting.stl"}");\n'
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
def test_printed_u_fitting_has_a_solid_hinge_boss(tmp_path):
    """The cavity alone is not enough: the boss between the blind bores must be material,
    or the yoke's stub pins would bear on nothing."""
    src = tmp_path / "u_eye.scad"
    src.write_text(
        f"include <{MOUNTS / 'throttle_77603/parameters.scad'}>\n"
        "intersection() {\n"
        f'    import("{MOUNTS / "throttle_u_fitting.stl"}");\n'
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
            'u_part="fitting"',
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
    """The rod eye's swept annulus at the published datum clears the tilted yoke's post; a
    2 mm displacement must hit the actual spigot."""
    src = tmp_path / "tilted_post.scad"
    src.write_text(
        f"include <{MOUNTS / 'throttle_77603/parameters.scad'}>\n"
        f"use <{MOUNTS / 'throttle_77603/throttle_u_fitting.scad'}>\n"
        "intersection() {\n"
        "  u_oriented() u_yoke();\n"
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


def test_manifest_pip_pairs_meet_mesh_release_gates(manifest):
    """The four co-printed joints must publish two bodies within the PIP limits."""
    expected = {
        "rod_capture_assembly",
        "throttle_u_fitting",
        "variant_c_pip",
        "variant_d_pip",
    }
    parts = manifest["parts"]
    actual = {
        name for name, part in parts.items() if part.get("pip_bodies", 1) > 1
    }
    assert actual == expected

    for name in expected:
        part = parts[name]
        assert part["connected_solids"] == 2, name
        assert part["pip_bodies"] == 2, name
        assert part["pip_min_gap_mm"] >= 0.35, name
        assert part["pip_droop_area_mm2"] <= 0.5, name


def test_readme_quoted_counts_match_the_catalog_and_manifest(manifest):
    """The README's headline counts are generated numbers, so they cannot drift."""
    from hardware.mounts.hardware_77603_catalog import PROBES_77603

    readme = (MOUNTS / "throttle_77603" / "README.md").read_text()
    piece_count = len(manifest["parts"])
    probe_runs = sum(len(probe.parameter_sets) for probe in PROBES_77603)
    named_checks = len({probe.check for probe in PROBES_77603})

    assert f"all {piece_count}\ncatalogued pieces" in readme
    assert f"all {probe_runs} CSG probe runs ({named_checks} named\nchecks" in readme


def test_stale_profile_exports_are_removed_without_touching_current_or_legacy(tmp_path):
    from hardware.mounts.build_printables import _remove_stale_profile_exports

    names = ("obsolete", "current", "legacy")
    for name in names:
        paths = (
            tmp_path / f"{name}.stl",
            tmp_path / f"prev_{name}.png",
            tmp_path / "3mf" / f"{name}.3mf",
        )
        for path in paths:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"artifact")

    removed = _remove_stale_profile_exports(
        {"obsolete", "current", "legacy"},
        {"current"},
        {"legacy"},
        tmp_path,
    )
    assert removed == ["obsolete"]
    assert not any(
        path.exists()
        for path in (
            tmp_path / "obsolete.stl",
            tmp_path / "prev_obsolete.png",
            tmp_path / "3mf" / "obsolete.3mf",
        )
    )
    for name in ("current", "legacy"):
        assert (tmp_path / f"{name}.stl").is_file()
        assert (tmp_path / f"prev_{name}.png").is_file()
        assert (tmp_path / "3mf" / f"{name}.3mf").is_file()


def test_published_flat_exports_match_both_catalogs(manifest):
    """The slicer-facing inventory has no missing or orphaned STL/3MF/preview files."""
    from hardware.mounts.build_printables import PARTS as LEGACY_PARTS
    from hardware.mounts.hardware_77603_catalog import PARTS_77603

    profile_names = {piece.name for piece in PARTS_77603}
    expected = profile_names | {name for name, _, _ in LEGACY_PARTS}
    assert set(manifest["parts"]) == profile_names
    assert {path.stem for path in MOUNTS.glob("*.stl")} == expected
    assert {path.stem for path in (MOUNTS / "3mf").glob("*.3mf")} == expected
    assert {
        path.stem.removeprefix("prev_") for path in MOUNTS.glob("prev_*.png")
    } == expected
