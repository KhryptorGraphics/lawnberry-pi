"""Artifact catalog for the Toro 77603 throttle/packaging build profile.

This is the single declaration of what gets rendered, what CSG probes must pass, and what
each piece's print metadata is. ``build_printables.py --profile hardware_77603`` consumes it.

Every printed piece is one watertight solid, printed bed-zero, inside the usable
400 x 400 x 490 mm envelope (10 mm reserve per XY edge, 10 mm Z on the 420 x 420 x 500
machine). Assemblies are scenes for views/drawings only; they are never validated as a
single printable part.
"""

from __future__ import annotations

from dataclasses import dataclass, field

THROTTLE = "throttle_77603"
CONTROL = "control_77603"

HORN_RADIUS_MIN_MM = 12.0  # mirrors horn_radius_min_mm in parameters.scad
HORN_RADIUS_DEFAULT_MM = 18.0  # mirrors horn_radius_default_mm in parameters.scad
HORN_RADIUS_MAX_MM = 30.0  # mirrors horn_radius_max_mm in parameters.scad
BUILD_VOLUME_MM = (420, 420, 500)
USABLE_VOLUME_MM = (400, 400, 490)
DRAWING_WATERMARK = "PROTOTYPE - FIT / LOAD / ELECTRICAL VALIDATION REQUIRED"


@dataclass(frozen=True, slots=True)
class Piece:
    """One printable part or fit coupon."""

    name: str
    source: str
    parameters: dict[str, object]
    quantity: int = 1
    orientation: str = "flat, largest face on the bed"
    material: str = "PETG (prototype; ASA unqualified pending printer/enclosure check)"
    note: str = ""
    # >1 marks a print-in-place piece: one print, this many captive bodies (pip_gate.py).
    bodies: int = 1


@dataclass(frozen=True, slots=True)
class Scene:
    """One rendered assembly view or projected drawing."""

    name: str
    source: str
    parameters: dict[str, object]
    camera: str
    kind: str = "view"  # "view" | "drawing"
    note: str = ""


@dataclass(frozen=True, slots=True)
class Probe:
    """A named CSG check.

    ``expect`` is ``empty`` for a negative-volume interference probe, or ``contact`` for a
    positive land/material-presence probe that must NOT be empty. ``contact`` proves only
    that named geometry carries material where the probe samples it. It is NOT proof of
    physical retention, fit or load capacity: no retention claim is made from non-empty
    contact alone.
    """

    check: str
    expect: str
    parameter_sets: tuple[dict[str, object], ...] = (field(default_factory=dict),)
    note: str = ""
    # Per-run openscad cap in seconds; None keeps the runner's default. Set only for a probe
    # whose real solids are genuinely slow to build, never to hide a hung render.
    timeout_s: int | None = None


def _t(part: str) -> dict[str, object]:
    # Generic "part" selector: enclosure, carriers.
    return {"part": part}


def _mount(part: str) -> dict[str, object]:
    # mg995_throttle_mount.scad reads the global `mount_part`.
    return {"mount_part": part}


def _carrier(kind: str) -> dict[str, object]:
    # board_carriers.scad reads the globals `carrier_kind` / `carrier_part`.
    return {"carrier_kind": kind, "carrier_part": "body"}


def _rod(part: str) -> dict[str, object]:
    # throttle_pushrod.scad reads the global `rod_part`. Sending `part` was silently
    # ignored by OpenSCAD, so every rod piece rendered the full assembly.
    return {"rod_part": part}


def _u(part: str) -> dict[str, object]:
    return {"u_part": part}


def _variant(letter: str, part: str, **extra: object) -> dict[str, object]:
    # throttle_horn_variants.scad reads `horn_variant` (distinct from the module's own
    # `variant` parameter, so -D is not shadowed by a signature default).
    return {"horn_variant": letter, "horn_part": part, **extra}


# ---- Printable pieces ---------------------------------------------------------------
MOUNT_PIECES = (
    # Derivative of the existing RIGHT arm head: the legacy head, raised with the left one,
    # plus a shelf, corbel and cradle walls under the steering servo. It replaces
    # arm_head_right for the throttle build.
    Piece(
        "throttle_arm_head",
        f"{THROTTLE}/throttle_arm_head.scad",
        {},
        orientation="mating face down, as the legacy arm head prints",
        note="derivative of arm_head_right; integral shelf and cradle walls carry the "
        "throttle MG995 below the raised steering servo; fit and joint strength unverified",
    ),
    Piece("servo_keeper", f"{THROTTLE}/mg995_throttle_mount.scad", _mount("keeper")),
    Piece(
        "servo_keeper_pin",
        f"{THROTTLE}/mg995_throttle_mount.scad",
        _mount("keeper_pin"),
        quantity=2,
        orientation="upright on its head",
        note="transverse pins through the keeper legs and cradle side walls; fit, clip "
        "retention and vibration life unverified",
    ),
    Piece(
        "servo_pin_keeper",
        f"{THROTTLE}/mg995_throttle_mount.scad",
        _mount("pin_keeper"),
        quantity=2,
        orientation="flat",
        note="split ring for each keeper pin; fatigue unverified",
    ),
    Piece("mg995_lug_gauge", f"{THROTTLE}/mg995_throttle_mount.scad", _mount("gauge")),
)


ROD_PIECES = (
    # Selector names match throttle_pushrod.scad's dispatcher. The slider, the receiver
    # sleeve and its threaded collet nut are one print-in-place export with three captive
    # bodies; all other rod parts are separately printed components required by the
    # assembly.
    Piece(
        "rod_segment_lower",
        f"{THROTTLE}/throttle_pushrod.scad",
        _rod("segment_lower"),
        orientation="upright, servo eye end on the bed",
        note="servo section: integral eye plate 16 mm below the tube axis, fused to the "
        "tube wall; the tube's first face over the plate needs support",
    ),
    Piece(
        "rod_segment_middle",
        f"{THROTTLE}/throttle_pushrod.scad",
        _rod("segment_middle"),
        orientation="lying flat, long axis on the bed X",
    ),
    Piece(
        "rod_segment_upper",
        f"{THROTTLE}/throttle_pushrod.scad",
        _rod("segment_upper"),
        orientation="lying flat, long axis on the bed X",
    ),
    Piece(
        "rod_capture_assembly",
        f"{THROTTLE}/throttle_pushrod.scad",
        _rod("capture_assembly"),
        orientation="upright, COLLET END DOWN: finger tips, nut front face and slider tip "
        "on the bed",
        note="print-in-place set: integral 45-degree sleeve lug rides in the slider groove; "
        "the captive threaded collet nut is hand-tightened to clamp the slider (no bolts, "
        "no tools); clamp force and vibration resistance unqualified",
        bodies=3,
    ),
    Piece(
        "splice_wedge",
        f"{THROTTLE}/throttle_pushrod.scad",
        _rod("splice_wedge"),
        quantity=4,
        note="one bar per section joint (servo section, two chain sections, sleeve) and "
        "one through the throttle-end body and the slider's window",
    ),
    Piece(
        "splice_keeper",
        f"{THROTTLE}/throttle_pushrod.scad",
        _rod("splice_keeper"),
        quantity=4,
        note="one keeper per transverse bar: three section joints and the throttle-end body",
    ),
    Piece(
        "rod_throttle_body",
        f"{THROTTLE}/throttle_pushrod.scad",
        _rod("throttle_body"),
        orientation="upright, rear face on the bed",
        note="sockets over the slider tip behind the fork, retained by a splice bar and "
        "keeper; its boss journals the fine screw",
    ),
    Piece(
        "rod_fine_screw",
        f"{THROTTLE}/throttle_pushrod.scad",
        _rod("fine_screw"),
        orientation="upright on its flange",
    ),
    Piece(
        "rod_fine_carriage_nut",
        f"{THROTTLE}/throttle_pushrod.scad",
        _rod("fine_carriage_nut"),
        orientation="upright",
    ),
    Piece(
        "rod_fine_jam_collar",
        f"{THROTTLE}/throttle_pushrod.scad",
        _rod("fine_jam_collar"),
        orientation="upright",
    ),
    Piece(
        "rod_throttle_fork",
        f"{THROTTLE}/throttle_pushrod.scad",
        _rod("throttle_fork"),
        note="throttle eye plate, side web, fine-nut lug; the eye drops over the U post",
    ),
    Piece(
        "rod_fit_gauge",
        f"{THROTTLE}/throttle_pushrod.scad",
        _rod("gauge"),
        note="slider passes, bore does not: proves the sliding allowance",
    ),
)


U_PIECES = (
    Piece(
        "throttle_u_fitting",
        f"{THROTTLE}/throttle_u_fitting.scad",
        _u("fitting"),
        orientation="as exported: mouth on the bed, yoke legs on the bed beside it",
        note="print-in-place: body and pitch yoke leave the bed as two captive bodies; "
        "no pin or keeper",
        bodies=2,
    ),
    Piece("throttle_u_gauge", f"{THROTTLE}/throttle_u_fitting.scad", _u("gauge")),
    Piece(
        "throttle_u_cap",
        f"{THROTTLE}/throttle_u_fitting.scad",
        _u("cap"),
        orientation="flat",
        note="traps the rod's throttle eye on the yoke's pivot post",
    ),
)

VARIANT_PIECES = (
    # Selector names match throttle_horn_variants.scad's per-variant dispatchers exactly;
    # every printable part of every kit is listed so the flat export set is complete.
    # Each part models its lowest point at z = 0 in its own frame (bed-zero export).
    Piece(
        "variant_a_post",
        f"{THROTTLE}/throttle_horn_variants.scad",
        _variant("A", "post"),
        orientation="upright",
    ),
    Piece("variant_a_cap", f"{THROTTLE}/throttle_horn_variants.scad", _variant("A", "cap")),
    Piece("variant_a_keeper", f"{THROTTLE}/throttle_horn_variants.scad", _variant("A", "keeper")),
    Piece("variant_b_shell_a", f"{THROTTLE}/throttle_horn_variants.scad", _variant("B", "shell_a")),
    Piece("variant_b_shell_b", f"{THROTTLE}/throttle_horn_variants.scad", _variant("B", "shell_b")),
    Piece(
        "variant_b_lug",
        f"{THROTTLE}/throttle_horn_variants.scad",
        _variant("B", "lug"),
        orientation="upright",
    ),
    Piece("variant_b_cap", f"{THROTTLE}/throttle_horn_variants.scad", _variant("B", "cap")),
    Piece(
        "variant_b_pin",
        f"{THROTTLE}/throttle_horn_variants.scad",
        _variant("B", "pin"),
        orientation="upright",
    ),
    Piece("variant_b_keeper", f"{THROTTLE}/throttle_horn_variants.scad", _variant("B", "keeper")),
    Piece(
        "variant_c_pip",
        f"{THROTTLE}/throttle_horn_variants.scad",
        _variant("C", "pip", radius_mm=HORN_RADIUS_DEFAULT_MM),
        orientation="flat, carrier and continuous-radius shoe co-printed at z = 0",
        note="print-in-place pair: opposing 45-degree dovetail flanks capture the shoe; "
        "closed rail ends retain it at both travel limits; the shoe's neck carries the "
        "wedge key's gable-roofed slot; horn contact remains provisional",
        bodies=2,
    ),
    Piece(
        "variant_c_wedge_key",
        f"{THROTTLE}/throttle_horn_variants.scad",
        _variant("C", "wedge_key"),
        orientation="flat, bearing underside on the bed",
        note="1:6 through-wedge: bears on the rail seat and the shoe roof and lifts the shoe "
        "onto the undercut; friction hold, preload retention and wear are unqualified",
    ),
    Piece(
        "variant_c_wedge_keeper",
        f"{THROTTLE}/throttle_horn_variants.scad",
        _variant("C", "wedge_keeper"),
        note="split ring on the key's tail groove: a backstop with about 2.4 mm slack; "
        "its 0.2 mm snap is unqualified",
    ),
    Piece("variant_c_cap", f"{THROTTLE}/throttle_horn_variants.scad", _variant("C", "cap")),
    Piece(
        "variant_c_pin",
        f"{THROTTLE}/throttle_horn_variants.scad",
        _variant("C", "pin"),
        orientation="upright",
    ),
    Piece("variant_c_keeper", f"{THROTTLE}/throttle_horn_variants.scad", _variant("C", "keeper")),
    Piece(
        "variant_d_pip",
        f"{THROTTLE}/throttle_horn_variants.scad",
        _variant("D", "pip"),
        orientation="upright, carrier and truncated ball co-printed at z = 0",
        note="print-in-place pair: carrier-integral spherical socket captures the "
        "truncated ball and its output post",
        bodies=2,
    ),
    Piece("variant_d_cap", f"{THROTTLE}/throttle_horn_variants.scad", _variant("D", "cap")),
    Piece(
        "variant_d_pin",
        f"{THROTTLE}/throttle_horn_variants.scad",
        _variant("D", "pin"),
        orientation="upright",
    ),
    Piece("variant_d_keeper", f"{THROTTLE}/throttle_horn_variants.scad", _variant("D", "keeper")),
    Piece("variant_e_carrier", f"{THROTTLE}/throttle_horn_variants.scad", _variant("E", "carrier")),
    Piece("variant_e_clevis", f"{THROTTLE}/throttle_horn_variants.scad", _variant("E", "clevis")),
    Piece(
        "variant_e_bushing",
        f"{THROTTLE}/throttle_horn_variants.scad",
        _variant("E", "bushing"),
        orientation="upright",
    ),
    Piece(
        "variant_e_eye_bushing",
        f"{THROTTLE}/throttle_horn_variants.scad",
        _variant("E", "eye_bushing"),
        orientation="upright",
    ),
    Piece(
        "variant_e_pin",
        f"{THROTTLE}/throttle_horn_variants.scad",
        _variant("E", "pin"),
        orientation="upright",
    ),
    Piece("variant_e_keeper", f"{THROTTLE}/throttle_horn_variants.scad", _variant("E", "keeper")),
    Piece(
        "variant_e_index_pin",
        f"{THROTTLE}/throttle_horn_variants.scad",
        _variant("E", "index_pin"),
        orientation="upright",
    ),
    Piece(
        "variant_e_index_keeper",
        f"{THROTTLE}/throttle_horn_variants.scad",
        _variant("E", "index_keeper"),
    ),
)

CONTROL_PIECES = (
    Piece(
        "control_enclosure_body",
        f"{CONTROL}/electronics_enclosure.scad",
        {"enclosure_part": "body"},
        orientation="open side up",
        note="printed in two stacked shells if taller than the usable Z",
    ),
    Piece(
        "control_enclosure_lid", f"{CONTROL}/electronics_enclosure.scad", {"enclosure_part": "lid"}
    ),
    Piece("control_gasket", f"{CONTROL}/electronics_enclosure.scad", {"enclosure_part": "gasket"}),
    Piece("carrier_pi5", f"{CONTROL}/board_carriers.scad", _carrier("pi5")),
    Piece("carrier_alta", f"{CONTROL}/board_carriers.scad", _carrier("alta")),
    Piece("carrier_pca9685", f"{CONTROL}/board_carriers.scad", _carrier("pca9685"), quantity=2),
    Piece("carrier_pololu_2806", f"{CONTROL}/board_carriers.scad", _carrier("pololu_2806")),
    Piece("carrier_hiletgo", f"{CONTROL}/board_carriers.scad", _carrier("hiletgo"), quantity=2),
    Piece("carrier_converter", f"{CONTROL}/board_carriers.scad", _carrier("converter"), quantity=4),
    Piece("carrier_harness_clamp", f"{CONTROL}/board_carriers.scad", _carrier("harness_clamp")),
)

PARTS_77603: tuple[Piece, ...] = (
    MOUNT_PIECES
    + ROD_PIECES
    + U_PIECES
    + VARIANT_PIECES
    + CONTROL_PIECES
)

# ---- Scene list ----------------------------------------------------------------------
# Anchor of the throttle station, mirroring the OpenSCAD derivation
# (throttle_77603/parameters.scad + ../enclosure_common.scad). The variant cameras aim
# here; aiming at the origin produced ten identical 5.8 kB blank frames. The throttle servo
# stands on the derivative head's shelf, BELOW the raised right steering servo:
#   servo x        = arm_joint_x + throttle_seat_x_local_mm                  = 253.30
#   servo y        = assembly_arm_y + throttle_seat_y_local_mm               = 123.00
#   horn underside = arm_joint_z + throttle_seat_local_mm
#                    + mg995_body_height_mm (= throttle_horn_base_mm())      = 410.58
# and the demonstration throttle end sits about a metre beyond the servo toward the
# lap-bar estimate, so the overview camera frames the whole chain, not just the station.
_SERVO_X = 253.30
_SERVO_Y = 123.00
_HORN_BASE_Z = 410.58


def _anchor() -> tuple[float, float, float]:
    """World position of the retained horn's underside plane on the shaft axis
    (throttle_horn_base_mm() in mg995_throttle_mount.scad), shared by every variant
    camera. The horn variants' frame contract fixes z = 0 here."""
    return (_SERVO_X, _SERVO_Y, _HORN_BASE_Z)


def _scene_variants() -> tuple[Scene, ...]:
    scenes = []
    ax, ay, az = _anchor()
    # The five horn interfaces sit just above the horn datum; aim there (the previous
    # framing centred 10 mm above the drawn horn for the same reason).
    for letter, dist in (("A", 240), ("B", 240), ("C", 240), ("D", 220), ("E", 240)):
        cam = f"{ax:.0f},{ay:.0f},{az + 10:.0f},60,0,35,{dist}"
        # The exploded scene lifts the horn interface 60 mm and the rod 96 mm, so its
        # camera aims above the datum and stands farther back to keep them in frame.
        cam_exploded = f"{ax:.0f},{ay:.0f},{az + 60:.0f},60,0,35,{dist + 140}"
        scenes.append(
            Scene(
                f"throttle_{letter.lower()}_assembled",
                f"{THROTTLE}/assemblies.scad",
                {"assembly": f"throttle_{letter.lower()}"},
                cam,
            )
        )
        scenes.append(
            Scene(
                f"throttle_{letter.lower()}_exploded",
                f"{THROTTLE}/assemblies.scad",
                {"assembly": f"throttle_{letter.lower()}", "exploded_mm": 60},
                cam_exploded,
            )
        )
        scenes.append(
            Scene(
                f"throttle_{letter.lower()}_drawing",
                f"{THROTTLE}/drawings.scad",
                {"horn_variant": letter},
                cam,
                kind="drawing",
            )
        )
    return tuple(scenes)


VIEWS_77603: tuple[Scene, ...] = (
    # The station plus its demonstration chain (tower to throttle end) spans about a
    # metre toward the lap-bar estimate, so the overview frames the whole run.
    Scene(
        "throttle_station",
        f"{THROTTLE}/assemblies.scad",
        {"assembly": "throttle_station"},
        "330,660,500,55,0,20,2400",
    ),
    Scene(
        "throttle_mount_service",
        f"{THROTTLE}/assemblies.scad",
        {"assembly": "throttle_mount_service"},
        f"{_anchor()[0]:.0f},{_anchor()[1] - 60:.0f},{_anchor()[2] - 60:.0f},62,0,18,600",
    ),
    Scene(
        "rod_collet",
        f"{THROTTLE}/throttle_pushrod.scad",
        _rod("assembly"),
        "16,0,832,60,0,35,260",
    ),
    # The collet camera aims at the threaded collet station measured from the rendered
    # assembly: tube axis at rod-local x = 16, nut and nose between z = 801 and 859 (the
    # finger tips at 857); the rod's origin is the servo eye, about 830 mm away.
    Scene(
        "throttle_u_fitting",
        f"{THROTTLE}/throttle_u_fitting.scad",
        _u("assembly"),
        "0,0,31,55,0,25,200",
    ),
    Scene(
        "electronics_packaging",
        f"{CONTROL}/assemblies.scad",
        {"assembly": "electronics_packaging"},
        "0,0,0,55,0,30,1400",
    ),
    Scene(
        "electronics_exploded",
        f"{CONTROL}/assemblies.scad",
        {"assembly": "electronics_packaging", "exploded_mm": 120},
        "0,0,0,55,0,30,1600",
    ),
    *_scene_variants(),
)

# ---- CSG probes ---------------------------------------------------------------------
PROBES_77603: tuple[Probe, ...] = (
    # Each entry names a branch that ACTUALLY EXISTS in throttle_77603/fit_checks.scad.
    # A label that matches no branch renders nothing, so the expectation would be tested
    # against an empty scene -- a false pass. Keep this list in lockstep with the file.
    #
    # U fitting: a lever that fits meets NO material, so pass-tests are "empty".
    Probe(
        "u_lever_passes",
        "empty",
        ({"check": "u_lever_passes"},),
        note="nominal lever section passes the specified opening",
    ),
    Probe("u_lever_passes_nominal", "empty", ({"check": "u_lever_passes_nominal"},)),
    Probe(
        "u_oversize_rejected",
        "contact",
        ({"check": "u_oversize_rejected"},),
        note="oversize lever is REFUSED: it meets wall material",
    ),
    Probe(
        "u_lever_passes_lean",
        "empty",
        ({"check": "u_lever_passes_lean"},),
        note="the whole cavity rotates with a lean; the lever still passes",
    ),
    Probe("u_clearance_applied", "empty", ({"check": "u_clearance_applied"},)),
    # Print-in-place U yoke: free through its range, and captive in every direction.
    Probe(
        "u_yoke_fit",
        "empty",
        ({"check": "u_yoke_fit"},),
        note="the printed-in-place yoke clears the body as printed",
    ),
    Probe(
        "u_yoke_pitch_range",
        "empty",
        tuple({"check": "u_yoke_pitch_range", "u_yoke_angle": a} for a in (-30, -15, 15, 30)),
        note="the yoke pitches through +/-joint_articulation_deg without contact",
    ),
    Probe(
        "u_yoke_captive",
        "contact",
        tuple(
            {"check": "u_yoke_captive", "u_yoke_shift": shift}
            for shift in (
                [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]
            )
        ),
        note="pushing the yoke 1 mm along any axis meets the body: it cannot leave",
    ),
    # Station: the shelf, corbel and cradle walls of the derivative right head, under the
    # raised steering servo. Branch names below match throttle_77603/fit_checks.scad
    # exactly; an unknown label fails the dispatcher's final assert.
    Probe(
        "shelf_fuses_to_rib",
        "contact",
        ({"check": "shelf_fuses_to_rib"},),
        note="the shelf overlaps the legacy head's rib: one printed piece, not two parts "
        "that happen to touch",
    ),
    Probe(
        "cradle_fuses_to_shelf",
        "contact",
        ({"check": "cradle_fuses_to_shelf"},),
        note="the cradle walls root into the shelf's slab",
    ),
    Probe(
        "shelf_supports_body",
        "contact",
        ({"check": "shelf_supports_body"},),
        note="head material directly under the servo body's footprint",
    ),
    Probe(
        "support_clears_servo_passages",
        "empty",
        ({"check": "support_clears_servo_passages"},),
        note="added material does not fill the steering servo's six holes or rear opening",
    ),
    Probe(
        "joint_bolts_have_tool_room",
        "empty",
        ({"check": "joint_bolts_have_tool_room"},),
        note="a 17 mm socket and 55 mm extension reach all four joint bolts past the shelf",
    ),
    Probe(
        "mount_vs_steering",
        "empty",
        ({"check": "mount_vs_steering"},),
        note="head, keeper stack and servo body clear the raised steering servo's case, "
        "moving bracket and adapter at 0/45/90 deg and its pushrod sweep",
    ),
    Probe(
        "linkage_vs_steering",
        "empty",
        tuple(
            {"check": "linkage_vs_steering", "link_variant": v, "link_sweep_deg": s}
            for v in "ABCDE"
            for s in (-30, 0, 30)
        ),
        note="each horn interface and rod servo section, swept +/-30 deg, clears the "
        "steering servo, its adapter and its pushrod sweep",
    ),
    Probe(
        "linkage_vs_head",
        "empty",
        tuple(
            {"check": "linkage_vs_head", "link_variant": v, "link_sweep_deg": s}
            for v in "ABCDE"
            for s in (-30, 0, 30)
        ),
        note="each horn interface and rod servo section, swept +/-30 deg, clears the "
        "head's rib, land, shelf and cradle walls",
    ),
    Probe(
        "rod_vs_steering",
        "empty",
        tuple({"check": "rod_vs_steering", "link_variant": v} for v in "ABCDE"),
        note="each variant's whole throttle rod, servo eye to throttle eye, clears the raised "
        "steering servo, its adapter and the falling steering pushrod it crosses in plan",
        # Builds the whole threaded rod and three full steering pushrods per variant: about
        # seven minutes single-threaded, longer under the build's concurrent renders.
        timeout_s=1800,
    ),
    Probe(
        "servo_lifts_out",
        "empty",
        ({"check": "servo_lifts_out"},),
        note="with its horn off, the MG995 rises out of the cradle without meeting the "
        "head or the steering servo: serviced without removing the steering servo",
    ),
    Probe(
        "keeper_lifts_out",
        "empty",
        ({"check": "keeper_lifts_out"},),
        note="with its pins drawn, the keeper rises clear of the walls below the steering "
        "servo",
    ),
    # Servo retention geometry. Contact probes prove geometric interference only: printed
    # pin strength, clip retention and vibration resistance remain bench-test items.
    Probe(
        "keeper_bears_on_lugs",
        "contact",
        ({"check": "keeper_bears_on_lugs"},),
        note="keeper material directly above the provisional lug band",
    ),
    Probe(
        "keeper_fits_cradle",
        "empty",
        ({"check": "keeper_fits_cradle"},),
        note="installed keeper does not interfere with the head's cradle walls",
    ),
    Probe(
        "keeper_pins_fit",
        "empty",
        ({"check": "keeper_pins_fit"},),
        note="installed keeper pins and rings clear the head and keeper",
    ),
    Probe(
        "keeper_pins_block_liftoff",
        "contact",
        ({"check": "keeper_pins_block_liftoff"},),
        note="lifting the keeper 1 mm meets both transverse pins",
    ),
    Probe(
        "keeper_pins_engage_cradle",
        "contact",
        ({"check": "keeper_pins_engage_cradle"},),
        note="the keeper pins bear in the head's side-wall bores",
    ),
    # Rod, collet, throttle-end body and joints.
    Probe(
        "rod_capture_pip_fit",
        "empty",
        ({"check": "rod_capture_pip_fit"},),
        note="as printed collet end down, the slider and sleeve are separate bodies with "
        "the integral lug inside the groove",
    ),
    Probe(
        "rod_collet_nut_clear",
        "empty",
        ({"check": "rod_collet_nut_clear"},),
        note="the captive collet nut clears the sleeve and the slider as printed",
    ),
    Probe(
        "rod_collet_nut_captive",
        "contact",
        tuple({"check": "rod_collet_nut_captive", "rod_collet_shift_mm": s} for s in (1, -1)),
        note="pushed 1 mm along the axis either way without turning, the nut meets the "
        "band's thread flanks",
    ),
    Probe(
        "rod_collet_nut_runs_free",
        "empty",
        tuple(
            {"check": "rod_collet_nut_runs_free", "rod_collet_advance_mm": a}
            for a in (1.2, -1.2)
        ),
        note="screwed 1.2 mm toward the tip or back, the nut still clears the sleeve: "
        "co-helical threads, cone contact only after 1.46 mm",
    ),
    Probe(
        "rod_collet_nut_drives_fingers",
        "contact",
        ({"check": "rod_collet_nut_drives_fingers"},),
        note="screwed forward by the clamp advance, the nut's cone overlaps the undeflected "
        "fingers, so it must close them onto the slider",
    ),
    Probe(
        "rod_capture_range_clear",
        "empty",
        tuple(
            {"check": "rod_capture_range_clear", "rod_capture_length_mm": length}
            for length in (940, 1066.8, 1200)
        ),
        note="the sleeve, its lug and the collet nut clear the slider through minimum, "
        "nominal and maximum settings",
    ),
    Probe(
        "rod_capture_clear",
        "empty",
        tuple(
            {"check": "rod_capture_clear", "rod_capture_length_mm": length}
            for length in (940, 1066.8, 1200)
        ),
        note="closed groove shoulder clears the sleeve lug at minimum, nominal and maximum",
    ),
    Probe(
        "rod_overlap_capture",
        "contact",
        tuple(
            {"check": "rod_overlap_capture", "rod_capture_length_mm": length}
            for length in (940, 1066.8, 1200)
        ),
        note="forced overtravel drives the groove shoulder into the integral sleeve lug",
    ),
    Probe(
        "rod_overlap_capture_reverse",
        "contact",
        ({"check": "rod_overlap_capture_reverse", "rod_capture_length_mm": 940},),
        note="reverse overtravel at minimum span drives the upper closed groove end into "
        "the sleeve's integral lug",
    ),
    Probe(
        "rod_body_fit",
        "empty",
        ({"check": "rod_body_fit"},),
        note="throttle-end body, bar, keeper, fine screw, jam collar, fork and fine nut "
        "clear the slider and each other at both ends of the fine travel",
    ),
    Probe(
        "rod_body_bar_holds",
        "contact",
        tuple(
            {"check": "rod_body_bar_holds", "rod_eye_target": t, "rod_eye_shift_mm": s}
            for t in ("body", "slider")
            for s in (1, -1)
        ),
        note="moving the body's bar 1 mm along the rod meets the body slot and the slider "
        "window, so the bar ties the body to the slider",
    ),
    Probe(
        "rod_body_keeper_stops_inward",
        "contact",
        ({"check": "rod_body_keeper_stops_inward"},),
        note="the keeper meets the body's face before the bar can walk inward",
    ),
    Probe(
        "rod_fine_screw_engages",
        "contact",
        ({"check": "rod_fine_screw_engages"},),
        note="the fine screw's flange and jam collar bear on the boss's two faces",
    ),
    Probe(
        "rod_body_clears_collet",
        "empty",
        tuple(
            {"check": "rod_body_clears_collet", "rod_capture_length_mm": length}
            for length in (940, 1066.8, 1200)
        ),
        note="the throttle-end body, fork and fine nut clear the sleeve and the fully "
        "advanced collet nut at minimum, nominal and maximum settings",
    ),
    Probe(
        "clevis_double_shear",
        "contact",
        tuple({"check": "clevis_double_shear", "clevis_ear_side": s} for s in (1, -1)),
        note="variant E's clevis has material around the hinge pin in each ear separately",
    ),
    # Section splices: the three section joints are retained by transverse bars.
    Probe(
        "splice_bars_fit",
        "empty",
        ({"check": "splice_bars_fit"},),
        note="every joint's bar and keeper clear the installed fixed chain",
    ),
    Probe(
        "splice_bar_holds_joint",
        "contact",
        tuple(
            {"check": "splice_bar_holds_joint", "splice_joint": j, "splice_shift_mm": s}
            for j in range(3)
            for s in (1, -1)
        ),
        note="moving a joint's bar 1 mm along the rod meets its boss and spigot slots",
    ),
    Probe(
        "splice_keeper_stops_inward",
        "contact",
        tuple({"check": "splice_keeper_stops_inward", "splice_joint": j} for j in range(3)),
        note="the keeper meets the boss before the bar can reach the slider passage",
    ),

    # Variant D: co-printed ball/socket clearance, angular travel and both socket mouths.
    Probe("d_ball_pip_fit", "empty", ({"check": "d_ball_pip_fit"},)),
    Probe(
        "d_ball_range_clear",
        "empty",
        tuple({"check": "d_ball_range_clear", "d_ball_angle_deg": a} for a in (-10, 10)),
        note="ball stem clears the carrier-integral socket through +/-10 degrees",
    ),
    Probe(
        "d_ball_captive",
        "contact",
        tuple(
            {"check": "d_ball_captive", "d_ball_shift_mm": shift}
            for shift in (
                [-3, 0, 0],
                [3, 0, 0],
                [0, -3, 0],
                [0, 3, 0],
                [0, 0, -3],
                [0, 0, 3],
            )
        ),
        note="the narrowed top and bottom mouths retain the ball against forced "
        "three-millimetre translation",
    ),

    Probe("c_shoe_pip_fit", "empty", ({"check": "c_shoe_pip_fit"},)),
    Probe(
        "c_shoe_range_clear",
        "empty",
        tuple({"check": "c_shoe_range_clear", "c_radius_mm": r} for r in (12, 18, 30)),
        note="the captured shoe slides freely at minimum, default and maximum radius",
    ),
    Probe(
        "c_shoe_captive",
        "contact",
        tuple(
            {"check": "c_shoe_captive", "c_radius_mm": r, "c_shoe_shift_mm": shift}
            for r in (12, 18, 30)
            for shift in ([0, 0, -1], [0, 0, 1], [0, -1, 0], [0, 1, 0])
        ),
        note="lower/upper dovetail flanks stop 1 mm vertical and transverse displacement",
    ),
    Probe(
        "c_shoe_end_stop",
        "contact",
        (
            {
                "check": "c_shoe_end_stop",
                "c_radius_mm": 12,
                "c_shoe_shift_mm": [-1, 0, 0],
            },
            {
                "check": "c_shoe_end_stop",
                "c_radius_mm": 30,
                "c_shoe_shift_mm": [1, 0, 0],
            },
        ),
        note="closed inboard and outboard rail ends block 1 mm travel past both radius limits",
    ),
    # Pin keepers: every split ring sits in a groove between its pin's end lip and the shaft.
    Probe(
        "pin_ring_seated_clear",
        "empty",
        tuple(
            {"check": "pin_ring_seated_clear", "pin_ring_kind": k}
            for k in ("shared", "index", "a_post", "e_hinge", "cross")
        ),
        note="each installed keeper ring clears its pin's groove, lip and shoulder",
    ),
    Probe(
        "pin_ring_held_by_lip",
        "contact",
        tuple(
            {"check": "pin_ring_held_by_lip", "pin_ring_kind": k}
            for k in ("shared", "index", "a_post", "e_hinge", "cross")
        ),
        note="pushing each installed ring 1 mm toward its pin's free end meets the end lip",
    ),
    # Variant C wedge lock. Interference is modelled as 0.05 mm per bearing face; CSG proves
    # where the faces bear, never the friction, holding force, wear or creep of the lock.
    Probe(
        "c_key_entry_clear",
        "empty",
        tuple({"check": "c_key_entry_clear", "c_radius_mm": r} for r in (12, 18, 30)),
        note="short of first contact the key touches only the rail seat it rests on",
    ),
    Probe(
        "c_key_bears_shoe",
        "contact",
        tuple({"check": "c_key_bears_shoe", "c_radius_mm": r} for r in (12, 18, 30)),
        note="driven home, the key's tapered gable bears on the shoe slot's matched roof",
    ),
    Probe(
        "c_key_bears_rail",
        "contact",
        tuple({"check": "c_key_bears_rail", "c_radius_mm": r} for r in (12, 18, 30)),
        note="driven home, the key's flat underside bears on the rail seat",
    ),
    Probe(
        "c_shoe_clamped",
        "contact",
        tuple({"check": "c_shoe_clamped", "c_radius_mm": r} for r in (12, 18, 30)),
        note="the key lifts the shoe's ramps onto the rail's 45-degree undercut faces",
    ),
    Probe(
        "c_lock_bearing_only",
        "empty",
        tuple(
            {"check": "c_lock_bearing_only", "c_radius_mm": r, "c_lock_stage": stage}
            for r in (12, 18, 30)
            for stage in (0.5, 1)
        ),
        note="mid-drive and home, all overlap lies in the seat, roof and undercut zones",
    ),
    Probe(
        "c_keeper_clear",
        "empty",
        tuple({"check": "c_keeper_clear", "c_radius_mm": r} for r in (12, 18, 30)),
        note="the keeper ring against the rail clears the driven key and lifted shoe",
    ),
    Probe(
        "c_keeper_stops_key",
        "contact",
        tuple({"check": "c_keeper_stops_key", "c_radius_mm": r} for r in (12, 18, 30)),
        note="withdrawing the key past the keeper slack meets the ring",
    ),
    Probe(
        "c_keeper_stops_on_rail",
        "contact",
        tuple({"check": "c_keeper_stops_on_rail", "c_radius_mm": r} for r in (12, 18, 30)),
        note="the ring cannot follow the key through the rail's -Y face",
    ),
    # Linkage joints, at each variant's demonstration pose. EMPTY probes prove the placed
    # parts clear; CONTACT probes prove a 1 mm displacement meets the retaining geometry.
    # Neither proves strength, wear or the real lever's arc.
    Probe(
        "linkage_servo_end_fit",
        "empty",
        tuple({"check": "linkage_servo_end_fit", "link_variant": v} for v in "ABCDE"),
        note="rod servo section clears the horn interface, mount and servo",
    ),
    Probe(
        "linkage_servo_eye_lift",
        "contact",
        tuple({"check": "linkage_servo_eye_lift", "link_variant": v} for v in "ABCDE"),
        note="lifting the servo eye 1 mm meets its cap or clevis",
    ),
    Probe(
        "linkage_servo_eye_pull",
        "contact",
        tuple({"check": "linkage_servo_eye_pull", "link_variant": v} for v in "ABCDE"),
        note="pulling the servo eye 1 mm along the run meets its pivot",
    ),
    Probe(
        "linkage_throttle_end_fit",
        "empty",
        tuple({"check": "linkage_throttle_end_fit", "link_variant": v} for v in "ABCDE"),
        note="rod moving end clears the U fitting on its pivot post",
    ),
    Probe(
        "linkage_throttle_eye_lift",
        "contact",
        tuple({"check": "linkage_throttle_eye_lift", "link_variant": v} for v in "ABCDE"),
        note="lifting the throttle eye 1 mm meets the U post's cap",
    ),
    Probe(
        "linkage_throttle_eye_pull",
        "contact",
        tuple({"check": "linkage_throttle_eye_pull", "link_variant": v} for v in "ABCDE"),
        note="pulling the throttle eye 1 mm back along the run meets the U post",
    ),
)

# ---- Inventory the packaging must show -----------------------------------------------
COMPONENT_INVENTORY = (
    ("raspberry_pi_5", 1, "owned", "primary computer"),
    ("libre_alta", 1, "owned", "hot standby"),
    ("pca9685", 2, "owned", "one per computer"),
    ("pololu_2806", 1, "owned assembled", "steering signal selection only"),
    ("hiletgo_relay", 2, "owned", "single-channel; one allocated to steering"),
    ("steering_converter", 2, "owned", "one per steering servo"),
    ("usb_c_c_converter", 2, "owned", "one per computer"),
    ("mg995", 1, "owned", "retained, not printed"),
    ("steering_servo_150kg", 2, "owned", "retained, not printed"),
    ("automotive_harness", 1, "owned, unverified", "relay bodies NOT confirmed"),
    ("pto_permission_contacts", 2, "1 candidate + 1 CONDITIONAL", "second contact not verified"),
    ("engine_stop_contacts", 2, "CONDITIONAL", "contact form and interface BLOCKED"),
)

MATERIAL_ASSUMPTIONS = {
    "PETG": "prototype/fit coupons and unpowered geometry; heat/UV/creep unqualified here",
    "ASA": "preferred for the enclosure if the printer/nozzle/enclosure are qualified",
    "PLA": "EXCLUDED for engine-bay service",
    "wall_layers_infill": "untested print parameters; record actual slicer settings per piece",
    "no_gcode": True,
}
