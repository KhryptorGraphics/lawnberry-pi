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

HORN_RADIUS_MIN_MM = 12.0   # mirrors horn_radius_min_mm in parameters.scad
HORN_RADIUS_MAX_MM = 30.0   # mirrors horn_radius_max_mm in parameters.scad
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
    positive retention/land probe that must NOT be empty.
    """

    check: str
    expect: str
    parameter_sets: tuple[dict[str, object], ...] = (field(default_factory=dict),)
    note: str = ""


def _t(part: str) -> dict[str, object]:
    # Generic "part" selector: enclosure, carriers.
    return {"part": part}


def _mount(part: str) -> dict[str, object]:
    # mg995_throttle_mount.scad reads the global `mount_part`.
    return {"mount_part": part}


def _rod(part: str, lock: str = "lead_screw") -> dict[str, object]:
    # throttle_pushrod.scad reads the globals `rod_part` / `rod_lock_kind`. Sending `part`
    # was silently ignored by OpenSCAD, so every rod/lock piece rendered the full assembly.
    return {"rod_part": part, "rod_lock_kind": lock}


def _u(part: str) -> dict[str, object]:
    return {"u_part": part}


def _variant(letter: str, part: str, **extra: object) -> dict[str, object]:
    # throttle_horn_variants.scad reads `horn_variant` (distinct from the module's own
    # `variant` parameter, so -D is not shadowed by a signature default).
    return {"horn_variant": letter, "horn_part": part, **extra}


# ---- Printable pieces ---------------------------------------------------------------
MOUNT_PIECES = (
    # Derivative of the existing RIGHT arm head: the original head plus the throttle column.
    # This is the piece the requirement asks for ("vertically extend and structurally
    # reinforce the existing right steering servo bracket"), not a bracket beside the design.
    Piece(
        "throttle_arm_head",
        f"{THROTTLE}/throttle_arm_head.scad",
        {},
        orientation="mating face down, as the legacy arm head prints",
        note="superset of arm_head_right: +141,100 mm3 (+47.1%) material, original servo land and holes kept",
    ),
    Piece(
        "riser_lower",
        f"{THROTTLE}/mg995_throttle_mount.scad",
        {**_mount("riser_lower"), "riser_index": 0},
        orientation="upright on its flange",
    ),
    Piece(
        "riser_upper",
        f"{THROTTLE}/mg995_throttle_mount.scad",
        {**_mount("riser_upper"), "riser_index": 1},
        orientation="upright on its flange",
    ),
    Piece("servo_cradle", f"{THROTTLE}/mg995_throttle_mount.scad", _mount("cradle")),
    Piece("servo_keeper", f"{THROTTLE}/mg995_throttle_mount.scad", _mount("keeper")),
    Piece("mg995_lug_gauge", f"{THROTTLE}/mg995_throttle_mount.scad", _mount("gauge")),
)

def _joint(kind: str) -> dict[str, object]:
    # printed_joints.scad reads the global `joint_part`.
    return {"joint_part": kind}


def joint_pieces() -> tuple[Piece, ...]:
    return (
        Piece("hinge_pin", f"{THROTTLE}/printed_joints.scad", _joint("pin"),
              quantity=6, orientation="upright on its head flange",
              note="captive printed hinge pin, used by variants A/B/C/D/E"),
        Piece("clevis", f"{THROTTLE}/printed_joints.scad", _joint("clevis"), quantity=2),
        Piece("bushing", f"{THROTTLE}/printed_joints.scad", _joint("bushing"),
              quantity=4, orientation="upright"),
        Piece("wear_socket", f"{THROTTLE}/printed_joints.scad", _joint("wear_socket"),
              quantity=2, orientation="upright"),
        Piece("endstop_collar", f"{THROTTLE}/printed_joints.scad", _joint("endstop_collar")),
        Piece("lock_nut", f"{THROTTLE}/printed_joints.scad", _joint("nut"),
              orientation="upright"),
        Piece("keyed_spigot", f"{THROTTLE}/printed_joints.scad", _joint("keyed_spigot"),
              quantity=3, orientation="upright"),
        Piece("trim_screw", f"{THROTTLE}/printed_joints.scad", _joint("screw"),
              orientation="upright, flange on the bed"),
        Piece("anti_backdrive_lip", f"{THROTTLE}/printed_joints.scad", _joint("anti_backdrive")),
        Piece("retainer_keeper", f"{THROTTLE}/printed_joints.scad", _joint("keeper"),
              quantity=8),
    )

ROD_PIECES = (
    # span 150 mm -> segment_count() == 1: a single printed rod piece, no splices. Longer
    # spans segment automatically; the count is derived, not fixed here.
    Piece("rod_segment", f"{THROTTLE}/throttle_pushrod.scad",
          _rod("segment_lower"), orientation="lying flat, long axis on the bed X",
          note="single piece for the 150 mm span; segmentation is automatic if the span grows"),
    Piece("rod_slider", f"{THROTTLE}/throttle_pushrod.scad", _rod("slider"),
          orientation="lying flat, long axis on the bed X"),
    Piece("rod_sleeve", f"{THROTTLE}/throttle_pushrod.scad", _rod("sleeve"),
          orientation="lying flat, long axis on the bed X"),
    Piece("splice_wedge", f"{THROTTLE}/throttle_pushrod.scad", _rod("splice_wedge"),
          quantity=0, note="only needed when segment_count() > 1; not used at the 150 mm span"),
    Piece("rod_fit_gauge", f"{THROTTLE}/throttle_pushrod.scad", _rod("gauge"),
          note="slider passes, bore does not: proves the sliding allowance"),
)

LOCK_PIECES = (
    Piece("lock_wedge_body", f"{THROTTLE}/throttle_pushrod.scad", _rod("lock_wedge_body")),
    Piece("lock_wedge_cam", f"{THROTTLE}/throttle_pushrod.scad", _rod("lock_wedge_cam")),
    Piece("lock_collet_body", f"{THROTTLE}/throttle_pushrod.scad",
          _rod("lock_collet_body"), orientation="upright"),
    Piece("lock_collet_collar", f"{THROTTLE}/throttle_pushrod.scad",
          _rod("lock_collet_collar"), orientation="upright"),
    Piece("lock_screw", f"{THROTTLE}/throttle_pushrod.scad", _rod("lock_screw"),
          orientation="upright, flange on the bed"),
    Piece("lock_jam_collar", f"{THROTTLE}/throttle_pushrod.scad", _rod("lock_jam_collar"),
          orientation="upright"),
)

U_PIECES = (
    Piece("throttle_u_body", f"{THROTTLE}/throttle_u_fitting.scad", _u("body")),
    Piece("throttle_u_neck", f"{THROTTLE}/throttle_u_fitting.scad", _u("neck")),
    Piece("throttle_u_gauge", f"{THROTTLE}/throttle_u_fitting.scad", _u("gauge")),
    Piece("throttle_u_pin", f"{THROTTLE}/throttle_u_fitting.scad", _u("pin"), quantity=2),
    Piece("throttle_u_retainer", f"{THROTTLE}/throttle_u_fitting.scad", _u("retainer")),
)

VARIANT_PIECES = (
    Piece("variant_a_eye", f"{THROTTLE}/throttle_horn_variants.scad",
          _variant("A", "eye"), orientation="upright"),
    Piece("variant_b_shell_a", f"{THROTTLE}/throttle_horn_variants.scad",
          _variant("B", "shell_a")),
    Piece("variant_b_shell_b", f"{THROTTLE}/throttle_horn_variants.scad",
          _variant("B", "shell_b")),
    Piece("variant_b_lug", f"{THROTTLE}/throttle_horn_variants.scad",
          _variant("B", "lug"), orientation="upright"),
    Piece("variant_c_carriage", f"{THROTTLE}/throttle_horn_variants.scad",
          _variant("C", "carriage")),
    Piece("variant_c_shoe", f"{THROTTLE}/throttle_horn_variants.scad", _variant("C", "shoe")),
    Piece("variant_d_ball_stem", f"{THROTTLE}/throttle_horn_variants.scad",
          _variant("D", "ball_stem"), orientation="upright on its base"),
    Piece("variant_d_cage", f"{THROTTLE}/throttle_horn_variants.scad",
          _variant("D", "cage"), orientation="upright"),
    Piece("variant_d_closure", f"{THROTTLE}/throttle_horn_variants.scad",
          _variant("D", "closure")),
    Piece("variant_e_clevis", f"{THROTTLE}/throttle_horn_variants.scad",
          _variant("E", "clevis")),
)

CONTROL_PIECES = (
    Piece("control_enclosure_body", f"{CONTROL}/electronics_enclosure.scad",
          {"enclosure_part": "body"}, orientation="open side up",
          note="printed in two stacked shells if taller than the usable Z"),
    Piece("control_enclosure_lid", f"{CONTROL}/electronics_enclosure.scad",
          {"enclosure_part": "lid"}),
    Piece("control_gasket", f"{CONTROL}/electronics_enclosure.scad",
          {"enclosure_part": "gasket"}),
    Piece("carrier_pi5", f"{CONTROL}/board_carriers.scad", {"carrier_kind": "pi5", "carrier_part": "body"}),
    Piece("carrier_alta", f"{CONTROL}/board_carriers.scad", {"carrier_kind": "alta", "carrier_part": "body"}),
    Piece("carrier_pca9685", f"{CONTROL}/board_carriers.scad", {"carrier_kind": "pca9685", "carrier_part": "body"}, quantity=2),
    Piece("carrier_pololu_2806", f"{CONTROL}/board_carriers.scad", {"carrier_kind": "pololu_2806", "carrier_part": "body"}),
    Piece("carrier_hiletgo", f"{CONTROL}/board_carriers.scad", {"carrier_kind": "hiletgo", "carrier_part": "body"}, quantity=2),
    Piece("carrier_converter", f"{CONTROL}/board_carriers.scad", {"carrier_kind": "converter", "carrier_part": "body"},
          quantity=4),
    Piece("carrier_harness_clamp", f"{CONTROL}/board_carriers.scad", {"carrier_kind": "harness_clamp", "carrier_part": "body"}),
)

PARTS_77603: tuple[Piece, ...] = (
    MOUNT_PIECES
    + joint_pieces()
    + ROD_PIECES
    + LOCK_PIECES
    + U_PIECES
    + VARIANT_PIECES
    + CONTROL_PIECES
)

# ---- Scene list ----------------------------------------------------------------------
# Anchor of the throttle station, mirroring parameters.scad. The variant cameras aim here;
# aiming at the origin produced ten identical 5.8 kB blank frames.
import math as _math

_SERVO_STATION_X = 209.55
_MG_WIDTH = 12.70
_OUTBOARD = 180.0
_TOWER_Y_OFFSET = 26.0
_ARM_TIP_Z = 430.24
_FOOT_T = 8.0
_RISER_Z = 120.0
_CRADLE_T = 8.0
_MG_BODY_H = 38.10
_MG_SHAFT = 12.70


def _anchor() -> tuple[float, float, float]:
    """World position of the MG995 horn, shared by every variant camera."""
    return (
        _SERVO_STATION_X + _MG_WIDTH / 2 + _OUTBOARD,
        123.0 + _TOWER_Y_OFFSET,
        _ARM_TIP_Z + _FOOT_T + _RISER_Z + _CRADLE_T + _MG_BODY_H + _MG_SHAFT,
    )


def _scene_variants() -> tuple[Scene, ...]:
    scenes = []
    ax, ay, az = _anchor()
    for letter, dist in (("A", 240), ("B", 240), ("C", 240), ("D", 220), ("E", 240)):
        cam = f"{ax:.0f},{ay:.0f},{az - 40:.0f},60,0,35,{dist}"
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
                cam,
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
    Scene("throttle_station", f"{THROTTLE}/assemblies.scad",
          {"assembly": "throttle_station"}, "160,120,430,55,0,20,1500"),
    Scene("throttle_mount_service", f"{THROTTLE}/assemblies.scad",
          {"assembly": "throttle_mount_service"},
          f"{_anchor()[0]:.0f},{_anchor()[1] - 60:.0f},{_anchor()[2] - 60:.0f},62,0,18,600"),
    Scene("lock_wedge_cam", f"{THROTTLE}/throttle_pushrod.scad",
          _rod("assembly", "wedge_cam"), "0,0,0,60,0,35,320"),
    Scene("lock_split_collet", f"{THROTTLE}/throttle_pushrod.scad",
          _rod("assembly", "split_collet"), "0,0,0,60,0,35,280"),
    Scene("lock_lead_screw", f"{THROTTLE}/throttle_pushrod.scad",
          _rod("assembly", "lead_screw"), "0,0,0,60,0,35,280"),
    Scene("throttle_u_fitting", f"{THROTTLE}/throttle_u_fitting.scad",
          _u("body"), "0,0,0,55,0,25,180"),
    Scene("electronics_packaging", f"{CONTROL}/assemblies.scad",
          {"assembly": "electronics_packaging"}, "0,0,0,55,0,30,1400"),
    Scene("electronics_exploded", f"{CONTROL}/assemblies.scad",
          {"assembly": "electronics_packaging", "exploded_mm": 120}, "0,0,0,55,0,30,1600"),
    *_scene_variants(),
)

# ---- CSG probes ---------------------------------------------------------------------
PROBES_77603: tuple[Probe, ...] = (
    # Each entry names a branch that ACTUALLY EXISTS in throttle_77603/fit_checks.scad.
    # A label that matches no branch renders nothing, so the expectation would be tested
    # against an empty scene -- a false pass. Keep this list in lockstep with the file.
    #
    # U fitting: a lever that fits meets NO material, so pass-tests are "empty".
    Probe("u_lever_passes", "empty", ({"check": "u_lever_passes"},),
          note="nominal lever section passes the specified opening"),
    Probe("u_lever_passes_nominal", "empty", ({"check": "u_lever_passes_nominal"},)),
    Probe("u_oversize_rejected", "contact", ({"check": "u_oversize_rejected"},),
          note="oversize lever is REFUSED: it meets wall material"),
    Probe("u_lever_passes_lean", "empty", ({"check": "u_lever_passes_lean"},),
          note="the whole cavity rotates with a lean; the lever still passes"),
    Probe("u_clearance_applied", "empty", ({"check": "u_clearance_applied"},)),
    # Mount: anchored to the right steering servo bracket.
    Probe("saddle_root_land", "contact", ({"check": "saddle_root_land"},),
          note="bracket carries material at the steering-servo face it bolts to"),
    Probe("mount_vs_linkage", "empty", ({"check": "mount_vs_linkage"},),
          note="throttle stack stays clear of the steering pushrod sweep"),
    Probe("mount_vs_steering", "empty", ({"check": "mount_vs_steering"},),
          note="throttle stack stays clear of the steering servo case and horn sweep"),
    # Servo retention.
    Probe("cradle_lug_capture", "contact", ({"check": "cradle_lug_capture"},)),
    Probe("keeper_capture", "contact", ({"check": "keeper_capture"},)),
    # Rod, locks and joints.
    Probe("rod_overlap_capture", "contact", ({"check": "rod_overlap_capture"},)),
    Probe("lock_wedge_grip", "contact", ({"check": "lock_wedge_grip"},)),
    Probe("lock_collet_grip", "contact", ({"check": "lock_collet_grip"},)),
    Probe("lock_screw_engagement", "contact", ({"check": "lock_screw_engagement"},)),
    Probe("endstop_capture", "contact", ({"check": "endstop_capture"},),
          note="anti-separation capture with every lock released"),
    Probe("clevis_double_shear", "contact", ({"check": "clevis_double_shear"},)),
    Probe("ball_cage_capture", "contact", ({"check": "ball_cage_capture"},)),
    Probe("c_radius_slide", "contact", ({"check": "c_radius_slide"},),
          note="continuous horn radius: the shoe sits at a non-default radius"),
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
