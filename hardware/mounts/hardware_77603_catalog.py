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


def _t(part: str) -> dict[str, object]:
    # Generic "part" selector: enclosure, carriers.
    return {"part": part}


def _mount(part: str) -> dict[str, object]:
    # mg995_throttle_mount.scad reads the global `mount_part`.
    return {"mount_part": part}


def _carrier(kind: str) -> dict[str, object]:
    # board_carriers.scad reads the globals `carrier_kind` / `carrier_part`.
    return {"carrier_kind": kind, "carrier_part": "body"}


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
    # Derivative of the existing RIGHT arm head: the legacy head plus ONE integral printed
    # support in the free Y band. It replaces arm_head_right for the throttle build.
    Piece(
        "throttle_arm_head",
        f"{THROTTLE}/throttle_arm_head.scad",
        {},
        orientation="mating face down, as the legacy arm head prints",
        note="derivative of arm_head_right; integral column carries the servo cradle "
        "directly (no riser); fit and joint strength unverified",
    ),
    Piece("servo_cradle", f"{THROTTLE}/mg995_throttle_mount.scad", _mount("cradle")),
    Piece("servo_keeper", f"{THROTTLE}/mg995_throttle_mount.scad", _mount("keeper")),
    Piece(
        "servo_seat_pin",
        f"{THROTTLE}/mg995_throttle_mount.scad",
        _mount("seat_pin"),
        orientation="upright on its head",
        note="transverse pin through the column and both cradle dowels; fit, clip "
        "retention and vibration life unverified",
    ),
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
        quantity=3,
        orientation="flat",
        note="split ring for the seat pin and both keeper pins; fatigue unverified",
    ),
    Piece("mg995_lug_gauge", f"{THROTTLE}/mg995_throttle_mount.scad", _mount("gauge")),
)


def _joint(kind: str) -> dict[str, object]:
    # printed_joints.scad reads the global `joint_part`.
    return {"joint_part": kind}


def joint_pieces() -> tuple[Piece, ...]:
    return (
        Piece(
            "hinge_pin",
            f"{THROTTLE}/printed_joints.scad",
            _joint("pin"),
            quantity=6,
            orientation="upright on its head flange",
            note="captive printed hinge pin for the U-neck and U-body joints; the rod's "
            "clevis pin and the five horn variants' index pins are their own parts",
        ),
        Piece("clevis", f"{THROTTLE}/printed_joints.scad", _joint("clevis"), quantity=2),
        Piece(
            "bushing",
            f"{THROTTLE}/printed_joints.scad",
            _joint("bushing"),
            quantity=4,
            orientation="upright",
        ),
        Piece(
            "wear_socket",
            f"{THROTTLE}/printed_joints.scad",
            _joint("wear_socket"),
            quantity=2,
            orientation="upright",
        ),
        Piece("endstop_collar", f"{THROTTLE}/printed_joints.scad", _joint("endstop_collar")),
        # Named for the shared joint library, not the rod's own lead-screw nut below:
        # both are printed nuts, and the flat export needs distinct names.
        Piece(
            "joint_lock_nut",
            f"{THROTTLE}/printed_joints.scad",
            _joint("nut"),
            orientation="upright",
        ),
        Piece(
            "keyed_spigot",
            f"{THROTTLE}/printed_joints.scad",
            _joint("keyed_spigot"),
            quantity=3,
            orientation="upright",
        ),
        Piece(
            "trim_screw",
            f"{THROTTLE}/printed_joints.scad",
            _joint("screw"),
            orientation="upright, flange on the bed",
        ),
        Piece("anti_backdrive_lip", f"{THROTTLE}/printed_joints.scad", _joint("anti_backdrive")),
        Piece("retainer_keeper", f"{THROTTLE}/printed_joints.scad", _joint("keeper"), quantity=8),
        # The shared printed_joints nut is exported as joint_lock_nut so the flat export
        # does not collide with the rod's own lead-screw lock_nut below.
        Piece("alignment_yoke", f"{THROTTLE}/printed_joints.scad", _joint("alignment_yoke")),
        Piece(
            "alignment_cross_block", f"{THROTTLE}/printed_joints.scad", _joint("alignment_cross")
        ),
    )


ROD_PIECES = (
    # Selector names match throttle_pushrod.scad's dispatcher exactly. The rod's own
    # "assembly" composes the sections, slider, capture hardware, fine stage and throttle
    # end; EVERY distinct printed part it composes is exported here, so the owner can
    # print the linkage from this catalog alone.
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
        "rod_slider",
        f"{THROTTLE}/throttle_pushrod.scad",
        _rod("slider"),
        orientation="lying flat, long axis on the bed X",
    ),
    Piece(
        "rod_sleeve",
        f"{THROTTLE}/throttle_pushrod.scad",
        _rod("sleeve"),
        orientation="lying flat, long axis on the bed X",
    ),
    Piece(
        "rod_capture_pin",
        f"{THROTTLE}/throttle_pushrod.scad",
        _rod("capture_pin"),
        orientation="upright on its head flange",
    ),
    Piece(
        "rod_capture_keeper",
        f"{THROTTLE}/throttle_pushrod.scad",
        _rod("capture_keeper"),
        orientation="upright",
    ),
    Piece("rod_spigot", f"{THROTTLE}/throttle_pushrod.scad", _rod("spigot"), orientation="upright"),
    Piece(
        "splice_wedge",
        f"{THROTTLE}/throttle_pushrod.scad",
        _rod("splice_wedge"),
        quantity=0,
        note="only needed when the span segments; count follows the span",
    ),
    Piece(
        "splice_keeper",
        f"{THROTTLE}/throttle_pushrod.scad",
        _rod("splice_keeper"),
        quantity=0,
        note="only needed when the span segments; count follows the span",
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
    # The rod's own anti-separation end stop. This is NOT printed_joints' endstop_collar:
    # that is the collar that clamps onto the sliding member, and both are exported.
    Piece(
        "rod_endstop", f"{THROTTLE}/throttle_pushrod.scad", _rod("endstop"), orientation="upright"
    ),
    Piece("rod_overstop_lug", f"{THROTTLE}/throttle_pushrod.scad", _rod("overstop_lug")),
    Piece("rod_overstop_stop", f"{THROTTLE}/throttle_pushrod.scad", _rod("overstop_stop")),
    # The rod's own clevis joint at the throttle end: ear pair, its bushing and its pin.
    # These are the rod's SELECTORS, distinct from the shared printed_joints pieces.
    Piece("rod_clevis", f"{THROTTLE}/throttle_pushrod.scad", _rod("clevis")),
    Piece(
        "rod_clevis_bushing",
        f"{THROTTLE}/throttle_pushrod.scad",
        _rod("clevis_bushing"),
        orientation="upright",
    ),
    Piece(
        "rod_clevis_pin",
        f"{THROTTLE}/throttle_pushrod.scad",
        _rod("clevis_pin"),
        orientation="upright on its head flange",
    ),
    Piece(
        "rod_fit_gauge",
        f"{THROTTLE}/throttle_pushrod.scad",
        _rod("gauge"),
        note="slider passes, bore does not: proves the sliding allowance",
    ),
)

LOCK_PIECES = (
    # The three coarse locks are ALTERNATIVES: a builder fits ONE family (wedge_cam,
    # split_collet or lead_screw), not all three. All three are exported so each concept
    # is independently manufacturable; the build notes must say which one was fitted.
    # Each lock's own keeper is exported with it; the shared printed_joints pieces are
    # catalogued separately.
    Piece("lock_wedge_body", f"{THROTTLE}/throttle_pushrod.scad", _rod("lock_wedge_body")),
    Piece("lock_wedge", f"{THROTTLE}/throttle_pushrod.scad", _rod("lock_wedge")),
    Piece("lock_wedge_cam", f"{THROTTLE}/throttle_pushrod.scad", _rod("lock_wedge_cam")),
    Piece("lock_wedge_keeper", f"{THROTTLE}/throttle_pushrod.scad", _rod("lock_wedge_keeper")),
    Piece(
        "lock_collet_body",
        f"{THROTTLE}/throttle_pushrod.scad",
        _rod("lock_collet_body"),
        orientation="upright",
    ),
    Piece(
        "lock_collet_collar",
        f"{THROTTLE}/throttle_pushrod.scad",
        _rod("lock_collet_collar"),
        orientation="upright",
    ),
    Piece(
        "lock_collet_keeper",
        f"{THROTTLE}/throttle_pushrod.scad",
        _rod("lock_collet_keeper"),
        orientation="upright",
    ),
    Piece(
        "lock_screw",
        f"{THROTTLE}/throttle_pushrod.scad",
        _rod("lock_screw"),
        orientation="upright, flange on the bed",
    ),
    Piece(
        "lock_nut",
        f"{THROTTLE}/throttle_pushrod.scad",
        _rod("lock_nut"),
        orientation="upright",
        note="the lead-screw lock's own nut; distinct from the "
        "shared printed_joints nut (joint_lock_nut)",
    ),
    Piece("lock_anti_backdrive", f"{THROTTLE}/throttle_pushrod.scad", _rod("lock_anti_backdrive")),
    Piece(
        "lock_jam_collar",
        f"{THROTTLE}/throttle_pushrod.scad",
        _rod("lock_jam_collar"),
        orientation="upright",
    ),
)

U_PIECES = (
    Piece("throttle_u_body", f"{THROTTLE}/throttle_u_fitting.scad", _u("body")),
    Piece("throttle_u_neck", f"{THROTTLE}/throttle_u_fitting.scad", _u("neck")),
    Piece("throttle_u_gauge", f"{THROTTLE}/throttle_u_fitting.scad", _u("gauge")),
    Piece(
        "throttle_u_pin",
        f"{THROTTLE}/throttle_u_fitting.scad",
        _u("pin"),
        note="the neck's lower Y-axis pin",
    ),
    Piece(
        "throttle_u_cap",
        f"{THROTTLE}/throttle_u_fitting.scad",
        _u("cap"),
        orientation="flat",
        note="traps the rod's throttle eye on the neck's pivot post",
    ),
    Piece("throttle_u_retainer", f"{THROTTLE}/throttle_u_fitting.scad", _u("retainer")),
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
    Piece("variant_c_carrier", f"{THROTTLE}/throttle_horn_variants.scad", _variant("C", "carrier")),
    Piece("variant_c_shoe", f"{THROTTLE}/throttle_horn_variants.scad", _variant("C", "shoe")),
    Piece(
        "variant_c_wedge_key", f"{THROTTLE}/throttle_horn_variants.scad", _variant("C", "wedge_key")
    ),
    Piece(
        "variant_c_wedge_keeper",
        f"{THROTTLE}/throttle_horn_variants.scad",
        _variant("C", "wedge_keeper"),
    ),
    Piece("variant_c_cap", f"{THROTTLE}/throttle_horn_variants.scad", _variant("C", "cap")),
    Piece(
        "variant_c_pin",
        f"{THROTTLE}/throttle_horn_variants.scad",
        _variant("C", "pin"),
        orientation="upright",
    ),
    Piece("variant_c_keeper", f"{THROTTLE}/throttle_horn_variants.scad", _variant("C", "keeper")),
    Piece("variant_d_carrier", f"{THROTTLE}/throttle_horn_variants.scad", _variant("D", "carrier")),
    Piece(
        "variant_d_ball_stem",
        f"{THROTTLE}/throttle_horn_variants.scad",
        _variant("D", "ball_stem"),
        orientation="upright on its base",
    ),
    Piece(
        "variant_d_cage",
        f"{THROTTLE}/throttle_horn_variants.scad",
        _variant("D", "cage"),
        orientation="upright",
    ),
    Piece(
        "variant_d_wear_socket",
        f"{THROTTLE}/throttle_horn_variants.scad",
        _variant("D", "wear_socket"),
        orientation="upright",
    ),
    Piece("variant_d_closure", f"{THROTTLE}/throttle_horn_variants.scad", _variant("D", "closure")),
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
    + joint_pieces()
    + ROD_PIECES
    + LOCK_PIECES
    + U_PIECES
    + VARIANT_PIECES
    + CONTROL_PIECES
)

# ---- Scene list ----------------------------------------------------------------------
# Anchor of the throttle station, mirroring the OpenSCAD derivation
# (throttle_77603/parameters.scad + ../enclosure_common.scad). The variant cameras aim
# here; aiming at the origin produced ten identical 5.8 kB blank frames. The station is
# now carried by the head's integral column directly over the steering-servo land: there
# is no riser and no outboard cantilever, so the old 180 mm outboard anchor is gone.
#   tower_world_x  = servo_station_x + column_width / 2 - servo_face_t - 1 = 218.30
#   tower_world_y  = assembly_arm_y + seat_y_local_mm                     = 152.00
#   horn underside = arm_joint_z + column_seat_local_mm + cradle_t_mm
#                    + mg995_body_height_mm (= throttle_horn_base_mm())  = 566.58
# and the demonstration throttle end sits about a metre beyond the tower toward the
# lap-bar estimate, so the overview camera frames the whole chain, not just the tower.
_TOWER_X = 218.30
_TOWER_Y = 152.00
_HORN_BASE_Z = 566.58


def _anchor() -> tuple[float, float, float]:
    """World position of the retained horn's underside plane on the shaft axis
    (throttle_horn_base_mm() in mg995_throttle_mount.scad), shared by every variant
    camera. The horn variants' frame contract fixes z = 0 here."""
    return (_TOWER_X, _TOWER_Y, _HORN_BASE_Z)


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
        "lock_wedge_cam",
        f"{THROTTLE}/throttle_pushrod.scad",
        _rod("assembly", "wedge_cam"),
        "29,0,852,60,0,35,260",
    ),
    Scene(
        "lock_split_collet",
        f"{THROTTLE}/throttle_pushrod.scad",
        _rod("assembly", "split_collet"),
        "20,0,852,60,0,35,240",
    ),
    Scene(
        "lock_lead_screw",
        f"{THROTTLE}/throttle_pushrod.scad",
        _rod("assembly", "lead_screw"),
        "22,0,867,60,0,35,300",
    ),
    # The lock cameras aim at the lock band (fixed chain 664 + sleeve-local 160..205,
    # crank 16 in X), measured from the rendered lock assemblies; the rod's origin is
    # the servo eye, about 850 mm away.
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
    # Mount: anchored to the right steering servo bracket. Branch names below match
    # throttle_77603/fit_checks.scad exactly; a label that matches no branch is a false
    # pass against an empty scene, which is why saddle_root_land (a branch that no longer
    # exists) was removed.
    Probe(
        "support_bears_on_land",
        "contact",
        ({"check": "support_bears_on_land"},),
        note="the integral column grows out of the arm head's servo-face land: one "
        "printed piece, not two parts that happen to touch",
    ),
    Probe(
        "seat_holes_are_bored",
        "contact",
        ({"check": "seat_holes_are_bored"},),
        note="the seat dowel holes are bored into real column material above the seat",
    ),
    Probe(
        "seat_dowels_fit",
        "empty",
        ({"check": "seat_dowels_fit"},),
        note="the cradle's locating dowels pass their holes without touching",
    ),
    Probe(
        "seat_cradle_clears_column",
        "empty",
        ({"check": "seat_cradle_clears_column"},),
        note="the seated cradle does not interfere with the column",
    ),
    Probe(
        "support_clears_servo_passages",
        "empty",
        ({"check": "support_clears_servo_passages"},),
        note="added material does not fill the steering servo's six holes or rear opening",
    ),
    Probe(
        "mount_vs_moving_steering",
        "empty",
        ({"check": "mount_vs_moving_steering"},),
        note="throttle stack stays clear of the steering servo's moving bracket at "
        "0/45/90 deg, not only at one pose",
    ),
    Probe(
        "mount_vs_linkage",
        "empty",
        ({"check": "mount_vs_linkage"},),
        note="throttle stack stays clear of the steering pushrod sweep",
    ),
    Probe(
        "mount_vs_steering",
        "empty",
        ({"check": "mount_vs_steering"},),
        note="throttle stack stays clear of the steering servo case and horn sweep",
    ),
    # Servo retention geometry. Contact probes prove geometric interference only: printed
    # pin strength, clip retention and vibration resistance remain bench-test items.
    Probe(
        "cradle_floor_supports_body",
        "contact",
        ({"check": "cradle_floor_supports_body"},),
        note="floor material directly under the servo body footprint",
    ),
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
        note="installed keeper does not interfere with the cradle",
    ),
    Probe(
        "seat_pin_fit",
        "empty",
        ({"check": "seat_pin_fit"},),
        note="installed seat pin and ring clear the column and cradle dowels",
    ),
    Probe(
        "seat_pin_blocks_liftoff",
        "contact",
        ({"check": "seat_pin_blocks_liftoff"},),
        note="lifting the cradle 1 mm meets the seat pin",
    ),
    Probe(
        "seat_pin_engages_column",
        "contact",
        ({"check": "seat_pin_engages_column"},),
        note="the seat pin bears in the column's cross bore",
    ),
    Probe(
        "keeper_pins_fit",
        "empty",
        ({"check": "keeper_pins_fit"},),
        note="installed keeper pins and rings clear the cradle and keeper",
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
        note="the keeper pins bear in the cradle side-wall bores",
    ),
    # Rod, locks and joints.
    Probe(
        "rod_capture_clear",
        "empty",
        tuple(
            {"check": "rod_capture_clear", "rod_capture_length_mm": length}
            for length in (940, 1066.8, 1200)
        ),
        note="capture pin and shoulder clear at both endpoints and nominal setting",
    ),
    Probe(
        "rod_overlap_capture",
        "contact",
        tuple(
            {"check": "rod_overlap_capture", "rod_capture_length_mm": length}
            for length in (940, 1066.8, 1200)
        ),
        note="forced overtravel meets the capture pin at each setting",
    ),
    Probe("lock_wedge_grip", "contact", ({"check": "lock_wedge_grip"},)),
    Probe("lock_collet_grip", "contact", ({"check": "lock_collet_grip"},)),
    Probe("lock_screw_engagement", "contact", ({"check": "lock_screw_engagement"},)),
    Probe(
        "endstop_capture",
        "contact",
        ({"check": "endstop_capture"},),
        note="end-stop collar material crosses the rod passage; anti-separation with "
        "every lock released is the design intent, not verified behaviour",
    ),
    Probe("clevis_double_shear", "contact", ({"check": "clevis_double_shear"},)),
    Probe("ball_cage_capture", "contact", ({"check": "ball_cage_capture"},)),
    Probe(
        "c_radius_slide",
        "contact",
        ({"check": "c_radius_slide"},),
        note="continuous horn radius: the shoe sits at a non-default radius",
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
