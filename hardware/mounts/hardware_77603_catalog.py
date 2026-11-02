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
    # OpenSCAD file, relative to hardware/mounts, that implements `check`.
    source: str = f"{THROTTLE}/fit_checks.scad"


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
        note="derivative of arm_head_right; integral shelf and cradle walls (four snap "
        "windows) carry the throttle MG995 below the raised steering servo; fit and joint "
        "strength unverified",
    ),
    Piece(
        "servo_keeper",
        f"{THROTTLE}/mg995_throttle_mount.scad",
        _mount("keeper"),
        orientation="frame on the bed, fingers up (as exported)",
        note="one-piece snap keeper: four spring fingers snap into the cradle windows and "
        "crush ribs press the ears; snap force, rib crush and vibration hold-down unverified",
    ),
    Piece("mg995_lug_gauge", f"{THROTTLE}/mg995_throttle_mount.scad", _mount("gauge")),
)


ROD_PIECES = (
    # Print-in-place throttle pushrod; its probes live in throttle_77603/rod_checks.scad. Four
    # prints carry nine captive bodies: two sections with joint levers, the capture set
    # (sleeve, slider, over-centre cam lever) and the fork with its detented thumbwheel. No
    # bars, keepers, screws or nuts. Clamp/holding force (185 N target) is unqualified: the
    # two coupons below are the bench test.
    Piece(
        "rod_segment_lower",
        f"{THROTTLE}/throttle_pushrod.scad",
        _rod("segment_lower"),
        orientation="upright, socket mouth down: boss mouth and joint lever on the bed; servo "
        "eye at the top",
        note="servo eye, 30 mm tube, 42 mm socket boss, print-in-place joint lever (open as "
        "printed)",
        bodies=2,
    ),
    Piece(
        "rod_segment_middle",
        f"{THROTTLE}/throttle_pushrod.scad",
        _rod("segment_middle"),
        orientation="upright, socket mouth down: boss mouth and joint lever on the bed; spigot "
        "at the top",
        note="34 mm square spigot with tooth pocket into rod_segment_lower; print-in-place "
        "joint lever on its own boss",
        bodies=2,
    ),
    Piece(
        "rod_capture_assembly",
        f"{THROTTLE}/throttle_pushrod.scad",
        _rod("capture_assembly"),
        orientation="upright, sleeve tip down: sleeve tip, slider tip and cam lever on the bed",
        note="sleeve (spigot, tube, cam housing, overtravel lug), slider (ticks, threaded tip), "
        "over-centre cam lever; holding unqualified (185 N target)",
        bodies=3,
    ),
    Piece(
        "rod_fork_assembly",
        f"{THROTTLE}/throttle_pushrod.scad",
        _rod("fork_assembly"),
        orientation="upright, wheel down; eye plate up",
        note="throttle-eye fork with captive detented thumbwheel: 0.25 mm per click, 20 mm "
        "range; screws onto the slider's thread",
        bodies=2,
    ),
    Piece(
        "rod_cam_coupon",
        f"{THROTTLE}/throttle_pushrod.scad",
        _rod("cam_coupon"),
        orientation="upright, tip down",
        note="bench coupon: 50 mm of the real cam housing, its lever and a square slider stub; "
        "pull to 185 N",
        bodies=3,
    ),
    Piece(
        "rod_joint_coupon",
        f"{THROTTLE}/throttle_pushrod.scad",
        _rod("joint_coupon"),
        orientation="upright, socket mouth down",
        note="bench coupon: 50 mm of the servo section's socket boss and its joint lever",
        bodies=2,
    ),
    Piece(
        "rod_joint_coupon_spigot",
        f"{THROTTLE}/throttle_pushrod.scad",
        _rod("joint_coupon_spigot"),
        orientation="upright, tube end down, spigot up",
        note="mating spigot slice with tooth pocket for rod_joint_coupon",
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
    # Everything inside the legacy enclosure body (box and lid unchanged). A removable chassis
    # is clamped under the box's four existing M4 bolt heads; every carrier hangs on its molded
    # cleats. No fasteners are added. Probes: control_77603/control_checks.scad.
    Piece(
        "chassis_base",
        f"{CONTROL}/board_carriers.scad",
        _carrier("chassis_base"),
        orientation="flat, hooks up",
        note="clamped on the four boss tops under the existing M4 bolt heads; 3.5 mm, as the "
        "legacy tray, so bolt grip length is unchanged",
    ),
    Piece(
        "chassis",
        f"{CONTROL}/board_carriers.scad",
        _carrier("chassis"),
        orientation="plate on the bed, panels vertical",
        note="167 x 120.5 x 151 rack with molded cleats; slides 6 mm -Y onto the base hooks "
        "and snaps a detent nub (press it with a pen tip to release)",
    ),
    Piece(
        "tower_rack",
        f"{CONTROL}/board_carriers.scad",
        _carrier("tower_rack"),
        orientation="shelf on the bed",
        note="Pi 5 stack (Active Cooler, AI HAT+, LC29H, HaLow HAT) below the Alta stack; "
        "locks on two mushroom studs plus a sprung nub",
    ),
    Piece(
        "tower_clamp",
        f"{CONTROL}/board_carriers.scad",
        _carrier("tower_clamp"),
        orientation="upside down, frame top on the bed",
        note="clamp frame pressing the stacks onto the rack; kit standoffs act as spacers only",
    ),
    Piece(
        "strap_bay",
        f"{CONTROL}/board_carriers.scad",
        _carrier("strap_bay"),
        quantity=6,
        orientation="upright as installed",
        note="print-in-place strap bar for the converters and relays (sizes unknown): box "
        "18-43.6 wide x 16.5-26.5 deep; strap and pawl holding unqualified",
        bodies=2,
    ),
    Piece(
        "harness_clamp",
        f"{CONTROL}/board_carriers.scad",
        _carrier("harness_clamp"),
        orientation="upright as installed",
        note="print-in-place strap bar for the automotive harness bundle, 8-26 mm",
        bodies=2,
    ),
    Piece(
        "ina_cradle",
        f"{CONTROL}/board_carriers.scad",
        _carrier("ina_cradle"),
        orientation="upside down, jaw lips and bar top on the bed",
        note="snaps onto the tower clamp frame; print-in-place ratchet jaw holds an INA3221 "
        "board flat (22-32 x 30-38.6, PCB 1.0-1.6: Adafruit #6062, CJMCU-3221, 30 x 26 "
        "breakouts) on 1.2 / 0.8 mm plastic lips kept >= 0.4 mm from its parts; holding "
        "force and vibration retention unqualified",
        bodies=2,
    ),
    Piece(
        "sma_clip",
        f"{CONTROL}/board_carriers.scad",
        _carrier("sma_clip"),
        orientation="upside down, plate top on the bed",
        note="snaps onto the tower clamp's front-right corner (jaws over both finger bands, leaf "
        "under the crossbar); holds the three tower-side SMA-F D-flat bulkhead jacks face-up for "
        "the lid's plugs (8 mm spanner); clamping force, torque capacity and vibration "
        "retention unqualified",
    ),
)

# Two HaLow antenna stations on the reprinted lid (the body is unchanged): the lid carries the
# bulkhead holes and blind post sockets; each post drops in and snaps, two grips per antenna.
# Print the grips for the MEASURED base-tube OD (25-40 mm).
ANTENNA_PIECES = (
    Piece(
        "antenna_post_left",
        "antenna_mount.scad",
        {"antenna_part": "post", "post_side": -1},
        orientation="lying on its back face, rails up, column along +Y",
        note="-X (Alta) station; spigot + flexure snap into the lid socket; mirror of the right "
        "post; snap retention, socket load and creep unqualified",
    ),
    Piece(
        "antenna_post_right",
        "antenna_mount.scad",
        {"antenna_part": "post", "post_side": 1},
        orientation="lying on its back face, rails up, column along +Y",
        note="+X (Pi) station; spigot + flexure snap into the lid socket; snap retention, socket "
        "load and creep unqualified",
    ),
    Piece(
        "antenna_grip",
        "antenna_mount.scad",
        {"antenna_part": "grip", "ant_tube_d_mm": 32},
        quantity=4,
        orientation="upright, bottom face on the bed (pull tab up)",
        note="print with ant_tube_d_mm = MEASURED base-tube OD (25-40; 32 is a photo estimate); "
        "same part at both stations; band/pawl retention, creep, vibration and UV unqualified",
    ),
)


PARTS_77603: tuple[Piece, ...] = (
    MOUNT_PIECES
    + ROD_PIECES
    + U_PIECES
    + VARIANT_PIECES
    + CONTROL_PIECES
    + ANTENNA_PIECES
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
        "rod_locks",
        f"{THROTTLE}/throttle_pushrod.scad",
        _rod("assembly"),
        "16,0,900,70,0,145,260",
        note="cam lever and coarse ticks on the capture set",
    ),
    Scene(
        "rod_joint",
        f"{THROTTLE}/throttle_pushrod.scad",
        _rod("assembly"),
        "16,0,585,70,0,145,240",
        note="middle-to-sleeve joint lever and tooth",
    ),
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
        "0,0,95,58,0,35,560",
    ),
    Scene(
        "electronics_contents",
        f"{CONTROL}/assemblies.scad",
        {"assembly": "electronics_contents"},
        "0,0,95,60,0,215,560",
    ),
    Scene(
        "electronics_exploded",
        f"{CONTROL}/assemblies.scad",
        {"assembly": "electronics_exploded"},
        "0,0,170,65,0,215,900",
    ),
    Scene(
        "electronics_lid_lifted",
        f"{CONTROL}/assemblies.scad",
        {"assembly": "electronics_lid_lifted"},
        "0,0,185,58,0,35,1080",
    ),
    Scene(
        "lid_antennas",
        "assemblies.scad",
        {"assembly": "lid_antennas"},
        "0,-40,230,62,0,330,650",
        note="lid cut on the -X station plane: both antenna stations, the GNSS SMA jack and the "
        "antennas' lower 300 mm",
    ),
    *_scene_variants(),
)

# ---- CSG probes ---------------------------------------------------------------------
_ANT = "antenna_checks.scad"
ANTENNA_PROBES: tuple[Probe, ...] = (
    Probe(
        "grip_contact",
        "contact",
        tuple({"check": "grip_contact", "probe_tube_d": d} for d in (25, 40)),
        note="crush ribs land on the tube, all four grips",
        source=_ANT,
    ),
    Probe(
        "grip_insertion_clear",
        "empty",
        tuple({"check": "grip_insertion_clear", "probe_tube_d": d} for d in (25, 40)),
        note="the tube slides through the band loops",
        source=_ANT,
    ),
    Probe(
        "grip_post_clear",
        "empty",
        ({"check": "grip_post_clear"},),
        note="grips (25 and 40) clear the post rails",
        source=_ANT,
    ),
    Probe(
        "grip_ledge_contact",
        "contact",
        ({"check": "grip_ledge_contact"},),
        note="lower grips seat on their ledges",
        source=_ANT,
    ),
    Probe(
        "grip_upper_ledge_contact",
        "contact",
        ({"check": "grip_upper_ledge_contact"},),
        note="upper grips seat on their ledges",
        source=_ANT,
    ),
    Probe(
        "post_socket_clear",
        "empty",
        ({"check": "post_socket_clear"},),
        note="the seated post clears its socket, recess and the lid",
        source=_ANT,
    ),
    Probe(
        "post_seat_contact",
        "contact",
        ({"check": "post_seat_contact"},),
        note="the column foot bears on the lid top",
        source=_ANT,
    ),
    Probe(
        "post_lock_retains",
        "contact",
        ({"check": "post_lock_retains"},),
        note="lifted 0.5 mm, the snap tooth meets the recess top",
        source=_ANT,
    ),
    Probe(
        "post_snap_deflect_clear",
        "empty",
        ({"check": "post_snap_deflect_clear"},),
        note="the leaf pressed in 1.2 mm clears socket wall and core",
        source=_ANT,
    ),
    Probe(
        "bulkhead_land_contact",
        "contact",
        ({"check": "bulkhead_land_contact"},),
        note="outer seal land under the flange/nut",
        source=_ANT,
    ),
    Probe(
        "bulkhead_inner_land_contact",
        "contact",
        ({"check": "bulkhead_inner_land_contact"},),
        note="inner land under the nut/shoulder",
        source=_ANT,
    ),
    Probe(
        "bulkhead_fit",
        "empty",
        ({"check": "bulkhead_fit"},),
        note="jack, D-body and nut fit the lid",
        source=_ANT,
    ),
    Probe(
        "pigtail_clear_lid",
        "empty",
        ({"check": "pigtail_clear_lid"},),
        note="pigtails clear the lid ribs and bosses",
        source=_ANT,
    ),
    Probe(
        "port_clear_body",
        "empty",
        ({"check": "port_clear_body"},),
        note="nuts, jacks, pigtails and bosses clear the body walls, flange and seal",
        source=_ANT,
    ),
    Probe(
        "underlid_in_keepout",
        "empty",
        ({"check": "underlid_in_keepout"},),
        note="under-lid parts stay inside antenna_lid_keepout()",
        source=_ANT,
    ),
    Probe(
        "no_pool_ring",
        "empty",
        ({"check": "no_pool_ring"},),
        note="nothing printed within 10 mm of either flange",
        source=_ANT,
    ),
    Probe(
        "lid_screw_access",
        "empty",
        ({"check": "lid_screw_access"},),
        note="all 12 lid-screw driver paths stay clear",
        source=_ANT,
    ),
    Probe(
        "antenna_vs_antenna",
        "empty",
        ({"check": "antenna_vs_antenna"},),
        note="the two stations (envelope, post, grips, bulkhead) clear each other",
        source=_ANT,
    ),
    Probe(
        "env_vs_body",
        "empty",
        ({"check": "env_vs_body"},),
        note="the two stations' full-length 40 mm envelopes, posts and grips clear the body",
        timeout_s=900,
        source=_ANT,
    ),
    Probe(
        "env_vs_arm_layer",
        "empty",
        ({"check": "env_vs_arm_layer"},),
        note="both full-length antenna stations on the lid clear the arm layer, arms and heads",
        timeout_s=900,
        source=_ANT,
    ),
    Probe(
        "env_vs_steering",
        "empty",
        ({"check": "env_vs_steering"},),
        note="both full-length antenna stations on the lid clear the steering servos, adapters "
        "and rods at faces 0-90 deg",
        timeout_s=900,
        source=_ANT,
    ),
    Probe(
        "env_vs_throttle",
        "empty",
        ({"check": "env_vs_throttle"},),
        note="both full-length antenna stations on the lid clear the throttle station",
        timeout_s=900,
        source=_ANT,
    ),
    Probe("gnss_jack_fit", "empty", ({"check": "gnss_jack_fit"},),
          note="GNSS SMA jack, D body, hex, washer and nut fit the land and D hole", source=_ANT),
    Probe("gnss_hex_bears", "contact", ({"check": "gnss_hex_bears"},),
          note="GNSS body hex bears on the 3 mm land in the under-lid counterbore", source=_ANT),
    Probe("gnss_land_backed", "empty", ({"check": "gnss_land_backed"},),
          note="the hex top face outside the hole is fully backed by the land", source=_ANT),
    Probe("gnss_dflat_keys", "contact", ({"check": "gnss_dflat_keys"},),
          note="turned 10 degrees, the GNSS D body meets the hole's flat (anti-rotation)",
          source=_ANT),
    Probe("gnss_seat_contact", "contact", ({"check": "gnss_seat_contact"},),
          note="GNSS lock washer lands on the lid top", source=_ANT),
    Probe("gnss_seat_backed", "empty", ({"check": "gnss_seat_backed"},),
          note="the whole washer annulus is backed by the flat lid top", source=_ANT),
    Probe("gnss_no_pool", "empty", ({"check": "gnss_no_pool"},),
          note="nothing printed within 10 mm of the GNSS washer", source=_ANT),
    Probe("gnss_plug_clear", "empty", ({"check": "gnss_plug_clear"},),
          note="mated antenna-cable plug clears the panel nut, washer and lid", source=_ANT),
    Probe("gnss_underlid_clear", "empty", ({"check": "gnss_underlid_clear"},),
          note="GNSS under-lid envelope clears the lid ribs/skirt and the body", source=_ANT),
    Probe("gnss_clear_stations", "empty", ({"check": "gnss_clear_stations"},),
          note="GNSS port clears both N stations, their envelopes and spanner rooms",
          source=_ANT),
    Probe("gnss_lid_screw_access", "empty", ({"check": "gnss_lid_screw_access"},),
          note="all 12 lid-screw driver paths clear the GNSS port room", source=_ANT),
    Probe(
        "env_vs_tower",
        "empty",
        ({"check": "env_vs_tower"},),
        note="both full-length antenna stations on the lid clear the camera tower stack",
        timeout_s=900,
        source=_ANT,
    ),
    Probe(
        "env_vs_hitch_plate",
        "empty",
        ({"check": "env_vs_hitch_plate"},),
        note="both full-length antenna stations on the lid clear the schematic hitch plate",
        timeout_s=900,
        source=_ANT,
    ),
)

_ROD = f"{THROTTLE}/rod_checks.scad"
_SPANS = (940, 1066.8, 1200)
_SHIFTS = ([1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1.5], [0, 0, -1.5])

_CTL = f"{CONTROL}/control_checks.scad"
# Single-run checks inside the enclosure: chassis, tower, boards and keep-outs.
_CTL_EMPTY = (
    "base_clear_body",
    "chassis_clear_body",
    "chassis_clear_base",
    "chassis_lowered_clear",
    "chassis_clear_bolt_heads",
    "contents_clear_keepouts",
    "contents_clear_body",
    "contents_clear_lid",
    "patch_clear",
    "pigtails_clear",
    "lid_bulkheads_clear",
    "tower_leads_clear",
    "gnss_jack_clear",
    "port_room_clear",
    "tower_clear_chassis",
    "tower_lowered_clear",
    "boards_clear_tower",
    "signal_boards_clear",
    "ina_cradle_clear_clamp",
    "ina_clear_others",
    "ina_access_clear",
    "sc_clip_clear_clamp",
    "sc_clip_clear_tower",
)
_CTL_CONTACT = (
    "base_seats_on_bosses",
    "chassis_seats_on_base",
    "chassis_hooks_hold",
    "chassis_detent_blocks",
    "tower_seats_on_chassis",
    "tower_studs_hold",
    "tower_lock_blocks",
    "stack_on_bosses",
    "clamp_pads_bear",
    "ina_cradle_seats",
    "ina_hooks_hold",
    "sc_clip_seats",
    "sc_clip_retained",
)
# (check, expect, number of `idx` stations): carriers 0-6 are the six strap bays and the harness
# clamp; bays/harness 0-1 are the minimum and maximum box or bundle; boards 0-2 the PCA9685s and
# the Pololu; headers 0-3 the stack's header joints; patch plugs 0-1 the Pi and Alta RJ45 ends
# of the Ethernet patch; INA3221 cradle: boards 0-1 the smallest and largest board of its
# capacity, saddle 0-1 the +x and -x stops, clamp travel 0-2 the -12 / 0 / +3 mm stack range,
# lift 0-1 the seated and the slid (tower-removal) pose; Adafruit #6062 0-1 the frame (jaw) and
# the bar (shoe). RF service joints (joint 0 Pi HaLow, 1 GNSS, 2 Alta HaLow): lead/cable reaches
# 0-5 = 2 x joint + end (0 board or lid end, 1 clip jack or plug barrel); SMA clip keyed 0-5 the
# +x/-x/+y/-y shifts and +-1 degree turns; jack probes 0-2 the joints; clip travel 0-2 the
# -12 / 0 / +3 mm range; spanner sweeps 0-2 lid off, 3-5 lid lifted 100 mm; lift 0-1 seated
# and slid.
_CTL_INDEXED = (
    ("patch_plug_seats", "contact", 2),
    ("lid_cable_reaches", "contact", 6),
    ("tower_lead_reaches", "contact", 6),
    ("header_engages", "contact", 4),
    ("carrier_hook_seats", "contact", 7),
    ("carrier_latch_under_land", "contact", 7),
    ("carrier_nominal_clear", "empty", 7),
    ("carrier_clear_others", "empty", 7),
    ("bay_holds_box", "contact", 2),
    ("bay_pawls_bear", "contact", 2),
    ("harness_holds_bundle", "contact", 2),
    ("harness_pawls_bear", "contact", 2),
    ("bay_strap_clear", "empty", 2),
    ("bay_box_clear_frame", "empty", 2),
    ("harness_strap_clear", "empty", 2),
    ("signal_board_captured", "contact", 3),
    ("ina_saddle_locates", "contact", 2),
    ("ina_board_clear_frame", "empty", 2),
    ("ina_board_clear_bar", "empty", 2),
    ("ina_jaw_holds", "contact", 2),
    ("ina_shoe_holds", "contact", 2),
    ("ina_bar_preloads", "contact", 2),
    ("ina_bar_clear", "empty", 2),
    ("ina_pawls_bear", "contact", 2),
    ("ina_travel_clear", "empty", 3),
    ("ina_lift_clear", "empty", 2),
    ("ina_adafruit_clear", "empty", 2),
    ("ina_adafruit_held", "contact", 2),
    ("sc_clip_keyed", "contact", 6),
    ("sc_jack_fit", "empty", 3),
    ("sc_hex_bears", "contact", 3),
    ("sc_dflat_keys", "contact", 3),
    ("sc_seat_contact", "contact", 3),
    ("sc_seat_backed", "empty", 3),
    ("sc_clip_clear", "empty", 3),
    ("sc_sweep_clear", "empty", 6),
    ("sc_lift_clear", "empty", 2),
)
CONTROL_PROBES: tuple[Probe, ...] = (
    *(Probe(c, "empty", ({"check": c},), source=_CTL) for c in _CTL_EMPTY),
    *(Probe(c, "contact", ({"check": c},), source=_CTL) for c in _CTL_CONTACT),
    *(
        Probe(c, e, tuple({"check": c, "idx": i} for i in range(n)), source=_CTL)
        for c, e, n in _CTL_INDEXED
    ),
)

PROBES_77603: tuple[Probe, ...] = (
    *ANTENNA_PROBES,
    *CONTROL_PROBES,
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
        note="with its fingers sprung free, the keeper rises clear of the walls below the "
        "steering servo",
    ),
    # Servo retention geometry. Contact probes prove geometric interference only: snap force,
    # rib crush and vibration hold-down remain bench-test items.
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
        "keeper_ribs_preload_lugs",
        "contact",
        ({"check": "keeper_ribs_preload_lugs"},),
        note="seated, the crush ribs overlap the provisional ears (designed 0.4 mm "
        "interference)",
    ),
    Probe(
        "keeper_snap_blocks_liftoff",
        "contact",
        ({"check": "keeper_snap_blocks_liftoff"},),
        note="lifted 0.5 mm, every barb meets its window roof",
    ),
    Probe(
        "keeper_snap_fingers_pass",
        "empty",
        tuple({"check": "keeper_snap_fingers_pass", "keeper_station": s} for s in (0, 1, 2)),
        note="fingers opened 0.7 mm slide down the walls past the ears at three stations",
    ),
    # Variant E: the clevis keeps material round the hinge pin in each ear.
    Probe(
        "clevis_double_shear",
        "contact",
        tuple({"check": "clevis_double_shear", "clevis_ear_side": s} for s in (1, -1)),
        note="variant E's clevis has material around the hinge pin in each ear separately",
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
            for k in ("shared", "index", "a_post", "e_hinge")
        ),
        note="each installed keeper ring clears its pin's groove, lip and shoulder",
    ),
    Probe(
        "pin_ring_held_by_lip",
        "contact",
        tuple(
            {"check": "pin_ring_held_by_lip", "pin_ring_kind": k}
            for k in ("shared", "index", "a_post", "e_hinge")
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
    # ---- Throttle pushrod and linkage: throttle_77603/rod_checks.scad ----
    Probe(
        "rod_capture_pip_fit",
        "empty",
        ({"check": "rod_capture_pip_fit"},),
        note="sleeve and slider separate as printed",
        source=_ROD,
    ),
    Probe(
        "rod_cam_pip_fit",
        "empty",
        ({"check": "rod_cam_pip_fit"},),
        note="cam lever clears sleeve and slider as printed",
        source=_ROD,
    ),
    Probe(
        "rod_cam_captive",
        "contact",
        tuple({"check": "rod_cam_captive", "rod_shift": s} for s in _SHIFTS),
        note="cam lever cannot leave the sleeve",
        source=_ROD,
    ),
    Probe(
        "rod_cam_swing_clear",
        "empty",
        tuple({"check": "rod_cam_swing_clear", "rod_phi": a} for a in (30, 60, 89)),
        note="lever swings without meeting its housing",
        source=_ROD,
    ),
    Probe(
        "rod_cam_open_clear",
        "empty",
        tuple({"check": "rod_cam_open_clear", "rod_setting": v} for v in _SPANS),
        note="open cam frees the slide",
        source=_ROD,
    ),
    Probe(
        "rod_cam_clamps",
        "contact",
        tuple({"check": "rod_cam_clamps", "rod_setting": v} for v in _SPANS),
        note="closed cam overlaps the seated slider by the preload after full pin play",
        source=_ROD,
    ),
    Probe(
        "rod_cam_overcentre",
        "contact",
        ({"check": "rod_cam_overcentre"},),
        note="the corner reaches past the closed flat",
        source=_ROD,
    ),
    Probe(
        "rod_cam_rest",
        "empty",
        ({"check": "rod_cam_rest"},),
        note="the closed flat stops short: over-centre valley",
        source=_ROD,
    ),
    Probe(
        "rod_capture_range_clear",
        "empty",
        tuple({"check": "rod_capture_range_clear", "rod_setting": v} for v in _SPANS),
        source=_ROD,
    ),
    Probe(
        "rod_overlap_capture",
        "contact",
        tuple({"check": "rod_overlap_capture", "rod_setting": v} for v in _SPANS),
        note="overtravel catch",
        source=_ROD,
    ),
    Probe(
        "rod_overlap_capture_reverse",
        "contact",
        ({"check": "rod_overlap_capture_reverse", "rod_setting": 940},),
        source=_ROD,
    ),
    Probe(
        "rod_fork_clears_sleeve",
        "empty",
        tuple({"check": "rod_fork_clears_sleeve", "rod_setting": v} for v in _SPANS),
        source=_ROD,
    ),
    Probe("rod_wheel_pip_fit", "empty", ({"check": "rod_wheel_pip_fit"},), source=_ROD),
    Probe(
        "rod_wheel_captive",
        "contact",
        tuple(
            {"check": "rod_wheel_captive", "rod_shift": s}
            for s in ([0, 0, 1], [0, 0, -1], [2, 0, 0], [-2, 0, 0], [0, 2, 0], [0, -2, 0])
        ),
        source=_ROD,
    ),
    Probe(
        "rod_wheel_detent_click",
        "contact",
        tuple({"check": "rod_wheel_detent_click", "rod_phi": a} for a in (11.25, -11.25)),
        note="half a notch: pawls must flex",
        source=_ROD,
    ),
    Probe(
        "rod_wheel_detent_seat",
        "empty",
        tuple({"check": "rod_wheel_detent_seat", "rod_phi": a} for a in (22.5, -22.5)),
        source=_ROD,
    ),
    Probe(
        "rod_wheel_runs_free",
        "empty",
        tuple({"check": "rod_wheel_runs_free", "rod_fine": f} for f in (0, 10, 20)),
        source=_ROD,
    ),
    Probe(
        "rod_wheel_drives",
        "contact",
        tuple(
            {"check": "rod_wheel_drives", "rod_fine": f, "rod_shift": s}
            for f in (0, 20)
            for s in ([0, 0, 0.8], [0, 0, -0.8])
        ),
        note="thread carries the fork at both fine limits",
        source=_ROD,
    ),
    Probe(
        "rod_fork_clear",
        "empty",
        tuple({"check": "rod_fork_clear", "rod_fine": f} for f in (0, 20)),
        source=_ROD,
    ),
    Probe(
        "rod_joint_pip_fit",
        "empty",
        tuple({"check": "rod_joint_pip_fit", "rod_joint": j} for j in (0, 1)),
        source=_ROD,
    ),
    Probe(
        "rod_joint_captive",
        "contact",
        tuple(
            {"check": "rod_joint_captive", "rod_joint": j, "rod_shift": s}
            for j in (0, 1)
            for s in _SHIFTS
        ),
        source=_ROD,
    ),
    Probe(
        "rod_joint_swing_clear",
        "empty",
        tuple(
            {"check": "rod_joint_swing_clear", "rod_joint": j, "rod_phi": a}
            for j in (0, 1)
            for a in (30, 60, 89)
        ),
        source=_ROD,
    ),
    Probe(
        "rod_joint_insert_clear",
        "empty",
        tuple({"check": "rod_joint_insert_clear", "rod_joint": j} for j in (0, 1)),
        note="spigot passes the open lever",
        source=_ROD,
    ),
    Probe(
        "rod_joint_clamps",
        "contact",
        tuple({"check": "rod_joint_clamps", "rod_joint": j} for j in (0, 1)),
        source=_ROD,
    ),
    Probe(
        "rod_joint_tooth_engages",
        "contact",
        tuple({"check": "rod_joint_tooth_engages", "rod_joint": j} for j in (0, 1)),
        source=_ROD,
    ),
    Probe(
        "rod_joint_tooth_clear",
        "empty",
        tuple({"check": "rod_joint_tooth_clear", "rod_joint": j} for j in (0, 1)),
        source=_ROD,
    ),
    Probe(
        "rod_joint_blocks_withdrawal",
        "contact",
        tuple(
            {"check": "rod_joint_blocks_withdrawal", "rod_joint": j, "rod_shift": s}
            for j in (0, 1)
            for s in ([0, 0, 0], [0, 0, 0.4])
        ),
        note="positive lock at +1 mm pull",
        source=_ROD,
    ),
    Probe(
        "rod_joint_blocks_rotation",
        "contact",
        tuple(
            {"check": "rod_joint_blocks_rotation", "rod_joint": j, "rod_phi": a}
            for j in (0, 1)
            for a in (3, -3)
        ),
        source=_ROD,
    ),
    Probe(
        "rod_joint_tooth_bears_on_boss",
        "contact",
        tuple({"check": "rod_joint_tooth_bears_on_boss", "rod_joint": j} for j in (0, 1)),
        note="withdrawal load goes to the boss wall, not the pins",
        source=_ROD,
    ),
    Probe(
        "rod_span_reachable",
        "empty",
        tuple({"check": "rod_span_reachable", "rod_setting": v} for v in (940, 1200)),
        source=_ROD,
    ),
    Probe(
        "rod_span_eye_located",
        "contact",
        tuple({"check": "rod_span_eye_located", "rod_setting": v} for v in (940, 1200)),
        source=_ROD,
    ),
    *(
        Probe(
            check,
            expect,
            tuple({"check": check, "link_variant": v} for v in "ABCDE"),
            source=_ROD,
        )
        for check, expect in (
            ("linkage_servo_end_fit", "empty"),
            ("linkage_servo_eye_lift", "contact"),
            ("linkage_servo_eye_pull", "contact"),
            ("linkage_throttle_end_fit", "empty"),
            ("linkage_throttle_eye_lift", "contact"),
            ("linkage_throttle_eye_pull", "contact"),
        )
    ),
    Probe(
        "linkage_vs_steering",
        "empty",
        tuple(
            {"check": "linkage_vs_steering", "link_variant": v, "link_sweep_deg": s}
            for v in "ABCDE"
            for s in (-30, 0, 30)
        ),
        timeout_s=900,
        source=_ROD,
    ),
    Probe(
        "linkage_vs_head",
        "empty",
        tuple(
            {"check": "linkage_vs_head", "link_variant": v, "link_sweep_deg": s}
            for v in "ABCDE"
            for s in (-30, 0, 30)
        ),
        source=_ROD,
    ),
    Probe(
        "rod_vs_steering",
        "empty",
        tuple({"check": "rod_vs_steering", "link_variant": v} for v in "ABCDE"),
        # Builds the whole rod and three full steering pushrods per variant.
        timeout_s=1800,
        source=_ROD,
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
