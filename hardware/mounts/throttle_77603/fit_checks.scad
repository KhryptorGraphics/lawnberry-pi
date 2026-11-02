// CSG probes for the 77603 throttle station.
//
// Each probe is a negative-volume interference test (must render EMPTY) or a positive
// retention/land test (must render NON-EMPTY). Probes intersect REAL built solids rather
// than re-deriving a formula, because a probe that duplicates the model's own arithmetic
// agrees with the model whether or not the model is right.
include <parameters.scad>
use <mg995_throttle_mount.scad>
use <throttle_u_fitting.scad>
use <throttle_horn_variants.scad>
use <printed_joints.scad>
use <../servo_pushrod_adapter.scad>  // the printed steering adapter, for clearance probes
use <throttle_arm_head.scad>   // the derivative head, for the attachment and clearance probes
use <../hitch_arm_layer.scad>   // legacy arm head, for the bearing probe
use <../pushrod.scad>           // legacy steering rod geometry, for the linkage-clearance probe
use <assemblies.scad>          // scene placements, reused by the linkage probes

check = "u_lever_passes";
link_variant = "E";
link_sweep_deg = 0;     // horn sweep about the neutral pose, degrees
clevis_ear_side = 1;     // clevis_double_shear: +1 upper ear, -1 lower ear
u_yoke_angle = 30;       // u_yoke_pitch_range: yoke pitch about the hinge axis, degrees
u_yoke_shift = [0, 0, 1];  // u_yoke_captive: displacement of the yoke, mm
d_ball_angle_deg = 10;      // d_ball_range_clear: prototype articulation angle
d_ball_shift_mm = [0, 0, 3]; // d_ball_captive: forced translation of the ball
c_radius_mm = horn_radius_default_mm;  // c_shoe_*: tested continuous radius
c_shoe_shift_mm = [0, 0, 1];           // c_shoe_*: forced movement of the shoe
c_lock_stage = 1;  // c_lock_*/c_key_*: fraction of the dovetail lift the wedge key has driven
pin_ring_kind = "shared";  // pin_ring_*: shared, index, a_post or e_hinge
keeper_station = 0;     // keeper_snap_fingers_pass: 0 at the wall top, 1 mid-wall, 2 near home
// ---- Helpers -----------------------------------------------------------------------
module at_servo() { translate(throttle_servo_base_mm()) children(); }
// The derivative right head as built, in world coordinates.
module head_world() { arm_vertical(arm_joint_t) throttle_arm_head(); }
// The MG995 reference body with its output shaft, in the cradle frame.
module mg995_body_with_shaft() {
    translate([-mg995_body_width_mm / 2, -mg995_body_length_mm / 2, 0])
        cube([mg995_body_width_mm, mg995_body_length_mm, mg995_body_height_mm]);
    translate([mg995_shaft_xy_mm[0], mg995_shaft_xy_mm[1], mg995_body_height_mm - 0.01])
        cylinder(d = mg995_shaft_d_mm, h = mg995_shaft_projection_mm);
}
// How far the servo, and the keeper, must rise to leave the cradle: the walls' height plus
// 2 mm, so the body's base (or the keeper's legs) passes over the wall tops.
service_lift_mm = cradle_wall_h_mm + 2;

// ---- U-fitting probes --------------------------------------------------------------
module u_lever_probe(scale = 1.0) {
    // A full-height lever section. Because the U's mouth is open at -Z and closed at +Z, a
    // lever that fits must pass through the void WITHOUT touching either the walls or the
    // closed top, so a solid intersection means "does not fit".
    d = throttle_u_inner_depth_mm * scale;
    w = throttle_u_inner_width_mm * scale;
    // Spans the open channel from just below the mouth up to just under the CLOSED TOP.
    // Reaching above the top would always intersect the lid and make every pass test fail.
    translate([-w / 2, -d / 2, -1])
        cube([w, d, throttle_u_engagement_mm + 0.8]);
}

// The raised RIGHT steering servo as it moves: its case, its moving bracket and the printed
// steering adapter at full forward, neutral and full reverse, and the WHOLE steering pushrod
// (the legacy rod geometry, not a guessed box). At each face the neutral rod is translated
// rigidly by the crank pin's displacement: rod_between_points() rightly refuses pin centres
// that do not match a rod setting. These are SAMPLED poses, not a bound on every real pose:
// the lever's arc at the clamp and the MyRIDE platform's motion are unmeasured, and either can
// move the clamp end differently from the crank.
// `near_station = true` keeps only each pushrod's segment in a box around the crank, which
// spans the whole station above the throttle stack, so station-local probes (the service
// lifts) still meet the rods there but their CGAL runs fit the probe timeout. Probes that
// follow the throttle rod forward (mount_vs_steering, rod_vs_steering) keep the whole rods.
module steering_envelope(near_station = false) {
    servo_body_envelope(1);
    for (face = [servo_face_forward_check_deg, servo_face_neutral_deg,
                 servo_face_reverse_deg]) {
        servo_moving_bracket(1, face);
        servo_pushrod_adapter_at(1, face);
        translate(servo_crank_pin_at(1, face) - servo_crank_pin(1)) intersection() {
            rod_between_points(servo_crank_pin(1), lapbar_pin(1), rod_nominal_setting());
            if (near_station)
                translate(servo_crank_pin(1) + [0, 60, 0]) cube([200, 260, 200], center = true);
        }
    }
}

// A socket and extension on each joint bolt, along the bolt's own axis from the head
// flange's top face: d 17 mm (10 mm AF nut socket), 55 mm long.
module joint_bolt_tools(len = 55) {
    for (x = [-arm_joint_bolt, arm_joint_bolt], y = [-arm_joint_bolt, arm_joint_bolt])
        rotate([0, arm_rake_deg, 0]) translate([x, y, arm_joint_flange_t + 0.5])
            cylinder(d = 17, h = len);
}

// The throttle linkage at this variant and horn sweep: horn interface and the rod's servo
// section (rod direction held at neutral).
module throttle_linkage_at(letter, sweep) {
    horn_interface(letter, sweep);
    rod_servo_end(letter, sweep);
}

// ---- Pin keeper pairs (pin_ring_* probes) ------------------------------------------------
// The shared printed_pin primitive and the real pins that use its lip-and-groove relief, each
// with its ring as installed. The shared pin is drawn in its own frame. (The servo keeper's
// cross pins and rings were retired with the one-piece snap keeper.)
module pin_ring_pin(kind) {
    if (kind == "shared") printed_pin(24);
    else keeper_probe_pin(kind);
}

module pin_ring_ring(kind) {
    if (kind == "shared") translate([0, 0, keeper_ring_seat_z()]) printed_keeper();
    else keeper_probe_ring(kind);
}

// ---- Snap keeper (keeper_* probes) ---------------------------------------------------------
// Insertion stations for the keeper, as lifts above its seated pose: barbs at the wall top,
// halfway down the wall, and just above their windows.
function keeper_station_lift(station) =
    let(top = cradle_wall_h_mm - keeper_barb_z0_mm)
    [top - 0.5, top / 2, keeper_barb_d_mm + keeper_barb_land_mm + 0.5][station];

// ---- Dispatch ----------------------------------------------------------------------
if (check == "u_lever_passes" || check == "u_lever_passes_nominal") {
    // EMPTY means the nominal lever section passes: it touches neither wall nor top.
    intersection() {
        throttle_u_fitting("body");
        u_lever_probe(1.0);
    }
} else if (check == "u_lever_passes_lean") {
    // EMPTY means the nominal lever still passes when the whole cavity carries a lean.
    intersection() {
        rotate([0, 12, 0]) throttle_u_fitting("body");
        rotate([0, 12, 0]) u_lever_probe(1.0);
    }
} else if (check == "u_oversize_rejected") {
    // SOLID means the oversize lever is correctly refused by the walls.
    intersection() {
        throttle_u_fitting("body");
        u_lever_probe(1.12);
    }
} else if (check == "u_clearance_applied") {
    // With refinement clearance applied the nominal lever must still pass cleanly.
    intersection() {
        throttle_u_fitting("body");
        u_lever_probe(1.0);
    }
} else if (check == "u_yoke_fit") {
    // EMPTY: the print-in-place yoke clears the body as printed.
    intersection() {
        u_body();
        u_yoke();
    }
} else if (check == "u_yoke_pitch_range") {
    // EMPTY, run at several angles: the yoke pitches about the hinge axis without contact.
    intersection() {
        u_body();
        u_yoke_pitched(u_yoke_angle);
    }
} else if (check == "u_yoke_captive") {
    // CONTACT, run for +/-1 mm along each axis: wherever the yoke is pushed, the stub pins,
    // bores, legs and boss stop it, so it cannot leave the body.
    intersection() {
        u_body();
        translate(u_yoke_shift) u_yoke();
    }
} else if (check == "shelf_fuses_to_rib") {
    // CONTACT: the shelf really overlaps the legacy head's rib, so head and shelf are one
    // printed piece rather than two solids that happen to touch.
    intersection() {
        arm_vertical(arm_joint_t) throttle_shelf();
        arm_head_assembly();
    }
} else if (check == "cradle_fuses_to_shelf") {
    // CONTACT: the cradle walls root into the shelf's slab.
    intersection() {
        at_servo() mg995_throttle_mount("cradle_walls");
        arm_vertical(arm_joint_t) throttle_shelf();
    }
} else if (check == "shelf_supports_body") {
    // CONTACT: the built head carries material directly under the servo body's footprint.
    intersection() {
        head_world();
        at_servo() translate([-mg995_body_width_mm / 2, -mg995_body_length_mm / 2, -0.6])
            cube([mg995_body_width_mm, mg995_body_length_mm, 0.5]);
    }
} else if (check == "support_clears_servo_passages") {
    // EMPTY: the added material must not fill the steering servo's six holes or its rear
    // opening, or the original servo could no longer be bolted on.
    intersection() {
        translate(arm_point(arm_joint_t)) throttle_arm_head();
        translate([-1, 0, arm_head_rise()]) {
            servo_rear_hole_cut(servo_face_dx() + 20);
            servo_rear_opening_cut(servo_face_dx() + 20);
        }
    }
} else if (check == "joint_bolts_have_tool_room") {
    // EMPTY: a socket and extension still reach all four joint bolts past the shelf.
    intersection() {
        throttle_arm_head();
        joint_bolt_tools();
    }
} else if (check == "mount_vs_steering") {
    // EMPTY: the head, the keeper stack and the servo body stay clear of the raised steering
    // servo -- its case, moving bracket and adapter at 0/45/90 deg -- and its pushrod sweep.
    intersection() {
        union() {
            head_world();
            mg995_throttle_mount("assembly");
            at_servo() mg995_body_with_shaft();
        }
        steering_envelope();
    }
} else if (check == "servo_lifts_out") {
    // EMPTY: with its horn removed, the MG995 rises straight out of the cradle (base past the
    // wall tops) without meeting the head or the steering servo above it, so the throttle
    // servo is serviced without removing the steering servo. The swept body starts 0.05 mm
    // above the shelf: seated, its base and the shelf share a plane, and that coplanar contact
    // renders without indicating any interference. The steering pushrods are kept only near
    // the station (`near_station`): beyond it they descend away toward the lap bar.
    intersection() {
        at_servo() hull() {
            translate([0, 0, 0.05]) mg995_body_with_shaft();
            translate([0, 0, service_lift_mm]) mg995_body_with_shaft();
        }
        union() {
            head_world();
            steering_envelope(near_station = true);
        }
    }
} else if (check == "keeper_lifts_out") {
    // EMPTY: released, the one-piece keeper rises until its fingers clear the cradle walls
    // without meeting the steering servo or its pushrods above it.
    intersection() {
        at_servo() translate([0, 0, service_lift_mm]) mg995_throttle_mount("keeper_installed");
        steering_envelope(near_station = true);
    }
} else if (check == "keeper_bears_on_lugs") {
    // CONTACT: the installed keeper has material directly above the provisional lug band.
    intersection() {
        mg995_throttle_mount("keeper_installed");
        for (s = [-1, 1])
            translate([-mg995_body_width_mm / 2,
                       s > 0 ? mg995_body_length_mm / 2 + joint_clearance_mm + 0.1
                             : -mg995_mount_pitch_fb_mm / 2 - 3,
                       cradle_flange_z1_mm + 0.25])
                cube([mg995_body_width_mm,
                      mg995_mount_pitch_fb_mm / 2 + 3 - mg995_body_length_mm / 2
                      - joint_clearance_mm - 0.1, 0.5]);
    }
} else if (check == "keeper_fits_cradle") {
    // EMPTY: the seated keeper -- frame, fingers and barbs in their windows -- does not
    // interfere with the head's cradle walls.
    intersection() {
        head_world();
        at_servo() mg995_throttle_mount("keeper_installed");
    }
} else if (check == "keeper_ribs_preload_lugs") {
    // CONTACT: seated (frame on the provisional ear tops), the crush ribs overlap the ears:
    // the designed 0.4 mm interference that presses the ears onto the wall tops. The ears
    // are lowered 0.05 mm so the frame's coplanar bearing face drops out and only the ribs'
    // overlap remains. A geometric overlap only; crush force and retention are bench items.
    intersection() {
        mg995_throttle_mount("keeper_installed");
        translate([0, 0, -0.05]) mg995_throttle_mount("flange_envelope");
    }
} else if (check == "keeper_snap_blocks_liftoff") {
    // CONTACT: lifted 0.5 mm (past the 0.2 mm catch play), every barb's catch face meets
    // its window's roof in the head's side wall.
    intersection() {
        at_servo() translate([0, 0, keeper_catch_play_mm + 0.3])
            mg995_throttle_mount("keeper_installed");
        head_world();
    }
} else if (check == "keeper_snap_fingers_pass") {
    // EMPTY, at each station: with every finger opened by keeper_snap_spread_mm at its
    // barb, the keeper slides down the walls -- barbs at the wall top, mid-wall and just
    // above their windows -- without meeting the head or the provisional ears.
    intersection() {
        at_servo() translate([0, 0, keeper_station_lift(keeper_station)])
            keeper_installed(spread = keeper_snap_spread_mm);
        union() {
            head_world();
            at_servo() mg995_throttle_mount("flange_envelope");
        }
    }
} else if (check == "clevis_double_shear") {
    // CONTACT, run once per ear: variant E's own clevis has material in a ring around the
    // hinge pin's axis inside that ear, so the eye between the ears is in double shear.
    // Frame: the clevis frame, bore centre at the origin.
    gap = eye_fork_gap();
    ear = e_ear_t_mm();
    z0 = clevis_ear_side > 0 ? gap / 2 + 0.3 : -gap / 2 - ear + 0.3;
    intersection() {
        e_clevis();
        translate([0, 0, z0])
            difference() {
                cylinder(d = bushing_od_mm + 3, h = ear - 0.6);
                translate([0, 0, -1]) cylinder(d = bushing_od_mm + 0.6, h = ear + 1);
            }
    }
} else if (check == "d_ball_pip_fit") {
    // EMPTY: the truncated ball and carrier-integral socket print as separate bodies.
    intersection() {
        d_carrier();
        translate([x_horn_arm_radius_mm, 0, 0]) d_ball_stem();
    }
} else if (check == "d_ball_range_clear") {
    // EMPTY: the stem/post clears the carrier-integral socket through the tested angle.
    intersection() {
        d_carrier();
        translate([x_horn_arm_radius_mm, 0, 0]) d_ball_stem_pitched(d_ball_angle_deg);
    }
} else if (check == "d_ball_captive") {
    // CONTACT: the socket prevents the ball leaving through either narrowed mouth.
    intersection() {
        d_carrier();
        translate([x_horn_arm_radius_mm, 0, 0])
            translate(d_ball_shift_mm) d_ball_stem();
    }
} else if (check == "c_shoe_pip_fit") {
    intersection() {
        c_carrier();
        c_shoe(c_radius_mm);
    }
} else if (check == "c_shoe_range_clear") {
    // The shoe slides continuously at the requested radius without rail interference.
    intersection() {
        c_carrier();
        c_shoe(c_radius_mm);
    }
} else if (check == "c_shoe_captive") {
    // CONTACT: upper/lower dovetail flanks stop vertical and transverse displacement.
    intersection() {
        c_carrier();
        translate(c_shoe_shift_mm) c_shoe(c_radius_mm);
    }
} else if (check == "c_shoe_end_stop") {
    // CONTACT: the closed inboard/outboard track ends retain the shoe at the travel limits.
    intersection() {
        c_carrier();
        translate(c_shoe_shift_mm) c_shoe(c_radius_mm);
    }
} else if (check == "c_key_entry_clear") {
    // EMPTY: withdrawn 0.5 mm short of first roof contact, the wedge key resting on the rail's
    // seat touches nothing else: neither the resting shoe nor the carrier below its seat skin.
    intersection() {
        c_key_placed(c_radius_mm, stage = 0, withdraw = 0.5);
        union() {
            c_shoe(c_radius_mm);
            difference() {
                c_carrier();
                c_seat_band();
            }
        }
    }
} else if (check == "c_key_bears_shoe") {
    // CONTACT: driven home with modelled preload, the key's gable bears on the shoe's roof.
    intersection() {
        c_key_placed(c_radius_mm, preload = c_lock_preload());
        c_shoe_lifted(c_radius_mm, preload = c_lock_preload());
    }
} else if (check == "c_key_bears_rail") {
    // CONTACT: at the same pose the key's flat underside bears on the rail's seat.
    intersection() {
        c_key_placed(c_radius_mm, preload = c_lock_preload());
        c_carrier();
    }
} else if (check == "c_shoe_clamped") {
    // CONTACT: at the same pose the lifted shoe's ramps bear on the rail's 45-degree undercut.
    intersection() {
        c_shoe_lifted(c_radius_mm, preload = c_lock_preload());
        c_carrier();
    }
} else if (check == "c_lock_bearing_only") {
    // EMPTY: at the requested drive stage (with preload) every key/shoe/carrier overlap lies
    // inside its designed bearing zone: seat skin, slot roof, dovetail undercut.
    union() {
        difference() {
            intersection() {
                c_key_placed(c_radius_mm, c_lock_stage, c_lock_preload());
                c_carrier();
            }
            c_seat_band(c_lock_preload());
        }
        difference() {
            intersection() {
                c_key_placed(c_radius_mm, c_lock_stage, c_lock_preload());
                c_shoe_lifted(c_radius_mm, c_lock_stage, c_lock_preload());
            }
            c_roof_band(c_radius_mm, c_lock_stage, c_lock_preload());
        }
        difference() {
            intersection() {
                c_shoe_lifted(c_radius_mm, c_lock_stage, c_lock_preload());
                c_carrier();
            }
            c_undercut_band(c_lock_preload());
        }
    }
} else if (check == "c_keeper_clear") {
    // EMPTY: the keeper ring placed against the rail's -Y face clears the preloaded key and
    // the lifted shoe (its face lands on the rail by construction, so the rail is excluded).
    intersection() {
        c_keeper_placed(c_radius_mm);
        union() {
            c_key_placed(c_radius_mm, preload = c_lock_preload());
            c_shoe_lifted(c_radius_mm, preload = c_lock_preload());
        }
    }
} else if (check == "c_keeper_stops_key") {
    // CONTACT: withdrawing the key 0.5 mm past the keeper's slack drives its groove shoulder
    // into the ring.
    intersection() {
        c_key_placed(c_radius_mm, preload = c_lock_preload(),
                     withdraw = c_keeper_slack_mm(c_lock_preload()) + 0.5);
        c_keeper_placed(c_radius_mm);
    }
} else if (check == "c_keeper_stops_on_rail") {
    // CONTACT: the ring dragged 0.5 mm toward +Y meets the rail's -Y face.
    intersection() {
        translate([0, 0.5, 0]) c_keeper_placed(c_radius_mm);
        c_carrier();
    }
} else if (check == "pin_ring_seated_clear") {
    // EMPTY: the keeper ring, installed in its pin's groove, clears the pin (groove, lip and
    // shaft shoulder).
    intersection() {
        pin_ring_pin(pin_ring_kind);
        pin_ring_ring(pin_ring_kind);
    }
} else if (check == "pin_ring_held_by_lip") {
    // CONTACT: the installed ring pushed 1 mm toward the pin's free end (-Z for every probed
    // pin) meets the end lip.
    intersection() {
        pin_ring_pin(pin_ring_kind);
        translate([0, 0, -1]) pin_ring_ring(pin_ring_kind);
    }
} else {
    assert(false, str("unknown check: ", check));
}
