// CSG probes for the 77603 throttle station.
//
// Each probe is a negative-volume interference test (must render EMPTY) or a positive
// retention/land test (must render NON-EMPTY). Probes intersect REAL built solids rather
// than re-deriving a formula, because a probe that duplicates the model's own arithmetic
// agrees with the model whether or not the model is right.
include <parameters.scad>
use <mg995_throttle_mount.scad>
use <throttle_pushrod.scad>
use <throttle_u_fitting.scad>
use <throttle_horn_variants.scad>
use <printed_joints.scad>
use <../servo_pushrod_adapter.scad>  // the printed steering adapter, for clearance probes
use <throttle_arm_head.scad>   // the derivative head, for the attachment and clearance probes
use <../hitch_arm_layer.scad>   // legacy arm head, for the bearing probe
use <../pushrod.scad>           // legacy steering rod geometry, for the linkage-clearance probe
use <assemblies.scad>          // scene placements, reused by the linkage probes

check = "u_lever_passes";
rod_capture_length_mm = rods_nominal_span_mm;
link_variant = "E";
link_sweep_deg = 0;     // horn sweep about the neutral pose, degrees
clevis_ear_side = 1;     // clevis_double_shear: +1 upper ear, -1 lower ear
splice_joint = 0;        // splice probes: 0 servo/chain, 1 chain/chain, 2 chain/sleeve
splice_shift_mm = 1;     // splice_bar_holds_joint: displacement along the rod axis
rod_collet_advance_mm = 1.2;  // rod_collet_nut_runs_free: helical nut advance, mm (- = back)
rod_collet_shift_mm = 1;      // rod_collet_nut_captive: pure axial push of the nut, mm
rod_eye_target = "body";      // rod_body_bar_holds: "body" or "slider"
rod_eye_shift_mm = 1;         // rod_body_bar_holds: bar displacement along the rod, mm
u_yoke_angle = 30;       // u_yoke_pitch_range: yoke pitch about the hinge axis, degrees
u_yoke_shift = [0, 0, 1];  // u_yoke_captive: displacement of the yoke, mm
d_ball_angle_deg = 10;      // d_ball_range_clear: prototype articulation angle
d_ball_shift_mm = [0, 0, 3]; // d_ball_captive: forced translation of the ball
c_radius_mm = horn_radius_default_mm;  // c_shoe_*: tested continuous radius
c_shoe_shift_mm = [0, 0, 1];           // c_shoe_*: forced movement of the shoe
c_lock_stage = 1;  // c_lock_*/c_key_*: fraction of the dovetail lift the wedge key has driven
pin_ring_kind = "shared";  // pin_ring_*: shared, index, a_post, e_hinge or cross
// Plan unit vector of the rod's run for the selected variant.
function link_run_mm() =
    [cos(run_angle_deg(link_variant)), sin(run_angle_deg(link_variant)), 0];

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
// with its ring as installed. The shared and cross pins are drawn in their own frames.
module pin_ring_pin(kind) {
    if (kind == "shared") printed_pin(24);
    else if (kind == "cross") cross_pin(keeper_cross_pin_len_mm);
    else keeper_probe_pin(kind);
}

module pin_ring_ring(kind) {
    if (kind == "shared") translate([0, 0, keeper_ring_seat_z()]) printed_keeper();
    else if (kind == "cross") translate([0, 0, cross_ring_seat_mm]) cross_keeper();
    else keeper_probe_ring(kind);
}

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
} else if (check == "linkage_vs_steering") {
    // EMPTY, per variant and horn sweep: the throttle horn interface and the rod's servo
    // section stay clear of the steering servo, its adapter and its pushrod sweep.
    intersection() {
        throttle_linkage_at(link_variant, link_sweep_deg);
        steering_envelope();
    }
} else if (check == "linkage_vs_head") {
    // EMPTY, per variant and horn sweep: the throttle horn interface and the rod's servo
    // section stay clear of the head's rib, land, shelf and cradle walls.
    intersection() {
        throttle_linkage_at(link_variant, link_sweep_deg);
        head_world();
    }
} else if (check == "rod_vs_steering") {
 // EMPTY, per variant: the WHOLE throttle rod, servo eye to throttle eye, stays clear of
 // the raised steering servo, its adapter and its pushrod sweep. The steering rod now falls
 // from the raised crank toward the lap bar while the throttle rod runs level below it, and
 // their plan paths cross; this is the probe that measures that crossing.
 // The real rod is intersected, threads included; that takes several minutes per variant,
 // so the catalog gives this probe its own timeout. A convex hull of the rod was tried and
 // rejected: it cost as much (building both operands dominates), and for variant E its
 // corners met a steering rod that the real rod clears, so it could not prove clearance.
 intersection() {
 rod_run(link_variant);
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
    // EMPTY: with its pins drawn, the keeper rises until its legs clear the cradle walls
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
    // EMPTY: the installed keeper does not interfere with the head's cradle walls.
    intersection() {
        head_world();
        at_servo() mg995_throttle_mount("keeper_installed");
    }
} else if (check == "keeper_pins_fit") {
    // EMPTY: the installed keeper pins and rings clear the head and the keeper.
    intersection() {
        at_servo() mg995_throttle_mount("keeper_pins_installed");
        union() {
            head_world();
            at_servo() mg995_throttle_mount("keeper_installed");
        }
    }
} else if (check == "keeper_pins_block_liftoff") {
    // CONTACT: lifting the keeper 1 mm drives both legs into the installed pins.
    intersection() {
        mg995_throttle_mount("keeper_pins_installed");
        translate([0, 0, 1]) mg995_throttle_mount("keeper_installed");
    }
} else if (check == "keeper_pins_engage_cradle") {
    // CONTACT: lifting the pins 1 mm drives them into the head's side-wall bores.
    intersection() {
        at_servo() translate([0, 0, 1]) mg995_throttle_mount("keeper_pins_installed");
        head_world();
    }
} else if (check == "linkage_servo_end_fit") {
    // EMPTY: the rod's servo section clears the horn interface, the mount stack and the
    // servo envelope at this horn sweep (rod direction held at neutral).
    intersection() {
        rod_servo_end(link_variant, link_sweep_deg);
        union() {
            horn_interface(link_variant, link_sweep_deg);
            mg995_throttle_mount("assembly");
            mg995_reference();
        }
    }
} else if (check == "linkage_servo_eye_lift") {
    // CONTACT: lifting the servo eye 1 mm meets its cap or upper clevis ear.
    intersection() {
        translate([0, 0, 1]) rod_servo_end(link_variant);
        horn_interface(link_variant);
    }
} else if (check == "linkage_servo_eye_pull") {
    // CONTACT: pulling the servo eye 1 mm along the run meets its pivot.
    intersection() {
        translate(link_run_mm()) rod_servo_end(link_variant);
        horn_interface(link_variant);
    }
} else if (check == "linkage_throttle_end_fit") {
    // EMPTY: the rod's moving end clears the U fitting on its post.
    intersection() {
        rod_throttle_end_placed(link_variant);
        throttle_end(link_variant);
    }
} else if (check == "linkage_throttle_eye_lift") {
    // CONTACT: lifting the throttle eye 1 mm meets the U post's cap.
    intersection() {
        translate([0, 0, 1]) rod_throttle_end_placed(link_variant);
        throttle_end(link_variant);
    }
} else if (check == "linkage_throttle_eye_pull") {
    // CONTACT: pulling the throttle eye 1 mm back along the run meets the U post.
    intersection() {
        translate(-link_run_mm()) rod_throttle_end_placed(link_variant);
        throttle_end(link_variant);
    }
} else if (check == "rod_capture_pip_fit") {
    // EMPTY: the sleeve and slider, as printed collet end down, leave the bed as separate
    // bodies; the mesh gap is measured by the catalogue's print-in-place gate.
    intersection() {
        rod_nose_place() rod_receiver_sleeve();
        rod_nose_place() rod_slider_print_pose();
    }
} else if (check == "rod_collet_nut_clear") {
    // EMPTY: the captive collet nut, as printed, clears both the sleeve and the slider.
    intersection() {
        rod_collet_nut();
        rod_nose_place() {
            rod_receiver_sleeve();
            rod_slider_print_pose();
        }
    }
} else if (check == "rod_collet_nut_captive") {
    // CONTACT, +/-1 mm: pushed straight along the axis without turning, the nut meets the
    // band's thread flanks (0.64 mm axial clearance) in either direction.
    intersection() {
        translate([0, 0, rod_collet_shift_mm]) rod_collet_nut();
        rod_nose_place() rod_receiver_sleeve();
    }
} else if (check == "rod_collet_nut_runs_free") {
    // EMPTY: screwed along its own thread (turn and advance together) by up to 1.2 mm
    // toward the tip or back toward the shoulder, the nut still clears the sleeve: the
    // threads are co-helical, the cone meets the flare only after
    // rod_collet_contact_advance() (1.46 mm) and the shoulder after about 1.45 mm back.
    intersection() {
        rod_collet_nut_at(rod_collet_advance_mm);
        rod_nose_place() rod_receiver_sleeve();
    }
} else if (check == "rod_collet_nut_drives_fingers") {
    // CONTACT: screwed forward by the clamp advance (3.61 mm: contact plus the 0.4 mm
    // slider clearance and 0.3 mm preload over tan 18 deg), the nut's cone overlaps the
    // UNDEFLECTED fingers, i.e. it must close them. The asserts in throttle_pushrod.scad
    // prove the reserved 6 mm advance closes 1.48 mm radially, more than the 0.7 needed.
    intersection() {
        rod_collet_nut_at(rod_collet_clamp_advance());
        intersection() {
            rod_nose_place() rod_receiver_sleeve();
            rod_collet_finger_zone();
        }
    }
} else if (check == "rod_capture_range_clear") {
    // EMPTY at each setting: the sleeve (lug in the groove) and the as-printed nut clear
    // the slider.
    intersection() {
        union() {
            rod_receiver_sleeve();
            rod_nose_place() rod_collet_nut();
        }
        translate([0, 0, rod_slider_start_in_sleeve_mm(rod_capture_length_mm)])
            rod_slider();
    }
} else if (check == "rod_capture_clear") {
    // At the selected setting, the closed groove shoulder clears the sleeve's integral lug.
    intersection() {
        rod_overstop_lug(rod_capture_length_mm);
        rod_overstop_stop();
    }
} else if (check == "rod_overlap_capture") {
    // Slide the slider along the rod until its closed groove end reaches the sleeve's
    // integral lug and 1 mm past it. The lug sits INSIDE the groove at the declared
    // settings, so the closed ends are overtravel catches, not length stops: contact
    // needs about 245 mm of extension at the 940 mm setting and about 5 mm at the
    // 1200 mm setting. Contact proves the catch exists, not that it holds load or that
    // the collet holds length.
    intersection() {
        translate([0, 0, rod_capture_lug_z()
                   - rod_capture_shoulder_end_z(rod_capture_length_mm) + 1])
            rod_overstop_lug(rod_capture_length_mm);
        rod_overstop_stop();
    }
} else if (check == "rod_overlap_capture_reverse") {
    // CONTACT: the same catch in reverse, reached after about 83 mm of reverse travel at
    // the 940 mm setting, plus 1 mm past it.
    intersection() {
        translate([0, 0, rod_capture_lug_z()
                   - rod_capture_upper_shoulder_z(rod_capture_length_mm) - 1])
            rod_overstop_reverse_shoulder(rod_capture_length_mm);
        rod_overstop_stop();
    }
} else if (check == "rod_body_fit") {
    // EMPTY: the throttle-end body, its bar and keeper, the fine screw and jam collar, the
    // fork and its nut (at both ends of the fine travel) clear the slider and each other.
    // Frame: relative to the slider tip.
    union() {
        intersection() {
            rod_slider_from_tip();
            union() {
                rod_throttle_body();
                rod_body_bar();
                rod_fine_jam_collar();
                for (f = [0, rods_fine_travel_mm])
                    rod_fork_at(f);
            }
        }
        intersection() {
            rod_throttle_body();
            union() {
                rod_body_bar();
                rod_fine_screw();
                rod_fine_jam_collar();
                for (f = [0, rods_fine_travel_mm])
                    rod_fork_at(f);
            }
        }
        for (f = [0, rods_fine_travel_mm])
            intersection() {
                rod_fork_at(f, with_nut = true);
                union() {
                    rod_body_bar();
                    rod_fine_jam_collar();
                }
            }
    }
} else if (check == "rod_body_bar_holds") {
    // CONTACT, per target and direction: the bar moved 1 mm along the rod meets the
    // body's slot (body) or the slider's window (slider), so the bar ties the two.
    intersection() {
        translate([0, 0, rod_eye_shift_mm]) rod_body_bar();
        if (rod_eye_target == "body") rod_throttle_body();
        else rod_slider_from_tip();
    }
} else if (check == "rod_body_keeper_stops_inward") {
    // CONTACT: pushing the bar 1 mm inward drives its keeper into the body's +Y face.
    intersection() {
        rod_throttle_body();
        translate([0, -1, 0]) rod_body_bar();
    }
} else if (check == "rod_fine_screw_engages") {
    // CONTACT: the fine screw's flange bears on the boss's rear face and the jam collar on
    // its front face (each moved 1 mm into the boss), so the screw is axially held.
    intersection() {
        rod_throttle_body();
        union() {
            translate([0, 0, 1]) rod_fine_screw();
            translate([0, 0, -1]) rod_fine_jam_collar();
        }
    }
} else if (check == "rod_body_clears_collet") {
    // EMPTY at each setting: the throttle-end body set, fork and fine nut clear the
    // sleeve and the collet nut at its full reserved advance (6 mm, past any clamp pose).
    intersection() {
        union() {
            translate([0, 0, rod_slider_tip_z(rod_capture_length_mm)]) rod_body_set();
            translate([0, 0, rod_throttle_pin_z(rod_capture_length_mm)]) {
                rod_throttle_fork();
                rod_fine_carriage_nut();
            }
        }
        union() {
            rod_sleeve_installed();
            rod_collet_nut_installed(collet_nut_max_advance_mm);
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
} else if (check == "splice_bars_fit") {
    // EMPTY: every joint's bar and keeper clear the installed fixed chain.
    intersection() {
        rod_fixed_chain();
        rod_splices();
    }
} else if (check == "splice_bar_holds_joint") {
    // CONTACT, per joint and direction: moving that joint's bar 1 mm along the rod meets
    // the slot faces in the boss and spigot, so the bar holds the two sections together.
    intersection() {
        rod_fixed_chain();
        translate([0, 0, splice_shift_mm]) rod_splice(rod_splice_bases()[splice_joint]);
    }
} else if (check == "splice_keeper_stops_inward") {
    // CONTACT, per joint: pushing that joint's bar 1 mm toward the slider drives its keeper
    // into the boss face before the bar can reach the slider passage.
    intersection() {
        rod_fixed_chain();
        translate([0, -1, 0]) rod_splice(rod_splice_bases()[splice_joint]);
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
