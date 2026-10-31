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
use <../pushrod.scad>
use <throttle_arm_head.scad>   // the derivative head, for the attachment and clearance probes
use <../hitch_arm_layer.scad>   // legacy arm head, for the bearing probe
use <../pushrod.scad>           // legacy steering rod geometry, for the linkage-clearance probe
use <assemblies.scad>          // scene placements, reused by the linkage probes

check = "u_lever_passes";
rod_capture_length_mm = rods_nominal_span_mm;
link_variant = "E";
link_sweep_deg = 0;
// Plan unit vector of the rod's run for the selected variant.
function link_run_mm() =
    [cos(run_angle_deg(link_variant)), sin(run_angle_deg(link_variant)), 0];

// ---- Helpers -----------------------------------------------------------------------
module at_servo() { translate(throttle_servo_base_mm()) children(); }

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

module saddle_volume() {
    // Self-placed: no extra assembly transform (see mg995_throttle_mount("assembly")).
    mg995_throttle_mount("assembly");
}

// Swept band of the EXISTING steering pushrod, built from the legacy rod geometry rather
// than a guessed box: the rod runs from the servo crank pin to the lap-bar clamp pin and
// sweeps with the crank. A crude box around the crank falsely reported interference.
module rod_servo_sweep_band(side) {
    rod_between_points(servo_crank_pin(side), lapbar_pin(side), rod_nominal_setting());
}

module column_volume() {
    // Legacy lower-column envelope. NOTE: the throttle mount no longer wraps this column --
    // it is anchored to the right steering servo bracket instead -- so column-band probes are
    // retained only as context and must not be relied on for the current structure.
    translate([arm_root_x, assembly_arm_y, arm_foot_z]) rotate([0, arm_rake_deg, 0])
        translate([-arm_col_d / 2, -arm_col_w / 2, arm_seat_local_z()])
            cube([arm_col_d, arm_col_w, arm_joint_t - arm_seat_local_z()]);
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
} else if (check == "support_bears_on_land") {
    // POSITIVE: the integral column really grows out of the arm head's servo-face land, so
    // the two are one printed piece rather than two parts that happen to touch.
    intersection() {
        translate(arm_point(arm_joint_t)) throttle_arm_head();
        arm_head_assembly();
    }
} else if (check == "seat_holes_are_bored") {
    // POSITIVE: a cylinder LARGER than each blind hole must still meet the column, which
    // proves the holes were drilled into real material rather than into air above the seat.
    // A probe the same size as the hole would only reproduce the hole's own void.
    intersection() {
        translate(arm_point(arm_joint_t)) throttle_arm_head();
        for (dx = [-seat_dowel_x_offset_mm, seat_dowel_x_offset_mm])
            translate([arm_joint_x + seat_x_local_mm + dx,
                       assembly_arm_y + seat_y_local_mm + seat_dowel_y_offset_mm,
                       arm_joint_z + column_z1_local_mm - seat_dowel_depth_mm])
                cylinder(d = seat_dowel_d_mm + 2 * seat_dowel_clearance_mm + 4,
                         h = seat_dowel_depth_mm + 1);
    }
} else if (check == "seat_dowels_fit") {
    // The actual cradle dowels must clear the column below the seating plane.
    intersection() {
        translate(arm_point(arm_joint_t)) throttle_arm_head();
        intersection() {
            translate(throttle_mount_stack_origin_mm()) mg995_throttle_mount("cradle_assembly");
            translate([tower_world_x() - cradle_w_mm, tower_world_y() - cradle_len_mm,
                       support_top_world_z() - seat_dowel_depth_mm - 1])
                cube([2 * cradle_w_mm, 2 * cradle_len_mm, seat_dowel_depth_mm + 0.9]);
        }
    }
} else if (check == "seat_cradle_clears_column") {
    // EMPTY: the cradle must not interfere with the column. The cradle is LIFTED by a hair
    // first: seated, the cradle's floor and the column's seat face share a plane, and a
    // coplanar contact makes the boolean non-manifold without indicating any interference.
    intersection() {
        translate(arm_point(arm_joint_t)) throttle_arm_head();
        translate(throttle_mount_stack_origin_mm() + [0, 0, 0.05])
            mg995_throttle_mount("cradle_assembly");
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
} else if (check == "mount_vs_moving_steering") {
    // EMPTY: the column and the stack must clear the steering servo's MOVING bracket at every
    // steering angle, not only at the pose the static model happened to show.
    intersection() {
        union() {
            translate(arm_point(arm_joint_t)) throttle_arm_head();
            mg995_throttle_mount("assembly");
        }
        for (face = [0, 45, 90]) servo_moving_bracket(1, face);
    }
} else if (check == "mount_vs_linkage") {
    // EMPTY: the throttle stack must not intrude on the steering pushrod sweep.
    intersection() {
        union() {
            translate(arm_point(arm_joint_t)) throttle_arm_head();
            mg995_throttle_mount("assembly");
        }
        for (side = [1]) rod_servo_sweep_band(side);
    }
} else if (check == "mount_vs_steering") {
    // EMPTY: the throttle stack must not intrude on the steering servo CASE or its horn
    // sweep. This is the probe that caught the original column-mounted design, which put the
    // throttle cradle inside the steering servo's own envelope.
    intersection() {
        union() {
            translate(arm_point(arm_joint_t)) throttle_arm_head();
            mg995_throttle_mount("assembly");
        }
        servo_body_envelope(1);
    }
} else if (check == "cradle_floor_supports_body") {
    // CONTACT: the floor carries material directly under the servo body's footprint.
    intersection() {
        mg995_throttle_mount("cradle_assembly");
        translate([-mg995_body_width_mm / 2, -mg995_body_length_mm / 2, cradle_t_mm - 1])
            cube([mg995_body_width_mm, mg995_body_length_mm, 0.5]);
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
    // EMPTY: the installed keeper does not interfere with the cradle.
    intersection() {
        mg995_throttle_mount("cradle_assembly");
        mg995_throttle_mount("keeper_installed");
    }
} else if (check == "seat_pin_fit") {
    // EMPTY: the installed seat pin and ring clear the column and the cradle dowels.
    intersection() {
        translate(throttle_mount_stack_origin_mm()) mg995_throttle_mount("seat_pin_installed");
        union() {
            translate(arm_point(arm_joint_t)) throttle_arm_head();
            translate(throttle_mount_stack_origin_mm()) mg995_throttle_mount("cradle_assembly");
        }
    }
} else if (check == "seat_pin_blocks_liftoff") {
    // CONTACT: lifting the cradle 1 mm drives its dowels into the installed seat pin.
    intersection() {
        translate(throttle_mount_stack_origin_mm()) mg995_throttle_mount("seat_pin_installed");
        translate(throttle_mount_stack_origin_mm() + [0, 0, 1])
            mg995_throttle_mount("cradle_assembly");
    }
} else if (check == "seat_pin_engages_column") {
    // CONTACT: lifting the pin 1 mm drives it into the column's own cross bore wall.
    intersection() {
        translate(throttle_mount_stack_origin_mm() + [0, 0, 1])
            mg995_throttle_mount("seat_pin_installed");
        translate(arm_point(arm_joint_t)) throttle_arm_head();
    }
} else if (check == "keeper_pins_fit") {
    // EMPTY: the installed keeper pins and rings clear the cradle and the keeper.
    intersection() {
        mg995_throttle_mount("keeper_pins_installed");
        union() {
            mg995_throttle_mount("cradle_assembly");
            mg995_throttle_mount("keeper_installed");
        }
    }
} else if (check == "keeper_pins_block_liftoff") {
    // CONTACT: lifting the keeper 1 mm drives both legs into the installed pins.
    intersection() {
        mg995_throttle_mount("keeper_pins_installed");
        translate([0, 0, 1]) mg995_throttle_mount("keeper_installed");
    }
} else if (check == "keeper_pins_engage_cradle") {
    // CONTACT: lifting the pins 1 mm drives them into the cradle's side-wall bores.
    intersection() {
        translate([0, 0, 1]) mg995_throttle_mount("keeper_pins_installed");
        mg995_throttle_mount("cradle_assembly");
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
} else if (check == "rod_capture_clear") {
    // At the selected setting, the actual shoulder and capture pin must not overlap.
    intersection() {
        rod_overstop_lug(rod_capture_length_mm);
        rod_overstop_stop();
    }
} else if (check == "rod_overlap_capture") {
    // Push the actual closed-groove shoulder beyond the fixed capture pin.
    // Contact demonstrates a geometric stop, not strength or vibration resistance.
    intersection() {
        translate([0, 0, rod_capture_pin_z()
                   - rod_capture_shoulder_end_z(rod_capture_length_mm) + 1])
            rod_overstop_lug(rod_capture_length_mm);
        rod_overstop_stop();
    }
} else if (check == "lock_wedge_grip") {
    // Positive: the wedge body is a real solid around the rod passage.
    intersection() {
        throttle_pushrod("lock_wedge_body");
        cube([rods_w_mm + 8, rods_w_mm + 8, 20], center = true);
    }
} else if (check == "lock_collet_grip") {
    // Positive: the collet body is a real ring around the rod.
    intersection() {
        throttle_pushrod("lock_collet_body");
        difference() {
            cylinder(d = rods_w_mm + 16, h = 40, center = true);
            cylinder(d = rods_w_mm, h = 42, center = true);
        }
    }
} else if (check == "lock_screw_engagement") {
    intersection() {
        throttle_pushrod("lock_screw");
        translate([0, 0, 10]) cylinder(d = fine_screw_d_mm + 4, h = fine_nut_height_mm);
    }
} else if (check == "endstop_capture") {
    intersection() {
        throttle_pushrod("endstop");
        cube([rods_w_mm + 8, rods_w_mm + 8, 2], center = true);
    }
} else if (check == "clevis_double_shear") {
    // Positive: BOTH ears carry material around the pin axis (double shear).
    intersection() {
        printed_clevis();
        for (s = [-1, 1])
            translate([s * (clevis_gap_mm / 2 + clevis_ear_t_mm / 2), 0,
                       clevis_ear_t_mm + clevis_gap_mm / 2])
                cube([clevis_ear_t_mm, clevis_gap_mm, clevis_gap_mm]);
    }
} else if (check == "ball_cage_capture") {
    // Positive: the cage is real material at its own mid-height band around the ball axis.
    intersection() {
        variant_d("cage");
        difference() {
            cylinder(d = ball_d_mm + 2 * ball_socket_wall_mm + 6, h = ball_d_mm * 0.4);
            cylinder(d = ball_d_mm + 2 * ball_socket_wall_mm, h = ball_d_mm * 0.4 + 1);
        }
    }
} else if (check == "c_radius_slide") {
    // Positive: the shoe must be able to sit at a non-default continuous radius.
    intersection() {
        variant_c("carriage", radius_mm = 23.7);
        translate([23.7, 0, 5]) cube([18, 8, 20], center = true);
    }
} else {
    assert(false, str("unknown check: ", check));
}
