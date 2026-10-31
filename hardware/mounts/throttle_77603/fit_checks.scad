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
use <../pushrod.scad>
use <../hitch_arm_layer.scad>   // legacy arm head, for the attachment probe   // legacy steering rod geometry, for the linkage-clearance probe

check = "u_lever_passes";

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

module saddle_wrap_only() {
    // Only the printed wrap halves, placed at the same station the assembly uses. The flange
    // hook is a separate piece that legitimately occupies the overhang.
    translate(throttle_servo_base_mm() + [0, 0, -riser_z_mm * 2])
        for (half = [-1, 1]) {
            mg995_throttle_mount("saddle_left");
            mg995_throttle_mount("saddle_right");
        }
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
} else if (check == "saddle_root_land") {
    // POSITIVE: the printed foot really contacts the legacy arm head where it bolts on.
    // Verified band: X 187.3..209.6, Y 141..157, Z 430.2..438.2. This probe exists so the
    // attachment cannot silently regress when placement parameters move.
    intersection() {
        translate([servo_station_x, servo_shaft_y(), arm_tip_z])
            mg995_throttle_mount("bracket_foot");
        arm_head_assembly();
    }
} else if (check == "mount_vs_linkage") {
    // EMPTY: the throttle stack must not intrude on the steering pushrod sweep.
    intersection() {
        mg995_throttle_mount("assembly");
        for (side = [1]) rod_servo_sweep_band(side);
    }
} else if (check == "mount_vs_steering") {
    // EMPTY: the throttle stack must not intrude on the steering servo CASE or its horn
    // sweep. This is the probe that caught the original column-mounted design, which put the
    // throttle cradle inside the steering servo's own envelope.
    intersection() {
        mg995_throttle_mount("assembly");
        servo_body_envelope(1);
    }
} else if (check == "cradle_lug_capture") {
    // Positive: the printed cradle carries material in the band below its own recess floor
    // (i.e. it has a floor, so the servo body cannot fall through).
    intersection() {
        mg995_throttle_mount("cradle");
        translate([0, 0, cradle_t_mm / 2]) cube([cradle_w_mm, cradle_len_mm, cradle_t_mm]);
    }
} else if (check == "keeper_capture") {
    // Positive: the printed keeper is a real bar spanning the cradle mouth.
    intersection() {
        mg995_throttle_mount("keeper");
        cube([cradle_w_mm, cradle_len_mm, keeper_t_mm]);
    }
} else if (check == "rod_overlap_capture") {
    intersection() {
        throttle_pushrod("sleeve");
        throttle_pushrod("slider");
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
