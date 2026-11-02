// CSG probes for the print-in-place throttle pushrod (throttle_pushrod.scad) and its linkage.
//
// Each probe intersects REAL built solids: EMPTY means no interference, NON-EMPTY (contact)
// means the named geometry carries material where the probe samples it. A contact probe is
// NOT proof of retention, clamp force or holding; those are coupon tests. Run one probe with
//   openscad --export-format binstl -o /dev/null -D 'check="name"' rod_checks.scad
// and treat any WARNING as a failure (an undefined name renders empty: a false pass).
//
// Frames: rod probes work in the rod's TUBE-AXIS frame at installed stations (see
// throttle_pushrod.scad); the wheel probes in the fork frame (wheel's servo face z = 0) or the
// slider-tip frame; the linkage probes in the world frame through assemblies.scad.
include <parameters.scad>
use <throttle_pushrod.scad>
use <../pip_rod_joints.scad>
use <assemblies.scad>
use <fit_checks.scad>            // steering_envelope(), head_world(), throttle_linkage_at()
use <mg995_throttle_mount.scad>

check = "rod_capture_pip_fit";
link_variant = "E";
link_sweep_deg = 0;              // horn sweep about the neutral pose, degrees
rod_setting = rods_nominal_span_mm;  // pin-to-pin setting, mm
rod_shift = [0, 0, 1];           // forced displacement for captive/drive probes, mm
rod_joint = 0;                   // 0: servo/middle joint, 1: middle/sleeve joint
rod_phi = 45;                    // lever angle: 0 as printed, 90 closed
rod_fine = 0;                    // fine (thumbwheel) extension, mm

// Plan unit vector of the rod's run for the selected variant (from the servo eye).
function rc_run() =
    [cos(run_angle_deg(link_variant)), sin(run_angle_deg(link_variant)), 0];
// Unit vector of the lever side (rod frame) and the closed lever's outward play.
function rc_side() = [cos(rod_side_deg()), sin(rod_side_deg()), 0];
// Under the clamp reaction a closed lever slides outward on its pins by the bore's play.
function rc_play() = pip_get(rod_cam_spec(), "gap") * rc_side();
// The seated slider moved a further 0.45 mm away from the cam: past the flat's reach
// (preload 0.3) but inside the corner's (preload + bump 0.6).
module rc_slider_far(length_mm) {
    translate(-0.45 * rc_side()) rod_slider_at(length_mm, seat = true);
}
// Fork and wheel installed at a setting (rod frame), without the slider.
module rc_fork_installed(length_mm) {
    translate([0, 0, rod_slider_tip_z(length_mm)])
        rod_fork_set(rod_fine_extension_mm(length_mm));
}
// A U-post-sized pin (6.2 mm) through the throttle eye's plane at rod-local z.
module rc_eye_pin(z) {
    translate([0, 0, z]) rotate([0, 90, 0]) cylinder(d = 6.2, h = 6, center = true, $fn = 48);
}

// ---- Dispatch ----------------------------------------------------------------------------
if (check == "rod_capture_pip_fit") {
    // EMPTY: as printed (tip down) the sleeve and slider are separate bodies, the lug in the
    // groove and the thread inside the bore with the gap all round.
    intersection() {
        rod_sleeve();
        rod_slider_print_pose();
    }
} else if (check == "rod_cam_pip_fit") {
    // EMPTY: the cam lever as printed clears the sleeve (cheeks, pins, window) and the slider.
    intersection() {
        rod_cam_lever(0);
        union() {
            rod_sleeve();
            rod_slider_print_pose();
        }
    }
} else if (check == "rod_cam_captive") {
    // CONTACT, per direction: pushed rod_shift from its print pose, the cam lever meets the
    // sleeve's pins or cheeks, so it cannot leave the sleeve.
    intersection() {
        translate(rod_shift) rod_cam_lever(0);
        rod_sleeve();
    }
} else if (check == "rod_cam_swing_clear") {
    // EMPTY at each angle: swinging from open to closed, the cam lever never meets its own
    // housing (lobe in the window, arm outside the wall, landing on the face when closed).
    intersection() {
        rod_cam_lever(rod_phi);
        rod_sleeve();
    }
} else if (check == "rod_cam_open_clear") {
    // EMPTY at each setting: the open cam lever clears the slider, so the slide is free.
    intersection() {
        rod_cam_lever(0);
        rod_slider_at(rod_setting);
    }
} else if (check == "rod_cam_clamps") {
    // CONTACT at each setting: closed, and pushed outward by its pins' full play, the lever's
    // flat still overlaps the seated slider by the preload: it clamps across the whole travel.
    intersection() {
        translate(rc_play()) rod_cam_lever(90);
        rod_slider_at(rod_setting, seat = true);
    }
} else if (check == "rod_cam_overcentre") {
    // CONTACT: at its corner angle (about 15 degrees before closed) the lever reaches a slider
    // set 0.45 mm beyond seating: the corner passes further than the closed flat (next probe).
    intersection() {
        translate(rc_play()) rod_cam_lever(90 - pip_get(rod_cam_spec(), "corner_deg"));
        rc_slider_far(rod_setting);
    }
} else if (check == "rod_cam_rest") {
    // EMPTY: closed, the flat stops short of that same far slider: the closed lever sits in a
    // valley behind the corner (over-centre), so opening it needs force.
    intersection() {
        translate(rc_play()) rod_cam_lever(90);
        rc_slider_far(rod_setting);
    }
} else if (check == "rod_capture_range_clear") {
    // EMPTY at each setting: the sleeve (lug in the groove) clears the slider.
    intersection() {
        rod_sleeve();
        rod_slider_at(rod_setting);
    }
} else if (check == "rod_overlap_capture") {
    // CONTACT at each setting: forced 1 mm past the overtravel catch (the groove's servo end
    // reaching the lug), the slider meets the sleeve's integral lug.
    intersection() {
        translate([0, 0, rod_lug_station()
                   - (rod_slider_start_z(rod_setting) + rod_groove_ends()[0]) + 1])
            rod_slider_at(rod_setting);
        rod_sleeve();
    }
} else if (check == "rod_overlap_capture_reverse") {
    // CONTACT: retracted 1 mm past the reverse catch (the groove's tip end reaching the lug's
    // top), the slider meets the lug.
    intersection() {
        translate([0, 0, rod_lug_station() + 7
                   - (rod_slider_start_z(rod_setting) + rod_groove_ends()[1]) - 1])
            rod_slider_at(rod_setting);
        rod_sleeve();
    }
} else if (check == "rod_fork_clears_sleeve") {
    // EMPTY at each setting: the fork and wheel clear the sleeve and the closed cam lever.
    intersection() {
        rc_fork_installed(rod_setting);
        union() {
            rod_sleeve();
            rod_cam_lever(90);
        }
    }
} else if (check == "rod_wheel_pip_fit") {
    // EMPTY: as printed the wheel clears the fork's posts, ridges, pawls and roof.
    intersection() {
        rod_wheel();
        rod_fork();
    }
} else if (check == "rod_wheel_captive") {
    // CONTACT, per direction: pushed rod_shift from its print pose, the wheel meets the posts'
    // ridges or the pawls: it cannot leave the fork.
    intersection() {
        translate(rod_shift) rod_wheel();
        rod_fork();
    }
} else if (check == "rod_wheel_detent_click") {
    // CONTACT, both ways: turned half a notch (rod_phi degrees) the wheel's rim lands on the
    // pawl teeth, so a pawl must flex outward to pass: a click.
    intersection() {
        rod_wheel_turned(rod_phi);
        rod_fork();
    }
} else if (check == "rod_wheel_detent_seat") {
    // EMPTY, both ways: turned a whole notch (rod_phi degrees) the pawls seat again.
    intersection() {
        rod_wheel_turned(rod_phi);
        rod_fork();
    }
} else if (check == "rod_wheel_runs_free") {
    // EMPTY at each fine extension (slider-tip frame): the wheel at its detent seat on the
    // slider's thread clears the slider: the threads mesh with their gap all round.
    intersection() {
        translate([0, 0, -rod_tip_back() + rod_fine]) rod_wheel_turned(rod_wheel_seat(rod_fine));
        rod_slider_from_tip();
    }
} else if (check == "rod_wheel_drives") {
    // CONTACT, per extension and direction: pushed 1 mm along the axis without turning, the
    // seated wheel meets the slider's thread flanks: the thread carries the fork.
    intersection() {
        translate([0, 0, -rod_tip_back() + rod_fine] + rod_shift)
            rod_wheel_turned(rod_wheel_seat(rod_fine));
        rod_slider_from_tip();
    }
} else if (check == "rod_fork_clear") {
    // EMPTY at each fine extension (slider-tip frame): the fork body, posts and pawls clear
    // the slider's thread and tip.
    intersection() {
        translate([0, 0, -rod_tip_back() + rod_fine]) rod_fork();
        rod_slider_from_tip();
    }
} else if (check == "rod_joint_pip_fit") {
    // EMPTY, per joint: the joint lever as printed clears its socket section.
    intersection() {
        rod_joint_lever(rod_joint, 0);
        rod_joint_socket_part(rod_joint);
    }
} else if (check == "rod_joint_captive") {
    // CONTACT, per joint and direction: pushed rod_shift from its print pose, the joint lever
    // meets its section's pins or cheeks.
    intersection() {
        translate(rod_shift) rod_joint_lever(rod_joint, 0);
        rod_joint_socket_part(rod_joint);
    }
} else if (check == "rod_joint_swing_clear") {
    // EMPTY, per joint and angle: the joint lever and its tooth swing from open to closed
    // without meeting the socket section (clamp window and tooth arc window).
    intersection() {
        rod_joint_lever(rod_joint, rod_phi);
        rod_joint_socket_part(rod_joint);
    }
} else if (check == "rod_joint_insert_clear") {
    // EMPTY, per joint: with the lever open the spigot slides fully home past it: the
    // centred spigot clears the socket section and the open lever.
    intersection() {
        rod_joint_spigot_part(rod_joint);
        union() {
            rod_joint_socket_part(rod_joint);
            rod_joint_lever(rod_joint, 0);
        }
    }
} else if (check == "rod_joint_clamps") {
    // CONTACT, per joint: closed and pushed outward by its pins' full play, the lever's flat
    // still overlaps the seated spigot by the preload.
    intersection() {
        translate(rc_play()) rod_joint_lever(rod_joint, 90);
        rod_joint_spigot_seated(rod_joint);
    }
} else if (check == "rod_joint_tooth_engages") {
    // CONTACT, per joint: the closed tooth reaches below the seated spigot's face (into the
    // spigot's square envelope): it is in the pocket, not merely against the face.
    intersection() {
        rod_joint_tooth(rod_joint, 90);
        pip_lever_seat(rod_joint_spec(), rod_side_deg()) rod_spigot_envelope(rod_joint);
    }
} else if (check == "rod_joint_tooth_clear") {
    // EMPTY, per joint: the closed tooth clears the pocket in the seated spigot and the arc
    // window in the boss wall.
    intersection() {
        rod_joint_tooth(rod_joint, 90);
        union() {
            rod_joint_spigot_seated(rod_joint);
            rod_joint_socket_part(rod_joint);
        }
    }
} else if (check == "rod_joint_blocks_withdrawal") {
    // CONTACT, per joint (and with the lever displaced along the rod, rod_shift): the seated
    // spigot pulled 1 mm out of its socket meets the closed tooth. The lock is positive.
    intersection() {
        translate(rod_shift) rod_joint_tooth(rod_joint, 90);
        translate([0, 0, 1]) rod_joint_spigot_seated(rod_joint);
    }
} else if (check == "rod_joint_blocks_rotation") {
    // CONTACT, per joint and direction: the spigot turned rod_phi degrees about the rod axis
    // (its square's corners meet the socket at about 2.1 degrees) meets the socket: the
    // square keys the joint against rotation.
    intersection() {
        rotate([0, 0, rod_phi]) rod_spigot_envelope(rod_joint);
        rod_joint_socket_part(rod_joint);
    }
} else if (check == "rod_joint_tooth_bears_on_boss") {
    // CONTACT, per joint: the closed tooth dragged 1 mm toward the mouth by a withdrawing
    // spigot meets the inner wall of its arc window in the boss. The withdrawal load therefore
    // runs spigot -> tooth -> boss wall, not through the lever's pins (which have ~1 mm of
    // play along the rod, see rod_*_captive).
    intersection() {
        translate([0, 0, 1]) rod_joint_tooth(rod_joint, 90);
        rod_joint_socket_part(rod_joint);
    }
} else if (check == "rod_span_reachable") {
    // EMPTY at each limit: a 6.2 mm post through rod-local z = setting passes the throttle
    // eye's 6.6 mm bore without touching the moving end: the eye is AT the setting.
    intersection() {
        rc_eye_pin(rod_setting);
        throttle_pushrod("throttle_end", length_mm = rod_setting);
    }
} else if (check == "rod_span_eye_located") {
    // CONTACT at each limit: the same post 1 mm further along meets the eye plate, so the
    // previous probe passes through the bore, not through empty space.
    intersection() {
        rc_eye_pin(rod_setting + 1);
        throttle_pushrod("throttle_end", length_mm = rod_setting);
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
        translate(rc_run()) rod_servo_end(link_variant);
        horn_interface(link_variant);
    }
} else if (check == "linkage_throttle_end_fit") {
    // EMPTY: the rod's moving end (slider, fork, wheel) clears the U fitting on its post.
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
        translate(-rc_run()) rod_throttle_end_placed(link_variant);
        throttle_end(link_variant);
    }
} else if (check == "linkage_vs_steering") {
    // EMPTY, per variant and horn sweep: the horn interface and the rod's servo section (with
    // its closed joint lever) clear the steering servo, its adapter and its pushrod sweep.
    intersection() {
        throttle_linkage_at(link_variant, link_sweep_deg);
        steering_envelope();
    }
} else if (check == "linkage_vs_head") {
    // EMPTY, per variant and horn sweep: the same linkage clears the head's rib, land, shelf
    // and cradle walls.
    intersection() {
        throttle_linkage_at(link_variant, link_sweep_deg);
        head_world();
    }
} else if (check == "rod_vs_steering") {
    // EMPTY, per variant: the WHOLE throttle rod, servo eye to throttle eye with all three
    // levers closed, clears the raised steering servo, its adapter and the falling steering
    // pushrod whose plan path it crosses. Slow: the real rod and three whole steering rods.
    intersection() {
        rod_run(link_variant);
        steering_envelope();
    }
} else {
    assert(false, str("unknown rod check: ", check));
}
