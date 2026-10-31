// Segmented, fully printed 77603 throttle pushrod with continuous coarse and fine adjustment.
//
// PROVISIONAL DESIGN.  The 940-1200 mm setting range is the requested illustrative
// adjustment range.  It is NOT a measured mower span, NOT a measured throttle stroke, and
// NOT a verified lever path.  Nothing in this file proves fit, load capacity, wear life,
// or actual travel.  Those remain coupon, bench, and mower tests.
//
// Mechanical concept
// -------------------
// * A fixed chain of printed square-tube sections is joined by keyed ring spigots.  Each
//   joint is retained by a separately printed transverse bar and a separately printed
//   keeper, so a joint cannot open by friction alone.
// * A fixed receiver sleeve surrounds a 21.2 mm square slider.  A printed capture pin
//   passes through the sleeve wall into a LONGITUDINAL, closed-ended groove in the
//   slider.  The pin may travel along that groove for continuous coarse adjustment, but
//   it cannot leave the groove, and the groove's closed ends stop the slider from
//   separating.  If every adjustment clamp is released or slips, the linkage remains
//   positively connected.  The end capture prevents separation; it does NOT guarantee
//   return-to-idle.
// * Each coarse clamp has an independent secondary keeper.  Wedge/cam, split collet, and
//   printed lead screw are distinct alternatives on the same sleeve/slider skeleton; none
//   is silently selected by this file.
// * A separate printed lead-screw stage gives up to 20 mm of nominally continuous fine
//   trim.  Printed screw accuracy, anti-backdrive life, and lever-fit accuracy are NOT
//   claimed.
//
// Axes and mating interfaces
// --------------------------
// * Rod-local +Z is the rod's longitudinal direction, from the servo pin toward the
//   throttle pin.
// * The servo pin centre is rod-local (0,0,0).
// * Both ends carry a horizontal EYE PLATE, 4 mm thick in rod-local X, 12 mm wide in Y,
//   with a 6.6 mm cut bore along X. The servo eye's bore centre is the origin; the
//   throttle eye's bore is at z = rod_throttle_pin_z(length). Both eyes lie in the
//   x = 0 plane.
// * The tube axis is CRANKED rod_eye_crank_mm above that plane (+X), so the 30 mm tube
//   clears the servo, keeper and horn hardware below the eye plane. Each eye plate runs
//   rod_eye_reach_mm out to the tube and is fused along the tube's -X wall.
// * Variants A-D drop the servo eye over a pivot post and trap it with a printed cap;
//   variant E holds it between two clevis ears. The throttle eye drops over the U
//   fitting's pivot post and is trapped by its cap. Both pivots are single shear except
//   variant E.
// * The fine stage, its fork arm and lug are on the tube's +X side, i.e. above the rod.
//
// The scene places the rod with rotate([0, -90, 0]) and a yaw, which maps rod-local
// X -> world +Z and rod-local Z -> the run direction. Both eye bores are vertical.
//
// Installation assumptions
// -----------------------
// * Assemble the slider into the receiver sleeve from the chain end BEFORE fitting the
//   sleeve to the fixed chain.  The slider's capture groove then receives the printed
//   capture pin, and the keeper is fitted last.
// * Printed pins, bars, keepers, and the lead screw are assembly/adjustment retainers,
//   not indexed length selectors.  There are no holes, notches, or detents in the length
//   adjustment path.
// * The screw's jam collar and anti-backdrive lip are friction features, not certified
//   load locks.  Printed clearances, material, loads, and wear parts still require
//   coupon and bench testing.
include <parameters.scad>
use <printed_joints.scad>

// ---- Section -----------------------------------------------------------------------
rod_w = rods_w_mm;                          // 30 mm outer square
rod_bore = rods_bore_mm;                    // 22 mm bore
rod_slider = rods_slider_mm;                // 21.2 mm keyed slider
rod_overlap = rods_min_overlap_mm;          // requested 60 mm minimum
rod_coarse_travel = rods_coarse_travel_mm;  // requested 240 mm
rod_fine_travel = rods_fine_travel_mm;       // requested 20 mm

// ---- Local print plan ---------------------------------------------------------------
// These are printable-section lengths and a kinematic layout.  They are not mower
// measurements.
rod_fixed_chain_len = 664;   // servo section + two chain sections, to the sleeve boss
rod_receiver_len = 216;      // receiver sleeve
rod_sleeve_end_z = rod_fixed_chain_len + rod_receiver_len;   // 880
rod_segment_lens = [278, 256, 246];  // printed section heights (eye section first)
rod_segment_offsets = [0, 232, 448];
rod_slider_len = 360;
rod_min_pin_to_pin = 940;    // illustrative shortest servo-pin to throttle-pin distance
rod_slider_start_min = rod_min_pin_to_pin - rod_slider_len - 8;   // 572
// The fork's bore sits 8 mm ahead of the slider's tip: the fork's plate is 9.8 mm
// thick along the rod and is centred on the bore, so it clears the slider's tip.
rod_fork_offset = 8;
rod_receiver_boss_len = 40;
rod_spigot_len = 40;
rod_spigot_od = 29.6;
rod_spigot_id = rod_bore;
rod_receiver_socket_od = 30.8;
rod_receiver_boss_od = 36;

rod_capture_pin_d = 5;
rod_capture_groove_w = 6;
rod_capture_groove_depth = 4.6;
rod_capture_groove_z0 = 15;
rod_capture_groove_z1 = 345;
rod_capture_z = 200;         // sleeve-local cross-pin station
rod_lock_z0 = 160;           // sleeve-local clamp band
rod_lock_z1 = 205;

rod_eye_plate_t = 4;           // vertical thickness of both eye plates
rod_eye_plate_w = 12;          // width across the run
rod_eye_bore_d = 5.8;
rod_eye_fork_gap = 4.8;        // 4 mm plate + 0.4 mm clearance per side (variant E)
// The tube axis sits this far above the eye plane. The eye plate's top face is then
// 1 mm inside the tube's -X wall, so the two fuse along rod_eye_fuse_len.
rod_eye_crank_mm = rod_w / 2 + rod_eye_plate_t / 2 - 1;      // 16
rod_eye_reach_mm = 18;         // servo bore to the tube's start, clear of horn hardware
rod_eye_fuse_len = 30;

// ---- Fine stage --------------------------------------------------------------------
// The fine stage is a screw running PARALLEL to the rod, i.e. along rod-local Z, offset
// in +X clear of the slider.  The printed nut is captive in the FORK's lug; the screw's
// plain end journals in the SLIDER's boss.  Turning the screw therefore slides the fork
// along the rod, which is the fine trim.  The screw, the nut, the lug's pocket and the
// slider's boss all share ONE axis, so the fine stage cannot bind.
//
// The screw's flange bears on the fork's front plate, which is AHEAD of the slider's tip
// at every setting, so the flange never fouls the slider.
rod_fine_screw_x = 32;          // the fine axis, outboard of the slider in +X
rod_fine_screw_len = 85;        // tip + thread + the flange
rod_fine_flange_z0 = -45;       // the slider's flange, relative to the slider's tip
rod_fine_flange_z1 = -29;
rod_fine_flange_x0 = rod_slider / 2 - 1; // overlap the slider to keep the flange attached
rod_fine_flange_x1 = 58;
rod_fine_boss_od = 30;          // the journal boss's outside
rod_fine_boss_z0 = -74;         // the boss, relative to the slider's tip: it journals
rod_fine_boss_z1 = -30;         // the screw's plain end over the whole fine travel
rod_fine_nut_bore_d = 26.1;     // the printed nut's outside, with a press fit
rod_fine_nut_h = 10;            // the printed nut's height (fine_nut_height_mm)
rod_fine_screw_bore_d = 16.8;   // the screw's clearance through the boss and the lug
// The screw's flange bears on the fork's front plate, at this station.
rod_fine_screw_z1 = 14;

// ---- Throttle eye on the fine-stage fork ---------------------------------------------
// The fork's origin is the throttle eye's bore, in the eye plane (x = -crank relative
// to the tube axis). The eye plate reaches back under the slider, a side web beside
// the slider rises to the arm, and the arm carries the captive fine nut's lug.
rod_fork_eye_back = 16;        // eye plate behind the bore, under the slider tip
rod_fork_eye_front = 6;        // eye plate ahead of the bore
rod_fork_web_y0 = rod_slider / 2 + 0.9;   // beside the slider
rod_fork_web_y1 = 16;
rod_fork_arm_x0 = 8;            // the arm reaching from the web to the fine screw
rod_fork_arm_x1 = rod_fine_screw_x + 18;
rod_fork_arm_y0 = 10;
rod_fork_arm_y1 = 26;
rod_fork_arm_z0 = 4.9;          // the arm sits ahead of the U fitting's cap
rod_fork_arm_z1 = 15;
// The fine nut's pocket, in the lug, on the screw's own axis.
rod_fork_nut_z0 = -28;          // the nut's pocket, open at the lug's rear face
rod_fork_nut_z1 = -28 + rod_fine_nut_h;
rod_fork_lug_y = 18;            // half width of the nut lug: it carries the 26 mm nut
rod_fork_lug_z1 = 10;           // the lug's front face
// The fork's bore station is rod_fork_offset, defined with the slider's start above.

// ---- Coarse-lock lead-screw clamp ---------------------------------------------------
// The screw's axis is RADIAL (rod-local X): it presses the shoe against the slider's
// +X flat.  The printed nut is captive in a boss that grows outboard from the saddle's
// +X bridge, so the screw's axis and the boss's axis are the same line.
rod_lead_axis_z = 32;           // the screw's axis station along the sleeve
rod_lead_screw_len = 26;
rod_lead_boss_x0 = rod_w / 2 + 3;   // the boss, outboard of the saddle's bridge
rod_lead_boss_x1 = rod_w / 2 + 20;
rod_lead_boss_od = 30;          // it carries the printed nut
rod_lead_nut_bore_d = 26.1;     // the printed nut's outside, with a press fit
rod_lead_screw_bore_d = 16.8;   // the screw's clearance through the boss
rod_lead_nut_x = rod_w / 2 + 12; // the nut's own station along the screw

// ---- Continuous adjustment mapping ----------------------------------------------------
// Coarse travel is used first; only the final 20 mm is represented by fine trim.  Both
// stages are continuous.  This mapping is deterministic for scene generation and is not
// advice about how an installer should choose a setting.
//
// The pin-to-pin span is measured from the SERVO PIN (rod-local z = 0) to the THROTTLE
// PIN, which is the fork's bore centre.  The fork sits a fixed distance ahead of the
// slider's tip, and the FINE stage slides it along the rod, so:
//
//   servo pin ......... z = 0
//   slider tip ........ z = rod_slider_tip_z
//   fork bore ......... z = rod_throttle_pin_z   (the throttle pin's station)
//
// The mapping is arranged so the shortest setting gives exactly 940 mm and the longest
// exactly 1200 mm.
function rod_coarse_extension_mm(length_mm) =
    min(rod_coarse_travel, max(0, length_mm - rod_min_pin_to_pin));

function rod_fine_extension_mm(length_mm) =
    min(rod_fine_travel,
        max(0, length_mm - rod_min_pin_to_pin - rod_coarse_travel));

function rod_slider_start_z(length_mm) =
    rod_slider_start_min + rod_coarse_extension_mm(length_mm);

function rod_slider_tip_z(length_mm) =
    rod_slider_start_z(length_mm) + rod_slider_len;

// The fine stage slides the FORK along the rod.  The fork's bore is the throttle pin's
// station, so the pin-to-pin span is the fork's bore station.
function rod_throttle_pin_z(length_mm) =
    rod_slider_tip_z(length_mm) + rod_fork_offset + rod_fine_extension_mm(length_mm);

// Axial stations used to test real capture-pin and closed-groove geometry.
function rod_capture_pin_z() = rod_fixed_chain_len + rod_capture_z;
function rod_capture_shoulder_end_z(length_mm) =
    rod_slider_start_z(length_mm) + rod_capture_groove_z0;

assert(rod_min_pin_to_pin == 940, "Illustrative shortest span must remain 940 mm");
assert(rod_coarse_travel + rod_fine_travel == 260,
       "Coarse plus fine travel must cover the 940-1200 mm requested range");
// The mapping must land EXACTLY on the requested span at both ends and at the
// nominal setting, otherwise the published pin-to-pin distance is not the one the
// geometry actually produces.
assert(abs(rod_throttle_pin_z(rods_span_min_mm) - rods_span_min_mm) < 1e-6,
       "The shortest setting must place the throttle pin exactly at 940 mm");
assert(abs(rod_throttle_pin_z(rods_span_max_mm) - rods_span_max_mm) < 1e-6,
       "The longest setting must place the throttle pin exactly at 1200 mm");
assert(abs(rod_throttle_pin_z(rods_nominal_span_mm) - rods_nominal_span_mm) < 1e-6,
       "The nominal setting must place the throttle pin exactly at its published span");
assert(rod_slider_len <= print_part_max_len_mm,
       "Slider must fit the declared print envelope");
assert(rod_receiver_len <= print_part_max_len_mm,
       "Receiver must fit the declared print envelope");
assert(max(rod_segment_lens) <= print_part_max_len_mm,
       "A fixed-chain section must fit the declared print envelope");
assert(rod_overlap >= 60, "The requested minimum overlap must remain at least 60 mm");
assert(rod_capture_groove_z1 - rod_capture_groove_z0 > rod_coarse_travel,
       "The capture groove must cover the whole coarse adjustment range");

// ---- Basic printed solids -------------------------------------------------------------
module rod_square_tube(len, w = rod_w, bore = rod_bore) {
    assert(bore > 0 && bore < w, "Tube bore must be positive and smaller than its outside");
    linear_extrude(height = len)
        difference() {
            square([w, w], center = true);
            square([bore, bore], center = true);
        }
}

module rod_square_ring(len, od, id) {
    assert(id > 0 && id < od, "Ring bore must be positive and smaller than its outside");
    linear_extrude(height = len)
        difference() {
            square([od, od], center = true);
            square([id, id], center = true);
        }
}

module rod_cross_bar(len = 34, w = keeper_t_mm, h = keeper_t_mm) {
    // A real transverse bar with a tapered outboard end.  It passes through aligned slots
    // in two printed parts and is trapped by a separate keeper.
    hull() {
        translate([-len / 2, 0, -h / 2]) cube([len - 5, w, h]);
        translate([len / 2 - 5, 0, -h / 2 + 0.8]) cube([5, w, h - 1.6]);
    }
}

module rod_bar_keeper() {
    // Split clip for a square bar. Capture and long-term retention require physical testing.
    linear_extrude(height = keeper_t_mm, center = true)
        difference() {
            circle(d = keeper_t_mm + 8);
            square([keeper_t_mm + 1, keeper_t_mm + 1], center = true);
            translate([0, -1]) square([keeper_t_mm + 8, 2]);
        }
}

// ---- Fixed-chain sections ---------------------------------------------------------------
module rod_receiver_boss() {
    // The enlarged socket accepts the outside of the previous section's ring spigot while
    // retaining a passage for the inner slider.  Its wall is real material.
    rod_square_ring(rod_receiver_boss_len, rod_receiver_boss_od,
                    rod_receiver_socket_od);
}

module rod_male_ring_spigot() {
    // Hollow keyed spigot.  Its central passage stays open for the slider, so the slider
    // and the fixed chain never occupy the same material.
    rod_square_ring(rod_spigot_len, rod_spigot_od, rod_spigot_id);
}

module rod_chain_section(len, receiver = true) {
    assert(len > rod_receiver_boss_len + 60,
           "A printed section must contain a receiver boss and a useful tube length");
    difference() {
        union() {
            if (receiver)
                rod_receiver_boss();
            // Bridge the socket to the smaller tube; their annuli otherwise do not touch.
            translate([0, 0, rod_receiver_boss_len - 2])
                rod_square_ring(4, rod_receiver_boss_od, rod_bore);
            translate([0, 0, rod_receiver_boss_len])
                rod_square_tube(len - rod_receiver_boss_len);
            // The ring overlaps the tube and extends 36 mm into the next boss.
            translate([0, 0, len - 4]) rod_male_ring_spigot();
        }
        // Transverse-bar slot in this section's bottom boss.
        if (receiver)
            translate([-keeper_t_mm / 2 - 0.6, rod_spigot_id / 2 - 0.2, 15])
                cube([keeper_t_mm + 1.2, rod_receiver_boss_od, keeper_t_mm]);
        // Transverse-bar slot in this section's male ring.
        translate([-keeper_t_mm / 2 - 0.6, rod_spigot_id / 2 - 0.2, len + 15])
            cube([keeper_t_mm + 1.2, rod_spigot_od, keeper_t_mm]);
    }
}

module rod_servo_section() {
    // The first printed section carries the servo eye integrally. Print frame: the eye
    // plate starts at z = 0 with its bore at z = 6, rod_eye_crank_mm below the tube axis;
    // the tube starts rod_eye_reach_mm beyond the bore and is fused to the plate along
    // its -X wall. Printed upright, the tube's first face over the plate needs support.
    tube_z0 = 6 + rod_eye_reach_mm;
    difference() {
        union() {
            translate([-rod_eye_crank_mm - rod_eye_plate_t / 2, -rod_eye_plate_w / 2, 0])
                cube([rod_eye_plate_t, rod_eye_plate_w, tube_z0 + rod_eye_fuse_len]);
            translate([0, 0, tube_z0])
                rod_square_tube(rod_segment_lens[0] - tube_z0);
            translate([0, 0, rod_segment_lens[0] - 4]) rod_male_ring_spigot();
        }
        // Nominal 5.8 mm servo pin bore along X, centred at printable z = 6.
        translate([-rod_eye_crank_mm - rod_eye_plate_t / 2 - 1, 0, 6]) rotate([0, 90, 0])
            cylinder(d = rod_eye_bore_d + 2 * joint_clearance_mm,
                     h = rod_eye_plate_t + 2);
        // Slot in this section's male ring for the next section's bar.
        translate([-keeper_t_mm / 2 - 0.6, rod_spigot_id / 2 - 0.2,
                   rod_segment_lens[0] + 15])
            cube([keeper_t_mm + 1.2, rod_spigot_od, keeper_t_mm]);
    }
}

// The servo section in rod-local coordinates: the eye's bore centre is the origin.
module rod_servo_eye() {
    translate([rod_eye_crank_mm, 0, -6]) rod_servo_section();
}

// ---- Receiver sleeve ------------------------------------------------------------------------
module rod_receiver_sleeve() {
    assert(rod_receiver_len > rod_receiver_boss_len + 20,
           "Receiver sleeve must have a socket and a usable slider passage");
    difference() {
        union() {
            rod_receiver_boss();
            // Bridge the socket's larger bore to the smaller receiver tube.
            translate([0, 0, rod_receiver_boss_len - 2])
                rod_square_ring(4, rod_receiver_boss_od, rod_bore);
            translate([0, 0, rod_receiver_boss_len])
                rod_square_ring(rod_receiver_len - rod_receiver_boss_len,
                                rod_w, rod_bore);
        }
        // Transverse-bar slot through the receiver boss.
        translate([-keeper_t_mm / 2 - 0.6, rod_spigot_id / 2 - 0.2, 15])
            cube([keeper_t_mm + 1.2, rod_receiver_boss_od, keeper_t_mm]);
        // Side service window for the three interchangeable coarse clamp bodies.  It
        // reaches through the +X wall so the clamp can press the slider's +X flat.
        translate([rod_slider / 2, -16, rod_lock_z0])
            cube([12, 32, rod_lock_z1 - rod_lock_z0]);
        // Printed capture-pin bore through the -X wall.
        translate([-rod_w / 2 - 4, 0, rod_capture_z]) rotate([0, 90, 0])
            cylinder(d = rod_capture_pin_d + 0.6, h = 12);
    }
}

// The capture pin's keeper geometry.  The keeper is a SEPARATE printed ring that is
// clipped onto the pin's turned-down neck, so the pin is captive without friction.
// The neck's outside diameter is below the keeper's bore, and the keeper's bore is
// below the shaft's diameter, so the ring sits on the neck and cannot pass the shaft.
rod_capture_neck_d = 4.4;
rod_keeper_bore_d = 4.7;
rod_keeper_od = 11;

module rod_capture_pin() {
    // One-head printed pin with a keeper BAND behind the head.  The head cannot pass
    // the sleeve wall; the keeper ring fills the gap between the head and the sleeve's
    // outer face, so the pin cannot back out of the slider's groove.
    //
    // An annular cut turns down the shaft's outside while leaving a solid core.
    difference() {
        union() {
            translate([-19, 0, 0]) rotate([0, 90, 0])
                cylinder(d = rod_capture_pin_d, h = 13);
            translate([-23, 0, 0]) rotate([0, 90, 0])
                cylinder(d = pin_head_d_mm, h = 4.2);
        }
        // Keeper band: the shaft's outside is turned down over this band.
        translate([-19, 0, 0]) rotate([0, 90, 0])
            pin_band_relief(d = rod_capture_pin_d, band_d = rod_capture_neck_d,
                            band_t = 4);
    }
}

module rod_capture_keeper() {
    // Split keeper ring, printed separately.  It is clipped onto the pin's neck and
    // bears on the sleeve's outer face, so the pin cannot travel inward and leave the
    // slider groove.  Its bore is smaller than the shaft, so it cannot pass over the
    // shaft and cannot be pushed off the pin.
    translate([-17, 0, 0]) rotate([0, 90, 0])
        difference() {
            cylinder(d = rod_keeper_od, h = 4, center = true);
            translate([0, 0, -3]) cylinder(d = rod_keeper_bore_d, h = 6);
            translate([0, -1.0, -3]) cube([rod_keeper_od, 2.0, 6]);
        }
}

// ---- Slider ---------------------------------------------------------------------------------
module rod_slider_shoulder() {
    // The closed lower end of the longitudinal capture groove.  This is the moving
    // material that a forced overtravel would drive into the fixed capture pin.
    translate([-rod_slider / 2, -rod_slider / 2, 0])
        cube([rod_slider, rod_slider, rod_capture_groove_z0]);
}

module rod_slider(len = rod_slider_len) {
    difference() {
        union() {
            translate([-rod_slider / 2, -rod_slider / 2, 0])
                cube([rod_slider, rod_slider, len]);
            // Integral fine-screw flange near the tip.  It sits beyond the sleeve end in
            // every setting, so it never fouls the sleeve.  It reaches outboard in +X to
            // carry the screw's journal boss.
            translate([rod_fine_flange_x0, -12, len + rod_fine_flange_z0])
                cube([rod_fine_flange_x1 - rod_fine_flange_x0, 24,
                      rod_fine_flange_z1 - rod_fine_flange_z0]);
            // Journal boss for the fine screw.  Its axis is rod-local Z, the same line
            // the fork's nut and the screw itself are on.  The boss reaches BACK from
            // the slider's tip so the screw's plain end journals in it while the nut
            // rides in the fork's lug ahead.
            translate([rod_fine_screw_x, 0, len + rod_fine_boss_z0])
                cylinder(d = rod_fine_boss_od,
                         h = rod_fine_boss_z1 - rod_fine_boss_z0);
        }
        // Continuous longitudinal capture groove with closed ends, on the -X face.
        translate([-rod_slider / 2 - 0.5, -rod_capture_groove_w / 2,
                   rod_capture_groove_z0])
            cube([rod_capture_groove_depth + 0.5, rod_capture_groove_w,
                  rod_capture_groove_z1 - rod_capture_groove_z0]);
        // The fine screw's journal bore, along rod-local Z, through the boss.
        translate([rod_fine_screw_x, 0, len + rod_fine_boss_z0 - 1])
            cylinder(d = rod_fine_screw_bore_d,
                     h = rod_fine_boss_z1 - rod_fine_boss_z0 + 2);
    }
}

// ---- Interchangeable coarse clamp bodies --------------------------------------------------------
module rod_lock_saddle() {
    // Printed saddle that hooks around the sleeve's +X side and reaches through the
    // sleeve's service window to the slider.  It keys into the sleeve's window edges, so
    // it cannot lift straight off; the clamp screw/wedge/collar supplies the force.
    difference() {
        union() {
            translate([-rod_w / 2 - 5, -rod_w / 2 - 5, 0])
                cube([rod_w + 10, 5, 55]);
            translate([-rod_w / 2 - 5, rod_w / 2, 0])
                cube([rod_w + 10, 5, 55]);
            translate([rod_w / 2, -rod_w / 2 - 5, 0])
                cube([5, rod_w + 10, 55]);
            // Shoe that reaches through the window to the slider's +X flat.
            translate([rod_slider / 2, -12, 8])
                cube([rod_w / 2 - rod_slider / 2 + 0.8, 24, 39]);
        }
        // Clearance around the sleeve OD.
        translate([-rod_w / 2 - 0.5, -rod_w / 2 - 0.5, -1])
            cube([rod_w + 1, rod_w + 1, 57]);
        // Clamp force channel through the outer bridge.
        translate([rod_w / 2 - 5, -6, 22]) rotate([0, 90, 0])
            cylinder(d = fine_screw_d_mm + 0.8, h = 12);
    }
}

module rod_lock_tongues() {
    // Keyed tongues locate the clamp in the sleeve's service window.
    for (s = [-1, 1])
        translate([rod_slider / 2 - 0.4, s * 14, 6])
            cube([rod_w / 2 - rod_slider / 2 + 1.4, 4, 43]);
}

module rod_lock_wedge_body() {
    difference() {
        union() {
            rod_lock_saddle();
            rod_lock_tongues();
        }
        // Tapered wedge channel, open toward the slider flat.
        translate([rod_slider / 2 - 1, -5, 24]) cube([20, 10, 8]);
        // Over-centre cam pivot bore, axis Y.
        translate([rod_w / 2 + 2.5, 14, 34]) rotate([90, 0, 0])
            cylinder(d = rod_capture_pin_d + 0.6, h = 28);
    }
}

module rod_lock_wedge() {
    linear_extrude(height = keeper_t_mm)
        polygon([[rod_slider / 2 + 0.4, -1],
                 [rod_w / 2 + 2, -3], [rod_w / 2 + 2, 3], [rod_slider / 2 + 0.4, 1]]);
}

module rod_lock_overcentre_cam() {
    difference() {
        union() {
            rotate([90, 0, 0]) cylinder(d = 14, h = 6, center = true);
            linear_extrude(height = 4, center = true)
                polygon([[0, -3], [28, -1], [28, 6], [0, 5]]);
        }
        rotate([90, 0, 0])
            cylinder(d = rod_capture_pin_d + 0.6, h = 8, center = true);
    }
}

module rod_lock_cam_keeper() { printed_keeper(); }

// Split collet: integral slotted shoes press the slider's +X flat; a large printed
// compression collar retains the saddle. Flexure and retention are unverified.
module rod_lock_collet_body() {
    difference() {
        union() {
            rod_lock_saddle();
            rod_lock_tongues();
            // Two split shoes that actually reach the slider flat.
            for (s = [-1, 1])
                translate([rod_slider / 2 + 0.2, s * 8 - 3, 8])
                    cube([rod_w / 2 + 2 - rod_slider / 2 - 0.2, 6, 39]);
        }
        // Relief slots leave the shoe roots attached to the outer bridge.
        for (s = [-1, 1])
            translate([rod_slider / 2, s * 8 - 0.6, 8])
                cube([6, 1.2, 39]);
    }
}

module rod_lock_collet_collar() {
    // Printed compression collar around the saddle's +X bridge.  Its skirt is longer than
    // the bridge it clamps, so it cannot slide off.
    difference() {
        linear_extrude(height = 20)
            difference() {
                square([42, 42], center = true);
                square([37.6, 37.6], center = true);
                // Open side for the split shoes.
                translate([0, -22]) square([44, 44]);
            }
        // Coarse printed compression ribs; not a standard thread.
        for (s = [-1, 1])
            translate([rod_w / 2 + 6, s * 13, 4]) rotate([90, 0, 0])
                cylinder(d = 3.6, h = 12);
    }
}

module rod_lock_collet_keeper() { printed_keeper(); }

// Printed lead-screw clamp.  The screw advances the shoe against the slider; the lip
// prevents free unwind and the jam collar is separately printed.
module rod_lock_lead_body() {
    difference() {
        union() {
            rod_lock_saddle();
            rod_lock_tongues();
        }
        // Screw bore through the outer bridge, along Z.
        translate([rod_w / 2 + 2.5, 0, -1]) rotate([-90, 0, 0])
            cylinder(d = fine_screw_d_mm + 0.8, h = 57);
    }
}

module rod_lock_lead_screw() { printed_screw(54); }
module rod_lock_lead_nut() { printed_nut(); }
module rod_lock_lead_lip() { anti_backdrive_lip(); }

module rod_lock_lead_jam() {
    linear_extrude(height = fine_jam_collar_h_mm)
        difference() {
            circle(d = fine_screw_d_mm + 12);
            circle(d = fine_screw_d_mm - 1.5);
        }
}

// ---- Throttle-side eye fork and fine trim -----------------------------------------------------
module rod_throttle_fork() {
    // Frame: x relative to the tube axis, z relative to the throttle eye's bore. The eye
    // plate lies in the eye plane (x = -crank) and drops over the U fitting's pivot post.
    // It reaches back under the slider's tip; a side web beside the slider rises to the
    // arm, which carries the captive fine nut's lug above the rod.
    eye_x0 = -rod_eye_crank_mm - rod_eye_plate_t / 2;
    difference() {
        union() {
            translate([eye_x0, -rod_eye_plate_w / 2, -rod_fork_eye_back])
                cube([rod_eye_plate_t, rod_eye_plate_w / 2 + rod_fork_web_y1,
                      rod_fork_eye_back + rod_fork_eye_front]);
            translate([eye_x0 + 1, rod_fork_web_y0, -rod_fork_eye_back])
                cube([rod_fork_arm_x0 + 1 - eye_x0 - 1, rod_fork_web_y1 - rod_fork_web_y0,
                      rod_fork_eye_back + rod_fork_arm_z1]);
            translate([rod_fork_arm_x0, rod_fork_arm_y0, rod_fork_arm_z0])
                cube([rod_fork_arm_x1 - rod_fork_arm_x0,
                      rod_fork_arm_y1 - rod_fork_arm_y0,
                      rod_fork_arm_z1 - rod_fork_arm_z0]);
            translate([rod_fine_screw_x - 18, -rod_fork_lug_y, rod_fork_nut_z0])
                cube([36, 2 * rod_fork_lug_y,
                      rod_fork_lug_z1 - rod_fork_nut_z0]);
        }
        // The throttle eye's bore along rod-local X (vertical in the world).
        translate([eye_x0 - 1, 0, 0]) rotate([0, 90, 0])
            cylinder(d = rod_eye_bore_d + 2 * joint_clearance_mm, h = rod_eye_plate_t + 2);
        // The fine nut's pocket in the lug, along rod-local Z: the nut is pressed into
        // it and cannot fall out.
        translate([rod_fine_screw_x, 0, rod_fork_nut_z0])
            cylinder(d = rod_fine_nut_bore_d, h = rod_fork_nut_z1 - rod_fork_nut_z0);
        // The fine screw's clearance bore through the lug, along the same line.
        translate([rod_fine_screw_x, 0, rod_fork_nut_z0])
            cylinder(d = rod_fine_screw_bore_d,
                     h = rod_fork_lug_z1 - rod_fork_nut_z0);
    }
}

module rod_fine_screw() {
    // Printed fine screw.  Its axis is rod-local Z, parallel to the rod and clear of the
    // slider, so the fine trim is a pure axial slide of the fork.  Its printed thread is
    // a trapezoid approximation; fractional turns give nominally continuous motion.
    // The screw is placed by its TIP at the fork's bore centre and runs back along -Z,
    // so it reaches the slider's journal boss.
    translate([rod_fine_screw_x, 0, rod_fine_screw_z1])
        rotate([180, 0, 0]) printed_screw(rod_fine_screw_len);
}

module rod_fine_carriage_nut() {
    // Printed nut that rides the fine screw inside the fork's lug.  Its axis is
    // rod-local Z, the same line as the screw it rides on and the slider's journal
    // boss, so all three share one axis.
    translate([rod_fine_screw_x, 0, rod_fork_nut_z0])
        printed_nut(h = rod_fine_nut_h);
}

module rod_fine_jam_collar() {
    translate([rod_fine_screw_x, 0, 0])
        linear_extrude(height = fine_jam_collar_h_mm)
            difference() {
                circle(d = fine_screw_d_mm + 12);
                circle(d = fine_screw_d_mm - 1.5);
            }
}

// The moving end at a setting, about the tube axis: slider, eye fork, fine nut and screw.
module rod_throttle_end(length_mm = rod_len) {
    translate([0, 0, rod_slider_start_z(length_mm)]) rod_slider();
    translate([0, 0, rod_throttle_pin_z(length_mm)]) {
        rod_throttle_fork();
        rod_fine_carriage_nut();
        rod_fine_screw();
    }
}

// ---- Named anti-separation / overtravel geometry ------------------------------------------------
// These are actual parts of the model, not numerical claims.  The moving lug is the
// slider's closed lower groove shoulder; the fixed stop is the receiver's capture pin.
// Translating the lug in +Z (the separation direction) drives it into the pin: installed
// clearance is empty, forced overtravel is blocked.
module rod_overstop_lug(length_mm = rod_len) {
    translate([0, 0, rod_slider_start_z(length_mm)]) rod_slider_shoulder();
}

module rod_overstop_stop() {
    translate([0, 0, rod_fixed_chain_len + rod_capture_z]) rod_capture_pin();
}

// ---- Three distinct, interchangeable coarse clamps ------------------------------------------------
module lock_wedge_cam(part = "assembly") {
    if (part == "assembly") {
        rod_lock_wedge_body();
        translate([0, 0, 34]) rod_lock_wedge();
        translate([rod_w / 2 + 2.5, 0, 34]) rod_lock_overcentre_cam();
        translate([rod_w / 2 + 2.5, 6, 34]) rod_lock_cam_keeper();
    } else if (part == "body") {
        rod_lock_wedge_body();
    } else if (part == "wedge") {
        rod_lock_wedge();
    } else if (part == "cam") {
        rod_lock_overcentre_cam();
    } else if (part == "keeper") {
        rod_lock_cam_keeper();
    } else {
        assert(false, str("unknown wedge part: ", part));
    }
}

module lock_split_collet(part = "assembly") {
    if (part == "assembly") {
        rod_lock_collet_body();
        translate([0, 0, 14]) rod_lock_collet_collar();
        translate([rod_w / 2 + 8, 0, 24]) rod_lock_collet_keeper();
    } else if (part == "body") {
        rod_lock_collet_body();
    } else if (part == "collar") {
        rod_lock_collet_collar();
    } else if (part == "keeper") {
        rod_lock_collet_keeper();
    } else {
        assert(false, str("unknown collet part: ", part));
    }
}

module lock_lead_screw(part = "assembly") {
    if (part == "assembly") {
        rod_lock_lead_body();
        translate([rod_w / 2 + 2.5, 0, 32]) rod_lock_lead_screw();
        translate([rod_w / 2 + 2.5, 0, 24]) rod_lock_lead_nut();
        translate([rod_w / 2 + 2.5, 6, 24]) rod_lock_lead_lip();
        translate([rod_w / 2 + 2.5, 0, 48]) rod_lock_lead_jam();
    } else if (part == "body") {
        rod_lock_lead_body();
    } else if (part == "screw") {
        rod_lock_lead_screw();
    } else if (part == "nut") {
        rod_lock_lead_nut();
    } else if (part == "anti_backdrive") {
        rod_lock_lead_lip();
    } else if (part == "jam") {
        rod_lock_lead_jam();
    } else {
        assert(false, str("unknown lead-screw part: ", part));
    }
}

module throttle_lock(kind = "wedge_cam", part = "assembly") {
    if (kind == "wedge_cam") lock_wedge_cam(part);
    else if (kind == "split_collet") lock_split_collet(part);
    else if (kind == "lead_screw") lock_lead_screw(part);
    else assert(false, str("unknown lock kind: ", kind));
}

// ---- Public assembly and printable selectors --------------------------------------------------------
rod_part = "assembly";
rod_len = 1066.8;   // illustrative default within the requested 940-1200 mm range
rod_lock_kind = "lead_screw";

module throttle_pushrod(part = rod_part, length_mm = rod_len,
                        lock_kind = rod_lock_kind) {
    assert(length_mm >= rods_span_min_mm && length_mm <= rods_span_max_mm,
           "Illustrative rod setting must lie within the requested 940-1200 mm range");

    if (part == "assembly") {
        // Every member is drawn about the tube axis, which is cranked rod_eye_crank_mm
        // above the eye plane; the servo eye's bore then lands on the rod-local origin.
        translate([rod_eye_crank_mm, 0, 0]) {
            translate([0, 0, rod_segment_offsets[0] - 6]) rod_servo_section();
            translate([0, 0, rod_segment_offsets[1]])
                rod_chain_section(rod_segment_lens[1]);
            translate([0, 0, rod_segment_offsets[2]])
                rod_chain_section(rod_segment_lens[2]);
            translate([0, 0, rod_fixed_chain_len]) rod_receiver_sleeve();
            // Positive capture pin and its separate keeper; the closed groove keeps the
            // sleeve pin captured over the whole coarse range.
            translate([0, 0, rod_fixed_chain_len + rod_capture_z]) rod_capture_pin();
            translate([0, 0, rod_fixed_chain_len + rod_capture_z]) rod_capture_keeper();
            // Selected coarse clamp, in the sleeve's service window.
            translate([0, 0, rod_fixed_chain_len + rod_lock_z0])
                throttle_lock(kind = lock_kind, part = "assembly");
            rod_throttle_end(length_mm);
        }
    } else if (part == "throttle_end") {
        // The moving end only (slider, eye fork, fine screw and nut), in rod-local
        // coordinates; used by the linkage probes.
        translate([rod_eye_crank_mm, 0, 0]) rod_throttle_end(length_mm);
    } else if (part == "segment_lower") {
        rod_servo_section();
    } else if (part == "segment_middle") {
        rod_chain_section(rod_segment_lens[1]);
    } else if (part == "segment_upper") {
        rod_chain_section(rod_segment_lens[2]);
    } else if (part == "slider") {
        rod_slider();
    } else if (part == "sleeve") {
        rod_receiver_sleeve();
    } else if (part == "capture_pin") {
        translate([0, 0, -6]) rotate([0, 90, 0]) rod_capture_pin();
    } else if (part == "capture_keeper") {
        translate([0, 0, 19]) rotate([0, -90, 0]) rod_capture_keeper();
    } else if (part == "spigot") {
        rod_male_ring_spigot();
    } else if (part == "splice_wedge") {
        translate([0, 0, keeper_t_mm / 2]) rod_cross_bar();
    } else if (part == "splice_keeper") {
        translate([0, 0, keeper_t_mm / 2]) rod_bar_keeper();
    } else if (part == "servo_eye") {
        rod_servo_eye();
    } else if (part == "throttle_fork") {
        translate([0, 0, -rod_fork_nut_z0]) rod_throttle_fork();
    } else if (part == "fine_screw") {
        printed_screw(rod_fine_screw_len);
    } else if (part == "fine_carriage_nut") {
        printed_nut(h = rod_fine_nut_h);
    } else if (part == "fine_jam_collar") {
        rod_fine_jam_collar();
    } else if (part == "endstop") {
        rod_slider_shoulder();
    } else if (part == "overstop_lug") {
        rod_slider_shoulder();
    } else if (part == "overstop_stop") {
        translate([0, 0, -6]) rotate([0, 90, 0]) rod_capture_pin();
    } else if (part == "lock_wedge_body") {
        lock_wedge_cam("body");
    } else if (part == "lock_wedge") {
        lock_wedge_cam("wedge");
    } else if (part == "lock_wedge_cam") {
        translate([0, 0, 7]) lock_wedge_cam("cam");
    } else if (part == "lock_wedge_keeper") {
        lock_wedge_cam("keeper");
    } else if (part == "lock_collet_body") {
        lock_split_collet("body");
    } else if (part == "lock_collet_collar") {
        lock_split_collet("collar");
    } else if (part == "lock_collet_keeper") {
        lock_split_collet("keeper");
    } else if (part == "lock_screw") {
        lock_lead_screw("screw");
    } else if (part == "lock_nut") {
        lock_lead_screw("nut");
    } else if (part == "lock_anti_backdrive") {
        lock_lead_screw("anti_backdrive");
    } else if (part == "lock_jam_collar") {
        lock_lead_screw("jam");
    } else if (part == "clevis") {
        printed_clevis();
    } else if (part == "clevis_bushing") {
        printed_bushing();
    } else if (part == "clevis_pin") {
        printed_pin(2 * clevis_ear_t_mm + clevis_gap_mm + 2 * keeper_t_mm);
    } else if (part == "gauge") {
        // Sliding-fit coupon: a plate with a closed square passage.
        linear_extrude(height = 14)
            difference() {
                translate([-20, -12]) square([40, 24]);
                translate([-rod_slider / 2, -rod_slider / 2])
                    square([rod_slider, rod_slider]);
            }
    } else {
        assert(false, str("unknown pushrod part: ", part));
    }
}

throttle_pushrod(rod_part);
