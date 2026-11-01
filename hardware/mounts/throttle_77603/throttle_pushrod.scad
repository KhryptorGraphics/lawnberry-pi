// Segmented, fully printed 77603 throttle pushrod: continuous coarse slide held by a
// print-in-place threaded split collet, plus a separate printed lead-screw fine trim.
//
// PROVISIONAL DESIGN.  The 940-1200 mm setting range is the requested illustrative
// adjustment range.  It is NOT a measured mower span, NOT a measured throttle stroke, and
// NOT a verified lever path.  Nothing in this file proves fit, load capacity, wear life,
// clamp force, or actual travel.  Those remain coupon, bench, and mower tests.
//
// Mechanical concept
// -------------------
// * A fixed chain of printed square-tube sections is joined by keyed ring spigots.  Each
//   joint is retained by a separately printed transverse bar and a separately printed
//   keeper, so a joint cannot open by friction alone.
// * The receiver sleeve and the 21.2 mm square slider print together with a captive
//   collet NUT as ONE print-in-place part, `rod_capture_assembly` (three bodies).  The
//   sleeve's integral 45-degree lug rides in the slider's closed-ended groove; the groove
//   ends are overtravel catches, not length stops.
// * LENGTH LOCK (no bolts, no tools): the sleeve's throttle end is a round nose -- an
//   unslotted band carrying a 45-degree-flank external thread, then four fingers slotted
//   at the square bore's corners, whose outside FLARES outward toward the tip (18-degree
//   cone).  The nut's inner cone opens toward the tip and sits over the flare with the
//   print-in-place gap.  Turning the nut by hand so it advances toward the tip drives its
//   cone onto the flare and closes each finger's flat inner face onto one slider flat.
//   The nut is captive both ways: the band's 45-degree shoulder stops it backing off, and
//   it cannot pass the flare while the slider is inside.  Scalloped grip, right-hand
//   thread: viewed from the servo end, CLOCKWISE tightens.
// * UNWINDING RESISTANCE is only thread self-locking (lead angle ~1.9 deg) plus collet
//   preload friction.  There is NO detent or ratchet; vibration resistance is unqualified.
// * REPEATABLE SETTING: V ticks engraved on the slider's +X flat (every 5 mm of coarse
//   setting, longer every 10 mm, longest every 50 mm) are read against the index notch in
//   the +X finger's tip face.  The fine stage adds up to 20 mm (2 mm per screw turn).
// * FINE TRIM: a separately printed throttle-end BODY sockets over the slider's plain tip
//   and is retained by the same transverse bar + keeper used at the splices.  Its boss
//   journals the printed fine screw (flange behind the boss, jam collar in front), and the
//   throttle-eye FORK rides the screw on its captive printed nut.  Printed screw accuracy
//   and fine-stage holding are NOT claimed.
//
// Axes and mating interfaces
// --------------------------
// * Rod-local +Z is the rod's longitudinal direction, from the servo pin toward the
//   throttle pin.  The servo pin centre is rod-local (0,0,0).
// * Both ends carry a horizontal EYE PLATE, 4 mm thick in rod-local X, 12 mm wide in Y,
//   with a 6.6 mm cut bore along X. The servo eye's bore centre is the origin; the
//   throttle eye (on the fork) has its bore at z = rod_throttle_pin_z(length). Both eyes
//   lie in the x = 0 plane.
// * The tube axis is CRANKED rod_eye_crank_mm above that plane (+X), so the 30 mm tube
//   clears the servo, keeper and horn hardware below the eye plane.
// * The fine stage, its boss and the fork's arm are on the tube's +X side (above the rod);
//   the slider's ticks and the collet's index notch are on +X too.
//
// The scene places the rod with rotate([0, -90, 0]) and a yaw, which maps rod-local
// X -> world +Z and rod-local Z -> the run direction. Both eye bores are vertical.
//
// Installation assumptions
// -----------------------
// * Print `rod_capture_assembly` COLLET END DOWN: finger tips, nut front face and slider
//   tip all start on the bed.  No capture pin, keeper or post-print assembly.
// * Fit the throttle-end body over the slider tip, push the bar through the body and the
//   slider's window, clip the keeper.  Then thread the fine screw through the boss, fit
//   the jam collar, run the fork's nut onto the screw.
// * Set length with the nut loose (as printed), read the tick at the index notch,
//   tighten the nut by hand, then trim with the fine screw.
include <parameters.scad>
use <printed_joints.scad>

// ---- Section -----------------------------------------------------------------------
rod_w = rods_w_mm;                          // 30 mm outer square
rod_bore = rods_bore_mm;                    // 22 mm bore
rod_slider = rods_slider_mm;                // 21.2 mm keyed slider
rod_overlap = rods_min_overlap_mm;          // requested 60 mm minimum
rod_coarse_travel = rods_coarse_travel_mm;  // requested 240 mm
rod_fine_travel = rods_fine_travel_mm;      // requested 20 mm

// ---- Local print plan ---------------------------------------------------------------
// These are printable-section lengths and a kinematic layout.  They are not mower
// measurements.
rod_fixed_chain_len = 664;   // servo section + two chain sections, to the sleeve boss
rod_receiver_len = 193;      // receiver sleeve, boss face to collet tip
rod_sleeve_end_z = rod_fixed_chain_len + rod_receiver_len;   // 857: the collet tip
// Printed section heights, eye section first. Each section's receiver boss butts against
// the end of the previous section's tube, with the previous section's male spigot in the
// boss socket; the transverse-bar slots then line up 15 mm above the butt face.
rod_segment_lens = [230, 220, 220];
// Placement of each section along the rod. The servo section is drawn 6 mm low so its eye
// bore lands on the rod-local origin, and every later section starts where the previous
// one ends.
rod_segment_offsets = [0, rod_segment_lens[0] - 6,
                       rod_segment_lens[0] + rod_segment_lens[1] - 6];
rod_slider_len = 380;
rod_min_pin_to_pin = 940;    // illustrative shortest servo-pin to throttle-pin distance
// The fork's bore sits 8 mm ahead of the slider's tip at zero fine extension.
rod_fork_offset = 8;
rod_slider_start_min = rod_min_pin_to_pin - rod_slider_len - rod_fork_offset;   // 552
rod_receiver_boss_len = 40;
rod_spigot_len = 40;
rod_spigot_od = 29.6;
rod_spigot_id = rod_bore;
rod_receiver_socket_od = 30.8;
rod_receiver_boss_od = 36;
rod_receiver_socket_depth = 38;

rod_capture_groove_w = 6;
rod_capture_groove_depth = 4.6;
rod_capture_groove_z0 = 15;
rod_capture_groove_z1 = 350;
rod_capture_z = 148;         // sleeve-local station of the integral lug (in the band)

rod_eye_plate_t = 4;           // vertical thickness of both eye plates
rod_eye_plate_w = 12;          // width across the run
rod_eye_bore_d = 5.8;
// The tube axis sits this far above the eye plane. The eye plate's top face is then
// 1 mm inside the tube's -X wall, so the two fuse along rod_eye_fuse_len.
rod_eye_crank_mm = rod_w / 2 + rod_eye_plate_t / 2 - 1;      // 16
rod_eye_reach_mm = 18;         // servo bore to the tube's start, clear of horn hardware
rod_eye_fuse_len = 30;

// ---- Print-in-place collet ----------------------------------------------------------
// Built in the NOSE frame: z = 0 at the finger tips, +z toward the servo, i.e. the print
// orientation.  rod_nose_place() maps it onto the sleeve's rod-oriented frame.
rod_collet_alpha = collet_cone_half_angle_deg;
rod_collet_g = collet_gap_mm;
rod_collet_gr = collet_gap_mm / cos(collet_cone_half_angle_deg);  // radial cone gap
rod_collet_p = collet_thread_pitch_mm;
rod_collet_h = collet_thread_depth_mm;
rod_collet_root = collet_thread_root_r_mm;
rod_collet_crest = rod_collet_root + rod_collet_h;
rod_collet_flare_len = collet_flare_len_mm;
rod_collet_finger_len = collet_finger_len_mm;
rod_collet_finger_r = collet_finger_r_mm;
rod_collet_tip_r = rod_collet_finger_r + rod_collet_flare_len * tan(rod_collet_alpha);
rod_collet_band_w0 = rod_collet_finger_len + (rod_collet_root - rod_collet_finger_r);
rod_collet_nut_h = 51;          // nut length, front face to rear face
rod_collet_shoulder_w0 = rod_collet_nut_h + 1;   // 45-degree shoulder starts here
rod_collet_collar_r = 21.4;     // round nose top: holds the square tube's corners (21.21)
rod_collet_nose_top = rod_collet_shoulder_w0 + (rod_collet_collar_r - rod_collet_root) + 1.2;
rod_collet_nut_r = 26.5;        // nut outside; 16 grip scallops cut it to 25.5
rod_collet_scallop_r = 3;
rod_collet_scallops = 16;
// Thread: 45-degree flanks, equal 0.65 mm root and crest flats.
rod_collet_flat = rod_collet_p / 2 - rod_collet_h;
rod_collet_b = (rod_collet_p - rod_collet_flat) / 2;      // ridge half-width at the root
rod_collet_c = rod_collet_b - rod_collet_h;              // ridge half-width at the crest
rod_collet_helix_w0 = rod_collet_band_w0 + 0.2 + rod_collet_b + 0.1;  // first ridge centre
rod_collet_ridge_a1 = (rod_collet_shoulder_w0 - rod_collet_helix_w0) / rod_collet_p * 360;
rod_collet_groove_a0 = -180;    // the nut's groove starts half a turn before the ridge
rod_collet_groove_a1 = (rod_collet_nut_h + 3 - rod_collet_helix_w0) / rod_collet_p * 360;
rod_collet_fn = 120;

// External ridge profile [r, z about the helix centre], extended 0.1 mm into the core
// along its flanks.
function rod_collet_ridge_profile() =
    let(r0 = rod_collet_root - 0.1, b = rod_collet_b + 0.1, c = rod_collet_c)
    [[r0, -b], [rod_collet_crest, -c], [rod_collet_crest, c], [r0, b]];

// The nut's groove: the ridge profile offset by the gap perpendicular to every face
// (mitred), cut off just inside the nut's bore so successive turns never overlap.
function rod_collet_groove_profile() =
    let(g = rod_collet_g, ri = rod_collet_root + g - 0.08,
        rc = rod_collet_crest + g, zc = rod_collet_c + g * (sqrt(2) - 1),
        zi = rod_collet_c + (rod_collet_crest - ri) + g * sqrt(2))
    [[ri, -zi], [rc, -zc], [rc, zc], [ri, zi]];

// Nut advance toward the tip at which its cone first meets the undeflected flare, and the
// advance that closes the fingers' slider clearance and adds the declared preload.
function rod_collet_contact_advance() = rod_collet_g / sin(rod_collet_alpha);
function rod_collet_clamp_advance() =
    rod_collet_contact_advance()
    + (sliding_clearance_mm + collet_preload_mm) / tan(rod_collet_alpha);
// Radial finger closure available within the reserved nut advance.
function rod_collet_closure_available() =
    collet_nut_max_advance_mm * tan(rod_collet_alpha) - rod_collet_gr;

// ---- Throttle-end body and fine stage ---------------------------------------------------
// All stations are relative to the slider's tip, t.  The body sockets over the slider
// behind the fork, so the slider's last rod_body_front_gap mm stay bare for the fork.
rod_body_back = 58;             // the body's rear face, behind the slider tip
rod_body_front = 22;            // the body's front face, behind the slider tip
rod_body_bar_z = -40;           // the transverse bar's station
rod_body_bar_y0 = -9;           // the bar's inner end, inside the slider's window
rod_fine_screw_x = 32;          // the fine axis, outboard of the slider in +X
rod_fine_screw_len = 85;        // shaft + its rear flange
rod_fine_web_x0 = rod_w / 2 - 1;   // the boss's web fuses into the socket's +X wall
rod_fine_web_x1 = 58;
rod_fine_boss_od = 30;          // the journal boss's outside
rod_fine_boss_z0 = -rod_body_back;
rod_fine_boss_z1 = -30;
rod_fine_screw_bore_d = 16.8;   // the screw's clearance through the boss and the fork lug
rod_fine_nut_bore_d = 26.1;     // the printed nut's outside, with a press fit
rod_fine_nut_h = 10;            // the printed nut's height (fine_nut_height_mm)
// The screw is axially fixed to the body: its flange bears on the boss's rear face and
// the jam collar on its front face (each drawn 0.2 mm off).  Its front end, relative to
// the slider tip:
rod_fine_screw_z1 = rod_fine_boss_z0 - 0.2 - keeper_t_mm + rod_fine_screw_len;  // +22.8

// ---- Throttle eye on the fine-stage fork ---------------------------------------------
// The fork's origin is the throttle eye's bore, in the eye plane (x = -crank relative
// to the tube axis). The eye plate reaches back under the slider tip, a side web beside
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
rod_fork_nut_z0 = -28;          // the nut's pocket, open at the lug's rear face
rod_fork_nut_z1 = -28 + rod_fine_nut_h;
rod_fork_lug_y = 18;            // half width of the nut lug: it carries the 26 mm nut
rod_fork_lug_z1 = 10;           // the lug's front face

// ---- Continuous adjustment mapping ----------------------------------------------------
// Coarse travel is used first; only the final 20 mm is fine trim.  Both are continuous.
// This mapping is deterministic for scene generation, not installer advice.
//
//   servo pin ......... z = 0
//   slider tip ........ z = rod_slider_tip_z
//   fork bore ......... z = rod_throttle_pin_z   (the throttle pin's station)
function rod_coarse_extension_mm(length_mm) =
    min(rod_coarse_travel, max(0, length_mm - rod_min_pin_to_pin));

function rod_fine_extension_mm(length_mm) =
    min(rod_fine_travel,
        max(0, length_mm - rod_min_pin_to_pin - rod_coarse_travel));

function rod_slider_start_z(length_mm) =
    rod_slider_start_min + rod_coarse_extension_mm(length_mm);

function rod_slider_tip_z(length_mm) =
    rod_slider_start_z(length_mm) + rod_slider_len;

function rod_throttle_pin_z(length_mm) =
    rod_slider_tip_z(length_mm) + rod_fork_offset + rod_fine_extension_mm(length_mm);

// Axial stations used to test the integral sleeve lug and closed slider groove.
function rod_capture_lug_z() = rod_fixed_chain_len + rod_capture_z;
function rod_slider_start_in_sleeve_mm(length_mm) =
    rod_slider_start_z(length_mm) - rod_fixed_chain_len;
function rod_capture_shoulder_end_z(length_mm) =
    rod_slider_start_z(length_mm) + rod_capture_groove_z0;
function rod_capture_upper_shoulder_z(length_mm) =
    rod_slider_start_z(length_mm) + rod_capture_groove_z1;
// Slider-local station read at the collet tip for a coarse setting (the tick positions).
function rod_tick_station(length_mm) =
    rod_sleeve_end_z - rod_slider_start_z(length_mm);
// The body's rearmost point (the screw's flange), behind the slider tip.
function rod_body_rear_z(length_mm) =
    rod_slider_tip_z(length_mm) + rod_fine_boss_z0 - 0.2 - keeper_t_mm;

assert(rod_min_pin_to_pin == 940, "Illustrative shortest span must remain 940 mm");
assert(rod_coarse_travel + rod_fine_travel == 260,
       "Coarse plus fine travel must cover the 940-1200 mm requested range");
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
assert(rod_segment_offsets[2] + rod_segment_lens[2] == rod_fixed_chain_len,
       "The fixed chain must end exactly at the receiver sleeve");
assert(rod_overlap >= 60, "The requested minimum overlap must remain at least 60 mm");
assert(rod_sleeve_end_z - rod_slider_start_z(rods_span_max_mm) >= rod_overlap,
       "The slider must overlap the sleeve by the minimum at the longest setting");
// The lug (7 mm tall) must stay inside the groove from 940 to 1200 mm and as printed.
assert(rod_capture_lug_z() - rod_slider_start_z(rods_span_max_mm) > rod_capture_groove_z0
       && rod_capture_lug_z() + 7 - rod_slider_start_z(rods_span_min_mm)
          < rod_capture_groove_z1
       && rod_capture_lug_z() + 7 - (rod_sleeve_end_z - rod_slider_len)
          < rod_capture_groove_z1,
       "The capture lug must stay inside the slider groove at every setting");
// The lug must sit in the unslotted band, not on a finger.
assert(rod_receiver_len - rod_capture_z - 7 > rod_collet_band_w0,
       "The capture lug must be on the unslotted band, behind the fingers");
// The throttle-end body clears the fully advanced nut at the shortest setting.
assert(rod_body_rear_z(rods_span_min_mm)
       - (rod_sleeve_end_z + collet_nut_max_advance_mm) >= 3,
       "The throttle-end body must clear the tightened collet nut at 940 mm");
// Collet: thread wall over the bore corners, radial closure, thread engagement, captive.
assert(rod_collet_root - rod_bore / sqrt(2) >= 3,
       "The threaded band must keep 3 mm of wall over the square bore's corners");
assert(rod_collet_closure_available() > sliding_clearance_mm + collet_preload_mm,
       "The nut's reserved advance must close the slider clearance plus preload");
assert(rod_collet_clamp_advance() < collet_nut_max_advance_mm,
       "The clamp advance must lie inside the reserved nut travel");
assert(rod_collet_nut_h - collet_nut_max_advance_mm
       - (rod_collet_helix_w0 - rod_collet_b - 0.1) >= 3 * rod_collet_p,
       "At full advance the nut must still engage three thread pitches");
assert(rod_collet_tip_r > rod_collet_root + rod_collet_g + 1,
       "The flare must be wider than the nut's thread bore, so the nut is captive");
assert(rod_collet_flat >= 0.6, "Thread flats too thin to print");

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

// ---- Section splice ---------------------------------------------------------------------------
// Each section joint is retained by one transverse bar through aligned slots in the boss and
// in the previous section's spigot, trapped by a keeper clipped onto the bar's outer end.
// The slot is 0.3 mm taller than the 4 mm bar, so the bar can be pushed home; that is the
// joint's axial play. The keeper stops the bar walking inward toward the slider; nothing
// but its grip stops it walking outward, so the joint's retention under vibration is a
// bench-test item.
rod_splice_slot_z = 15;                       // slot floor above the joint's butt face
rod_splice_slot_h = keeper_t_mm + 0.3;
rod_splice_bar_len = 34;

// Slot through the +Y wall only, from just outside the slider passage outward. `base` is
// the butt face's z in the cutting part's frame, `reach` how far out the slot runs.
module rod_splice_slot(base, reach) {
    translate([-keeper_t_mm / 2 - 0.6, rod_spigot_id / 2 - 0.2,
               base + rod_splice_slot_z - (rod_splice_slot_h - keeper_t_mm) / 2])
        cube([keeper_t_mm + 1.2, reach, rod_splice_slot_h]);
}

// A bar with its inner end at y = y0 and its keeper 0.2 mm off the face at y = face,
// centred at z = bar_z.  Shared by the splices and the throttle-end body.
module rod_bar_and_keeper(bar_z, y0, face) {
    translate([keeper_t_mm / 2, y0 + rod_splice_bar_len / 2, bar_z])
        rotate([0, 0, 90]) rod_cross_bar(len = rod_splice_bar_len);
    translate([0, face + 0.2 + keeper_t_mm / 2, bar_z])
        rotate([-90, 0, 0]) rod_bar_keeper();
}

// Bar and keeper installed at a joint whose butt face is at z = `base` (tube-axis frame).
// The bar's inner end stops at the bore wall, 0.4 mm clear of the slider; its tapered end
// leaves the boss on +Y and the keeper sits 0.2 mm off the boss face.
module rod_splice(base) {
    rod_bar_and_keeper(base + rod_splice_slot_z + keeper_t_mm / 2, rod_bore / 2,
                       rod_receiver_boss_od / 2);
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
            rod_splice_slot(0, rod_receiver_boss_od);
        // Transverse-bar slot in this section's male ring.
        rod_splice_slot(len, rod_spigot_od);
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
        rod_splice_slot(rod_segment_lens[0], rod_spigot_od);
    }
}

// The servo section in rod-local coordinates: the eye's bore centre is the origin.
module rod_servo_eye() {
    translate([rod_eye_crank_mm, 0, -6]) rod_servo_section();
}

// A self-supporting lug rooted 0.4 mm into the sleeve's -X wall. Its 45-degree face is on
// the TIP side (+Z), which is DOWN when the capture assembly prints collet end down. It
// engages the slider groove with the specified per-side clearance.
module rod_capture_lug(z = rod_capture_z) {
    width = rod_capture_groove_w - 2 * sliding_clearance_mm;
    depth = rod_capture_groove_depth + sliding_clearance_mm;
    translate([-rod_bore / 2 - sliding_clearance_mm, width / 2, z])
        rotate([90, 0, 0])
            linear_extrude(height = width)
                polygon([[0, 0], [depth, 0], [depth, 2], [0, depth + 2]]);
}

// ---- Collet nose (nose frame: z = 0 at the finger tips, +z toward the servo) -----------------
// Maps the nose frame onto the sleeve's rod-oriented frame and back (an involution).
module rod_nose_place() {
    translate([0, 0, rod_receiver_len]) rotate([180, 0, 0]) children();
}

module rod_collet_nose_solid() {
    $fn = rod_collet_fn;
    w_s0 = rod_collet_shoulder_w0;
    union() {
        // Flare: radius grows toward the tip, so as printed it is an upward-facing cone.
        cylinder(h = rod_collet_flare_len, r1 = rod_collet_tip_r, r2 = rod_collet_finger_r);
        translate([0, 0, rod_collet_flare_len - 0.01])
            cylinder(h = rod_collet_finger_len - rod_collet_flare_len + 0.02,
                     r = rod_collet_finger_r);
        // 45-degree step from the finger shank up to the band root.
        translate([0, 0, rod_collet_finger_len])
            cylinder(h = rod_collet_root - rod_collet_finger_r + 0.01,
                     r1 = rod_collet_finger_r, r2 = rod_collet_root);
        translate([0, 0, rod_collet_band_w0])
            cylinder(h = w_s0 - rod_collet_band_w0 + 0.01, r = rod_collet_root);
        // 45-degree shoulder: stops the nut backing off and carries the square tube.
        translate([0, 0, w_s0])
            cylinder(h = rod_collet_collar_r - rod_collet_root + 0.01,
                     r1 = rod_collet_root, r2 = rod_collet_collar_r);
        translate([0, 0, w_s0 + rod_collet_collar_r - rod_collet_root])
            cylinder(h = rod_collet_nose_top - w_s0 - (rod_collet_collar_r - rod_collet_root),
                     r = rod_collet_collar_r);
        // External thread on the unslotted band.
        translate([0, 0, rod_collet_helix_w0])
            helical_sweep(rod_collet_ridge_profile(), rod_collet_p, 0, rod_collet_ridge_a1);
    }
}

// Finger slots centred on the square bore's corners, with a 45-degree roof so the slot
// end is self-supporting as printed.
module rod_collet_slots() {
    w = collet_slot_w_mm;
    for (k = [0:3]) rotate([0, 0, 45 + 90 * k])
        translate([10, 0, 0]) rotate([90, 0, 90])
            linear_extrude(height = 20)
                polygon([[-w / 2, -1], [w / 2, -1], [w / 2, rod_collet_finger_len],
                         [0, rod_collet_finger_len + w / 2],
                         [-w / 2, rod_collet_finger_len]]);
}

// Index notch: a small V in the +X finger's tip face, the reading line for the ticks.
module rod_collet_index_notch() {
    translate([10, 0, 0]) rotate([90, 0, 90])
        linear_extrude(height = 15)
            polygon([[-1, -1], [1, -1], [0, 1]]);
}

// The captive collet nut in the nose frame, as printed (front face on the bed).
module rod_collet_nut() {
    $fn = rod_collet_fn;
    difference() {
        linear_extrude(height = rod_collet_nut_h)
            difference() {
                circle(r = rod_collet_nut_r);
                for (k = [0:rod_collet_scallops - 1])
                    rotate([0, 0, 360 * k / rod_collet_scallops])
                        translate([rod_collet_nut_r + rod_collet_scallop_r - 1, 0])
                            circle(r = rod_collet_scallop_r, $fn = 48);
            }
        // Inner cone over the flare, extended 1 mm below the bed and 0.5 mm into the bore.
        translate([0, 0, -1])
            cylinder(h = rod_collet_flare_len + 1.5,
                     r1 = rod_collet_tip_r + rod_collet_gr + tan(rod_collet_alpha),
                     r2 = rod_collet_finger_r + rod_collet_gr
                          - 0.5 * tan(rod_collet_alpha));
        // Thread bore (the internal ridges' tips) above the cone.
        translate([0, 0, rod_collet_flare_len])
            cylinder(h = rod_collet_nut_h, r = rod_collet_root + rod_collet_g);
        // Internal thread groove, out through the rear face.
        translate([0, 0, rod_collet_helix_w0])
            helical_sweep(rod_collet_groove_profile(), rod_collet_p,
                          rod_collet_groove_a0, rod_collet_groove_a1);
    }
}

// The nut moved along its own thread by `advance` mm toward the tip (negative = backed
// off toward the shoulder), in the nose frame.
module rod_collet_nut_at(advance = 0) {
    translate([0, 0, -advance]) rotate([0, 0, -360 * advance / rod_collet_p])
        rod_collet_nut();
}

// The fingers' undeflected region, nose frame, for the drive probe.
module rod_collet_finger_zone() {
    translate([0, 0, -1]) cylinder(h = rod_collet_finger_len + 1, r = 30);
}

// ---- Receiver sleeve (rod-oriented sleeve frame: boss face z = 0, collet tip at z = len) ------
module rod_receiver_sleeve() {
    assert(rod_receiver_len > rod_receiver_boss_len + rod_collet_nose_top + 20,
           "Receiver sleeve must have a socket, a tube and the collet nose");
    tube_z1 = rod_receiver_len - rod_collet_nose_top + 1;
    union() {
        difference() {
            union() {
                cube([rod_receiver_boss_od, rod_receiver_boss_od,
                      2 * rod_receiver_socket_depth], center = true);
                // 45-degree step from the boss down to the tube: self-supporting as printed.
                translate([0, 0, rod_receiver_socket_depth])
                    linear_extrude(height = (rod_receiver_boss_od - rod_w) / 2,
                                   scale = rod_w / rod_receiver_boss_od)
                        square(rod_receiver_boss_od, center = true);
                translate([0, 0, rod_receiver_socket_depth])
                    linear_extrude(height = tube_z1 - rod_receiver_socket_depth)
                        square(rod_w, center = true);
                rod_nose_place() rod_collet_nose_solid();
            }
            // Below the boss face.
            translate([0, 0, -50]) cube(100, center = true);
            // Spigot socket and the slider passage.
            translate([0, 0, -1]) linear_extrude(height = rod_receiver_socket_depth + 1)
                square(rod_receiver_socket_od, center = true);
            translate([0, 0, rod_receiver_socket_depth - 2])
                linear_extrude(height = rod_receiver_len)
                    square(rod_bore, center = true);
            // Transverse-bar slot through the receiver boss.
            rod_splice_slot(0, rod_receiver_boss_od);
            rod_nose_place() {
                rod_collet_slots();
                rod_collet_index_notch();
            }
        }
        rod_capture_lug();
    }
}

// ---- Slider ---------------------------------------------------------------------------------
module rod_slider_shoulder() {
    // The closed lower end of the longitudinal groove drives into the sleeve's
    // integral capture lug at the long-span overtravel catch.
    translate([-rod_slider / 2, -rod_slider / 2, 0])
        cube([rod_slider, rod_slider, rod_capture_groove_z0]);
}

// Graduation ticks: a V across the +X flat, 0.4 mm deep, 45-degree faces.
module rod_tick(s, len) {
    translate([rod_slider / 2, -len / 2, s]) rotate([-90, 0, 0])
        linear_extrude(height = len)
            polygon([[0.1, -0.5], [-0.4, 0], [0.1, 0.5]]);
}

module rod_slider(len = rod_slider_len) {
    difference() {
        translate([-rod_slider / 2, -rod_slider / 2, 0])
            cube([rod_slider, rod_slider, len]);
        // Continuous longitudinal capture groove with closed ends, on the -X face.
        translate([-rod_slider / 2 - 0.5, -rod_capture_groove_w / 2,
                   rod_capture_groove_z0])
            cube([rod_capture_groove_depth + 0.5, rod_capture_groove_w,
                  rod_capture_groove_z1 - rod_capture_groove_z0]);
        // Window for the throttle-end body's transverse bar, through Y.
        translate([-keeper_t_mm / 2 - 0.6, -rod_slider / 2 - 1,
                   len + rod_body_bar_z - rod_splice_slot_h / 2])
            cube([keeper_t_mm + 1.2, rod_slider + 2, rod_splice_slot_h]);
        // Ticks for each 5 mm of coarse setting, read at the collet tip.
        for (setting = [rod_min_pin_to_pin:5:rod_min_pin_to_pin + rod_coarse_travel])
            rod_tick(rod_tick_station(setting),
                     setting % 50 == 0 ? 14 : (setting % 10 == 0 ? 9 : 5));
    }
}

// Printable PIP set, collet end down: sleeve, nut and slider in the nose frame, all three
// on the bed. The slider sits at its print pose (tip flush with the finger tips).
module rod_capture_assembly() {
    rod_nose_place() {
        rod_receiver_sleeve();
        translate([0, 0, rod_receiver_len - rod_slider_len]) rod_slider();
    }
    rod_collet_nut();
}

// ---- Throttle-end body (frame: z relative to the slider tip, tube-axis x/y) ----------------
module rod_throttle_body() {
    z0 = -rod_body_back;
    difference() {
        union() {
            translate([0, 0, z0]) rod_square_tube(rod_body_back - rod_body_front);
            // Web from the socket's +X wall out to the boss, along the boss's length.
            translate([rod_fine_web_x0, -12, z0])
                cube([rod_fine_web_x1 - rod_fine_web_x0, 24, rod_fine_boss_z1 - z0]);
            translate([rod_fine_screw_x, 0, z0])
                cylinder(d = rod_fine_boss_od, h = rod_fine_boss_z1 - z0, $fn = 72);
        }
        // Bar slot through the +Y wall.
        translate([-keeper_t_mm / 2 - 0.6, rod_bore / 2 - 0.2,
                   rod_body_bar_z - rod_splice_slot_h / 2])
            cube([keeper_t_mm + 1.2, rod_w, rod_splice_slot_h]);
        // The fine screw's journal bore.
        translate([rod_fine_screw_x, 0, z0 - 1])
            cylinder(d = rod_fine_screw_bore_d, h = rod_fine_boss_z1 - z0 + 2, $fn = 72);
    }
}

module rod_body_bar() {
    rod_bar_and_keeper(rod_body_bar_z, rod_body_bar_y0, rod_w / 2);
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

// The fine screw, axially fixed to the body: flange on the boss's rear face, shaft forward
// through the boss and the fork's nut. Frame: relative to the slider tip.
module rod_fine_screw() {
    translate([rod_fine_screw_x, 0, rod_fine_screw_z1])
        rotate([180, 0, 0]) printed_screw(rod_fine_screw_len);
}

// The printed nut in the fork's lug. Frame: relative to the fork's bore.
module rod_fine_carriage_nut() {
    translate([rod_fine_screw_x, 0, rod_fork_nut_z0])
        printed_nut(h = rod_fine_nut_h);
}

// Jam collar on the screw against the boss's front face. Frame: relative to the slider tip.
module rod_fine_jam_collar() {
    translate([rod_fine_screw_x, 0, rod_fine_boss_z1 + 0.2])
        linear_extrude(height = fine_jam_collar_h_mm)
            difference() {
                circle(d = fine_screw_d_mm + 12);
                circle(d = fine_screw_d_mm - 1.5);
            }
}

// The throttle-end body set at the slider tip: body, bar, keeper, screw and jam collar.
module rod_body_set() {
    rod_throttle_body();
    rod_body_bar();
    rod_fine_screw();
    rod_fine_jam_collar();
}

// The moving end at a setting, about the tube axis: slider, body set, fork and its nut.
module rod_throttle_end(length_mm = rod_len) {
    translate([0, 0, rod_slider_start_z(length_mm)]) rod_slider();
    translate([0, 0, rod_slider_tip_z(length_mm)]) rod_body_set();
    translate([0, 0, rod_throttle_pin_z(length_mm)]) {
        rod_throttle_fork();
        rod_fine_carriage_nut();
    }
}

// ---- Named anti-separation/overtravel geometry --------------------------------------
// The moving shoulder is the closed lower end of the slider's groove; the fixed stop is
// the receiver sleeve's integral capture lug. At the declared 940-1200 mm settings the
// lug sits INSIDE the groove, so the groove ends act as overtravel catches rather than
// length stops. The collet is what must hold the length.
module rod_overstop_lug(length_mm = rod_len) {
    translate([0, 0, rod_slider_start_z(length_mm)]) rod_slider_shoulder();
}

module rod_overstop_stop() {
    translate([0, 0, rod_fixed_chain_len]) rod_capture_lug();
}

// Upper closed groove end, clipped from the actual slider, blocks reverse overtravel.
module rod_overstop_reverse_shoulder(length_mm = rod_len) {
    translate([0, 0, rod_slider_start_z(length_mm)])
        intersection() {
            rod_slider();
            translate([-rod_slider / 2, -rod_slider / 2, rod_capture_groove_z1])
                cube([rod_slider, rod_slider, rod_slider_len - rod_capture_groove_z1]);
        }
}

// The collet nut in the tube-axis frame, advanced along its thread by `advance` mm.
module rod_collet_nut_installed(advance = 0) {
    translate([0, 0, rod_fixed_chain_len]) rod_nose_place() rod_collet_nut_at(advance);
}

// Probe helpers (fit_checks.scad `use`s this file, so it cannot see its variables).
// The slider at its print pose in the sleeve frame (tip flush with the finger tips).
module rod_slider_print_pose() {
    translate([0, 0, rod_receiver_len - rod_slider_len]) rod_slider();
}
// The slider in the throttle-end frame (z relative to its tip).
module rod_slider_from_tip() {
    translate([0, 0, -rod_slider_len]) rod_slider();
}
// The fork, optionally with its fine nut, at fine extension `fine`, slider-tip frame.
module rod_fork_at(fine = 0, with_nut = false) {
    translate([0, 0, rod_fork_offset + fine]) {
        rod_throttle_fork();
        if (with_nut) rod_fine_carriage_nut();
    }
}
// The receiver sleeve installed at the end of the fixed chain, tube-axis frame.
module rod_sleeve_installed() {
    translate([0, 0, rod_fixed_chain_len]) rod_receiver_sleeve();
}

// The fixed chain as installed, about the tube axis: servo section, two chain sections and
// the receiver sleeve, each butted against the previous one.
module rod_fixed_chain() {
    translate([0, 0, rod_segment_offsets[0] - 6]) rod_servo_section();
    translate([0, 0, rod_segment_offsets[1]]) rod_chain_section(rod_segment_lens[1]);
    translate([0, 0, rod_segment_offsets[2]]) rod_chain_section(rod_segment_lens[2]);
    translate([0, 0, rod_fixed_chain_len]) rod_receiver_sleeve();
}

// Butt-face z of the three section joints, about the tube axis.
function rod_splice_bases() = [rod_segment_offsets[1], rod_segment_offsets[2],
                               rod_fixed_chain_len];

// The three section joints' bars and keepers, about the tube axis.
module rod_splices() {
    for (base = rod_splice_bases()) rod_splice(base);
}

// ---- Public assembly and printable selectors --------------------------------------------------------
rod_part = "assembly";
rod_len = 1066.8;   // illustrative default within the requested 940-1200 mm range

module throttle_pushrod(part = rod_part, length_mm = rod_len) {
    assert(length_mm >= rods_span_min_mm && length_mm <= rods_span_max_mm,
           "Illustrative rod setting must lie within the requested 940-1200 mm range");

    if (part == "assembly") {
        // Every member is drawn about the tube axis, which is cranked rod_eye_crank_mm
        // above the eye plane; the servo eye's bore then lands on the rod-local origin.
        // The collet nut is drawn at first contact with the flare.
        translate([rod_eye_crank_mm, 0, 0]) {
            rod_fixed_chain();
            rod_splices();
            rod_collet_nut_installed(rod_collet_contact_advance());
            rod_throttle_end(length_mm);
        }
    } else if (part == "throttle_end") {
        // The moving end only (slider, body set, fork and fine nut), in rod-local
        // coordinates; used by the linkage probes.
        translate([rod_eye_crank_mm, 0, 0]) rod_throttle_end(length_mm);
    } else if (part == "segment_lower") {
        rod_servo_section();
    } else if (part == "segment_middle") {
        rod_chain_section(rod_segment_lens[1]);
    } else if (part == "segment_upper") {
        rod_chain_section(rod_segment_lens[2]);
    } else if (part == "capture_assembly") {
        rod_capture_assembly();
    } else if (part == "splice_wedge") {
        translate([0, 0, keeper_t_mm / 2]) rod_cross_bar();
    } else if (part == "splice_keeper") {
        translate([0, 0, keeper_t_mm / 2]) rod_bar_keeper();
    } else if (part == "servo_eye") {
        rod_servo_eye();
    } else if (part == "throttle_body") {
        // Print frame: rear face (screw-flange side) on the bed.
        translate([0, 0, rod_body_back]) rod_throttle_body();
    } else if (part == "throttle_fork") {
        translate([0, 0, -rod_fork_nut_z0]) rod_throttle_fork();
    } else if (part == "fine_screw") {
        printed_screw(rod_fine_screw_len);
    } else if (part == "fine_carriage_nut") {
        printed_nut(h = rod_fine_nut_h);
    } else if (part == "fine_jam_collar") {
        translate([-rod_fine_screw_x, 0, -rod_fine_boss_z1 - 0.2]) rod_fine_jam_collar();
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
