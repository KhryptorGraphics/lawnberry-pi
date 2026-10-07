// Printed, front-fitted sway bars tying each arm to the camera tower.
// Each handed half-jaw occupies only its side of the tower centre plane.
// Together the halves form a rear-open U, fitted from +Y toward the box.
// The intact outboard wall carries the bar. Two short M4 screws per half
// bear outside the tube; neither the jaw nor screws enter the USB cable bore.
// These are first-article position braces, not a qualified steering load path.
//
// LOAD: these are sway/position bars, not the primary steering load path. The
// servo reaction still goes servo -> arm -> sandwich layer -> hitch plate, and that
// path is unqualified. A bar or pad failure is a stop-work condition, not a
// tolerable mode. Printed ASA only; no strength, fatigue or creep claim is made
// for any of it. Bench-load before powered steering.
//
// Print each bar Lying flat (broad face on the bed) so the in-plane bending stress
// runs along the filament, the same rule as camera_tower.scad's segments.
include <enclosure_common.scad>
arm_brace_part = "bar_0";      // bar_0 | bar_1 | bar_2
arm_brace_side = "right";      // right | left; explicit handed print variants

// The selected part's station index. indexOf on a list is not available in this
// OpenSCAD, so the station is selected by name.
function arm_brace_index() =
    arm_brace_part == "bar_0" ? 0 : arm_brace_part == "bar_1" ? 1 : 2;
function arm_brace_station_z() = arm_brace_z[arm_brace_index()];
// A column pad's inboard mate face is a plane of constant assembly X. The
// local wedge's slope cancels the arm's rake; the offset comes from the same
// plane equation used by arm_pad_x_at(), not from an STL measurement.
function arm_brace_station_pad_x(i) =
    arm_point(arm_brace_t(arm_brace_z[i]))[0] + arm_brace_mate_x();
// Keep the printed bar just shy of the pad; M5 clamping closes the gap.
arm_brace_bar_clearance = 0.2;
function arm_brace_bar_len(i = -1) =
    arm_brace_station_pad_x(i < 0 ? arm_brace_index() : i)
    - tower_od/2 - arm_brace_bar_clearance;
function arm_brace_gusset_t() = 10;
function arm_brace_gusset_w() = arm_brace_bar_w+12;
// The wedge is raked, so its two M5 centres on the plane x=mate_x land
// ABOVE the arm axis: z_local = -mate_x*tan(rake) +/- pitch/(2*cos).
// Both the bar's bores and its end land use this shared projection.
function arm_brace_bolt_shift() = -arm_brace_mate_x()*tan(arm_rake_deg);
function arm_brace_bolt_z(sign) =
    arm_brace_bolt_shift() + sign*arm_brace_pad_bolt_pitch/(2*arm_cos());

assert(arm_brace_part == "bar_0" || arm_brace_part == "bar_1"
       || arm_brace_part == "bar_2", "arm_brace_part must be bar_0, bar_1 or bar_2");
assert(arm_brace_side == "right" || arm_brace_side == "left",
       "arm_brace_side must be right or left");
assert(arm_brace_pair_gap > 0 && arm_brace_pair_gap < tower_od,
       "Handed jaws need a positive centre gap smaller than the tower");
assert(arm_brace_z[0] - arm_brace_bar_w/2 > tongue_gusset_h,
       "Lowest sway web runs into the tongue gussets");
assert(arm_brace_z[0] + arm_brace_collar_t/2 < tower_base_top_z-tower_flange_t
       && arm_brace_z[1] - arm_brace_gusset_w()/2
          > tower_base_top_z+tower_flange_t,
       "Sway bars overlap the tower or segment joint flange");
assert(arm_brace_bar_len() > arm_brace_gusset_t(),
       "Pad face is too close to the tower column for the bar gusset");

// Keep the outboard +X wall and half the front wall, with the rear (-Y) open.
// Mirroring this half at the opposite arm leaves the shared centre gap.
// Approach from +Y AFTER the tower is built; -Y would drive the front wall
// through the tube. Captured M4 nuts are top-loaded and screws bear outside.
module arm_brace_collar() {
    difference() {
        cube([arm_brace_collar_od,arm_brace_collar_od,arm_brace_collar_t],
             center=true);
        cube([arm_brace_collar_id,arm_brace_collar_id,
              arm_brace_collar_t+2],center=true);
        translate([-arm_brace_collar_od/2-1,-arm_brace_collar_od/2-1,
                   -arm_brace_collar_t/2-1])
            cube([arm_brace_collar_od+2,
                  (arm_brace_collar_od-arm_brace_collar_id)/2+2,
                  arm_brace_collar_t+2]);
        // Remove the opposite half: two complete collars cannot share a station.
        translate([-arm_brace_collar_od/2-1,-arm_brace_collar_od/2-1,
                   -arm_brace_collar_t/2-1])
            cube([arm_brace_collar_od/2+arm_brace_pair_gap/2+1,
                  arm_brace_collar_od+2,arm_brace_collar_t+2]);
        for(y=[-arm_brace_collar_bolt_y,arm_brace_collar_bolt_y]) {
            translate([tower_od/2-1,y,0]) rotate([0,90,0])
                cylinder(d=m4_clear,
                         h=(arm_brace_collar_od-tower_od)/2+3);
            // 3.6 mm-deep M4 nut trap behind 4.4 mm of the outer wall;
            // top-entry slot permits assembly without an unsupported insert.
            translate([tower_od/2+2,y-4.2,-4.2])
                cube([3.6,8.4,8.4]);
            translate([tower_od/2+2,y-4.2,4.2])
                cube([3.6,8.4,arm_brace_collar_t/2-4.2+0.5]);
        }
    }
}
// The bar proper, in a frame whose origin is the tower column axis, +Z up, +X
// outboard along the bar. Its +X end face is the pad mating face.
// i selects the station; defaulting to the selected part keeps the single-part
// render identical while letting the assembly view draw all three at true length.
module arm_brace_bar(i = -1) {
    idx = i < 0 ? arm_brace_index() : i;
    len = arm_brace_bar_len(idx);
    face = arm_brace_station_pad_x(idx);
    difference() {
        union() {
            arm_brace_collar();
            translate([tower_od/2+0.2,-arm_brace_bar_t/2,-arm_brace_bar_w/2])
                cube([len-0.2,arm_brace_bar_t,arm_brace_bar_w]);
            translate([face-arm_brace_bar_clearance-arm_brace_gusset_t(),
                       -arm_brace_bar_t/2,
                       arm_brace_bolt_shift()-arm_brace_gusset_w()/2])
                cube([arm_brace_gusset_t(),arm_brace_bar_t,arm_brace_gusset_w()]);
        }
        // Both M5 bores must meet the raked pad's two nut pockets in the
        // assembled frame. Drill through the far-end gusset and the face.
        for(sign=[-1,1])
            translate([face-arm_brace_gusset_t()-1,0,arm_brace_bolt_z(sign)])
                rotate([0,90,0])
                    cylinder(d=m5_clear,h=arm_brace_gusset_t()+2);
    }
}
module arm_brace_bar_at(i) {
    translate([0,arm_brace_y,arm_brace_z[i]]) arm_brace_bar(i);
}
// All three stations at once, for the assembly view.
module arm_brace_bars_assembly() { for(i=[0:2]) arm_brace_bar_at(i); }
// One side's full set. The convention MUST match arm_assembly_side(): side < 0 is the
// right-hand (+x) side, unmirrored; side > 0 is the left (-x) side, mirrored. These two
// used to be inverted relative to each other, so the assembly view drew every bar
// across the machine to the opposite arm.
module arm_brace_bars_side(side) {
    if (side < 0) arm_brace_bars_assembly();
    else mirror([1,0,0]) arm_brace_bars_assembly();
}

// rotate([90,0,90]) sends local (x,y,z) to bed (z,x,y).
// Derive each handed part's minimum X before rotation; the rear-wall cut
// and long web determine the other two minima.
module arm_brace_bar_print() {
    min_x = arm_brace_side == "left"
        ? -arm_brace_station_pad_x(arm_brace_index())+arm_brace_bar_clearance
        : arm_brace_pair_gap/2;
    translate([arm_brace_bar_w/2,-min_x,arm_brace_collar_id/2-1])
        rotate([90,0,90])
            if(arm_brace_side == "left") mirror([1,0,0]) arm_brace_bar();
            else arm_brace_bar();
}

// The contact probes sample each actual pad independently; this ECHO is
// diagnostic only and cannot substitute for built-solid clearance checks.
echo(str("arm_support: ", arm_brace_part, " station z=", arm_brace_station_z(),
         " bar_len_mm=", arm_brace_bar_len(),
         " tower_face_x=", tower_od/2,
         " pad_face_x=", arm_brace_station_pad_x(arm_brace_index()),
         " side=", arm_brace_side,
         " print envelope=", arm_brace_station_pad_x(arm_brace_index())
             -arm_brace_bar_clearance-arm_brace_pair_gap/2, " x ",
         arm_brace_collar_od, " x ", arm_brace_gusset_w(), " mm; sway only, not the",
         " steering load path"));

if(arm_brace_part == "bar_0" || arm_brace_part == "bar_1"
   || arm_brace_part == "bar_2") arm_brace_bar_print();
else assert(false,"arm_brace_part must be bar_0, bar_1 or bar_2");
