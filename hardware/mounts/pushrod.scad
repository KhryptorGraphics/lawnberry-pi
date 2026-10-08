// Drive-lever pushrods for the Toro TimeCutter MAX 50 in MyRIDE (model 77502).
// FIRST-ARTICLE FIT / LOAD-TEST ONLY: not strength-certified for powered steering.
//
// LENGTH IS ESTIMATED. The rod is generated from servo_crank_pin() and lapbar_pin()
// in enclosure_common.scad: the measured 8-1/4 in / 1 ft 6 in servo station plus a
// Toro-photo estimate of the lap-bar clamp point (TORO_77502_LINKAGE.md). Measure
// the three lapbar_pin_* values and the crank, rebuild, then print.
//
// Each rod is one straight two-force member. Both M6 pin axes are global X, so the
// linkage moves in fore-and-aft planes even though the lap bar sits farther outboard
// than the servo crank. The rod is therefore skewed by rod_skew_deg; each eye fitting
// is turned so its faces stay square to its pin while the line between pin centres
// stays on the rod axis (no eccentric bending). The skew is built in: the same three
// outer pieces and inner member serve both sides, the left rod turned 180 degrees
// about its own axis.
//
// Section: 40 mm square tube, 5 mm walls, in three bolted pieces (each <= 420 mm),
// with a 29.1 mm solid inner bar sliding in the last piece. The old 32 mm tube /
// 21.1 mm bar stretched to this span buckles below the servo stall force.
// Two steel M6 bolts lock the inner bar at one of rod_index_count settings; each
// spigot splice takes two more. The servo end is a single eye on the kit crank's M6
// pin, outboard of the metal arm (one M6, washers set the trim); the disc's perimeter
// holes carry nothing and the arm bolts to the disc spline at the drawing's 61.4 mm
// envelope. The lap-bar end is the single eye that the printed clamp's paired yoke
// ears hold in double shear.
//
// Public interface (use <pushrod.scad>):
//   rod_outer_piece(k), rod_outer(), rod_inner(), rod_assembly(index)
//   rod_eye_fitting(), rod_between_points(p1, p2, index), rod_gauge(index)
//   rod_pin_length(index), rod_last_setting(), rod_lock_x(station)
// CLI: -D rod_part="servo_end"|"middle"|"sleeve"|"inner"|"gauge"; -D rod_gauge_index=0|1|2.
include <enclosure_common.scad>

rod_part = "servo_end";
rod_index = 1;                      // 0 shortest, 1 = estimated neutral span, 2 longest
rod_gauge_index = 1;

rod_w = 40;
rod_wall = 5;
rod_bore_w = rod_w-2*rod_wall;
rod_slide_clr = 0.45;               // per side; tune with rod_gauge()
rod_inner_w = rod_bore_w-2*rod_slide_clr;
rod_spigot_w = rod_bore_w-0.4;      // 0.2 mm per side at each splice
rod_bolt_bore = 6.6;                // M6 clearance for lock and splice bolts
rod_pin_bore = clamp_pin_bore_mm;
rod_index_step = 15;
rod_index_count = 3;
rod_nominal_index = 1;
rod_lock_pitch = 45;                // three steps: inner holes land 15 mm apart, never merge
rod_lock_margin = 18;
rod_min_overlap = 84;
rod_inner_reach = 380;              // inner pin to inner tip
rod_outer_piece_count = 4;
// Each spigot joint is multi-position, which is what removes the last measurement from the build:
// the rod's length is set by which holes the joint bolts pass through, so it spans the machine's
// fore-aft uncertainty without anyone reading a tape. Both pieces of a joint carry the SAME hole
// row at rod_splice_step pitch, so shifting the joint by any whole step keeps both bolts in holes.
// The bolt pair must be spaced a whole number of steps (30 = 2 steps), which is why the row joints
// use [15, 45] and not [15, 35]: 20 mm would make every second position unbolt-able.
// The joint that feeds the inner bar is deliberately NOT one of these: its spigot sits in the same
// bore the inner bar slides through, so it stays short and single-position. That is why the rod has
// four outer pieces rather than three - the extra joint is what buys the adjustment range.
rod_splice_step = 15;
rod_splice_positions = 7;           // joint offsets, 0..6 steps -> 90 mm per row joint
rod_splice_bolts = [15, 15+2*rod_splice_step];
rod_splice_len = 150;               // row joint overlap: deepest hole at 45+90 plus margin
rod_splice_bolts_short = [15, 35];  // the inner-feeding joint: one position only
rod_splice_len_short = 45;
rod_splice_plug = 15;
function rod_splice_len_at(k) =
    k == rod_outer_piece_count-2 ? rod_splice_len_short : rod_splice_len;
// The bolt offsets are per joint too, so a probe can ask the right question of the right joint.
function rod_splice_offsets_at(k) =
    k == rod_outer_piece_count-2 ? rod_splice_bolts_short : rod_splice_bolts;
// What the adjustment buys, stated in millimetres rather than asserted in prose:
rod_fwd_tolerance_mm = 101.6;       // +/- 4 in on the fore-aft estimate (TORO_77502_LINKAGE.md)
function rod_splice_range_mm() =
    (rod_splice_positions-1)*rod_splice_step*(rod_outer_piece_count-1);
function rod_lock_range_mm() = (rod_index_count-1)*rod_index_step;
function rod_adjust_range_mm() = rod_splice_range_mm()+rod_lock_range_mm();
rod_eye_len = clamp_rod_eye_length_mm;
rod_eye_w = clamp_rod_eye_width_mm;
rod_eye_t = clamp_rod_eye_t_mm;
rod_pin_edge = rod_eye_len/2;
rod_neck_len = 8;
rod_taper_len = 20;
rod_fit_len = rod_pin_edge+rod_neck_len+rod_taper_len;
rod_bridge_len = 20;
rod_servo_stub = 70;                // narrow bar clears the servo body through crank travel
rod_servo_flare = 20;
rod_tube_start = rod_servo_stub+rod_servo_flare;
rod_bed_max = 420;

rod_neutral = lapbar_pin(1)-servo_crank_pin(1);
rod_lateral_offset = rod_neutral[0];
rod_planar_span = norm([rod_neutral[1], rod_neutral[2]]);
rod_nominal_len = norm(rod_neutral);
rod_skew_deg = asin(rod_lateral_offset/rod_nominal_len);
rod_trim_shift = rod_index_step*sin(rod_skew_deg);  // eye-plane change per index step

function rod_pin_length(index) = rod_nominal_len+(index-rod_nominal_index)*rod_index_step;
function rod_last_setting() = rod_index_count-1;
rod_min_len = rod_pin_length(0);
rod_max_len = rod_pin_length(rod_last_setting());
rod_outer_end = rod_max_len-rod_inner_reach+rod_min_overlap;
rod_outer_span = rod_outer_end+rod_pin_edge;
function rod_outer_boundary(k) = -rod_pin_edge+k*rod_outer_span/rod_outer_piece_count;
function rod_lock_x(station) = rod_outer_end-rod_lock_margin-(1-station)*rod_lock_pitch;
function rod_inner_tip(index) = rod_pin_length(index)-rod_inner_reach;
function rod_outer_piece_len(k) = rod_outer_span/rod_outer_piece_count
                         +(k < rod_outer_piece_count-1 ? rod_splice_len_at(k) : 0);
rod_inner_part_len = rod_pin_edge+rod_inner_reach;
rod_vent_x = (rod_tube_start+rod_outer_boundary(1)-rod_splice_plug)/2;
// `use <pushrod.scad>` imports functions, not variables: public values go through these.
function rod_nominal_setting() = rod_nominal_index;
function rod_skew() = rod_skew_deg;
function rod_outer_count() = rod_outer_piece_count;
function rod_splice_offsets() = rod_splice_bolts;
function rod_splice_step_mm() = rod_splice_step;
function rod_splice_positions_count() = rod_splice_positions;
function rod_splice_len_mm(k) = rod_splice_len_at(k);
function rod_bolt_bore_mm() = rod_bolt_bore;

assert(rod_part == "servo_end" || rod_part == "middle_a" || rod_part == "middle_b"
       || rod_part == "sleeve" || rod_part == "inner" || rod_part == "gauge",
       "rod_part must be servo_end, middle_a, middle_b, sleeve, inner or gauge");
assert(rod_index >= 0 && rod_index <= rod_last_setting(),
       "rod_index must select a configured lock setting");
assert(rod_lateral_offset >= 0,
       "The lap-bar clamp pin must be outboard of (or level with) the servo crank pin");
assert(rod_trim_shift+0.5 <= (servo_adapter_gap_mm-rod_eye_t)/2,
       "Trim moves the servo eye beyond the double-shear adapter's shim clearance");
assert(rod_lock_pitch % rod_index_step == 0 && rod_lock_pitch/rod_index_step
       >= rod_index_count, "Lock holes for different settings would merge");
assert(rod_lock_pitch-rod_bolt_bore >= 8 && rod_index_step-rod_bolt_bore >= 8,
       "Leave at least 8 mm of material between adjacent M6 holes");
assert(rod_min_overlap >= 2*rod_lock_margin+rod_lock_pitch,
       "Minimum overlap must hold both lock bolts with end margins");
assert(rod_inner_tip(0) > rod_outer_boundary(rod_outer_piece_count-1)
       +rod_splice_len_at(rod_outer_piece_count-2)+2,
       "At the shortest setting the inner bar reaches the last splice spigot");
assert(rod_outer_boundary(1)-rod_splice_plug-rod_tube_start >= 40,
       "Servo-end piece is too short for its stub, flare and hollow tube");
assert(rod_lock_x(0)-rod_bolt_bore > rod_outer_boundary(rod_outer_piece_count-1)
       +rod_splice_len_at(rod_outer_piece_count-2), "Sleeve lock holes run into its splice spigot");
assert(rod_inner_reach-rod_lock_margin >= rod_max_len-rod_lock_x(0),
       "Deepest inner lock hole lacks tip margin");
assert(rod_min_len-rod_lock_x(1) >= rod_fit_len+rod_bridge_len+rod_bolt_bore,
       "Nearest inner lock hole runs into the eye transition");
assert(max([for(k=[0:rod_outer_piece_count-1]) rod_outer_piece_len(k)])+rod_pin_edge
       <= rod_bed_max && rod_inner_part_len+rod_pin_edge <= rod_bed_max,
       "A printed rod piece exceeds the 420 mm bed; add an outer piece");
// Splice rows: adjacent holes must keep 8 mm of material, and the deepest hole has to sit inside
// the overlap with an end margin, or the joint cannot be bolted at its last position.
assert(rod_splice_step-rod_bolt_bore >= 8, "Splice rows would merge into a slotted hole");
assert(rod_splice_bolts[1]+(rod_splice_positions-1)*rod_splice_step+8 <= rod_splice_len,
       "Deepest splice hole runs past the overlap; lengthen rod_splice_len");
// The point of the joint rows: span the mower's fore-aft uncertainty with printed parts, so the
// clamp point stays a choice inside the estimated band instead of a tape reading.
assert(rod_adjust_range_mm() >= 2*rod_fwd_tolerance_mm,
       "Rod adjustment range is under the estimate's uncertainty; add splice positions");
assert(rod_eye_t+0.5 <= clamp_rod_gap_mm, "Rod eye exceeds the clamp yoke clearance");
assert(rod_pin_edge+rod_neck_len >= clamp_rod_eye_sweep_radius_mm()+clamp_yoke_sweep_clearance_mm,
       "Thin eye neck must extend past the clamp yoke sweep before the rod widens");
assert(rod_pin_edge >= 1.5*rod_pin_bore, "Eye pin edge distance is too small");

// Pin at the origin, fitting axis +X, pin axis Y; 8 mm eye thickness along Y.
// Turned by -rod_skew_deg about Z wherever it is placed on the rod axis.
module rod_eye_fitting() {
    difference() {
        union() {
            translate([-rod_pin_edge,-rod_eye_t/2,-rod_eye_w/2])
                cube([rod_pin_edge+rod_eye_len/2+rod_neck_len+1,rod_eye_t,rod_eye_w]);
            // Ends just inside the bridge hull so no coplanar faces round to slivers.
            translate([rod_pin_edge+rod_neck_len,0,0]) rotate([0,90,0])
                linear_extrude(height=rod_taper_len-0.5,
                               scale=[(rod_inner_w-0.4)/rod_eye_w,
                                      (rod_inner_w-0.4)/rod_eye_t])
                    square([rod_eye_w,rod_eye_t],center=true);
        }
        rotate([90,0,0]) cylinder(d=rod_pin_bore,h=rod_eye_t+2,center=true);
    }
}
module rod_square(x0, x1, w) {
    translate([x0,-w/2,-w/2]) cube([x1-x0,w,w]);
}
// Fitting frame at the rod pin x=pin, facing the rod (dir=+1 at the servo, -1 at
// the lap bar), turned so the eye faces stay square to the global pin axis.
module rod_end_frame(pin, dir) {
    translate([pin,0,0]) rotate([0,0,-rod_skew_deg])
        if (dir > 0) children(); else mirror([1,0,0]) children();
}
// Fitting plus a hull into the 29.1 mm bar centred on the pin-to-pin axis.
module rod_end(pin, dir) {
    rod_end_frame(pin,dir) rod_eye_fitting();
    hull() {
        rod_end_frame(pin,dir) rod_square(rod_fit_len-1,rod_fit_len,rod_inner_w);
        rod_square(pin+dir*(rod_fit_len+rod_bridge_len)-(dir > 0 ? 1 : 0),
                   pin+dir*(rod_fit_len+rod_bridge_len)+(dir > 0 ? 0 : 1),rod_inner_w);
    }
}
module rod_tube(x0, x1) {
    difference() {
        rod_square(x0,x1,rod_w);
        rod_square(x0-1,x1+1,rod_bore_w);
    }
}
module rod_vertical_bores(xs, d=rod_bolt_bore) {
    for(x=xs) translate([x,0,-rod_w]) cylinder(d=d,h=2*rod_w);
}
// Outer pieces in the rod frame: servo pin at the origin, rod axis +X.
module rod_outer_piece(k) {
    x0 = rod_outer_boundary(k);
    x1 = rod_outer_boundary(k+1);
    last = k == rod_outer_piece_count-1;
    difference() {
        union() {
            if (k == 0) {
                rod_end(0,1);
                rod_square(rod_fit_len+rod_bridge_len-0.5,rod_servo_stub,rod_inner_w);
                translate([rod_servo_stub,0,0]) rotate([0,90,0])
                    linear_extrude(height=rod_servo_flare+1,scale=rod_w/rod_inner_w)
                        square(rod_inner_w,center=true);
                rod_tube(rod_tube_start,x1);
            } else rod_tube(x0,x1);
            if (!last) {
                // Fused plug inside this piece, then the spigot into the next piece. The joint
                // that feeds the inner bar gets the short spigot (see the parameters above).
                rod_square(x1-rod_splice_plug,x1,rod_bore_w+0.2);
                rod_square(x1-1,x1+rod_splice_len_at(k),rod_spigot_w);
            }
        }
        // Row joints: both sides carry the same pitched row, so any whole-step shift aligns a pair
        // of holes. The inner-feeding joint keeps a single pair, because its spigot is short.
        if (k > 0 && k <= rod_outer_piece_count-2)
            rod_vertical_bores([for(s=rod_splice_bolts, i=[0:rod_splice_positions+1])
                                x0+s+i*rod_splice_step]);
        else if (k > 0) rod_vertical_bores([for(s=rod_splice_offsets_at(k)) x0+s]);
        if (!last && k < rod_outer_piece_count-2)
            rod_vertical_bores([for(s=rod_splice_bolts, i=[0:rod_splice_positions+1])
                                x1+s+i*rod_splice_step]);
        else if (!last) rod_vertical_bores([for(s=rod_splice_offsets_at(k)) x1+s]);
        if (last) rod_vertical_bores([rod_lock_x(0),rod_lock_x(1)]);
        // The servo piece's hollow run is otherwise a sealed print void.
        if (k == 0) rod_vertical_bores([rod_vent_x],5);
    }
}
module rod_outer() { for(k=[0:rod_outer_piece_count-1]) rod_outer_piece(k); }
// Inner member in its own frame: lap-bar pin at the origin, bar running to -X.
module rod_inner() {
    difference() {
        union() {
            rod_end(0,-1);
            rod_square(-rod_inner_reach,-(rod_fit_len+rod_bridge_len)+0.5,rod_inner_w);
        }
        rod_vertical_bores([for(i=[0:rod_last_setting()], s=[0:1])
                            rod_lock_x(s)-rod_pin_length(i)]);
    }
}
module rod_assembly(index = rod_index) {
    assert(index >= 0 && index <= rod_last_setting(), "rod_assembly index out of range");
    rod_outer();
    translate([rod_pin_length(index),0,0]) rod_inner();
    for(s=[0:1]) %translate([rod_lock_x(s),0,-rod_w/2-12])
        cylinder(d=rod_bolt_bore*0.94,h=rod_w+24);
    echo(str("rod index=",index," pin_length_mm=",rod_pin_length(index),
             " inner_overlap_mm=",rod_outer_end-rod_inner_tip(index),
             " steel_M6=2_lock+",2*(rod_outer_piece_count-1),"_splice"));
}
// Places the rod between two pin centres. Both pin axes are global X; the X offset
// between them must equal the built skew at this index, so measure and rebuild
// rather than forcing a different lateral offset.
module rod_between_points(p1, p2, index = rod_nominal_index) {
    v = p2-p1;
    len = norm(v);
    e1 = v/len;
    s = e1[0];
    sgn = s < 0 ? -1 : 1;
    c = cos(rod_skew_deg);
    e2 = (sgn*[1,0,0]-abs(s)*e1)/c;
    e3 = cross(e1,e2);
    assert(abs(len-rod_pin_length(index)) < 0.5,
           "Pin centres do not match this rod setting; choose another index or rebuild");
    assert(abs(asin(abs(s))-rod_skew_deg) < 0.2,
           "Pin centres need a different built skew; rebuild with measured endpoints");
    translate(p1) multmatrix([[e1[0],e2[0],e3[0],0],[e1[1],e2[1],e3[1],0],
                              [e1[2],e2[2],e3[2],0],[0,0,0,1]])
        rod_assembly(index);
}
module rod_gauge(index = rod_gauge_index) {
    assert(index >= 0 && index <= 2, "rod_gauge_index must be 0, 1, or 2");
    clr = rod_slide_clr+(index-1)*0.2;
    difference() {
        cube([rod_w,rod_w,rod_w]);
        translate([-0.1,(rod_w-rod_inner_w)/2-clr,(rod_w-rod_inner_w)/2-clr])
            cube([rod_w+0.2,rod_inner_w+2*clr,rod_inner_w+2*clr]);
    }
}

echo(str("pushrod Toro 77502 estimate: planar_span_mm=",rod_planar_span,
         " lateral_offset_mm=",rod_lateral_offset," skew_deg=",rod_skew_deg,
         " pin_lengths_mm=",rod_min_len,"/",rod_nominal_len,"/",rod_max_len,
         " trim_shift_per_step_mm=",rod_trim_shift,
         " pieces_mm=",[for(k=[0:rod_outer_piece_count-1]) rod_outer_piece_len(k)+rod_pin_edge],
         "+inner ",rod_inner_part_len+rod_pin_edge,"; fit/load testing only"));

// Every piece prints lying flat: rod axis along bed X, bolts vertical.
if (rod_part == "servo_end") translate([0,0,rod_w/2]) rod_outer_piece(0);
if (rod_part == "middle_a")
    translate([-rod_outer_boundary(1),0,rod_w/2]) rod_outer_piece(1);
if (rod_part == "middle_b")
    translate([-rod_outer_boundary(2),0,rod_w/2]) rod_outer_piece(2);
if (rod_part == "sleeve")
    translate([-rod_outer_boundary(3),0,rod_w/2]) rod_outer_piece(3);
if (rod_part == "inner") translate([rod_inner_reach,0,rod_inner_w/2]) rod_inner();
if (rod_part == "gauge") rod_gauge();
