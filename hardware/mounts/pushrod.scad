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
// stays on the rod axis (no eccentric bending). The skew is built in: the same prints
// serve both sides, the left rod turned 180 degrees about its own axis.
//
// NO PURCHASED FASTENERS IN THE ROD. Per side: three prints, five bodies.
//   pushrod_servo_end  servo eye, 70 mm stub, 40 mm square tube whose mouth is the joint
//                      socket, with a print-in-place (PIP) joint lever             2 bodies
//   pushrod_sleeve     29.6 mm square spigot, 40 mm sleeve whose mouth carries the PIP rack
//                      lever                                                        2 bodies
//   pushrod_inner      29.1 mm solid bar, lap-bar eye, 19-pocket rack on its +Y face  1 body
// The only steel left at the rod is the two mating pins: the M6 pin in the servo adapter's
// clevis and the M6 pin through the lap-bar clamp's yoke ears, both in double shear.
// Section: 40 mm square tube, 5 mm walls, and the 29.1 mm inner bar (the old 32 mm tube /
// 21.1 mm bar stretched to this span buckles below the servo stall force).
//
// LOCKS: the shared lever of pip_rod_joints.scad, scaled up for this rod.
// * SECTION JOINT. The sleeve's square spigot seats rod_spigot_insert deep in the servo end's
//   bore until its shoulder meets the mouth; the square keys rotation. Closing the joint lever
//   clamps the spigot (lobe flat, over-centre) AND swings a tooth through an arc window in the
//   tube wall into a pocket in the spigot. Tooth flanks, window and pocket walls are arcs about
//   the lever pivot, so pull or push on the rod passes through the pivot and cannot cam the
//   lever open; the load path is spigot -> tooth -> the window's wall (rod_withdraw_blocked).
//   Withdrawal, push and rotation are blocked POSITIVELY, not by friction.
// * ADJUSTABLE PORTION. The same lever on the sleeve's mouth drops its tooth into one of
//   rod_index_count pockets on the bar's +Y face, rod_index_step apart: one setting per pocket.
//   The lobe adds clamp friction, but the tooth is MANDATORY: friction alone near the ~540 N
//   stall is unqualified, and a slipping rod silently changes the steering trim. There is no
//   setting between pockets; finer trim is the servo's software centre offset.
// * Each bore is offset toward its lever by its member's side play, so a locked member,
//   pressed off the lever by the lobe, sits exactly on the rod axis and the end pins stay
//   coaxial with the pieces.
// * Holding target, UNQUALIFIED: each positive lock must hold rod_hold_target_n axially in
//   both directions before the rod steers the mower. Bench-test the coupons (rod_part
//   joint_coupon + joint_coupon_spigot, rack_coupon + rack_coupon_bar), which carry the same
//   modules as the rod pieces plus a Ø6.6 test-pin bore in a solid grip.
//
// PRINTING. The two PIP pieces print UPRIGHT with their lever mouth on the bed: the lever
// prints open beside the mouth, arm radial on the bed, and every body touches the bed. The
// servo end stands 446 mm with its eye on top; the sleeve stands 406 mm with its spigot on
// top. The inner bar prints lying flat (rod +Z up) as before, so its rack pockets are slots in a
// vertical side face and the webs between them shear across the layers, not along one.
// The levers sit on the rod's +Y face, which rod_between_points() turns OUTBOARD on both sides.
//
// Public interface (use <pushrod.scad>):
//   rod_assembly(index, phi), rod_between_points(p1, p2, index), rod_gauge(index)
//   rod_pin_length(index), rod_last_setting(), rod_nominal_setting(), rod_skew()
//   rod_eye_fitting(), rod_servo_end(), rod_sleeve(), rod_inner(), rod_outer(phi)
//   rod_joint_lever(phi), rod_rack_lever(phi), rod_joint_tooth(phi), rod_rack_tooth(phi)
//   (phi: 0 open as printed, 90 closed); probe helpers listed beside their definitions.
// CLI: -D rod_part="servo_end"|"sleeve"|"inner"|"gauge"|"joint_coupon"|"joint_coupon_spigot"
//      |"rack_coupon"|"rack_coupon_bar"; -D rod_gauge_index=0|1|2.
include <enclosure_common.scad>
use <pip_rod_joints.scad>

rod_part = "servo_end";
rod_index = 9;                      // a rack pocket: 0 shortest .. 18 longest, 9 nominal
rod_gauge_index = 1;

rod_w = 40;
rod_wall = 5;
rod_bore_w = rod_w-2*rod_wall;
rod_slide_clr = 0.45;               // per side; tune with rod_gauge()
rod_inner_w = rod_bore_w-2*rod_slide_clr;
rod_spigot_w = rod_bore_w-0.4;      // 0.2 mm per side in the socket
rod_pin_bore = clamp_pin_bore_mm;
// Rack: one setting per pocket, 12 mm apart (finer than the old 15 mm bolt steps).
rod_index_step = 12;
rod_index_count = 19;
rod_nominal_index = 9;
// What the adjustment buys, stated in millimetres rather than asserted in prose:
rod_fwd_tolerance_mm = 101.6;       // +/- 4 in on the fore-aft estimate (TORO_77502_LINKAGE.md)
rod_legacy_span_mm = 210;           // the bolted rod's 210 mm adjustment, kept as a floor
// Layout along the rod, servo pin at x = 0 (print-plan values, not mower measurements).
rod_joint_x = 430;                  // servo end's socket mouth = the sleeve's shoulder
rod_sleeve_x = 766;                 // sleeve mouth: the rack lever
rod_spigot_insert = 70;             // spigot depth in the socket
rod_spigot_lead = 1.5;              // 45-degree lead chamfer on the spigot and bar tips
rod_plug = 5;                       // solid between the spigot and the sleeve's bore roof
rod_roof = rod_bore_w/2;            // bore ends in a 45-degree pyramid (prints unsupported)
rod_inner_reach = 380;              // inner pin to inner tip
rod_min_overlap = 90;               // bar inside the sleeve at the longest setting
rod_tip_margin = 2;                 // bar tip to the sleeve's bore roof at the shortest
rod_bed_x = 398;                    // lying-flat inner bar, usable bed X 400
rod_print_h = 488;                  // upright pieces, usable height 490
rod_hold_target_n = 2*540;          // 2 x the ~540 N stall at a 30 mm horn (UNQUALIFIED)
// Levers (pip_rod_joints.scad), the same for both locks. Load-bearing sizes: the tooth's
// shear face is rod_tooth_w x rod_lever_w; its bearing depth is rod_pocket_d - 0.8 mm tip
// clearance; the pins only locate the lever and carry the clamp reaction.
rod_lever_side = 0;                 // levers on the rod's +Y face
rod_lever_n = [0,cos(rod_lever_side),sin(rod_lever_side)];  // the lever face's outward normal
rod_lever_pivot_out = 4;
rod_lever_arm = 40;
rod_lever_w = 20;
rod_lever_pin_r = 4;
rod_tooth_lt = 30;                  // tooth centre radius about the pivot (= 2.5 pitches)
rod_tooth_w = 7;                    // tooth thickness along the rod, closed
rod_pocket_d = 4.5;
rod_lever_gap = 0.4;                // PIP gap (pip_lever_spec default)
// Bore offsets toward the lever: the member's side play.
rod_joint_dz = (rod_bore_w-rod_spigot_w)/2;
rod_rack_dz = rod_slide_clr;
rod_coupon_grip = 27;               // solid test-pin grip on each coupon
rod_coupon_spigot_stub = 30;

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

rod_neutral = lapbar_pin(1)-servo_crank_pin(1);
rod_lateral_offset = rod_neutral[0];
rod_planar_span = norm([rod_neutral[1], rod_neutral[2]]);
rod_nominal_len = norm(rod_neutral);
rod_skew_deg = asin(rod_lateral_offset/rod_nominal_len);
rod_trim_shift = rod_index_step*sin(rod_skew_deg);  // eye-plane change per index step

function rod_joint_spec() =
    pip_lever_spec(rod_w/2-rod_joint_dz, rod_bore_w/2, rod_spigot_w/2,
                   pivot_out=rod_lever_pivot_out, arm=rod_lever_arm, width=rod_lever_w,
                   pin_r=rod_lever_pin_r, gap=rod_lever_gap, tooth_lt=rod_tooth_lt,
                   tooth_w=rod_tooth_w, pocket_d=rod_pocket_d);
function rod_rack_spec() =
    pip_lever_spec(rod_w/2-rod_rack_dz, rod_bore_w/2, rod_inner_w/2,
                   pivot_out=rod_lever_pivot_out, arm=rod_lever_arm, width=rod_lever_w,
                   pin_r=rod_lever_pin_r, gap=rod_lever_gap, tooth_lt=rod_tooth_lt,
                   tooth_w=rod_tooth_w, pocket_d=rod_pocket_d);
function rod_pin_length(index) = rod_nominal_len+(index-rod_nominal_index)*rod_index_step;
function rod_last_setting() = rod_index_count-1;
rod_min_len = rod_pin_length(0);
rod_max_len = rod_pin_length(rod_last_setting());
function rod_adjust_range_mm() = rod_max_len-rod_min_len;
function rod_inner_tip(index) = rod_pin_length(index)-rod_inner_reach;
rod_fit_reach = rod_fit_len+rod_bridge_len;         // pin to the clean 29.1 mm bar
rod_bore_end = rod_joint_x+rod_plug+rod_roof;        // sleeve bore full section from here
// Closed tooth centre on the rod, and pocket k's centre on the bar (bar pin at 0, bar to -X).
function rod_tooth_x(s, mouth) = mouth-pip_get(s,"rb")-rod_tooth_lt;
function rod_pocket_x(k) = rod_tooth_x(rod_rack_spec(),rod_sleeve_x)-rod_pin_length(k);
rod_pocket_half = rod_tooth_w/2+rod_lever_gap;
// `use <pushrod.scad>` imports functions, not variables: public values go through these.
function rod_nominal_setting() = rod_nominal_index;
function rod_skew() = rod_skew_deg;

assert(rod_part == "servo_end" || rod_part == "sleeve" || rod_part == "inner"
       || rod_part == "gauge" || rod_part == "joint_coupon" || rod_part == "joint_coupon_spigot"
       || rod_part == "rack_coupon" || rod_part == "rack_coupon_bar",
       "rod_part must be servo_end, sleeve, inner, gauge, joint_coupon, joint_coupon_spigot, rack_coupon or rack_coupon_bar");
assert(rod_index >= 0 && rod_index <= rod_last_setting(),
       "rod_index must select a rack pocket");
assert(rod_lateral_offset >= 0,
       "The lap-bar clamp pin must be outboard of (or level with) the servo crank pin");
assert(rod_trim_shift+0.5 <= (servo_adapter_gap_mm-rod_eye_t)/2,
       "Trim moves the servo eye beyond the double-shear adapter's shim clearance");
// The point of the rack: span the mower's fore-aft uncertainty with printed parts, so the clamp
// point stays a choice inside the estimated band instead of a tape reading.
assert(rod_adjust_range_mm() >= 2*rod_fwd_tolerance_mm && rod_adjust_range_mm() >= rod_legacy_span_mm,
       "Rack span is under the estimate's uncertainty or the bolted rod's 210 mm; add pockets");
assert(rod_index_step-2*rod_pocket_half >= 4,
       "Rack pockets would leave less than 4 mm of bar between them");
// The lobe's clamp flat sits under the pivot, rod_tooth_lt from the tooth: half a pitch off the
// pocket row puts it on the bar web between two pockets at every setting, not over a pocket.
assert(abs(rod_tooth_lt % rod_index_step - rod_index_step/2) < 0.01,
       "The rack lever's clamp flat would land over a pocket");
assert(rod_sleeve_x-rod_inner_tip(rod_last_setting()) >= rod_min_overlap,
       "At the longest setting the bar overlaps the sleeve too little");
assert(rod_inner_tip(0) >= rod_bore_end+rod_tip_margin,
       "At the shortest setting the bar tip reaches the sleeve's bore roof");
assert(rod_pocket_x(rod_last_setting())-rod_pocket_half+rod_inner_reach >= 20,
       "The deepest rack pocket lacks bar-tip margin");
assert(-rod_fit_reach-(rod_pocket_x(0)+rod_pocket_half) >= 8,
       "The nearest rack pocket runs into the eye transition");
assert(rod_min_len-rod_fit_reach >= rod_sleeve_x+2,
       "At the shortest setting the eye transition enters the sleeve mouth");
assert(rod_tooth_x(rod_joint_spec(),rod_joint_x)-rod_pocket_half
       -(rod_joint_x-rod_spigot_insert+rod_spigot_lead) >= 10,
       "The joint pocket runs off the spigot's tip");
assert(rod_joint_x+rod_pin_edge+1 <= rod_print_h && rod_sleeve_x-rod_joint_x+rod_spigot_insert <= rod_print_h,
       "An upright rod piece exceeds the usable print height");
assert(rod_inner_reach+rod_pin_edge+1 <= rod_bed_x, "The inner bar exceeds the usable bed X");
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
// Square frustum along X: w0 at x0 to w1 at x1.
module rod_frustum_x(x0, w0, x1, w1) {
    hull() {
        translate([x0,-w0/2,-w0/2]) cube([0.01,w0,w0]);
        translate([x1-0.01,-w1/2,-w1/2]) cube([0.01,w1,w1]);
    }
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
// Bore cavity of a tube: a 45-degree pyramid roof from its apex at x=apex, then the full
// bore to just past x1 (the mouth). Placed at the bore's offset by the caller.
module rod_bore_cavity(apex, x1) {
    rod_frustum_x(apex,0.02,apex+rod_roof,rod_bore_w);
    rod_square(apex+rod_roof-0.01,x1+1,rod_bore_w);
}
// Maps a pip_rod_joints.scad lever placement onto the rod: the lever's housing axis is the
// rod's +X (mouth facing +X at x=mouth), its bore centre offset dz along rod_lever_n, the lever
// on that face. Lever-local +x (outward) -> rod_lever_n; its pivot axis -> rod Z.
module rod_lever_frame(s, mouth, dz) {
    translate(dz*rod_lever_n) multmatrix([[0,0,1,0],[1,0,0,0],[0,1,0,0],[0,0,0,1]])
        pip_at_lever(s,mouth,rod_lever_side) children();
}
module rod_joint_cuts() { rod_lever_frame(rod_joint_spec(),rod_joint_x,rod_joint_dz) pip_lever_housing_cuts(rod_joint_spec()); }
module rod_joint_adds() { rod_lever_frame(rod_joint_spec(),rod_joint_x,rod_joint_dz) pip_lever_housing_adds(rod_joint_spec()); }
module rod_rack_cuts() { rod_lever_frame(rod_rack_spec(),rod_sleeve_x,rod_rack_dz) pip_lever_housing_cuts(rod_rack_spec()); }
module rod_rack_adds() { rod_lever_frame(rod_rack_spec(),rod_sleeve_x,rod_rack_dz) pip_lever_housing_adds(rod_rack_spec()); }
// Pockets, in the frame of a member that sits on the rod axis (its locked pose).
module rod_joint_pocket() { rod_lever_frame(rod_joint_spec(),rod_joint_x,rod_joint_dz) pip_lever_pocket(rod_joint_spec()); }
module rod_rack_pocket() { rod_lever_frame(rod_rack_spec(),rod_sleeve_x,rod_rack_dz) pip_lever_pocket(rod_rack_spec()); }
// Levers and their teeth at angle phi: 0 open (as printed), 90 closed.
module rod_joint_lever(phi = 90) {
    rod_lever_frame(rod_joint_spec(),rod_joint_x,rod_joint_dz) rotate([0,phi,0]) pip_lever(rod_joint_spec());
}
module rod_rack_lever(phi = 90) {
    rod_lever_frame(rod_rack_spec(),rod_sleeve_x,rod_rack_dz) rotate([0,phi,0]) pip_lever(rod_rack_spec());
}
module rod_joint_tooth(phi = 90) {
    rod_lever_frame(rod_joint_spec(),rod_joint_x,rod_joint_dz) rotate([0,phi,0]) pip_lever_tooth(rod_joint_spec());
}
module rod_rack_tooth(phi = 90) {
    rod_lever_frame(rod_rack_spec(),rod_sleeve_x,rod_rack_dz) rotate([0,phi,0]) pip_lever_tooth(rod_rack_spec());
}
// Servo end, rod frame: eye, stub, flare, tube; its bore (offset rod_joint_dz) is the socket.
module rod_servo_end() {
    difference() {
        union() {
            rod_end(0,1);
            rod_square(rod_fit_reach-0.5,rod_servo_stub,rod_inner_w);
            translate([rod_servo_stub,0,0]) rotate([0,90,0])
                linear_extrude(height=rod_servo_flare+1,scale=rod_w/rod_inner_w)
                    square(rod_inner_w,center=true);
            rod_square(rod_tube_start,rod_joint_x,rod_w);
        }
        translate(rod_joint_dz*rod_lever_n) rod_bore_cavity(rod_tube_start,rod_joint_x);
        rod_joint_cuts();
    }
    rod_joint_adds();
}
// The sleeve's spigot (rod frame, on the rod axis), with the joint pocket and a lead chamfer.
module rod_spigot() {
    x0 = rod_joint_x-rod_spigot_insert;
    difference() {
        union() {
            rod_frustum_x(x0,rod_spigot_w-2*rod_spigot_lead,x0+rod_spigot_lead,rod_spigot_w);
            rod_square(x0+rod_spigot_lead-0.01,rod_joint_x+0.5,rod_spigot_w);
        }
        rod_joint_pocket();
    }
}
// Sleeve, rod frame: spigot, plug, 40 mm tube; its bore (offset rod_rack_dz) takes the bar.
module rod_sleeve() {
    difference() {
        union() {
            rod_spigot();
            rod_square(rod_joint_x,rod_sleeve_x,rod_w);
        }
        translate(rod_rack_dz*rod_lever_n) rod_bore_cavity(rod_joint_x+rod_plug,rod_sleeve_x);
        rod_rack_cuts();
    }
    rod_rack_adds();
}
// Inner member in its own frame: lap-bar pin at the origin, bar running to -X. Pocket k is
// the one the rack tooth enters at setting k.
module rod_inner() {
    difference() {
        union() {
            rod_end(0,-1);
            rod_square(-rod_inner_reach+rod_spigot_lead-0.01,-rod_fit_reach+0.5,rod_inner_w);
            rod_frustum_x(-rod_inner_reach,rod_inner_w-2*rod_spigot_lead,
                          -rod_inner_reach+rod_spigot_lead,rod_inner_w);
        }
        for(k=[0:rod_last_setting()]) translate([-rod_pin_length(k),0,0]) rod_rack_pocket();
    }
}
module rod_inner_at(index) { translate([rod_pin_length(index),0,0]) rod_inner(); }
module rod_outer(phi = 90) {
    rod_servo_end();
    rod_sleeve();
    rod_joint_lever(phi);
    rod_rack_lever(phi);
}
module rod_assembly(index = rod_index, phi = 90) {
    assert(index >= 0 && index <= rod_last_setting(), "rod_assembly index out of range");
    rod_outer(phi);
    rod_inner_at(index);
    echo(str("rod index=",index," pin_length_mm=",rod_pin_length(index),
             " inner_overlap_mm=",rod_sleeve_x-rod_inner_tip(index),
             " purchased_fasteners_in_rod=0 (end pins: servo M6, clamp-yoke M6)"));
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
// Probe helpers. The pocket-free envelope of each locked member, so a probe can ask whether
// a closed tooth sits in a real pocket void rather than in a re-derived one.
module rod_joint_member_envelope() {
    rod_square(rod_joint_x-rod_spigot_insert+rod_spigot_lead,rod_joint_x,rod_spigot_w);
}
module rod_rack_member_envelope(index) {
    translate([rod_pin_length(index),0,0])
        rod_square(-rod_inner_reach+rod_spigot_lead,-rod_fit_reach,rod_inner_w);
}
// A ring round a pin axis (global X, i.e. rod Y turned by the skew) at rod x: it meets the
// eye's material only if the eye is really there.
module rod_pin_ring(x) {
    translate([x,0,0]) rotate([0,0,-rod_skew_deg]) rotate([90,0,0])
        difference() {
            cylinder(d=rod_pin_bore+4,h=4,center=true,$fn=48);
            cylinder(d=rod_pin_bore+0.2,h=6,center=true,$fn=48);
        }
}

// Bench coupons: the rod's own lock modules in short pieces with a solid grip and a Ø6.6
// test-pin bore (test-rig hardware only). Load each pair axially in both directions.
module rod_cross_bore(x, axis_y = true) {
    translate([x,0,0]) rotate(axis_y ? [90,0,0] : [0,0,0])
        cylinder(d=rod_pin_bore,h=rod_w+2,center=true,$fn=32);
}
module rod_coupon_housing(mouth, dz, depth) {
    x0 = mouth-depth-rod_roof-rod_coupon_grip;
    difference() {
        rod_square(x0,mouth,rod_w);
        translate(dz*rod_lever_n) rod_bore_cavity(x0+rod_coupon_grip,mouth);
        rod_cross_bore(x0+rod_coupon_grip/2);
        children();
    }
}
rod_joint_coupon_depth = rod_spigot_insert+3;
rod_rack_coupon_bar = [rod_pocket_x(rod_nominal_index)-54, rod_pocket_x(rod_nominal_index)+78];
rod_rack_coupon_depth = rod_sleeve_x-(rod_pin_length(rod_nominal_index)+rod_rack_coupon_bar[0])+3;
module rod_joint_coupon() {
    rod_coupon_housing(rod_joint_x,rod_joint_dz,rod_joint_coupon_depth) rod_joint_cuts();
    rod_joint_adds();
}
module rod_joint_coupon_spigot() {
    difference() {
        union() {
            rod_spigot();
            rod_square(rod_joint_x,rod_joint_x+rod_coupon_spigot_stub,rod_w);
        }
        rod_cross_bore(rod_joint_x+rod_coupon_spigot_stub/2+3);
    }
}
module rod_rack_coupon() {
    rod_coupon_housing(rod_sleeve_x,rod_rack_dz,rod_rack_coupon_depth) rod_rack_cuts();
    rod_rack_adds();
}
// Bar frame, cut between pockets; the test-pin bore runs along Y beneath the pockets.
module rod_rack_coupon_bar() {
    difference() {
        intersection() {
            rod_inner();
            rod_square(rod_rack_coupon_bar[0],rod_rack_coupon_bar[1],rod_w);
        }
        rod_cross_bore(rod_rack_coupon_bar[1]-12,false);
    }
}

echo(str("pushrod Toro 77502 estimate: planar_span_mm=",rod_planar_span,
         " lateral_offset_mm=",rod_lateral_offset," skew_deg=",rod_skew_deg,
         " pin_lengths_mm=",rod_min_len,"/",rod_nominal_len,"/",rod_max_len,
         " rack_pitch_mm=",rod_index_step," settings=",rod_index_count,
         " trim_shift_per_step_mm=",rod_trim_shift,
         " print_heights_mm servo_end=",rod_joint_x+rod_pin_edge," sleeve=",
         rod_sleeve_x-rod_joint_x+rod_spigot_insert," inner_flat=",rod_inner_reach+rod_pin_edge,
         " overlap_at_max_mm=",rod_sleeve_x-rod_inner_tip(rod_last_setting()),
         " hold_target_n=",rod_hold_target_n,"; fit/load testing only"));

// Upright pieces: rod +X turned to -Z, mouth on the bed, lever open (as printed).
module rod_upright(mouth) { translate([0,0,mouth]) rotate([0,90,0]) children(); }
if (rod_part == "servo_end") rod_upright(rod_joint_x) { rod_servo_end(); rod_joint_lever(0); }
if (rod_part == "sleeve") rod_upright(rod_sleeve_x) { rod_sleeve(); rod_rack_lever(0); }
// The bar prints lying flat, rod +Z up; its rack is on the +Y side face.
if (rod_part == "inner") translate([rod_inner_reach,0,rod_inner_w/2]) rod_inner();
if (rod_part == "gauge") rod_gauge();
if (rod_part == "joint_coupon") rod_upright(rod_joint_x) { rod_joint_coupon(); rod_joint_lever(0); }
if (rod_part == "joint_coupon_spigot")
    rod_upright(rod_joint_x+rod_coupon_spigot_stub) rod_joint_coupon_spigot();
if (rod_part == "rack_coupon") rod_upright(rod_sleeve_x) { rod_rack_coupon(); rod_rack_lever(0); }
if (rod_part == "rack_coupon_bar")
    translate([-rod_rack_coupon_bar[0],0,rod_inner_w/2]) rod_rack_coupon_bar();
