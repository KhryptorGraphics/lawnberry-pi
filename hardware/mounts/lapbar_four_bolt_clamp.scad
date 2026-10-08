// FIRST-ARTICLE FIT / LOAD-TEST ONLY. Not powered-operation certified.
// The bar size is set by choosing a sleeve from the set in common.scad (clamp_bar_od_mm), so no
// caliper reading is needed; printed-plastic clamp creep, fatigue, and torque capacity are unknown.
// Hand-cycle the linkage and inspect for slip, whitening, cracking, and loss of clamp
// after every test. Do not infer a safe bolt torque from this model; use no steel
// backing plate, but do not rely on that as evidence of operating safety.
//
// Local tube axis is X. The two mating halves split at Z=0 in the assembly.
// Four vertical M5 bolts occupy X=+/- clamp_bolt_x_mm, Y=+/- clamp_bolt_y_mm
// (2 longitudinal x 2 transverse). The mating rod pin axis is Z.
//
// Use from another file with: use <lapbar_four_bolt_clamp.scad>;
// Callable modules: clamp_anchor(), clamp_cap(), clamp_insert(), clamp_inserts_assembly(),
// clamp_assembly(). The clamp bore is fixed; a sleeve from clamp_bar_od_mm sets the bar size.

include <common.scad>
clamp_part = "anchor";                 // anchor | cap | insert | gauge
clamp_insert_index = 1;                // sleeve size: 0 tight, 1 nominal, 2 loose
// Public assembly-space pin centre, before any per-bar clocking transform.
// lapbar_on_bar in assemblies.scad maps local +Z (this pin axis) to global +Y.
function clamp_pin_local() = [0, clamp_w_mm/2 + clamp_yoke_reach_mm/2, 0];
// The bar the assembly views draw. The real bar is chosen by fitting a sleeve from the set, so
// there is no single nominal diameter any more; the middle entry stands in for the illustration.
function clamp_display_bar_mm() = clamp_insert_bar_mm(floor((clamp_insert_count()-1)/2));
function clamp_half_height_mm() = clamp_half_h_mm;
function clamp_width_mm() = clamp_w_mm;
function clamp_length_mm() = clamp_l_mm;
function clamp_bolt_x_mm_value() = clamp_bolt_x_mm;
function clamp_bolt_y_mm_value() = clamp_bolt_y_mm;

// Axis-X cylindrical groove is cut from one half of each print blank. Both mating
// faces remain flat and the groove leaves 4.25 mm nominal web to M5 hole edges.
module clamp_anchor() {
    assert(clamp_insert_count() >= 1, "clamp_bar_od_mm needs at least one bar size");
    assert(clamp_bolt_y_mm - clamp_r_mm - clamp_bolt_clear_mm / 2 >= 4,
           "Increase transverse bolt offset for at least 4 mm groove-to-hole web");
    assert(clamp_rod_gap_mm >= clamp_rod_eye_t_mm+0.5,
           "Increase yoke gap/coupon clearance around the printed rod eye");
    assert(clamp_rod_gap_mm < 2*clamp_half_h_mm,
           "Yoke gap leaves insufficient material in the paired ears");
    assert(clamp_yoke_span_mm <= clamp_l_mm-8,
           "Widen clamp length for sound yoke-ear edge margins");
    assert(clamp_yoke_span_mm/2
           >= clamp_rod_eye_sweep_radius_mm()+clamp_yoke_sweep_clearance_mm,
           "Yoke span must clear the full planar eye sweep");
    assert(clamp_yoke_reach_mm/2
           >= clamp_rod_eye_sweep_radius_mm()+clamp_yoke_sweep_clearance_mm,
           "Yoke reach must clear the full planar eye sweep");
    assert(clamp_shell_mm-clamp_yoke_root_overlap_mm
           >= clamp_nut_af_mm/(2*cos(30))+1,
           "Yoke root intrudes into an M5 nut pocket; increase shell or reduce overlap");
    assert(clamp_half_h_mm <= 500, "Clamp height exceeds printer Z");
    assert(clamp_l_mm <= 420 && clamp_w_mm+clamp_yoke_reach_mm <= 420,
           "Clamp footprint exceeds 420 mm printer bed");
    clamp_anchor_print();
}

module clamp_anchor_print() {
    difference() {
        union() {
            translate([-clamp_l_mm/2,-clamp_w_mm/2,0])
                cube([clamp_l_mm,clamp_w_mm,clamp_half_h_mm]);
            // Lower yoke ear becomes the negative-Z/negative-Y clamp half at assembly.
            translate([-clamp_yoke_span_mm/2,
                       clamp_w_mm/2-clamp_yoke_root_overlap_mm,0])
                cube([clamp_yoke_span_mm,clamp_yoke_reach_mm+clamp_yoke_root_overlap_mm,
                      clamp_half_h_mm-clamp_rod_gap_mm/2]);
        }
        translate([-clamp_l_mm/2-1,0,clamp_half_h_mm])
            rotate([0,90,0]) cylinder(r=clamp_r_mm,h=clamp_l_mm+2);
        for(x=[-clamp_bolt_x_mm,clamp_bolt_x_mm],y=[-clamp_bolt_y_mm,clamp_bolt_y_mm]) {
            translate([x,y,-0.1]) cylinder(d=clamp_bolt_clear_mm,h=clamp_half_h_mm+0.2);
            translate([x,y,-0.1]) cylinder(d=clamp_nut_af_mm/cos(30),
                h=clamp_nut_depth_mm+0.1,$fn=6);
        }
        translate([0,clamp_pin_local()[1],-1])
            cylinder(d=clamp_pin_bore_mm,h=clamp_half_h_mm+2);
    }
}

module clamp_cap() {
    assert(clamp_insert_count() >= 1, "clamp_bar_od_mm needs at least one bar size");
    assert(clamp_bolt_y_mm - clamp_r_mm - clamp_bolt_clear_mm / 2 >= 4,
           "Increase transverse bolt offset for at least 4 mm groove-to-hole web");
    assert(clamp_rod_gap_mm >= clamp_rod_eye_t_mm+0.5
           && clamp_rod_gap_mm < 2*clamp_half_h_mm,
           "Yoke opening must clear the rod eye while leaving two sound ears");
    assert(clamp_yoke_span_mm <= clamp_l_mm-8,
           "Widen clamp length for yoke-ear end margins");
    assert(clamp_yoke_span_mm/2
           >= clamp_rod_eye_sweep_radius_mm()+clamp_yoke_sweep_clearance_mm,
           "Yoke span must clear the full planar eye sweep");
    assert(clamp_yoke_reach_mm/2
           >= clamp_rod_eye_sweep_radius_mm()+clamp_yoke_sweep_clearance_mm,
           "Yoke reach must clear the full planar eye sweep");
    assert(clamp_shell_mm-clamp_yoke_root_overlap_mm
           >= clamp_nut_af_mm/(2*cos(30))+1,
           "Yoke root intrudes into an M5 nut pocket; increase shell or reduce overlap");
    assert(clamp_half_h_mm <= 500 && clamp_l_mm <= 420
           && clamp_w_mm+clamp_yoke_reach_mm <= 420,
           "Clamp cap exceeds the 420x420x500 printer envelope");
    clamp_cap_print();
}

module clamp_cap_print() {
    difference() {
        union() {
            translate([-clamp_l_mm/2,-clamp_w_mm/2,0])
                cube([clamp_l_mm,clamp_w_mm,clamp_half_h_mm]);
            // Upper yoke ear, paired with the lower anchor ear around the rod eye.
            translate([-clamp_yoke_span_mm/2,
                       clamp_w_mm/2-clamp_yoke_root_overlap_mm,
                       clamp_rod_gap_mm/2])
                cube([clamp_yoke_span_mm,clamp_yoke_reach_mm+clamp_yoke_root_overlap_mm,
                      clamp_half_h_mm-clamp_rod_gap_mm/2]);
        }
        translate([-clamp_l_mm/2-1,0,0])
            rotate([0,90,0]) cylinder(r=clamp_r_mm,h=clamp_l_mm+2);
        for(x=[-clamp_bolt_x_mm,clamp_bolt_x_mm],y=[-clamp_bolt_y_mm,clamp_bolt_y_mm])
            translate([x,y,-0.1]) cylinder(d=clamp_bolt_clear_mm,h=clamp_half_h_mm+0.2);
        translate([0,clamp_pin_local()[1],-1])
            cylinder(d=clamp_pin_bore_mm,h=clamp_half_h_mm+2);
    }
}

// Translate the print-oriented anchor down so its groove/nut pockets face the
// split plane and the cap/anchor bore halves align about the tube centre at z=0.
module clamp_assembly(index=clamp_insert_index) {
    translate([0,0,-clamp_half_h_mm]) clamp_anchor_print();
    clamp_cap_print();
    clamp_inserts_assembly(index);
}

// ---- Interchangeable split sleeve: one clamp body, a set of bar sizes ------------------
// The clamp bore fits the LARGEST sleeve, so the body, bolts, yoke and pin are printed once per
// side whatever the bar measures. The bar size is set by fitting a sleeve: the set spans the
// plausible lever range, and each sleeve's bore is a slip fit over its own bar OD, so trying one
// on the bar is the whole fit check - no calipers needed. ONE printed half serves both clamp
// halves - turn it 180 deg about the bar axis and its flanges land on the opposite end faces -
// so print two per
// side of the chosen size. Flanges bear on the clamp's end faces to stop the
// sleeve walking along the bar; they are not structural.
module clamp_insert_half(index=clamp_insert_index) {
    bore = clamp_insert_bore_mm(index);
    od = clamp_insert_od_mm;
    flange_od = clamp_insert_flange_od_mm(index);
    // The flanges are bumps ON the tube, which projects past the body's end faces. A flange
    // inside the body's length is wider than its bore and collides; one that merely touches the
    // tube's end face is a partial face-to-face union and comes out non-manifold. Overlapping it
    // in 3D on a longer tube is the only arrangement that is both clean and outside the body.
    tube_len = clamp_l_mm+2*clamp_insert_protrusion_mm;
    flange_in = clamp_l_mm/2+clamp_insert_flange_gap_mm;
    assert(index >= 0 && index < clamp_insert_count(),
           "clamp_insert_index out of range; see clamp_bar_od_mm");
    assert(clamp_insert_wall_at_mm(index) >= 2.5,
           "Sleeve wall under 2.5 mm for this bar: raise clamp_body_bore_mm");
    assert(flange_od/2+clamp_bolt_clear_mm/2+1 <= clamp_bolt_y_mm,
           "Sleeve flange runs into an M5 bolt hole; increase clamp_bolt_y_mm");
    assert(flange_in+clamp_insert_flange_t_mm+0.2 <= tube_len/2,
           "Sleeve tube too short past the body for its flanges; raise clamp_insert_protrusion_mm");
    difference() {
        intersection() {
            union() {
                translate([-tube_len/2,0,0]) rotate([0,90,0])
                    difference() {
                        cylinder(d=od,h=tube_len);
                        translate([0,0,-1]) cylinder(d=bore,h=tube_len+2);
                    }
                for(x=[-flange_in-clamp_insert_flange_t_mm,flange_in])
                    translate([x,0,0]) rotate([0,90,0])
                        difference() {
                            cylinder(d=flange_od,h=clamp_insert_flange_t_mm);
                            translate([0,0,-1])
                                cylinder(d=bore,h=clamp_insert_flange_t_mm+2);
                        }
            }
            translate([-100,-100,-100]) cube([200,200,100]);   // keep the lower half only
        }
    }
}
// Print pose: the same solid lifted so its split face sits on the bed.
module clamp_insert(index=clamp_insert_index) {
    translate([0,0,clamp_insert_flange_od_mm(index)/2]) clamp_insert_half(index);
}
// A sleeve fitted to both clamp halves, in the clamp's own frame.
module clamp_inserts_assembly(index=clamp_insert_index) {
    clamp_insert_half(index);
    rotate([180,0,0]) clamp_insert_half(index);
}

// The old one-piece gauge coupon is gone: the sleeve's bore is the same slip fit over its own bar
// OD, so the sleeve IS the coupon, and one print both checks the bar and does the job.

if (clamp_part == "anchor") clamp_anchor();
else if (clamp_part == "cap") clamp_cap();
else if (clamp_part == "insert") clamp_insert(clamp_insert_index);
else assert(false, "clamp_part must be anchor, cap or insert");
