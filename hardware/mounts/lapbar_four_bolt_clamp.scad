// FIRST-ARTICLE FIT / LOAD-TEST ONLY. Not powered-operation certified.
// Measure the actual lever tube OD with calipers; the default is only a sample
// coupon size. Printed-plastic clamp creep, fatigue, and torque capacity are unknown.
// Hand-cycle the linkage and inspect for slip, whitening, cracking, and loss of clamp
// after every test. Do not infer a safe bolt torque from this model; use no steel
// backing plate, but do not rely on that as evidence of operating safety.
//
// Local tube axis is X. The two mating halves split at Z=0 in the assembly.
// Four vertical M5 bolts occupy X=+/- clamp_bolt_x_mm, Y=+/- clamp_bolt_y_mm
// (2 longitudinal x 2 transverse). The mating rod pin axis is Z.
//
// Use from another file with: use <lapbar_four_bolt_clamp.scad>;
// Callable modules: clamp_anchor(), clamp_cap(), clamp_assembly().

include <common.scad>
clamp_part = "anchor";                 // anchor | cap | gauge
// Public assembly-space pin centre, before any per-bar clocking transform.
// lapbar_on_bar in assemblies.scad maps local +Z (this pin axis) to global +Y.
function clamp_pin_local() = [0, clamp_w_mm/2 + clamp_yoke_reach_mm/2, 0];
function clamp_tube_diameter_mm() = clamp_tube_od_mm;
function clamp_half_height_mm() = clamp_half_h_mm;
function clamp_width_mm() = clamp_w_mm;
function clamp_length_mm() = clamp_l_mm;
function clamp_bolt_x_mm_value() = clamp_bolt_x_mm;
function clamp_bolt_y_mm_value() = clamp_bolt_y_mm;

// Axis-X cylindrical groove is cut from one half of each print blank. Both mating
// faces remain flat and the groove leaves 4.25 mm nominal web to M5 hole edges.
module clamp_anchor() {
    assert(clamp_tube_od_mm > 0, "Measure tube OD and set clamp_tube_od_mm > 0");
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
    assert(clamp_tube_od_mm > 0, "Measure tube OD and set clamp_tube_od_mm > 0");
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
module clamp_assembly() {
    translate([0,0,-clamp_half_h_mm]) clamp_anchor_print();
    clamp_cap_print();
}

// One-piece fit coupon: a flat annular sample for checking the measured bore size.
// Single-solid gauge coupon. Index 0/1/2 samples tight/nominal/loose diametral fit.
module clamp_gauge(index=clamp_gauge_index) {
    assert(index>=0 && index<=2, "clamp_gauge_index must be 0, 1 or 2");
    clearance=clamp_tube_clearance_mm+(index-1)*0.2;
    difference() {
        cylinder(d=clamp_tube_od_mm+clearance+2*clamp_gauge_wall_mm,h=3);
        translate([0,0,-0.1]) cylinder(d=clamp_tube_od_mm+clearance,h=3.2);
    }
}

if (clamp_part == "anchor") clamp_anchor();
else if (clamp_part == "cap") clamp_cap();
else if (clamp_part == "gauge") clamp_gauge(clamp_gauge_index);
else assert(false, "clamp_part must be anchor, cap or gauge");
