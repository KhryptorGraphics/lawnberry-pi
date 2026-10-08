// Moving U-CROSSPLATE to pushrod, not a bolt-on disc horn. FDM ASA first article.
// User identified the broad8-hole plate in60883207...jpg as the MOVING member.
// Hole positions/plate radius are photo-fitting capacities, not verified factory dimensions.
// Eight through bolts with broad METAL washers/nuts; double-shear M6 eye.
// Face45 = neutral, face90 = reverse. Eye is up at45 through a+45deg clock offset.
// NO powered-steering strength/fatigue qualification is implied by a valid mesh.
include <enclosure_common.scad>
adapter_part = "right"; // right | left | gauge | assembly
adapter_face_angle_deg = servo_face_neutral_deg;

module servo_adapter_mount_slots(h=120) {
    travel=servo_moving_hole_travel_mm();
    for(p=servo_moving_hole_points_mm()) hull()
        for(dx=[-travel[0]/2,travel[0]/2],dv=[-travel[1]/2,travel[1]/2])
            translate([p[0]+dx,p[1]+dv,-20])
                cylinder(d=servo_adapter_bore_d_mm,h=h);
}
module servo_pushrod_adapter(crank_r=servo_crank_r) {
    pin=servo_adapter_pin_local(crank_r);
    inner=servo_adapter_ear_inner_mm;
    outer=inner+servo_adapter_gap_mm;
    ear=servo_adapter_ear_t_mm;
    r=servo_adapter_pivot_od_mm/2;
    assert(servo_adapter_gap_mm >= clamp_rod_eye_t_mm+1,"Clevis must clear the rod eye and shims");
    assert(pin[1]-r > 16,"Crank radius puts the pivot into the pad/fastener land");
    difference() {
        union() {
            translate([-servo_adapter_pad_w_mm/2,-servo_adapter_pad_l_mm/2,0])
                cube([servo_adapter_pad_w_mm,servo_adapter_pad_l_mm,servo_adapter_pad_t_mm]);
            // Root bridge connects both ears BEHIND the eye; no web fills the moving-eye gap.
            translate([28,-22,0]) cube([servo_adapter_outer_x_mm()-28,8,8]);
            for(x=[inner-ear,outer]) hull() {
                translate([x,-22,0]) cube([ear,8,8]);
                translate([x,pin[1],pin[2]]) rotate([0,90,0]) cylinder(r=r,h=ear);
            }
        }
        servo_adapter_mount_slots();
        // A nut/washer/tool can reach every through-bolt from the exposed pad face.
        for(p=servo_moving_hole_points_mm()) hull()
            for(dx=[-2,2],dv=[-4,4])
                translate([p[0]+dx,p[1]+dv,servo_adapter_pad_t_mm])
                    cylinder(d=servo_adapter_washer_od_mm+0.4,h=80);
        // Complete X-axis pivot passage; never thread a small printed ear.
        translate([inner-ear-1,pin[1],pin[2]]) rotate([0,90,0])
            cylinder(d=m6_clear,h=2*ear+servo_adapter_gap_mm+2);
        // Geometric centre/tangent witness, safely away from the eight mounting passages.
        translate([-0.5,-6,servo_adapter_pad_t_mm-0.6]) cube([1,12,1]);
        translate([-3,5,servo_adapter_pad_t_mm-0.6]) rotate([0,0,45]) cube([4,1,1]);
    }
}
module servo_pushrod_adapter_at(side,face_angle_deg=servo_face_neutral_deg,crank_r=servo_crank_r) {
    servo_adapter_pose(side,face_angle_deg) servo_pushrod_adapter(crank_r);
}
module servo_adapter_print(left=false,crank_r=servo_crank_r) {
    pin=servo_adapter_pin_local(crank_r);
    translate([0,0,max(0,servo_adapter_pivot_od_mm/2-pin[2])])
        if(left) mirror([1,0,0]) servo_pushrod_adapter(crank_r);
        else servo_pushrod_adapter(crank_r);
}
module servo_adapter_fit_gauge() {
    difference() {
        translate([-servo_adapter_pad_w_mm/2,-servo_adapter_pad_l_mm/2,0])
            cube([servo_adapter_pad_w_mm,servo_adapter_pad_l_mm,3]);
        servo_adapter_mount_slots();
    }
}
if(adapter_part=="right") servo_adapter_print();
else if(adapter_part=="left") servo_adapter_print(true);
else if(adapter_part=="gauge") servo_adapter_fit_gauge();
else if(adapter_part=="assembly") {
    color([0.7,0.7,0.7]) servo_moving_bracket(1,adapter_face_angle_deg);
    color([0.16,0.55,0.6]) servo_pushrod_adapter_at(1,adapter_face_angle_deg);
} else assert(false,"Unknown adapter_part");
