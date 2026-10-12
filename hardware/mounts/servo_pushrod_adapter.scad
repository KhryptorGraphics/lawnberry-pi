// Moving U-CROSSPLATE to pushrod, not a bolt-on disc horn. FDM ASA first article.
// Mounting holes are the arm heads' six photo-derived M2.5 passages, ±0.4 mm slot trim and
// Ø13.5 opening (servo_rear_hole_*); orientation on the moving plate is an assumption.
// Strength: 10 mm pad, two 9 mm full-length ear webs, Ø24 pivot bosses, a clevis floor and
// an inboard gusset. Steel M2.5 through-bolts, OD6 washers and AF5 nuts on the flat pad top.
// Face45 = neutral, face90 = reverse. Eye is up at45 through a+45deg clock offset.
// NO powered-steering strength/fatigue qualification is implied by a valid mesh.
include <enclosure_common.scad>
adapter_part = "right"; // right | left | gauge | assembly
adapter_face_angle_deg = servo_face_neutral_deg;

// Pad-normal (Z) version of arm_servo_holes(): same slots and opening, h = pad thickness.
module servo_adapter_mount_cut(h) {
    pts=servo_adapter_hole_points_mm();
    for(i=[0:len(pts)-1]) {
        travel=servo_rear_hole_travel_mm(i);   // [bracket Y -> V, bracket Z -> X]
        hull() for(dx=[-travel[1]/2,travel[1]/2],dv=[-travel[0]/2,travel[0]/2])
            translate([pts[i][0]+dx,pts[i][1]+dv,-1])
                cylinder(d=servo_mount_clear_d_mm,h=h+2);
    }
    translate([0,0,-1]) cylinder(d=servo_rear_clearance_d_mm,h=h+2);
}
// X reach of the nut-driver envelope around the outermost passage.
function servo_adapter_tool_x_mm() =
    max([for(p=servo_adapter_hole_points_mm()) p[0]])
    +servo_rear_hole_travel_mm(0)[1]/2+servo_mount_tool_d_mm/2;
// Engraved arrow on the top face: points +X, toward the bracket's (±10, 18.5) row and the clevis.
module servo_adapter_orientation_mark(top) {
    translate([0,0,top-0.6]) linear_extrude(1) {
        translate([-29,-0.5]) square([7,1]);
        polygon([[-18,0],[-22,2.5],[-22,-2.5]]);
    }
}
module servo_pushrod_adapter(crank_r=servo_crank_r) {
    pin=servo_adapter_pin_local(crank_r);
    inner=servo_adapter_ear_inner_mm;
    outer=inner+servo_adapter_gap_mm;
    ear=servo_adapter_ear_t_mm;
    t=servo_adapter_pad_t_mm;
    l=servo_adapter_pad_l_mm;
    v0=-l/2;
    r=servo_adapter_pivot_od_mm/2;
    tool_x=servo_adapter_tool_x_mm();
    floor_v=servo_adapter_floor_v_max_mm;
    assert(servo_adapter_gap_mm >= clamp_rod_eye_t_mm+1,"Clevis must clear the rod eye and shims");
    assert(pin[1]-r > 16,"Crank radius puts the pivot into the pad/fastener land");
    assert(servo_adapter_pad_w_mm/2 >= inner-ear,"Inboard ear must grow out of the pad");
    assert(tool_x+1 < inner-ear,"Inboard ear/gusset must leave nut-driver access");
    assert(norm([pin[1]-floor_v,pin[2]-t]) > clamp_rod_eye_sweep_radius_mm()+2,
           "Clevis floor must stay outside the rod-eye sweep");
    difference() {
        union() {
            translate([-servo_adapter_pad_w_mm/2,v0,0]) cube([servo_adapter_pad_w_mm,l,t]);
            // Each ear is a web from the whole pad length up to its pivot boss.
            for(x=[inner-ear,outer]) hull() {
                translate([x,v0,0]) cube([ear,l,t]);
                translate([x,pin[1],pin[2]]) rotate([0,90,0]) cylinder(r=r,h=ear);
            }
            // Clevis floor ties both ears together well behind the moving eye.
            translate([inner-0.01,v0,0]) cube([servo_adapter_gap_mm+0.02,floor_v-v0,t]);
            // Gusset ramp from the pad top into the inboard ear, outboard of every nut/tool.
            hull() {
                translate([tool_x+1,v0,0]) cube([inner-ear-tool_x-1,l,t]);
                translate([inner-ear-1,v0,0]) cube([1.01,l,t+8]);
            }
        }
        servo_adapter_mount_cut(t);
        // Complete X-axis pivot passage; never thread a printed ear.
        translate([inner-ear-1,pin[1],pin[2]]) rotate([0,90,0])
            cylinder(d=m6_clear,h=2*ear+servo_adapter_gap_mm+2);
        servo_adapter_orientation_mark(t);
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
        servo_adapter_mount_cut(3);
        servo_adapter_orientation_mark(3);
    }
}
if(adapter_part=="right") servo_adapter_print();
else if(adapter_part=="left") servo_adapter_print(true);
else if(adapter_part=="gauge") servo_adapter_fit_gauge();
else if(adapter_part=="assembly") {
    color([0.7,0.7,0.7]) servo_moving_bracket(1,adapter_face_angle_deg);
    color([0.16,0.55,0.6]) servo_pushrod_adapter_at(1,adapter_face_angle_deg);
} else assert(false,"Unknown adapter_part");
