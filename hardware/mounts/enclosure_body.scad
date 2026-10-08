// Upright electronics enclosure with an integral rear hitch tongue. ASA, bed-down.
// Floor glands point DOWN with drip loops. NOT a certified watertight box.
//
// MOUNT: the tongue (a 16 mm slab off the bottom of the +Y wall, two vertical
// gussets up that wall) rests on the mower's horizontal hitch plate and bolts
// through the plate's existing ball hole. The box stands vertical BESIDE the
// plate and hangs from the tongue as a cantilever. Print bed is z=0 = floor
// bottom = tongue bottom, so no assembly offset exists.
//
// Cantilever check (2 kg box, 6x dynamic factor, bolt 168 mm from box centre):
//   moment at the hitch bolt      ~24 N*m (incl. the camera tower on the tongue)
//   tongue slab alone at the bolt  3.9 MPa  -> FOS 10.7 vs 42 MPa ASA in-plane
//   tongue root with gussets      <0.2 MPa  -> FOS >60 even vs 11 MPa interlayer
//   hitch bolt tension            ~780 N   (40 mm prying lever) -- trivial for M8+
// Bed-down printing puts the slab's bending stress along the filament. The
// gussets are layer-stacked, but their stress is negligible. The single bolt
// resists twist only by friction over the tongue's contact patch: torque it to
// the hardware's rating and, if the plate may be drilled, use the two optional
// M6 anti-twist holes flanking it. Recheck after the first season.
//
// TOWER: six M5 holes and a 30 x 50 mm connector passage in the +Y wall take
// camera_tower.scad's base; two M5 holes in the tongue take its foot.
// Fit a split gasket/seal AFTER threading cables; do not obstruct the sump drain.
include <enclosure_common.scad>
body_part = "body";                // body | hitch_gauge

module enclosure_tongue() {
    y0 = box_inner_w / 2;          // start inside the wall thickness, fused to floor + wall
    difference() {
        union() {
            translate([-tongue_w / 2, y0, 0]) cube([tongue_w, tongue_end_y() - tongue_w / 2 - y0, tongue_t]);
            translate([0, tongue_end_y() - tongue_w / 2, 0]) intersection() {
                cylinder(d = tongue_w, h = tongue_t);
                translate([-tongue_w / 2, 0, 0]) cube([tongue_w, tongue_w, tongue_t]);
            }
            // Two gussets from the tongue up the +Y wall, outside the tower base
            for (x = [-1, 1])
                translate([x * (tongue_w / 2 - tongue_gusset_t / 2) - tongue_gusset_t / 2, 0, 0])
                    rotate([90, 0, 90]) linear_extrude(height = tongue_gusset_t)
                        polygon([[y0, tongue_t - 0.1], [tongue_bolt_y() - 25, tongue_t - 0.1],
                                 [y0, tongue_t + tongue_gusset_h]]);
        }
        translate([0, tongue_bolt_y(), -1]) cylinder(d = hitch_hole_d, h = tongue_t + 2);
        for (x = [-1, 1]) translate([x * tongue_aux_bolt_x, tongue_bolt_y(), -1])
            cylinder(d = m6_clear, h = tongue_t + 2);
        for (p = tower_foot_bolts) translate([p[0], p[1], -1]) cylinder(d = tower_wall_bolt_d, h = tongue_t + 2);
    }
}

module enclosure_body() {
    difference() {
        union() {
            difference() {
                plate(box_outer_l,box_outer_w,box_top,box_corner);
                translate([0,0,box_floor])
                    plate(box_inner_l,box_inner_w,box_top+1,box_corner-box_wall);
            }
            translate([0,0,box_top-box_flange_t]) difference() {
                plate(box_flange_l,box_flange_w,box_flange_t,box_corner+box_flange);
                translate([0,0,-1]) plate(box_inner_l,box_inner_w,box_flange_t+2,box_corner-box_wall);
            }
            for(p=tray_fixings) translate([p[0],p[1],box_floor-0.1])
                cylinder(d=tray_support_d,h=tray_lift+0.1);
            enclosure_tongue();
        }
        translate([0,0,box_top-seal_depth]) enclosure_seal_path(seal_depth+0.5);
        for(p=lid_screw_points) {
            translate([p[0],p[1],box_top-box_flange_t-1]) cylinder(d=lid_screw_d,h=box_flange_t+2);
            translate([p[0],p[1],box_top-box_flange_t-0.1])
                cylinder(d=lid_nut_pocket_d,h=lid_nut_pocket_depth+0.1,$fn=6);
        }
        for(p=gland_centres) translate([p[0],p[1],-1]) cylinder(d=gland_cutout_d,h=box_floor+2);
        translate([box_outer_l/2+1,28,30]) rotate([0,-90,0]) cylinder(d=vent_cutout_d,h=box_wall+2);
        for(p=tray_fixings) translate([p[0],p[1],-1]) cylinder(d=tray_fixing_d,h=tray_bottom_z+2);
        // Tower base bolts and cable port through the +Y wall
        for(p=tower_base_bolts) translate([p[0],box_inner_w/2-1,p[1]])
            rotate([-90,0,0]) cylinder(d=tower_wall_bolt_d,h=box_wall+2);
        translate([-tower_port_w/2,box_inner_w/2-1,tower_port_z-tower_port_h/2])
            cube([tower_port_w,box_wall+2,tower_port_h]);
    }
}

// Small coupon: just the hitch hole. Fit it on the mower's ball/bolt/hole first.
module hitch_gauge() {
    side = max(hitch_hole_d + 30, 60);
    difference() {
        plate(side, side, 3, 8);
        translate([0, 0, -1]) cylinder(d = hitch_hole_d, h = 5);
    }
}

if (body_part == "body") enclosure_body();
else if (body_part == "hitch_gauge") hitch_gauge();
else assert(false, "body_part must be body or hitch_gauge");
