// Removable electronics tray, ASA, flat underside at z=0.
// Pi5 rotated90deg: real 58x49 pattern is offset10mm from board centre.
// Four M2.5 bores go THROUGH base and posts; no blind holes or duplicated feet.
// Utility stack's M3 columns sit OUTSIDE the Pi outline. Four relay levels on right.
include <enclosure_common.scad>

module electronics_tray() {
    difference() {
        union() {
            plate(tray_length,tray_width,tray_thickness,4);
            for(p=pi_holes) translate([p[0],p[1],tray_thickness-0.1]) cylinder(d=pi_post_d,h=pi_post_h+0.1);
        }
        for(p=pi_holes) translate([p[0],p[1],-1]) cylinder(d=pi_screw_d,h=tray_thickness+pi_post_h+2);
        for(p=tray_fixings) translate([p[0],p[1],-1]) cylinder(d=tray_fixing_d,h=tray_thickness+2);
        for(p=concat(relay_anchors,utility_anchors))
            translate([p[0],p[1],-1]) cylinder(d=m3_clear,h=tray_thickness+2);
        for(p=gland_centres) translate([p[0],p[1],-1]) cylinder(d=20,h=tray_thickness+2);
        // Drain/strain relief, away from posts, support bosses and stack anchors.
        for(x=[-20,20]) translate([x,-49,-1]) rotate([0,0,90]) slot(3.4,8,tray_thickness+2);
        for(x=[-40,0,40]) translate([x,55,-1]) slot(3.4,12,tray_thickness+2);
    }
}
module enclosure_tray_assembly() { translate([0,0,tray_bottom_z]) electronics_tray(); }
electronics_tray();
