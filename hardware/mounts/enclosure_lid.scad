// Print outside-face down. Closed assembly flips ribs/skirt into -Z.
// M4 pan/socket screws and flat washers sit on the flat exterior surface.
// No assumed countersink/head shape or heat-set insert dimensions.
include <enclosure_common.scad>
module enclosure_lid() {
    difference() {
        union() {
            plate(lid_length,lid_width,lid_thickness,box_corner+box_flange+lid_skirt_out);
            translate([0,0,lid_thickness-0.1]) difference() {
                plate(lid_length,lid_width,lid_skirt_drop+0.1,box_corner+box_flange+lid_skirt_out);
                translate([0,0,-1])
                    plate(box_flange_l+2*lid_skirt_clearance,box_flange_w+2*lid_skirt_clearance,
                          lid_skirt_drop+2,box_corner+box_flange+lid_skirt_clearance);
            }
            // Perimeter ribs hang 4mm, not half-height centred cubes.
            for(y=[-52,52]) translate([-74,y-lid_rib_w/2,lid_thickness-0.1])
                cube([148,lid_rib_w,lid_rib_h+0.1]);
            for(x=[-73,73]) translate([x-lid_rib_w/2,-54,lid_thickness-0.1])
                cube([lid_rib_w,108,lid_rib_h+0.1]);
        }
        for(p=lid_screw_points) translate([p[0],p[1],-1]) cylinder(d=lid_screw_d,h=lid_thickness+2);
    }
}
module enclosure_lid_closed() { enclosure_lid_transform() enclosure_lid(); }
enclosure_lid();
