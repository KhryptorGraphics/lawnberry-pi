// Print outside-face down. Closed assembly flips ribs/skirt into -Z.
// M4 pan/socket screws and flat washers sit on the flat exterior surface.
// No assumed countersink/head shape or heat-set insert dimensions.
// The HEAD lid (12 x M4, skirt, cord-seal face, ribs) plus the two HaLow antenna stations of
// antenna_mount.scad at its back (-Y) corners: a D-flat bulkhead hole through the plate inside
// the seal line, the HEAD ribs cut back round each bulkhead, and a blind post socket in a boss
// under the plate. The socket opens on the outside face, so its roof (the boss floor) prints
// as a 12.4 mm bridge.
include <enclosure_common.scad>
use <antenna_mount.scad>
module enclosure_lid_ribs() {
    // Perimeter ribs hang 4mm, not half-height centred cubes.
    for(y=[-52,52]) translate([-74,y-lid_rib_w/2,lid_thickness-0.1])
        cube([148,lid_rib_w,lid_rib_h+0.1]);
    for(x=[-73,73]) translate([x-lid_rib_w/2,-54,lid_thickness-0.1])
        cube([lid_rib_w,108,lid_rib_h+0.1]);
}
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
            difference() {
                enclosure_lid_ribs();
                enclosure_lid_transform() antenna_lid_rib_clear();
            }
            enclosure_lid_transform() antenna_lid_add();
        }
        for(p=lid_screw_points) translate([p[0],p[1],-1]) cylinder(d=lid_screw_d,h=lid_thickness+2);
        enclosure_lid_transform() antenna_lid_cut();
    }
}
module enclosure_lid_closed() { enclosure_lid_transform() enclosure_lid(); }
enclosure_lid();
