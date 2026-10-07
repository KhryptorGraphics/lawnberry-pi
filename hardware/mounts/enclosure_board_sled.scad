// Real removable drilling carriers; exact supplier PCB patterns remain unknown.
// Never tie over exposed components. Drill for insulating PCB spacers after
// comparing the purchased boards with the documented usable envelope.
// The four OUTER M3 bores are stack anchors, NOT component mounting holes.
include <enclosure_common.scad>
sled_type = "relay";               // relay | utility

module enclosure_board_sled(kind = "relay") {
    assert(kind == "relay" || kind == "utility");
    l = kind == "relay" ? relay_sled_l : utility_sled_l;
    w = kind == "relay" ? relay_sled_w : utility_sled_w;
    ax = kind == "relay" ? 25 : 33.5;
    ay = kind == "relay" ? 42 : 40;
    difference() {
        plate(l,w,sled_t,4);
        for(x=[-ax,ax],y=[-ay,ay]) translate([x,y,-1]) cylinder(d=m3_clear,h=sled_t+2);
        // Parallel, NON-intersecting slots. They cannot sever a loose island.
        for(x=[-14,14]) translate([x,0,-1]) rotate([0,0,90]) slot(m3_clear,60,sled_t+2);
        // Short transverse tie slots clear every stack anchor and PCB screw slot.
        for(y=[-43,43]) translate([0,y,-1]) slot(3.4,18,sled_t+2);
    }
}
module enclosure_sleds_assembly() {
    for(z=relay_deck_z) translate([relay_pos[0],relay_pos[1],z]) enclosure_board_sled("relay");
    for(z=utility_deck_z) translate([utility_pos[0],utility_pos[1],z]) enclosure_board_sled("utility");
}
enclosure_board_sled(sled_type);
