// Hollow camera tower: stands on the enclosure's hitch tongue, bolts to its +Y
// wall, and lifts camera_mount.scad's camera_foot above the seat back. The bore
// routes a USB camera cable down the column and through a wall port into the
// box, between the two electronics stacks. ASA.
//
// PARTS (select with -D tower_part=...):
//   base      L-foot on the tongue + wall flange + first 160 mm of column with the
//             cable port, a sump below the port and a drain. Prints FLANGE-DOWN
//             (wall plate on the bed, column horizontal, supports under the
//             column and its joint flange) so column bending stress runs along
//             the filament. Bed footprint ~110 x 175, height ~75.
//   segment   Flanged column length; N identical pieces (tower_seg_count()).
//             Prints LYING FLAT along the bed on one tube face. Bottom flange is
//             plain; top flange carries the spigot into the next piece.
//   cap       Closes the column, seats the camera foot (2x M4 at +-22 with
//             side-entry nut traps), lets the cable out sideways toward the box
//             below a solid roof so rain cannot run straight down the bore.
//   backing   Drilling template for a metal strip INSIDE the box behind the wall
//             flange: spreads the six M5 loads over the 4 mm ASA wall.
//
// LOAD (0.96 kg column + 0.3 kg camera/foot/cap, 6x dynamic, lateral):
//   base moment  ~46 N*m  -> 50x50x5 tube 3.7 MPa: FOS 11 in-plane, 2.9 across
//                layers. THAT is why every piece prints with its axis horizontal.
//   wall bolts   ~150 N each (6x M5 on 80x100) -> ~2.8 MPa washer bearing on ASA
//   joint bolts  ~200 N each (4x M4 on 56 pitch at 0.5 m)
// The joint spigot (0.4 mm/side clearance) is alignment only; the flange bolts
// carry the moment. No fatigue data for FDM ASA exists here: cycle it, inspect
// flange roots and the wall after the first season, and brace the column to
// the box lid flange if it hums at engine speed.
//
// tower_height (tongue top to camera surface) is a PLACEHOLDER 900 mm: measure
// from the hitch plate to the required line of sight over the seat back first.
include <enclosure_common.scad>
tower_part = "base";               // base | segment | cap | backing

tower_bore = tower_od - 2 * tower_wall;
spigot_od = tower_bore - 2 * tower_spigot_clear;
spigot_wall = 3.5;
wall_y = box_outer_w / 2;          // box wall outer face
foot_top_z = tongue_t + tower_base_t;
nut_af = 7.2;                      // M4 nut across flats + allowance
nut_t = 3.4;

module tower_tube(len) {
    difference() {
        translate([-tower_od / 2, -tower_od / 2, 0]) cube([tower_od, tower_od, len]);
        translate([-tower_bore / 2, -tower_bore / 2, -1]) cube([tower_bore, tower_bore, len + 2]);
    }
}
module tower_joint_flange() {
    difference() {
        translate([-tower_flange / 2, -tower_flange / 2, 0]) cube([tower_flange, tower_flange, tower_flange_t]);
        translate([-tower_bore / 2, -tower_bore / 2, -1]) cube([tower_bore, tower_bore, tower_flange_t + 2]);
        for (x = [-1, 1], y = [-1, 1])
            translate([x * tower_flange_bolt, y * tower_flange_bolt, -1]) cylinder(d = m4_clear, h = tower_flange_t + 2);
    }
}
// Spigot with a 2 mm collar below z=0 spanning the full tube OD: the collar is
// what fuses the spigot to the flange it stands on (a 39.2 mm square inside a
// 40 mm bore would otherwise touch nothing). Bore narrows to the spigot's
// inner size (32 mm) for those 2 mm -- still passes a USB-A plug.
module tower_spigot() {
    inner = spigot_od - 2 * spigot_wall;
    difference() {
        union() {
            translate([-tower_od / 2, -tower_od / 2, -2]) cube([tower_od, tower_od, 2]);
            translate([-spigot_od / 2, -spigot_od / 2, 0]) cube([spigot_od, spigot_od, tower_spigot_len]);
        }
        translate([-inner / 2, -inner / 2, -3]) cube([inner, inner, tower_spigot_len + 4]);
    }
}

// ---- base, in ASSEMBLY coordinates (box frame): column centred at (0, tower_y) ----
module tower_base_assembly() {
    foot_y1 = tower_y + tower_od / 2;
    column_top = tower_base_top_z - tower_flange_t;
    web_len = tower_y - tower_od / 2 - wall_y - tower_base_t;
    difference() {
        union() {
            translate([-tower_base_w / 2, wall_y, tongue_t]) cube([tower_base_w, tower_base_t, tower_base_h]);
            translate([-tower_base_w / 2, wall_y, tongue_t]) cube([tower_base_w, foot_y1 - wall_y, tower_base_t]);
            translate([0, tower_y, foot_top_z - 0.1]) tower_tube(column_top - foot_top_z + 0.1);
            for (x = [-1, 1])
                translate([x * (tower_od / 2 - 4) - 4, wall_y + tower_base_t - 0.1, foot_top_z - 0.1])
                    cube([8, web_len + 0.2, column_top - foot_top_z]);
            translate([0, tower_y, column_top - 0.1]) tower_joint_flange_solid(tower_flange_t + 0.1);
            translate([0, tower_y, tower_base_top_z]) tower_spigot();
        }
        // bore through column and flange, stopping under the spigot collar...
        translate([-tower_bore / 2, tower_y - tower_bore / 2, foot_top_z + 0.9])
            cube([tower_bore, tower_bore, tower_base_top_z - 2 - (foot_top_z + 0.9)]);
        // ...then the collar's narrower throat through the solid flange top
        translate([-(spigot_od - 2 * spigot_wall) / 2, tower_y - (spigot_od - 2 * spigot_wall) / 2, tower_base_top_z - 3])
            cube([spigot_od - 2 * spigot_wall, spigot_od - 2 * spigot_wall, 4]);
        for (x = [-1, 1], y = [-1, 1])
            translate([x * tower_flange_bolt, tower_y + y * tower_flange_bolt, column_top - 1])
                cylinder(d = m4_clear, h = tower_flange_t + 2);
        // cable port: column bore -> webs gap -> wall flange (box wall is cut in enclosure_body.scad)
        translate([0, wall_y - 1, tower_port_z]) rotate([-90, 0, 0])
            cylinder(d = tower_port_d, h = tower_y - wall_y + 2);
        // sump drain out the +Y face, 4 mm above the bore floor
        translate([0, tower_y, foot_top_z + 0.9 + 4]) rotate([-90, 0, 0]) cylinder(d = 5, h = tower_od);
        for (p = tower_base_bolts) translate([p[0], wall_y - 1, p[1]]) rotate([-90, 0, 0])
            cylinder(d = tower_wall_bolt_d, h = tower_base_t + 2);
        for (p = tower_foot_bolts) translate([p[0], p[1], tongue_t - 1]) cylinder(d = tower_wall_bolt_d, h = tower_base_t + 2);
    }
}
module tower_joint_flange_solid(t) {
    translate([-tower_flange / 2, -tower_flange / 2, 0]) cube([tower_flange, tower_flange, t]);
}
// Print orientation: wall-flange face on the bed. rotate([90,0,0]) maps y->z,
// so the wall face (y=wall_y) lands at z=wall_y; shift it down to z=0.
module tower_base_print() {
    translate([0, 0, -wall_y]) rotate([90, 0, 0]) tower_base_assembly();
}

// ---- segment, local: bottom flange face at z=0, top spigot tip at len+spigot ----
module tower_segment(len = tower_seg_len()) {
    union() {
        tower_joint_flange();
        tower_tube(len);
        translate([0, 0, len - tower_flange_t]) tower_joint_flange();
        translate([0, 0, len]) tower_spigot();
    }
}
// Short piece for exploded views only; not a printable part.
module tower_segment_stub() { tower_segment(60); }
module tower_segment_print() {
    translate([0, 0, tower_flange / 2]) rotate([0, 90, 0]) tower_segment();
}

// ---- cap, local: flange underside at z=0 ----
module tower_cap() {
    cav_h = tower_spigot_len + 3;
    difference() {
        translate([-tower_flange / 2, -tower_flange / 2, 0]) cube([tower_flange, tower_flange, tower_cap_t]);
        // socket for the top spigot, with headroom
        translate([-tower_bore / 2, -tower_bore / 2, -1]) cube([tower_bore, tower_bore, cav_h + 1]);
        // cable exit toward the box (-Y), from the socket out through the side, under the roof
        translate([-9, -tower_flange / 2 - 1, cav_h - 11]) cube([18, tower_flange / 2 - tower_bore / 2 + 2, 10]);
        for (x = [-1, 1], y = [-1, 1])
            translate([x * tower_flange_bolt, y * tower_flange_bolt, -1]) cylinder(d = m4_clear, h = tower_cap_t + 2);
        // camera_foot bolts through the roof, nuts slid in from the +-X faces
        for (x = [-1, 1]) {
            translate([x * cam_foot_bolt_x, 0, cav_h]) cylinder(d = m4_clear, h = tower_cap_t);
            translate([x > 0 ? cam_foot_bolt_x - nut_af / 2 : -tower_flange / 2 - 1, -nut_af / 2, cav_h + 2])
                cube([tower_flange / 2 - cam_foot_bolt_x + nut_af / 2 + 1, nut_af, nut_t]);
        }
    }
}

// ---- internal backing strip template: wall bolts + port, sits inside on the +Y wall ----
module tower_backing() {
    difference() {
        plate(tower_base_w, 130, 3, 6);
        for (p = tower_base_bolts) translate([p[0], p[1] - 90, -1]) cylinder(d = tower_wall_bolt_d, h = 5);
        translate([0, tower_port_z - 90, -1]) cylinder(d = tower_port_d, h = 5);
    }
}

// ---- assembly helper (box frame): base, N segments, cap ----
module camera_tower_assembly() {
    tower_base_assembly();
    for (i = [0 : tower_seg_count() - 1])
        translate([0, tower_y, tower_flange_z(i)]) tower_segment();
    translate([0, tower_y, tower_flange_z(tower_seg_count())]) tower_cap();
}
function tower_cap_top_z() = tower_flange_z(tower_seg_count()) + tower_cap_t;
assert(tower_cap_t >= tower_spigot_len + 3 + 2 + nut_t + 6, "Cap roof too thin for the nut traps");

echo(str("camera_tower: segments=", tower_seg_count(), " x ", tower_seg_len(), " mm; cap top z=", tower_cap_top_z()));

if (tower_part == "base") tower_base_print();
else if (tower_part == "segment") tower_segment_print();
else if (tower_part == "cap") tower_cap();
else if (tower_part == "backing") tower_backing();
else assert(false, "tower_part must be base, segment, cap or backing");
