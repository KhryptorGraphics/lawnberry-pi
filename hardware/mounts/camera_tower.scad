// Hollow ASA camera mast with bottom-loading enclosed cameras.
// Pi Camera 2 is lower; a 304.8 mm tube ABOVE its housing carries the stereo housing.
// camera_stack.scad adds the housings/trays/cameras to this structural column.
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
//   extension  304.8 mm flange-to-flange tube BETWEEN the two camera housings.
//   cap        Closed roof over the upper housing's rear cable spine.
//   backing   Drilling template for a metal strip INSIDE the box behind the wall
//             flange: spreads the six M5 loads over the 4 mm ASA wall.
//
// The taller two-camera stack has NOT been load/vibration qualified. Previous
// single-camera mass/moment/FOS estimates do not apply. Print tube axes horizontal,
// use metal washers, inspect joints, and bench-test the mast before mower use.
// tower_height specifies the lower Pi lens centre, not the cap or the upper lens.
include <enclosure_common.scad>
tower_part = "base";               // base | segment | extension | cap | backing

tower_bore = tower_od - 2 * tower_wall;
spigot_od = tower_bore - 2 * tower_spigot_clear;
spigot_wall = 3.5;
wall_y = box_outer_w / 2;          // box wall outer face
foot_top_z = tongue_t + tower_base_t;
nut_af = 7.2;                      // M4 nut across flats + allowance
nut_t = 3.4;

// ---- sway-bar brace stations: the bar collars clamp around the column, and their M4 screws must
// pass through the COMPLETE square tube - both walls - so collar pair and column become one bolted
// joint. Stations are z = [110, 190, 250]; the base reaches tower_base_top_z, so the stations above
// it fall inside the FIRST segment, in that segment's local frame. Both must fit there or the shared
// segment part cannot carry them, hence the assert. Through-holes in the bore are tolerable because
// the base already drains its sump out the +Y face.
function tower_brace_seg_z() =
    [for (z = arm_brace_z) if (z > tower_base_top_z) z - tower_base_top_z];
assert(len(tower_brace_seg_z()) == 0
       || max(tower_brace_seg_z()) < tower_seg_len() - tower_flange_t,
       "A brace station falls outside the first segment; the shared segment cannot carry its bores");
// zs are in the caller's frame; y0 is the column's y offset there (tower_y absolute, 0 local).
module tower_brace_bores(zs, y0 = 0) {
    for (z = zs, y = [-arm_brace_collar_bolt_y, arm_brace_collar_bolt_y])
        translate([-tower_od / 2 - 1, y0 + y, z]) rotate([0, 90, 0])
            cylinder(d = m4_clear, h = tower_od + 2);
}
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
// inner size (32.2 mm) for those 2 mm; checked against the connector envelope.
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
            // Closed connector duct between box wall and tube; the upper relief lets a rigid
            // plug turn ABOVE the brace bolts rather than colliding with their cross-shafts.
            translate([-18,wall_y+tower_base_t-0.1,tower_port_z-tower_port_h/2-3])
                cube([36,tower_y-18-(wall_y+tower_base_t)+0.1,77]);
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
        translate([-tower_port_w/2,wall_y-1,tower_port_z-tower_port_h/2])
            cube([tower_port_w,tower_y-wall_y+2,tower_port_h]);
        translate([-tower_port_w/2,wall_y+tower_base_t,tower_port_z-tower_port_h/2])
            cube([tower_port_w,tower_y-wall_y-tower_base_t+2,
                  column_top+1.7-(tower_port_z-tower_port_h/2)]);
        // sump drain out the +Y face, 4 mm above the bore floor
        translate([0, tower_y, foot_top_z + 0.9 + 4]) rotate([-90, 0, 0]) cylinder(d = 5, h = tower_od);
        // Weep vent for the former open web gap now covered by the connector duct.
        // Open this otherwise sealed print cavity; it is separate from the cable passage.
        translate([0,wall_y+tower_base_t+web_len/2,foot_top_z+4])
            rotate([0,90,0]) cylinder(d=5,h=tower_od/2+2);
        for (p = tower_base_bolts) translate([p[0], wall_y - 1, p[1]]) rotate([-90, 0, 0])
            cylinder(d = tower_wall_bolt_d, h = tower_base_t + 2);
        for (p = tower_foot_bolts) translate([p[0], p[1], tongue_t - 1]) cylinder(d = tower_wall_bolt_d, h = tower_base_t + 2);
        // Brace station 0 sits on the base: screws through the complete tube, both walls.
        tower_brace_bores([for (z = arm_brace_z) if (z <= tower_base_top_z) z], tower_y);
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
    difference() {
        union() {
            tower_joint_flange();
            tower_tube(len);
            translate([0, 0, len - tower_flange_t]) tower_joint_flange();
            translate([0, 0, len]) tower_spigot();
        }
        // Both column brace stations fall in the first segment, drilled here in local coordinates.
        tower_brace_bores(tower_brace_seg_z(), 0);
    }
}
// Short piece for exploded views only; not a printable part.
module tower_segment_stub() { tower_segment(60); }
module tower_segment_print() {
    translate([0, 0, tower_flange / 2]) rotate([0, 90, 0]) tower_segment();
}

// Separate extension: no redundant brace holes in the exposed upper mast.
module tower_camera_extension() {
    tower_joint_flange();
    tower_tube(camera_extension_mm);
    translate([0,0,camera_extension_mm-tower_flange_t]) tower_joint_flange();
    translate([0,0,camera_extension_mm]) tower_spigot();
}
module tower_camera_extension_print() {
    translate([0,0,tower_flange/2]) rotate([0,90,0]) tower_camera_extension();
}

// ---- cap, local: flange underside at z=0 ----
module tower_cap() {
    cav_h = tower_spigot_len + 3;
    difference() {
        translate([-tower_flange / 2, -tower_flange / 2, 0]) cube([tower_flange, tower_flange, tower_cap_t]);
        // socket for the top spigot, with headroom
        translate([-tower_bore / 2, -tower_bore / 2, -1]) cube([tower_bore, tower_bore, cav_h + 1]);
        for (x = [-1, 1], y = [-1, 1])
            translate([x * tower_flange_bolt, y * tower_flange_bolt, -1])
                cylinder(d = m4_clear, h = tower_cap_t + 2);
    }
}

// ---- internal backing strip template: wall bolts + port, sits inside on the +Y wall ----
module tower_backing() {
    difference() {
        plate(tower_base_w, 130, 3, 6);
        for (p = tower_base_bolts) translate([p[0], p[1] - 90, -1]) cylinder(d = tower_wall_bolt_d, h = 5);
        translate([-tower_port_w/2,tower_port_z-90-tower_port_h/2,-1])
            cube([tower_port_w,tower_port_h,5]);
    }
}

// ---- structural column; camera_stack.scad inserts the two housings in the gaps ----
module camera_tower_assembly() {
    tower_base_assembly();
    for (i = [0 : tower_seg_count() - 1])
        translate([0, tower_y, tower_flange_z(i)]) tower_segment();
    translate([0,tower_y,camera_lower_z()+camera_housing_h]) tower_camera_extension();
    translate([0,tower_y,tower_upper_cap_z()]) tower_cap();
}
function tower_cap_top_z() = tower_upper_cap_z() + tower_cap_t;
assert(tower_cap_t >= tower_spigot_len + 3 + 6, "Closed cap roof must be at least 6 mm thick");
assert(camera_extension_mm + tower_spigot_len <= 420, "Camera extension exceeds the printer bed");

echo(str("camera_tower: segments=", tower_seg_count(), " x ", tower_seg_len(), " mm; cap top z=", tower_cap_top_z()));

if (tower_part == "base") tower_base_print();
else if (tower_part == "segment") tower_segment_print();
else if (tower_part == "extension") tower_camera_extension_print();
else if (tower_part == "cap") tower_cap();
else if (tower_part == "backing") tower_backing();
else assert(false, "tower_part must be base, segment, extension, cap or backing");
