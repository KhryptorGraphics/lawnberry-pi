// Compute tower: the Libre Alta stack under the Pi 5 stack, held by TWO prints and no
// purchased fastener, standing on the chassis (chassis.scad) on three pedestals.
//
//   RACK: a shelf with four bosses at the Pi hole pattern, a back wall behind the microSD
//   end and two posts beside the boards' front corners. Each boss sits on a flexure tab cut
//   into the shelf (2.4 mm thick, 22 mm long): the tabs are the stack's spring, so the
//   clamp's 1.5 mm ratchet steps always end with the stack preloaded. Ratchet racks run up
//   the posts' outer faces and up two strips on the back wall. The shelf drops onto the
//   chassis pedestals' locating cones and two chassis snap fingers hook over its long edges.
//   CLAMP (one print): a ladder frame over the top board with four pads at the hole pattern
//   and four pawl fingers (two on the posts, two on the back wall). Push it down onto the
//   stack; it ratchets and holds. Release: spread the two post fingers and pull the two back
//   fingers forward, then lift.
//
// The boards form ONE column at each Pi-pattern hole: the owner's kit M-F standoffs screw to
// each other through every board (they are spacers, never anchors). The boss spigots (2.3 x
// 1.2, shear only) locate the Alta's holes; the clamp pads' 3.2 bores take the top standoff
// threads. Stack (bottom up): Alta, HaLow HAT, [standoffs], Pi 5 + Active Cooler, AI HAT+ on
// its supplied 16 mm stacking header, LC29H on ONE purchased 2x20 extra-tall stacking
// header, HaLow HAT on top. Spacings are ESTIMATES in parameters.scad (s_*): the ratchet
// takes 12 mm shorter / 3 mm taller total stack without a reprint.
//
// Frame (kept from the cleat frame it was designed in): back wall front face at
// y = cr_plate_front, board v along +x (ctw_v0_x at v = 0), board u along -y (ctw_u0_y at
// u = 0), shelf top at z = 0. ctw_to_body() turns it 180 degrees about Z into the body:
// board u -> body +Y, board v -> body -X.
//
// Printability: RACK prints upright on its shelf; CLAMP prints upside down (frame top on the
// bed, pads and fingers up). Ratchet catch faces are short horizontal overhangs (1.2/1.6);
// all other overhangs are 45 degrees.
include <cleat_carrier.scad>

function ctw_z_st() = -ctw_root_above_shelf;           // shelf top, local z
function ctw_z_sb() = ctw_z_st() - ctw_shelf_t;
function ctw_x0() = ctw_v0_x - ctw_post_gap - ctw_post_t;   // rack -v extreme (post face)
function ctw_x1() = ctw_v0_x + hatp_pcb[1] + ctw_post_gap + ctw_post_t;
ctw_stack_gap = 0.1;          // nominal: boards 0.1 above the bosses (probes push them down)
ctw_pad_gap = 0.2;            // nominal: pads 0.2 above the top board
ctw_tab_w = 7;
ctw_tab_slot = 1.0;
// Accessors for files that `use` this one (variables are not shared by `use`).
function ctw_stack_gap_mm() = ctw_stack_gap;
function ctw_pad_gap_mm() = ctw_pad_gap;
ctw_tab_t = 2.4;
ctw_tab_len = 22;             // root to boss centre; eps = 1.5*2.4*2.0/22^2 = 1.49 %
function ctw_tab_strain() = 1.5 * ctw_tab_t * 2.0 / pow(ctw_tab_len, 2);
assert(ctw_tab_strain() <= 0.015, "Tower boss flexure strain exceeds 1.5 %");

// Board frame -> cleat local. w is absolute local z.
module ctw_at_board(w = 0) {
    multmatrix([[0, 1, 0, ctw_v0_x], [-1, 0, 0, ctw_u0_y], [0, 0, 1, w], [0, 0, 0, 1]]) children();
}
function ctw_xy(u, v) = [ctw_v0_x + v, ctw_u0_y - u];
// Board-bottom z of each level (local), with the nominal stack gap.
function ctw_level_w(level) =
    ctw_z_st() + ctw_stack_gap + (
        level == "alta" ? ctw_alta_bot() :
        level == "hra" ? ctw_hra_bot() :
        level == "pi" ? ctw_pi_bot() :
        level == "ai" ? ctw_ai_bot() :
        level == "lc" ? ctw_lc_bot() :
        level == "hrp" ? ctw_hrp_bot() : 0);
// Clamp frame underside at travel dz (local z).
function ctw_clamp_z(dz = 0) = ctw_z_st() + ctw_stack_gap + ctw_clamp_bot(dz);
// Rack tooth undersides: one stop per 1.5 mm of clamp travel, 0.3 above the pawl's catch.
function ctw_tooth_zs() =
    [for (k = [0 : round((ctw_adjust + ctw_adjust_up) / rt_pitch)])
        ctw_clamp_z(-ctw_adjust + rt_pitch * k) - ctw_finger_tooth + 0.3];
function ctw_back_top() = max(ctw_tooth_zs()) + rt_h + 2.5;
// Local frame -> body: +90 degrees about Z (board u -> body +X, v -> body +Y).
module ctw_to_body(dx = 0, dz = 0) {
    multmatrix([[0, -1, 0, ctw_tx + ctw_u0_y + dx], [1, 0, 0, ctw_ty - ctw_v0_x],
                [0, 0, 1, ctw_shelf_z() + dz], [0, 0, 0, 1]]) children();
}
// Board (u, v, w) -> body point, w above the shelf top.
function ctw_body(u, v, w = 0) = [ctw_tx + u, ctw_ty + v, ctw_shelf_z() + w];

// ---- Rack --------------------------------------------------------------------------------
module ctw_shelf() {
    u1 = ctw_shelf_u[1];
    // 0.5 wider than the posts' outer faces so no shelf face is coplanar with a post face.
    v0 = -ctw_post_gap - ctw_post_t - 0.5;
    v1 = hatp_pcb[1] + ctw_post_gap + ctw_post_t + 0.5;
    ctw_at_board(ctw_z_sb()) difference() {
        union() {
            box3([ctw_shelf_u[0] - 0.01, v0, 0], [u1, v1, ctw_shelf_t]);
            // Toe past the back wall: its end face is the lock catch (chassis.scad).
            box3([ctw_toe_u, v0, 0], [ctw_shelf_u[0], ctw_toe_v_max, ctw_shelf_t]);
        }
        // Airflow / wiring window between the tabs.
        box3([ctw_window[0][0], ctw_window[0][1], -1], [ctw_window[1][0], ctw_window[1][1], ctw_shelf_t + 1]);
        // Keyholes over the chassis mushroom studs: the tower is lowered 6 mm -u of its seat
        // (entry over the head), then slid +u; the floor under each pocket is ctw_floor_t.
        for (st = ctw_studs) {
            hull() for (du = [0, ctw_slide]) translate([st[0] + du, st[1], -1])
                cylinder(d = ctw_key_w, h = ctw_shelf_t + 2);
            translate([st[0] + ctw_slide, st[1], -1]) cylinder(d = ctw_key_pocket_d, h = ctw_shelf_t + 2);
            hull() for (du = [0, ctw_slide]) translate([st[0] + du, st[1], ctw_floor_t])
                cylinder(d = ctw_key_pocket_d, h = ctw_shelf_t);
        }
        for (h = pi5_holes) {
            back = h[0] < 30;
            // The tab runs from its root toward the board edge, boss at the tip.
            tu0 = back ? h[0] - 4 : h[0] - ctw_tab_len;
            tu1 = back ? h[0] + ctw_tab_len : h[0] + 4;
            // Slot around three sides of the tab.
            difference() {
                box3([tu0 - (back ? ctw_tab_slot : 0), h[1] - ctw_tab_w / 2 - ctw_tab_slot, -1],
                     [tu1 + (back ? 0 : ctw_tab_slot), h[1] + ctw_tab_w / 2 + ctw_tab_slot, ctw_shelf_t + 1]);
                box3([tu0, h[1] - ctw_tab_w / 2, -2], [tu1, h[1] + ctw_tab_w / 2, ctw_shelf_t + 2]);
            }
            // Thin the tab to the flexure thickness (bottom stays on the bed).
            box3([tu0, h[1] - ctw_tab_w / 2 - 0.01, ctw_tab_t],
                 [tu1, h[1] + ctw_tab_w / 2 + 0.01, ctw_shelf_t + 1]);
        }
    }
    // Bosses on the tab tips, spigots into the Alta's holes.
    for (h = pi5_holes) ctw_at_board(ctw_z_sb()) translate([h[0], h[1], ctw_tab_t - 0.01]) {
        cylinder(d = ctw_boss_od, h = ctw_shelf_t - ctw_tab_t + ctw_boss_h + 0.01);
        translate([0, 0, ctw_shelf_t - ctw_tab_t + ctw_boss_h])
            cylinder(d = ctw_spigot_d, h = ctw_spigot_h);
    }
}

// Rack teeth as part of one 2D profile (outward distance n, z) so no tooth meets a face
// edge-on: the profile is extruded as a whole (post, or backing strip, plus teeth).
module ctw_teeth_2d() {
    for (z = ctw_tooth_zs()) polygon([[-0.01, z], [rt_h, z], [-0.01, z + rt_h + 0.01]]);
}

module ctw_posts() {
    yp0 = ctw_u0_y - (ctw_post_u[1] - 0.5);   // stops 0.5 short of the shelf front face
    yp1 = ctw_u0_y - ctw_post_u[0];
    z_top = ctw_z_st() + ctw_post_top();
    // -x post: profile in (x, z), teeth on the outer (-x) face.
    xz_prism(yp0, yp1) union() {
        translate([ctw_x0(), ctw_z_st() - 0.01]) square([ctw_post_t, z_top - ctw_z_st() + 0.01]);
        translate([ctw_x0(), 0]) mirror([1, 0]) ctw_teeth_2d();
    }
    xz_prism(yp0, yp1) union() {
        translate([ctw_x1() - ctw_post_t, ctw_z_st() - 0.01])
            square([ctw_post_t, z_top - ctw_z_st() + 0.01]);
        translate([ctw_x1(), 0]) ctw_teeth_2d();
    }
}

// Back racks: a backing strip proud of the plate by ctw_back_proud, teeth on it, extruded
// along x as one profile; the strip's back is buried 0.5 in the plate.
ctw_back_proud = 0.3;
function ctw_back_face() = cr_plate_front - ctw_back_proud;
module ctw_back_racks() {
    zs = ctw_tooth_zs();
    for (r = ctw_back_finger_x)
        yz_prism(r[0] + 0.5, r[1] - 0.5) union() {
            translate([ctw_back_face(), zs[0] - 1]) square([ctw_back_proud + 0.5, zs[len(zs) - 1] - zs[0] + 3]);
            translate([ctw_back_face(), 0]) mirror([1, 0]) ctw_teeth_2d();
        }
}

// Back wall behind the microSD end (u -11 .. -8): carries the two back ratchet strips.
module ctw_back_wall() {
    box3([ctw_x0(), cr_plate_front, ctw_z_sb()], [ctw_x1(), cr_plate_back, ctw_back_top()]);
}
module tower_rack() {
    ctw_back_wall();
    ctw_shelf();
    ctw_posts();
    ctw_back_racks();
}

// ---- Clamp -------------------------------------------------------------------------------
ctw_bar_w = 7;
ctw_back_u = [-4.7, 1];       // back crossbar overlaps the back fingers by 0.8 in y
ctw_front_u = ctw_post_u;
ctw_pad_od = ctw_boss_od;
ctw_pad_bore = 3.2;           // M2.5 standoff thread clearance
function ctw_finger_v() = [
    -ctw_post_gap - ctw_post_t - 0.4 - rt_tooth_out - rt_finger_t,     // -side body start
    hatp_pcb[1] + ctw_post_gap + ctw_post_t + 0.4 + rt_tooth_out];     // +side body start
function ctw_back_finger_y() =
    [ctw_back_face() - 0.4 - rt_tooth_out - rt_finger_t, ctw_back_face() - 0.4 - rt_tooth_out];

module tower_clamp(dz = 0) {
    zc = ctw_clamp_z(dz);
    zf = zc - ctw_finger_tooth;
    fv = ctw_finger_v();
    z_pad = ctw_level_w("hrp") + board_t + ctw_pad_gap + dz;
    ctw_at_board(zc) {
        // Ladder frame: two long bars over the hole rows, back and front crossbars.
        for (v = [0, hatp_pcb[1] - ctw_bar_w])
            box3([ctw_back_u[0], v, 0], [ctw_front_u[1], v + ctw_bar_w, ctw_clamp_t]);
        box3([ctw_back_u[0], 0, 0], [ctw_back_u[1], hatp_pcb[1], ctw_clamp_t]);
        // Front crossbar over the posts; it reaches out to the post fingers only across the
        // finger band, so nothing passes the shelf edges outside it.
        box3([ctw_front_u[0], 0, 0], [ctw_front_u[1], hatp_pcb[1], ctw_clamp_t]);
        box3([ctw_post_u[0] + 2, fv[0], 0], [ctw_post_u[1] - 2, fv[1] + rt_finger_t, ctw_clamp_t]);
    }
    // Pads with standoff-thread bores.
    for (h = pi5_holes) ctw_at_board(z_pad) translate([h[0], h[1], 0]) difference() {
        cylinder(d = ctw_pad_od, h = zc - z_pad + 0.01);
        translate([0, 0, -1]) cylinder(d = ctw_pad_bore, h = 7);
    }
    // Post fingers (flex in v), teeth toward the posts: each finger + tooth is one profile
    // in (v, z) extruded along u.
    ctw_at_board(0) for (s = [0, 1]) {
        v0 = fv[s];
        translate([ctw_post_u[0] + 2, 0, 0]) rotate([90, 0, 90])
            linear_extrude(height = ctw_post_u[1] - ctw_post_u[0] - 4) union() {
                translate([v0, zc - ctw_finger_len]) square([rt_finger_t, ctw_finger_len + 0.01]);
                if (s == 0)
                    polygon([[v0 + rt_finger_t - 0.01, zf], [v0 + rt_finger_t + rt_tooth_out, zf],
                             [v0 + rt_finger_t - 0.01, zf - rt_tooth_out]]);
                else
                    polygon([[v0 + 0.01, zf], [v0 - rt_tooth_out, zf], [v0 + 0.01, zf - rt_tooth_out]]);
            }
    }
    // Back fingers (flex in y), teeth toward the back plate racks; each rises 3 mm into the
    // back crossbar.
    by = ctw_back_finger_y();
    for (r = ctw_back_finger_x)
        yz_prism(r[0], r[1]) union() {
            translate([by[0], zc - ctw_finger_len]) square([rt_finger_t, ctw_finger_len + 3]);
            polygon([[by[1] - 0.01, zf], [by[1] + rt_tooth_out, zf], [by[1] - 0.01, zf - rt_tooth_out]]);
        }
}

// ---- Print poses -------------------------------------------------------------------------
module tower_rack_print() { translate([0, 0, -ctw_z_sb()]) tower_rack(); }
module tower_clamp_print() {
    zc = ctw_clamp_z(0);
    translate([0, 0, zc + ctw_clamp_t]) rotate([180, 0, 0]) tower_clamp(0);
}
