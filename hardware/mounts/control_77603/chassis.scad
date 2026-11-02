// Chassis: the removable, bolt-free internal frame of the electronics enclosure, and its
// base. Body coordinates of the unchanged legacy enclosure (../enclosure_body.scad).
//
//   BASE (one print, chassis_base): a U-frame - two rails over the four floor bosses and a
//   +Y crossbar - clamped onto the boss tops by the four EXISTING box-to-arm-layer M4 bolts
//   (it replaces the legacy tray at the same 3.5 mm thickness, so the bolts keep their grip
//   length). Four hooks stand on the rails, each with a 45-degree lip pointing +Y; a finger
//   in the +X rail carries the detent nub. The -Y side stays open over the floor glands.
//   CHASSIS (one print): a 4 mm plate on the base plus three panels. Lower it 6 mm +Y of its
//   seat (the hooks rise through its windows), push it -Y: the lips ride onto the chamfered
//   window edges and hold it down, and the detent nub pops up into the plate's detent hole.
//   Release: press the nub down through that hole (pen tip) and push the chassis +Y, lift.
//   The panels carry molded French-cleat strips (../electronics_rail.scad) for the strap
//   bays and the harness clamp, clips for the PCA9685 x2 and the Pololu #2806, and the
//   tower seat: two mushroom studs, two pads and the tower-lock finger.
//
// Printability: the base prints flat (hooks up, 45-degree lip undersides); the chassis
// prints plate-down (panels vertical, strips' only flat overhang is the 2 mm catch land,
// mushroom heads are 45-degree cones). Strength, stiffness and vibration are NOT qualified.
include <cleat_carrier.scad>
use <compute_tower.scad>

module ch_outline_2d() {
    r = ch_corner_r;
    hull() for (x = [ch_x[0] + r, ch_x[1] - r], y = [ch_y[0] + r, ch_y[1] - r])
        translate([x, y]) circle(r = r);
}
module ch_outline(z0, z1) { translate([0, 0, z0]) linear_extrude(height = z1 - z0) ch_outline_2d(); }

// Cleat local frame (wall face y = 0, interior y < 0, strip along +x, z = 0 at root) on a
// chassis panel face. `start` is the body coordinate of local x = 0 along the face.
// Exact matrices (no trigonometric round-off, so coplanar faces stay coplanar).
function face_matrix(face, start, root) =
    face == "-Y" ? [[-1, 0, 0, start], [0, -1, 0, ch_face_y], [0, 0, 1, root], [0, 0, 0, 1]] :
    face == "-X" ? [[0, -1, 0, -ch_face_x], [1, 0, 0, start], [0, 0, 1, root], [0, 0, 0, 1]] :
    face == "+X" ? [[0, 1, 0, ch_face_x], [-1, 0, 0, start], [0, 0, 1, root], [0, 0, 0, 1]] :
    undef;
module at_face(face, start, root) {
    assert(!is_undef(face_matrix(face, start, root)), str("unknown face ", face));
    multmatrix(face_matrix(face, start, root)) children();
}
module at_strip(i) {
    s = ch_strips[i];
    at_face(s[0], s[2], s[1]) children();
}

// ---- Base -------------------------------------------------------------------------------
function cb_z0() = tray_bottom_z;
function cb_z1() = tray_bottom_z + cb_t;
module cb_hook(h) {
    yp = h[1] + cb_hook_post[1] / 2;                 // post +Y face
    ye = yp + ch_hook_cl;                            // plate window edge, seated
    zu0 = ch_top() - ch_edge_chamfer - ch_hook_cl + ch_hook_cl * sqrt(2);
    ztop = ch_top() + cb_hook_top;
    box3([h[0] - cb_hook_post[0] / 2, h[1] - cb_hook_post[1] / 2, cb_z1() - 0.01],
         [h[0] + cb_hook_post[0] / 2, yp, ztop]);
    // Lip over the plate's chamfered window edge: 45-degree underside 0.2 off the chamfer.
    yz_prism(h[0] - cb_hook_post[0] / 2, h[0] + cb_hook_post[0] / 2) polygon([
        [yp - 0.01, zu0], [yp + cb_hook_lip, zu0 + cb_hook_lip],
        [yp + cb_hook_lip, ztop], [yp - 0.01, ztop]]);
}
// Base detent finger region in the +X rail.
function cb_det_x() = [cb_det_xy[0] - cb_det_w / 2, cb_det_xy[0] + cb_det_w / 2];
function cb_det_y1() = cb_det_xy[1] + 3;            // free end
// Window in the -X rail under the chassis tower-lock finger.
function ch_lock_x() = [ctw_body(ctw_toe_u, 0)[0] - 0.4 - ch_lock_w, ctw_body(ctw_toe_u, 0)[0] - 0.4];
module chassis_base(nub = true) {
    z0 = cb_z0();
    z1 = cb_z1();
    dx = cb_det_x();
    lx = ch_lock_x();
    difference() {
        union() {
            for (s = [-1, 1]) {
                x0 = s < 0 ? -cb_rail_x[1] : cb_rail_x[0];
                x1 = s < 0 ? -cb_rail_x[0] : cb_rail_x[1];
                box3([x0, cb_y[0], z0], [x1, cb_y[1], z1]);
            }
            box3([-cb_rail_x[0] - 0.01, cb_bar_y[0], z0], [cb_rail_x[0] + 0.01, cb_bar_y[1], z1]);
        }
        for (p = tray_fixings) translate([p[0], p[1], z0 - 1]) cylinder(d = tray_fixing_d, h = cb_t + 2);
        // Detent finger: slot round two sides and the free end, thinned from the top.
        box3([dx[0] - 1, cb_det_root_y, z0 - 1], [dx[0], cb_det_y1() + 1, z1 + 1]);
        box3([dx[1], cb_det_root_y, z0 - 1], [dx[1] + 1, cb_det_y1() + 1, z1 + 1]);
        box3([dx[0] - 1, cb_det_y1(), z0 - 1], [dx[1] + 1, cb_det_y1() + 1, z1 + 1]);
        box3([dx[0] - 0.01, cb_det_root_y + 1, z0 + cb_det_t], [dx[1] + 0.01, cb_det_y1() + 0.01, z1 + 1]);
        // Window under the chassis tower-lock finger (it flexes down).
        box3([-cb_rail_x[1] - 1, ch_lock_y[0] + 4, z0 - 1], [lx[1] + 0.5, ch_lock_y[1] + 1, z1 + 1]);
    }
    for (h = cb_hooks) cb_hook(h);
    if (nub) translate([cb_det_xy[0], cb_det_xy[1], z0 + cb_det_t - 0.01]) {
        hn = ch_seat_z + cb_det_engage - (z0 + cb_det_t);
        cylinder(d = cb_det_nub_d, h = hn - 0.5 + 0.01);
        translate([0, 0, hn - 0.5]) cylinder(d1 = cb_det_nub_d, d2 = cb_det_nub_d - 1, h = 0.5);
    }
}

// ---- Chassis plate ------------------------------------------------------------------------
module ch_plate() {
    z0 = ch_seat_z;
    z1 = ch_top();
    lx = ch_lock_x();
    difference() {
        ch_outline(z0, z1);
        // Hook windows: the lowered hook (post + lip, 6 mm -Y in plate terms) drops through,
        // then the plate slides -Y and the window's chamfered +Y edge goes under the lip.
        for (h = cb_hooks) {
            yp = h[1] + cb_hook_post[1] / 2;
            ye = yp + ch_hook_cl;
            box3([h[0] - cb_hook_post[0] / 2 - 0.4, yp - cb_hook_post[1] - ch_slide - ch_hook_cl, z0 - 1],
                 [h[0] + cb_hook_post[0] / 2 + 0.4, ye, z1 + 1]);
            yz_prism(h[0] - cb_hook_post[0] / 2 - 0.4, h[0] + cb_hook_post[0] / 2 + 0.4) polygon([
                [ye - 0.01, z1 - ch_edge_chamfer], [ye + ch_edge_chamfer + 0.01, z1 + 0.01],
                [ye - 0.01, z1 + 0.01]]);
        }
        // Bolt heads (washer + head envelope), slotted for the slide.
        for (p = tray_fixings) hull() for (dy = [0, -ch_slide])
            translate([p[0], p[1] + dy, z0 - 1]) cylinder(d = ch_head_hole_d, h = ch_plate_t + 2);
        // Detent hole: the base nub pops up into it at the seat.
        translate([cb_det_xy[0], cb_det_xy[1], z0 - 1]) cylinder(d = ch_det_hole_d, h = ch_plate_t + 2);
        // Gland cable passages, slotted for the slide.
        for (g = gland_centres) hull() for (dy = [0, -ch_slide])
            translate([g[0], g[1] + dy, z0 - 1]) cylinder(d = ch_gland_hole_d, h = ch_plate_t + 2);
        // Breather fitting nut (+X wall, y 28, z 30): plate edge set back, slotted for the slide.
        box3([ch_vent_plate_x, 28 - 10.5 - ch_slide, z0 - 1], [ch_x[1] + 1, 28 + 10.5, z1 + 1]);
        // Tower-lock finger: slot round two sides and the free end, thinned from the top.
        box3([lx[0] - ch_lock_slot, ch_lock_y[0], z0 - 1], [lx[0], ch_lock_y[1] + ch_lock_slot, z1 + 1]);
        box3([lx[1], ch_lock_y[0], z0 - 1], [lx[1] + ch_lock_slot, ch_lock_y[1] + ch_lock_slot, z1 + 1]);
        box3([lx[0] - ch_lock_slot, ch_lock_y[1], z0 - 1], [lx[1] + ch_lock_slot, ch_lock_y[1] + ch_lock_slot, z1 + 1]);
        box3([lx[0] - 0.01, ch_lock_y[0] + 1, z0 + ch_lock_t], [lx[1] + 0.01, ch_lock_y[1] + 0.01, z1 + 1]);
    }
}

// ---- Tower seat ----------------------------------------------------------------------------
function ctw_shelf_bot_z() = ch_top() + ctw_pedestal_h;
module ch_tower_seat() {
    zb = ctw_shelf_bot_z();
    for (st = ctw_studs) translate([ctw_body(st[0], st[1])[0], ctw_body(st[0], st[1])[1], 0]) {
        translate([0, 0, ch_top() - 0.01]) cylinder(d = ctw_ped_d, h = ctw_pedestal_h + 0.01);
        zn = zb + ctw_floor_t + 0.4;
        translate([0, 0, zb - 0.01]) cylinder(d = ctw_stud_neck_d, h = zn - zb + 0.02);
        hc = (ctw_stud_head_d - ctw_stud_neck_d) / 2;
        translate([0, 0, zn]) cylinder(d1 = ctw_stud_neck_d, d2 = ctw_stud_head_d, h = hc);
        translate([0, 0, zn + hc - 0.01]) cylinder(d = ctw_stud_head_d, h = zb + ctw_shelf_t - zn - hc + 0.01);
    }
    for (pd = ctw_pads) translate([ctw_body(pd[0], pd[1])[0], ctw_body(pd[0], pd[1])[1], ch_top() - 0.01])
        cylinder(d = ctw_pad_d, h = ctw_pedestal_h + 0.01);
    // Lock nub on the finger's free end; catch face 0.4 behind the toe's end face.
    lx = ch_lock_x();
    box3([lx[0], ch_lock_y[1] - ch_lock_nub_len, ch_seat_z + ch_lock_t - 0.01],
         [lx[1], ch_lock_y[1], zb + ch_lock_engage]);
}

// ---- Panels, strips, board clips ---------------------------------------------------------------
module ch_panels() {
    z0 = ch_top() - 0.01;
    intersection() {
        ch_outline(z0, ch_ceiling);
        union() {
            difference() {
                box3([ch_x[0], ch_y[0], z0], [ch_x[1], ch_face_y, ch_my_top]);
                // Over the floor-gland keep-outs (locknut + cable bend to z 30.5).
                for (g = gland_centres)
                    box3([g[0] - ch_gland_hole_d / 2 - 0.5, ch_y[0] - 1, z0 - 1],
                         [g[0] + ch_gland_hole_d / 2 + 0.5, ch_face_y + 1, ch_gland_notch_z]);
                // Over the -Y bolt heads while the chassis is lowered 6 mm +Y of its seat.
                for (p = tray_fixings) if (p[1] < 0)
                    box3([p[0] - ch_head_hole_d / 2 - 0.5, ch_y[0] - 1, z0 - 1],
                         [p[0] + ch_head_hole_d / 2 + 0.5, ch_face_y + 1, cb_z1() + bolt_head_h + 0.7]);
            }
            box3([ch_x[0], ch_side_y[0], z0], [-ch_face_x, ch_side_y[1], ch_mx_top]);
            difference() {
                box3([ch_face_x, ch_side_y[0], z0], [ch_x[1], ch_side_y[1], ch_px_top]);
                // Breather fitting, slotted for the slide.
                box3([ch_face_x - 1, ch_vent_cut[0][0], z0 - 1], [ch_x[1] + 1, ch_vent_cut[1][0], ch_vent_cut[1][1]]);
            }
        }
    }
}
// Strips are clipped to the plate outline (a strip end at a rounded corner would otherwise
// leave a sliver outside the panel).
module ch_strip_set() {
    intersection() {
        ch_outline(ch_top() - 0.01, ch_ceiling);
        union() for (i = [0 : len(ch_strips) - 1]) at_strip(i) wall_cleat(ch_strips[i][3], 1);
    }
}

// Board in the plane parallel to the plate: x [bx, bx+bw], z [bz, bz+bh], back face at
// sig_board_back(), thickness bt.
function sig_board_back() = cr_plate_front - sig_standoff;
module board_holder(bx, bz, bw, bh, bt) {
    cl = sig_cl;
    lip = sig_lip;
    yb = sig_board_back();
    yf = yb - bt;
    ylf = yf - cl - 1.2;
    yhf = yf - cl - 3.0;
    zt = bz + bh;
    for (m = [0, 1]) translate([m * (2 * bx + bw), 0, 0]) mirror([m, 0, 0]) {
        // Bottom pocket: floor, side wall, standoff, L-shaped lip over the corner.
        difference() {
            box3([bx - cl - 2, ylf, bz - cl - 2], [bx + 5, cr_plate_front + 0.01, bz + 5]);
            box3([bx - cl, yf - cl, bz - cl], [bx + 6, yb + cl, bz + 6]);
            box3([bx + lip, ylf - 1, bz + lip], [bx + 6, yf - cl + 0.01, bz + 6]);
        }
        yz_prism(bx - cl - 2, bx + 5) polygon([
            [cr_plate_front + 0.01, bz - cl - 2 + 0.01], [ylf, bz - cl - 2 + 0.01],
            [cr_plate_front + 0.01, bz - cl - 2 - (cr_plate_front - ylf)]]);
        // Top clip: finger outside the pocket wall, flexing in x.
        xf1 = bx - cl - 2 - 0.6;
        xf0 = xf1 - rt_finger_t;
        z_fb = zt - 30;
        box3([xf0, yhf, z_fb - 4], [xf1, cr_plate_front + 0.01, z_fb]);
        yz_prism(xf0, xf1) polygon([
            [cr_plate_front + 0.01, z_fb - 4 + 0.01], [yhf, z_fb - 4 + 0.01],
            [cr_plate_front + 0.01, z_fb - 4 - (cr_plate_front - yhf)]]);
        box3([xf0, yhf, z_fb - 0.01], [xf1, cr_plate_front - 0.6, zt + cl + 3.6]);
        // Hook arm over the board front at the corner: 45-degree lead-in in plan, 45-degree
        // underside in elevation.
        intersection() {
            translate([0, 0, zt - 10]) linear_extrude(height = 10 + cl + 1) polygon([
                [xf0 + 0.05, yhf + 0.05], [bx - cl, yhf + 0.05], [bx + lip, yhf + lip + cl],
                [bx + lip, yf - cl], [xf0 + 0.05, yf - cl]]);
            xz_prism(yhf - 1, yf) polygon([
                [xf0 + 0.05, zt - 9.4], [xf1, zt - 9.4], [bx + lip, zt - 9.4 + (bx + lip - xf1)],
                [bx + lip, zt + cl + 1], [xf0 + 0.05, zt + cl + 1]]);
        }
        // Ceiling over the board's top corner (kept 0.05 inside the finger's faces).
        // Front and left faces offset from the hook arm's so no two faces are coplanar.
        xz_prism(yhf + 0.15, cr_plate_front - 0.65) polygon([
            [xf0 + 0.1, zt + cl], [bx - cl, zt + cl], [bx + lip, zt + cl + lip + cl],
            [bx + lip, zt + cl + 3.55], [xf0 + 0.1, zt + cl + 3.55]]);
    }
}

module ch_board_clips() {
    for (b = sig_boards)
        at_face(b[1], b[2], 0) translate([0, -cr_plate_front, 0])
            board_holder(b[3], b[4], b[5][0], b[5][1], b[6]);
}

// ---- Whole chassis -----------------------------------------------------------------------------
// Seated pose; (dy, dz) offsets it (dy = +ch_slide is the lowered pose).
module chassis(dy = 0, dz = 0, lock_nub = true) {
    translate([0, dy, dz]) {
        ch_plate();
        ch_panels();
        ch_strip_set();
        ch_board_clips();
        if (lock_nub) ch_tower_seat();
        else difference() {
            ch_tower_seat();
            box3([ch_lock_x()[0] - 1, ch_lock_y[0], ch_seat_z], [ch_lock_x()[1] + 1, ch_lock_y[1] + 1, 40]);
        }
    }
}

module chassis_print() { translate([0, 0, -ch_seat_z]) chassis(); }
module chassis_base_print() { translate([0, 0, -cb_z0()]) chassis_base(); }
