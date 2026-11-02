// INA3221 holder: a CAPACITY cradle that snaps onto the compute tower's clamp frame
// (compute_tower.scad), the board flat with its parts up, directly under the lid: terminal
// screws and plugs are reached from above with the lid off. The owner's module is not known;
// one print takes any board 22.0..32.0 across x 30..38.6 along, PCB 1.0..1.6 (parameters.scad,
// INA3221 section: Adafruit #6062, CJMCU-3221 and 30 x 26 breakouts fall inside).
//
//   FRAME (one body): two rails lying on the clamp's long bars, a fixed jaw across their -x
//   ends, a cross member near their +x ends. The jaw's ledge and lip take the board's near
//   long edge (1.2 mm each). Two snap leaves hang outside the long bars at the -x end and click
//   under their outer edges; a saddle (an inner tab against the front crossbar's -x face, an
//   outer tab on rail A past its +x face) locates the cradle along x.
//   BAR (print-in-place body): a beam between the rails with a pawl finger at each end that
//   ratchets in a groove along its rail (1.5 mm steps, the strap bay's tooth form), and a
//   cantilever leaf carrying the SHOE: ledge under and lip over the middle 8 mm of the board's
//   far long edge (0.8 mm reach). The leaf takes up the step with 1.0..2.5 mm preload. Every
//   gripping feature keeps >= 0.4 mm from a board's parts: lay the edge whose parts come
//   closest (the Adafruit's header edge) under the shoe, the terminal edge on the jaw. Push
//   the bar toward the jaw to clamp; release: push both pawl finger tips toward the board (pen
//   tip), pull the bar back. Remove the cradle: pull both snap leaves outward, lift its -x
//   end, then lift the saddle off the crossbar. The board touches plastic only, never metal.
//
// Frame of this file: body x/y, z above the clamp frame's top face (ina_at_clamp() places it
// at any clamp travel). Prints UPSIDE DOWN on its top plane (z = ina_z_top): lips, beam,
// fingers and rails start on the bed; the leaves and tabs rise from it; the ledges' faces and
// the snap catch faces are <= 1.6 mm overhangs, every other overhang is 45 degrees.
// Holding force, vibration retention and creep are NOT qualified.
include <cleat_carrier.scad>
use <compute_tower.scad>

assert(ctw_front_bar_u() == ctw_post_u, "INA cradle saddle assumes the clamp's front crossbar spans ctw_post_u");

function ina_x0() = ina_jaw_x - ina_jaw_t;                      // cradle -x end
function ina_xbar(d) = ina_jaw_x + d;                            // bar reference plane at stop d
// Pawl catch face (x, faces +x) at stop d; rack tooth k's catch face (faces -x) sits on it.
function ina_xf(d) = ina_xbar(d) + ina_beam_x0 - ina_finger_len + rt_tooth_out;
function ina_bar_out() = [ctw_ty, ctw_ty + hatp_pcb[1]];         // clamp long bars' outer faces
function ina_fo() = [ina_rail_y[0] + 0.4, ina_rail_y[1] - 0.4];   // fingers' outer faces
function ina_groove_x() = [ina_xf(ina_d_min) - rt_tooth_out - 0.8, ina_xf(ina_d_print) + 1.0];
function ina_catch_z() = -ctw_clamp_t - 0.4;                     // snap lip catch face
function ina_side_sign(i) = i == 0 ? -1 : 1;                      // rail A at -y, rail B at +y

// Groove in rail i's rack face, profile (y, z): 45-degree flanks, floor rt_tooth_out deep.
module ina_groove_2d(i) {
    s = ina_side_sign(i);
    f = ina_rail_y[i];
    z0 = ina_tooth_z0;
    th = ina_tooth_h;
    polygon([[f - s * 0.01, z0 - 0.8], [f + s * rt_tooth_out, z0 + 0.8],
             [f + s * rt_tooth_out, z0 + th - 0.8], [f - s * 0.01, z0 + th + 0.8]]);
}
// Rack teeth on the groove floor: catch faces at the pawl catch face of every stop.
module ina_rack(i) {
    s = ina_side_sign(i);
    yg = ina_rail_y[i] + s * rt_tooth_out;
    g = ina_groove_x();
    intersection() {
        yz_prism(g[0], g[1]) ina_groove_2d(i);
        translate([0, 0, ina_tooth_z0 - 1]) linear_extrude(height = ina_tooth_h + 2)
            for (k = [0 : ina_steps]) let(x = ina_xf(ina_d_k(k)))
                polygon([[x, yg + s * 0.01], [x, yg - s * rt_h], [x + rt_h, yg + s * 0.01]]);
    }
}

// ---- Frame -----------------------------------------------------------------------------------
module ina_snap_leaf(i) {
    s = ina_side_sign(i);
    bo = ina_bar_out()[i];
    yi = bo + s * 0.4;                         // leaf face toward the clamp bar
    yo = yi + s * ina_hook_t;
    x0 = ina_x0();
    zc = ina_catch_z();
    zb = zc - (0.4 + ina_hook_reach);
    box3([x0, min(yi, yo), zb], [x0 + ina_hook_len, max(yi, yo), ina_z_top]);
    // Root block back to the rail (the leaf is free beyond it).
    rail_out = ina_rail_y[i] + s * ina_rail_t;
    box3([x0, min(yo, rail_out - s * 0.01), 0], [x0 + 2, max(yo, rail_out - s * 0.01), ina_z_top]);
    // Lip under the long bar's outer edge: flat catch face up, 45-degree lead-in below.
    yz_prism(x0 + ina_hook_len - 4, x0 + ina_hook_len) polygon([
        [yi + s * 0.01, zc], [bo - s * ina_hook_reach, zc], [yi + s * 0.01, zb]]);
}

module ina_frame() {
    x0 = ina_x0();
    z1 = ina_z_top;
    ya = ina_rail_y[0];
    yb = ina_rail_y[1];
    hl = ina_along_range[1] / 2 + 0.35;
    difference() {
        union() {
            box3([x0, ya - ina_rail_t, 0], [ina_tab_x[1], ya, z1]);        // rail A (to the outer tab)
            box3([x0, yb, 0], [ina_cross_x[1], yb + ina_rail_t, z1]);      // rail B
        }
        for (i = [0, 1]) yz_prism(ina_groove_x()[0], ina_groove_x()[1]) ina_groove_2d(i);
    }
    for (i = [0, 1]) ina_rack(i);
    // Fixed jaw with its ledge and lip over the board's near long edge.
    box3([x0, ya - 0.01, 0], [ina_jaw_x, yb + 0.01, z1]);
    box3([ina_jaw_x - 0.01, ina_yc - hl, ina_ledge_z0], [ina_jaw_x + ina_lip, ina_yc + hl, ina_z_board]);
    box3([ina_jaw_x - 0.01, ina_yc - hl, ina_lip_z0], [ina_jaw_x + ina_lip, ina_yc + hl, z1]);
    // Cross member and the saddle: inner tab against the front crossbar's -x face, outer tab
    // (rail A only) past its +x face.
    box3([ina_cross_x[0], ya - 0.01, 0], [ina_cross_x[1], yb + 0.01, z1]);
    box3([ina_cross_x[0], ina_yc - 15, -ina_tab_drop], [ina_cross_x[1], ina_yc + 15, 0.01]);
    box3([ina_tab_x[0], ya - ina_rail_t, -ina_tab_drop], [ina_tab_x[1], ya, 0.01]);
    for (i = [0, 1]) ina_snap_leaf(i);
}

// ---- Bar (print-in-place) ---------------------------------------------------------------------
module ina_pawl(i, d) {
    s = ina_side_sign(i);
    fo = ina_fo()[i];
    xf = ina_xf(d);
    z0 = ina_tooth_z0;
    th = ina_tooth_h;
    out = rt_tooth_out;
    intersection() {
        translate([0, 0, z0]) linear_extrude(height = th)
            polygon([[xf, fo - s * 0.01], [xf, fo + s * out], [xf - out, fo - s * 0.01]]);
        yz_prism(xf - out - 1, xf + 1) polygon([
            [fo - s * 0.01, z0], [fo + s * out, z0 + out],
            [fo + s * out, z0 + th - out], [fo - s * 0.01, z0 + th]]);
    }
}

module ina_bar(d = ina_d_print) {
    xb = ina_xbar(d);
    fo = ina_fo();
    zb = ina_bar_z0;
    z1 = ina_z_top;
    xbk = xb + ina_beam_x0;
    box3([xbk, fo[0], zb], [xbk + ina_beam_t, fo[1], z1]);
    // Pawl fingers along the rails, flexing in y, teeth outward into the grooves.
    box3([xbk - ina_finger_len, fo[0], zb], [xbk + 0.01, fo[0] + rt_finger_t, z1]);
    box3([xbk - ina_finger_len, fo[1] - rt_finger_t, zb], [xbk + 0.01, fo[1], z1]);
    for (i = [0, 1]) ina_pawl(i, d);
    // Cantilever leaf (root toward -y) and the shoe on its free end.
    yl0 = ina_yc - ina_leaf_len;
    ys = [ina_yc - ina_shoe_len / 2, ina_yc + ina_shoe_len / 2];
    box3([xb + ina_shoe_x[1], yl0 - 2, zb], [xbk + 0.01, yl0, z1]);
    box3([xb + ina_shoe_x[1], yl0 - 0.01, zb], [xb + ina_shoe_x[1] + ina_leaf_t, ys[1], z1]);
    box3([xb + ina_shoe_x[0], ys[0], zb], [xb + ina_shoe_x[1] + 0.01, ys[1], z1]);
    xc = xb + ina_shoe_x[0];
    box3([xc - ina_shoe_lip, ys[0], ina_ledge_z0], [xc + 0.01, ys[1], ina_z_board]);
    box3([xc - ina_shoe_lip, ys[0], ina_lip_z0], [xc + 0.01, ys[1], z1]);
}

// part: "frame" | "bar" | "assembly"; d = bar stop (ina_d_k) or the printed ina_d_print.
module ina_cradle(part = "assembly", d = ina_d_print) {
    if (part == "frame") ina_frame();
    else if (part == "bar") ina_bar(d);
    else { ina_frame(); ina_bar(d); }
}
// Cradle frame -> body at clamp travel `travel`, with the tower moved by (dx, dz).
module ina_at_clamp(travel = 0, dx = 0, dz = 0) {
    translate([dx, 0, ctw_shelf_z() + ctw_clamp_top_w(travel) + dz]) children();
}
// A board [across, along] resting 0.05 off the jaw face and the ledges (its seat faces).
function ina_board_origin(b) = [ina_jaw_x + 0.05, ina_yc - b[1] / 2, ina_z_board + 0.05];
// Bar position whose shoe face just meets board b (the leaf deflected to the board edge).
function ina_d_touch(b) = b[0] + ina_bulge + 0.1;

module ina_cradle_print() { translate([0, 0, ina_z_top]) rotate([180, 0, 0]) ina_cradle("assembly", ina_d_print); }
