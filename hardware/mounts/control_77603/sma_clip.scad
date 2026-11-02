// SMA service clip: ONE print, no fasteners, riding the compute tower's clamp frame
// (compute_tower.scad) at its front-right corner. It holds the three tower-side SMA-female D-flat
// bulkhead jacks face-up (owner decision 2026-11-02): [0] Pi HaLow, [1] GNSS, [2] Alta HaLow
// (sc_joints, parameters.scad radio section). Each jack is the end of a U.FL -> SMA-F lead that
// stays mated to its board; the lid's cables end in SMA-male right-angle plugs screwed onto these
// jacks. To open the lid: lift it ~100 mm, unscrew the three plugs with an 8 mm spanner.
//
//   SEAT: the north foot stands on the front crossbar (x 15.5..24) and the +Y finger band, the
//   south bridge on the -Y finger band (x 17.5..22). JAWS: a pair straddles each finger band
//   (inner faces sc_jaw_cl off its sides: X location); the north pair's -Y face sits sc_jaw_cl off
//   the frame's +Y face, the south pair's +Y face off its -Y face (Y location). Two pairs 62 mm
//   apart key the clip against the plugs' mating torque (7-10 in-lb, Amphenol RF): each jack's
//   D-flat keys it in the clip, the clip's jaws key it on the frame. SNAP: a leaf hangs down the
//   crossbar's +X face and its lip hooks under the crossbar (pull the release tab +X 1.4 mm,
//   lift). PLATE: 5 mm, carried by the north column and the south web; each jack's body hex sits
//   in a counterbore under a gn_land_t land, a lock washer + nut on the flat top (the lid
//   GNSS port's land and D hole, antenna_mount sma_land_cut()).
//   Off limits (agreed with the INA3221 cradle, 2026-11-02): x <= 15.1 at the crossbar (cross
//   member, inner tab), the cradle's lift path over rail A (x 14.7..26.6, y -6.75..-2.45 above
//   Zc+8.4), its I2C lead's envelope (ina_i2c_run) and the finger rooms; the clip stays below
//   Zc+33.6, under the INA lead's +X run at Zc+35..40 between the Pi and GNSS jacks.
//
// Frame: body x/y, z above the clamp frame's top face Zc (sc_at_clamp() places it at any clamp
// travel). Prints UPSIDE DOWN on its plate top (z = sc_plate_z[1]): the column, web and leaf rise
// from the plate; the foot and bridges cap them (the south bridge on a 45-degree gusset), the jaws
// rise from those; the leaf root and the release tab are 45-degree wedges; the lip's catch face
// is a 1.4 mm overhang. Holding force, torque capacity, vibration retention and creep are NOT
// qualified; rotation is limited to the jaws' play (sc_clip_keyed probes).
include <cleat_carrier.scad>
use <compute_tower.scad>
use <../antenna_mount.scad>

assert(ctw_front_bar_u() == ctw_post_u, "SMA clip assumes the clamp's front crossbar spans ctw_post_u");

function sc_bar_y() = [ctw_ty, ctw_ty + hatp_pcb[1]];             // frame -Y / +Y faces
function sc_band_x() = [ctw_tx + ctw_post_u[0] + 2, ctw_tx + ctw_post_u[1] - 2];   // finger bands
function sc_catch_z() = -ctw_clamp_t - sc_lip_cl;                  // lip catch face
function sc_jaw_x() = [[sc_x0, sc_band_x()[0] - sc_jaw_cl], [sc_band_x()[1] + sc_jaw_cl, sc_x1]];

module sc_xz(y0, y1) { translate([0, y1, 0]) rotate([90, 0, 0]) linear_extrude(height = y1 - y0) children(); }

module sc_jaws(y0, y1) {
    for (j = sc_jaw_x()) box3([j[0], y0, -sc_jaw_drop], [j[1], y1, 0.01]);
}
module sc_body() {
    pz = sc_plate_z;
    // North foot + column on the crossbar and the +Y band, up to the plate.
    box3([sc_x0, sc_north_y[0], 0], [sc_x1, sc_north_y[1], pz[1]]);
    sc_jaws(sc_bar_y()[1] + sc_jaw_cl, sc_north_y[1]);
    // South bridge on the -Y band, its gusset and the web up to the plate.
    box3([sc_x0, sc_south_y[0], 0], [sc_web_x[1], sc_south_y[1], sc_bridge_t]);
    sc_jaws(sc_south_y[0], sc_south_y[1]);
    sc_xz(sc_south_y[0], sc_south_y[1]) polygon([
        [sc_x0 + 1, sc_bridge_t - 0.5], [sc_web_x[0] + 0.01, sc_bridge_t - 0.5],
        [sc_web_x[0] + 0.01, sc_bridge_t - 0.5 + sc_web_x[0] - sc_x0 - 1]]);
    box3([sc_web_x[0], sc_south_y[0], 0], [sc_web_x[1], sc_plate_y[0] + 0.01, pz[1]]);
    // Plate.
    box3([sc_x0, sc_plate_y[0], pz[0]], [sc_plate_x1, sc_plate_y[1], pz[1]]);
    // Snap leaf: root wedge on the column, leaf down the crossbar's +X face, lip under the
    // crossbar, release tab at its foot.
    sc_xz(sc_leaf_y[0], sc_leaf_y[1]) {
        w = sc_leaf_x[1] - sc_x1 + 0.1;
        polygon([[sc_x1 - 0.1, sc_leaf_root_z], [sc_leaf_x[1], sc_leaf_root_z],
                 [sc_leaf_x[1], sc_leaf_root_z + 0.5], [sc_x1 - 0.1, sc_leaf_root_z + 0.5 + w]]);
        translate([sc_leaf_x[0], sc_tab[1]])
            square([sc_leaf_x[1] - sc_leaf_x[0], sc_leaf_root_z - sc_tab[1] + 0.01]);
        lt = sc_x1 - sc_lip_reach;
        polygon([[sc_leaf_x[0] + 0.01, sc_catch_z()], [lt, sc_catch_z()],
                 [sc_leaf_x[0] + 0.01, sc_catch_z() - (sc_leaf_x[0] + 0.01 - lt)]]);
        polygon([[sc_leaf_x[1] - 0.01, sc_tab[1]], [sc_tab[0], sc_tab[1]], [sc_tab[0], sc_tab[1] + 0.8],
                 [sc_leaf_x[1] - 0.01, sc_tab[1] + 0.8 + sc_tab[0] - sc_leaf_x[1]]]);
    }
}
// The clip in its frame: jack lands cut into the plate.
module sma_clip() {
    difference() {
        sc_body();
        for (j = sc_joints) sma_land_cut(j, sc_plate_z[1], sc_plate_z[0] - 1);
    }
}
// Clip frame -> body at clamp travel `travel`, with the tower moved by (dx, dz).
module sc_at_clamp(travel = 0, dx = 0, dz = 0) {
    translate([dx, 0, ctw_shelf_z() + ctw_clamp_top_w(travel) + dz]) children();
}
module sma_clip_print() { translate([0, 0, sc_plate_z[1]]) rotate([180, 0, 0]) sma_clip(); }
