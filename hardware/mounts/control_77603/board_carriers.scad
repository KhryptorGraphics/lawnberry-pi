// Interface carriers that hang on the chassis cleat strips: the strap bay (converters and
// relays) and the harness clamp; plus the printable dispatch for every control piece.
//
// Strap bay (one print, six copies): a box stands on the ledge against three back ribs. The
// print-in-place STRAP BAR rides between the two posts; a pawl finger at each end of the
// bar ratchets in a toothed groove along its post (1.5 mm steps), and a spring leaf on the
// bar takes up the step with 1.0-2.5 mm preload. Push the bar onto the box to clamp; to
// release, press both finger tops toward the middle (1.4 mm slot) and pull the bar away.
// Box capacity (DESIGN, parameters.scad): width along the panel <= 44, depth 15.5-30,
// height >= 18 up to each slot's capacity (lay_bays). Holding force, vibration retention
// and creep are NOT qualified.
//
// The harness clamp is the same bay without the ledge, 30 mm between posts, 7.5-30 mm stops:
// the bundle runs vertically between the posts and the bar presses it onto the back ribs.
//
// Print orientation: bays and the harness clamp print UPRIGHT as installed (plate bottom,
// ledge and strap bar on the bed, hook up). `carrier_part = "body"` renders each printable
// piece in its print pose, bed-zero.
include <cleat_carrier.scad>
use <compute_tower.scad>
use <chassis.scad>
use <ina_holder.scad>
use <sma_clip.scad>

// ---- Strap bay ---------------------------------------------------------------------------
function bay_W(w_in) = w_in + 2 * bay_post;
function bay_dk(d_max, k) = d_max - rt_pitch * k;
function bay_front_y(d_max) = bay_ref_y - (d_max + 3.5) - bay_bar_t - 0.6;
function bay_yf(d) = bay_ref_y - d - 4;        // pawl catch-face y with the bar at stop d
function bay_zt0() = -bay_drop + bay_tooth_z0;
// Leaf span: bay_leaf_span, or less where the crossbar is shorter (harness clamp).
function bay_leaf_span_for(w_in) =
    min(bay_leaf_span, bay_W(w_in) - 2 * (bay_post + 0.4 + rt_finger_t + bay_finger_slot) - 4);

module bay_groove_2d() {
    zt0 = bay_tooth_z0;   // relative to the ledge bottom, shifted by the caller
    th = bay_tooth_h;
    polygon([[bay_post + 0.01, zt0 - 0.8], [bay_post - bay_groove_d, zt0 + 0.8],
             [bay_post - bay_groove_d, zt0 + th - 0.8], [bay_post + 0.01, zt0 + th + 0.8]]);
}

module bay_post_left(W, d_max, steps) {
    z_lb = -bay_drop;
    front = bay_front_y(d_max);
    g0 = front + 1.0;
    g1 = bay_yf(bay_dk(d_max, steps)) + rt_tooth_out + 1.0;
    difference() {
        box3([0, front, z_lb], [bay_post, cr_plate_front + 0.01, z_lb + bay_post_h]);
        translate([0, 0, z_lb]) xz_prism(g0, g1) bay_groove_2d();
    }
    // Rack teeth inside the groove: catch faces at y_k face +y (toward the wall).
    intersection() {
        translate([0, 0, z_lb]) xz_prism(g0, g1) bay_groove_2d();
        translate([bay_post - bay_groove_d, 0, z_lb + bay_tooth_z0 - 1])
            linear_extrude(height = bay_tooth_h + 2)
                for (k = [0 : steps]) let(s = bay_yf(bay_dk(d_max, k)))
                    polygon([[-0.01, s - rt_h], [rt_h, s], [-0.01, s]]);
    }
}

module bay_frame(w_in = bay_w_in, d_max = bay_d_max, steps = bay_steps, ledge = true) {
    W = bay_W(w_in);
    z_lb = -bay_drop;
    g = cr_gap(0, W);
    d_min = bay_dk(d_max, steps);
    rib_z0 = z_lb + (ledge ? bay_ledge_t : 0);
    cr_hanger(0, W, z_lb);
    // Back ribs: two full-height side ribs, one centre rib under the latch tongue.
    for (x = [9, W - 11])
        box3([x, bay_ref_y, rib_z0], [x + 2, cr_plate_front + 0.01, cr_hook_top]);
    box3([W / 2 - 1, bay_ref_y, rib_z0],
         [W / 2 + 1, cr_plate_front + 0.01, cr_tongue_free - cr_tongue_slot - 0.6]);
    if (ledge) difference() {
        box3([bay_post - 0.01, bay_ref_y - d_min, z_lb],
             [W - bay_post + 0.01, cr_plate_front + 0.01, z_lb + bay_ledge_t]);
        // Recess under the bar's spring leaf at the closest stops.
        box3([W / 2 - bay_leaf_span_for(w_in) / 2 - 1, bay_ref_y - d_min - 1, z_lb - 1],
             [W / 2 + bay_leaf_span_for(w_in) / 2 + 1, bay_ref_y - d_min + bay_leaf + 0.5,
              z_lb + bay_ledge_t + 1]);
    }
    bay_post_left(W, d_max, steps);
    translate([W, 0, 0]) mirror([1, 0, 0]) bay_post_left(W, d_max, steps);
}

// The strap bar with its back face at distance d from the rib plane.
module bay_bar(w_in = bay_w_in, d = bay_d_max + 3.5) {
    W = bay_W(w_in);
    z_lb = -bay_drop;
    yb = bay_ref_y - d;
    xf0 = bay_post + 0.4;
    xf1 = xf0 + rt_finger_t;
    xs1 = xf1 + bay_finger_slot;
    zt0 = z_lb + bay_tooth_z0;
    yf = bay_yf(d);
    xc = W / 2;
    box3([xs1, yb - bay_bar_t, z_lb], [W - xs1, yb, z_lb + bay_bar_h]);
    for (m = [0, 1]) translate([m * W, 0, 0]) mirror([m, 0, 0]) {
        box3([xf0, yb - bay_bar_t, z_lb], [xs1 + 0.01, yb, z_lb + bay_finger_root]);
        box3([xf0, yb - bay_bar_t, z_lb], [xf1, yb, z_lb + bay_bar_h]);
        intersection() {
            translate([0, 0, zt0]) linear_extrude(height = bay_tooth_h)
                polygon([[xf0 + 0.01, yf], [xf0 - rt_tooth_out, yf], [xf0 + 0.01, yf + rt_tooth_out + 0.01]]);
            xz_prism(yf - 1, yf + rt_tooth_out + 1) polygon([
                [xf0 + 0.01, zt0], [xf0 - rt_tooth_out, zt0 + rt_tooth_out],
                [xf0 - rt_tooth_out, zt0 + bay_tooth_h - rt_tooth_out], [xf0 + 0.01, zt0 + bay_tooth_h]]);
        }
    }
    // Spring leaf: a 1.2 mm arc bulging bay_leaf toward the box, inside the crossbar's ends.
    span = bay_leaf_span_for(w_in);
    n = 16;
    outer = [for (i = [0 : n]) let(x = xc - span / 2 + span * i / n, t = 2 * (x - xc) / span)
                 [x, yb + bay_leaf * (1 - t * t)]];
    inner = [for (i = [n : -1 : 0]) [outer[i][0], outer[i][1] - bay_leaf_t]];
    translate([0, 0, z_lb]) linear_extrude(height = bay_bar_h)
        polygon(concat(outer, inner));
}

// Ratchet stop index that clamps a box of depth d (leaf compressed 1.0..2.5 mm).
function bay_stop_for(d, d_max, steps) =
    let(ks = [for (k = [0 : steps]) if (bay_dk(d_max, k) >= d) k]) ks[len(ks) - 1];

module strap_bay(part = "assembly", d = bay_d_max + 3.5) {
    if (part == "frame") bay_frame();
    else if (part == "bar") bay_bar(bay_w_in, d);
    else { bay_frame(); bay_bar(bay_w_in, d); }
}
module harness_clamp(part = "assembly", d = hc_d_max() + 3.5) {
    if (part == "frame") bay_frame(hc_w_in, hc_d_max(), hc_steps(), false);
    else if (part == "bar") bay_bar(hc_w_in, d);
    else { bay_frame(hc_w_in, hc_d_max(), hc_steps(), false); bay_bar(hc_w_in, d); }
}
function hc_d_max() = hc_bundle_d[1];
function hc_steps() = 15;

// ---- Printable dispatch ------------------------------------------------------------------
// carrier_kind: strap_bay | harness_clamp | chassis | chassis_base | tower_rack | tower_clamp
//               | ina_cradle | sma_clip
carrier_kind = "strap_bay";
carrier_part = "body";

module control_carrier_print(kind = carrier_kind) {
    if (kind == "strap_bay") translate([0, 0, bay_drop]) strap_bay("assembly");
    else if (kind == "harness_clamp") translate([0, 0, bay_drop]) harness_clamp("assembly");
    else if (kind == "chassis") chassis_print();
    else if (kind == "chassis_base") chassis_base_print();
    else if (kind == "tower_rack") tower_rack_print();
    else if (kind == "tower_clamp") tower_clamp_print();
    else if (kind == "ina_cradle") ina_cradle_print();
    else if (kind == "sma_clip") sma_clip_print();
    else assert(false, str("unknown carrier kind: ", kind));
}

if (carrier_part == "body") control_carrier_print(carrier_kind);
else assert(false, str("unknown carrier part: ", carrier_part));
