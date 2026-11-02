// Reference solids of the electronics, for scenes and probes. NOT printable.
//
// Outlines are the published ones (parameters.scad, URLs there); every part height,
// connector envelope and placement marked ESTIMATE in parameters.scad is an envelope chosen
// to be generous, not a measurement. Board frame (u, v, w): u along the long edge from the
// microSD / left end, v from the USB-C / bottom edge, w = 0 at the PCB underside.
include <parameters.scad>

// PCB slab with the Pi-pattern mounting holes that fall inside it (2.7, Pi 5 drawing;
// the Alta, AI HAT+ and HaLow HATs share the pattern: the Alta's is ASSUMED, TO VERIFY).
module rb_slab(l, wv, v0 = 0) {
    difference() {
        translate([0, v0, 0]) cube([l, wv, board_t]);
        for (h = pi5_holes) if (h[1] > v0 && h[1] < v0 + wv)
            translate([h[0], h[1], -1]) cylinder(d = pi5_hole_d, h = board_t + 2, $fn = 24);
    }
}
// 2x20 male pin field above the PCB top (Pi pattern: centre u = 29, v = 52.5).
module rb_hdr_pins(h) {
    translate([pi5_gpio_u - hdr_len / 2, hdr_v - hdr_w / 2, board_t - 0.01])
        cube([hdr_len, hdr_w, h + 0.01]);
}
// 2x20 female socket under the PCB.
module rb_hdr_socket(h) {
    translate([pi5_gpio_u - hdr_len / 2, hdr_v - hdr_w / 2, -h]) cube([hdr_len, hdr_w, h + 0.01]);
}
module rb_block(p0, s) { translate(p0) cube(s); }

// Raspberry Pi 5 + Active Cooler.
module rb_pi5(headers = true) {
    color([0.14, 0.50, 0.18]) rb_slab(pi5_pcb[0], pi5_pcb[1]);
    color([0.70, 0.72, 0.75]) {
        rb_block([cooler_uv0[0], cooler_uv0[1], board_t - 0.01], [cooler_xy[0], cooler_xy[1], cooler_h + 0.01]);
        // RJ45 + USB stacks on the u = 85 edge, 3 mm overhang (drawing); height ESTIMATE.
        rb_block([64, 2, board_t - 0.01], [pi5_pcb[0] + pi5_port_overhang - 64, 53, pi5_port_h]);
        // USB-C and micro-HDMI on the v = 0 edge (centres from the drawing; 1 mm overhang EST).
        for (c = [[pi5_usbc_u, 9], [pi5_hdmi_u[0], 7.5], [pi5_hdmi_u[1], 7.5]])
            rb_block([c[0] - c[1] / 2, -1, board_t - 0.01], [c[1], 8, 3.5]);
    }
    color([0.25, 0.25, 0.25]) rb_block([7, 7, -under_h], [71, 42, under_h + 0.01]);
    if (headers) color([0.1, 0.1, 0.1]) rb_hdr_pins(male_pin_h);
}

// Libre Computer Alta: 85.6 x 56.5 measured; Pi 3B+-style layout ASSUMED.
module rb_alta(headers = true) {
    color([0.92, 0.70, 0.25]) rb_slab(alta_pcb[0], alta_pcb[1]);
    color([0.70, 0.72, 0.75]) {
        rb_block([12, 8, board_t - 0.01], [40, 38, alta_hs_h + 0.01]);          // heatsink EST
        rb_block([64, 2, board_t - 0.01], [alta_pcb[0] + pi5_port_overhang - 64, 53, pi5_port_h]);
        // USB-C 10.6, HDMI 32, audio 53.5 on the v = 0 edge (Pi 3B+ positions, EST).
        for (c = [[10.6, 9], [32, 15], [53.5, 7]])
            rb_block([c[0] - c[1] / 2, -1.5, board_t - 0.01], [c[1], 9, 6]);
    }
    color([0.25, 0.25, 0.25]) rb_block([7, 7, -under_h], [71, 42, under_h + 0.01]);
    if (headers) color([0.1, 0.1, 0.1]) rb_hdr_pins(male_pin_h);
}

// AI HAT+ on its supplied 16 mm stacking header (female under, pins up).
module rb_aihat(headers = true) {
    color([0.14, 0.45, 0.70]) rb_slab(hatp_pcb[0], hatp_pcb[1]);
    color([0.2, 0.2, 0.2]) rb_block([24, 18, board_t - 0.01], [hailo_xy[0], hailo_xy[1], 2.5]);
    if (headers) color([0.1, 0.1, 0.1]) {
        rb_hdr_socket(s_pi_ai - hdr_base_h);
        rb_hdr_pins(stack_pin_h);
    }
}

// LC29H(DA) on the purchased extra-tall stacking header (DEFAULT stack).
module rb_lc29h(headers = true) {
    color([0.15, 0.40, 0.25]) rb_slab(lc_pcb[0], lc_pcb[1], lc_v0);
    color([0.85, 0.75, 0.2]) rb_block([lc_ufl_uv[0] - 2, lc_ufl_uv[1] - 2, board_t - 0.01], [4, 4, 3]);
    if (headers) color([0.1, 0.1, 0.1]) {
        rb_hdr_socket(socket_h);
        rb_hdr_pins(stack_pin_h);
    }
}

// Pi 5 HaLow: Heltec HT-HC01P mini-PCIe card (published 50.95 x 30 x 3.75, U.FL position from the
// card drawing) on a Pi carrier HAT (model TO CONFIRM: 65 x 56 envelope, carrier parts, socket and
// card placement ESTIMATE). Fingers in the socket at u ~3..12, card running +u to its far (U.FL)
// end, shield block on the card top, the U.FL receptacle beside the far-end screw.
module rb_halow_hc01p(headers = true) {
    zc = board_t + hc_card_z;                       // card underside
    cu = hc_card_u();
    color([0.20, 0.20, 0.55]) rb_slab(hr_pcb[0], hr_pcb[1]);
    color([0.25, 0.25, 0.25]) {
        rb_block([hc_socket_u[0], hc_card_v[0] - 1, board_t - 0.01],
            [hc_socket_u[1] - hc_socket_u[0], hc_card_v[1] - hc_card_v[0] + 2, hc_card_z + hc_card_t + 0.01]);
        // Carrier parts off the card's footprint and the clamp pads (ESTIMATE): a strip along the
        // v = 0 edge between the hole pads.
        rb_block([8, 1, board_t - 0.01], [49, hc_card_v[0] - 2, hc_carrier_h + 0.01]);
    }
    color([0.70, 0.72, 0.75]) {
        rb_block([cu[0], hc_card_v[0], zc], [cu[1] - cu[0], hc_card_v[1] - hc_card_v[0], hc_card_t]);
        rb_block([cu[0] + hc_shield[0][0], hc_card_v[0] + hc_shield[0][1], zc + hc_card_t - 0.01],
            [hc_shield[1][0], hc_shield[1][1], hc_shield[1][2] + 0.01]);
        // Far-end screw standoff (card holes 2.9 from the far end), ESTIMATE d 4.5.
        translate([cu[1] - 2.9, hc_card_v[0] + 2.9, board_t - 0.01]) cylinder(d = 4.5, h = hc_card_z + 0.02, $fn = 24);
    }
    color([0.85, 0.75, 0.2]) rb_block([hc_ufl_uv()[0] - ufl_body[0] / 2, hc_ufl_uv()[1] - ufl_body[1] / 2,
        zc + hc_card_t - 0.01], [ufl_body[0], ufl_body[1], ufl_body[2] + 0.01]);
    if (headers) color([0.1, 0.1, 0.1]) rb_hdr_socket(socket_h);
}
// Alta HaLow: Seeed WM1302 Pi HAT carrying the Wio-WM6108 mini-PCIe card (finger end in J1 at
// u ~57-61, card running -u), its U.FL receptacle on the card top near the far (screw) end.
module rb_halow_wm1302(headers = true) {
    zc = board_t + wm_card_z;                       // card underside
    color([0.20, 0.20, 0.55]) rb_slab(hr_pcb[0], hr_pcb[1]);
    color([0.25, 0.25, 0.25]) rb_block([wm_socket_u[0], wm_card_v[0] - 1, board_t - 0.01],
        [wm_socket_u[1] - wm_socket_u[0], wm_card_v[1] - wm_card_v[0] + 2, wm_card_z + wm_card_t + 0.01]);
    color([0.70, 0.72, 0.75]) {
        rb_block([wm_card_u[0], wm_card_v[0], zc], [wm_card_u[1] - wm_card_u[0], wm_card_v[1] - wm_card_v[0], wm_card_t]);
        rb_block([wm_module_u0, wm_card_v[0], zc + wm_card_t - 0.01],
            [wm_card_u[1] - wm_module_u0, wm_card_v[1] - wm_card_v[0], wm_module_h + 0.01]);
        // Card screw standoffs J2/J3 (board file: u 9.33, v 15.90 / 40.10), ESTIMATE d 4.5.
        for (v = [15.9, 40.1]) translate([9.33, v, board_t - 0.01]) cylinder(d = 4.5, h = wm_card_z + 0.02, $fn = 24);
    }
    color([0.85, 0.75, 0.2]) rb_block([wm_ufl_uv[0] - ufl_body[0] / 2, wm_ufl_uv[1] - ufl_body[1] / 2,
        zc + wm_card_t - 0.01], [ufl_body[0], ufl_body[1], ufl_body[2] + 0.01]);
    if (headers) color([0.1, 0.1, 0.1]) rb_hdr_socket(socket_h);
}

// Board-mounted parts on the signal plate, in the carrier frame (board in the x-z plane):
// PCB slab, front envelope (headers + plugs, inset 2.6 from the edges) and back envelope
// (pin tails, inset 5.5: pin positions TO MEASURE on the clones).
module rb_plate_board(bx, bz, bw, bh, bt, front_h, yb) {
    yf = yb - bt;
    color([0.25, 0.50, 0.80]) translate([bx, yf, bz]) cube([bw, bt, bh]);
    color([0.1, 0.1, 0.1]) {
        translate([bx + 2.6, yf - front_h, bz + 2.6]) cube([bw - 5.2, front_h + 0.01, bh - 5.2]);
        // Pin tails: kept 5.5 inside the edges (the clip pockets hold the corners).
        translate([bx + 5.5, yb - 0.01, bz + 5.5]) cube([bw - 11, pol_tail_h + 0.01, bh - 11]);
    }
}

// A converter / relay as a plain box in a strap bay: [width along the wall, depth, height].
module rb_bay_box(size) { color([0.55, 0.35, 0.20]) cube(size); }

// INA3221 breakout as a capacity envelope: PCB [across x along x t] (x = 0 is the edge on the
// fixed jaw, x = w the edge under the shoe); parts above it and pin tails below it kept
// ina_edge_keep[0] / [1] off those two long edges and ina_end_keep off the short edges
// (outlines in parameters.scad; heights ESTIMATE).
module rb_ina3221(w, l, t = ina_t) {
    k = ina_edge_keep;
    color([0.45, 0.20, 0.55]) cube([w, l, t]);
    color([0.10, 0.35, 0.20]) {
        translate([k[0], ina_end_keep, t - 0.01])
            cube([w - k[0] - k[1], l - 2 * ina_end_keep, ina_top_h + 0.01]);
        translate([k[0], ina_end_keep, -ina_tail_h])
            cube([w - k[0] - k[1], l - 2 * ina_end_keep, ina_tail_h + 0.01]);
    }
}

// Adafruit #6062 in the same frame, in its REQUIRED orientation (terminal edge on the jaw at
// x = 0, header edge under the shoe), with its header soldered and a 2.54 mm Dupont housing
// plugged along the whole row (ESTIMATE envelope). `grow` inflates every part (not the PCB)
// in x and y: an empty intersection with grow g proves at least g of clearance.
module rb_ada_box(p0, p1, z0, z1, g) {
    translate([p0[0] - g, p0[1] - g, z0]) cube([p1[0] - p0[0] + 2 * g, p1[1] - p0[1] + 2 * g, z1 - z0]);
}
module rb_ina3221_adafruit(grow = 0, t = ina_t) {
    w = ina_board_adafruit[0];
    // Board frame (x along the 38.354 edge, y from the header edge) -> cradle frame.
    translate([w, 0, 0]) rotate([0, 0, 90]) {
        color([0.10, 0.10, 0.12]) cube([ina_board_adafruit[1], w, t]);
        color([0.15, 0.15, 0.15]) {
            rb_ada_box(ada_hdr[0], ada_hdr[1], t - 0.01, t + ada_hdr_h, grow);
            rb_ada_box(ada_hdr_pads[0], ada_hdr_pads[1], -ina_tail_h, 0.01, grow);
        }
        color([0.20, 0.20, 0.75]) rb_ada_box(ada_hdr[0], ada_hdr[1], t + ada_hdr_h - 0.01,
                                             t + ada_hdr_h + ada_dupont_h, grow);
        color([0.20, 0.55, 0.30]) for (x = ada_tb_x) {
            rb_ada_box([x + ada_tb_dx[0], ada_tb_y[0]], [x + ada_tb_dx[1], ada_tb_y[1]], t - 0.01, t + ada_tb_h, grow);
            rb_ada_box([x - 2.7, ada_tb_pads_y[0]], [x + 2.6, ada_tb_pads_y[1]], -ina_tail_h, 0.01, grow);
        }
        color([0.85, 0.85, 0.85]) for (q = ada_qt) rb_ada_box(q[0], q[1], t - 0.01, t + ada_qt_h, grow);
    }
}
