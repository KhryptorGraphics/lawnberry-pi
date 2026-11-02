// Reference solids of the electronics, for scenes and probes. NOT printable.
//
// Outlines are the published ones (parameters.scad, URLs there); every part height,
// connector envelope and placement marked ESTIMATE in parameters.scad is an envelope chosen
// to be generous, not a measurement. Board frame (u, v, w): u along the long edge from the
// microSD / left end, v from the USB-C / bottom edge, w = 0 at the PCB underside.
include <parameters.scad>

// PCB slab with the Pi-pattern mounting holes that fall inside it (2.7, Pi 5 drawing;
// the Alta, AI HAT+ and HaLow HAT share the pattern: the Alta's is ASSUMED, TO VERIFY).
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

// HaLow HAT envelope (65 x 56, female-only header) with a mini-PCIe-format card; RF
// connector UNKNOWN, so both an edge SMA + right-angle plug and a card U.FL get room.
module rb_halow(headers = true) {
    color([0.20, 0.20, 0.55]) rb_slab(hr_pcb[0], hr_pcb[1]);
    color([0.70, 0.72, 0.75]) {
        rb_block([hr_card_uv[0], hr_card_uv[1], board_t + hr_card_z], hr_card);
        rb_block([hr_card_uv[0] - 3, hr_card_uv[1], board_t - 0.01], [6, hr_card[1], hr_card_z + hr_card[2]]);
    }
    color([0.85, 0.75, 0.2]) {
        rb_block([hr_card_uv[0] + hr_card[0] - 7, hr_card_uv[1] + 4,
                  board_t + hr_card_z + hr_card[2] - 0.01], [6, 6, hr_ufl_h + 0.01]);
        rb_block([hr_sma_u[0], hr_pcb[1] - 2, board_t - 0.01],
                 [hr_sma_u[1] - hr_sma_u[0], 2 + hr_sma_reach, hr_sma_h]);
    }
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
