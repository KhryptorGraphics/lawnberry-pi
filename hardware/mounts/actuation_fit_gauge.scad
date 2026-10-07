// Fit coupons for the retained E-stop frame interface and servo side pattern.
// Steering clamps now use lapbar_four_bolt_clamp.scad's measured-OD gauge;
// the previous round-U-bolt/V-seat gauges no longer apply to steering.
// gauge 1: selected rail cross-section and square-U-bolt pitch for E-stop only.
// gauge 2: RDS51150 fixed-holder SIDE 24x24 M2.5 pattern.
include <common.scad>
gauge = 1;

module actuation_fit_gauge() {
    if (gauge == 1) difference() {
        plate(110, 70, 4, 5);
        translate([0, 0, 2]) cube([frame_rail_w + 0.8, frame_rail_h + 0.8, 6], center = true);
        for (x = [-45, 45], y = [-24, 24])
            translate([x, y, -1]) cylinder(d = frame_ubolt_leg_hole_d, h = 6);
    }
    else if (gauge == 2) difference() {
        plate(38, 38, 3, 3);
        for (x = [-12, 12], y = [-12, 12])
            translate([x, y, -1]) cylinder(d = servo_side_hole_clear_d, h = 5);
    }
    else assert(false, "gauge must be 1 or 2");
}
actuation_fit_gauge();
