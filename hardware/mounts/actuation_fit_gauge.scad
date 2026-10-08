// Fit coupons for the retained E-stop frame interface and six-hole rear bracket.
// Steering clamps now use lapbar_four_bolt_clamp.scad's measured-OD gauge;
// the previous round-U-bolt/V-seat gauges no longer apply to steering.
// gauge 1: selected rail cross-section and square-U-bolt pitch for E-stop only.
// gauge 2: photo-derived stationary rear bracket slots + rear-opening capacity.
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
        plate(60, 38, 3, 3);
        // Rotate shared full-X cutters to bed-normal; coupon axes are -Z,Y.
        translate([0,0,-1]) rotate([0,-90,0]) {
            servo_rear_hole_cut(5);
            servo_rear_opening_cut(5);
        }
    }
    else assert(false, "gauge must be 1 or 2");
}
actuation_fit_gauge();
