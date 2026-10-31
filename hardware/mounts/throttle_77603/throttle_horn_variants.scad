// Five complete throttle linkage variants, each interfacing the RETAINED existing MG995
// X-horn without altering the servo, the horn or its factory shaft retainer.
//
// Variants are deliberately different mechanisms, not one mechanism with renamed parts:
//   A  direct captive joint through an existing X-horn arm hole
//   B  printed X-shaped reinforcement shell over the horn, distributing reaction
//   C  continuously variable effective horn radius on a captive radial carriage
//   D  printed ball with a replaceable split wear socket in a structural cage
//   E  double-shear clevis with replaceable bushings and a captive printed pin
//
// No variant is selected as a winner. All are printable-only; no bought ball link, clevis,
// pin, bolt or insert appears in any of them.
include <parameters.scad>
use <printed_joints.scad>

horn_r = x_horn_arm_radius_mm;
horn_t = x_horn_thickness_mm;
horn_hole = x_horn_hole_d_mm;
shaft_clear = x_horn_hub_clear_d_mm;

// ---- A: direct joint through an existing horn arm hole ------------------------------
module variant_a(part = "assembly") {
    if (part == "assembly") {
        variant_a("eye");
        translate([0, 0, -clevis_ear_t_mm - keeper_t_mm]) variant_a("pin");
        translate([0, clevis_gap_mm, -clevis_ear_t_mm - keeper_t_mm]) variant_a("keeper");
    } else if (part == "eye") {
        // Single-shear eye sharing the existing horn hole. Built as ONE 2D profile then
        // extruded, so the bore cannot detach a flange ring from the eye.
        // Printed lying flat: the 2D profile is centred on the origin, so extruding from
        // z=0 puts the lowest point exactly on the bed.
        linear_extrude(height = clevis_ear_t_mm)
            difference() {
                union() {
                    circle(d = horn_hole + 14);
                    translate([-clevis_gap_mm / 2, 0]) square([clevis_gap_mm, 11]);
                }
                circle(d = horn_hole + joint_clearance_mm);
            }
    } else if (part == "pin") {
        printed_pin(clevis_ear_t_mm + 2 * keeper_t_mm);
    } else if (part == "keeper") {
        printed_keeper();
    } else {
        assert(false, str("A: unknown part ", part));
    }
}

// ---- B: printed X-shaped reinforcement shell ---------------------------------------
module variant_b(part = "assembly") {
    if (part == "assembly") {
        variant_b("shell_a");
        variant_b("shell_b");
        variant_b("lug");
        translate([0, 0, -20]) variant_b("pin");
    } else if (part == "shell_a" || part == "shell_b") {
        // Two halves clamp the retained X horn; the shell carries the reaction across all
        // four arms instead of one 2 mm arm. Central hub stays open for the shaft retainer.
        difference() {
            union() {
                cylinder(d = 2 * horn_r + 8, h = horn_t + (part == "shell_a" ? 4 : 2));
                for (k = [0:3])
                    rotate([0, 0, 45 + k * 90])
                        translate([horn_r * 0.45, -5, 0]) cube([horn_r * 0.75, 10, horn_t + 4]);
            }
            translate([0, 0, -1]) cylinder(d = shaft_clear, h = horn_t + 8);
            for (k = [0:3])
                rotate([0, 0, 45 + k * 90])
                    translate([horn_r, 0, -1]) cylinder(d = horn_hole + joint_clearance_mm,
                                                        h = horn_t + 8);
            if (part == "shell_b")
                translate([-40, -40, (horn_t + 4) / 2])
                    cube([80, 80, 40]);
        }
    } else if (part == "lug") {
        // Outboard joint lug, captured under the shell halves.
        difference() {
            translate([-clevis_gap_mm / 2 - 2, -clevis_ear_t_mm, 0])
                cube([clevis_gap_mm + 4, clevis_ear_t_mm, 24]);
            rotate([90, 0, 0]) cylinder(d = pin_d_mm + 2 * joint_clearance_mm,
                                        h = clevis_ear_t_mm + 2, center = true);
        }
    } else if (part == "pin") {
        printed_pin(clevis_ear_t_mm + 2 * keeper_t_mm);
    } else {
        assert(false, str("B: unknown part ", part));
    }
}

// ---- C: continuously variable effective horn radius --------------------------------
function c_radius(radius_mm) = min(horn_radius_max_mm, max(horn_radius_min_mm, radius_mm));

module variant_c(part = "assembly", radius_mm = horn_radius_default_mm) {
    r = c_radius(radius_mm);
    if (part == "assembly") {
        variant_c("carriage", radius_mm = r);
        translate([0, 0, 0]) variant_c("shoe", radius_mm = r);
        translate([0, 0, keeper_t_mm]) variant_c("wedge", radius_mm = r);
        translate([0, 0, -16]) variant_c("pin", radius_mm = r);
    } else if (part == "carriage") {
        // Captive radial dovetail carriage: continuous slot, no indexing holes anywhere.
        difference() {
            union() {
                translate([-12, -10, 0]) cube([24, 20, 10]);
                translate([r - 16, -8, 0]) cube([34, 16, 10]);
                translate([r - 8, -6, 10]) cube([18, 12, 12]);
            }
            // Continuous radial slot: the shoe slides anywhere along it.
            translate([-14, -4.2, -1]) cube([r + 32, 8.4, 12]);
            // Keeper land for the secondary retainer.
            translate([r - 6, -keeper_t_mm / 2, 12]) cube([keeper_t_mm, keeper_t_mm, 12]);
            translate([0, 0, -1]) cylinder(d = shaft_clear, h = 12);
        }
    } else if (part == "shoe") {
        // Sliding shoe whose eye carries the rod pin; the wedge pinches it in the dovetail.
        difference() {
            translate([-9, -4.2 + joint_clearance_mm, 0])
                cube([18, 8.4 - 2 * joint_clearance_mm, 10]);
            translate([0, 0, 10]) rotate([0, 0, 0])
                cylinder(d = pin_d_mm + 2 * joint_clearance_mm, h = 12);
        }
    } else if (part == "wedge") {
        printed_keeper();
    } else if (part == "pin") {
        printed_pin(12);
    } else {
        assert(false, str("C: unknown part ", part));
    }
}

// ---- D: captured printed ball with replaceable wear socket --------------------------
module variant_d(part = "assembly") {
    if (part == "assembly") {
        variant_d("ball_stem");
        translate([0, 0, ball_d_mm * 0.75]) variant_d("socket");
        translate([0, 0, ball_d_mm * 0.75 + ball_socket_wall_mm]) variant_d("cage");
        translate([0, ball_d_mm, ball_d_mm * 0.75]) variant_d("closure");
    } else if (part == "ball_stem") {
        // Printed ball on a plate, lifted so the lowest point sits on the bed. Built as an
        // explicit hull pair instead of sphere-plus-cylinder: a cylinder meeting a sphere
        // leaves a zero-area seam triangle.
        translate([0, 0, ball_d_mm]) union() {
            hull() {
                translate([0, 0, 0]) sphere(d = ball_d_mm);
                translate([0, 0, -ball_d_mm * 0.35]) sphere(d = ball_d_mm * 0.55);
            }
            difference() {
                translate([-9, -clevis_ear_t_mm / 2, -ball_d_mm])
                    cube([18, clevis_ear_t_mm, ball_d_mm]);
                rotate([90, 0, 0])
                    cylinder(d = pin_d_mm + 2 * joint_clearance_mm,
                             h = clevis_ear_t_mm + 2, center = true);
            }
        }
    } else if (part == "socket") {
        // Replaceable sacrificial wear liner.
        printed_wear_socket();
    } else if (part == "cage") {
        // Structural cage: retains the ball even if the liner loses preload.
        difference() {
            cylinder(d = ball_d_mm + 2 * ball_socket_wall_mm + 6, h = ball_d_mm * 1.15);
            sphere(d = ball_d_mm + 2 * ball_socket_wall_mm);
            translate([0, 0, -1]) cylinder(d = ball_d_mm * 0.55, h = ball_d_mm * 0.6);
        }
    } else if (part == "closure") {
        // Independent captive closure: the ball cannot lift out if the cage relaxes.
        difference() {
            cylinder(d = ball_d_mm + 2 * ball_socket_wall_mm + 8, h = keeper_t_mm);
            cylinder(d = ball_d_mm * 0.75, h = keeper_t_mm + 1);
            for (s = [-1, 1])
                translate([s * (ball_d_mm / 2 + ball_socket_wall_mm), -keeper_t_mm / 2, -1])
                    cube([keeper_t_mm, keeper_t_mm, keeper_t_mm + 2]);
        }
    } else {
        assert(false, str("D: unknown part ", part));
    }
}

// ---- E: double-shear clevis with replaceable bushings ------------------------------
module variant_e(part = "assembly") {
    if (part == "assembly") {
        variant_e("clevis");
        variant_e("bushing");
        variant_e("pin");
        variant_e("keeper");
    } else if (part == "clevis") {
        printed_clevis();
    } else if (part == "bushing") {
        printed_bushing();
    } else if (part == "pin") {
        printed_pin(2 * clevis_ear_t_mm + clevis_gap_mm + 2 * keeper_t_mm);
    } else if (part == "keeper") {
        printed_keeper();
    } else {
        assert(false, str("E: unknown part ", part));
    }
}

// ---- Dispatcher --------------------------------------------------------------------
// The selector names are DISTINCT from the module parameter names, so `-D horn_variant="B"`
// cannot be shadowed by a module signature default. Same reason for horn_part.
horn_variant = "A";
horn_part = "assembly";

module throttle_horn_variant(variant = horn_variant, part = horn_part,
                             radius_mm = horn_radius_default_mm) {
    if (variant == "A") variant_a(part);
    else if (variant == "B") variant_b(part);
    else if (variant == "C") variant_c(part, radius_mm = radius_mm);
    else if (variant == "D") variant_d(part);
    else if (variant == "E") variant_e(part);
    else assert(false, str("unknown variant: ", variant));
}

throttle_horn_variant(horn_variant, horn_part);
