// Bolt-free electronics mounting interface: the STRIP side.
//
// The removable control_77603 chassis molds horizontal French-cleat strips on its panels
// (the enclosure body is unchanged); every electronics carrier hangs on one strip by a
// 45-degree hook and is held down by a spring latch that snaps under the strip's catch
// land. No screws, standoffs or inserts.
//
// Contract (shared by control_77603/chassis.scad and the control_77603 carriers). Local frame:
//   * the panel face (the "wall" in wall_cleat() and below) is the plane y = 0; the carrier
//     side is y < 0;
//   * the strip runs along +X from x = 0 to x = len;
//   * z = 0 is the ROOT of the hook groove, where the top bevel meets the wall.
//
// Profile (y, z), counter-clockwise from the groove root:
//   (0, 0) -> (-t, t)             top bevel, 45 degrees, rising away from the wall
//   (-t, t) -> (-t, t - h)        outer face
//   (-t, t - h) -> (-t + land, t - h)   CATCH LAND: horizontal, faces down
//   (-t + land, t - h) -> (0, ...)       45-degree underside back to the wall
// A carrier's hook drops into the V between the bevel and the wall (gravity pulls it down
// and toward the wall); its latch tooth snaps up under the catch land, so the carrier can
// neither lift out nor pull away until the latch is released by hand.
//
// Key notches across the top bevel every cleat_notch_pitch_mm stop a hooked carrier from
// sliding along the strip; a carrier's hook carries matching key ribs.
//
// Printability: the strip is molded on a wall that prints vertically. The bevels are
// 45 degrees; the only down-facing horizontal face is the cleat_land_mm catch land.
//
// Geometry only: retention force, vibration life and creep are NOT qualified.

cleat_t_mm = 6;                 // protrusion from the wall face
cleat_h_mm = 12;                // outer-face height
cleat_land_mm = 2;              // horizontal catch land under the outer edge
cleat_clearance_mm = 0.4;       // per mating face, carrier side applies it
cleat_notch_pitch_mm = 10;      // key notch pitch along the strip
cleat_notch_w_mm = 2.4;         // notch width along X (a 2.0 mm key rib + clearance)
cleat_notch_d_mm = 1.5;         // notch depth, normal to the bevel

// Accessors, so a file that `use`s this one (which shares functions, never variables)
// reads the same numbers.
function cleat_t() = cleat_t_mm;
function cleat_h() = cleat_h_mm;
function cleat_land() = cleat_land_mm;
function cleat_clearance() = cleat_clearance_mm;
function cleat_notch_pitch() = cleat_notch_pitch_mm;
function cleat_notch_w() = cleat_notch_w_mm;
function cleat_notch_d() = cleat_notch_d_mm;

// The catch land's height relative to the groove root, and how far it reaches from the
// wall. A carrier latch tooth must sit at z <= cleat_land_z() - cleat_clearance_mm and
// inside y in [cleat_land_y0(), cleat_land_y1()].
function cleat_land_z() = cleat_t_mm - cleat_h_mm;
function cleat_land_y0() = -cleat_t_mm;
function cleat_land_y1() = -cleat_t_mm + cleat_land_mm;
// Lowest point of the strip (its underside meets the wall here), relative to the root.
function cleat_bottom_z() = cleat_land_z() - (cleat_t_mm - cleat_land_mm);
// The bevel's line is z = -y; a hook face offset by the clearance lies on z = -y + this.
function cleat_hook_offset_mm() = cleat_clearance_mm * sqrt(2);

module cleat_profile_2d() {
    t = cleat_t_mm;
    polygon([
        [0, 0],
        [-t, t],
        [-t, cleat_land_z()],
        [cleat_land_y1(), cleat_land_z()],
        [0, cleat_bottom_z()],
    ]);
}

// One strip, x in [0, len]. `wall_overlap_mm` buries the back face into the wall so the
// union fuses; pass the wall thickness or less.
module wall_cleat(len, wall_overlap_mm = 1) {
    assert(len >= 2 * cleat_notch_pitch_mm, "cleat strip shorter than two notch pitches");
    difference() {
        // Profile is drawn in (y, z); extrude along X.
        rotate([90, 0, 90]) linear_extrude(height = len)
            union() {
                cleat_profile_2d();
                // Fuse into the wall.
                translate([0, cleat_bottom_z()])
                    square([wall_overlap_mm, -cleat_bottom_z()]);
            }
        // Key notches: slots cut normal to the top bevel, centred on the pitch grid,
        // leaving half a pitch of plain strip at each end.
        for (x = [cleat_notch_pitch_mm / 2 : cleat_notch_pitch_mm : len - cleat_notch_pitch_mm / 2])
            translate([x - cleat_notch_w_mm / 2, 0, 0])
                rotate([90, 0, 90]) linear_extrude(height = cleat_notch_w_mm)
                    // A band of depth cleat_notch_d_mm under the bevel line z = -y.
                    polygon([
                        [0.01, -0.01],
                        [-cleat_t_mm - 0.01, cleat_t_mm + 0.01],
                        [-cleat_t_mm - 0.01, cleat_t_mm - cleat_notch_d_mm * sqrt(2)],
                        [0.01, -cleat_notch_d_mm * sqrt(2)],
                    ]);
    }
}

// Notch centres along a strip of length len, so carriers can place key ribs on them.
function cleat_notch_xs(len) =
    [for (x = [cleat_notch_pitch_mm / 2 : cleat_notch_pitch_mm : len - cleat_notch_pitch_mm / 2]) x];

