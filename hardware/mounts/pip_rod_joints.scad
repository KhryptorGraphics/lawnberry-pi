// Print-in-place (PIP) rod joint and lock primitives, shared by the throttle pushrod
// (throttle_77603/throttle_pushrod.scad) and the steering pushrods (pushrod.scad).
//
// Self-contained: no include, no top-level variables, no geometry at top level, so callers
// `use <pip_rod_joints.scad>` (root) or `use <../pip_rod_joints.scad>` (a subdirectory). Every
// size is passed in; the mating features (socket, spigot, pocket, thread) are sized by the
// caller through the same spec. Nothing here proves clamp force, holding force, detent
// torque, wear life or vibration resistance: those are coupon and bench tests.
//
// Print-in-place rules followed (checked on the exported mesh by pip_gate.py): every gap
// between two bodies is `gap` (0.4 mm default; 0.45 on helices), every body reaches the bed,
// and no face of one body that overhangs more than 45 degrees lies within 1 mm above another.
//
// -----------------------------------------------------------------------------------------
// 1. LEVER: one design for a cam friction lock and for a positive joint latch.
// -----------------------------------------------------------------------------------------
// A rigid lever turns 90 degrees on two pins that stand inward from two cheeks on the
// housing. Its lobe reaches through a window in the housing wall onto the clamped member:
//   * open (as printed): the lobe's open face is `gap` off the member; the arm sticks out
//     radially and lies on the bed;
//   * closing: the lobe's corner (radius rmax) passes the member about acos(hf/rmax) before
//     closed, then a FLAT (distance hf) lands. hf takes up the member's side play, the
//     pivot's radial play (`gap`) and `preload`; rmax adds `bump` beyond hf. Closed, the lever
//     sits in that over-centre valley and its arm rests on the housing face;
//   * optional TOOTH (tooth_lt > 0): an annular sector under the arm, concentric with the
//     pivot. Closed, it passes through an arc window in the housing wall into a pocket in the
//     member. Its flanks and the pocket's walls are arcs about the pivot, so it swings in
//     without sweeping the pocket and an axial load on the member is radial to the pivot:
//     it cannot cam the lever open. That is a POSITIVE axial lock, not friction. The load
//     path is member -> tooth -> the arc window's wall in the housing (gap 0.4 each side),
//     NOT the pins: the D-section pins leave the lever ~1 mm of play along the rod axis.
// Pins: round at their sides (the clamp reaction is radial and bears there), flat top and
// bottom with 45-degree chamfers, in a round bore 0.4 mm larger; that is what keeps every
// overhang more than 1 mm from the other body as printed.
// Scale for load with `width`, `pin_r`, `tooth_w`, `tooth_lt`, `pocket_d` and the cheeks.
//
// LEVER-LOCAL FRAME (all lever modules): origin on the pivot axis, which is lever-local y.
// +x is the outward normal of the clamped face (the member lies at -x). +z points toward the
// housing's MOUTH, which is the face that prints on the bed, so +z is DOWN as printed.
// rotate([0, phi, 0]) closes the lever: phi = 0 open (as printed), 90 closed.
// pip_at_lever() maps lever-local geometry onto a housing whose mouth is at z = mouth_z and
// faces +Z (mouth_dir = 1) or -Z (mouth_dir = -1), on the side at angle side_deg about Z.
//
// Spec (key/value list; read with pip_get):
//   pip_lever_spec(housing_half, bore_half, member_half, ...) returns the spec. Keys:
//   W B F clr xp ro hf rmax zlo rb arm w gap pin_r pin_bore_r pin_flat pin_x0 pin_len
//   cheek_t cheek_wall roof tooth_lt tooth_w pocket_d td corner_deg
// Modules (lever-local unless stated):
//   pip_lever(s)                 lever body, open pose (tooth included when tooth_lt > 0)
//   pip_lever_tooth(s)           the tooth alone (for withdrawal probes)
//   pip_lever_housing_adds(s)    cheeks + pins: UNION with the housing
//   pip_lever_housing_cuts(s)    clamp window (+ tooth arc window): SUBTRACT from the housing
//   pip_lever_pocket(s)          tooth pocket: SUBTRACT from the member at its SEATED pose
//   pip_at_lever(s, mouth_z, side_deg = 90, mouth_dir = 1) children
//   pip_lever_seat(s, side_deg = 90)  translate a member into its seated pose (children)
//
// -----------------------------------------------------------------------------------------
// 2. DETENTED THUMBWHEEL on a 45-degree-flank thread.
// -----------------------------------------------------------------------------------------
// A wheel prints captive in a cage (two posts whose V ridges run in the wheel's rim groove)
// and screws along a ridge helix on the caller's screw. Two flexure pawls on the posts drop
// into `detents` rim notches: pitch / detents per click. The caller's body over the wheel
// uses pip_wheel_body_profile() (45-degree roof) and must not turn with the wheel.
// Spec: pip_wheel_spec(crest_r, ...). Keys:
//   crest root depth pitch flat b c gap hgap wheel_r wheel_h detents notch_d detent_h pawl_h
//   pawl_t pawl_tooth_deg post_a pawl_a post_r1 ridge_z roof_z0
// Modules:
//   pip_thread_ridge(ws, a1)     external ridge helix, first centre at z = 0, to angle a1
//   pip_wheel(ws)                the wheel as printed (frame: servo face z = 0, axis z)
//   pip_wheel_cage(ws, top_z)    posts, ridges and pawls from the bed up to top_z
// Functions:
//   pip_wheel_body_profile(ws, top_z, bore_r)  (r, z) half-section to rotate_extrude
//   pip_wheel_seat_turn(ws, dz)  wheel turn (deg from print pose) at the nearest detent, when
//                                its servo face is dz above the ridge helix's first centre
//
// -----------------------------------------------------------------------------------------
// 3. SQUARE SOCKET / SPIGOT helpers (axis z, mouth facing +Z at z = m).
// -----------------------------------------------------------------------------------------
//   pip_square(z0, z1, w), pip_square_frustum(z0, w0, z1, w1)
//   pip_socket_boss(m, boss_w, boss_len, tube_w)          outside: boss + 45-degree step
//   pip_socket_cavity(m, socket_w, straight, bore_w)     inside: socket + 45-degree pyramid
//   pip_spigot(m, spigot_w, insert, flare, tube_w)       spigot + 45-degree flare to the tube
//   pip_helical_sweep(profile, pitch, a0, a1, step_deg)  convex [r, z] profile on a helix

// ---- Spec access ----------------------------------------------------------------------
function pip_get(s, k) =
    let(i = search([k], s)[0])
    assert(is_num(i), str("pip spec has no key ", k))
    s[i][1];

// ---- Lever spec -----------------------------------------------------------------------
// housing_half: half-width of the housing's outside at the lever (square section)
// bore_half:    half-width of its bore at the lever
// member_half:  half-width of the clamped member, centred (side play = bore - member)
// pivot_out:    pivot axis beyond the housing face
function pip_lever_spec(housing_half, bore_half, member_half, pivot_out = 3, arm = 30,
                        width = 12, pin_r = 3, gap = 0.4, preload = 0.3, bump = 0.3,
                        cheek_t = 4.5, cheek_wall = 2.6, roof = 1.2, pin_len = 3.5,
                        tooth_lt = 0, tooth_w = 4.5, pocket_d = 2.8, tooth_clear = 0.8) =
    let(W = housing_half, B = bore_half, F = member_half, clr = B - F,
        xp = W + pivot_out,
        ro = xp - F - gap,
        hf = xp - (F - clr) + gap + preload,
        rmax = hf + bump,
        R = pin_r + gap,
        // Pin flats: every bore face overhanging > 45 degrees stays >= 1 mm above the pin,
        // every pin face overhanging > 45 degrees >= 1 mm above the bore's floor.
        pin_flat = R * cos(45) - 1.1,
        pin_x0 = sqrt(R * R - pow(pin_flat + 1.15, 2)) - 0.1,
        td = W - (F - clr - pocket_d + tooth_clear))
    assert(B > F && W > B, "pip lever: need housing > bore > member")
    assert(ro > R + 2, "pip lever: the lobe must wrap its pin bore (raise pivot_out)")
    assert(pin_flat > 0.5 && pin_x0 > 0.5, "pip lever: pin too small for its gap")
    assert(tooth_lt == 0 || tooth_lt + tooth_w / 2 < arm - 2, "pip lever: tooth past the arm")
    assert(tooth_lt == 0 || td > W - B + 1, "pip lever: the tooth must reach the member")
    [["W", W], ["B", B], ["F", F], ["clr", clr], ["xp", xp], ["ro", ro], ["hf", hf],
     ["rmax", rmax], ["zlo", W - xp], ["rb", rmax], ["arm", arm], ["w", width],
     ["gap", gap], ["pin_r", pin_r], ["pin_bore_r", R], ["pin_flat", pin_flat],
     ["pin_x0", pin_x0], ["pin_len", pin_len], ["cheek_t", cheek_t],
     ["cheek_wall", cheek_wall], ["roof", roof], ["tooth_lt", tooth_lt],
     ["tooth_w", tooth_w], ["pocket_d", pocket_d], ["td", td],
     ["corner_deg", acos(hf / rmax)], ["preload", preload], ["bump", bump]];

// ---- Lever: 2D profiles in the lever plane (x, z) ------------------------------------------
module pip_prism(w) {
    rotate([90, 0, 0]) linear_extrude(height = w, center = true) children();
}
module pip_annulus_2d(r1, r2) {
    difference() {
        circle(r = r2, $fn = 240);
        circle(r = r1, $fn = 240);
    }
}
module pip_lever_2d(s) {
    ro = pip_get(s, "ro"); hf = pip_get(s, "hf"); rm = pip_get(s, "rmax");
    zl = pip_get(s, "zlo"); rb = pip_get(s, "rb");
    union() {
        // Lobe: open face at x = -ro, flat at z = -hf (faces the member when closed), corner
        // on the circle rmax between them.
        intersection() {
            circle(r = rm, $fn = 120);
            translate([-ro, -hf]) square([2 * rm, 2 * rm]);
        }
        // Arm: its z = zlo face rests on the housing when closed; its z = rb face is the bed.
        translate([0, zl]) square([pip_get(s, "arm"), rb - zl]);
        // Filler between the lobe and the bed.
        translate([-ro, 0]) square([ro + 0.01, rb]);
    }
}
// The lever's bore: round, so its seat on the pin is round in every pose.
module pip_bore_2d(s) {
    circle(r = pip_get(s, "pin_bore_r"), $fn = 48);
}
// The cheek pin: round at its sides (where the clamp reaction bears when closed), flat top
// and bottom at +/- pin_flat, 45-degree chamfers under it.
module pip_pin_2d(s) {
    r = pip_get(s, "pin_r"); f = pip_get(s, "pin_flat"); x0 = pip_get(s, "pin_x0");
    intersection() {
        circle(r = r, $fn = 48);
        translate([-2 * r, -f]) square([4 * r, 2 * f]);
        polygon([[-x0 - 4 * r, f - 4 * r], [-x0, f], [x0, f], [x0 + 4 * r, f - 4 * r]]);
    }
}
module pip_tooth_2d(s) {
    lt = pip_get(s, "tooth_lt"); w = pip_get(s, "tooth_w"); td = pip_get(s, "td");
    intersection() {
        pip_annulus_2d(lt - w / 2, lt + w / 2);
        translate([0, pip_get(s, "zlo") - td]) square([2 * (lt + w), td + 0.01]);
    }
}

// ---- Lever: solids (lever-local) ---------------------------------------------------------------
module pip_lever(s) {
    difference() {
        union() {
            pip_prism(pip_get(s, "w")) pip_lever_2d(s);
            if (pip_get(s, "tooth_lt") > 0) pip_lever_tooth(s);
        }
        pip_prism(pip_get(s, "w") + 2) pip_bore_2d(s);
    }
}
module pip_lever_tooth(s) {
    pip_prism(pip_get(s, "w")) pip_tooth_2d(s);
}
module pip_lever_housing_adds(s) {
    w = pip_get(s, "w"); g = pip_get(s, "gap"); ct = pip_get(s, "cheek_t");
    rb = pip_get(s, "rb");
    xf = pip_get(s, "W") - pip_get(s, "xp");       // the housing face, lever-local
    y0 = w / 2 + g;
    x0 = xf - 1;
    x1 = pip_get(s, "pin_bore_r") + pip_get(s, "cheek_wall");
    z0 = -x1;
    for (side = [-1, 1]) {
        // Cheek: its bed end rises at 45 degrees from 0.5 mm above the mouth face inside the
        // housing wall, so it is self-supporting and none of its outline lies on the mouth
        // face or on the window's side planes (coincident outlines leave zero-area triangles).
        intersection() {
            translate([x0, side > 0 ? y0 : -y0 - ct, z0]) cube([x1 - x0, ct, rb - z0]);
            translate([0, side * (y0 + ct / 2), 0]) pip_prism(ct + 2)
                polygon([[x0 - 1, z0 - 1], [x1 + 1, z0 - 1],
                         [x1 + 1, rb - 0.5 - (x1 + 1 - x0)], [x0 - 1, rb + 0.5]]);
        }
        pin_y1 = y0 + 0.5;                       // rooted 0.5 mm into the cheek
        pin_y0 = w / 2 - pip_get(s, "pin_len");
        translate([0, side * (pin_y0 + pin_y1) / 2, 0])
            pip_prism(pin_y1 - pin_y0) pip_pin_2d(s);
    }
}
// The tooth's swing shell (an annulus about the pivot, closed side z < 0) widened by the
// gap, between lever-local x0 and x1.
module pip_tooth_shell(s, x0, x1) {
    lt = pip_get(s, "tooth_lt"); w = pip_get(s, "tooth_w"); g = pip_get(s, "gap");
    yw = pip_get(s, "w") / 2 + g;
    intersection() {
        pip_prism(2 * yw) pip_annulus_2d(lt - w / 2 - g, lt + w / 2 + g);
        translate([x0, -yw - 1, -(lt + w + 5)]) cube([x1 - x0, 2 * yw + 2, lt + w + 5]);
    }
}
module pip_lever_housing_cuts(s) {
    g = pip_get(s, "gap"); xp = pip_get(s, "xp");
    yw = pip_get(s, "w") / 2 + g;
    x0 = pip_get(s, "B") - 1 - xp;
    x1 = pip_get(s, "W") + 0.01 - xp;
    ztop = -(pip_get(s, "hf") + pip_get(s, "roof"));
    translate([x0, -yw, ztop]) cube([x1 - x0, 2 * yw, pip_get(s, "rb") + 1 - ztop]);
    if (pip_get(s, "tooth_lt") > 0) pip_tooth_shell(s, x0, x1);
}
// The tooth pocket, in lever-local coordinates of the SEATED member.
module pip_lever_pocket(s) {
    pip_tooth_shell(s, pip_get(s, "F") - pip_get(s, "clr") - pip_get(s, "pocket_d")
                       - pip_get(s, "xp"),
                    pip_get(s, "W") - pip_get(s, "xp"));
}
module pip_at_lever(s, mouth_z, side_deg = 90, mouth_dir = 1) {
    translate([0, 0, mouth_z]) rotate(mouth_dir > 0 ? [0, 0, 0] : [180, 0, 0])
        rotate([0, 0, side_deg]) translate([pip_get(s, "xp"), 0, -pip_get(s, "rb")])
            children();
}
// The closed lever pushes its member across its side play, away from the lever.
module pip_lever_seat(s, side_deg = 90) {
    translate(-pip_get(s, "clr") * [cos(side_deg), sin(side_deg), 0]) children();
}

// ---- Helix ------------------------------------------------------------------------------------
// Sweeps a convex [r, z] profile (counter-clockwise in the r-z plane, z about the helix centre)
// along a right-hand helix of `pitch` from angle a0 to a1 (degrees; the centre rises
// pitch * a / 360). One polyhedron: every side triangle contains a profile edge, so no facet
// is steeper than the profile flank it approximates.
module pip_helical_sweep(profile, pitch, a0, a1, step_deg = 5) {
    n = max(1, ceil((a1 - a0) / step_deg));
    m = len(profile);
    pts = [for (i = [0:n])
               let(a = a0 + (a1 - a0) * i / n, rise = pitch * a / 360)
               for (p = profile) [p[0] * cos(a), p[0] * sin(a), p[1] + rise]];
    sides = [for (i = [0:n - 1]) for (k = [0:m - 1])
                 let(k1 = (k + 1) % m, A = i * m + k, B = i * m + k1,
                     C = (i + 1) * m + k1, D = (i + 1) * m + k)
                 each [[A, B, C], [A, C, D]]];
    caps = [[for (k = [m - 1:-1:0]) k], [for (k = [0:m - 1]) n * m + k]];
    polyhedron(points = pts, faces = concat(sides, caps), convexity = 4);
}

// ---- Thumbwheel spec ---------------------------------------------------------------------------
// crest_r: the screw thread's crest radius (the screw passes its housing's bore with this).
function pip_wheel_spec(crest_r, depth = 1.35, pitch = 4, wheel_r = 16, wheel_h = 12,
                        detents = 16, notch_d = 1.2, detent_h = 4.5, gap = 0.4, hgap = 0.45,
                        pawl_t = 1.6, pawl_tooth_deg = 56, post_a = [85, 130],
                        pawl_a = [52, 85.5], cage_wall = 4, ridge_z = 8.5) =
    let(flat = pitch / 2 - depth, b = (pitch - flat) / 2)
    assert(flat >= 0.6, "pip wheel: thread flats too thin to print")
    assert(wheel_h >= 3 * pitch, "pip wheel: engage at least three pitches")
    assert(wheel_r - notch_d - 1.5 > crest_r + hgap + 2, "pip wheel: rim wall too thin")
    assert(detents % 2 == 0, "pip wheel: both pawls (180 degrees apart) must seat at once")
    [["crest", crest_r], ["root", crest_r - depth], ["depth", depth], ["pitch", pitch],
     ["flat", flat], ["b", b], ["c", b - depth], ["gap", gap], ["hgap", hgap],
     ["wheel_r", wheel_r], ["wheel_h", wheel_h], ["detents", detents],
     ["notch_d", notch_d], ["detent_h", detent_h], ["pawl_h", detent_h - 1.2],
     ["pawl_t", pawl_t], ["pawl_tooth_deg", pawl_tooth_deg], ["post_a", post_a],
     ["pawl_a", pawl_a], ["post_r1", wheel_r + gap + cage_wall], ["ridge_z", ridge_z],
     ["roof_z0", wheel_h + gap], ["mm_per_click", pitch / detents]];

function pip_ridge_profile(ws) =
    let(r0 = pip_get(ws, "root") - 0.1, b = pip_get(ws, "b") + 0.1, c = pip_get(ws, "c"),
        cr = pip_get(ws, "crest"))
    [[r0, -b], [cr, -c], [cr, c], [r0, b]];
// The wheel's groove: the ridge offset by hgap normal to every face (mitred), cut off just
// inside the wheel's bore so successive turns never overlap.
function pip_groove_profile(ws) =
    let(g = pip_get(ws, "hgap"), cr = pip_get(ws, "crest"), c = pip_get(ws, "c"),
        ri = pip_get(ws, "root") + g - 0.08, rc = cr + g,
        zc = c + g * (sqrt(2) - 1), zi = c + (cr - ri) + g * sqrt(2))
    [[ri, -zi], [rc, -zc], [rc, zc], [ri, zi]];
function pip_wheel_seat_turn(ws, dz) =
    let(t = 360 * dz / pip_get(ws, "pitch"), step = 360 / pip_get(ws, "detents"))
    round(t / step) * step;
// The caller's body over the wheel: bore, 45-degree roof (lowest just outside the rim),
// outside at post_r1, flat top at top_z.
function pip_wheel_body_profile(ws, top_z, bore_r) =
    let(r = pip_get(ws, "wheel_r") + pip_get(ws, "gap"), z0 = pip_get(ws, "roof_z0"),
        r1 = pip_get(ws, "post_r1"))
    [[bore_r, z0 + r - bore_r], [r, z0], [r1, z0 + r1 - r], [r1, top_z], [bore_r, top_z]];

module pip_thread_ridge(ws, a1) {
    pip_helical_sweep(pip_ridge_profile(ws), pip_get(ws, "pitch"), 0, a1);
}
module pip_notch_2d(ws) {
    d = pip_get(ws, "notch_d");
    polygon([[-d, 0], [1.5, d + 1.5], [1.5, -d - 1.5]]);
}
module pip_capture_ridge_2d(ws) {
    r = pip_get(ws, "wheel_r"); z = pip_get(ws, "ridge_z");
    polygon([[r - 1.1, z - 0.3], [r + 1.0, z - 2.4], [r + 1.0, z + 2.4], [r - 1.1, z + 0.3]]);
}
// The wheel as printed: notches aligned with the pawl teeth.
module pip_wheel(ws) {
    r = pip_get(ws, "wheel_r"); h = pip_get(ws, "wheel_h"); dh = pip_get(ws, "detent_h");
    n = pip_get(ws, "detents"); p = pip_get(ws, "pitch");
    difference() {
        union() {
            linear_extrude(height = dh)
                difference() {
                    circle(r = r, $fn = 160);
                    for (k = [0:n - 1])
                        rotate([0, 0, pip_get(ws, "pawl_tooth_deg") + k * 360 / n])
                            translate([r, 0]) pip_notch_2d(ws);
                }
            translate([0, 0, dh - 0.01]) cylinder(r = r, h = h - dh + 0.01, $fn = 160);
        }
        // 72 facets offset half a helix step: no bore vertex lies on the groove's 5-degree
        // profile planes (coincident ones leave zero-area triangles).
        translate([0, 0, -1]) rotate([0, 0, 2.5])
            cylinder(r = pip_get(ws, "root") + pip_get(ws, "hgap"), h = h + 2, $fn = 72);
        pip_helical_sweep(pip_groove_profile(ws), p, -360, (h + 4) / p * 360);
        rotate_extrude($fn = 160) offset(delta = pip_get(ws, "gap")) pip_capture_ridge_2d(ws);
    }
}
module pip_sector(r0, r1, a0, a1, z0, z1) {
    translate([0, 0, z0]) rotate([0, 0, a0])
        rotate_extrude(angle = a1 - a0, $fn = 180) translate([r0, 0]) square([r1 - r0, z1 - z0]);
}
// Two posts (180 degrees apart) from the bed to top_z, each with a V ridge in the wheel's
// rim groove and a free flexure pawl whose tooth sits in a notch with the gap all round.
module pip_wheel_cage(ws, top_z) {
    r_in = pip_get(ws, "wheel_r") + pip_get(ws, "gap");
    pa = pip_get(ws, "post_a"); wa = pip_get(ws, "pawl_a");
    for (s = [0, 180]) rotate([0, 0, s]) {
        pip_sector(r_in, pip_get(ws, "post_r1") - 0.4, pa[0], pa[1], 0, top_z);
        rotate([0, 0, pa[0] + 0.5])
            rotate_extrude(angle = pa[1] - pa[0] - 1, $fn = 180) pip_capture_ridge_2d(ws);
        pip_sector(r_in, r_in + pip_get(ws, "pawl_t"), wa[0], wa[1], 0, pip_get(ws, "pawl_h"));
        linear_extrude(height = pip_get(ws, "pawl_h"))
            intersection() {
                rotate([0, 0, pip_get(ws, "pawl_tooth_deg")])
                    translate([pip_get(ws, "wheel_r"), 0])
                        offset(delta = -pip_get(ws, "gap")) pip_notch_2d(ws);
                circle(r = r_in + pip_get(ws, "pawl_t") - 0.1, $fn = 160);
            }
    }
}

// ---- Square socket / spigot -------------------------------------------------------------------
module pip_square(z0, z1, w) {
    translate([-w / 2, -w / 2, z0]) cube([w, w, z1 - z0]);
}
module pip_square_frustum(z0, w0, z1, w1) {
    hull() {
        translate([0, 0, z0]) linear_extrude(height = 0.01) square(w0, center = true);
        translate([0, 0, z1 - 0.01]) linear_extrude(height = 0.01) square(w1, center = true);
    }
}
// Boss outside: boss_w square from m - boss_len to the mouth m, then a 45-degree step down to
// the tube (self-supporting with the mouth on the bed).
module pip_socket_boss(m, boss_w, boss_len, tube_w) {
    pip_square(m - boss_len, m, boss_w);
    pip_square_frustum(m - boss_len - (boss_w - tube_w) / 2, tube_w, m - boss_len + 0.01, boss_w);
}
// Socket inside: socket_w square from the mouth to depth `straight`, then a 45-degree pyramid
// down to the bore (the spigot's end corner seats on it).
module pip_socket_cavity(m, socket_w, straight, bore_w) {
    pip_square(m - straight, m + 1, socket_w);
    pip_square_frustum(m - straight - (socket_w - bore_w) / 2, bore_w, m - straight + 0.01, socket_w);
}
// Spigot: spigot_w square from m - insert to the mouth, then a 45-degree flare to the tube.
module pip_spigot(m, spigot_w, insert, flare, tube_w) {
    pip_square(m - insert, m, spigot_w);
    pip_square_frustum(m - 0.01, spigot_w, m + flare, tube_w);
}
