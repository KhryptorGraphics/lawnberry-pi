// Bottom-loaded, opaque lens-only camera housings. FDM ASA; gasket parts use TPU/foam.
// VERIFIED: Pi2 drawing PCB23.862x25, R2 corners, lens housing8.5 square,
// lens centre14.4/12.5, M2 holes x2/14.5,y2/23. SVPRO dimensional image PCB80x16.5,
// lens bases14x13 and USB-C width9; its illustrated16mm height has an ambiguous datum.
// Barrel diameters, PCB thicknesses, lens projection and stereo baseline are UNKNOWN.
// Defaults below are fitting capacities, never a claim of measured camera fit.
include <enclosure_common.scad>
use <camera_tower.scad>
camera_part = "pi_housing";
camera_hinge_gap = 8.6;             // sensor foot interface, unchanged
camera_wall_mm = 3;
camera_front_y_mm = -105;
camera_front_wall_mm = 8;
camera_rear_y_mm = -36;
camera_tray_t_mm = 4;
camera_fit_mm = 0.4;
camera_face_y_mm = -108;
camera_face_t_mm = 2.6;
camera_face_h_mm = 77;
camera_key_h_mm = 62;
camera_insert_y_mm = -110.4;
camera_insert_t_mm = 2;
camera_pcb_front_y_mm = -96;
camera_groove_mm = 3.2;            // maximum PCB+pad stack, NOT measured PCB thickness
camera_edge_engagement_mm = 0.8;   // verify these narrow lands are component-free on actual PCB
camera_insert_offset_x_mm = 0;     // coarse indexed tile setting, selected using actual camera
camera_insert_offset_z_mm = 0;
camera_insert_bore_mm = 0;         // 0 selects prototype default; choose with the try-fit gauge
camera_stereo_lens_offsets_mm = [[0,0],[0,0]]; // independently selected X/Z coarse settings, not baseline data
camera_pi_bore_capacity_mm = 8;
camera_stereo_bore_capacity_mm = 14;

function camera_shell_width(kind) = kind == "pi" ? 90 : 180;
function camera_bottom_screw_points(kind) =
    [for (x = [-1,1], y = [-93,-45]) [x*(camera_shell_width(kind)/2+5),y]];
// Offset Pi fixings around the two upright PCB guides: nuts/washers sit on the
// foot, not through its edge-rail at x≈-16 or x≈11.
function camera_carrier_screw_points(kind) =
    kind == "pi" ? [[-10,-67],[17,-67]] : [[-65,-65],[65,-65]];
function camera_throat_mm() = tower_od-2*tower_wall-2*tower_spigot_clear-7;
function camera_board_outline_mm(kind) = kind == "pi" ? [23.862,25] : [80,16.5];
function camera_board_bounds_xz(kind) = kind == "pi" ? [-14.4,9.462,27.5,52.5] : [-40,40,31.75,48.25];
function camera_board_front_y_mm(kind="pi",y_shift=0) = camera_pcb_front_y_mm+y_shift;
function camera_carrier_y_travel_mm(kind="pi") = kind == "pi" ? [-11.8,10] : [-8,10];
function camera_pcb_thickness_capacity_mm() = [0.6,3.0];
function camera_insert_front_y_mm(kind) = kind == "pi" ? -110 : camera_insert_y_mm;
function camera_insert_thickness_mm(kind) = kind == "pi" ? 1.6 : camera_insert_t_mm;
function camera_pi_lens_base_relief_side_mm(bore=0) =
    max(9.3,(bore > 0 ? bore : camera_insert_bore_d_mm("pi"))+1.3);
function camera_pi_lens_base_relief_depth_mm() = 0.9;
function camera_lens_rim_seal_y_mm(kind) = kind == "pi" ? -109.3 : -108.4;
function camera_lens_rim_seal_t_mm(kind) = kind == "pi" ? 0.3 : 0.8;
function camera_lens_front_plane_mm(kind="pi") = kind == "pi" ? -110 : -111;
function camera_lens_projection_capacity_mm(kind="pi") = kind == "pi" ? [2.2,24] : [7,25];
function camera_groove_front_lip_mm(kind) = kind == "pi" ? 0.4 : 1.2;
function camera_key_width_mm(kind) = kind == "pi" ? 54 : 104;
function camera_key_top_mm() = camera_key_h_mm;
function camera_fastener_key_top_mm() = 73;
function camera_face_top_mm() = camera_face_h_mm;
function camera_front_seal_gap_mm() = 0.4;
function camera_insert_seal_gap_mm() = 0.4;
function camera_insert_centres(kind) = kind == "pi" ? [0] : [-25,25];
function camera_insert_width_mm(kind) = kind == "pi" ? 52 : 46;
function camera_insert_height_mm() = 74;
function camera_insert_x_travel_mm(kind) = kind == "pi" ? [-2,2] : [-1,1];
function camera_insert_z_travel_mm() = [-3,3];
function camera_insert_bore_d_mm(kind) = camera_insert_bore_mm > 0 ? camera_insert_bore_mm :
    (kind == "pi" ? camera_pi_bore_capacity_mm : camera_stereo_bore_capacity_mm);
function camera_bore_choices_mm(kind) = kind == "pi" ? [5,6,7,8,9,10,11,12] : [8,10,12,14,16,18,20];
function camera_fov_capacity_deg(kind) = kind == "pi" ? 110 : 85; // design cones, not calibrated optics
function camera_insert_flare_mm(kind) = camera_insert_thickness_mm(kind)*tan(camera_fov_capacity_deg(kind)/2);
function camera_port_windows_xz(kind) = kind == "pi" ? [[-22,22,20,60]] :
    [[-43,-6,24,56],[6,43,24,56]];
function camera_insert_screw_points(kind,index=0) =
    [for (z = [13,67]) [camera_insert_centres(kind)[index],z]];
// Coarse selection + fine screw travel is checked against the actual port window.
// Stereo holes remain separate and every possible setting retains an opaque ligament.
function camera_lens_centre_capacity_x_mm(kind,index=0,bore=0) =
    let(d=bore > 0 ? bore : camera_insert_bore_d_mm(kind),
        r=d/2+camera_insert_flare_mm(kind), b=camera_port_windows_xz(kind)[index])
    [b[0]+r+0.5,b[1]-r-0.5];
function camera_lens_centre_capacity_z_mm(kind,bore=0) =
    let(d=bore > 0 ? bore : camera_insert_bore_d_mm(kind), r=d/2+camera_insert_flare_mm(kind),
        b=camera_port_windows_xz(kind)[0])
    [b[2]+r+0.5,b[3]-r-0.5];
function camera_lens_offset_xz_mm(kind,index=0) = kind == "pi" ?
    [camera_insert_offset_x_mm,camera_insert_offset_z_mm] : camera_stereo_lens_offsets_mm[index];
function camera_lens_centres_xz(kind) =
    [for (i = [0:len(camera_insert_centres(kind))-1])
        let(o=camera_lens_offset_xz_mm(kind,i)) [camera_insert_centres(kind)[i]+o[0],40+o[1]]];
function camera_insert_washer_od_mm() = 20;
function camera_groove_spans_x_mm(kind,upper=false) = kind == "pi" ? [[-13.6,3]] :
    (upper ? [[-39.2,-32],[32,39.2]] : [[-39.2,39.2]]);
function camera_board_retainer_x_mm(kind) = camera_board_bounds_xz(kind)[1]+2.5;
function camera_station_part_count(kind) = 6+3*len(camera_insert_centres(kind));

assert(camera_throat_mm() >= 30,"Mast throat too small for connector route");

// One shared bottom-open cut for shell and face seal. Upper insert nuts also travel in an open key.
module camera_bottom_key_cuts(kind,y=camera_front_y_mm-1,depth=camera_front_wall_mm+2) {
    translate([-camera_key_width_mm(kind)/2,y,-1])
        cube([camera_key_width_mm(kind),depth,camera_key_h_mm+1]);
    for (x = camera_insert_centres(kind))
        translate([x-6,y,-1]) cube([12,depth,camera_fastener_key_top_mm()+1]);
}
module camera_shell(kind = "pi") {
    w = camera_shell_width(kind);
    throat = camera_throat_mm();
    assert(kind == "pi" || kind == "stereo","Unknown camera housing kind");
    difference() {
        union() {
            // Keep the structural spine/flanges/socket/cable turn independent of the service floor.
            translate([-tower_od/2,-tower_od/2,0]) cube([tower_od,tower_od,camera_housing_h]);
            tower_joint_flange();
            translate([0,0,camera_housing_h-tower_flange_t]) tower_joint_flange();
            translate([0,0,camera_housing_h]) tower_spigot();
            translate([-w/2,camera_front_y_mm,0]) cube([w,camera_front_wall_mm,camera_housing_h]);
            for (x = [-w/2,w/2-camera_wall_mm])
                translate([x,camera_front_y_mm,0]) cube([camera_wall_mm,69,camera_housing_h]);
            translate([-w/2,camera_rear_y_mm-camera_wall_mm,0])
                cube([w,camera_wall_mm+0.1,camera_housing_h]);
            translate([-25,camera_rear_y_mm,0]) cube([50,11.1,camera_housing_h]);
            translate([-w/2,camera_front_y_mm,camera_housing_h-camera_wall_mm])
                cube([w,69,camera_wall_mm]);
            for (p = camera_bottom_screw_points(kind))
                translate([p[0]-6,p[1]-6,0]) cube([12,12,12]);
        }
        translate([-throat/2,-throat/2,-1]) cube([throat,throat,camera_housing_h+17]);
        translate([-20,-20,-1]) cube([40,40,tower_spigot_len+2]);
        translate([-15,-40,15]) cube([30,57,50]);
        for (x = [-1,1], y = [-1,1])
            translate([x*tower_flange_bolt,y*tower_flange_bolt,-1])
                cylinder(d=m4_clear,h=camera_housing_h+2);
        for (x = [-1,1], y = [-1,1])
            translate([x*tower_flange_bolt,y*tower_flange_bolt,tower_flange_t])
                cylinder(d=8.6,h=camera_housing_h-2*tower_flange_t);
        // OPEN TO BOTTOM: protruding lenses and edge-guide arms lift without hitting a closed hole.
        camera_bottom_key_cuts(kind);
        for (p = camera_bottom_screw_points(kind)) {
            translate([p[0],p[1],-1]) cylinder(d=m3_clear,h=14);
            translate([p[0]>0 ? p[0]-3.4 : p[0]-7,p[1]-2.9,6]) cube([10.4,5.8,2.8]);
        }
    }
}

module camera_face_window_cuts(kind,depth=6,y=camera_face_y_mm-1) {
    for (b = camera_port_windows_xz(kind))
        translate([b[0],y,b[2]]) cube([b[1]-b[0],depth,b[3]-b[2]]);
}
module camera_insert_fastener_cuts(kind,y,depth) {
    for (i = [0:len(camera_insert_centres(kind))-1],p = camera_insert_screw_points(kind,i))
        translate([p[0],y,p[1]]) rotate([-90,0,0]) cylinder(d=m3_clear,h=depth);
}
module camera_bottom_tray(kind = "pi") {
    w = camera_shell_width(kind);
    rim_w = w-2*camera_wall_mm-2*camera_fit_mm;
    difference() {
        union() {
            // Opaque front is integral to the removable floor; camera and closing face move together.
            translate([-w/2-12,camera_face_y_mm,-camera_tray_t_mm])
                cube([w+24,camera_rear_y_mm-camera_face_y_mm,camera_tray_t_mm]);
            translate([-w/2,camera_face_y_mm,-0.1]) cube([w,camera_face_t_mm,camera_face_h_mm+0.1]);
            difference() {
                translate([-rim_w/2,-96.6,0]) cube([rim_w,57.2,6]);
                translate([-rim_w/2+2,-94.6,-1]) cube([rim_w-4,53.2,8]);
                // The guide arms/forward PCB need the same unobstructed bottom loading key.
                translate([-camera_key_width_mm(kind)/2,-98,-1]) cube([camera_key_width_mm(kind),8,8]);
            }
        }
        camera_face_window_cuts(kind);
        camera_insert_fastener_cuts(kind,camera_face_y_mm-1,6);
        camera_floor_gasket(kind);
        for (p = camera_bottom_screw_points(kind))
            translate([p[0],p[1],-5]) cylinder(d=m3_clear,h=7);
        for (p = camera_carrier_screw_points(kind))
            translate([p[0],p[1],-5]) cylinder(d=m3_clear,h=7);
    }
}

// Horizontal edge grooves: slide PCB from +X toward the fixed left stop on the BENCH.
// Never bend a populated board over a snap lip. Right cap is screwed on afterwards.
// Pi CSI right edge is open; stereo upper centre is open for its9mm USB-C connector.
module camera_edge_groove(kind,upper=false) {
    b = camera_board_bounds_xz(kind);
    z = upper ? b[3]-camera_edge_engagement_mm : b[2]-2.5;
    gh = 2.5+camera_edge_engagement_mm;
    for (s = camera_groove_spans_x_mm(kind,upper)) difference() {
        translate([s[0],camera_pcb_front_y_mm-camera_groove_front_lip_mm(kind),z])
            cube([s[1]-s[0],camera_groove_mm+1.2+camera_groove_front_lip_mm(kind),gh]);
        translate([s[0]-0.1,camera_pcb_front_y_mm,z+(upper ? -0.1 : 2.5)])
            cube([s[1]-s[0]+0.2,camera_groove_mm,camera_edge_engagement_mm+0.2]);
    }
}
module camera_groove_cassette(kind) {
    b = camera_board_bounds_xz(kind);
    beam_w = kind == "pi" ? 46 : 146;
    difference() {
        union() {
            // Keep the stereo floor clear of the tray's rear sealing rim at +10 mm
            // adjustment; its bolt slots retain closed ends at both travel limits.
            translate([-beam_w/2,kind == "pi" ? -80 : -78,0])
                cube([beam_w,kind == "pi" ? 28 : 26,6]);
            // Pi lower arms stay behind the opaque face, while raised guides enter its covered relief.
            for (x = [b[0]-3,b[1]+0.8])
                translate([x,camera_pcb_front_y_mm+(kind == "pi" ? 2.8 : -1.2),0])
                    cube([2.2,kind == "pi" ? 41.2 : 45.2,b[2]-1]);
            // Pi CSI exits the middle of its right edge: reserve12mm behind that edge, no lip to fold over.
            translate([b[1]+0.8,camera_pcb_front_y_mm+(kind == "pi" ? 12 : camera_groove_mm),b[2]-5])
                cube([2.2,1.2,b[3]-b[2]+10]);
            // Rear corner webs join partial edge bars to the uprights, away from the PCB faces.
            translate([b[0]-3,camera_pcb_front_y_mm+camera_groove_mm,b[2]-2.5])
                cube([b[1]-b[0]+6,1.2,2.5]);
            // Stereo's24mm upper-centre gap also clears the USB-C connector's rear-side projection.
            for (s = kind == "pi" ? [[b[0]-3,b[1]+3]] : [[b[0]-3,-12],[12,b[1]+3]])
                translate([s[0],camera_pcb_front_y_mm+camera_groove_mm,b[3]])
                    cube([s[1]-s[0],1.2,2.5]);
            camera_edge_groove(kind);
            camera_edge_groove(kind,true);
            // Left end-stop and right retainer screw lands lie outside the verified board outline.
            translate([b[0]-3,camera_pcb_front_y_mm-camera_groove_front_lip_mm(kind),b[2]-5])
                cube([3,camera_groove_mm+1.2+camera_groove_front_lip_mm(kind),b[3]-b[2]+10]);
            for (z = [b[2]-2.5,b[3]+2.5])
                translate([b[1],camera_pcb_front_y_mm-camera_groove_front_lip_mm(kind),z-2.5])
                    cube([5,(kind == "pi" ? 13.2 : camera_groove_mm+1.2)+camera_groove_front_lip_mm(kind),5]);
        }
        for (p = camera_carrier_screw_points(kind))
            translate([p[0],p[1]+1,-1])
                rotate([0,0,90]) slot(m3_clear,kind == "pi" ? 22 : 18,8);
        for (z = [b[2]-2.5,b[3]+2.5])
            translate([camera_board_retainer_x_mm(kind),camera_pcb_front_y_mm-2,z])
                rotate([-90,0,0]) cylinder(d=2.4,h=kind == "pi" ? 21 : 9);
    }
}
module camera_pi_carrier() { camera_groove_cassette("pi"); }
module camera_stereo_carrier() { camera_groove_cassette("stereo"); }
module camera_board_retainer(kind="pi") {
    b = camera_board_bounds_xz(kind);
    difference() {
        union() {
            // Rear screw plate avoids shallow Pi optics; Pi bridge leaves its middle CSI edge unobstructed.
            translate([b[1]+0.15,camera_pcb_front_y_mm+(kind == "pi" ? 14 : 4.8),b[2]-5])
                cube([5,2,b[3]-b[2]+10]);
            if (kind == "pi") {
                for (z = [b[2]-5,b[3]-0.8])
                    translate([b[1]+0.15,camera_pcb_front_y_mm+4.8,z])
                        cube([5,9.3,5.8]);
                for (z = [b[2]+0.1,b[3]-0.8])
                    translate([b[1]+0.15,camera_pcb_front_y_mm,z])
                        cube([0.6,4.9,0.7]);
            } else translate([b[1]+0.15,camera_pcb_front_y_mm,b[2]+0.1])
                cube([0.6,4.9,b[3]-b[2]-0.2]);
        }
        for (z = [b[2]-2.5,b[3]+2.5])
            translate([camera_board_retainer_x_mm(kind),camera_pcb_front_y_mm-1,z])
                rotate([-90,0,0]) cylinder(d=2.4,h=kind == "pi" ? 20 : 9);
    }
}
module camera_carrier_assembly(kind = "pi",y_shift=0) {
    tr = camera_carrier_y_travel_mm(kind);
    assert(y_shift >= tr[0] && y_shift <= tr[1],"Carrier fore/aft seating exceeds slot capacity");
    translate([0,y_shift,0]) {
        if (kind == "pi") camera_pi_carrier(); else camera_stereo_carrier();
        camera_board_retainer(kind);
    }
}

// Separate opaque lens inserts install along +Y from the FRONT after bottom loading.
// They do NOT sweep a closed bore over a protruding lens. Loosen M3s to align using the camera jig.
// Choose a coarse off-centre variant, then fine X/Z slot adjustment. No fixed stereo baseline.
module camera_lens_insert(kind="pi",index=0,bore=0,dx=undef,dz=undef) {
    cx = camera_insert_centres(kind)[index];
    d = bore > 0 ? bore : camera_insert_bore_d_mm(kind);
    w = camera_insert_width_mm(kind);
    xr = camera_lens_centre_capacity_x_mm(kind,index,d);
    zr = camera_lens_centre_capacity_z_mm(kind,d);
    travel = camera_insert_x_travel_mm(kind)[1];
    offset = camera_lens_offset_xz_mm(kind,index);
    ox = is_undef(dx) ? offset[0] : dx;
    oz = is_undef(dz) ? offset[1] : dz;
    assert(d >= min(camera_bore_choices_mm(kind)) && d <= max(camera_bore_choices_mm(kind)),
           "Bore outside try-fit capacity; do not claim unknown camera dimensions");
    assert(cx+ox-travel >= xr[0] && cx+ox+travel <= xr[1],"Lens bore/flare exceeds covered X capacity");
    assert(40+oz-3 >= zr[0] && 40+oz+3 <= zr[1],"Lens bore/flare exceeds covered Z capacity");
    difference() {
        translate([cx-w/2,camera_insert_front_y_mm(kind),3])
            cube([w,camera_insert_thickness_mm(kind),camera_insert_height_mm()]);
        // Rear bore fits the non-optical rim; the outward flare leaves optical access.
        // Overrun prevents a near-tangent print-face sliver (Pi STL triangle 572).
        translate([cx+ox,camera_insert_front_y_mm(kind)-0.25,40+oz]) rotate([-90,0,0])
            cylinder(d1=d+2*camera_insert_flare_mm(kind)+0.3,d2=d,
                     h=camera_insert_thickness_mm(kind)+0.5,$fn=90);
        // Blind REAR recess clears the verified8.5-square Pi lens base at shallow seating.
        // It stops0.7mm behind the opaque front; the exterior remains ONE round lens-only opening.
        if (kind == "pi") {
            side = camera_pi_lens_base_relief_side_mm(d);
            translate([cx+ox-side/2,-109.3,40+oz-side/2]) cube([side,1,side]);
        }
        // Seal each full slot with20mm OD metal+compliant washers;10mm washers expose extreme trim.
        for (p = camera_insert_screw_points(kind,index))
            hull() for (x = [-travel,travel],z = [-3,3])
                translate([p[0]+x,camera_insert_front_y_mm(kind)-1,p[1]+z]) rotate([-90,0,0])
                    cylinder(d=m3_clear,h=4);
    }
}
module camera_lens_inserts_assembly(kind="pi") {
    for (i = [0:len(camera_insert_centres(kind))-1]) camera_lens_insert(kind,i);
}
// Continuous interface gaskets are printable TPU solids or cutting templates for closed-cell foam.
// 0.4mm installed gaps require thicker compliant stock, not an asserted weather/IP rating.
module camera_floor_gasket(kind="pi") {
    w = camera_shell_width(kind);
    // Compressed0.8mm floor perimeter sits in the tray recess without moving mast datums.
    difference() {
        translate([-w/2,camera_front_y_mm,-0.8]) cube([w,69,0.8]);
        translate([-w/2+camera_wall_mm,camera_front_y_mm+camera_front_wall_mm,-1])
            cube([w-2*camera_wall_mm,58,2]);
        translate([-camera_key_width_mm(kind)/2,camera_front_y_mm-1,-1])
            cube([camera_key_width_mm(kind),camera_front_wall_mm+2,2]);
    }
}
module camera_face_gasket(kind="pi") {
    w = camera_shell_width(kind);
    difference() {
        translate([-w/2,camera_front_y_mm-0.4,0]) cube([w,0.4,camera_face_h_mm]);
        camera_bottom_key_cuts(kind,camera_front_y_mm-1,2);
    }
}
module camera_insert_gasket(kind="pi",index=0) {
    cx = camera_insert_centres(kind)[index];
    w = camera_insert_width_mm(kind);
    difference() {
        translate([cx-w/2,camera_face_y_mm-0.4,3]) cube([w,0.4,camera_insert_height_mm()]);
        camera_face_window_cuts(kind,3,camera_face_y_mm-1);
        camera_insert_fastener_cuts(kind,camera_face_y_mm-1,3);
    }
}
module camera_seals_assembly(kind="pi") {
    camera_floor_gasket(kind);
    camera_face_gasket(kind);
    for (i = [0:len(camera_insert_centres(kind))-1]) {
        camera_insert_gasket(kind,i);
        camera_lens_rim_gasket(kind,i);
    }
}
module camera_lens_rim_gasket(kind="pi",index=0,bore=0) {
    d = bore > 0 ? bore : camera_insert_bore_d_mm(kind);
    p = camera_lens_centres_xz(kind)[index];
    // Ring fits behind the insert, over the actual non-optical barrel rim. Do not cover glass/pupil.
    difference() {
        translate([p[0],camera_lens_rim_seal_y_mm(kind),p[1]]) rotate([-90,0,0])
            cylinder(d=kind == "pi" ? camera_pi_lens_base_relief_side_mm(d) : d+3,
                     h=camera_lens_rim_seal_t_mm(kind));
        translate([p[0],camera_lens_rim_seal_y_mm(kind)-0.1,p[1]]) rotate([-90,0,0])
            cylinder(d=d-0.6,h=camera_lens_rim_seal_t_mm(kind)+0.2);
    }
}

// Try-fit barrel gauge: use the real non-optical lens rim, never press on glass.
// Pick the smallest loose bore, then use opaque tile variants for optical centre alignment.
module camera_lens_fit_gauge(kind="pi") {
    ds = camera_bore_choices_mm(kind);
    pitch = 28;
    difference() {
        cube([len(ds)*pitch,30,2]);
        for (i = [0:len(ds)-1]) translate([i*pitch+14,15,-1]) cylinder(d=ds[i],h=4);
    }
    for (i = [0:len(ds)-1]) translate([i*pitch+4,2,2])
        linear_extrude(0.4) text(str(ds[i]),size=3);
}
module camera_insert_print(kind="pi",index=0,bore=0,dx=undef,dz=undef) {
    translate([0,0,-camera_insert_front_y_mm(kind)]) rotate([90,0,0])
        camera_lens_insert(kind,index,bore,dx,dz);
}
// Kit tiles/strips are joined by thin breakaway sprues so each kit is ONE printable
// solid (one STL/3MF, one bed placement). Snap pieces off and deburr before use.
function camera_kit_tile_hw_mm(kind) = kind == "pi" ? 26 : 23;
camera_kit_tile_ylo_mm = -77;      // tile local bed-Y extent [-77,-3]
module camera_lens_tile_fit_kit(kind="pi") {
    // Actual camera is the fitting jig; indexed X/Z variants need no caliper-derived baseline.
    xr = camera_lens_centre_capacity_x_mm(kind,0);
    zr = camera_lens_centre_capacity_z_mm(kind);
    cx = camera_insert_centres(kind)[0];
    tr = camera_insert_x_travel_mm(kind)[1];
    hw = camera_kit_tile_hw_mm(kind);
    ylo = camera_kit_tile_ylo_mm;
    function ok(i,j) = cx+2*(i-3)-tr >= xr[0] && cx+2*(i-3)+tr <= xr[1] &&
                       40+2*(j-2)-3 >= zr[0] && 40+2*(j-2)+3 <= zr[1];
    for (i = [0:6],j = [0:4]) if (ok(i,j)) translate([i*58,j*80,0])
        // Label ENGRAVED in a tile corner so it can never float off as its own body.
        difference() {
            translate([-cx,0,0]) camera_insert_print(kind,0,0,2*(i-3),2*(j-2));
            translate([-hw+3,ylo+3,camera_insert_thickness_mm(kind)-0.4]) linear_extrude(1)
                text(str(2*(i-3),",",2*(j-2)),size=3.5);
        }
    // Present tiles form a rectangular subgrid (independent i/j filters), so bridging
    // each tile to its left and lower neighbour connects the whole kit.
    for (i = [0:6],j = [0:4]) if (ok(i,j)) {
        if (i > 0 && ok(i-1,j))
            translate([(i-1)*58+hw-1,j*80+ylo+30,0]) cube([58-2*hw+2,14,1.2]);
        if (j > 0 && ok(i,j-1))
            translate([i*58-6,(j-1)*80+ylo+73,0]) cube([12,8,1.2]);
    }
}
module camera_groove_pad_kit(kind="pi") {
    // Removable strip gauges/shims; thin insulating compliant tape may replace these ASA gauges.
    // Slide shims from the open +X end, no force. Select stack by actual board, without calipers.
    x0 = camera_board_bounds_xz(kind)[0];
    for (i = [0:5],upper = [false,true],s = camera_groove_spans_x_mm(kind,upper)) {
        c = i%3; y = floor(i/3)*20+(upper ? 8 : 0);
        translate([c*85+(s[0]-x0),y,0]) cube([s[1]-s[0],0.8,0.4*(i+1)]);
        // breakaway tab from the strip back to its column spine
        translate([c*85-3,y,0]) cube([s[0]-x0+4,0.8,0.4]);
    }
    for (c = [0:2]) translate([c*85-5,-3,0]) cube([2.5,33.8,0.4]);   // column spines
    translate([-5,-3,0]) cube([2*85+2.5,2,0.4]);                       // rail joining spines
}

// Keep the existing foot for sensor_carrier.scad; cameras no longer mount on this pivot.
module camera_foot() {
    difference() {
        union() {
            plate(60,40,4,4);
            for (x = [-camera_hinge_gap/2-4,camera_hinge_gap/2]) {
                translate([x,-8,3]) cube([4,16,13]);
                translate([x,0,16]) rotate([0,90,0]) cylinder(r=8,h=4);
            }
        }
        translate([-12,0,16]) rotate([0,90,0]) cylinder(d=m4_clear,h=24);
        for (x = [-22,22]) translate([x,0,-1]) rotate([0,0,90]) slot(m4_clear,12,6);
    }
}

module camera_shell_print(kind = "pi") {
    translate([0,0,tower_flange/2]) rotate([-90,0,0]) camera_shell(kind);
}
module camera_tray_print(kind = "pi") { translate([0,0,camera_tray_t_mm]) camera_bottom_tray(kind); }
module camera_pi_carrier_print() { translate([0,0,96.4]) rotate([90,0,0]) camera_pi_carrier(); }
module camera_stereo_carrier_print() { translate([0,0,97.2]) rotate([90,0,0]) camera_stereo_carrier(); }
module camera_retainer_print(kind="pi") {
    // Rear plate on bed; support the projecting corner bridges, never print on the tiny PCB-edge tabs.
    translate([0,0,camera_pcb_front_y_mm+(kind == "pi" ? 16 : 6.8)])
        rotate([-90,0,0]) camera_board_retainer(kind);
}
module camera_floor_gasket_print(kind="pi") { translate([0,0,0.8]) camera_floor_gasket(kind); }
module camera_face_gasket_print(kind="pi") {
    translate([0,0,105.4]) rotate([90,0,0]) camera_face_gasket(kind);
}
module camera_insert_gasket_print(kind="pi",index=0) {
    translate([0,0,108.4]) rotate([90,0,0]) camera_insert_gasket(kind,index);
}
module camera_rim_gasket_print(kind="pi",index=0) {
    translate([0,0,-camera_lens_rim_seal_y_mm(kind)]) rotate([90,0,0]) camera_lens_rim_gasket(kind,index);
}

if (camera_part == "pi_housing") camera_shell_print("pi");
else if (camera_part == "pi_bottom") camera_tray_print("pi");
else if (camera_part == "pi_carrier") camera_pi_carrier_print();
else if (camera_part == "pi_retainer") camera_retainer_print("pi");
else if (camera_part == "pi_lens_insert") camera_insert_print("pi");
else if (camera_part == "pi_face_gasket") camera_face_gasket_print("pi");
else if (camera_part == "pi_floor_gasket") camera_floor_gasket_print("pi");
else if (camera_part == "pi_insert_gasket") camera_insert_gasket_print("pi");
else if (camera_part == "pi_rim_gasket") camera_rim_gasket_print("pi");
else if (camera_part == "pi_lens_gauge") camera_lens_fit_gauge("pi");
else if (camera_part == "pi_tile_fit_kit") camera_lens_tile_fit_kit("pi");
else if (camera_part == "pi_groove_pad_kit") camera_groove_pad_kit("pi");
else if (camera_part == "stereo_housing") camera_shell_print("stereo");
else if (camera_part == "stereo_bottom") camera_tray_print("stereo");
else if (camera_part == "stereo_carrier") camera_stereo_carrier_print();
else if (camera_part == "stereo_retainer") camera_retainer_print("stereo");
else if (camera_part == "stereo_lens_insert_left") camera_insert_print("stereo",0);
else if (camera_part == "stereo_lens_insert_right") camera_insert_print("stereo",1);
else if (camera_part == "stereo_face_gasket") camera_face_gasket_print("stereo");
else if (camera_part == "stereo_floor_gasket") camera_floor_gasket_print("stereo");
else if (camera_part == "stereo_insert_gasket_left") camera_insert_gasket_print("stereo",0);
else if (camera_part == "stereo_insert_gasket_right") camera_insert_gasket_print("stereo",1);
else if (camera_part == "stereo_rim_gasket_left") camera_rim_gasket_print("stereo",0);
else if (camera_part == "stereo_rim_gasket_right") camera_rim_gasket_print("stereo",1);
else if (camera_part == "stereo_lens_gauge") camera_lens_fit_gauge("stereo");
else if (camera_part == "stereo_tile_fit_kit") camera_lens_tile_fit_kit("stereo");
else if (camera_part == "stereo_groove_pad_kit") camera_groove_pad_kit("stereo");
else if (camera_part == "foot") camera_foot();
else assert(false,"Unknown camera_part");
