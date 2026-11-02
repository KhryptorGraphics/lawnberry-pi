// Electronics contract for the Pi 5 + Libre Alta tower and the interface parts, all held
// bolt-free on a removable CHASSIS inside the unchanged legacy enclosure body
// (../enclosure_body.scad: interior 169 x 128 x 180, floor top z 4.5, lid ribs down to
// z 180.5). The chassis seats on the box's four existing floor bosses (tray_fixings) under
// the heads of the existing box-to-arm-layer M4 bolts: keyhole slots slide under printed
// studs clamped by those bolts, and a snap detent stops the slide. Nothing is molded into
// the box walls; the carriers hang on French-cleat strips molded on the chassis panels.
//
// Every number below is one of: a published dimension (source URL on the line), a DESIGN
// value chosen here, or an ESTIMATE / TO MEASURE. No fit, load, thermal, ingress or
// vibration claim is made beyond what control_checks.scad shows with real solids.
// Board data: local://board-dims.md (BoardResearch, 2026-11-01), URLs repeated per line.
include <../enclosure_common.scad>
use <../electronics_rail.scad>

control_geometry_verified = false;
control_watermark = "PROTOTYPE - FIT / LOAD / ELECTRICAL VALIDATION REQUIRED";

// Print-in-place gap, shared value with the throttle package (not shared by `use`).
joint_clearance_mm = 0.4;

// =====================================================================================
// Carrier family: the CARRIER side of the electronics_rail.scad cleat
// =====================================================================================
// Frame (cleat local): wall face y = 0, interior y < 0, strip along +x, z = 0 at the hook
// groove root. A carrier is drawn at its NOMINAL pose: every mating face 0.4 mm
// (cleat_clearance) off the strip. Gravity then drops it onto the bevel (cr_seat_shift).
cr_c = cleat_clearance();                        // 0.4, per mating face
cr_hook_off = cleat_hook_offset_mm();            // hook face on z = -y + this
cr_plate_back = -(cleat_t() + cr_c);             // -6.4: 0.4 off the strip's outer face
cr_plate_t = 3;                                  // DESIGN
cr_plate_front = cr_plate_back - cr_plate_t;     // -9.4
cr_hook_wall = -cr_c;                            // hook's wall-side face
cr_hook_top = cleat_t() + 6;                     // 12: hook block top
cr_rib_w = cleat_notch_w() - 0.4;                // 2.0 key rib in a 2.4 notch
cr_rib_below_bevel = cleat_notch_d() - cr_c;     // rib bottom 1.1 below the bevel line
// Latch: a tongue freed from the plate by two slots, hanging from a root above the hook,
// with a tooth that snaps up under the strip's catch land.
cr_tongue_w = 10;
cr_tongue_slot = 1;
cr_tongue_t = 2.1;                               // back face flush with the plate back
cr_tooth_d = 2.0;                                // reaches 1.6 under the 2.0 land
cr_tooth_top = cleat_land_z() - cr_c;            // -6.4: 0.4 under the land (nominal)
cr_tooth_flat = 1.0;
cr_tooth_bottom = cr_tooth_top - cr_tooth_flat - cr_tooth_d;   // 45-degree lead-in below
cr_tongue_free = cr_tooth_bottom - 2.6;          // -12: free end
cr_tongue_root = 15;
cr_plate_top = 18;                               // plate above the root
cr_lip = 1.6;                                    // grip lip, ahead of the plate front face
// Snap strain, simple cantilever eps = 1.5 t delta / L^2. delta = tooth reach 2.0 (the
// tooth tip must pass the strip's outer face, 1.6 engagement + 0.4 clearance); L = from the
// thin tongue's top (1 mm under the root, where it steps to the plate thickness) to the
// tooth's mid flat. DESIGN LIMIT 1.5 % for repeated PETG snaps (same ESTIMATE as the
// enclosure lid latch; no datasheet cited).
function cr_tongue_len() = (cr_tongue_root - 1) - (cr_tooth_top - cr_tooth_flat / 2);
function cr_latch_strain() = 1.5 * cr_tongue_t * cr_tooth_d / pow(cr_tongue_len(), 2);
cr_foot_h = 4;                                   // wall-bearing feet at the plate bottom
cr_foot_w = 6;
cr_foot_max_top = cleat_bottom_z() - cr_c;       // feet stay below the strip
// Gravity seat: the hook face lands on the bevel, the hook on the wall, the plate on the
// strip's outer face (all three 0.4 gaps close together).
cr_seat_shift = [0, cr_c, -(cr_c + cr_hook_off)];
// Vertical play before the latch tooth meets the land, from the nominal pose.
cr_latch_gap = cleat_land_z() - cr_tooth_top;    // 0.4
// Hook gap in z (vertical drop to the bevel from the nominal pose).
cr_hook_gap_z = cr_hook_off;
assert(cr_latch_strain() <= 0.015, "Carrier latch strain exceeds the 1.5 % PETG design limit");
assert(cr_tooth_bottom > cleat_bottom_z(), "Latch tooth must sit beside the strip, not under it");

// =====================================================================================
// Ratchet (bay strap bars and the tower clamp share the tooth form)
// =====================================================================================
rt_pitch = 1.5;          // DESIGN step
rt_h = 1.2;              // rack tooth height off its face, 45-degree ramp
rt_tooth_out = 1.6;      // pawl tooth protrusion off its finger
rt_finger_t = 1.8;
rt_engage = rt_tooth_out - rt_h + 0.4;   // pawl tooth tip sits 0.4 off the rack face
// Pawl deflection to pass one tooth = engagement 0.8 + 0.4 clearance.
function rt_pass_deflection() = rt_h - 0.4 + 0.4;

// =====================================================================================
// Strap bay: 4 converters + 2 HiLetgo relays, one print x 6 (PIP: frame + strap bar)
// =====================================================================================
// A box stands on the ledge against three back ribs; the print-in-place strap bar slides
// toward the wall and its two pawl fingers ratchet in grooves along the posts. A spring
// leaf on the bar takes up the 1.5 mm ratchet step. Box capacity (DESIGN):
//   width along the wall <= bay_w_in, depth from the ribs bay_box_d_range, height >= 18.
bay_w_in = 44;                       // HiLetgo 30A ~40 wide (UNVERIFIED, board-dims §9)
bay_post = 5;
bay_w = bay_w_in + 2 * bay_post;     // 54
bay_rib_proud = 2;
bay_ref_y = cr_plate_front - bay_rib_proud;   // box back face plane (rib tips)
bay_drop = 18;                        // ledge bottom below the strip root (short frame)
bay_ledge_t = 3;
bay_bar_t = 5;
bay_bar_h = 22;
bay_post_h = 24;
bay_leaf = 2.5;                       // spring leaf bulge toward the box
bay_leaf_t = 1.2;
bay_leaf_span = 30;
bay_d_max = 26.5;                        // bar back face from the rib plane, furthest stop
bay_steps = 7;
function bay_d_k(k) = bay_d_max - rt_pitch * k;      // ratchet stops
bay_d_min = bay_d_k(bay_steps);                      // 16
bay_d_print = bay_d_max + 3.5;                      // printed (unarmed) position
bay_box_d_range = [bay_d_min + 0.5, bay_d_max];    // 16.5 .. 26.5 with 1.0-2.5 leaf preload
bay_finger_slot = 1.4;
bay_finger_root = 3;                   // finger fused to the bar below this height
bay_tooth_z0 = 15;                     // pawl tooth base above the ledge bottom
bay_tooth_h = 6;
bay_groove_d = 1.6;
bay_front = bay_ref_y - bay_d_print - bay_bar_t - 0.6;   // post front end
// Stop k engages the box when the leaf is compressed 1.0 .. 2.5: the smallest stop at or
// beyond the box depth d. Search from the far stop toward the wall.
function bay_k_for_box(d) =
    let(ks = [for (k = [0 : bay_steps]) if (bay_d_k(k) >= d) k]) ks[len(ks) - 1];
function bay_pawl_strain() =
    1.5 * rt_finger_t * rt_pass_deflection()
    / pow(bay_tooth_z0 + bay_tooth_h / 2 - bay_finger_root, 2);
assert(bay_pawl_strain() <= 0.015, "Bay pawl strain exceeds the 1.5 % PETG design limit");

// Reference boxes (DESIGN CAPACITY envelopes, not purchased parts).
bay_box_min = [18, bay_box_d_range[0], 20];
bay_box_max = [bay_w_in - 0.4, bay_box_d_range[1], 80];
// HiLetgo 30A relay reference box, board-dims §9: "size the bay for about 75 x 42 x 26" (EST).
relay_box = [42, 26, 75];

// Harness clamp: the same bay without the ledge, narrower, its bar stops closer to the
// wall; the bundle runs vertically between the posts.
hc_w_in = 30;
hc_w = hc_w_in + 2 * bay_post;        // 40
hc_bundle_d = [8, 26];                // DESIGN range of the automotive harness bundle

// =====================================================================================
// Signal boards: 2x PCA9685 + Pololu #2806, held by clips molded on the chassis signal
// panel (boards parallel to the panel, no hole positions assumed)
// =====================================================================================
// Adafruit #815 fab print, https://learn.adafruit.com/16-channel-pwm-servo-driver/downloads :
// 2.45 x 0.88 in = 62.2 x 22.4 (the vendor text says 62.5 x 25.4: clones TO MEASURE).
pca_pcb = [22.4, 62.2];               // mounted upright: x = short edge, z = long edge
pca_t = 1.6;                          // ESTIMATE (not published)
pca_pin_h = 8.5;                      // ESTIMATE std 0.1 in header above the PCB
pca_plug_h = 16.5;                    // ESTIMATE servo plug top above the PCB
// Pololu dimension drawing, https://www.pololu.com/file/0J2021/pololu-4-channel-rc-servo-multiplexer-dimensions.pdf
pol_pcb = [24.6, 24.6];
pol_t = 1.02;
pol_pin_h = 8.5;                       // pins above the board top (drawing)
pol_tail_h = 2.0;                      // pin tails below (drawing)
sig_standoff = 3;                      // pin tails ~2.0 below the PCB (drawing, Pololu)
sig_cl = 0.3;                          // board pocket clearance per side
sig_lip = 2.0;                         // lip reach over a board face at its corners
// Clip finger: deflects lip + clearance = 2.3 over 25 mm -> eps = 1.5*1.8*2.3/625.
function sig_clip_strain() = 1.5 * rt_finger_t * (sig_lip + sig_cl) / pow(25, 2);
assert(sig_clip_strain() <= 0.015, "Signal board clip strain exceeds 1.5 %");

// =====================================================================================
// Compute tower: Alta stack under the Pi 5 stack, two prints (rack + ratchet clamp)
// =====================================================================================
// Board frame (u, v, w): u along the 85 edge from the microSD end, v from the USB-C edge,
// w up from the PCB underside. Pi 5 drawing RP-008347,
// https://datasheets.raspberrypi.com/rpi5/raspberry-pi-5-mechanical-drawing.pdf
pi5_pcb = [85, 56];
pi5_holes = [[3.5, 3.5], [61.5, 3.5], [3.5, 52.5], [61.5, 52.5]];
pi5_hole_d = 2.7;
pi5_usbc_u = 11.2;                     // drawing
pi5_hdmi_u = [25.8, 39.2];             // drawing
pi5_rj45_v = 10.2;                     // drawing
pi5_usb_v = [29.1, 47];                // drawing
pi5_port_overhang = 3;                 // RJ45/USB past the u = 85 edge (drawing)
pi5_gpio_u = 29;                       // header centre (drawing)
pi5_port_h = 16;                       // ESTIMATE USB stack top above the PCB (TO MEASURE)
board_t = 1.6;                         // ESTIMATE every PCB here (not published)
under_h = 3;                           // ESTIMATE underside parts envelope (TO MEASURE)
// Active Cooler, https://datasheets.raspberrypi.com/cooling/raspberry-pi-active-cooler-product-brief.pdf
cooler_xy = [63.5, 42.5];
cooler_h = 13.7;
cooler_uv0 = [1, 6.5];                 // ESTIMATE placement over the SoC
// AI HAT+, https://datasheets.raspberrypi.com/ai-hat-plus/raspberry-pi-ai-hat-plus-product-brief.pdf
// 65 x 56.5, Pi hole pattern, supplied 16 mm stacking header (passes GPIO up).
hatp_pcb = [65, 56.5];
hailo_xy = [17, 17];
// LC29H(DA), https://www.waveshare.com/wiki/LC29H(XX)_GPS/RTK_HAT : 65 x 30.5, holes 58 x 23.5.
// Female-only 2x20 (Waveshare package list). DEFAULT stack: ONE purchased 2x20 extra-tall
// stacking header replaces it on the LC29H so GPIO passes up to the HaLow HAT on top.
lc_pcb = [65, 30.5];
lc_v0 = pi5_pcb[1] - lc_pcb[1];        // sits along the GPIO edge
lc_ufl_uv = [57, 44.5];                // ESTIMATE GNSS U.FL (image: near the right end)
// HaLow HAT (owner's "heltec halow hat"; repo logs: Seeed WM6108 card on a Pi-style HAT).
// Modelled as a full Pi HAT envelope (Heltec HT-HR01 outline 65 x 56,
// https://resource.heltec.cn/download/HT-HR01/Datasheet/HT-HR01_V1.0.0.pdf) with a
// female-only header, carrying a mini-PCIe-format card (Heltec HC01P drawing 50.95 x 30 x 3.75,
// https://heltec.org/project/ht-hc01p/). RF connector type UNKNOWN: an SMA on the HAT edge
// AND a U.FL on the card both get cable room. Carrier model + RF connector: TO CONFIRM.
hr_pcb = [65, 56];
hr_card = [50.95, 30, 3.75];
hr_card_uv = [8, 13];                  // ESTIMATE card placement on the HAT
hr_card_z = 5.2;                       // ESTIMATE mPCIe socket: card underside above the PCB
hr_top_h = 9;                          // card + shield envelope above the HAT PCB (ESTIMATE)
hr_ufl_h = 2.5;                        // ESTIMATE mated U.FL height above the card
hr_sma_u = [12, 24];                   // ESTIMATE (HT-HR01 photo: left of centre)
hr_sma_reach = 14;                     // DESIGN room for an edge SMA + right-angle plug: TO CONFIRM
hr_sma_h = 7;                          // ESTIMATE SMA body height above the PCB top
// Libre Computer Alta (AML-A311D-CC): 85.6 x 56.5 measured (https://bret.dk/libre-computer-alta-review-big-cottonwood);
// Pi 3B+ layout and 58 x 49 holes ASSUMED (TO VERIFY).
alta_pcb = [85.6, 56.5];
alta_hs_h = 9;                         // ESTIMATE heatsink envelope: must fit under the HaLow HAT
// GPIO headers (ESTIMATES; the 16 mm stacking header is the AI HAT+'s, per its brief).
hdr_len = 50.8;                        // 2 x 20 at 2.54
hdr_w = 5.1;
hdr_v = 52.5;                          // centre line, Pi pattern
hdr_base_h = 2.5;                      // male header plastic base
male_pin_h = 8.5;                      // pins above a board (Pi, Alta)
stack_pin_h = 6;                       // stacking-header pins above a pass-through HAT
socket_h = 8.5;                        // female socket under a HAT
hdr_engage_min = 3;                    // probe: engagement at least this
// Stack spacings, PCB top to the next PCB bottom (kit standoffs reused as spacers).
s_alta_hr = 11;                        // ESTIMATE std HAT standoff (heatsink <= 9: TO MEASURE)
s_hr_pi = 15;                          // ESTIMATE: card 9 + U.FL 2.5 + Pi underside 3
s_pi_ai = 16;                          // HAT+ spec: 15 min, 16 ideal over the Active Cooler
s_ai_lc = 11;                          // ESTIMATE std HAT standoff on the AI HAT+ pins
s_lc_hr = 11;                          // ESTIMATE std HAT standoff on the LC29H stacking pins
// Rack frame (kept from the cleat frame it was designed in): back wall front face at
// y = cr_plate_front, board v along +x, board u along -y, shelf top at z = 0.
ctw_sd_gap = 8;                        // microSD end to the back wall: AI HAT+ FFC loop room
ctw_u0_y = cr_plate_front - ctw_sd_gap;   // y of the board u = 0 edge
ctw_v0_x = 4;                          // local x of the board v = 0 edge
ctw_shelf_t = 4;
ctw_boss_h = 5;                        // Alta underside clearance (ESTIMATE >= under_h)
ctw_boss_od = 5.6;                     // inside the ~6 mm hole pad
ctw_spigot_d = 2.3;                    // into the 2.7 hole
ctw_spigot_h = 1.2;                    // shorter than the PCB: shear only, no bending pin
ctw_post_u = [61.5, 70];               // clear of HDMI/audio; x >= 15.5 clears the camera-port mouth
ctw_post_gap = 1.2;                    // board edge to post; the boss-tab slot runs inside it
ctw_post_t = 4;
ctw_shelf_u = [-ctw_sd_gap, 70];
ctw_root_above_shelf = 0;              // local z origin at the shelf top
ctw_clamp_t = 5;
ctw_clamp_clear = hr_top_h + hr_ufl_h + 0.5;   // clamp frame above the top HaLow card + U.FL
ctw_finger_len = 32;
ctw_finger_tooth = 30;                 // tooth top below the clamp frame underside
ctw_adjust = 12;                       // clamp travel below nominal (shorter stack)
ctw_adjust_up = 3;                     // clamp travel above nominal (bay B6 sits above)
ctw_back_finger_x = [[6, 14], [50.5, 58.5]];   // rack local x of the back ratchet strips
// Stack heights above the shelf top (functions so the probes use the same chain).
function ctw_alta_bot() = ctw_boss_h;
function ctw_hra_bot() = ctw_alta_bot() + board_t + s_alta_hr;
function ctw_pi_bot() = ctw_hra_bot() + board_t + s_hr_pi;
function ctw_ai_bot() = ctw_pi_bot() + board_t + s_pi_ai;
function ctw_lc_bot() = ctw_ai_bot() + board_t + s_ai_lc;
function ctw_hrp_bot() = ctw_lc_bot() + board_t + s_lc_hr;
function ctw_top_board() = ctw_hrp_bot() + board_t;
function ctw_clamp_bot(dz = 0) = ctw_top_board() + ctw_clamp_clear + dz;
function ctw_post_top() = ctw_clamp_bot(-ctw_adjust) - 0.4;
assert(alta_hs_h <= s_alta_hr - 1, "Alta heatsink envelope does not fit under its HaLow HAT");
assert(cooler_h <= s_pi_ai - 1, "Active Cooler does not fit under the AI HAT+");
assert(hr_top_h + hr_ufl_h + under_h <= s_hr_pi, "Alta HaLow card + U.FL and the Pi underside collide");
function ctw_clamp_strain() = 1.5 * rt_finger_t * rt_pass_deflection() / pow(ctw_finger_tooth, 2);
assert(ctw_clamp_strain() <= 0.015, "Tower clamp pawl strain exceeds 1.5 %");

// Toe: the shelf runs past the back wall to u = ctw_toe_u. Its end face is the tower's lock
// catch: a chassis plate finger's nub springs up behind it (see chassis.scad).
ctw_toe_u = -20;
ctw_toe_v_max = 46;                    // toe notched clear of the (-62, 52) box bolt head
// Pedestals under the shelf, board (u, v): two mushroom studs (keyhole lock) and two pads,
// clear of the boss flexure tabs, the airflow window and each other's 6 mm slide.
ctw_pedestal_h = 3;                    // shelf bottom above the chassis plate top
ctw_studs = [[4, 28], [44, 28]];
ctw_pads = [[29, 2.5], [29, 53.5]];
ctw_stud_neck_d = 6;
ctw_stud_head_d = 10;                  // 45-degree cone head (printable upright)
ctw_ped_d = 10;
ctw_pad_d = 5;
ctw_floor_t = 1.6;                     // shelf pocket floor under each mushroom head
ctw_key_w = ctw_stud_neck_d + 0.6;
ctw_key_pocket_d = ctw_stud_head_d + 1;
ctw_slide = 6;                         // lowered 6 mm -X of its seat, then slid +X
ctw_window = [[16, 14], [36, 42]];     // shelf airflow / wiring window, board (u, v)

// =====================================================================================
// Chassis and layout (body coordinates, final seated position)
// =====================================================================================
// Box facts used here (HEAD enclosure_common.scad, enclosure_body.scad, enclosure_lid.scad):
// interior x +-84.5, y +-64, floor top z 4.5; bosses d 12 at tray_fixings, tops at
// tray_bottom_z = 14.5; PG9 glands at gland_centres (z <= 30.5 kept for the cable bend,
// gland_pass_d 24); breather on the +X wall at y 28, z 30; camera port in the +Y wall
// x +-15, z 85..135; camera-tower M5 bolts through the +Y wall at x +-40, z 40/90/140; lid
// ribs (lid_rib_w wide) at y = +-52 and x = +-73 hang to z 180.5.
ch_ceiling = 179;                      // every chassis part and payload stays below this
// BASE: one printed U-frame (two rails over the bosses + a +Y crossbar) clamped onto the
// boss tops by the four EXISTING box-to-arm-layer M4 bolts. It is as thick as the legacy
// tray (3.5 mm at HEAD), so those bolts keep their grip length. Four hooks stand on it.
cb_t = 3.5;                            // = HEAD's legacy tray_thickness under these bolt heads
cb_rail_x = [54, 70];                  // |x| span of each rail
cb_y = [-58.5, 60];
cb_bar_y = [44, 60];                   // +Y crossbar (the -Y side stays open over the glands)
cb_hooks = [[-60, -40], [60, -40], [-60, 20], [60, 20]];
cb_hook_post = [6, 6];
cb_hook_lip = 3.8;
cb_hook_top = 3;                       // lip top above the chassis plate top
// Base detent: a finger in the +X rail, thinned from the top, with a nub that pops up into
// the chassis plate's detent hole at the seat (release: press the nub through the hole).
cb_det_xy = [62, -9];
cb_det_w = 5;
cb_det_root_y = -36;
cb_det_t = 2.0;
cb_det_nub_d = 3.5;
cb_det_engage = 2.0;                   // nub top above the chassis plate underside
function cb_det_strain() =
    1.5 * cb_det_t * (cb_det_engage + 0.2) / pow(cb_det_xy[1] - cb_det_root_y, 2);
assert(cb_det_strain() <= 0.015, "Base detent strain exceeds 1.5 %");
bolt_head_d = 9;                       // ESTIMATE M4 washer (DIN 125 OD 9) envelope (TO MEASURE)
bolt_head_h = 4.8;                     // ESTIMATE washer 0.8 + M4 head 4 (TO MEASURE)
// CHASSIS plate on the base: lowered 6 mm +Y of its seat (the hooks rise through its
// windows), slid -Y under the hook lips; the base detent then stops it sliding back.
ch_seat_z = tray_bottom_z + cb_t;      // 18: plate underside on the base top
ch_plate_t = 4;
function ch_top() = ch_seat_z + ch_plate_t;
ch_slide = 6;
ch_x = [-83.5, 83.5];
ch_y = [-63, 57.5];
ch_corner_r = 5;                       // box inner corners are r 4
ch_edge_chamfer = 1.5;                 // 45-degree chamfer on each hook window's +Y top edge
ch_hook_cl = 0.2;
ch_head_hole_d = bolt_head_d + 0.6;
ch_det_hole_d = cb_det_nub_d + 0.8;
// Tower lock finger: cut in the plate under the toe's end, thinned to ch_lock_t from the
// top, flexing DOWN (the base's -X rail is windowed under it). Its nub stands
// ch_lock_engage above the shelf bottom, 0.4 behind the toe's end face. The lowered tower's
// toe presses it down; after the 6 mm slide it springs up behind the toe. Release: press
// the nub down, slide the tower -X.
ch_lock_engage = 1.2;
ch_lock_t = 2.0;
ch_lock_w = 6;                         // finger width (x)
ch_lock_y = [-4, 24];                  // root at y -4, nub at the free end
ch_lock_nub_len = 8;                   // nub length along y at the free end
ch_lock_slot = 1.0;
function ch_lock_strain() =
    1.5 * ch_lock_t * (ch_lock_engage + 0.4)
    / pow(ch_lock_y[1] - ch_lock_nub_len / 2 - ch_lock_y[0], 2);
assert(ch_lock_strain() <= 0.015, "Tower lock finger strain exceeds 1.5 %");
// Panels: inner faces carry the cleat strips and board clips.
ch_panel_t = 3;
ch_face_x = 80.5;                      // +-X panels' inner faces
ch_face_y = -60;                       // -Y panel inner face
ch_side_y = [-12.5, 56.6];             // side panels' y range (bays B1-B5 end at y -13)
ch_my_top = 160;
ch_mx_top = 160;
ch_px_top = 169;                       // under the lid post socket (z >= 170)
ch_vent_cut = [[9, 17], [41, 43]];     // +X panel slot round the breather [y0, z0], [y1, z1]
ch_vent_plate_x = 76;                  // plate edge under the breather fitting (y 12 .. 38.5)
ch_gland_hole_d = 26;
ch_gland_notch_z = 31;                 // -Y panel notched over each gland keep-out (to z 30.5)
// Strips on the chassis panels: [face, root_z, start, len]. Face "-Y": the back panel,
// local +x = body -X (start = the carrier's +X end). Face "-X": local +x = body +Y. Face
// "+X": local +x = body -Y. Each carrier hangs at strip local x = 0 on its own strip, so
// every copy of a carrier shares one rib phase and one print.
ch_strips = [
    ["-Y", 50, 28, 50],      // 0 bay B1   seated ledge bottom 31.03 (gland keep-out 30.5)
    ["-Y", 50, 83.5, 50],    // 1 bay B2
    ["-Y", 133, -29, 50],    // 2 bay B3   seated ledge 114.03: over the Pololu, under the -X antenna
    ["-Y", 147, 27, 50],     // 3 bay B4   seated ledge 128.03: over the tower clamp's finger band
    ["-Y", 128.5, 83, 50],   // 4 bay B5   seated ledge 109.53: over B2's relay, under the +X antenna
    ["-X", 147, 0, 50],      // 5 bay B6   over the tower's -X end
    ["-Y", 50, -43, 40],     // 6 harness clamp (seated feet from z 31.03)
];
// Payload in each bay (every bay is the same print) and its height capacity above the
// ledge top (DESIGN; the probes stand a box of this height in the bay).
lay_bays = [
    ["B1", 0, "HiLetgo 30A relay K_PI", 90],
    ["B2", 1, "HiLetgo 30A relay K_ALTA", 75],
    ["B3", 2, "steering converter L", 44],
    ["B4", 3, "steering converter R", 47.5],
    ["B5", 4, "battery-to-USB-C (Pi)", 48.5],
    ["B6", 5, "battery-to-USB-C (Alta)", 47.5],
];
lay_harness = 6;
// Signal boards: [name, face, start, bx, bz, size, thickness]; holder frame = cleat frame
// of that face (bx along the face from `start`, bz absolute z).
sig_boards = [
    ["PCA9685 (Pi)", "+X", 56, 5.2, 102, pca_pcb, pca_t],
    ["PCA9685 (Alta)", "+X", 56, 37.5, 102, pca_pcb, pca_t],
    ["Pololu 2806", "-Y", -43, 7.7, 85, pol_pcb, pol_t],
];
// Tower placement: board u -> body +X, v -> body +Y; u = 0 at x ctw_tx, v = 0 at y ctw_ty.
ctw_tx = -46;
ctw_ty = -6.6;
function ctw_shelf_z() = ch_top() + ctw_pedestal_h + ctw_shelf_t;
// Keep-outs (body boxes [[x0, y0, z0], [x1, y1, z1]]) that carriers and the chassis avoid.
ko_cam_port = [[-15, 56, 85], [15, 64, 135]];                      // port mouth
ko_nuts = [for (x = [-40, 40], z = [40, 90, 140]) [[x - 6, 55, z - 6], [x + 6, 64, z + 6]]];  // ESTIMATE
ko_vent = [[76.5, 18, 20], [84.5, 38, 40]];                        // ESTIMATE fitting nut
ko_usbc = [[-42.5, -37, 31], [-27, -7, 74]];                         // Pi/Alta USB-C plugs + bend
ko_plugs = [[42.5, -8, 31], [80, 50, 90]];                          // RJ45/USB plugs + bend

// =====================================================================================
// Pi 5 <-> Alta Ethernet patch (owner decision 2026-11-01): one short patch cable between the
// two RJ45 jacks. The jacks face +X one above the other (u = 85 edge, v 10.2 centre, Pi 5
// drawing; Alta position ASSUMED from its Pi 3B+-style layout), 29 mm apart. A STANDARD
// straight patch does NOT fit: its boot ends ~32 mm past the jack face (x 74), 6.5 mm short
// of the +X panel, and a 6 mm Cat6 cable needs a 25 mm bend radius. Specified instead: TWO
// 90-degree side-angle plugs (cable leaves toward -Y) on a FLAT or SLIM (<= 4 mm) Cat6 patch,
// 0.15 m, its slack coiled (bend radius >= 12 mm ASSUMED) in the free channel between the
// jacks and bays B2/B5. Plug and housing sizes are ESTIMATES (TO MEASURE on the cable bought).
patch_plug_w = 13.5;                   // ESTIMATE RJ45 plug body width (y)
patch_plug_h = 11.7;                   // ESTIMATE RJ45 plug body height (z)
patch_insert = 10;                     // ESTIMATE plug length inside the jack
patch_jack_h = 13.5;                   // ESTIMATE jack height above the PCB top
patch_body_out = 20;                   // ESTIMATE side-angle housing length past the jack face
patch_exit_x = [11, 19];               // ESTIMATE cable exit window, past the jack face
patch_cable_t = 7;                     // stub envelope across the cable (flat 6.5 or slim 4 OD)
patch_bend_r = 12;                     // ASSUMED minimum bend radius (flat or slim cable)
patch_len = 150;                       // cable to buy, plug end to plug end
patch_channel = [[46, -12.4, 37], [79.5, -5.2, 79]];   // slack coil envelope (x 33.5 >= 2 R)
assert(patch_channel[1][0] - patch_channel[0][0] >= 2 * patch_bend_r + patch_cable_t,
       "Patch slack channel narrower than one bend diameter");

// =====================================================================================
// HaLow antennas on the LID (owner decision 2026-11-01): two N bulkheads at the lid's back
// corners, one pigtail each (N bulkhead -> HaLow HAT): the -X one feeds the Alta's HaLow HAT,
// the +X one the Pi's. 1.13 mm micro-coax minimum bend radius ASSUMED 10 mm (board-dims: UNK,
// TO MEASURE); every corridor junction leaves room for that radius. The lid lifts with both
// antennas, so unplug the pigtails at the HATs first.
// Under-lid antenna envelopes: AntennaMount's antenna_lid_keepout() (../antenna_mount.scad):
// per station (x = -64.5 feeds the Alta HaLow, +64.5 the Pi HaLow; y -48) a 30 mm cylinder
// from z 162 to the lid, and the lid post-socket boss x +-(31.5..82.5), y -18..2, z >= 170.
// Each pigtail leaves its station toward +Y at centreline z 164 (162.5..165.5).
pig_bend_r = 10;                       // ASSUMED
// Pigtail envelopes (both cables share them), body boxes, in route order:
pig_route = [
    [[-72, -32.9, 162.2], [-59, -9, 169.5]],          // Alta: +Y out of its station, over B3
    [[-72, -24, 162.2], [-45, -9, 169.5]],            //   the R10 turn toward +X
    [[-72, -15, 162.2], [-15.5, -9, 169.5]],          //   along the bays' front, under the socket
    [[59, -32.9, 162.2], [72, -9, 169.5]],            // Pi: +Y out of its station, over B5
    [[45, -24, 162.2], [72, -9, 169.5]],              //   the R10 turn toward -X
    [[-24, -15, 162.2], [72, -9, 169.5]],             //   along the bays' front, over the tower
    [[-24, -21.5, 162.2], [-15.5, 62.5, 169.5]],      // both: forward over the tower clamp
    [[-21.5, 49.5, 150], [-15.5, 62.5, 169.5]],       //   the R10 turn down at the +Y wall
    [[-21.5, 56.5, 49.5], [-15.5, 62.5, 169.5]],      //   down the +Y gap to the HAT RF ends
];
// Where each pigtail meets its HaLow HAT's RF end (edge SMA or card U.FL, both TO CONFIRM):
// [0] Alta HaLow, [1] Pi HaLow; z of the connector band.
pig_end_z = [[49.5, 55], [109.5, 114.5]];
// Cable to buy (centreline estimate incl. bends and 25 mm service slack, rounded up).
pig_len = [400, 400];
