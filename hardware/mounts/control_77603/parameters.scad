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
use <../antenna_mount.scad>          // SMA bulkhead data (gn_*_mm()) for the SMA clip

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
// HaLow radios (owner 2026-11-02): a Heltec mini-PCIe card on a Pi carrier board on the Pi 5 (top
// of the stack) and a Seeed card on its Pi HAT on the Alta (under the Pi). Both carriers have a
// female-only 2x20; the Seeed HAT is 65 x 56 (hr_pcb), the Heltec carrier is Pi-size (hcp_pcb).
hr_pcb = [65, 56];
hr_top_h = 9;                          // ESTIMATE parts envelope above either HaLow HAT PCB
hr_ufl_h = 2.5;                        // U.FL mated height max, Hirose U.FL catalog p.4:
                                       // https://media.digikey.com/pdf/Data%20Sheets/Hirose%20PDFs/U.FL_Series.pdf
// Pi 5: Heltec HT-HC01P mini-PCIe card (MM6108, SPI) on a Pi carrier board (owner 2026-11-02;
// config/limits.yaml:17 records "Heltec HaLow HAT owns GPIO4/9/10/11 on this Pi"). The owner
// states the carrier is the same size as the Pi 5: modelled as the Pi outline on the Pi hole
// pattern (hcp_pcb, 85 x 56). Heltec's own carrier, the "HT-HC01P Raspberry Pi HAT"
// (https://heltec.org/project/ht-hc01p-hat/), is a 65 x 56 HAT in its product photos (four corner
// holes about 3.5 mm in from BOTH ends, the 2x20 between them): carrier model and length TO MEASURE;
// the larger owner-stated outline is the one probed. Socket and card placement are ESTIMATES
// (photos: socket at one short end, card along the long axis to two screw posts; here the card
// runs +u from its finger edge at u 7.65 to its far, U.FL end at u 58.6, hc_card_u1).
hcp_pcb = pi5_pcb;                     // owner 2026-11-02: "same size as the Pi 5" (85 x 56)
// Card drawing https://heltec.org/wp-content/uploads/2025/05/HC01P_DEMENSION-976x1024.png and
// datasheet https://resource.heltec.cn/download/HT-HC01P/Datasheet/HT-HC01P_Rev.1.0.0.pdf p.17:
// 50.95 x 30 x 3.75; IPEX (U.FL) on the component side, 2.61 from the far (hole) end and 10.46
// from the long edge away from the key notch; shield 18.5 x 14 x 2.75 starting 20.95 from the
// finger edge, 8.0 from that long edge; holes 2.9 from the far end and the long edges.
hc_card = [50.95, 30];
hc_card_u1 = 58.6;                     // ESTIMATE far (U.FL) end; card u 7.65..58.6
hc_card_v = [13, 43];                  // ESTIMATE (centred like the Seeed HAT's card)
hc_card_z = 5.2;                       // ESTIMATE card underside above the carrier PCB top
hc_card_t = 1.0;                       // drawing: card 3.75 overall, shield block 2.75
hc_shield = [[20.95, 8.0], [18.5, 14, 2.75]];   // [from finger edge, from bottom edge], size
hc_ufl = [2.61, 10.46];                // from the far end, from the bottom (v-low, ASSUMED) edge
hc_carrier_h = 3;                      // ESTIMATE carrier parts (socket excluded) above its PCB
hc_socket_u = [3, 12];                 // ESTIMATE mini-PCIe socket round the fingers
function hc_card_u() = [hc_card_u1 - hc_card[0], hc_card_u1];
function hc_ufl_uv() = [hc_card_u1 - hc_ufl[0], hc_card_v[0] + hc_ufl[1]];
// Lid-side cables end in SMA-male RIGHT-ANGLE plugs at the tower clip (sma_clip.scad); the plug
// nut's overhang past the mating plane is MIL-STD-348A's (antenna_mount gn_plug_overhang).
// L-com LCCN45958, https://www.l-com.com/Images/Downloadables/2D/LCCN45958_2D.pdf (Wellshow
// C01BA31174008A: 16.6 / 13.1 / barrel 10.8, https://www.wellshow.com/?sdm_process_download=1&download_id=30727)
ra_len = 17.5;                         // coupling-nut open end to the far face of the body
ra_axis = 13.1;                        // nut open end to the cable axis
ra_nut_len = 8.4;                      // coupling nut (5/16 in hex)
ra_w = 7.92;                           // body width (and height round the plug axis)
ra_barrel = 10.8;                      // plug axis to the crimp-barrel end (Wellshow; L-com 9.5)
ra_barrel_d = 3.9;                     // crimp ferrule
assert(hc_card_z + hc_card_t + max(hc_shield[1][2], hr_ufl_h) <= hr_top_h,
       "HC01P card + U.FL exceed the HaLow HAT parts envelope");
// Alta: Seeed Wio-WM6108 mini-PCIe card (Quectel FGH100M-H) on its required carrier, the Seeed
// WM1302 Pi HAT (https://www.seeedstudio.com/Wio-WM6108-Wi-Fi-HaLow-mini-PCIe-Module-p-6394.html).
// HAT board file (frame = this board frame, 65 x 56): https://files.seeedstudio.com/products/113100022/WM1302%20Pi%20Hat%20v1.0.brd
// Card: https://files.seeedstudio.com/wiki/wifi_halow/res/Wio-WM6108_V30.kicad_pcb ; antenna connector
// CN1 "CN_IPEX" = Hirose U.FL-R-SMT-1 (https://files.seeedstudio.com/wiki/wifi_halow/res/Wio-WM6108_V30_SCH_20241107.pdf).
wm_card_u = [6.4, 57.35];              // far (screw) end .. finger end in socket J1 (u 53.2..61.4)
wm_card_v = [13, 43];
wm_card_z = 5.2;                       // ESTIMATE card underside above the HAT PCB top (socket)
wm_card_t = 1.0;                       // ESTIMATE
wm_module_u0 = 13;                     // ESTIMATE module + shield start (the far end has the U.FL)
wm_module_h = 2.75;                    // ESTIMATE above the card (HC01P card total 3.75)
wm_socket_u = [52.5, 62];              // ESTIMATE J1 envelope round its contacts
wm_ufl_uv = [8.8, 23.3];               // card U.FL centre (from both CAD files, inferred)
ufl_body = [3, 2.6, 1.25];             // U.FL-R-SMT-1 receptacle, u x v x h (Hirose catalog p.3)
assert(wm_card_z + wm_card_t + max(wm_module_h, hr_ufl_h) <= hr_top_h,
       "Seeed card + U.FL exceed the HaLow HAT parts envelope");
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
// RF service joints (owner decisions 2026-11-01 / 2026-11-02): two HaLow antennas on N
// bulkheads at the lid's back corners (-X feeds the Alta, +X the Pi 5) and the GNSS SMA jack in
// the lid (../antenna_mount.scad). The lid separates from the boards at THREE SMA joints on a
// printed clip riding the tower clamp (sma_clip.scad): to open the lid, lift it ~100 mm and
// unscrew the three joints with an 8 mm spanner; the fragile U.FLs stay mated.
//   TOWER side (tower-borne, mated once at stack build, never unplugged in service): a U.FL
//   (IPEX MHF I) plug -> SMA-female bulkhead lead per radio, its jack held face-up in the clip.
//   LID side (lid-borne): N-female bulkhead -> SMA-male RIGHT-ANGLE plug pigtails (HaLow) and an
//   SMA-female bulkhead -> SMA-male right-angle jumper (GNSS; its bulkhead is the lid jack).
// The clip's three jacks use the lid jack's published SMA-F D-flat bulkhead data (antenna_mount
// gn_*: Taoglas CAB.011 / Linx CSH-SGFB drawings, MIL-STD-348A), each on a gn_land_t land.
// 1.13 mm micro-coax everywhere: minimum bend radius 6.8 mm published for Taoglas 1.13 cable
// (https://www.taoglas.com/specifications/30021F020000A_Taoglas.pdf). Each cable's corners are
// filleted at its own radius (tl_bend_r, ll_bend_r: R10 where the corridors allow, never under
// 6.8); the fillets must fit their legs and the cable must stay inside its corridors (asserted).
// Under-lid envelopes: AntennaMount's antenna_lid_keepout(): per HaLow station (x +-64.5,
// y -48) a 30 mm cylinder from z 162, the post-socket bosses x +-(31.5..82.5), y -18..2,
// z >= 170, and the GNSS jack column (16 mm round gn_xy_mm() = (51, 41.6) from z 160).
pig_bend_r = 10;                       // DESIGN preferred bend radius (published minimum 6.8)
gj_bend_r = 7;                         // DESIGN GNSS jumper U-turn radius (published minimum 6.8)
gj_u_floor = 98;                       // DESIGN floor of the GNSS jumper's U corridor (ko_plugs top z 90)
lid_lift_slack = 100;                  // owner rule: the lid lifts ~100 mm before the joints open
tower_dress_slack = 20;                // DESIGN minimum spare on a tower-side lead
// Joints [0] Pi HaLow, [1] GNSS, [2] Alta HaLow: jack axes (body x, y; DESIGN), the direction of
// each mated right-angle plug's cable barrel and of each joint's spanner swing (degrees about +Z
// from +X). Each barrel points away from its own swing; JA's lid-lifted cable rises outside the
// others' swings (sc_sweep probes).
sc_joints = [[25, 3.2], [33.2, 27.6], [37, 44.5]];
sc_names = ["Pi HaLow", "GNSS", "Alta HaLow"];
sc_barrel_dir = [90, 0, 180];
sc_sweep_dir = [180, 200, 210];
// Heights above the clamp frame's top Zc (clamp travel t moves the clip and everything on it).
sc_plate_z = [28.6, 33.6];             // clip plate; each jack's land is its top gn_land_t
sc_face_z = sc_plate_z[1] - gn_land_t_mm() + gn_thread_len_mm();   // jack (mating) face: 42.0
function sc_nut0_z() = sc_face_z - gn_plug_overhang_mm();          // plug nut open end: 38.57
function sc_barrel_z() = sc_nut0_z() + ra_axis;                    // plug cable axis: 51.67
function sc_top_z() = sc_nut0_z() + ra_len;                        // plug body top: 56.07
function sc_dir(i) = [cos(sc_barrel_dir[i]), sin(sc_barrel_dir[i])];
function sc_barrel_end(i) = sc_joints[i] + ra_barrel * sc_dir(i);
// 8 mm open-end spanner (published heads 19-20.5 wide, 3.8-5.5 thick, 90-141 long: IS 2028:2004
// Table 1/4 (DIN 3110, ISO 10102) https://law.resource.org/pub/in/bis/S01/is.2028.2004.pdf ;
// Gedore 7 R 8 https://shop.gedore.com/en-at/2297051/ ; Wera 6003 Joker 8
// https://www.weratools.com/en-us/tools/6003-joker-combination-wrench) on the plug's 5/16 in
// coupling nut. SMA mating torque 7-10 in-lb (Amphenol RF https://assets.alliedelec.com/image/upload/v1582062086/MKT/LP/Suppliers/amphenol-rf/sma-connectors/Amphenol_RF_SMA_Connectors_Specs.pdf
// p.2); bulkhead nut 15 in-lb (Cinch https://www.cinch.com/media/product-information/products/rf-microwave/connectors/pi-CCS-JOHN-142-0593-401.pdf p.2).
// Swing room per joint: a +-15 degree sector (a 15-degree offset head flipped turns the nut in
// 30-degree steps) of radius sc_sw_r in the slab sc_sw_slab on the plug nut, the head's ring
// round the nut, and the column over the sector beyond the head ring up to sc_hand_top: the
// spanner comes down there and slides onto the nut. 56 mm is all the box gives (B6, B4 and the
// INA board): published 8 mm spanners are 90-141 long, so hold one short of its end or use a
// stubby spanner (TO CONFIRM by trial).
sc_sw_r = 56;                          // DESIGN (free radius)
sc_sw_half = 15;
sc_sw_slab = [40.5, 46.0];             // above Zc: over the INA I2C run (Zc+40), within the plug nut
sc_head_r = 12;                        // >= half the 20.5 head + nut clearance
sc_hand_top = 60;                      // descent column top above the box rim (a hand's height)
assert(sc_sw_slab[0] >= sc_nut0_z() && sc_sw_slab[1] <= sc_nut0_z() + ra_nut_len,
       "Spanner slab leaves the plug's coupling nut");
assert(sc_sw_slab[1] - sc_sw_slab[0] >= 5.5, "Spanner slab thinner than a 5.5 mm head");
// Clip (sma_clip.scad; body x/y, z above Zc). It sits on the front crossbar of the clamp frame
// (x 15.5..24) and on both finger bands (x 17.5..22) beyond the long bars; jaw pairs straddle
// the two bands, their inner faces sc_jaw_cl off the band sides and the frame's +-Y faces; a
// snap leaf on the crossbar's +X face hooks under it.
sc_x0 = ctw_tx + ctw_post_u[0] + 0.1;  // 15.6: 0.5 off the INA cradle's cross member (x <= 15.1)
sc_x1 = ctw_tx + ctw_post_u[1];        // 24.0: the crossbar's +X face
sc_jaw_cl = 0.2;                       // DESIGN locating clearance (band sides, frame +-Y faces)
sc_jaw_drop = 4.5;                     // jaws down the band's 5 mm sides (1.0 over the posts at t -12)
sc_bridge_t = 3;                       // foot / bridges on the crossbar and band tops
sc_north_y = [38, 55.6];               // foot + column; 0.4 short of the camera port with the tower slid
sc_south_y = [-12.5, ctw_ty - sc_jaw_cl];   // south bridge: 0.5 off bays B4/B5 (y <= -13)
sc_web_x = [29.6, 32.6];               // south web from the plate to the south bridge
sc_plate_x1 = 42.6;                    // plate +X edge: 0.4 off the lid GNSS jack column (x >= 43)
sc_plate_y = [-2.0, 50.0];             // -Y edge: 0.45 off the INA cradle's lift path (y <= -2.45)
sc_leaf_x = [sc_x1 + 0.4, sc_x1 + 2.4];
sc_leaf_y = [39, 45];
sc_leaf_root_z = 16;
sc_lip_reach = 1.0;                    // lip under the crossbar's underside
sc_lip_cl = 0.4;                       // lip catch face under the crossbar
sc_tab = [29.0, -8.4];                 // release tab: out to x, leaf bottom z
function sc_leaf_len() = sc_leaf_root_z + ctw_clamp_t + sc_lip_cl;
function sc_leaf_strain() = 1.5 * (sc_leaf_x[1] - sc_leaf_x[0]) * (sc_lip_reach + sc_lip_cl) / pow(sc_leaf_len(), 2);
assert(sc_leaf_strain() <= 0.015, "SMA clip snap leaf strain exceeds 1.5 %");
// Tower-side lead corridors [lead, [box]] (body, nominal stack), lead = joint index:
tl_route = [
    [0, [[11.55, 14, 114], [29.2, 20, 117.4]]],    // Pi: +X under the clamp crossbar and INA cradle
    [0, [[24.5, -2.4, 114], [29.2, 20, 129.8]]],   //   outboard of the crossbar, up to JA's heat-shrink
    [1, [[13.05, 34, 95.2], [41.2, 42, 105]]],     // GNSS: +X under the Pi's HaLow carrier (underside z 106.1)
    [1, [[39.6, 21.5, 95.2], [45.6, 43.5, 120]]],  //   rise past the carrier's +X edge (x 39) + spare coil
    [1, [[32.2, 22, 117.2], [45.6, 38, 129.8]]],   //   -X over the carrier (its parts <= z 116.7) to JB
    [2, [[-39.5, 18.05, 54.6], [-33.5, 49.5, 60.2]]],   // Alta: +Y under the Pi from the card U.FL
    [2, [[-39.5, 46, 52], [10, 54.8, 61]]],        //   +X behind the stack (y < 55: clear of the camera nuts)
    [2, [[9, 49.6, 52], [15, 62.5, 61]]],          //   +Y round the tower's +Y post
    [2, [[9, 56.5, 52], [33, 62.5, 61]]],          //   +X past the post
    [2, [[26.5, 56.5, 52], [33, 62.5, 129.8]]],    //   rise in the +Y gap (x < 34: clear of the camera nuts)
    [2, [[26.5, 56.5, 120], [40.7, 62.5, 127]]],   //   +X between the camera nuts' z bands
    [2, [[34.3, 41.2, 120], [40.7, 62.5, 129.8]]], //   -Y to JC's heat-shrink
];
// Closed centrelines (U.FL centre -> heat-shrink end), corners filleted at tl_bend_r. They run
// diagonally where the corridors allow, so every fillet fits its legs and the cable stays inside its
// corridors (asserted below; found by a numerical search, rounded to 0.01 mm).
tl_way = [
    [[9.99, 16.86, 115.15], [11.85, 16.86, 115.19], [20.03, 18.35, 114.13], [26.68, 12.99, 116.8],
     [27.72, 5.27, 120.85], [25, 3.2, 127.41], [25, 3.2, 129.9]],
    [[11, 37.9, 96.3], [12.27, 37.9, 96.3], [26.75, 36.57, 99.58], [45.59, 38.32, 103.26],
     [40.34, 30.54, 117.33], [33.2, 27.6, 126.01], [33.2, 27.6, 129.9]],
    [[-37.2, 16.7, 55.7], [-37.2, 47.16, 57.17], [-4.41, 51.05, 57.03], [2.91, 50.03, 56.83],
     [14.4, 57.57, 58.54], [30.93, 57.82, 59.49], [30.42, 58.76, 92.73], [26.59, 61.99, 114.37],
     [34.75, 61.32, 123.07], [37, 44.5, 120.46], [37, 44.5, 129.9]],
];
// Bend radius per tower lead: R10 for the Pi and GNSS leads. The Alta lead's 6 mm corridors (short
// jogs round the tower's +Y post and between the camera nuts) take R8 at most: the search found no
// R9 or R10 path inside them. All are above the published 6.8.
tl_bend_r = [pig_bend_r, pig_bend_r, 8];
// U.FL plug -> SMA-F D-flat bulkhead, 1.13 mm: Pi Taoglas CAB.011 95 mm; GNSS the Waveshare kit
// cable ("IPEX 1 to SMA cable ~17cm", panel-mount SMA-F with nut and washer per the pack photo
// https://www.waveshare.com/img/devkit/accBoard/LC29H-AA-GPS-HAT/LC29H-GPS-HAT-details-pack.jpg;
// D-flat and panel range TO CONFIRM); Alta 250 mm (TO BUY).
tl_len = [95, 170, 250];
// 1 mm reach stubs: [lead][0 = board end, 1 = clip end] (body boxes, nominal).
tl_stub = [
    [[[10.5, 15.8, 114.1], [11.56, 17.9, 115]], [[24, 2.2, 129.7], [26, 4.2, 130.9]]],
    [[[12.05, 36, 95.5], [13.06, 39.8, 97.5]], [[32.2, 26.6, 129.7], [34.2, 28.6, 130.9]]],
    [[[-38.7, 17.0, 54.6], [-35.7, 18.06, 55.7]], [[36, 43.5, 129.7], [38, 45.5, 130.9]]],
];
// Lid-side cable corridors [cable, [box]] (lid closed, clip at nominal travel). They stay clear
// of the INA3221 I2C lead's envelope (ina_i2c_run) at every clamp travel. Each HaLow pigtail leaves
// its N bulkhead (z 170) inside the station envelope (AntennaMount halow_lid_keepout(), reserved
// for the bulkhead, its nut and the pigtail's exit): its exit box lies inside that envelope and its
// first corridor box runs back into it, so the reserved space is continuous. ec_pigtails()
// subtracts the envelope, so the probed corridors never overlap the bulkhead's room.
ll_route = [
    [0, [[59, -36, 162.2], [72, -10.4, 169.5]]],    // Pi: +Y out of the +X station, over B5
    [0, [[39, -16.4, 162.2], [72, -10.4, 169.5]]],  //   -X along the bays' front (y < -10: INA descent)
    [0, [[39, -16.4, 162.2], [45.6, 7.6, 169.5]]],  //   +Y under the +X boss (z < 170)
    [0, [[39, 2.4, 162.2], [45.6, 7.6, 180]]],      //   up beside the boss and the INA run (y >= 8)
    [0, [[39, 2.4, 168.5], [45.6, 22, 180]]],       //   +Y over the INA run (Zc+40 <= 167.7)
    [0, [[23, 18, 168.5], [45.6, 22, 180]]],        //   -X between JA and JB
    [0, [[23, 14.05, 168.5], [27, 22, 180]]],       //   -Y into JA's barrel
    [0, [[45.6, 2.4, 168.5], [72, 25.2, 180]]],     //   Pi slack loops over the INA descent
    [0, [[62, -50, 162.2], [67, -35, 170.6]]],      //   exit from the bulkhead, inside the envelope
    [1, [[49.5, 38.6, gj_u_floor], [52.5, 44.6, 159.95]]],  // GNSS jumper: down from the lid jack ...
    [1, [[49.5, 26.1, gj_u_floor], [52.5, 44.6, gj_u_floor + 6]]],     //   U-turn (100 mm lift slack hangs here)
    [1, [[49.5, 26.1, gj_u_floor], [52.5, 29.1, 172]]],     //   ... up the loft
    [1, [[44.05, 25.6, 168.5], [52.5, 29.6, 178.4]]],   //   R7 turn into JB's barrel
    [2, [[-72, -36, 162.2], [-59, -9, 169.5]]],     // Alta: +Y out of the -X station, over B3
    [2, [[-72, -15, 162.2], [-16, -9, 169.5]]],     //   +X along the bays' front, under the -X boss
    [2, [[-24, -15, 162.2], [-16, -9, 180]]],       //   up beside B4's front
    [2, [[-24, -9, 168.5], [-16, 47, 180]]],        //   +Y over the tower top (INA <= Zc+40)
    [2, [[-24, 41, 168.5], [26.15, 47, 180]]],      //   +X into JC's barrel
    [2, [[-31, -9, 168.5], [10, 40.6, 180]]],       //   Alta slack loops over the tower top
    [2, [[-67, -50, 162.2], [-62, -35, 170.6]]],    //   exit from the bulkhead, inside the envelope
];
// Closed centrelines, shortest path (N jack cable exit / lid GNSS heat-shrink end -> plug barrel
// end), corners filleted at ll_bend_r: the lift slack is the cable bought beyond these. The HaLow
// pigtails were searched as the tower leads; the GNSS jumper's is its shortest route, and
// ll_way_installed(1) is the path it takes (its spare hangs in the U).
ll_way = [
    [[64.5, -48, 170], [64.5, -48, 162.25], [60.75, -14.37, 165.16], [43.24, -12.14, 167.42],
     [42.68, -0.01, 168.3], [41.67, 13.71, 171.24], [40.43, 20.46, 176.18], [25, 21.85, 176.4],
     [25, 14, 176.4]],
    [[51, 41.6, 161.1], [51, 41.6, 154.1], [51, 27.6, 154.1], [51, 27.6, 176.4], [44, 27.6, 176.4]],
    [[-64.5, -48, 170], [-64.5, -48, 162.68], [-65.79, -13.48, 163.6], [-21.46, -10.85, 168.55],
     [-20.36, 5.31, 175.77], [-17.01, 44.5, 176.4], [26.2, 44.5, 176.4]],
];
// Bend radius per lid cable. Each HaLow pigtail leaves its bulkhead straight down at z 170 and
// must be level above the bays (corridor floor z 162.2): R <= 170 - 162.2 - 1.13 / 2 = 7.2, so R7.
// The GNSS jumper's U-turn is R7 too (gj_bend_r).
pig_lid_bend_r = 7;
ll_bend_r = [pig_lid_bend_r, gj_bend_r, pig_lid_bend_r];
// Reach stubs [cable][0 = lid end, 1 = plug barrel end] (body boxes, lid closed, nominal).
ll_stub = [
    [[[63, -34.0, 164], [66, -32.85, 167]], [[24, 13.4, 175.4], [26, 14.1, 177.4]]],
    [[[50, 40.6, 159.9], [52, 42.6, 161]], [[43.4, 26.6, 175.4], [44.1, 28.6, 177.4]]],
    [[[-66, -34.0, 164], [-63, -32.85, 167]], [[26.1, 43.5, 175.4], [26.8, 45.5, 177.4]]],
];
// N-female bulkhead -> SMA-male right-angle (HaLow) and SMA-F bulkhead -> SMA-M right-angle
// (GNSS) to buy, 1.13 mm cable (TO BUY).
ll_len = [250, 150, 300];

// ---- Cable centrelines: fillet fit, true length, containment ---------------------------------
// A route is a polyline whose corners are filleted at its bend radius r: a corner of deflection
// theta uses r tan(theta / 2) of each leg and shortens the route by 2 r tan(theta / 2) - r theta.
// The fillets must fit their legs (fillet_slack >= 0) and every point of the cable's surface (8
// round each centreline sample) must lie in the route's reserved boxes (corridors and connector
// stubs); the first and last cable_skip mm sit inside the connectors and are not tested.
coax_d = 1.13;                         // Taoglas 30021FC020000A jacket OD 1.13 +- 0.05 (sheet above)
coax_min_bend_r = 6.8;                 // published minimum bend radius, same sheet
cable_skip = 1.5;                      // DESIGN connector body length not tested at each end
function vsum(v) = len(v) == 0 ? 0 : [for (x = v) 1] * v;
function seg_len(p, i) = norm(p[i + 1] - p[i]);
function seg_dir(p, i) = (p[i + 1] - p[i]) / seg_len(p, i);
function corner_deg(p, k) = k <= 0 || k >= len(p) - 1 ? 0
    : acos(max(-1, min(1, seg_dir(p, k - 1) * seg_dir(p, k))));
function fillet_t(p, k, r) = r * tan(corner_deg(p, k) / 2);
function poly_len(p) = vsum([for (i = [0 : len(p) - 2]) seg_len(p, i)]);
function route_len(p, r) = poly_len(p)
    - vsum([for (k = [1 : len(p) - 2]) 2 * fillet_t(p, k, r) - r * corner_deg(p, k) * PI / 180]);
function fillet_slack(p, r) = min([for (i = [0 : len(p) - 2])
    seg_len(p, i) - fillet_t(p, i, r) - fillet_t(p, i + 1, r)]);
function cable_line(a, b) = let(m = max(1, ceil(norm(b - a) / 0.5))) [for (j = [0 : m]) a + (b - a) * j / m];
function cable_arc(p, k, r) = let(a = seg_dir(p, k - 1), b = seg_dir(p, k), th = corner_deg(p, k),
        nb = b - (a * b) * a, n = nb / norm(nb), c = p[k] - a * fillet_t(p, k, r) + n * r,
        m = max(2, ceil(th / 5)))
    [for (j = [0 : m]) c + r * (-n * cos(th * j / m) + a * sin(th * j / m))];
function cable_pts(p, r) = [for (k = [0 : len(p) - 2]) let(
        s = p[k] + seg_dir(p, k) * fillet_t(p, k, r), e = p[k + 1] - seg_dir(p, k) * fillet_t(p, k + 1, r))
    each concat(cable_line(s, e), corner_deg(p, k + 1) > 1e-6 ? cable_arc(p, k + 1, r) : [])];
function in_box(q, b) = q[0] >= b[0][0] && q[0] <= b[1][0] && q[1] >= b[0][1] && q[1] <= b[1][1]
    && q[2] >= b[0][2] && q[2] <= b[1][2];
// The cable-surface points of route p (bend radius r) outside `boxes`: empty when the cable fits.
function cable_escapes(p, r, boxes) = let(P = cable_pts(p, r), e = p[len(p) - 1])
    [for (i = [1 : len(P) - 2]) if (norm(P[i] - p[0]) > cable_skip && norm(P[i] - e) > cable_skip)
        let(t0 = P[i + 1] - P[i - 1]) if (norm(t0) > 1e-9) let(t = t0 / norm(t0),
            ref = abs(t[2]) < 0.9 ? [0, 0, 1] : [1, 0, 0], n1 = cross(t, ref) / norm(cross(t, ref)),
            n2 = cross(t, n1))
        for (w = [0 : 45 : 315]) let(q = P[i] + coax_d / 2 * (cos(w) * n1 + sin(w) * n2))
            if (len([for (b = boxes) if (in_box(q, b)) 1]) == 0) q];
function tl_boxes(i) = concat([for (r = tl_route) if (r[0] == i) r[1]], tl_stub[i]);
function ll_boxes(i) = concat([for (r = ll_route) if (r[0] == i) r[1]], ll_stub[i]);
function tl_route_len(i) = route_len(tl_way[i], tl_bend_r[i]);
function ll_route_len(i) = route_len(ll_way[i], ll_bend_r[i]);
// The GNSS jumper as installed: its shortest route with the U's floor lowered until the route is
// the cable bought (both U corners stay square, so the spare adds twice the drop).
function gj_u_z() = ll_way[1][1][2] - (ll_len[1] - ll_route_len(1)) / 2;
function ll_way_installed(i) = i != 1 ? ll_way[i] : [ll_way[1][0],
    [ll_way[1][1][0], ll_way[1][1][1], gj_u_z()], [ll_way[1][2][0], ll_way[1][2][1], gj_u_z()],
    ll_way[1][3], ll_way[1][4]];
for (i = [0 : 2]) {
    assert(min(tl_bend_r[i], ll_bend_r[i]) >= coax_min_bend_r,
           str(sc_names[i], " cable bend radius under the published minimum"));
    assert(fillet_slack(tl_way[i], tl_bend_r[i]) >= 0,
           str(sc_names[i], " tower lead: its R", tl_bend_r[i], " fillets do not fit its legs"));
    assert(fillet_slack(ll_way_installed(i), ll_bend_r[i]) >= -1e-9,
           str(sc_names[i], " lid cable: its R", ll_bend_r[i], " fillets do not fit its legs"));
    let(e = cable_escapes(tl_way[i], tl_bend_r[i], tl_boxes(i)))
        assert(len(e) == 0, str(sc_names[i], " tower lead leaves its corridors near ", len(e) ? e[0] : ""));
    let(e = cable_escapes(ll_way_installed(i), ll_bend_r[i], ll_boxes(i)))
        assert(len(e) == 0, str(sc_names[i], " lid cable leaves its corridors near ", len(e) ? e[0] : ""));
    assert(tl_len[i] >= tl_route_len(i) + tower_dress_slack,
           str(sc_names[i], " tower lead shorter than its route + dress slack"));
    assert(ll_len[i] >= ll_route_len(i) + lid_lift_slack,
           str(sc_names[i], " lid cable shorter than its route + lid-lift slack"));
}
// The GNSS jumper's spare length (bought - shortest closed route) hangs in its U, shared by both
// legs: the U corridor's floor must reach the depth that length needs.
assert(gj_u_z() - coax_d / 2 >= gj_u_floor, "GNSS jumper U corridor too shallow for its spare length");

// =====================================================================================
// INA3221 3-channel power monitor (owner: inside the enclosure). I2C 0x40 on the Pi bus
// (config/hardware.yaml; the Pi's PCA9685 is strapped to 0x41, config/tractor.yaml); sense
// leads from the external 30 A/75 mV (solar) and 50 A/75 mV (battery) shunts. The owner's
// module is NOT known, so the holder is a CAPACITY cradle (ina_holder.scad): one print takes
// any board inside the ranges below. It snaps onto the tower clamp's ladder frame (board flat,
// parts up, directly under the lid) and rides with the tower.
// Common outlines (long x short side):
//   Adafruit #6062: 38.354 x 22.86, EAGLE outline (layer 20) of
//     https://github.com/adafruit/Adafruit-INA3221-Breakout-PCB ; 38.6 x 22.9 x 10.5 over the
//     parts, https://www.adafruit.com/product/6062 . Two 2.5 plated holes at (2.54, 20.32) and
//     (35.814, 20.32); three 3.5 mm 2-pole terminal blocks on one long edge (parts end 2.1 from
//     it); a 15-pin 0.1 in header row 2.54 from the other long edge (header shipped loose);
//     STEMMA QT sockets centred on both short edges (0.45 from them). Same board file.
//   Purple / black CJMCU-3221 ("MCU-3221"): 38.3 x 31.7 (vendor; also "PCB size 38*32"),
//     https://www.jh-electronica.com/mcu-3221-ina3221-three-way-i2c-output-current-power-monitor.shtml
//     Holes, headers, terminals and thickness: TO MEASURE.
//   yesyes INA3221-R010: 30 x 26, http://www.yesyes.info/index.php/electronics-products/ina3221-rnone/
//   SwitchDoc Labs Grove/pin INA3221: no outline on the reachable vendor pages (TO MEASURE).
// Cradle frame of the board: ACROSS = body x (the jaws close on the two LONG edges), ALONG =
// body y. Capacity (DESIGN): across 22.0..32.0, along 30..38.6, PCB 1.0..1.6 thick.
ina_across_range = [22.0, 32.0];
ina_along_range = [30, 38.6];
ina_t_range = [1.0, 1.6];              // ESTIMATE (none of the boards publishes a thickness)
ina_t = 1.6;                           // modelled PCB; a 1.0 board has 0.9 vertical play
ina_top_h = 10.5;                      // ESTIMATE parts above the PCB (Adafruit: 10.5 overall)
ina_tail_h = 2.0;                      // ESTIMATE pin tails below the PCB
// Part-free strip along each gripped long edge, both faces: [jaw edge, shoe edge] = lip
// reach + the 0.4 mm clearance every gripping feature keeps from a board's parts.
ina_part_cl = 0.4;
ina_lip = 1.2;                         // fixed jaw lip / ledge reach
ina_shoe_lip = 0.8;                    // shoe lip / ledge reach (the Adafruit header edge sits here)
ina_edge_keep = [ina_lip + ina_part_cl, ina_shoe_lip + ina_part_cl];
ina_end_keep = 0.4;                    // short edges (Adafruit STEMMA QT sockets end 0.45 in)
// Reference boards [across, along]: probes use the capacity corners, the scene the CJMCU.
ina_board_adafruit = [22.86, 38.354];
ina_board_cjmcu = [31.7, 38.3];
ina_board_yesyes = [26, 30];
ina_board_probe = [[ina_across_range[0], ina_along_range[0]], [ina_across_range[1], ina_along_range[1]]];
// Adafruit #6062 parts, board frame (x along its 38.354 edge, y from its HEADER edge), from
// the EAGLE .brd cited above: header row centre y 2.54, pins x 1.27..36.83, plastic 2.54 wide
// over x 0..38.1; terminal blocks J3/J2/J1 centred x 10.16/19.05/27.94, footprint x -3.6..+3.4,
// y 13.745..20.745 (2.115 from the terminal edge); STEMMA QT sockets x 0.454..4.754 and
// 33.6..37.9, y 8.33..14.53. Heights and pads are ESTIMATES: header plastic 2.5; a plugged
// 2.54 mm female Dupont housing 14 above it over the whole row; terminal blocks 8.9 (10.5
// overall less the 1.6 PCB); STEMMA QT 3.0; pads under the header (1.7) and blocks, tails 2.0.
// REQUIRED ORIENTATION in the cradle: terminal edge on the fixed jaw, header edge under the shoe.
ada_hdr = [[0, 1.27], [38.1, 3.81]];
ada_hdr_h = 2.5;                       // ESTIMATE
ada_dupont_h = 14;                     // ESTIMATE plugged Dupont housing above the header plastic
ada_tb_x = [10.16, 19.05, 27.94];
ada_tb_dx = [-3.6, 3.4];
ada_tb_y = [13.745, 20.745];
ada_tb_h = 8.9;                        // ESTIMATE
ada_qt = [[[0.454, 8.33], [4.754, 14.43]], [[33.6, 8.43], [37.9, 14.53]]];
ada_qt_h = 3.0;                        // ESTIMATE
ada_hdr_pads = [[0.42, 1.69], [37.68, 3.39]];   // ESTIMATE 1.7 pads under the header pins
ada_tb_pads_y = [16.2, 18.1];          // ESTIMATE pads under the terminal-block pins
assert(ada_hdr[0][1] - ina_shoe_lip >= ina_part_cl, "Shoe lip within 0.4 mm of the Adafruit header");
assert(ina_board_adafruit[0] - ada_tb_y[1] - ina_lip >= ina_part_cl,
       "Jaw lip within 0.4 mm of the Adafruit terminal blocks");
// Cradle (body x/y; z above the clamp frame's top face).
ina_yc = ctw_ty + hatp_pcb[1] / 2;     // 21.65: board centre line = clamp frame centre line
ina_jaw_x = -29.5;                     // fixed jaw face (board's near long edge); bay B6 ends at x -33.9
ina_jaw_t = 2.5;
ina_z_board = 3.0;                     // PCB underside (tails hang into the clamp frame's opening)
ina_z_top = 8.0;                       // cradle top plane (it prints upside down on this plane)
ina_cl = 0.3;                          // lip underside above the modelled PCB top
ina_ledge_z0 = 1.5;
ina_lip_z0 = ina_z_board + ina_t + ina_cl;   // 4.9
// Ratchet jaw (print-in-place bar), stops as the strap bay: d = the bar's reference plane
// measured from the jaw face. The shoe's contact face sits ina_bulge in front of that plane
// on a cantilever leaf, which takes up the 1.5 mm step with 1.0..2.5 mm preload.
ina_bulge = 2.5;
ina_d_max = ina_across_range[1];       // 32.0
ina_steps = 7;
function ina_d_k(k) = ina_d_max - rt_pitch * k;
ina_d_min = ina_d_k(ina_steps);        // 21.5
ina_d_print = ina_d_max + 3.5;         // printed (open) position: pawls clear of the first tooth
function ina_k_for(w) = let(ks = [for (k = [0 : ina_steps]) if (ina_d_k(k) >= w) k]) ks[len(ks) - 1];
function ina_stop(w) = ina_d_k(ina_k_for(w));
assert(ina_across_range[0] >= ina_d_min + 0.5, "INA3221 jaw: smallest board under-preloaded");
// Bar parts relative to its reference plane x_bar = ina_jaw_x + d (x offsets):
ina_shoe_x = [-ina_bulge, 0.5];        // shoe block: contact face .. leaf
ina_leaf_t = 1.2;                      // cantilever leaf along y, flexing in x
ina_leaf_gap = ina_bulge + 0.4;        // room behind the leaf for the full compression
ina_beam_x0 = ina_shoe_x[1] + ina_leaf_t + ina_leaf_gap;   // 4.6: beam back face
ina_beam_t = 2.5;
ina_leaf_len = 18;                     // leaf root to the shoe centre (along y)
ina_shoe_len = 8;
ina_finger_len = 16;                   // pawl finger from the beam back face toward -x
ina_bar_z0 = 0.5;                      // bar underside above the clamp top
// Rails: rack faces outside the longest board's ends by the finger plus ina_end_gap.
ina_end_gap = 3.0;
ina_rail_t = 3.5;
ina_rail_y = [ina_yc - (ina_along_range[1] / 2 + ina_end_gap + rt_finger_t + 0.4),
              ina_yc + (ina_along_range[1] / 2 + ina_end_gap + rt_finger_t + 0.4)];
ina_tooth_z0 = 1.8;                    // pawl tooth band (45-degree ends) inside the rail groove
ina_tooth_h = 4.4;
// Clamp interface: long bars' outer faces at y = ctw_ty and ctw_ty + hatp_pcb[1], their
// underside ctw_clamp_t below the top; front crossbar x = ctw_tx + ctw_post_u.
ina_hook_t = 1.8;
ina_hook_len = 24;                     // snap leaf from the cradle's -x end (2 mm root block)
ina_hook_reach = 1.2;                  // lip under the long bar's outer edge
ina_tab_drop = 3;                      // saddle tabs below the clamp top
ina_cross_x = [ina_jaw_x + ina_d_print + ina_beam_x0 + ina_beam_t + 0.4, ctw_tx + ctw_post_u[0] - 0.4];
ina_tab_x = [ctw_tx + ctw_post_u[1] + 0.4, ctw_tx + ctw_post_u[1] + 2.2];
assert(ina_cross_x[1] - ina_cross_x[0] >= 1.5, "INA3221 cradle cross member thinner than 1.5");
// Snap strains, simple cantilever eps = 1.5 t delta / L^2, 1.5 % PETG DESIGN limit.
function ina_pawl_strain() = 1.5 * rt_finger_t * rt_pass_deflection() / pow(ina_finger_len - rt_tooth_out / 2, 2);
function ina_leaf_strain() = 1.5 * ina_leaf_t * ina_bulge / pow(ina_leaf_len, 2);
function ina_hook_strain() = 1.5 * ina_hook_t * (ina_hook_reach + 0.4) / pow(ina_hook_len - 4, 2);
assert(ina_pawl_strain() <= 0.015, "INA3221 jaw pawl strain exceeds 1.5 %");
assert(ina_leaf_strain() <= 0.015, "INA3221 shoe leaf strain exceeds 1.5 %");
assert(ina_hook_strain() <= 0.015, "INA3221 cradle snap strain exceeds 1.5 %");
// The INA3221's I2C lead (4-wire Dupont from the plugged header housing to the Pi's I2C bus at the
// +X panel), envelope agreed with the radio side 2026-11-02 (body x/y, z above the clamp frame's
// top; it rides the clamp): up the column over the header housing, +X between the SMA clip's Pi
// and GNSS joints, down the loft beside the +X panel. ESTIMATE route; the radio parts keep clear.
ina_i2c_run = [
    [[-12, 2, 21], [-6, 41, 40]],
    [[-12, 8, 35], [46, 16, 40]],
    [[46, -10, -19.7], [59, 25, 40]],
];
assert(sc_sw_slab[0] >= ina_i2c_run[1][1][2] + 0.4, "Spanner slab within 0.4 of the INA I2C run");
assert(sc_plate_z[1] <= ina_i2c_run[1][0][2] - 0.4, "SMA clip plate within 0.4 of the INA I2C run");
