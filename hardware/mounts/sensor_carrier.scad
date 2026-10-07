// Drill-to-fit carrier for an UNKNOWN sensor PCB/housing (BNO085, VL53L0X,
// BME280, INA3221 or ZED-F9P carrier). No vendor board pattern is assumed.
// Use camera_mount.scad's foot and M4 pivot, or bolt flat via side slots.
// FDM ASA. Rigidly lock an IMU after calibration; do not use foam isolation.
// Small bare boards require insulating standoffs and suitable weather protection.
include <common.scad>
sensor_plate_w = 64;
sensor_plate_h = 50;
sensor_plate_t = 3;
sensor_pcb_holes = [];              // measured [x,y] coordinates, origin plate centre
sensor_pcb_bore = 3.4;              // match actual PCB screw size; default M3 drilling blank
sensor_hinge_y = -sensor_plate_h / 2 - 8;

module sensor_carrier() {
    difference() {
        union() {
            plate(sensor_plate_w, sensor_plate_h, sensor_plate_t, 4);
            translate([-4, sensor_hinge_y - 1, 0]) cube([8, 13, sensor_plate_t]);
            translate([-4, sensor_hinge_y, 8]) rotate([0, 90, 0]) cylinder(r = 8, h = 8);
        }
        for (p = sensor_pcb_holes) {
            assert(abs(p[0]) <= sensor_plate_w / 2 - 12 &&
                   abs(p[1]) <= sensor_plate_h / 2 - 5,
                   "PCB hole would collide with tie slots or plate edge");
            translate([p[0], p[1], -1]) cylinder(d = sensor_pcb_bore, h = sensor_plate_t + 2);
        }
        // These are enclosure straps / M3 mounting slots, NOT claimed PCB holes.
        for (x = [-sensor_plate_w / 2 + 6, sensor_plate_w / 2 - 6])
            translate([x, 0, -1]) rotate([0, 0, 90]) slot(3.4, 28, sensor_plate_t + 2);
        translate([-6, sensor_hinge_y, 8]) rotate([0, 90, 0]) cylinder(d = m4_clear, h = 12);
    }
}

module sensor_carrier_placed(angle = 90) {
    translate([0, 0, 16]) rotate([angle, 0, 0])
        translate([0, -sensor_hinge_y, -8]) sensor_carrier();
}

sensor_carrier();
