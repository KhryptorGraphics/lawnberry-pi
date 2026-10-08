"""W4 ID-pass labels: exact round trip, rounding tolerance, occlusion, drift detection."""

import numpy as np
import pytest

from workshop.idpass import (
    MAX_ID,
    TOLERANCE,
    IdPassError,
    boxes,
    decode,
    exposure_problem,
    id_to_rgb,
    yolo_line,
)


def test_exposure_gate_flags_black_and_blown_frames_only():
    def frame(v):
        return np.full((8, 8, 3), v, np.uint8)

    assert exposure_problem(frame(0)) == "black"
    assert exposure_problem(frame(255)) == "blown"
    assert exposure_problem(frame(5)) is None  # dim shade is real data
    assert exposure_problem(frame(190)) is None  # flat empty lawn is real data


def test_every_id_round_trips_within_tolerance():
    ids = np.arange(1, MAX_ID + 1)
    rgb = np.array([id_to_rgb(int(i)) for i in ids], dtype=np.int32)
    for delta in (-TOLERANCE, 0, TOLERANCE):
        img = np.clip(rgb + delta, 0, 255).astype(np.uint8)[None]
        assert (decode(img)[0] == ids).all()


def test_temporal_residue_and_off_palette_pixels_decode_as_background():
    img = np.array([[[4, 0, 4], [0, 6, 0], [40, 40, 40], [128, 0, 0]]], np.uint8)
    assert (decode(img) == 0).all()


def test_id_zero_and_overflow_rejected():
    for bad in (0, MAX_ID + 1):
        with pytest.raises(ValueError):
            id_to_rgb(bad)


def test_boxes_are_tight_on_visible_pixels_only():
    img = np.zeros((40, 60, 3), np.uint8)
    img[5:20, 10:30] = id_to_rgb(1)  # rock, partly hidden
    img[10:25, 20:40] = id_to_rgb(2)  # person in front, occludes rock's corner
    got = boxes(decode(img), {1: 7, 2: 0}, min_pixels=1)
    assert sorted(got) == [(0, 20, 10, 39, 24, 300), (7, 10, 5, 29, 19, 15 * 20 - 10 * 10)]


def test_tiny_instances_dropped():
    img = np.zeros((10, 10, 3), np.uint8)
    img[0, 0] = id_to_rgb(3)
    assert boxes(decode(img), {3: 1}, min_pixels=2) == []


def test_small_stray_islands_treated_as_edge_blend():
    """RTX blends ~1 px where silhouettes meet (measured 3080 Ti): a stray id
    in small islands is background, not contamination."""
    img = np.zeros((40, 60, 3), np.uint8)
    img[5:20, 10:30] = id_to_rgb(1)  # (0,0,85)
    img[10:25, 20:40] = id_to_rgb(63)  # (255,255,255)
    # measured blend row: id 21 = (85,85,85) along the shared edge
    img[10:12, 20:22] = id_to_rgb(21)  # 4-px island << max_stray_pixels (200)
    got = boxes(decode(img), {1: 7, 63: 0}, min_pixels=1)
    assert sorted((g[0], g[1]) for g in got) == [(0, 20), (7, 10)]  # class, x0


def test_large_stray_island_still_fails_loudly():
    """A big unassigned region means a material leaked past the override."""
    img = np.zeros((40, 60, 3), np.uint8)
    img[0:20, 0:20] = id_to_rgb(1)  # 400 px >> max_stray_pixels (200)
    with pytest.raises(IdPassError):
        boxes(decode(img), {2: 0})


def test_object_ids_are_doubling_safe():
    from workshop.idpass import OBJECT_IDS

    assert len(OBJECT_IDS) >= 8
    doubled = {tuple(min(255, 2 * c) for c in id_to_rgb(i)) for i in OBJECT_IDS}
    palette = {id_to_rgb(i) for i in OBJECT_IDS}
    # a doubled object colour is either itself (no 85 channel) or no object id
    for i in OBJECT_IDS:
        d = tuple(min(255, 2 * c) for c in id_to_rgb(i))
        assert d == id_to_rgb(i) or d not in palette
    assert doubled  # non-empty


def test_doubled_region_folds_into_source_box():
    """Measured 3080 Ti artefact: id (0,0,85) rendered as (0,0,170) over most
    of the object. The doubled pixels belong to the same instance."""
    img = np.zeros((40, 60, 3), np.uint8)
    img[5:20, 10:30] = (0, 0, 170)  # doubled id 1
    img[18:20, 28:30] = id_to_rgb(1)
    got = boxes(decode(img), {1: 7, 63: -1}, min_pixels=1)
    assert got == [(7, 10, 5, 29, 19, 300)]


def test_small_level2_blend_is_not_folded_beside_large_doubled_surface():
    """A large foldable surface must not make a distant 10-pixel blend fold too."""
    img = np.zeros((40, 60, 3), np.uint8)
    img[5:10, 5:10] = id_to_rgb(1)  # object A
    img[10:30, 10:30] = (0, 0, 170)  # large doubled surface folds into object A
    img[30:32, 50:55] = (0, 0, 170)  # separate 10-pixel blend stays background
    got = boxes(decode(img), {1: 7, 63: -1}, min_pixels=1)
    assert got == [(7, 5, 5, 29, 29, 425)]


def test_yolo_line_uses_inclusive_pixel_bounds():
    assert yolo_line(4, 0, 0, 9, 4, 20, 10) == "4 0.250000 0.250000 0.500000 0.500000"


def test_native_boxes_merge_sub_instances_and_drop_tiny():
    """Replicator annotator output -> boxes: sub-labelled parts of one spawned asset merge,
    ground/unlabelled ids are ignored, and < min_pixels instances are dropped."""
    from workshop.idpass import native_boxes

    dt = [(k, "<i4") for k in ("semanticId", "x_min", "y_min", "x_max", "y_max")]
    tight = np.array([(0, 10, 5, 29, 19), (1, 25, 15, 34, 24), (2, 50, 0, 50, 0)], dt)
    paths = ["/W/obj_00", "/W/obj_00/asset/wheel", "/W/obj_01"]
    seg = np.zeros((40, 60), np.uint32)
    seg[5:20, 10:30] = 2  # obj_00 body
    seg[15:25, 25:35] = 3  # its sub-labelled wheel
    seg[0, 50] = 4  # obj_01: 1 visible pixel
    ids = {"0": "BACKGROUND", "2": "/W/obj_00", "3": "/W/obj_00/asset/wheel", "4": "/W/obj_01"}
    got = native_boxes(tight, paths, seg, ids, {"/W/obj_00": 7, "/W/obj_01": 0})
    assert got == [(7, 10, 5, 34, 24, int((seg == 2).sum() + (seg == 3).sum()))]
