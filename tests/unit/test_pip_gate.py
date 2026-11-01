"""Unit tests for the print-in-place mesh gate.

The gate decides whether a multi-body export leaves the bed as releasable bodies. It is
exercised indirectly by the published manifest, but a change inside `_min_gap` or
`_droop_area` could weaken it while the manifest still records plausible numbers, so the
thresholds are pinned here directly.
"""

from __future__ import annotations

import numpy as np
import pytest

from hardware.mounts.pip_gate import (
    PIP_DROOP_AREA_LIMIT_MM2,
    PIP_MIN_GAP_MM,
    pip_report,
)


def _box_tris(x0, y0, z0, x1, y1, z1):
    """Axis-aligned box as 12 triangles, wound outward."""
    v = [
        (x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0),
        (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1),
    ]
    faces = [
        (0, 3, 2), (0, 2, 1),  # bottom (normal -Z)
        (4, 5, 6), (4, 6, 7),  # top
        (0, 1, 5), (0, 5, 4),  # -Y
        (1, 2, 6), (1, 6, 5),  # +X
        (2, 3, 7), (2, 7, 6),  # +Y
        (3, 0, 4), (3, 4, 7),  # -X
    ]
    tris = np.array([[v[a], v[b], v[c]] for a, b, c in faces], dtype=float)
    return tris


def _two_bodies(gap_mm, upper_dz=0.0):
    """Two bed-touching boxes separated horizontally by ``gap_mm``.

    The upper body optionally sits ``upper_dz`` above the lower one so a flat overhang
    hangs over its neighbour.
    """
    lower = _box_tris(0.0, 0.0, 0.0, 10.0, 10.0, 10.0)
    upper = _box_tris(10.0 + gap_mm, 0.0, upper_dz, 20.0 + gap_mm, 10.0, 10.0 + upper_dz)
    triangles = np.concatenate([lower, upper])
    labels = np.array([0] * len(lower) + [1] * len(upper))
    return triangles, labels


def test_two_bodies_with_a_wide_gap_pass():
    triangles, labels = _two_bodies(1.0)
    report = pip_report(triangles, labels)
    assert report["pip_bodies"] == 2
    assert report["pip_min_gap_mm"] >= PIP_MIN_GAP_MM
    assert report["pip_droop_area_mm2"] <= PIP_DROOP_AREA_LIMIT_MM2


def test_two_bodies_closer_than_the_minimum_gap_are_rejected():
    triangles, labels = _two_bodies(PIP_MIN_GAP_MM / 2)
    with pytest.raises(ValueError, match="PIP gap"):
        pip_report(triangles, labels)


def test_a_body_that_does_not_reach_the_bed_is_rejected():
    lower = _box_tris(0.0, 0.0, 0.0, 10.0, 10.0, 10.0)
    floating = _box_tris(20.0, 0.0, 2.0, 30.0, 10.0, 12.0)
    triangles = np.concatenate([lower, floating])
    labels = np.array([0] * len(lower) + [1] * len(floating))
    with pytest.raises(ValueError, match="does not reach the bed"):
        pip_report(triangles, labels)


def test_a_flat_overhang_over_another_body_is_rejected():
    """A horizontal ceiling above the neighbouring body is a weld risk.

    The overhanging body stands on the bed as a column and ledge; the ledge's underside
    hangs 0.5 mm above the top of the separate neighbouring body.
    """
    column = _box_tris(0.0, 0.0, 0.0, 2.0, 10.0, 10.0)
    ledge = _box_tris(0.0, 0.0, 8.0, 8.0, 10.0, 10.0)
    overhanger = np.concatenate([column, ledge])
    neighbour = _box_tris(3.0, 0.0, 0.0, 7.0, 10.0, 7.5)
    triangles = np.concatenate([overhanger, neighbour])
    labels = np.array([0] * len(overhanger) + [1] * len(neighbour))
    with pytest.raises(ValueError, match="flat overhang"):
        pip_report(triangles, labels)
