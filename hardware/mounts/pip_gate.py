"""Print-in-place gate for multi-body STL exports.

A print-in-place (PIP) piece is ONE print that leaves the bed as several captive bodies, so
its mesh is several disjoint closed shells. ``mesh_report`` proves each shell is a closed,
positive solid; this module proves the shells can actually print apart:

* every body touches the bed. A body that starts in mid-air would lay its first layer on
  its neighbour, 0.4 mm below, and fuse to it;
* the measured gap between every pair of bodies is at least ``min_gap_mm``. The gap is
  measured on the facets the slicer will print, not on the design circles;
* no body has a flat overhang (steeper than 45 degrees from vertical) within
  ``droop_mm`` above another body. Such a face sags onto its neighbour and welds the joint.

The gate is geometric. It cannot prove that a given printer, nozzle, material and slicer
profile will actually release the joint: that is a coupon test.
"""

from __future__ import annotations

import math
from collections import defaultdict

import numpy as np

# Below this the gap between facets is too small to stay open on a 0.4 mm nozzle FDM print.
# Designs use 0.4 mm per side; 0.05 mm is left for facet chords on curved faces.
PIP_MIN_GAP_MM = 0.35
# A flat overhang this close above another body is treated as a weld risk.
PIP_DROOP_MM = 1.0
# Faces whose normal points down more steeply than this are flat overhangs (45 degrees).
_OVERHANG_NZ = -math.cos(math.radians(45.0)) - 0.01
# Area of flat overhang tolerated over another body, for facet slivers on chamfer seams.
PIP_DROOP_AREA_LIMIT_MM2 = 0.5
_CELL = 2.0
_EDGE_STEP = 0.25


def _closest_distance(points: np.ndarray, tris: np.ndarray) -> np.ndarray:
    """Exact distance from each point to the nearest triangle (Ericson's region test)."""
    p = points[:, None, :]
    a, b, c = (tris[None, :, k, :] for k in range(3))
    ab, ac, ap = b - a, c - a, p - a
    d1 = np.einsum("pmk,pmk->pm", np.broadcast_to(ab, ap.shape), ap)
    d2 = np.einsum("pmk,pmk->pm", np.broadcast_to(ac, ap.shape), ap)
    bp = p - b
    d3 = np.einsum("pmk,pmk->pm", np.broadcast_to(ab, bp.shape), bp)
    d4 = np.einsum("pmk,pmk->pm", np.broadcast_to(ac, bp.shape), bp)
    cp = p - c
    d5 = np.einsum("pmk,pmk->pm", np.broadcast_to(ab, cp.shape), cp)
    d6 = np.einsum("pmk,pmk->pm", np.broadcast_to(ac, cp.shape), cp)
    va = d3 * d6 - d5 * d4
    vb = d5 * d2 - d1 * d6
    vc = d1 * d4 - d3 * d2
    with np.errstate(divide="ignore", invalid="ignore"):
        denom = va + vb + vc
        v_in = vb / denom
        w_in = vc / denom
        v_ab = d1 / (d1 - d3)
        w_ac = d2 / (d2 - d6)
        w_bc = (d4 - d3) / ((d4 - d3) + (d5 - d6))
    closest = a + v_in[..., None] * ab + w_in[..., None] * ac
    cases = [
        ((d1 <= 0) & (d2 <= 0), np.broadcast_to(a, closest.shape)),
        ((d3 >= 0) & (d4 <= d3), np.broadcast_to(b, closest.shape)),
        ((vc <= 0) & (d1 >= 0) & (d3 <= 0), a + v_ab[..., None] * ab),
        ((d6 >= 0) & (d5 <= d6), np.broadcast_to(c, closest.shape)),
        ((vb <= 0) & (d2 >= 0) & (d6 <= 0), a + w_ac[..., None] * ac),
        (
            (va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0),
            b + w_bc[..., None] * (c - b),
        ),
    ]
    # Apply in reverse so the earliest matching region wins, as in the scalar algorithm.
    for mask, value in reversed(cases):
        closest = np.where(mask[..., None], value, closest)
    return np.sqrt(((p - closest) ** 2).sum(axis=-1)).min(axis=1)


def _surface_samples(tris: np.ndarray) -> np.ndarray:
    """Vertices, centroids and points along every edge at most _EDGE_STEP apart."""
    samples = [tris.reshape(-1, 3), tris.mean(axis=1)]
    for i, j in ((0, 1), (1, 2), (2, 0)):
        start, end = tris[:, i], tris[:, j]
        length = np.linalg.norm(end - start, axis=1)
        steps = np.maximum(1, np.ceil(length / _EDGE_STEP).astype(int))
        for n in np.unique(steps):
            if n < 2:
                continue
            sel = steps == n
            t = (np.arange(1, n) / n)[None, :, None]
            edge_points = (
                start[sel][:, None] + t * (end[sel] - start[sel])[:, None]
            )
            samples.append(edge_points.reshape(-1, 3))
    return np.unique(np.round(np.concatenate(samples), 4), axis=0)


def _grid(tris: np.ndarray, reach: float) -> dict[tuple[int, int, int], list[int]]:
    """Map each grid cell to the triangles whose box, grown by ``reach``, overlaps it."""
    cells: dict[tuple[int, int, int], list[int]] = defaultdict(list)
    low = np.floor((tris.min(axis=1) - reach) / _CELL).astype(int)
    high = np.floor((tris.max(axis=1) + reach) / _CELL).astype(int)
    for index, (lo, hi) in enumerate(zip(low, high, strict=True)):
        for x in range(lo[0], hi[0] + 1):
            for y in range(lo[1], hi[1] + 1):
                for z in range(lo[2], hi[2] + 1):
                    cells[(x, y, z)].append(index)
    return cells


def _min_gap(samples: np.ndarray, tris: np.ndarray, reach: float) -> float:
    """Smallest sample-to-surface distance below ``reach`` (``reach`` if none is closer)."""
    cells = _grid(tris, reach)
    keys = np.floor(samples / _CELL).astype(int)
    groups: dict[tuple[int, int, int], list[int]] = defaultdict(list)
    for index, key in enumerate(map(tuple, keys)):
        if key in cells:
            groups[key].append(index)
    best = reach
    for key, members in groups.items():
        candidate = tris[cells[key]]
        pts = samples[members]
        for start in range(0, len(pts), 256):
            best = min(best, float(_closest_distance(pts[start : start + 256], candidate).min()))
    return best


def _face_samples(tri: np.ndarray, step: float = 0.5) -> np.ndarray:
    """Barycentric grid on one triangle, at most ``step`` apart along its longest edge."""
    longest = max(np.linalg.norm(tri[i] - tri[(i + 1) % 3]) for i in range(3))
    n = max(1, int(math.ceil(longest / step)))
    grid = [(i / n, j / n) for i in range(n + 1) for j in range(n + 1 - i)]
    weights = np.array([(1 - u - v, u, v) for u, v in grid])
    return weights @ tri


def _droop_area(down: np.ndarray, others: np.ndarray, droop_mm: float) -> float:
    """Area of flat overhang faces with another body's surface just below them.

    Each overhang triangle is sampled on a grid, and its area is counted in proportion to
    the samples that have another body less than ``droop_mm`` directly beneath.
    """
    if len(down) == 0 or len(others) == 0:
        return 0.0
    cells = _grid(others, droop_mm)
    area = 0.0
    for tri in down:
        samples = _face_samples(tri)
        hits = 0
        for point in samples:
            key = tuple(np.floor(point / _CELL).astype(int))
            below: set[int] = set()
            for dz in range(0, int(math.ceil(droop_mm / _CELL)) + 2):
                below.update(cells.get((key[0], key[1], key[2] - dz), ()))
            if not below:
                continue
            cand = others[sorted(below)]
            a, b, c = cand[:, 0], cand[:, 1], cand[:, 2]
            v0, v1 = b[:, :2] - a[:, :2], c[:, :2] - a[:, :2]
            v2 = point[None, :2] - a[:, :2]
            den = v0[:, 0] * v1[:, 1] - v1[:, 0] * v0[:, 1]
            # Vertical facets project to zero area; give them a dummy denominator and
            # exclude them, so the division never produces NaN.
            usable = np.abs(den) > 1e-12
            safe = np.where(usable, den, 1.0)
            u = (v2[:, 0] * v1[:, 1] - v1[:, 0] * v2[:, 1]) / safe
            v = (v0[:, 0] * v2[:, 1] - v2[:, 0] * v0[:, 1]) / safe
            inside = usable & (u >= 0) & (v >= 0) & (u + v <= 1)
            if not inside.any():
                continue
            z = (a[:, 2] + u * (b[:, 2] - a[:, 2]) + v * (c[:, 2] - a[:, 2]))[inside]
            gap = point[2] - z
            if ((gap > 0) & (gap < droop_mm)).any():
                hits += 1
        if hits:
            tri_area = 0.5 * float(np.linalg.norm(np.cross(tri[1] - tri[0], tri[2] - tri[0])))
            area += tri_area * hits / len(samples)
    return area


def pip_report(
    triangles: np.ndarray,
    labels: np.ndarray,
    min_gap_mm: float = PIP_MIN_GAP_MM,
    droop_mm: float = PIP_DROOP_MM,
) -> dict:
    """Check a multi-body print. ``labels`` gives each triangle's body index."""
    bodies = sorted(set(labels.tolist()))
    tris = {body: triangles[labels == body] for body in bodies}
    zmin = {body: float(tris[body][:, :, 2].min()) for body in bodies}
    floating = [round(z, 3) for z in zmin.values() if abs(z) > 0.001]
    if floating:
        raise ValueError(f"PIP body does not reach the bed: z_min={floating}")
    reach = max(min_gap_mm, droop_mm) + 0.5
    gap = reach
    for i in bodies:
        for j in bodies:
            if i < j:
                gap = min(
                    gap,
                    _min_gap(_surface_samples(tris[j]), tris[i], reach),
                    _min_gap(_surface_samples(tris[i]), tris[j], reach),
                )
    if gap < min_gap_mm:
        raise ValueError(f"PIP gap {gap:.3f} mm is below {min_gap_mm} mm")
    droop = 0.0
    for body in bodies:
        own = tris[body]
        normal = np.cross(own[:, 1] - own[:, 0], own[:, 2] - own[:, 0])
        unit = normal / np.linalg.norm(normal, axis=1, keepdims=True)
        flat = (unit[:, 2] < _OVERHANG_NZ) & (own[:, :, 2].min(axis=1) > 0.001)
        others = np.concatenate([tris[b] for b in bodies if b != body])
        droop += _droop_area(own[flat], others, droop_mm)
    if droop > PIP_DROOP_AREA_LIMIT_MM2:
        raise ValueError(
            f"PIP flat overhang over another body: {droop:.2f} mm^2 within {droop_mm} mm"
        )
    return {
        "pip_bodies": len(bodies),
        "pip_min_gap_mm": round(gap, 3),
        "pip_droop_area_mm2": round(droop, 3),
    }
