"""Instance-ID pass encoding for Isaac Sim synthetic data on the V100 (W4).

Isaac Sim's synthetic-data CUDA kernels (segmentation, 2D boxes) ship only
sm_75..sm_90 SASS without PTX, so on the V100 (sm_70) every annotator other
than ``rgb`` returns empty data. Labels are therefore rendered as a second
**ID pass** through the RGB path. Each labelled instance gets a flat emissive
colour, and lights, GI, mesh lights, AA, auto-exposure and tonemapping are off.

Measured on the V100 with tonemap op 0: the output is linear, at
``pixel = colour * emissive_intensity * 81.6`` per channel, and
``EMISSIVE_INTENSITY`` maps colour 1.0 to 255. Real-time RTX keeps a little
temporal history from the preceding beauty frame (observed residue of about
4/255), so IDs use a sparse palette, levels {0, 85, 170, 255} per channel
(63 IDs per frame). Decoding accepts a pixel only within ``TOLERANCE`` of a
palette colour; anything else is background.

On the RTX 3080 Ti, doubling and 1-px edge blends are measured, but comparison
against native Replicator boxes also found cross-object ID assignments that
this decoder cannot detect. Use native labels for 3080 Ti training; use this
ID pass on the V100, where native annotator kernels are unavailable, and for
diagnostics.
"""

from __future__ import annotations

import numpy as np

LEVEL = 85
LEVELS = 4  # per channel: 0, 85, 170, 255
MAX_ID = LEVELS**3 - 1  # 63; id 0 is background
TOLERANCE = 20
COUNTS_PER_UNIT = 81.6  # measured: rendered counts per unit emissive colour x intensity
EMISSIVE_INTENSITY = 255.0 / COUNTS_PER_UNIT


def _levels(instance_id: int) -> tuple[int, int, int]:
    r, rem = divmod(instance_id, LEVELS * LEVELS)
    g, b = divmod(rem, LEVELS)
    return r, g, b


def _from_levels(r: int, g: int, b: int) -> int:
    return r * LEVELS * LEVELS + g * LEVELS + b


# On the 3080 Ti, parts of an emissive object intermittently render at exactly
# 2x (measured: id 1 = (0,0,85) appearing as (0,0,169) over 100k+ px in 150 of
# 1751 v1 frames). Doubling maps level 1 -> 2 and keeps 0 and 3 (255 clips), so
# object ids use only levels {0, 1, 3}: a doubled colour then always contains a
# level 2, which no object id has, and halving that level folds it back to its
# unique source id. Ground (MAX_ID, all 3s) is doubling-stable. 25 ids >= the
# 8 objects per frame isaac_sdg spawns.
OBJECT_IDS: tuple[int, ...] = tuple(
    i for i in range(1, MAX_ID) if all(c in (0, 1, 3) for c in _levels(i))
)


def _large_component_indices(mask: np.ndarray, min_pixels: int) -> list[np.ndarray]:
    """Return flat pixel indices for 8-connected components above the size gate."""
    h, w = mask.shape
    seen = np.zeros_like(mask)
    active = np.flatnonzero(mask)
    components = []
    for seed in active:
        y0, x0 = divmod(int(seed), w)
        if seen[y0, x0]:
            continue
        seen[y0, x0] = True
        stack = [int(seed)]
        pixels = []
        while stack:
            flat = stack.pop()
            y, x = divmod(flat, w)
            pixels.append(flat)
            for ny in range(max(0, y - 1), min(h, y + 2)):
                for nx in range(max(0, x - 1), min(w, x + 2)):
                    if mask[ny, nx] and not seen[ny, nx]:
                        seen[ny, nx] = True
                        stack.append(ny * w + nx)
        if len(pixels) > min_pixels:
            components.append(np.asarray(pixels, dtype=np.intp))
    return components


def fold_doubled(
    ids: np.ndarray, assigned: dict[int, int], min_fold_pixels: int = 200
) -> np.ndarray:
    """Map large 2x-brightness surfaces to their assigned source id (returns a copy).

    Folds a connected component of an unassigned level-2 id only when its source
    (every level-2 channel halved to level 1) is assigned AND the component is
    larger than ``min_fold_pixels``. Real doubled surfaces measured 100k+ px;
    1-px edge blends (<= 65 px) can contain a 170 channel and must not be
    attributed to an unrelated object (they remain strays = background).
    """
    out = ids.copy()
    for iid in np.unique(ids):
        iid = int(iid)
        if iid == 0 or iid in assigned:
            continue
        lv = _levels(iid)
        if 2 not in lv:
            continue
        src = _from_levels(*(1 if c == 2 else c for c in lv))
        if src not in assigned:
            continue
        mask = ids == iid
        if mask.sum() <= min_fold_pixels:
            continue
        for indices in _large_component_indices(mask, min_fold_pixels):
            out.flat[indices] = src
    return out


def id_to_rgb(instance_id: int) -> tuple[int, int, int]:
    """8-bit target colour for an instance id (1..MAX_ID)."""
    if not 1 <= instance_id <= MAX_ID:
        raise ValueError(f"instance id {instance_id} outside 1..{MAX_ID}")
    r, rem = divmod(instance_id, LEVELS * LEVELS)
    g, b = divmod(rem, LEVELS)
    return (r * LEVEL, g * LEVEL, b * LEVEL)


def id_to_emissive(instance_id: int) -> tuple[float, float, float]:
    """Emissive colour (0..1) to author on the ID material."""
    return tuple(c / 255.0 for c in id_to_rgb(instance_id))  # type: ignore[return-value]


def decode(img: np.ndarray) -> np.ndarray:
    """HxWx3 uint8 ID-pass image -> HxW int32 instance ids (0 = background/unknown)."""
    c = img[..., :3].astype(np.int32)
    q = np.clip((c + LEVEL // 2) // LEVEL, 0, LEVELS - 1)
    ok = (np.abs(c - q * LEVEL) <= TOLERANCE).all(axis=-1)
    ids = q[..., 0] * LEVELS * LEVELS + q[..., 1] * LEVELS + q[..., 2]
    return np.where(ok, ids, 0).astype(np.int32)


def _largest_component(mask: np.ndarray) -> int:
    """Size of the largest 8-connected component of a boolean mask (numpy only).

    Iterative flood fill via row-span expansion: ID-pass stray masks are tiny
    edge blends, so this stays O(pixels) with no scipy dependency.
    """
    h, w = mask.shape
    seen = np.zeros_like(mask)
    best = 0
    for y0 in range(h):
        xs = np.nonzero(mask[y0] & ~seen[y0])[0]
        for x0 in xs:
            if seen[y0, x0]:
                continue
            size = 0
            stack = [(y0, x0)]
            while stack:
                y, x = stack.pop()
                if y < 0 or y >= h or x < 0 or x >= w:
                    continue
                if not mask[y, x] or seen[y, x]:
                    continue
                seen[y, x] = True
                size += 1
                stack.extend(
                    (
                        (y - 1, x - 1),
                        (y - 1, x),
                        (y - 1, x + 1),
                        (y, x - 1),
                        (y, x + 1),
                        (y + 1, x - 1),
                        (y + 1, x),
                        (y + 1, x + 1),
                    )
                )
            best = max(best, size)
    return best


class IdPassError(RuntimeError):
    """Decoded ids do not match the ids that were assigned (ID pass not exact)."""


def exposure_problem(img: np.ndarray, black_below: float = 3.0, blown_above: float = 250.0):
    """``"black"`` / ``"blown"`` if a beauty frame is unusable, else ``None``.

    Measured on the 3080 Ti with seed 1. About 3% of frames come out black (mean < 3) because
    the camera spawns inside or under an object. Extra render steps never recover them.
    Genuinely dim frames, such as shade under an oak, start at a mean of 4.2. A run on a
    GPU shared with another job came out flat white (mean 255) after frame 0. Contrast is
    not gated: an empty lawn is legitimately flat (std 1.6).
    """
    mean = float(img[..., :3].mean())
    if mean < black_below:
        return "black"
    if mean > blown_above:
        return "blown"
    return None


def boxes(
    ids: np.ndarray,
    assigned: dict[int, int],
    min_pixels: int = 20,
    max_stray_pixels: int = 200,
) -> list[tuple[int, int, int, int, int, int]]:
    """Tight visible boxes per instance.

    ``assigned`` maps instance id -> class index. Returns
    ``(class_idx, x0, y0, x1, y1, pixels)`` with inclusive pixel bounds.

    RTX renders object silhouettes with ~1 px of colour blending where two
    materials meet. Measured on the 3080 Ti (13 rejected smoke frames):
    ~0.3% of pixels, largest stray component 65 px (median 43), e.g. id 21 =
    (85,85,85) between ids 1 = (0,0,85) and 63 = (255,255,255); identical with
    DLSS/AA off and after 12 flush steps. ``max_stray_pixels`` = 200 is ~3x the
    observed max. Unassigned ids forming islands no larger than it are treated
    as blend = background. A *large* unassigned island means the pass is
    contaminated (a material leaked past the override) and the frame must not
    be labelled: ``IdPassError``.
    """
    ids = fold_doubled(ids, assigned)
    present = np.unique(ids)
    big_stray = []
    for iid in present:
        iid = int(iid)
        if iid == 0 or iid in assigned:
            continue
        if _largest_component(ids == iid) > max_stray_pixels:
            big_stray.append(iid)
    if big_stray:
        raise IdPassError(f"decoded unassigned ids {big_stray[:8]}")
    out = []
    for iid in present:
        iid = int(iid)
        if iid == 0 or iid not in assigned:  # small strays = background
            continue
        ys, xs = np.nonzero(ids == iid)
        if len(xs) < min_pixels:
            continue
        x0, y0, x1, y1 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
        out.append((assigned[iid], x0, y0, x1, y1, len(xs)))
    return out


def native_boxes(
    tight: np.ndarray,
    tight_paths: list[str],
    seg: np.ndarray,
    seg_paths: dict,
    spawned: dict[str, int],
    min_pixels: int = 20,
) -> list[tuple[int, int, int, int, int, int]]:
    """Replicator ``bounding_box_2d_tight`` + ``instance_id_segmentation`` -> :func:`boxes` tuples.

    ``tight`` is the box annotator's structured array (``x_min``..``y_max``, visible pixels,
    inclusive, measured equal to the segmentation extent) and ``tight_paths`` its
    ``info["primPaths"]``; ``seg`` is the HxW id image and ``seg_paths`` its
    ``info["idToLabels"]`` (id -> mesh prim path). ``spawned`` maps each labelled prim path to
    its class. Boxes and pixels under a spawned path are merged into it (meshes, or an asset's
    own sub-labels); instances with fewer than ``min_pixels`` visible pixels are dropped, as in
    :func:`boxes`.
    """

    def owner(path: str) -> str | None:
        return next((r for r in spawned if path == r or path.startswith(r + "/")), None)

    pixels: dict[str, int] = {}
    paths = {int(k): v for k, v in seg_paths.items()}
    for iid, n in zip(*np.unique(seg, return_counts=True), strict=True):
        root = owner(paths.get(int(iid), ""))
        if root:
            pixels[root] = pixels.get(root, 0) + int(n)
    merged: dict[str, tuple[int, int, int, int]] = {}
    for rec, path in zip(tight, tight_paths, strict=True):
        root = owner(path)
        if root is None:
            continue
        b = tuple(int(rec[k]) for k in ("x_min", "y_min", "x_max", "y_max"))
        m = merged.get(root, b)
        merged[root] = (min(m[0], b[0]), min(m[1], b[1]), max(m[2], b[2]), max(m[3], b[3]))
    return [
        (spawned[r], *b, pixels.get(r, 0))
        for r, b in merged.items()
        if pixels.get(r, 0) >= min_pixels
    ]


def yolo_line(cls: int, x0: int, y0: int, x1: int, y1: int, w: int, h: int) -> str:
    cx, cy = (x0 + x1 + 1) / 2 / w, (y0 + y1 + 1) / 2 / h
    bw, bh = (x1 - x0 + 1) / w, (y1 - y0 + 1) / h
    return f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"
