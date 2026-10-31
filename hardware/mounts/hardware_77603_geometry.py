"""Pure numerical helpers for the 77603 throttle station report.

These functions exist so the kinematic/strength numbers in the build report come from one
auditable implementation rather than being retyped per variant. They are NOT a substitute
for the CSG collision probes: those test the actual modelled solids.

Units are millimetres, newtons and megapascals throughout; ``euler_buckling_n`` therefore
returns newtons from ``E`` in MPa (N/mm^2), ``I`` in mm^4 and spans in mm.
"""

from __future__ import annotations

import math


class GeometryError(ValueError):
    """Raised when an input cannot describe real geometry."""


def _finite(name: str, value: float) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise GeometryError(f"{name} must be finite, got {value!r}")
    return number


def _positive(name: str, value: float) -> float:
    number = _finite(name, value)
    if number <= 0:
        raise GeometryError(f"{name} must be positive, got {value!r}")
    return number


def _non_negative(name: str, value: float) -> float:
    number = _finite(name, value)
    if number < 0:
        raise GeometryError(f"{name} must not be negative, got {value!r}")
    return number


def solve_endpoint_mm(
    horn_pin_mm: tuple[float, float, float],
    endpoint_x_mm: float,
    endpoint_z_mm: float,
    rod_length_mm: float,
) -> tuple[float, float, float]:
    """Return the forward (+Y) throttle endpoint at the given X/Z and rod length.

    A fixed-length two-force member joins the horn pin to the throttle attachment. With the
    attachment constrained in X and Z, the remaining degree of freedom is Y, so

        dy = sqrt(L^2 - dx^2 - dz^2),  dy >= 0

    and the endpoint is ``(x, y_pin + dy, z)``. A negative radicand means the requested
    endpoint is farther from the pin than the rod can reach, which is a real assembly
    failure rather than a value to clamp.
    """
    if len(horn_pin_mm) != 3:
        raise GeometryError("horn_pin_mm must be a 3-tuple")
    px, py, pz = (_finite("horn_pin", v) for v in horn_pin_mm)
    ex = _finite("endpoint_x_mm", endpoint_x_mm)
    ez = _finite("endpoint_z_mm", endpoint_z_mm)
    length = _positive("rod_length_mm", rod_length_mm)

    dx = ex - px
    dz = ez - pz
    remaining = length * length - dx * dx - dz * dz
    if remaining < 0:
        raise GeometryError(
            "throttle endpoint is unreachable: rod length "
            f"{length:.3f} mm is shorter than the {math.hypot(dx, dz):.3f} mm offset"
        )
    return (ex, py + math.sqrt(remaining), ez)


def reachable(rod_length_mm: float, dx_mm: float, dz_mm: float) -> bool:
    """Whether a rod of this length can span the given X/Z offset."""
    length = _positive("rod_length_mm", rod_length_mm)
    return length * length - dx_mm * dx_mm - dz_mm * dz_mm >= 0


def torque_nm(force_n: float, radius_mm: float) -> float:
    """Servo torque needed for a tangential force at a horn radius."""
    force = _non_negative("force_n", force_n)
    radius = _non_negative("radius_mm", radius_mm)
    return force * radius / 1000.0


def euler_buckling_n(
    modulus_mpa: float,
    second_moment_mm4: float,
    span_mm: float,
    effective_length_factor: float = 1.0,
) -> float:
    """Euler critical load ``Pcr = pi^2 E I / (K L)^2`` in newtons.

    ``K = 1`` is pinned-pinned, the idealisation used for the printed rod. Real printed
    members have imperfection and layer-direction weaknesses, so this is an upper bound,
    never a validated working load.
    """
    modulus = _positive("modulus_mpa", modulus_mpa)
    inertia = _positive("second_moment_mm4", second_moment_mm4)
    span = _positive("span_mm", span_mm)
    factor = _positive("effective_length_factor", effective_length_factor)
    return math.pi**2 * modulus * inertia / (factor * span) ** 2


def square_tube_second_moment_mm4(
    across_flats_mm: float, wall_mm: float
) -> float:
    """Second moment of a square tube about a centroidal axis."""
    outer = _positive("across_flats_mm", across_flats_mm)
    wall = _positive("wall_mm", wall_mm)
    inner = outer - 2 * wall
    if inner <= 0:
        raise GeometryError("wall thickness leaves no bore")
    return (outer**4 - inner**4) / 12.0


def horn_radius_for_travel_mm(travel_mm: float, sweep_deg: float) -> float:
    """Horn radius that turns a linear throttle travel into a given angular sweep."""
    travel = _non_negative("travel_mm", travel_mm)
    sweep = _positive("sweep_deg", sweep_deg)
    if sweep >= 180:
        raise GeometryError("sweep must be under 180 degrees")
    return travel / math.radians(sweep)
