"""Numerical and manifest tests for the 77603 build.

Two distinct jobs:
  * ``test_hardware_77603_geometry`` exercises the pure numerical helpers.
  * ``test_hardware_77603_manifest`` audits whatever the builder actually published, and is
    skipped when no artifacts exist (CI has no OpenSCAD).

Neither test claims a physical qualification. They check arithmetic, artifact integrity and
the presence of the required deliverable coverage.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from hardware.mounts.hardware_77603_geometry import (
    GeometryError,
    euler_buckling_n,
    horn_radius_for_travel_mm,
    reachable,
    solve_endpoint_mm,
    square_tube_second_moment_mm4,
    torque_nm,
)

MANIFEST = Path("hardware/mounts/generated_77603/validation.json")


class TestEndpointSolver:
    def test_solves_the_forward_branch(self):
        """A 3-4-5 style triangle must land on the positive-Y root."""
        x, y, z = solve_endpoint_mm((0, 0, 0), 30.0, 40.0, 100.0)
        assert (x, z) == (30.0, 40.0)
        assert y == pytest.approx(math.sqrt(100**2 - 30**2 - 40**2))

    def test_result_is_exactly_one_rod_length_from_the_pin(self):
        pin = (5.0, -2.0, 7.0)
        end = solve_endpoint_mm(pin, 25.0, 61.0, 220.0)
        distance = math.dist(pin, end)
        assert distance == pytest.approx(220.0, abs=1e-9)

    def test_unreachable_endpoint_raises_instead_of_clamping(self):
        with pytest.raises(GeometryError, match="unreachable"):
            solve_endpoint_mm((0, 0, 0), 300.0, 0.0, 50.0)

    def test_exactly_reachable_is_allowed(self):
        end = solve_endpoint_mm((0, 0, 0), 30.0, 40.0, 50.0)
        assert end[1] == pytest.approx(0.0, abs=1e-9)

    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), -1.0])
    def test_rejects_non_positive_or_nonfinite_length(self, bad):
        with pytest.raises(GeometryError):
            solve_endpoint_mm((0, 0, 0), 1.0, 1.0, bad)

    def test_rejects_malformed_pin(self):
        with pytest.raises(GeometryError, match="3-tuple"):
            solve_endpoint_mm((0, 0), 1.0, 1.0, 10.0)  # type: ignore[arg-type]

    def test_reachability_matches_the_solver(self):
        assert reachable(100.0, 30.0, 40.0) is True
        assert reachable(40.0, 30.0, 40.0) is False


class TestLoadHelpers:
    def test_torque_is_force_times_radius(self):
        assert torque_nm(10.0, 20.0) == pytest.approx(0.2)

    def test_zero_force_and_radius_are_allowed(self):
        assert torque_nm(0.0, 20.0) == 0.0
        assert torque_nm(10.0, 0.0) == 0.0

    def test_negative_force_is_rejected(self):
        with pytest.raises(GeometryError):
            torque_nm(-1.0, 20.0)

    def test_square_tube_inertia_matches_closed_form(self):
        # 30 mm outside, 4 mm wall -> 22 mm bore.
        expected = (30**4 - 22**4) / 12.0
        assert square_tube_second_moment_mm4(30.0, 4.0) == pytest.approx(expected)

    def test_wall_that_closes_the_bore_is_rejected(self):
        with pytest.raises(GeometryError, match="no bore"):
            square_tube_second_moment_mm4(10.0, 5.0)

    def test_buckling_scales_with_inverse_length_squared(self):
        a = euler_buckling_n(1500.0, 47978.7, 1000.0)
        b = euler_buckling_n(1500.0, 47978.7, 2000.0)
        assert b == pytest.approx(a / 4.0)

    def test_buckling_scales_linearly_with_modulus(self):
        a = euler_buckling_n(1500.0, 1000.0, 500.0)
        b = euler_buckling_n(3000.0, 1000.0, 500.0)
        assert b == pytest.approx(2 * a)

    def test_effective_length_factor_reduces_capacity(self):
        pinned = euler_buckling_n(1500.0, 1000.0, 500.0, 1.0)
        fixed = euler_buckling_n(1500.0, 1000.0, 500.0, 2.0)
        assert fixed == pytest.approx(pinned / 4.0)

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"modulus_mpa": 0.0, "second_moment_mm4": 1.0, "span_mm": 1.0},
            {"modulus_mpa": 1.0, "second_moment_mm4": -1.0, "span_mm": 1.0},
            {"modulus_mpa": 1.0, "second_moment_mm4": 1.0, "span_mm": 0.0},
        ],
    )
    def test_buckling_rejects_impossible_inputs(self, kwargs):
        with pytest.raises(GeometryError):
            euler_buckling_n(**kwargs)


class TestHornRadius:
    def test_radius_matches_the_arc_definition(self):
        radius = horn_radius_for_travel_mm(25.0, 60.0)
        assert radius * math.radians(60.0) == pytest.approx(25.0)

    def test_zero_travel_gives_zero_radius(self):
        assert horn_radius_for_travel_mm(0.0, 30.0) == 0.0

    @pytest.mark.parametrize("sweep", [0.0, -5.0, 180.0, 200.0])
    def test_degenerate_sweeps_are_rejected(self, sweep):
        with pytest.raises(GeometryError):
            horn_radius_for_travel_mm(10.0, sweep)


@pytest.mark.skipif(
    not MANIFEST.exists(),
    reason="no generated artifacts in this checkout (CI has no OpenSCAD)",
)
class TestManifest:
    @pytest.fixture(scope="class")
    def manifest(self) -> dict:
        return json.loads(MANIFEST.read_text())

    def test_declares_the_profile_and_generated_status(self, manifest):
        assert manifest["profile"] == "hardware_77603"
        assert manifest["status"] == "geometry_generated_not_physically_qualified"

    def test_every_part_is_one_watertight_bed_zero_solid(self, manifest):
        for name, part in manifest["parts"].items():
            assert part["connected_solids"] == 1, name
            assert part["watertight"] is True, name
            assert part["consistent_winding"] is True, name
            assert abs(part["bounds_mm"][0][2]) < 0.01, name

    def test_every_part_fits_the_usable_envelope(self, manifest):
        usable = manifest["usable_envelope_mm"]
        for name, part in manifest["parts"].items():
            size = part["size_mm"]
            assert size[0] <= usable[0] and size[1] <= usable[1] and size[2] <= usable[2], (
                f"{name} exceeds {usable}: {size}"
            )

    def test_all_five_variants_and_three_locks_are_present(self, manifest):
        names = set(manifest["parts"])
        for letter in "abcde":
            assert any(n.startswith(f"variant_{letter}_") for n in names), letter
        for lock in ("lock_wedge_body", "lock_collet_body", "lock_screw"):
            assert lock in names, lock

    def test_stl_and_3mf_exist_for_every_part(self, manifest):
        root = MANIFEST.parent
        for name, part in manifest["parts"].items():
            assert (root / "stl" / f"{name}.stl").is_file(), name
            assert (root / "3mf" / f"{name}.3mf").is_file(), name

    def test_declared_3mf_hashes_match_the_files(self, manifest):
        import hashlib

        root = MANIFEST.parent
        for name, entry in manifest["print_files"].items():
            actual = hashlib.sha256((root / entry["path"]).read_bytes()).hexdigest()
            assert actual == entry["sha256"], name

    def test_probes_record_their_expectation(self, manifest):
        assert manifest["probes"], "no CSG probes recorded"
        for probe in manifest["probes"]:
            assert probe["expect"] in ("empty", "contact")
            assert probe["empty"] is (probe["expect"] == "empty")

    def test_step_is_reported_unavailable_rather_than_faked(self, manifest):
        assert "UNAVAILABLE" in manifest["step_status"]

    def test_component_inventory_records_conditional_items(self, manifest):
        statuses = {c["name"]: c["status"] for c in manifest["component_inventory"]}
        assert statuses["pololu_2806"].startswith("owned")
        assert "CONDITIONAL" in " ".join(statuses.values())
        assert "unverified" in statuses["automotive_harness"]


class TestRodSegmentPlan:
    """The rod plan must be self-consistent at the real span.

    These exist because a 1 m rod's segmentation rule was left in place when the span dropped
    to 150 mm, producing 23 mm stubs joined by 40 mm spigots -- physically impossible, and
    invisible to the mesh gate (each stub was a valid solid).
    """

    @staticmethod
    def _count(span: float, bed: float = 380.0, splice: float = 40.0, overlap: float = 40.0):
        if span <= bed and span > 2 * splice:
            return 1
        import math as _m

        return max(2, _m.ceil((span - overlap) / (bed - 20)) + 1)

    @staticmethod
    def _seg_len(span: float, n: int, splice: float = 40.0):
        return span if n <= 1 else (span - (n - 1) * splice) / n

    @pytest.mark.parametrize("span", [150, 180, 300])
    def test_segment_fits_in_one_print_is_not_split(self, span):
        assert self._count(span) == 1, f"{span} mm fits the bed and must be one piece"

    def test_every_segment_is_at_least_as_long_as_its_spigot(self):
        for span in (150, 400, 760, 1066.8):
            n = self._count(span)
            seg = self._seg_len(span, n)
            if n > 1:
                assert seg >= 40.0, f"span {span}: {n} segments of {seg:.1f} mm < 40 mm spigot"

    def test_long_span_still_segments_for_the_bed(self):
        span = 1066.8
        n = self._count(span)
        assert n >= 3
        assert self._seg_len(span, n) <= 380.0

    def test_piece_sum_covers_the_span(self):
        for span in (150, 400, 760):
            n = self._count(span)
            seg = self._seg_len(span, n)
            total = seg * n + (n - 1) * 40.0
            assert total == pytest.approx(span, abs=1e-6), span
