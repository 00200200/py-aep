"""Auto-bezier derivation rules, measured on AE 2018 and AE 2026.

After Effects recomputes an auto-bezier keyframe's tangents and ease from the
flag and ignores whatever is stored, so its own files legitimately carry stale
or zero values on those keyframes. Every expectation here is an AE measurement,
not a restatement of the implementation - see
`.claude/plans/write-path-companion-state.md`.
"""

from __future__ import annotations

import pytest

from py_aep.resolvers.interpolation import (
    _DEFAULT_INFLUENCE,
    auto_spatial_tangents,
    auto_temporal_speeds,
)


class TestAutoSpatialTangents:
    """Tangent is +/- one sixth of the chord between the neighbours."""

    def test_interior_uses_chord_not_time_weighting(self) -> None:
        """The discriminating case: neighbours that are NOT equidistant in
        time. AE gives +/-(600-0)/6; the time-weighted rule this replaced
        would give in=[-50] out=[150]."""
        values = [[0.0, 0.0], [100.0, 0.0], [600.0, 0.0]]
        out_tangent, in_tangent = auto_spatial_tangents(values, 1)
        assert out_tangent == pytest.approx([100.0, 0.0])
        assert in_tangent == pytest.approx([-100.0, 0.0])

    def test_endpoints_use_the_single_neighbour_chord(self) -> None:
        """AE writes BOTH tangents at an endpoint, from the chord to its one
        neighbour: k0 gets +/-(100-0)/6, k2 gets +/-(600-100)/6."""
        values = [[0.0, 0.0], [100.0, 0.0], [600.0, 0.0]]
        first_out, first_in = auto_spatial_tangents(values, 0)
        assert first_out == pytest.approx([100.0 / 6.0, 0.0])
        assert first_in == pytest.approx([-100.0 / 6.0, 0.0])

        last_out, last_in = auto_spatial_tangents(values, 2)
        assert last_out == pytest.approx([500.0 / 6.0, 0.0])
        assert last_in == pytest.approx([-500.0 / 6.0, 0.0])

    def test_two_keyframes_share_the_same_chord(self) -> None:
        values = [[0.0, 0.0], [300.0, 0.0]]
        for index in (0, 1):
            out_tangent, in_tangent = auto_spatial_tangents(values, index)
            assert out_tangent == pytest.approx([50.0, 0.0])
            assert in_tangent == pytest.approx([-50.0, 0.0])

    def test_three_dimensional(self) -> None:
        values = [[0.0, 0.0, 0.0], [100.0, 50.0, 20.0], [600.0, 300.0, 90.0]]
        out_tangent, in_tangent = auto_spatial_tangents(values, 1)
        assert out_tangent == pytest.approx([100.0, 50.0, 15.0])
        assert in_tangent == pytest.approx([-100.0, -50.0, -15.0])

    def test_single_keyframe_has_no_tangents(self) -> None:
        out_tangent, in_tangent = auto_spatial_tangents([[10.0, 20.0]], 0)
        assert out_tangent == [0.0, 0.0]
        assert in_tangent == [0.0, 0.0]

    def test_ae2018_diamond(self) -> None:
        """The five-key diamond in samples/versions/ae2018. Every tangent AE
        reports for it is chord/6, including both endpoints."""
        values = [
            [100.0, 540.0, 0.0],
            [960.0, 100.0, 0.0],
            [1820.0, 540.0, 0.0],
            [960.0, 980.0, 0.0],
            [100.0, 540.0, 0.0],
        ]
        expected_out = [
            [143.333, -73.333, 0.0],
            [286.667, 0.0, 0.0],
            [0.0, 146.667, 0.0],
            [-286.667, 0.0, 0.0],
            [-143.333, -73.333, 0.0],
        ]
        for index, expected in enumerate(expected_out):
            out_tangent, in_tangent = auto_spatial_tangents(values, index)
            assert out_tangent == pytest.approx(expected, abs=0.001)
            assert in_tangent == pytest.approx([-c for c in expected], abs=0.001)


class TestAutoTemporalSpeeds:
    """Speed is the through-slope across the neighbours, per dimension."""

    def test_interior_through_slope(self) -> None:
        """AE 2018 Rotate Z, 0 -> 180 -> 360 at t = 0 / 2 / 4: speed 90."""
        values = [[0.0], [180.0], [360.0]]
        times = [0.0, 2.0, 4.0]
        in_speeds, out_speeds = auto_temporal_speeds(values, times, 1)
        assert in_speeds == pytest.approx([90.0])
        assert out_speeds == pytest.approx([90.0])

    def test_endpoints_zero_on_the_outward_side(self) -> None:
        values = [[0.0], [180.0], [360.0]]
        times = [0.0, 2.0, 4.0]
        first_in, first_out = auto_temporal_speeds(values, times, 0)
        assert first_in == pytest.approx([0.0])
        assert first_out == pytest.approx([90.0])

        last_in, last_out = auto_temporal_speeds(values, times, 2)
        assert last_in == pytest.approx([90.0])
        assert last_out == pytest.approx([0.0])

    def test_per_dimension_slope(self) -> None:
        """AE 2026, 2-D Scale 100->200->400 and 100->120->150 over 2 s:
        speeds 150 and 25, NOT one shared scalar."""
        values = [[100.0, 100.0], [200.0, 120.0], [400.0, 150.0]]
        times = [0.0, 1.0, 2.0]
        in_speeds, out_speeds = auto_temporal_speeds(values, times, 1)
        assert in_speeds == pytest.approx([150.0, 25.0])
        assert out_speeds == pytest.approx([150.0, 25.0])

    def test_three_dimensional_scale(self) -> None:
        values = [[100.0, 100.0, 100.0], [200.0, 120.0, 110.0], [400.0, 150.0, 300.0]]
        times = [0.0, 1.0, 2.0]
        in_speeds, _ = auto_temporal_speeds(values, times, 1)
        assert in_speeds == pytest.approx([150.0, 25.0, 100.0])

    def test_two_keyframes(self) -> None:
        """AE 2026, opacity 0 -> 100 over 4 s: slope 25 on the inward side."""
        values = [[0.0], [100.0]]
        times = [0.0, 4.0]
        first_in, first_out = auto_temporal_speeds(values, times, 0)
        assert first_in == pytest.approx([0.0])
        assert first_out == pytest.approx([25.0])

    def test_single_keyframe_has_no_speed(self) -> None:
        in_speeds, out_speeds = auto_temporal_speeds([[42.0]], [1.0], 0)
        assert in_speeds == [0.0]
        assert out_speeds == [0.0]

    def test_zero_span_does_not_divide_by_zero(self) -> None:
        values = [[0.0], [10.0], [20.0]]
        times = [1.0, 1.0, 1.0]
        in_speeds, out_speeds = auto_temporal_speeds(values, times, 1)
        assert in_speeds == [0.0]
        assert out_speeds == [0.0]

    def test_default_influence_is_one_sixth(self) -> None:
        assert _DEFAULT_INFLUENCE == pytest.approx(100.0 / 6.0)
