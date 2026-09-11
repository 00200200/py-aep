"""Separating and recombining Position must move the values, not just a bit.

`transform_unseparated.aep` and `transform_separated.aep` are the same AE
project one setting apart (same comp, geometry, fps, layer name and layer id
15, AE build 25.6x101), so py-aep's separate must reproduce the second from
the first. Flipping only the tdsb bit left the followers dead, and AE then
ignored the separation entirely.
"""

from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest
from helpers import parse_project_fresh

from py_aep import parse as parse_aep
from py_aep.enums import KeyframeInterpolationType
from py_aep.models.items.composition import CompItem
from py_aep.models.properties.property import Property

SAMPLES_DIR = Path(__file__).parent.parent.parent / "samples" / "models" / "property"


def _position(comp: CompItem) -> Property:
    return comp.layers[0].transform["ADBE Position"]


def _state(comp: CompItem) -> dict[str, object]:
    position = _position(comp)
    return {
        "separated": position.dimensions_separated,
        "leader_stored": position._read_cdat_raw(),
        "followers": [
            (
                position.get_separation_follower(dim).value,
                position.get_separation_follower(dim)._is_live(),
            )
            for dim in range(3)
        ],
    }


class TestSeparate:
    def test_reproduces_the_ae_authored_separated_file(self, tmp_path: Path) -> None:
        expected = _state(
            parse_aep(SAMPLES_DIR / "transform_separated.aep").project.compositions[0]
        )

        project = parse_project_fresh(SAMPLES_DIR / "transform_unseparated.aep")
        _position(project.compositions[0]).dimensions_separated = True
        out = tmp_path / "separated.aep"
        project.save(out)

        assert _state(parse_aep(out).project.compositions[0]) == expected

    def test_followers_take_the_leader_components(self) -> None:
        project = parse_project_fresh(SAMPLES_DIR / "transform_unseparated.aep")
        position = _position(project.compositions[0])
        assert position.value == pytest.approx([960.0, 540.0, 0.0])

        position.dimensions_separated = True

        assert position.get_separation_follower(0).value == pytest.approx(960.0)
        assert position.get_separation_follower(1).value == pytest.approx(540.0)

    def test_z_stays_synthetic_on_a_2d_layer(self) -> None:
        """AE materializes X and Y but leaves Z alone unless the layer is 3D."""
        project = parse_project_fresh(SAMPLES_DIR / "transform_unseparated.aep")
        position = _position(project.compositions[0])

        position.dimensions_separated = True

        assert position.get_separation_follower(0)._is_live()
        assert position.get_separation_follower(1)._is_live()
        assert not position.get_separation_follower(2)._is_live()

    def test_leader_is_parked_on_its_default(self) -> None:
        """While separated the leader is dead, and AE stores the default
        there rather than the old position."""
        project = parse_project_fresh(SAMPLES_DIR / "transform_unseparated.aep")
        comp = project.compositions[0]
        position = _position(comp)
        position.get_separation_follower(0)  # force follower creation
        position.value = [111.0, 222.0, 0.0]

        position.dimensions_separated = True

        assert position._read_cdat_raw() == pytest.approx(
            [comp.width / 2, comp.height / 2, 0.0]
        )
        # ...while the reported value still tracks the followers.
        assert position.value == pytest.approx([111.0, 222.0, 0.0])


class TestRecombine:
    def test_folds_the_followers_back_into_the_leader(self, tmp_path: Path) -> None:
        project = parse_project_fresh(SAMPLES_DIR / "transform_separated.aep")
        position = _position(project.compositions[0])
        position.get_separation_follower(0).value = 111.0
        position.get_separation_follower(1).value = 222.0

        position.dimensions_separated = False

        out = tmp_path / "recombined.aep"
        project.save(out)
        reloaded = _position(parse_aep(out).project.compositions[0])
        assert reloaded.dimensions_separated is False
        assert reloaded.value == pytest.approx([111.0, 222.0, 0.0])

    def test_followers_revert_to_dead(self, tmp_path: Path) -> None:
        project = parse_project_fresh(SAMPLES_DIR / "transform_separated.aep")
        position = _position(project.compositions[0])
        assert position.get_separation_follower(0)._is_live()

        position.dimensions_separated = False

        out = tmp_path / "recombined.aep"
        project.save(out)
        reloaded = _position(parse_aep(out).project.compositions[0])
        for dimension in range(3):
            assert not reloaded.get_separation_follower(dimension)._is_live()


class TestGuards:
    def test_is_idempotent(self) -> None:
        """AE's dimensionsSeparated is idempotent. Re-running the transfer
        would re-seed the followers from a leader the first call already
        reset, destroying the separated values."""
        project = parse_project_fresh(SAMPLES_DIR / "transform_separated.aep")
        position = _position(project.compositions[0])
        position.get_separation_follower(0).value = 777.0

        position.dimensions_separated = True

        assert position.get_separation_follower(0).value == pytest.approx(777.0)

    def test_non_leader_is_rejected(self) -> None:
        """AE silently ignores this; py-aep raises."""
        project = parse_project_fresh(SAMPLES_DIR / "transform_separated.aep")
        follower = _position(project.compositions[0]).get_separation_follower(0)
        with pytest.raises(ValueError, match="separation leader"):
            follower.dimensions_separated = True

    def test_animated_recombine_keeps_the_keyframes(self, tmp_path: Path) -> None:
        """The old flag-only write orphaned the follower keyframes and left
        the leader on a stale static value - the animation vanished.

        The leader lands on the UNION of the followers' times. X and Y do
        not share a timeline in this fixture (X keys at 0/5/9, Y at 0/3/9) -
        that is the point of separating them - so folding the two lists
        together positionally would put X's t=5 value beside Y's t=3 one
        and drop a key.
        """
        project = parse_project_fresh(SAMPLES_DIR / "keyframe_separated_dimensions.aep")
        position = _position(project.compositions[0])
        followers = [position.get_separation_follower(dim) for dim in range(3)]
        x_times = [k.time for k in followers[0].keyframes]
        y_times = [k.time for k in followers[1].keyframes]
        assert x_times != y_times
        expected_times = sorted(set(x_times) | set(y_times))
        expected = [
            [follower.value_at_time(time) for follower in followers]
            for time in expected_times
        ]

        position.dimensions_separated = False

        out = tmp_path / "recombined.aep"
        project.save(out)
        reloaded = _position(parse_aep(out).project.compositions[0])
        assert reloaded.dimensions_separated is False
        assert [k.time for k in reloaded.keyframes] == pytest.approx(expected_times)
        for index, keyframe in enumerate(reloaded.keyframes):
            composed = cast("list[float]", keyframe.value)
            assert composed[:2] == pytest.approx(expected[index][:2])


class TestAnimatedSeparation:
    """The keyframe transfer AE performs, measured on AE 2026.

    AE turns each spatial tangent into a per-dimension ease with
    `speed = |tangent| * 100` and `influence = 1/dt` percent. The numbers
    below come from a probe over segment durations of 0.5, 2.5, 5 and 12 s -
    the last needing influence 0.0833, below AE's own dialog minimum.
    """

    TIMES = [0.0, 0.5, 3.0, 8.0, 20.0]
    VALUES = [
        [50.0, 50.0, 0.0],
        [150.0, 90.0, 0.0],
        [300.0, 60.0, 0.0],
        [420.0, 180.0, 0.0],
        [560.0, 100.0, 0.0],
    ]
    TANGENTS = [
        ([0.0, 0.0, 0.0], [12.0, 5.0, 0.0]),
        ([-30.0, -8.0, 0.0], [44.0, 16.0, 0.0]),
        ([-21.0, -9.0, 0.0], [33.0, 27.0, 0.0]),
        ([-18.0, -24.0, 0.0], [60.0, 12.0, 0.0]),
        ([-15.0, -7.0, 0.0], [0.0, 0.0, 0.0]),
    ]

    def _authored_leader(self) -> Property:
        project = parse_project_fresh(SAMPLES_DIR / "effect_point_speed.aep")
        position = _position(project.compositions[0])
        while position.keyframes:
            position.remove_key(0)
        for time, value in zip(self.TIMES, self.VALUES):
            position.add_key(time)
            position.keyframes[-1].value = list(value)
        for keyframe in position.keyframes:
            keyframe.in_interpolation_type = KeyframeInterpolationType.BEZIER
            keyframe.out_interpolation_type = KeyframeInterpolationType.BEZIER
        for keyframe, (incoming, outgoing) in zip(position.keyframes, self.TANGENTS):
            keyframe.in_spatial_tangent = list(incoming)
            keyframe.out_spatial_tangent = list(outgoing)
        return position

    def test_follower_ease_matches_after_effects(self) -> None:
        position = self._authored_leader()

        position.dimensions_separated = True

        follower = position.get_separation_follower(0)
        # (in speed, in influence, out speed, out influence) per keyframe.
        expected = [
            (0.0, 100.0 / 6.0, 1200.0, 2.0),
            (3000.0, 2.0, 4400.0, 0.4),
            (2100.0, 0.4, 3300.0, 0.2),
            (1800.0, 0.2, 6000.0, 1.0 / 12.0),
            (1500.0, 1.0 / 12.0, 0.0, 100.0 / 6.0),
        ]
        for keyframe, wanted in zip(follower.keyframes, expected):
            assert keyframe.in_temporal_ease[0].speed == pytest.approx(wanted[0])
            assert keyframe.in_temporal_ease[0].influence == pytest.approx(wanted[1])
            assert keyframe.out_temporal_ease[0].speed == pytest.approx(wanted[2])
            assert keyframe.out_temporal_ease[0].influence == pytest.approx(wanted[3])
            assert keyframe.temporal_continuous is True

    def test_leader_keyframes_move_to_the_followers(self) -> None:
        position = self._authored_leader()

        position.dimensions_separated = True

        assert position.keyframes == []
        for dimension in range(2):
            follower = position.get_separation_follower(dimension)
            assert len(follower.keyframes) == len(self.TIMES)
            assert [k.time for k in follower.keyframes] == pytest.approx(self.TIMES)

    def test_round_trip_is_lossless(self) -> None:
        """Values and tangents come back exactly, which is also what AE does."""
        position = self._authored_leader()

        position.dimensions_separated = True
        position.dimensions_separated = False

        assert len(position.keyframes) == len(self.TIMES)
        for index, keyframe in enumerate(position.keyframes):
            assert keyframe.time == pytest.approx(self.TIMES[index])
            assert keyframe.value == pytest.approx(self.VALUES[index])
            assert keyframe.in_spatial_tangent == pytest.approx(self.TANGENTS[index][0])
            assert keyframe.out_spatial_tangent == pytest.approx(
                self.TANGENTS[index][1]
            )


class TestLeaderIsNotWritable:
    """A separated leader refuses writes, as After Effects does.

    Probed on AE 2026: `setValue` and `setValueAtTime` on a separated
    Position both raise ("Can not 'set value' with this property, because
    the property or a parent property is hidden"), while the X / Y / Z
    followers accept writes. py-aep used to accept the write silently and
    park it in the leader's dead `cdat`, where nothing ever read it back.
    """

    def test_value_write_is_rejected(self) -> None:
        position = _position(
            parse_project_fresh(SAMPLES_DIR / "transform_separated.aep").compositions[0]
        )
        assert position.dimensions_separated is True
        with pytest.raises(ValueError, match="dimensions are separated"):
            position.value = [11.0, 22.0, 0.0]

    def test_set_value_at_time_is_rejected(self) -> None:
        position = _position(
            parse_project_fresh(SAMPLES_DIR / "transform_separated.aep").compositions[0]
        )
        with pytest.raises(ValueError, match="dimensions are separated"):
            position.set_value_at_time(1.0, [11.0, 22.0, 0.0])

    def test_rejected_write_changes_nothing(self) -> None:
        comp = parse_project_fresh(
            SAMPLES_DIR / "transform_separated.aep"
        ).compositions[0]
        before = _state(comp)
        with pytest.raises(ValueError):
            _position(comp).value = [11.0, 22.0, 0.0]
        assert _state(comp) == before

    def test_unseparated_leader_is_still_writable(self) -> None:
        """The guard must not leak to an ordinary Position."""
        position = _position(
            parse_project_fresh(SAMPLES_DIR / "transform_unseparated.aep").compositions[
                0
            ]
        )
        assert position.dimensions_separated is False
        position.value = [11.0, 22.0, 0.0]
        assert position.value == pytest.approx([11.0, 22.0, 0.0])

    def test_followers_take_the_write_instead(self, tmp_path: Path) -> None:
        """What AE tells you to do instead, and it round-trips."""
        project = parse_project_fresh(SAMPLES_DIR / "transform_separated.aep")
        position = _position(project.compositions[0])
        position.get_separation_follower(0).value = 11.0
        position.get_separation_follower(1).value = 22.0

        assert position.value == pytest.approx([11.0, 22.0, 0.0])
        out = tmp_path / "modified.aep"
        project.save(out)

        reparsed = _position(parse_aep(out).project.compositions[0])
        assert reparsed.dimensions_separated is True
        assert reparsed.value == pytest.approx([11.0, 22.0, 0.0])


class TestSeparatedCompensation:
    """`Layer.parent` compensates a separated Position through the followers.

    AE refuses the scripting write but does exactly this itself on a
    reparent, so the private path stays even though the public setter does
    not (`Property._set_separated_value`).
    """

    def test_internal_write_reaches_the_followers(self) -> None:
        comp = parse_project_fresh(
            SAMPLES_DIR / "transform_separated.aep"
        ).compositions[0]
        position = _position(comp)
        parked = position._read_cdat_raw()

        position._set_separated_value(
            [11.0, 22.0, 0.0], cast("list[Property]", position._separation_followers())
        )

        assert position.value == pytest.approx([11.0, 22.0, 0.0])
        # The dead leader stays on the default AE parks it on.
        assert position._read_cdat_raw() == pytest.approx(parked)

    def test_untouched_z_stays_synthetic_on_a_2d_layer(self) -> None:
        """Only changed components are written, so a Z that does not move is
        not materialized - matching what AE writes for a 2D layer."""
        comp = parse_project_fresh(
            SAMPLES_DIR / "transform_separated.aep"
        ).compositions[0]
        position = _position(comp)
        assert position.get_separation_follower(2)._is_live() is False

        position._set_separated_value(
            [11.0, 22.0, 0.0], cast("list[Property]", position._separation_followers())
        )

        assert position.get_separation_follower(2)._is_live() is False
