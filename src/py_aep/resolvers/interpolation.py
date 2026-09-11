"""Interpolation utilities for keyframe-based property evaluation.

Implements HOLD, LINEAR, and BEZIER interpolation for
`Property.value_at_time()`.  Pure Python with no external dependencies.

The algorithms are ported from `lottie-web <https://github.com/airbnb/
lottie-web>`_, the industry-standard renderer for Lottie/bodymovin
animations exported from After Effects:

- **Temporal ease** uses a normalised [0, 1] -> [0, 1] cubic-bezier
  easing function (`BezierEasing`), with an 11-point sample table,
  Newton-Raphson refinement, and binary-subdivision fallback (ported
  from `BezierEaser.js`).
- **Spatial paths** are pre-sampled into a 150-segment polyline with
  per-segment partial lengths (ported from `bez.js`).
- **Arc-length reparameterisation** walks the segment table linearly,
  interpolating between adjacent sample points.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, cast

from py_aep.enums import KeyframeInterpolationType

if TYPE_CHECKING:
    from ..models.properties.keyframe import Keyframe
    from ..models.properties.keyframe_ease import KeyframeEase

_DEFAULT_INFLUENCE = 100.0 / 6.0  # 16.6667 %

# Number of polyline segments for spatial bezier path approximation.
# lottie-web uses getDefaultCurveSegments() which defaults to 150.
_CURVE_SEGMENTS = 150


# ---------------------------------------------------------------------------
# BezierEasing - port of lottie-web 3rd_party/BezierEaser.js
# ---------------------------------------------------------------------------
# Implements a cached unit-square bezier easing function.
# Given control points (cx1, cy1, cx2, cy2), maps x in [0,1] -> y in [0,1].
# Uses precomputed sample table + Newton-Raphson with binary subdivision
# fallback, matching Firefox's nsSMILKeySpline.

_NEWTON_ITERATIONS = 4
_NEWTON_MIN_SLOPE = 0.001
_SUBDIVISION_PRECISION = 0.0000001
_SUBDIVISION_MAX_ITERATIONS = 10
_SAMPLE_TABLE_SIZE = 11
_SAMPLE_STEP_SIZE = 1.0 / (_SAMPLE_TABLE_SIZE - 1.0)


def _bez_A(a1: float, a2: float) -> float:
    return 1.0 - 3.0 * a2 + 3.0 * a1


def _bez_B(a1: float, a2: float) -> float:
    return 3.0 * a2 - 6.0 * a1


def _bez_C(a1: float) -> float:
    return 3.0 * a1


def _calc_bezier(t: float, a1: float, a2: float) -> float:
    """Calculate bezier value at t for one axis (x or y)."""
    return ((_bez_A(a1, a2) * t + _bez_B(a1, a2)) * t + _bez_C(a1)) * t


def _get_slope(t: float, a1: float, a2: float) -> float:
    """Calculate bezier derivative at t for one axis."""
    return 3.0 * _bez_A(a1, a2) * t * t + 2.0 * _bez_B(a1, a2) * t + _bez_C(a1)


def _binary_subdivide(x: float, a: float, b: float, x1: float, x2: float) -> float:
    """Binary subdivision fallback to find t for given x."""
    current_t = 0.0
    for _ in range(_SUBDIVISION_MAX_ITERATIONS):
        current_t = a + (b - a) / 2.0
        current_x = _calc_bezier(current_t, x1, x2) - x
        if current_x > 0.0:
            b = current_t
        else:
            a = current_t
        if abs(current_x) <= _SUBDIVISION_PRECISION:
            break
    return current_t


def _newton_raphson_iterate(x: float, guess_t: float, x1: float, x2: float) -> float:
    """Newton-Raphson iteration to refine t for given x."""
    for _ in range(_NEWTON_ITERATIONS):
        slope = _get_slope(guess_t, x1, x2)
        if slope == 0.0:
            return guess_t
        current_x = _calc_bezier(guess_t, x1, x2) - x
        guess_t -= current_x / slope
    return guess_t


class _BezierEasing:
    """Cached unit-square bezier easing function.

    Port of lottie-web's BezierEasing class.
    """

    __slots__ = ("_cx1", "_cy1", "_cx2", "_cy2", "_samples")

    def __init__(self, cx1: float, cy1: float, cx2: float, cy2: float) -> None:
        self._cx1 = cx1
        self._cy1 = cy1
        self._cx2 = cx2
        self._cy2 = cy2
        # Precompute sample table
        self._samples = [
            _calc_bezier(i * _SAMPLE_STEP_SIZE, cx1, cx2)
            for i in range(_SAMPLE_TABLE_SIZE)
        ]

    def get(self, x: float) -> float:
        """Map x in [0,1] to eased y in [0,1]."""
        # Linear shortcut
        if self._cx1 == self._cy1 and self._cx2 == self._cy2:
            return x
        if x == 0.0:
            return 0.0
        if x == 1.0:
            return 1.0
        return _calc_bezier(self._get_t_for_x(x), self._cy1, self._cy2)

    def _get_t_for_x(self, x: float) -> float:
        """Find t parameter for a given x value."""
        # Find the sample interval
        interval_start = 0.0
        current_sample = 1
        last_sample = _SAMPLE_TABLE_SIZE - 1

        while current_sample != last_sample and self._samples[current_sample] <= x:
            interval_start += _SAMPLE_STEP_SIZE
            current_sample += 1
        current_sample -= 1

        # Interpolate for initial guess
        denom = self._samples[current_sample + 1] - self._samples[current_sample]
        if denom == 0.0:
            dist = 0.0
        else:
            dist = (x - self._samples[current_sample]) / denom
        guess_t = interval_start + dist * _SAMPLE_STEP_SIZE

        initial_slope = _get_slope(guess_t, self._cx1, self._cx2)
        if initial_slope >= _NEWTON_MIN_SLOPE:
            return _newton_raphson_iterate(x, guess_t, self._cx1, self._cx2)
        if initial_slope == 0.0:
            return guess_t
        return _binary_subdivide(
            x,
            interval_start,
            interval_start + _SAMPLE_STEP_SIZE,
            self._cx1,
            self._cx2,
        )


# Cache for easing functions (mirroring lottie-web's BezierFactory cache)
_easing_cache: dict[tuple[float, float, float, float], _BezierEasing] = {}


def _get_bezier_easing(cx1: float, cy1: float, cx2: float, cy2: float) -> _BezierEasing:
    """Get or create a cached BezierEasing instance."""
    key = (cx1, cy1, cx2, cy2)
    cached = _easing_cache.get(key)
    if cached is not None:
        return cached
    easing = _BezierEasing(cx1, cy1, cx2, cy2)
    _easing_cache[key] = easing
    return easing


# ---------------------------------------------------------------------------
# Speed/influence -> bezier control points conversion
# ---------------------------------------------------------------------------
# AEP stores temporal ease as (speed, influence) pairs.
# Lottie JSON stores temporal ease as bezier control points (o.x, o.y, i.x, i.y)
# in a normalized [0,1] coordinate space.
#
# The bodymovin exporter converts from AEP to Lottie format as follows:
#   outgoing: cx1 = out_influence / 100, cy1 = out_speed_normalized
#   incoming: cx2 = 1 - in_influence / 100, cy2 = 1 - in_speed_normalized
#
# Where speed_normalized = speed * influence_fraction * dt / (v1 - v0)
# For spatial properties, speed is in position-units/sec.


def _ease_to_bezier_1d(
    t0: float,
    t1: float,
    v0: float,
    v1: float,
    out_ease: KeyframeEase,
    in_ease: KeyframeEase,
    out_override: tuple[float, float] | None = None,
    in_override: tuple[float, float] | None = None,
) -> tuple[float, float, float, float]:
    """Convert AEP speed/influence ease to normalized bezier control points.

    Returns (cx1, cy1, cx2, cy2) for use with BezierEasing.
    """
    dt = t1 - t0
    dv = v1 - v0

    if out_override is not None:
        o_spd, o_inf = out_override
    else:
        o_spd, o_inf = out_ease.speed, out_ease.influence
    if in_override is not None:
        i_spd, i_inf = in_override
    else:
        i_spd, i_inf = in_ease.speed, in_ease.influence

    # Normalized influence (x-axis control points)
    cx1 = o_inf / 100.0
    cx2 = 1.0 - i_inf / 100.0

    # Normalized speed (y-axis control points)
    # cy1 = outgoing speed * influence_fraction * dt / dv
    # cy2 = 1 - incoming speed * influence_fraction * dt / dv
    if abs(dv) > 1e-12:
        cy1 = o_spd * cx1 * dt / dv
        cy2 = 1.0 - i_spd * (1.0 - cx2) * dt / dv
    else:
        # No value change: linear
        cy1 = cx1
        cy2 = cx2

    return cx1, cy1, cx2, cy2


# ---------------------------------------------------------------------------
# Spatial bezier path - port of lottie-web bez.js buildBezierData
# ---------------------------------------------------------------------------


def _cubic_bezier_nd(
    p0: list[float],
    p1: list[float],
    p2: list[float],
    p3: list[float],
    t: float,
) -> list[float]:
    """Evaluate N-dimensional cubic bezier at parameter t.

    Uses same formula as lottie-web:
    (1-t)^3 * P0 + 3*(1-t)^2*t * P1 + 3*(1-t)*t^2 * P2 + t^3 * P3

    Note: In lottie-web, P1 = pt1 + outTangent (absolute control point),
    P2 = pt2 + inTangent (absolute control point). We follow the same
    convention here.
    """
    u = 1.0 - t
    u2 = u * u
    u3 = u2 * u
    t2 = t * t
    t3 = t2 * t
    return [
        u3 * p0[d] + 3.0 * u2 * t * p1[d] + 3.0 * u * t2 * p2[d] + t3 * p3[d]
        for d in range(len(p0))
    ]


def _point_on_line_2d(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    x3: float,
    y3: float,
) -> bool:
    """Check if (x3,y3) is collinear with segment (x1,y1)->(x2,y2).

    Port of lottie-web's pointOnLine2D.
    """
    det = x1 * y2 + y1 * x3 + x2 * y3 - x3 * y2 - y3 * x1 - x2 * y1
    return -0.001 < det < 0.001


def _point_on_line_3d(
    x1: float,
    y1: float,
    z1: float,
    x2: float,
    y2: float,
    z2: float,
    x3: float,
    y3: float,
    z3: float,
) -> bool:
    """Check if (x3,y3,z3) is collinear with segment in 3D.

    Port of lottie-web's pointOnLine3D.
    """
    if z1 == 0.0 and z2 == 0.0 and z3 == 0.0:
        return _point_on_line_2d(x1, y1, x2, y2, x3, y3)
    dist1 = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2)
    dist2 = math.sqrt((x3 - x1) ** 2 + (y3 - y1) ** 2 + (z3 - z1) ** 2)
    dist3 = math.sqrt((x3 - x2) ** 2 + (y3 - y2) ** 2 + (z3 - z2) ** 2)
    if dist1 > dist2:
        diff = dist1 - dist2 - dist3 if dist1 > dist3 else dist3 - dist2 - dist1
    elif dist3 > dist2:
        diff = dist3 - dist2 - dist1
    else:
        diff = dist2 - dist1 - dist3
    return -0.0001 < diff < 0.0001


class _BezierPathData:
    """Pre-sampled spatial bezier path with arc-length data.

    Port of lottie-web's BezierData / buildBezierData.
    Stores N+1 sample points along the curve with their partial
    lengths, enabling arc-length reparameterization via linear
    interpolation in the segment table.
    """

    __slots__ = ("points", "partial_lengths", "segment_length")

    def __init__(
        self,
        v0: list[float],
        v1: list[float],
        out_tangent: list[float],
        in_tangent: list[float],
    ) -> None:
        ndim = len(v0)
        # Convert to absolute control points (lottie-web convention)
        p1 = [v0[d] + out_tangent[d] for d in range(ndim)]
        p2 = [v1[d] + in_tangent[d] for d in range(ndim)]

        # Check for collinear (straight line) - use 2 segments only
        is_collinear = False
        if ndim == 2 and (v0[0] != v1[0] or v0[1] != v1[1]):
            is_collinear = _point_on_line_2d(
                v0[0], v0[1], v1[0], v1[1], p1[0], p1[1]
            ) and _point_on_line_2d(v0[0], v0[1], v1[0], v1[1], p2[0], p2[1])
        elif ndim == 3 and (v0[0] != v1[0] or v0[1] != v1[1] or v0[2] != v1[2]):
            is_collinear = _point_on_line_3d(
                v0[0],
                v0[1],
                v0[2],
                v1[0],
                v1[1],
                v1[2],
                p1[0],
                p1[1],
                p1[2],
            ) and _point_on_line_3d(
                v0[0],
                v0[1],
                v0[2],
                v1[0],
                v1[1],
                v1[2],
                p2[0],
                p2[1],
                p2[2],
            )

        n_segs = 2 if is_collinear else _CURVE_SEGMENTS

        self.points: list[list[float]] = []
        self.partial_lengths: list[float] = []
        total_length = 0.0
        last_point: list[float] | None = None

        for k in range(n_segs):
            perc = k / (n_segs - 1) if n_segs > 1 else 0.0
            point = _cubic_bezier_nd(v0, p1, p2, v1, perc)

            pt_dist = 0.0
            if last_point is not None:
                pt_dist = math.sqrt(
                    sum((point[d] - last_point[d]) ** 2 for d in range(ndim))
                )
            total_length += pt_dist

            self.points.append(point)
            self.partial_lengths.append(pt_dist)
            last_point = point

        self.segment_length = total_length


def _get_point_on_path(
    bezier_data: _BezierPathData,
    eased_perc: float,
) -> list[float]:
    """Get position along a pre-sampled bezier path at arc-length fraction.

    Port of lottie-web's PropertyFactory interpolateValue (spatial branch).
    Walks the precomputed polyline segments to find the point at the
    given arc-length distance.
    """
    if eased_perc <= 0.0:
        return list(bezier_data.points[0])
    if eased_perc >= 1.0:
        return list(bezier_data.points[-1])

    distance = bezier_data.segment_length * eased_perc
    added_length = 0.0
    n_pts = len(bezier_data.points)

    for j in range(n_pts):
        added_length += bezier_data.partial_lengths[j]

        if distance == 0.0 or eased_perc == 0.0 or j == n_pts - 1:
            return list(bezier_data.points[j])

        if (
            distance >= added_length
            and distance < added_length + bezier_data.partial_lengths[j + 1]
        ):
            seg_perc = (distance - added_length) / bezier_data.partial_lengths[j + 1]
            pt_a = bezier_data.points[j]
            pt_b = bezier_data.points[j + 1]
            return [pt_a[d] + (pt_b[d] - pt_a[d]) * seg_perc for d in range(len(pt_a))]

    # Fallback: return last point
    return list(bezier_data.points[-1])


def _tangents_are_zero(tangent: list[float] | None) -> bool:
    if tangent is None:
        return True
    return all(abs(x) < 1e-12 for x in tangent)


# ---------------------------------------------------------------------------
# Auto-bezier computation (same as original - needed for AEP binary data)
# ---------------------------------------------------------------------------


_AUTO_BEZIER_CHORD_SCALE = 1.0 / 6.0


def auto_spatial_tangents(
    values: list[list[float]], index: int
) -> tuple[list[float], list[float]]:
    """AE's auto-bezier spatial tangents for `values[index]`.

    The tangent is one sixth of the chord between the keyframe's two
    neighbours, applied symmetrically: `in = -out`. An endpoint uses the
    chord to its single neighbour, and still gets both tangents. A lone
    keyframe gets zeros.

    Measured on AE 2018 (a five-key diamond, every tangent matching) and on
    AE 2026 with deliberately uneven time spacing, which rules out the
    time-weighted variant this used to implement - that agrees with AE only
    when the neighbours are equidistant in time.

    The result is expressed in whatever space `values` are in, so callers
    working in raw chunk units and callers working in resolved units both
    get a self-consistent answer.

    Returns:
        An `(out_tangent, in_tangent)` pair.
    """
    ndim = len(values[index])
    zero = [0.0] * ndim
    if len(values) < 2:
        return zero, list(zero)
    low = values[index - 1] if index > 0 else values[index]
    high = values[index + 1] if index < len(values) - 1 else values[index]
    out_tangent = [(high[d] - low[d]) * _AUTO_BEZIER_CHORD_SCALE for d in range(ndim)]
    return out_tangent, [-component for component in out_tangent]


def auto_temporal_speeds(
    values: list[list[float]], times: list[float], index: int
) -> tuple[list[float], list[float]]:
    """AE's auto-bezier temporal speeds for `values[index]`, per dimension.

    An interior keyframe takes the through-slope across its neighbours on
    both sides. An endpoint takes the slope of its single segment on the
    inward side and zero on the outward side. Influence is always
    `_DEFAULT_INFLUENCE`.

    Multi-dimensional properties get a per-dimension slope (measured on AE
    2026: 2-D Scale 100->200->400 / 100->120->150 yields speeds 150 and 25).

    Returns:
        An `(in_speeds, out_speeds)` pair, one entry per dimension.
    """
    ndim = len(values[index])
    zero = [0.0] * ndim
    count = len(values)
    if count < 2:
        return list(zero), list(zero)

    def slope(lo: int, hi: int) -> list[float]:
        span = times[hi] - times[lo]
        if span <= 0:
            return [0.0] * ndim
        return [(values[hi][d] - values[lo][d]) / span for d in range(ndim)]

    if index == 0:
        return list(zero), slope(0, 1)
    if index == count - 1:
        return slope(count - 2, count - 1), list(zero)
    through = slope(index - 1, index + 1)
    return through, list(through)


def roving_keyframe_times(keyframes: list[Keyframe]) -> dict[int, float]:
    """Times for every roving keyframe, spaced by arc length.

    A roving keyframe's time is derived, not stored: AE places it so the
    speed along the spatial path is constant between the enclosing
    non-roving anchors, `t = t_a + (t_b - t_a) * arc_so_far / arc_total`.

    The metric is the length of the actual curve, not the chord between
    keyframes - measured on AE 2026 with one segment bowed by +/-400 px
    tangents, where chord length predicts 0.667 s and AE produced 2.2205 s.
    The 150-sample approximation in `_BezierPathData` reproduces AE's stored
    times to the exact timebase unit for both that case and a flattened one.

    Returns:
        A `{keyframe index: time in seconds}` mapping covering only the
        roving keyframes that sit between two anchors.
    """
    result: dict[int, float] = {}
    anchors = [i for i, kf in enumerate(keyframes) if not kf.roving]
    if len(anchors) < 2:
        return result

    for start, end in zip(anchors, anchors[1:]):
        if end - start < 2:
            continue
        lengths: list[float] = []
        for i in range(start, end):
            first, second = keyframes[i].value, keyframes[i + 1].value
            if not isinstance(first, list) or not isinstance(second, list):
                lengths = []
                break
            out_tangent = keyframes[i].out_spatial_tangent or [0.0] * len(first)
            in_tangent = keyframes[i + 1].in_spatial_tangent or [0.0] * len(second)
            lengths.append(
                _BezierPathData(first, second, out_tangent, in_tangent).segment_length
            )
        total = sum(lengths)
        if not lengths or total <= 0:
            continue
        span_start, span_end = keyframes[start].time, keyframes[end].time
        travelled = 0.0
        for offset, index in enumerate(range(start + 1, end)):
            travelled += lengths[offset]
            result[index] = span_start + (span_end - span_start) * travelled / total
    return result


def _compute_auto_spatial_tangents(
    keyframes: list[Keyframe],
) -> list[tuple[list[float] | None, list[float] | None]]:
    """Resolve spatial tangents for interpolation, deriving auto-bezier ones.

    Unlike the old implementation this does NOT defer to the stored tangents
    when they are non-zero: AE recomputes them from the flag and ignores what
    is on disk, so its own files legitimately carry stale values.
    """
    values = [kf.value for kf in keyframes]
    all_vectors = all(isinstance(value, list) for value in values)
    result: list[tuple[list[float] | None, list[float] | None]] = []
    for i, kf in enumerate(keyframes):
        value = values[i]
        if not kf.spatial_auto_bezier or not isinstance(value, list):
            result.append((kf.out_spatial_tangent, kf.in_spatial_tangent))
            continue
        if not all_vectors:
            result.append(([0.0] * len(value), [0.0] * len(value)))
            continue
        result.append(auto_spatial_tangents(cast("list[list[float]]", values), i))
    return result


def _compute_auto_temporal_ease(
    keyframes: list[Keyframe],
) -> list[tuple[float, float, float, float]]:
    """Compute temporal ease for keyframes with `temporal_auto_bezier`.

    Returns (out_speed, out_influence, in_speed, in_influence) per keyframe.
    """
    n = len(keyframes)
    result: list[tuple[float, float, float, float]] = []

    for i in range(n):
        kf = keyframes[i]
        is_auto = kf.temporal_auto_bezier
        out_e = kf.out_temporal_ease[0] if kf.out_temporal_ease else None
        in_e = kf.in_temporal_ease[0] if kf.in_temporal_ease else None

        out_spd = out_e.speed if out_e else 0.0
        out_inf = out_e.influence if out_e else 0.0
        in_spd = in_e.speed if in_e else 0.0
        in_inf = in_e.influence if in_e else 0.0

        if not is_auto or (out_inf > 0.0 or in_inf > 0.0):
            result.append((out_spd, out_inf, in_spd, in_inf))
            continue

        v = kf.value
        v_scalar = v if isinstance(v, (int, float)) else None

        if v_scalar is None:
            result.append((out_spd, out_inf, in_spd, in_inf))
            continue

        if i == 0 and n > 1:
            nv = keyframes[1].value
            dt = keyframes[1].time - kf.time
            if isinstance(nv, (int, float)) and dt > 0:
                out_spd = (float(nv) - float(v_scalar)) / dt
            out_inf = _DEFAULT_INFLUENCE
            in_inf = _DEFAULT_INFLUENCE
        elif i == n - 1 and n > 1:
            pv = keyframes[i - 1].value
            dt = kf.time - keyframes[i - 1].time
            if isinstance(pv, (int, float)) and dt > 0:
                in_spd = (float(v_scalar) - float(pv)) / dt
            in_inf = _DEFAULT_INFLUENCE
            out_inf = _DEFAULT_INFLUENCE
        else:
            pv = keyframes[i - 1].value
            nv = keyframes[i + 1].value
            dt_total = keyframes[i + 1].time - keyframes[i - 1].time
            if (
                isinstance(pv, (int, float))
                and isinstance(nv, (int, float))
                and dt_total > 0
            ):
                slope = (float(nv) - float(pv)) / dt_total
                out_spd = slope
                in_spd = slope
            out_inf = _DEFAULT_INFLUENCE
            in_inf = _DEFAULT_INFLUENCE

        result.append((out_spd, out_inf, in_spd, in_inf))

    return result


# ---------------------------------------------------------------------------
# Segment interpolation - lottie-web approach
# ---------------------------------------------------------------------------


def _interpolate_bezier_1d(
    t: float,
    t0: float,
    t1: float,
    v0: float,
    v1: float,
    out_ease: KeyframeEase,
    in_ease: KeyframeEase,
    out_override: tuple[float, float] | None = None,
    in_override: tuple[float, float] | None = None,
) -> float:
    """Interpolate 1D using lottie-web's BezierEasing approach.

    Converts speed/influence to normalized bezier control points,
    then uses BezierEasing to get the eased progress, then lerps.
    """
    dt = t1 - t0
    if dt == 0:
        return v0

    # Convert AEP ease to normalized bezier control points
    cx1, cy1, cx2, cy2 = _ease_to_bezier_1d(
        t0, t1, v0, v1, out_ease, in_ease, out_override, in_override
    )

    # Get easing function (cached)
    easing = _get_bezier_easing(cx1, cy1, cx2, cy2)

    # Normalized time fraction
    x = (t - t0) / dt

    # Eased progress
    perc = easing.get(x)

    # Linear interpolation in value space
    return v0 + (v1 - v0) * perc


def _interpolate_spatial_bezier(
    t: float,
    t0: float,
    t1: float,
    kf0: Keyframe,
    kf1: Keyframe,
    out_tan_override: list[float] | None = None,
    in_tan_override: list[float] | None = None,
) -> list[float]:
    """Interpolate spatial property using lottie-web's approach.

    1. Builds a polyline approximation of the spatial bezier path
    2. Converts temporal ease to a BezierEasing function
    3. Maps time -> eased arc-length fraction
    4. Walks the polyline to find the position
    """
    v0 = kf0.value
    v1 = kf1.value
    if not isinstance(v0, list) or not isinstance(v1, list):
        if isinstance(v0, list):
            return [float(x) for x in v0]
        if isinstance(v0, (int, float)):
            return [float(v0)]
        return []

    dt = t1 - t0
    if dt == 0:
        return list(v0)

    ndim = len(v0)
    out_tan = out_tan_override or kf0.out_spatial_tangent or [0.0] * ndim
    in_tan = in_tan_override or kf1.in_spatial_tangent or [0.0] * ndim
    straight_path = _tangents_are_zero(out_tan) and _tangents_are_zero(in_tan)

    # Build temporal easing function
    out_ease = kf0.out_temporal_ease[0] if kf0.out_temporal_ease else None
    in_ease = kf1.in_temporal_ease[0] if kf1.in_temporal_ease else None

    has_explicit_ease = (
        out_ease is not None
        and in_ease is not None
        and (out_ease.influence > 0 or in_ease.influence > 0)
    )

    if has_explicit_ease:
        assert out_ease is not None and in_ease is not None

        # For spatial properties, we convert speed/influence to normalized
        # bezier control points using the segment distance
        if straight_path:
            seg_dist = math.sqrt(sum((v1[d] - v0[d]) ** 2 for d in range(ndim)))
            path_deriv_mag_0 = seg_dist
            path_deriv_mag_1 = seg_dist
        else:
            p1 = [v0[d] + out_tan[d] for d in range(ndim)]
            p2 = [v1[d] + in_tan[d] for d in range(ndim)]
            path_deriv_mag_0 = 3.0 * math.sqrt(
                sum((p1[d] - v0[d]) ** 2 for d in range(ndim))
            )
            path_deriv_mag_1 = 3.0 * math.sqrt(
                sum((v1[d] - p2[d]) ** 2 for d in range(ndim))
            )

        # Convert to normalized speed (progress/sec -> progress fraction)
        out_inf_frac = out_ease.influence / 100.0
        in_inf_frac = in_ease.influence / 100.0

        ds_dt_0 = out_ease.speed / path_deriv_mag_0 if path_deriv_mag_0 > 1e-12 else 0.0
        ds_dt_1 = in_ease.speed / path_deriv_mag_1 if path_deriv_mag_1 > 1e-12 else 0.0

        # Bezier control points in normalized (time_frac, progress) space
        cx1 = out_inf_frac
        cy1 = ds_dt_0 * out_inf_frac * dt

        cx2 = 1.0 - in_inf_frac
        cy2 = 1.0 - ds_dt_1 * in_inf_frac * dt

        cy1 = max(0.0, min(1.0, cy1))
        cy2 = max(0.0, min(1.0, cy2))

        easing = _get_bezier_easing(cx1, cy1, cx2, cy2)
        x = (t - t0) / dt
        perc = easing.get(x)
    else:
        perc = (t - t0) / dt

    # For straight paths, just lerp
    if straight_path:
        return [v0[d] + (v1[d] - v0[d]) * perc for d in range(ndim)]

    # Build spatial bezier path data (polyline approximation)
    # In a full implementation this would be cached; for correctness
    # comparison we build it each time.
    bezier_data = _BezierPathData(v0, v1, out_tan, in_tan)

    # Walk the path to get the position at the eased arc-length fraction
    return _get_point_on_path(bezier_data, perc)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def interpolate_keyframes(
    time: float,
    keyframes: list[Keyframe],
    is_spatial: bool,
) -> list[float] | float | None:
    """Compute the interpolated value at `time` from a keyframe list.

    Uses lottie-web's algorithms:
    - BezierEasing for temporal ease (normalized unit-square bezier)
    - Polyline path approximation for spatial bezier
    - Linear arc-length interpolation along sampled path

    Args:
        time: Time in seconds.
        keyframes: Sorted list of keyframes.
        is_spatial: Whether the property is spatial.

    Returns:
        Interpolated value, or `None` if no keyframes.
    """
    if not keyframes:
        return None

    n = len(keyframes)

    # Before first keyframe or single keyframe
    if n == 1 or time <= keyframes[0].time:
        return cast("list[float] | float | None", keyframes[0].value)

    # After last keyframe
    if time >= keyframes[-1].time:
        return cast("list[float] | float | None", keyframes[-1].value)

    # Find the segment
    right_idx = 0
    for i in range(1, n):
        if keyframes[i].time >= time:
            right_idx = i
            break

    left_idx = right_idx - 1
    kf_left = keyframes[left_idx]
    kf_right = keyframes[right_idx]

    t0 = kf_left.time
    t1 = kf_right.time

    # At exactly a keyframe time
    if abs(time - t0) < 1e-12:
        return cast("list[float] | float | None", kf_left.value)
    if abs(time - t1) < 1e-12:
        return cast("list[float] | float | None", kf_right.value)

    # Determine interpolation type
    out_type = kf_left.out_interpolation_type
    in_type = kf_right.in_interpolation_type

    # HOLD
    if out_type == KeyframeInterpolationType.HOLD:
        return cast("list[float] | float | None", kf_left.value)
    if in_type == KeyframeInterpolationType.HOLD:
        return cast("list[float] | float | None", kf_right.value)

    v0 = kf_left.value
    v1 = kf_right.value

    # LINEAR
    if out_type == KeyframeInterpolationType.LINEAR:
        ratio = (time - t0) / (t1 - t0)
        if isinstance(v0, list) and isinstance(v1, list):
            return [v0[d] + (v1[d] - v0[d]) * ratio for d in range(len(v0))]
        if isinstance(v0, (int, float)) and isinstance(v1, (int, float)):
            return float(v0) + (float(v1) - float(v0)) * ratio
        return cast("list[float] | float | None", v0)

    # Pre-compute auto-bezier tangents/ease
    auto_tangents: list[tuple[list[float] | None, list[float] | None]] | None
    auto_ease: list[tuple[float, float, float, float]] | None

    has_auto_spatial = is_spatial and any(kf.spatial_auto_bezier for kf in keyframes)
    auto_tangents = (
        _compute_auto_spatial_tangents(keyframes) if has_auto_spatial else None
    )

    has_auto_temporal = any(kf.temporal_auto_bezier for kf in keyframes)
    auto_ease = _compute_auto_temporal_ease(keyframes) if has_auto_temporal else None

    # BEZIER
    if out_type == KeyframeInterpolationType.BEZIER:
        if is_spatial and isinstance(v0, list) and isinstance(v1, list):
            out_tan_ov: list[float] | None = None
            in_tan_ov: list[float] | None = None
            if auto_tangents:
                out_tan_ov = auto_tangents[left_idx][0]
                in_tan_ov = auto_tangents[right_idx][1]
            return _interpolate_spatial_bezier(
                time,
                t0,
                t1,
                kf_left,
                kf_right,
                out_tan_override=out_tan_ov,
                in_tan_override=in_tan_ov,
            )

        # Non-spatial: per-dimension 1D bezier
        out_ov: tuple[float, float] | None = None
        in_ov: tuple[float, float] | None = None
        if auto_ease:
            ae_left = auto_ease[left_idx]
            ae_right = auto_ease[right_idx]
            out_ov = (ae_left[0], ae_left[1])
            in_ov = (ae_right[2], ae_right[3])

        if isinstance(v0, list) and isinstance(v1, list):
            result: list[float] = []
            for d in range(len(v0)):
                out_e = (
                    kf_left.out_temporal_ease[d]
                    if d < len(kf_left.out_temporal_ease)
                    else kf_left.out_temporal_ease[0]
                )
                in_e = (
                    kf_right.in_temporal_ease[d]
                    if d < len(kf_right.in_temporal_ease)
                    else kf_right.in_temporal_ease[0]
                )
                result.append(
                    _interpolate_bezier_1d(
                        time,
                        t0,
                        t1,
                        v0[d],
                        v1[d],
                        out_e,
                        in_e,
                        out_override=out_ov,
                        in_override=in_ov,
                    )
                )
            return result

        if isinstance(v0, (int, float)) and isinstance(v1, (int, float)):
            out_e_1d = (
                kf_left.out_temporal_ease[0] if kf_left.out_temporal_ease else None
            )
            in_e_1d = (
                kf_right.in_temporal_ease[0] if kf_right.in_temporal_ease else None
            )
            if out_e_1d and in_e_1d:
                return _interpolate_bezier_1d(
                    time,
                    t0,
                    t1,
                    float(v0),
                    float(v1),
                    out_e_1d,
                    in_e_1d,
                    out_override=out_ov,
                    in_override=in_ov,
                )
            ratio = (time - t0) / (t1 - t0)
            return float(v0) + (float(v1) - float(v0)) * ratio

    # Fallback
    return cast("list[float] | float | None", v0)
