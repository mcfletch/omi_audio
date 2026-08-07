"""Where a sound is, and how loud it is from where you are standing.

Every function here is a pure function of geometry returning a **linear
amplitude multiplier** in ``[0, 1]``.  The engine multiplies them together --
distance, cone, emitter gain, source gain -- and hands the product to the mixer,
so each curve can be reasoned about, plotted and tested on its own.

Two attenuation models live here, because two published specifications describe
how a sound fades with distance and neither can express the other:

* **The glTF model** (:func:`distance_gain`, :func:`cone_gain`).  A distance
  curve plus a directional cone, from ``KHR_audio_emitter``, which takes them
  from the Web Audio API's ``PannerNode``.  This is the model authoring tools
  export and the one new content should use.
* **The VRML97 model** (:func:`ellipsoid_reach`, :func:`ellipsoid_gain`,
  :func:`ellipsoid_gain_at`).  Two ellipsoids sharing a focus at the sound, with
  a ramp between them that is linear in decibels.  A VRML97 ``Sound`` node says
  exactly this and nothing else can express it, so it is implemented rather than
  approximated.  See `docs/VRML97.md <../../docs/VRML97.md>`_ for the field
  mapping.

Panning is shared: :meth:`Listener.azimuth_elevation` puts the source in the
listener's own frame and :func:`equal_power_pan` turns the azimuth into a pair
of ear gains.  **The output is stereo, and only the azimuth reaches it** -- see
:func:`equal_power_pan` for what that costs.

References:
    ``KHR_audio_emitter``
    https://github.com/omigroup/gltf-extensions/tree/main/extensions/2.0/KHR_audio_emitter

    Web Audio API, "Spatialization"
    https://webaudio.github.io/web-audio-api/#Spatialization

    ISO/IEC 14772-1:1997 (VRML97) 6.42 ``Sound``
    https://www.web3d.org/documents/specifications/14772/V2.0/part1/nodesRef.html#Sound
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Any, Protocol, TypeAlias
from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

#: Radians in a full turn.  ``KHR_audio_emitter`` defaults both cone angles to
#: this, meaning "not a cone at all".
TAU = 2.0 * math.pi

#: The attenuation VRML97 puts at the outer ellipsoid, in decibels.  It calls
#: that inaudible, so the gain is forced to zero beyond it; the resulting step
#: from 0.1 to 0 is smoothed by the mixer's per-block gain ramp rather than
#: being fudged here, so this function stays the specification's own curve.
INAUDIBLE_DB = -20.0


class DistanceModel(str, Enum):
    """How an emitter's gain falls off with distance (``KHR_audio_emitter``).

    The values are the strings the glTF extension writes, so a parsed document's
    ``distanceModel`` is usable as a member without translation.
    """

    LINEAR = 'linear'
    INVERSE = 'inverse'
    EXPONENTIAL = 'exponential'


class ShapeType(str, Enum):
    """Whether a positional emitter radiates evenly or within a cone."""

    OMNIDIRECTIONAL = 'omnidirectional'
    CONE = 'cone'


#: Anything three numbers can be read out of: a tuple, a list, a numpy array.
Vector: TypeAlias = Sequence[float] | NDArray[Any]


class Rotation(Protocol):
    """A rotation that can be applied to a homogeneous vector on the left.

    ``quaternion * [x, y, z, w]`` is the whole of the operation
    :meth:`Listener.from_view_platform` performs, so it is the whole of what a
    rotation has to provide.
    """

    def __mul__(self, vector: Sequence[float]) -> Any:
        ...


class ViewPlatform(Protocol):
    """A camera pose, in the shape a scenegraph presents one.

    Stated as a :class:`~typing.Protocol` rather than a base class because the
    coupling is meant to stay this thin: any object with a place and a facing
    can be the listener, and requiring a particular camera class would make the
    library care which renderer sits above it.
    """

    @property
    def position(self) -> Vector:
        """World-space position of the camera."""

    @property
    def quaternion(self) -> Rotation:
        """The camera's orientation."""


def _unit(vector: Vector) -> NDArray[np.float64]:
    """``vector`` scaled to unit length; a zero vector is returned unchanged."""
    array = np.asarray(vector, dtype='d')[:3]
    length = float(np.linalg.norm(array))
    return array if length == 0.0 else array / length


def distance_gain(
    distance: float,
    model: DistanceModel | str = DistanceModel.INVERSE,
    ref_distance: float = 1.0,
    max_distance: float = 0.0,
    rolloff_factor: float = 1.0,
) -> float:
    """Gain for a source ``distance`` away under one of the glTF distance models.

    These are the extension's three formulas, implemented as written::

        inverse      ref / (ref + rolloff * (max(d, ref) - ref))
        exponential  (max(d, ref) / ref) ** -rolloff
        linear       1 - rolloff * (clamp(d, ref, max) - ref) / (max - ref)

    ``ref_distance`` is where each curve reads 1.0 and inside which nothing is
    attenuated, and ``rolloff_factor`` is how sharply it falls.  **Only the
    linear model uses** ``max_distance``, which is where it reaches silence;
    the other two fall away for ever and never quite reach it.

    That last point is worth stating plainly because the extension's prose says
    something else -- it calls ``maxDistance`` "the maximum distance between the
    emitter and listener, beyond which the audio cannot be heard", which no
    formula implements.  The formulas are what Web Audio's ``PannerNode`` does
    and therefore what Blender, Godot and three.js content is authored against,
    so the formulas are what is implemented here.  Use
    :meth:`~omi_audio.model.PositionalProperties.in_range` where the prose
    reading is what an application wants; see ``docs/SPATIALISATION.md``.

    A ``linear`` emitter therefore needs a ``max_distance`` greater than its
    ``ref_distance`` to mean anything.  Given one that is not, it is audible
    within ``ref_distance`` and silent past it.

    Raises:
        ValueError: if ``ref_distance`` is not positive, which every model
            divides by.
    """
    if ref_distance <= 0.0:
        raise ValueError('ref_distance must be greater than zero, not %r' % (ref_distance,))
    model = DistanceModel(model)
    # Inside the reference distance every model reads 1.0.  The far end is the
    # linear model's business alone -- see the docstring.
    near = max(distance, ref_distance)
    if model is DistanceModel.INVERSE:
        return ref_distance / (ref_distance + rolloff_factor * (near - ref_distance))
    if model is DistanceModel.EXPONENTIAL:
        return float((near / ref_distance) ** -rolloff_factor)
    if max_distance <= ref_distance:
        # A linear ramp with no room to ramp: audible at the reference distance
        # and silent everywhere past it.
        return 1.0 if distance <= ref_distance else 0.0
    span = (min(near, max_distance) - ref_distance) / (max_distance - ref_distance)
    return max(0.0, min(1.0, 1.0 - rolloff_factor * span))


def cone_gain(
    angle: float,
    inner_angle: float = TAU,
    outer_angle: float = TAU,
    outer_gain: float = 0.0,
) -> float:
    """Gain for a listener ``angle`` radians off an emitter's forward axis.

    ``inner_angle`` and ``outer_angle`` are *angular diameters* -- the whole cone
    from side to side -- so the boundaries are at half of each.  Inside the inner
    cone there is no attenuation; outside the outer cone the gain is
    ``outer_gain``; between them it interpolates linearly.

    The defaults describe a full sphere, which is why an emitter that never sets
    them is never attenuated by direction.
    """
    inner = abs(inner_angle) / 2.0
    outer = abs(outer_angle) / 2.0
    angle = abs(angle)
    if angle <= inner:
        return 1.0
    if angle >= outer:
        return outer_gain
    across = (angle - inner) / (outer - inner)
    return (1.0 - across) + outer_gain * across


def ellipsoid_reach(front: float, back: float, cos_theta: float) -> float:
    """Distance from a VRML97 ``Sound``'s location to its ellipsoid surface.

    The four distance fields describe an ellipsoid **with one focus at the
    sound**, reaching ``front`` along ``direction`` and ``back`` against it.
    Measuring from the focus rather than the centre makes the surface a focal
    conic, whose polar form collapses to::

        reach(theta) = 2 * front * back / ((front + back) - (front - back) * cos(theta))

    -- the harmonic mean of the two distances at right angles, and each distance
    itself along the axis.  ``cos_theta`` is the cosine of the angle between the
    sound's ``direction`` and the direction to the listener.

    A ``front`` or ``back`` of zero describes an ellipsoid with no interior, so
    the reach is zero in every direction.
    """
    if front <= 0.0 or back <= 0.0:
        return 0.0
    return 2.0 * front * back / ((front + back) - (front - back) * cos_theta)


def ellipsoid_gain(
    distance: float,
    cos_theta: float,
    min_front: float = 1.0,
    min_back: float = 1.0,
    max_front: float = 10.0,
    max_back: float = 10.0,
) -> float:
    """VRML97's gain between a ``Sound``'s inner and outer ellipsoids.

    Full volume inside the inner ellipsoid, silence outside the outer, and
    between them a ramp that is linear **in decibels** from 0 dB down to
    :data:`INAUDIBLE_DB`.  Linear in decibels rather than in amplitude is what
    makes the sound fade the way a listener expects instead of vanishing at the
    end of the ramp.

    Coincident ellipsoids leave no room to ramp, so the sound is at full volume
    inside them and silent outside.

    Most callers want :func:`ellipsoid_gain_at`, which works ``distance`` and
    ``cos_theta`` out from the two world positions and the sound's direction.
    """
    inner = ellipsoid_reach(min_front, min_back, cos_theta)
    outer = ellipsoid_reach(max_front, max_back, cos_theta)
    if distance <= inner:
        return 1.0
    if distance > outer:
        return 0.0
    across = (distance - inner) / (outer - inner)
    return float(10.0 ** (INAUDIBLE_DB * across / 20.0))


def ellipsoid_gain_at(
    location: Vector,
    direction: Vector,
    listener_position: Vector,
    min_front: float = 1.0,
    min_back: float = 1.0,
    max_front: float = 10.0,
    max_back: float = 10.0,
) -> float:
    """:func:`ellipsoid_gain` for a ``Sound`` at a place, facing a way.

    ``location``, ``direction`` and ``listener_position`` are all **world
    space**: a VRML97 ``Sound``'s ``location`` and ``direction`` are given in
    its local coordinate system, so a caller with a transform stack applies it
    before calling.

    This is the geometry every consumer of the VRML97 model would otherwise
    derive for itself -- the glTF cone path has the equivalent done for it
    inside the engine -- so it lives here once, where it is tested.

    A sound with no direction at all has no front and no back to tell apart, so
    its ``front`` distances apply in every direction.
    """
    offset = np.asarray(listener_position, dtype='d')[:3] - np.asarray(location, dtype='d')[:3]
    distance = float(np.linalg.norm(offset))
    if distance == 0.0:
        return 1.0                          # standing on the sound: fully inside
    axis = _unit(direction)
    cos_theta = 1.0 if not np.any(axis) else float(np.dot(axis, offset / distance))
    return ellipsoid_gain(distance, cos_theta, min_front=min_front, min_back=min_back,
                          max_front=max_front, max_back=max_back)


def equal_power_pan(azimuth: float) -> tuple[float, float]:
    """Left and right gains for a mono source ``azimuth`` radians off centre.

    Positive azimuth is to the listener's right.  The two gains trace a quarter
    circle, so ``left**2 + right**2`` is 1 at every angle: panning moves a sound
    across the stereo field without changing how loud it is.

    A source behind the listener is folded onto its mirror image in front --
    behind-and-right pans right.  Two loudspeakers cannot put a sound behind
    anybody, and the fold is what the Web Audio API specifies rather than
    something chosen here.

    **Elevation is not represented.**  A pair of ear gains has one degree of
    freedom and azimuth spends it, so a sound directly overhead and one dead
    ahead are indistinguishable.  Stereo headphones are what this library aims
    at; height needs an HRTF, and surround needs more than two channels.
    Neither is implemented, and both would be a new function rather than a
    change to this one.
    """
    if azimuth < -math.pi / 2:
        azimuth = -math.pi - azimuth
    elif azimuth > math.pi / 2:
        azimuth = math.pi - azimuth
    across = (azimuth + math.pi / 2) / math.pi
    return math.cos(across * math.pi / 2), math.sin(across * math.pi / 2)


@dataclass(frozen=True)
class Listener:
    """Where the ears are, and which way they face.

    One per engine, refreshed each frame from the view platform.  ``forward``
    and ``up`` are normalised on construction so callers may pass whatever
    length falls out of their own maths.
    """

    #: World-space position of the listener.
    position: NDArray[np.float64]
    #: Unit vector the listener faces; ``-Z`` in the glTF and VRML97 default view.
    forward: NDArray[np.float64]
    #: Unit vector out of the top of the listener's head.
    up: NDArray[np.float64]

    def __init__(self, position: Vector = (0.0, 0.0, 0.0),
                 forward: Vector = (0.0, 0.0, -1.0),
                 up: Vector = (0.0, 1.0, 0.0)) -> None:
        object.__setattr__(self, 'position', np.asarray(position, dtype='d')[:3])
        object.__setattr__(self, 'forward', _unit(forward))
        object.__setattr__(self, 'up', _unit(up))

    @property
    def right(self) -> NDArray[np.float64]:
        """Unit vector out of the listener's right ear."""
        return _unit(np.cross(self.forward, self.up))

    @classmethod
    def from_view_platform(cls, platform: ViewPlatform) -> Listener:
        """The listener implied by a camera's pose.

        ``platform`` is anything satisfying :class:`ViewPlatform` -- a
        scenegraph's view platform, or an application's own camera object.
        Duck-typed on purpose: the pose is all this needs, and requiring a
        particular camera class would make the library care which renderer is
        above it.

        The camera *is* the listener: a scene with sound in it wants the two to
        agree, and there is nothing an application would have to keep in step if
        the pose is read from the platform every frame.
        """
        rotation = platform.quaternion
        forward = np.asarray(rotation * [0.0, 0.0, -1.0, 0.0], dtype='d')[:3]
        up = np.asarray(rotation * [0.0, 1.0, 0.0, 0.0], dtype='d')[:3]
        return cls(position=np.asarray(platform.position, dtype='d')[:3],
                   forward=forward, up=up)

    def distance_to(self, point: Vector) -> float:
        """Straight-line distance from the listener to ``point``."""
        return float(np.linalg.norm(np.asarray(point, dtype='d')[:3] - self.position))

    def azimuth_elevation(self, point: Vector) -> tuple[float, float]:
        """``point`` as a bearing from the listener, in radians.

        Azimuth is measured in the listener's horizontal plane: 0 dead ahead,
        positive to the right, ``+-pi`` behind.  Elevation is the angle out of
        that plane, positive overhead.

        Both are returned, but only the azimuth reaches the stereo output; see
        :func:`equal_power_pan`.  The elevation is here because it is free to
        compute and an application may want it -- to duck a sound that is
        overhead, or to drive its own filter.

        A point *at* the listener has no bearing at all; it is reported as dead
        ahead so that a sound placed on the camera pans to the centre rather
        than producing a division by zero.
        """
        offset = np.asarray(point, dtype='d')[:3] - self.position
        length = float(np.linalg.norm(offset))
        if length == 0.0:
            return 0.0, 0.0
        offset = offset / length
        vertical = float(np.dot(offset, self.up))
        horizontal = offset - vertical * self.up
        elevation = math.asin(max(-1.0, min(1.0, vertical)))
        if not np.any(horizontal):
            # Directly overhead or underfoot: no horizontal bearing exists.
            return 0.0, elevation
        azimuth = math.atan2(float(np.dot(horizontal, self.right)),
                             float(np.dot(horizontal, self.forward)))
        return azimuth, elevation
