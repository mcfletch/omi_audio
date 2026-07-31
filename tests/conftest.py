"""Fixtures shared across the suite.

Two kinds of thing live here.  A stand-in for the pose source an application
feeds the listener from, and the engines every behavioural test needs -- built
once, in one place, so a change to how an engine is wired for testing is a
change in one file rather than in each of them.

Plain helpers (clips, level arithmetic, the ``.wav`` writer) are in
``support.py``, which tests import directly.
"""

import numpy as np
import pytest

from omi_audio import _backend, synth
from omi_audio.device import NullDevice
from omi_audio.engine import AudioEngine

from support import RATE


class _Rotation:
    """A unit quaternion, multiplied against a homogeneous vector on the left.

    ``quaternion * [x, y, z, w]`` is the operation the listener performs, and
    the fourth component rides through untouched, as it does for a rotation.
    """

    def __init__(self, axis=(0.0, 1.0, 0.0), radians=0.0):
        axis = np.asarray(axis, dtype='d')
        self.vector = axis / np.linalg.norm(axis) * np.sin(radians / 2.0)
        self.scalar = float(np.cos(radians / 2.0))

    def __mul__(self, vector):
        given = np.asarray(vector, dtype='d')
        point = given[:3]
        cross = np.cross(self.vector, point)
        rotated = point + 2.0 * np.cross(self.vector, cross + self.scalar * point)
        return np.concatenate([rotated, given[3:]])


class _Pose:
    """A place and a facing, in the shape a view platform presents them."""

    def __init__(self, position=(0.0, 0.0, 0.0), axis=(0.0, 1.0, 0.0), radians=0.0):
        self.position = np.asarray(position, dtype='d')
        self.quaternion = _Rotation(axis, radians)


@pytest.fixture
def pose():
    """Build a view-platform stand-in: ``pose(position=..., radians=...)``.

    Its rotation rotates vectors for real -- a fake that returned its input
    would let a broken transform pass.
    """
    return _Pose


@pytest.fixture
def engine():
    """An engine wired to silence, so tests read the mix rather than hear it."""
    made = AudioEngine(device=NullDevice(sample_rate=RATE), voices=8)
    try:
        yield made
    finally:
        made.close()


@pytest.fixture
def sounding_engine(engine):
    """:func:`engine`, holding one full-scale test tone named ``'test'``.

    Long enough that no test runs off the end of it, and unfaded, so a level
    read anywhere in it is the level the chain produced.
    """
    engine.clips.put('test', synth.tone(440.0, 5.0, sample_rate=RATE,
                                        amplitude=1.0, fade=0.0))
    return engine


@pytest.fixture
def no_backend(monkeypatch):
    """Make the optional ``miniaudio`` backend look absent.

    One patch point covers decoding and playback both, which is the whole
    reason the import lives in :mod:`omi_audio._backend` rather than in each of
    them.
    """
    monkeypatch.setattr(_backend, 'backend', lambda: None)
