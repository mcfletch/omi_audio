"""A sound placed in the stereo field: mono in, two ears out.

The control thread works out where the sound is -- distance, cone and pan,
with :mod:`omi_audio.spatial` -- and hands a :class:`Panner` the two ear gains
with :meth:`Panner.set_gains`, as it does a voice's; the panner ramps to them
across the next block.

With :attr:`Panner.propagation_delay` set, the sound also arrives that many
seconds late, through a delay line of up to ``max_propagation_delay``
seconds.  Setting it each frame to the distance over the speed of sound is
the Doppler effect: an approaching sound's delay shrinks, so it plays
faster and higher, and a receding one's grows.

Given a :class:`~omi_audio.spatial.HeadResponse` (:meth:`Panner.set_head`),
each ear also hears the sound as a spherical head shapes it: the far ear
later by the interaural delay, and each through a first-order shelf whose
high-frequency gain is the ear's shadow -- the Web Audio panner's HRTF model,
made from the geometry of a head rather than from measured responses.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from numpy.typing import NDArray

from omi_audio._numbers import finite
from omi_audio.graph.core import Graph, Node
from omi_audio.graph.filters import Delay
from omi_audio.spatial import SPEED_OF_SOUND, HeadResponse

__all__ = ['MAX_HEAD_DELAY', 'SPEED_OF_SOUND', 'HeadShadow', 'Panner']

#: The longest interaural delay an ear's line holds, in seconds: a head of
#: 0.2 m radius, more than any listener's.
MAX_HEAD_DELAY = 0.002


class HeadShadow:
    """One ear's head shadow: a first-order shelf, 1 at low frequencies and
    ``shadow`` at high, turning over at the speed of sound over the head's
    radius (Brown and Duda's), by the bilinear transform

    Keeps its last input and output from block to block.
    """

    def __init__(self) -> None:
        self.x1 = 0.0
        self.y1 = 0.0

    def settle(self) -> None:
        """Start from silence"""
        self.x1 = self.y1 = 0.0

    def run(self, signal: NDArray[np.float32], shadow: float, head_radius: float,
            rate: float, out: NDArray[np.float32]) -> None:
        """`signal` through the shelf into `out`"""
        twice_rate = 2.0 * rate
        turn = 2.0 * SPEED_OF_SOUND / max(head_radius, 1e-6)
        norm = twice_rate + turn
        b0 = (shadow * twice_rate + turn) / norm
        b1 = (turn - shadow * twice_rate) / norm
        pole = (twice_rate - turn) / norm
        x = np.asarray(signal, dtype=np.float64)
        given = b0 * x
        given[0] += b1 * self.x1
        given[1:] += b1 * x[:-1]
        self.y1 = _first_order(given, pole, self.y1, out)
        self.x1 = float(x[-1]) if x.shape[0] else self.x1


def _first_order(given: NDArray[np.float64], pole: float, start: float,
                 out: NDArray[np.float32]) -> float:
    """``out[n] = given[n] + pole out[n-1]`` from ``out[-1] = start``; the last value

    In stretches short enough that the pole's powers stay within six
    decades, each a cumulative sum.
    """
    frames = given.shape[0]
    if not frames:
        return start
    if pole == 0.0:
        out[:frames] = given
        return float(given[-1])
    reach = abs(pole)
    stretch = max(1, min(256, int(6.0 / -math.log10(reach)))) if reach < 1.0 else 1
    carry = start
    steps = np.arange(1, stretch + 1, dtype=np.float64)
    for at in range(0, frames, stretch):
        count = min(stretch, frames - at)
        powers = pole ** steps[:count]
        part = np.cumsum(given[at:at + count] / powers * pole) * powers / pole
        part += powers * carry
        out[at:at + count] = part
        carry = float(part[-1])
    return carry


class Panner(Node):
    """Its input mixed to mono and placed by a pair of ear gains."""

    def __init__(self, left: float = 0.0, right: float = 0.0,
                 max_propagation_delay: float = 1.0, **named: Any) -> None:
        named.setdefault('channel_count', 1)
        named.setdefault('channel_count_mode', 'explicit')
        super().__init__(**named)
        self._left = self._left_now = finite(left)
        self._right = self._right_now = finite(right)
        self._delay = Delay(max_delay_time=max_propagation_delay)
        self._mono = np.zeros((1, 0), dtype=np.float32)
        self._head: HeadResponse | None = None
        self._ears = (Delay(max_delay_time=MAX_HEAD_DELAY), Delay(max_delay_time=MAX_HEAD_DELAY))
        self._shadows = (HeadShadow(), HeadShadow())
        self._heard = np.zeros((2, 0), dtype=np.float32)

    @property
    def gains(self) -> tuple[float, float]:
        """The left and right gains being ramped to."""
        return self._left, self._right

    def set_gains(self, left: float, right: float) -> None:
        """Ramp to ``left`` and ``right`` across the next block; non-finite is silence."""
        self._left = finite(left)
        self._right = finite(right)

    @property
    def head(self) -> HeadResponse | None:
        """The head each ear hears through; None for the gains alone"""
        return self._head

    def set_head(self, response: HeadResponse | None) -> None:
        """Hear through `response`'s delays and shadows from the next block; None for none"""
        self._head = response
        for ear, late in zip(self._ears, (response.left_delay, response.right_delay)
                             if response is not None else (0.0, 0.0), strict=True):
            ear.delay_time = late

    def settle(self) -> None:
        super().settle()
        self._left_now, self._right_now = self._left, self._right
        self._delay.settle()
        for ear in self._ears:
            ear.settle()
        for shadow in self._shadows:
            shadow.settle()

    @property
    def propagation_delay(self) -> float:
        """Seconds the sound takes to arrive; 0 for none."""
        return self._delay.delay_time

    @propagation_delay.setter
    def propagation_delay(self, value: float) -> None:
        self._delay.delay_time = value

    @property
    def max_propagation_delay(self) -> float:
        """The longest :attr:`propagation_delay`, in seconds; setting it empties the line."""
        return self._delay.max_delay_time

    @max_propagation_delay.setter
    def max_propagation_delay(self, value: float) -> None:
        self._delay.max_delay_time = value

    def prepare(self, graph: Graph) -> None:
        self._mono = np.zeros((1, graph.max_block), dtype=np.float32)
        self._heard = np.zeros((2, graph.max_block), dtype=np.float32)
        self._delay.bind(graph)
        for ear in self._ears:
            ear.bind(graph)

    def output_channels(self, _computed: int) -> int:
        return 2

    def process(self, block: NDArray[np.float32], out: NDArray[np.float32],
                frames: int) -> None:
        graph = self._graph
        if graph is None:  # pragma: no cover - rendering means a graph
            return
        mono = block[0]
        if self._delay.delay_time > 0.0 or self._delay.reached > 0.0:
            delayed = self._mono[:, :frames]
            self._delay.process(block[:1], delayed, frames)
            mono = delayed[0]
        head = self._head
        if head is None:
            self._left_now = _place(mono, self._left_now, self._left, out[0, :frames], graph)
            self._right_now = _place(mono, self._right_now, self._right, out[1, :frames], graph)
            return
        heard = self._heard[:, :frames]
        source = mono.reshape(1, -1)
        for side, (ear, shadow, gain) in enumerate(zip(
                self._ears, self._shadows, (head.left_shadow, head.right_shadow), strict=True)):
            ear.process(source, heard[side:side + 1], frames)
            shadow.run(heard[side], gain, head.head_radius, graph.sample_rate, heard[side])
        self._left_now = _place(heard[0], self._left_now, self._left, out[0, :frames], graph)
        self._right_now = _place(heard[1], self._right_now, self._right, out[1, :frames], graph)


def _place(mono: NDArray[np.float32], start: float, end: float,
           target: NDArray[np.float32], graph: Graph) -> float:
    """``mono`` times a gain ramped from ``start`` to ``end``, into ``target``."""
    if start == end:
        np.multiply(mono, end, out=target)
    else:
        np.multiply(mono, graph.ramp(start, end, target.shape[0]), out=target)
    return end
