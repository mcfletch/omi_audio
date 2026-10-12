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
from typing import Any, NamedTuple

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


class _Shelf(NamedTuple):
    """A head shadow's shelf at one rate and head radius, with its pole's powers."""

    rate: float
    head_radius: float
    #: Twice the rate, and twice the frequency the shelf turns over at in
    #: radians a second: the two terms of the bilinear transform.
    twice_rate: float
    turn: float
    pole: float
    #: ``pole^-m`` and ``pole^n`` for ``m``, ``n`` from 0; their length is
    #: the stretch a block is filtered in.
    inverse: NDArray[np.float64]
    forward: NDArray[np.float64]


def _shelf(head_radius: float, rate: float, longest: int = 512) -> _Shelf:
    """The shelf turning over at the speed of sound over ``head_radius``, at ``rate``.

    The pole's powers cover a stretch short enough that ``pole^-m`` stays
    below 1e100, and no longer than ``longest`` frames.
    """
    twice_rate = 2.0 * rate
    turn = 2.0 * SPEED_OF_SOUND / max(head_radius, 1e-6)
    pole = (twice_rate - turn) / (twice_rate + turn)
    reach = abs(pole)
    stretch = longest
    if 0.0 < reach < 1.0:
        stretch = max(1, min(longest, int(100.0 / -math.log10(reach))))
    if pole == 0.0:
        return _Shelf(rate, head_radius, twice_rate, turn, pole, np.ones(1), np.ones(1))
    forward = reach ** np.arange(stretch, dtype=np.float64)
    if pole < 0.0:
        forward[1::2] *= -1.0
    return _Shelf(rate, head_radius, twice_rate, turn, pole, 1.0 / forward, forward)


class HeadShadow:
    """One ear's head shadow: a first-order shelf, 1 at low frequencies and
    ``shadow`` at high, turning over at the speed of sound over the head's
    radius (Brown and Duda's), by the bilinear transform

    Keeps its last input and output from block to block.  The recursion
    ``y[n] = g[n] + p y[n-1]`` is taken a stretch at a time in closed form,
    ``y[n] = p^(n+1) y[-1] + p^n sum(p^-m g[m])``, as the biquad's poles are.
    """

    def __init__(self) -> None:
        self.x1 = 0.0
        self.y1 = 0.0
        self._design: _Shelf | None = None
        self._given = np.zeros(0)
        self._scratch = np.zeros(0)

    def settle(self) -> None:
        """Start from silence"""
        self.x1 = self.y1 = 0.0

    def design(self, head_radius: float, rate: float, longest: int) -> None:
        """Make the shelf for ``head_radius`` at ``rate``, for blocks up to ``longest`` frames.

        The control thread's: :meth:`run` given another radius or rate, or a
        longer block, makes the shelf itself, which allocates.
        """
        current = self._design
        if current is None or (current.rate, current.head_radius) != (rate, head_radius):
            self._design = _shelf(head_radius, rate)
        if self._given.shape[0] < longest:
            self._given = np.zeros(longest)
            self._scratch = np.zeros(longest)

    def run(self, signal: NDArray[np.float32], shadow: float, head_radius: float,
            rate: float, out: NDArray[np.float32]) -> None:
        """`signal` through the shelf into `out`, which may be `signal` itself"""
        frames = signal.shape[0]
        if not frames:
            return
        shelf = self._design
        if (shelf is None or shelf.rate != rate or shelf.head_radius != head_radius
                or self._given.shape[0] < frames):
            self.design(head_radius, rate, frames)
            shelf = self._design
            assert shelf is not None
        norm = shelf.twice_rate + shelf.turn
        b0 = (shadow * shelf.twice_rate + shelf.turn) / norm
        b1 = (shelf.turn - shadow * shelf.twice_rate) / norm
        last = float(signal[-1])
        # The feed-forward half: g[n] = b0 x[n] + b1 x[n-1].
        given = self._given[:frames]
        scratch = self._scratch[:frames]
        np.multiply(signal, b1, out=scratch)
        np.multiply(signal, b0, out=given)
        given[0] += b1 * self.x1
        np.add(given[1:], scratch[:-1], out=given[1:])
        self.x1 = last
        pole, stretch = shelf.pole, shelf.forward.shape[0]
        carry = self.y1
        # With the pole at nought there is no recursion: the output is `given`.
        for at in range(0, frames if pole else 0, stretch):
            count = min(stretch, frames - at)
            part = given[at:at + count]
            np.multiply(part, shelf.inverse[:count], out=part)
            np.cumsum(part, out=part)
            np.multiply(part, shelf.forward[:count], out=part)
            if carry:
                np.multiply(shelf.forward[:count], carry * pole, out=scratch[:count])
                np.add(part, scratch[:count], out=part)
            carry = float(part[-1])
        self.y1 = carry if pole else float(given[-1])
        np.copyto(out, given, casting='same_kind')


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
        graph = self._graph
        if response is not None and graph is not None:
            for shadow in self._shadows:
                shadow.design(response.head_radius, graph.sample_rate, graph.max_block)
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
        head = self._head
        if head is not None:
            for shadow in self._shadows:
                shadow.design(head.head_radius, graph.sample_rate, graph.max_block)

    def output_channels(self, _computed: int) -> int:
        return 2

    def silent(self) -> bool:
        """Whether both ears' gains are nought and the ramps to them finished."""
        return (self._left == 0.0 and self._right == 0.0
                and self._left_now == 0.0 and self._right_now == 0.0) or super().silent()

    def resume(self, missed: int) -> None:
        """Empty the propagation line and each ear's line and shadow."""
        self._delay.resume(missed)
        for ear in self._ears:
            ear.resume(missed)
        for shadow in self._shadows:
            shadow.settle()

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
