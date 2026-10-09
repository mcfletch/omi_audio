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
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from omi_audio._numbers import finite
from omi_audio.graph.core import Graph, Node
from omi_audio.graph.filters import Delay

__all__ = ['SPEED_OF_SOUND', 'Panner']

#: Metres a second, in air at about 20 degrees Celsius.
SPEED_OF_SOUND = 343.0


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

    @property
    def gains(self) -> tuple[float, float]:
        """The left and right gains being ramped to."""
        return self._left, self._right

    def set_gains(self, left: float, right: float) -> None:
        """Ramp to ``left`` and ``right`` across the next block; non-finite is silence."""
        self._left = finite(left)
        self._right = finite(right)

    def settle(self) -> None:
        super().settle()
        self._left_now, self._right_now = self._left, self._right
        self._delay.settle()

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
        self._delay.bind(graph)

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
        self._left_now = _place(mono, self._left_now, self._left, out[0, :frames], graph)
        self._right_now = _place(mono, self._right_now, self._right, out[1, :frames], graph)


def _place(mono: NDArray[np.float32], start: float, end: float,
           target: NDArray[np.float32], graph: Graph) -> float:
    """``mono`` times a gain ramped from ``start`` to ``end``, into ``target``."""
    if start == end:
        np.multiply(mono, end, out=target)
    else:
        np.multiply(mono, graph.ramp(start, end, target.shape[0]), out=target)
    return end
