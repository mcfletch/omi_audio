"""A dynamics compressor: loud passages turned down, the whole raised to match.

The Web Audio API's ``DynamicsCompressorNode`` names the controls this node
has, and its make-up gain; the detector and the gain smoothing are this
module's own:

* The level is the peak of every channel over each stretch of
  :data:`DETECTOR_FRAMES` frames, in decibels.
* The static curve passes a level below ``threshold - knee / 2`` unchanged,
  divides how far it is above ``threshold`` by ``ratio`` beyond
  ``threshold + knee / 2``, and joins the two with a quadratic across the knee,
  so the slope changes smoothly from 1 to ``1 / ratio``.
* The gain reduction moves toward the curve's with a time constant of
  ``attack`` seconds when it is deepening and ``release`` seconds when it is
  recovering, once a stretch, and is interpolated across the stretch.
* The make-up gain is ``(1 / g) ** 0.6`` for the gain ``g`` the curve gives a
  full-scale signal, as the Web Audio API specifies, so the compressed signal
  is about as loud as the input was.

There is no look-ahead: a transient sharper than the attack passes partly
uncompressed.  :attr:`DynamicsCompressor.reduction` is the gain reduction
applied at the end of the last block, in decibels (zero or less).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from numpy.typing import NDArray

from omi_audio._numbers import finite
from omi_audio.graph.core import Graph, Node

__all__ = ['DETECTOR_FRAMES', 'DynamicsCompressor', 'static_curve']

#: Frames per detector stretch: under a millisecond at the usual rates.
DETECTOR_FRAMES = 32

def static_curve(level: float, threshold: float, knee: float, ratio: float) -> float:
    """The output level, in decibels, for a steady input ``level`` in decibels."""
    over = level - threshold
    if knee > 0.0 and abs(over) <= knee / 2.0:
        return level + (1.0 / ratio - 1.0) * (over + knee / 2.0) ** 2 / (2.0 * knee)
    if over > 0.0:
        return threshold + over / ratio
    return level


class DynamicsCompressor(Node):
    """A compressor over every channel, driven by their loudest.

    ``threshold`` (-100 to 0) and ``knee`` (0 to 40) are in decibels,
    ``ratio`` (1 to 20) is the input change in decibels for one decibel of
    output change above the threshold, and ``attack`` and ``release`` are the
    time constants of the gain change, in seconds.
    """

    def __init__(self, threshold: float = -24.0, knee: float = 30.0, ratio: float = 12.0,
                 attack: float = 0.003, release: float = 0.25, gain: float = 1.0,
                 **named: Any) -> None:
        super().__init__(gain=gain, **named)
        self.threshold = threshold
        self.knee = knee
        self.ratio = ratio
        self.attack = attack
        self.release = release
        #: The gain reduction at the end of the last block, in decibels.
        self.reduction = 0.0
        self._level_now = 1.0
        self._loudest = np.zeros(0, dtype=np.float32)
        self._peaks = np.zeros(0, dtype=np.float32)
        self._levels = np.zeros(0)
        self._starts = np.zeros(0, dtype=np.intp)
        self._magnitude = np.zeros((1, 0), dtype=np.float32)
        self._ends = np.zeros(0)
        self._begins = np.zeros(0)
        self._gains = np.zeros(0)
        self._fractions = np.zeros(0)

    def settings(self) -> tuple[float, float, float, float, float]:
        """The controls, each limited to its range."""
        return (max(-100.0, min(0.0, finite(self.threshold))),
                max(0.0, min(40.0, finite(self.knee))),
                max(1.0, min(20.0, finite(self.ratio)) or 1.0),
                max(0.0, finite(self.attack)), max(0.0, finite(self.release)))

    def prepare(self, graph: Graph) -> None:
        stretches = -(-graph.max_block // DETECTOR_FRAMES)
        self._peaks = np.zeros(stretches, dtype=np.float32)
        self._levels = np.zeros(stretches)
        self._starts = np.arange(0, graph.max_block, DETECTOR_FRAMES, dtype=np.intp)
        self._magnitude = np.zeros((graph.max_channels, graph.max_block), dtype=np.float32)
        self._loudest = np.zeros(graph.max_block, dtype=np.float32)
        self._ends = np.zeros(stretches)
        self._begins = np.zeros(stretches)
        self._gains = np.zeros(stretches * DETECTOR_FRAMES)
        self._fractions = np.arange(1, DETECTOR_FRAMES + 1, dtype='d') / DETECTOR_FRAMES

    def resume(self, missed: int) -> None:  # noqa: ARG002 - the base method's signature is kept
        """Start from no reduction."""
        self.reduction = 0.0
        self._level_now = 1.0

    def process(self, block: NDArray[np.float32], out: NDArray[np.float32],
                frames: int) -> None:
        threshold, knee, ratio, attack, release = self.settings()
        channels = block.shape[0]
        stretches = -(-frames // DETECTOR_FRAMES)
        magnitude = self._magnitude[:channels, :frames]
        np.abs(block, out=magnitude)
        loudest = self._loudest[:frames]
        np.max(magnitude, axis=0, out=loudest)
        peaks = self._peaks[:stretches]
        np.maximum.reduceat(loudest, self._starts[:stretches], out=peaks)
        levels = self._levels[:stretches]
        np.maximum(peaks, 1e-10, out=levels)
        np.log10(levels, out=levels)
        levels *= 20.0
        rate = self.sample_rate
        attack_keep = _keep(attack, rate)
        release_keep = _keep(release, rate)
        makeup = -0.6 * (static_curve(0.0, threshold, knee, ratio))
        reduction = self.reduction
        ends = self._ends[:stretches]
        for index in range(stretches):
            level = float(levels[index])
            target = static_curve(level, threshold, knee, ratio) - level
            keep = attack_keep if target < reduction else release_keep
            reduction = target + (reduction - target) * keep
            ends[index] = reduction
        self.reduction = reduction
        # Linear gains at each stretch's end, makeup included, and each
        # stretch ramped from the one before.
        np.add(ends, makeup, out=ends)
        np.multiply(ends, math.log(10.0) / 20.0, out=ends)
        np.exp(ends, out=ends)
        begins = self._begins[:stretches]
        begins[0] = self._level_now
        begins[1:] = ends[:-1]
        self._level_now = float(ends[-1])
        gains = self._gains[:stretches * DETECTOR_FRAMES].reshape(stretches, DETECTOR_FRAMES)
        np.subtract(ends, begins, out=ends)
        np.multiply(ends[:, None], self._fractions[None, :], out=gains)
        np.add(gains, begins[:, None], out=gains)
        flat = self._gains[:frames]
        for channel in range(channels):
            np.multiply(block[channel], flat, out=out[channel, :frames], casting='same_kind')


def _keep(seconds: float, rate: int) -> float:
    """How much of the gap to the target is left after one detector stretch."""
    if seconds <= 0.0:
        return 0.0
    return math.exp(-DETECTOR_FRAMES / (seconds * rate))
