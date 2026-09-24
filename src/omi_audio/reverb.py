"""The sound of the place the listener is in: a reverb on the whole mix.

A tunnel, a cellar or a stairwell gives back everything played in it, the
listener's own footsteps and engine included, and that is a property of the
place rather than of any one sound. So this runs on the summed mix, after the
voices are added up, and one setting covers every sound playing.

The reverb is a bank of feedback comb filters, the arrangement Schroeder
described: each comb is a delay line fed back into itself, so a sound comes
back again and again, a little quieter and a little darker each time. Several
combs with delays that share no common factor fill in each other's gaps, and
the left and right ears get slightly different delays so the tail spreads
across the stereo field instead of sitting in the middle.

It is built for the audio thread the way :mod:`~omi_audio.mixer` is:

* The delay lines are allocated once, at construction, for the longest delay
  any comb uses, and :meth:`Reverb.process` writes into them in place.
* A comb's feedback reads a sample written one delay ago, so a stretch of the
  block no longer than the shortest delay reads only samples written before
  it started. The block is processed in stretches of that length, each one
  vectorised, and no sample-by-sample loop runs in Python.
* The damping, which darkens each return, is a two-tap average of the samples
  read back. It is applied to what is read rather than inside the recursion,
  which keeps the stretch vectorised and still takes the top off each pass in
  turn, since each pass reads what the previous one wrote.

:attr:`Reverb.level` is how much of the reverb is heard, from nought to one,
and it is ramped across a block as a voice's gain is, so a listener walking
into a tunnel hears the reverb arrive rather than switch on. At nought, once
the ramp has finished, the processing stops and the delay lines are cleared,
so a place that has no reverb costs nothing.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

__all__ = ['Reverb', 'COMB_DELAYS', 'SPREAD', 'feedback_for']

#: The comb delays, in seconds. Schroeder's own choice of roughly 30 to 45
#: milliseconds, picked so no two share a factor, which keeps their echoes
#: from landing together and ringing at one pitch.
COMB_DELAYS = (0.0297, 0.0371, 0.0411, 0.0437)

#: How much later the right ear's combs are than the left's, in seconds. A
#: millisecond is enough to decorrelate the two tails, so the reverb is heard
#: around the listener rather than in the middle of the head.
SPREAD = 0.00052


def feedback_for(delay: float, decay: float) -> float:
    """The feedback gain that makes a comb of ``delay`` seconds fall 60 dB in ``decay``.

    A sound circulating in a comb is multiplied by the feedback once a
    ``delay``; falling by a thousand times over ``decay`` seconds is falling by
    ``10 ** (-3 * delay / decay)`` per pass. A ``decay`` of nought or less is
    no feedback at all.
    """
    if decay <= 0.0:
        return 0.0
    return float(10.0 ** (-3.0 * float(delay) / float(decay)))


class Reverb:
    """A stereo comb-filter reverb over blocks of up to ``max_block`` frames.

    ``decay`` is the reverberation time in seconds, the time a sound takes to
    fall by 60 dB. ``damping`` is how much of the top each return loses, from
    nought (none) to one (the most). ``level`` is how loud the reverb is
    against the dry mix. All three are plain floats the control thread may
    write at any time.
    """

    def __init__(self, sample_rate: int, max_block: int) -> None:
        self.sample_rate = int(sample_rate)
        self.decay = 1.2
        self.damping = 0.4
        self.level = 0.0
        #: The level the last block ended at, which the next one ramps from.
        self._level = 0.0
        #: Each comb's delay in frames, one row per ear.
        self._delays = np.array(
            [[max(2, int(round(d * self.sample_rate))) for d in COMB_DELAYS],
             [max(2, int(round((d + SPREAD) * self.sample_rate)))
              for d in COMB_DELAYS]], dtype=np.int64)
        longest = int(self._delays.max())
        #: The combs' delay lines: ear, comb, and the samples it last wrote.
        self._lines = np.zeros((2, len(COMB_DELAYS), longest), dtype=np.float32)
        #: Where each line will be written next.
        self._cursor = np.zeros((2, len(COMB_DELAYS)), dtype=np.int64)
        #: The last sample each line read, for the damping's two-tap average.
        self._previous = np.zeros((2, len(COMB_DELAYS)), dtype=np.float32)
        #: The longest stretch that can be processed in one go: every comb
        #: reads only what was written before the stretch began.
        self._stretch = int(self._delays.min())
        size = int(max_block)
        self._wet = np.zeros((size, 2), dtype=np.float32)
        self._read = np.zeros(self._stretch, dtype=np.float32)
        self._damped = np.zeros(self._stretch, dtype=np.float32)
        self._written = np.zeros(self._stretch, dtype=np.float32)
        self._steps = np.arange(1, size + 1, dtype=np.float32)
        self._ramp = np.zeros(size, dtype=np.float32)
        self._idle = True

    @property
    def active(self) -> bool:
        """Whether the next block will be processed at all.

        Still true for the one block after the level reaches nought, which is
        the block that clears the delay lines.
        """
        return not (self.level <= 0.0 and self._level <= 0.0 and self._idle)

    def process(self, out: NDArray[np.float32], frames: int) -> None:
        """Add the reverb of ``out``'s first ``frames`` frames into it, in place."""
        level = max(0.0, min(1.0, _finite(self.level)))
        if level <= 0.0 and self._level <= 0.0:
            if not self._idle:
                self._lines.fill(0.0)
                self._previous.fill(0.0)
                self._idle = True
            return
        self._idle = False
        dry = out[:frames]
        wet = self._wet[:frames]
        wet.fill(0.0)
        decay = _finite(self.decay)
        damping = max(0.0, min(1.0, _finite(self.damping))) * 0.5
        for ear in range(2):
            source = dry[:, ear]
            target = wet[:, ear]
            for comb in range(len(COMB_DELAYS)):
                gain = feedback_for(self._delays[ear, comb] / self.sample_rate, decay)
                self._comb(ear, comb, source, target, frames, gain, damping)
        np.multiply(wet, 1.0 / len(COMB_DELAYS), out=wet)
        ramp = self._ramp[:frames]
        if level == self._level:
            ramp.fill(level)
        else:
            np.multiply(self._steps[:frames], (level - self._level) / frames, out=ramp)
            np.add(ramp, self._level, out=ramp)
        self._level = level
        wet *= ramp[:, None]
        dry += wet

    def _comb(self, ear: int, comb: int, source: NDArray[np.float32],
              target: NDArray[np.float32], frames: int, gain: float,
              damping: float) -> None:
        """Run one comb over the block, a stretch at a time, adding into ``target``."""
        line = self._lines[ear, comb]
        delay = int(self._delays[ear, comb])
        cursor = int(self._cursor[ear, comb])
        previous = float(self._previous[ear, comb])
        done = 0
        while done < frames:
            count = min(self._stretch, frames - done)
            read = self._read[:count]
            # The ring is one delay long, so what sits at the cursor was
            # written one delay ago.
            _take(line, cursor, count, delay, read)
            damped = self._damped[:count]
            # Each return is the average of itself and the one before, weighted
            # by the damping: the top of the tail goes first, as it does in a
            # room with anything soft in it.
            damped[0] = previous
            if count > 1:
                damped[1:] = read[:count - 1]
            np.multiply(damped, damping, out=damped)
            previous = float(read[count - 1])
            np.multiply(read, 1.0 - damping, out=read)
            np.add(read, damped, out=read)
            written = self._written[:count]
            np.multiply(read, gain, out=written)
            np.add(written, source[done:done + count], out=written)
            _put(line, cursor, written, delay)
            target[done:done + count] += read
            cursor = (cursor + count) % delay
            done += count
        self._cursor[ear, comb] = cursor
        self._previous[ear, comb] = previous


def _take(line: NDArray[np.float32], start: int, count: int, size: int,
          into: NDArray[np.float32]) -> None:
    """Copy ``count`` samples of a ring of ``size`` from ``start`` into ``into``."""
    start %= size
    first = min(count, size - start)
    into[:first] = line[start:start + first]
    if first < count:
        into[first:count] = line[:count - first]


def _put(line: NDArray[np.float32], start: int, samples: NDArray[np.float32],
         size: int) -> None:
    """Write ``samples`` into a ring of ``size`` from ``start``."""
    count = samples.shape[0]
    start %= size
    first = min(count, size - start)
    line[start:start + first] = samples[:first]
    if first < count:
        line[:count - first] = samples[first:]


def _finite(value: float) -> float:
    """``value`` as a float, or nought where it is not a finite number."""
    number = float(value)
    return number if math.isfinite(number) else 0.0
