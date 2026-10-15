"""Filters: the biquad, the delay line and the wave shaper.

**The biquad** is the second-order filter of the Web Audio API's
``BiquadFilterNode``, its coefficients those of Robert Bristow-Johnson's
"Audio EQ Cookbook" as that specification gives them
(:func:`biquad_coefficients`).  A recursive filter is a loop over samples,
which in Python would cost microseconds a sample; here it is factored into
its two poles, ``1 / ((1 - p1 z^-1)(1 - p2 z^-1))``, and each first-order
recursion ``r[n] = u[n] + p r[n-1]`` is evaluated a stretch at a time in
closed form, ``r[n] = p^(n+1) r[-1] + p^n sum(p^-m u[m])``, with NumPy's
cumulative sum.  The stretch is short enough that ``p^-m`` stays far inside
the range of a double, so the result agrees with the sample-by-sample filter
to rounding.  The filter's state is kept as its last two inputs and outputs,
as the direct form keeps it, so a change of coefficients carries on from the
same signal.

**The delay line** holds ``max_delay_time`` seconds per channel and reads at
a fractional position, so a delay that changes glides across a block.

**The wave shaper** maps each sample through a curve, as the Web Audio API's
``WaveShaperNode`` does, optionally at two or four times the rate to keep
the harmonics the curve adds from folding back below Nyquist.
"""

from __future__ import annotations

import cmath
import math
from typing import Any, NamedTuple

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from numpy.typing import ArrayLike, NDArray

from omi_audio._numbers import finite
from omi_audio.graph.core import Graph, Node

__all__ = ['FILTER_TYPES', 'LONGEST_STRETCH', 'OVERSAMPLING', 'BiquadFilter', 'Delay', 'WaveShaper',
           'biquad_coefficients']

#: The biquad's responses, as the Web Audio API names them.
FILTER_TYPES = ('lowpass', 'highpass', 'bandpass', 'lowshelf', 'highshelf',
                'peaking', 'notch', 'allpass')

#: How much faster than the graph a :class:`WaveShaper` may run its curve.
OVERSAMPLING = ('none', '2x', '4x')

#: The largest ``p^-m`` a stretch of the recursion may reach.
_REACH = 300.0

#: The longest stretch a pole's powers are tabled for, in frames.  A block
#: longer than this is filtered in several stretches; a table this long costs
#: a redesign tens of microseconds, so a frequency swept every frame is cheap.
LONGEST_STRETCH = 512


def biquad_coefficients(kind: str, frequency: float, sample_rate: float,
                        q: float = 1.0, gain: float = 0.0,
                        detune: float = 0.0) -> tuple[float, float, float, float, float]:
    """``(b0, b1, b2, a1, a2)`` of a biquad, normalised so ``a0`` is 1.

    ``frequency`` is in hertz and moved by ``detune`` cents; ``q`` is the
    quality factor, which the low- and high-pass filters read in decibels as
    the Web Audio API specifies; ``gain`` is the shelving and peaking filters'
    boost in decibels.  A frequency at or beyond zero or Nyquist, and a ``q``
    of zero, give the limit the formulas reach there: a low-pass at Nyquist
    passes everything, an all-pass with no ``q`` inverts, and so on.

    Raises:
        ValueError: for a ``kind`` that is not one of :data:`FILTER_TYPES`.
    """
    if kind not in FILTER_TYPES:
        raise ValueError('filter type %r is not one of %s' % (kind, FILTER_TYPES))
    nyquist = sample_rate / 2.0
    frequency = finite(frequency) * 2.0 ** (finite(detune) / 1200.0)
    amplitude = 10.0 ** (finite(gain) / 40.0)
    q = finite(q)
    identity = (1.0, 0.0, 0.0, 0.0, 0.0)
    silence = (0.0, 0.0, 0.0, 0.0, 0.0)
    boost = (amplitude * amplitude, 0.0, 0.0, 0.0, 0.0)
    low, high = frequency <= 0.0, frequency >= nyquist
    if low or high:
        edges = {
            'lowpass': (silence, identity), 'highpass': (identity, silence),
            'bandpass': (silence, silence), 'lowshelf': (identity, boost),
            'highshelf': (boost, identity),
        }
        return edges.get(kind, (identity, identity))[1 if high else 0]
    if q <= 0.0 and kind in ('bandpass', 'notch', 'allpass', 'peaking'):
        return {'bandpass': identity, 'notch': silence,
                'allpass': (-1.0, 0.0, 0.0, 0.0, 0.0), 'peaking': boost}[kind]
    w0 = 2.0 * math.pi * frequency / sample_rate
    cos, sin = math.cos(w0), math.sin(w0)
    alpha_q = sin / (2.0 * q) if q > 0.0 else 0.0
    alpha_db = sin / (2.0 * 10.0 ** (q / 20.0))
    alpha_s = sin / 2.0 * math.sqrt(2.0)
    root = 2.0 * alpha_s * math.sqrt(amplitude)
    a = amplitude
    if kind == 'lowpass':
        b, d = ((1 - cos) / 2, 1 - cos, (1 - cos) / 2), (1 + alpha_db, -2 * cos, 1 - alpha_db)
    elif kind == 'highpass':
        b, d = ((1 + cos) / 2, -(1 + cos), (1 + cos) / 2), (1 + alpha_db, -2 * cos, 1 - alpha_db)
    elif kind == 'bandpass':
        b, d = (alpha_q, 0.0, -alpha_q), (1 + alpha_q, -2 * cos, 1 - alpha_q)
    elif kind == 'notch':
        b, d = (1.0, -2 * cos, 1.0), (1 + alpha_q, -2 * cos, 1 - alpha_q)
    elif kind == 'allpass':
        b, d = (1 - alpha_q, -2 * cos, 1 + alpha_q), (1 + alpha_q, -2 * cos, 1 - alpha_q)
    elif kind == 'peaking':
        b = (1 + alpha_q * a, -2 * cos, 1 - alpha_q * a)
        d = (1 + alpha_q / a, -2 * cos, 1 - alpha_q / a)
    elif kind == 'lowshelf':
        b = (a * ((a + 1) - (a - 1) * cos + root), 2 * a * ((a - 1) - (a + 1) * cos),
             a * ((a + 1) - (a - 1) * cos - root))
        d = ((a + 1) + (a - 1) * cos + root, -2 * ((a - 1) + (a + 1) * cos),
             (a + 1) + (a - 1) * cos - root)
    else:
        b = (a * ((a + 1) + (a - 1) * cos + root), -2 * a * ((a - 1) + (a + 1) * cos),
             a * ((a + 1) + (a - 1) * cos - root))
        d = ((a + 1) - (a - 1) * cos + root, 2 * ((a - 1) - (a + 1) * cos),
             (a + 1) - (a - 1) * cos - root)
    return b[0] / d[0], b[1] / d[0], b[2] / d[0], d[1] / d[0], d[2] / d[0]


class _Pole(NamedTuple):
    """One pole, and the powers of it a stretch of the recursion multiplies by."""

    pole: complex
    #: ``p^-m`` for ``m`` from 0; its length is the stretch.
    inverse: NDArray[np.complex128]
    #: ``p^n`` for ``n`` from 0.
    forward: NDArray[np.complex128]
    #: ``p^(n+1)``.
    onward: NDArray[np.complex128]


def _pole(pole: complex, longest: int) -> _Pole:
    """``pole`` with its powers over a stretch whose ``p^-m`` stays below 1e130.

    The powers are made from the pole's size and angle, ``|p|^n`` and
    ``n * arg(p)``, which costs a few real operations on the stretch where
    raising a complex number to each power costs a complex logarithm apiece.
    """
    pole = complex(pole)
    size = abs(pole)
    if size == 0.0 or size >= 1.0:
        stretch = longest
    else:
        stretch = max(1, min(longest, int(_REACH / -math.log(size))))
    if size == 0.0:
        forward = np.zeros(stretch, dtype=complex)
        forward[0] = 1.0
        return _Pole(pole, forward.copy(), forward, forward * pole)
    steps = np.arange(stretch, dtype='d')
    turn = steps * cmath.phase(pole)
    forward = np.empty(stretch, dtype=complex)
    inverse = np.empty(stretch, dtype=complex)
    np.cos(turn, out=forward.real)
    np.sin(turn, out=forward.imag)
    np.conjugate(forward, out=inverse)
    grown = np.exp(steps * math.log(size))
    forward *= grown
    inverse /= grown
    return _Pole(pole, inverse, forward, forward * pole)


class _Design(NamedTuple):
    """A biquad's coefficients and poles, swapped in whole."""

    b0: float
    b1: float
    b2: float
    first: _Pole
    second: _Pole


class BiquadFilter(Node):
    """A second-order filter of one of :data:`FILTER_TYPES`, on every channel.

    ``frequency`` in hertz, ``detune`` in cents, ``q`` the quality factor and
    ``filter_gain`` the shelving and peaking boost in decibels; see
    :func:`biquad_coefficients`.  :attr:`~omi_audio.graph.Node.gain` is the
    linear gain on the output, as on every node.
    """

    def __init__(self, kind: str = 'lowpass', frequency: float = 350.0,
                 q: float = 1.0, detune: float = 0.0, filter_gain: float = 0.0,
                 gain: float = 1.0, **named: Any) -> None:
        super().__init__(gain=gain, **named)
        if kind not in FILTER_TYPES:
            raise ValueError('filter type %r is not one of %s' % (kind, FILTER_TYPES))
        self._kind = kind
        self._frequency = finite(frequency)
        self._q = finite(q)
        self._detune = finite(detune)
        self._filter_gain = finite(filter_gain)
        self._design: _Design | None = None
        self._state = np.zeros((1, 4))
        self._x = np.zeros(0)
        self._u = np.zeros(0, dtype=complex)
        self._v = np.zeros(0, dtype=complex)
        self._y = np.zeros(0, dtype=complex)
        self._scratch = np.zeros(0, dtype=complex)
        self._carry = np.zeros(0, dtype=complex)

    @property
    def kind(self) -> str:
        """Which response, one of :data:`FILTER_TYPES`."""
        return self._kind

    @kind.setter
    def kind(self, value: str) -> None:
        if value not in FILTER_TYPES:
            raise ValueError('filter type %r is not one of %s' % (value, FILTER_TYPES))
        self._kind = value
        self._redesign()

    @property
    def frequency(self) -> float:
        """The corner, centre or shelf frequency, in hertz."""
        return self._frequency

    @frequency.setter
    def frequency(self, value: float) -> None:
        self._frequency = finite(value)
        self._redesign()

    @property
    def q(self) -> float:
        """The quality factor; in decibels for the low- and high-pass."""
        return self._q

    @q.setter
    def q(self, value: float) -> None:
        self._q = finite(value)
        self._redesign()

    @property
    def detune(self) -> float:
        """Cents the frequency is moved by."""
        return self._detune

    @detune.setter
    def detune(self, value: float) -> None:
        self._detune = finite(value)
        self._redesign()

    @property
    def filter_gain(self) -> float:
        """The shelving and peaking boost, in decibels."""
        return self._filter_gain

    @filter_gain.setter
    def filter_gain(self, value: float) -> None:
        self._filter_gain = finite(value)
        self._redesign()

    def coefficients(self) -> tuple[float, float, float, float, float]:
        """``(b0, b1, b2, a1, a2)`` at the graph's rate."""
        return biquad_coefficients(self._kind, self._frequency, self.sample_rate,
                                   q=self._q, gain=self._filter_gain, detune=self._detune)

    def response(self, frequency: float) -> complex:
        """The filter's complex response at ``frequency`` hertz."""
        b0, b1, b2, a1, a2 = self.coefficients()
        z = cmath.exp(-2j * math.pi * frequency / self.sample_rate)
        return (b0 + b1 * z + b2 * z * z) / (1.0 + a1 * z + a2 * z * z)

    def _redesign(self) -> None:
        """Work out the coefficients and poles; the control thread's job."""
        if self._graph is None:
            return
        b0, b1, b2, a1, a2 = self.coefficients()
        root = cmath.sqrt(a1 * a1 - 4.0 * a2)
        longest = min(self._graph.max_block, LONGEST_STRETCH)
        first = _pole((-a1 + root) / 2.0, longest)
        if root.imag:
            # The poles are a conjugate pair, and so are their powers.
            second = _Pole(first.pole.conjugate(), first.inverse.conj(),
                           first.forward.conj(), first.onward.conj())
        else:
            second = _pole((-a1 - root) / 2.0, longest)
        self._design = _Design(b0, b1, b2, first, second)

    def prepare(self, graph: Graph) -> None:
        size = graph.max_block
        self._state = np.zeros((graph.max_channels, 4))
        self._x = np.zeros(size + 2)
        self._u = np.zeros(size, dtype=complex)
        self._v = np.zeros(size, dtype=complex)
        self._y = np.zeros(size, dtype=complex)
        self._scratch = np.zeros(size, dtype=complex)
        self._carry = np.zeros(size, dtype=complex)
        self._redesign()

    def resume(self, missed: int) -> None:  # noqa: ARG002 - the base method's signature is kept
        """Start from silence: no input or output remembered."""
        self._state.fill(0.0)

    def process(self, block: NDArray[np.float32], out: NDArray[np.float32],
                frames: int) -> None:
        design = self._design
        if design is None:  # pragma: no cover - prepared before any render
            return
        x = self._x[:frames + 2]
        u = self._u[:frames]
        v = self._v[:frames]
        y = self._y[:frames]
        for channel in range(block.shape[0]):
            # State: the last two inputs and outputs, newest first.
            x1, x2, y1, y2 = self._state[channel]
            x[0], x[1] = x2, x1
            x[2:] = block[channel]
            # The feed-forward half: u[n] = b0 x[n] + b1 x[n-1] + b2 x[n-2].
            np.multiply(x[2:], design.b0, out=u)
            self._fold(u, x[1:-1], design.b1)
            self._fold(u, x[:-2], design.b2)
            # The poles, one at a time; the first stage's state is what the
            # second's output history implies, v[-1] = y[-1] - p2 y[-2].
            self._recur(u, design.first, y1 - design.second.pole * y2, v)
            self._recur(v, design.second, complex(y1), y)
            np.copyto(out[channel, :frames], y.real, casting='same_kind')
            if frames >= 2:
                self._state[channel] = (x[-1], x[-2], y[-1].real, y[-2].real)
            else:
                self._state[channel] = (x[-1], x1, y[-1].real, y1)

    def _fold(self, u: NDArray[np.complex128], x: NDArray[np.float64],
              weight: float) -> None:
        """``u += weight * x``."""
        if weight:
            scratch = self._scratch[:u.shape[0]]
            np.multiply(x, weight, out=scratch)
            np.add(u, scratch, out=u)

    def _recur(self, given: NDArray[np.complex128], pole: _Pole, start: complex,
               target: NDArray[np.complex128]) -> None:
        """``target[n] = given[n] + p target[n-1]``, from ``target[-1] = start``."""
        if pole.pole == 0:
            np.copyto(target, given)
            return
        stretch = pole.inverse.shape[0]
        frames = given.shape[0]
        carry = start
        for at in range(0, frames, stretch):
            count = min(stretch, frames - at)
            part = target[at:at + count]
            scratch = self._scratch[:count]
            np.multiply(given[at:at + count], pole.inverse[:count], out=scratch)
            np.cumsum(scratch, out=scratch)
            np.multiply(scratch, pole.forward[:count], out=part)
            onward = self._carry[:count]
            np.multiply(pole.onward[:count], carry, out=onward)
            np.add(part, onward, out=part)
            carry = complex(part[-1])


class Delay(Node):
    """Its input, ``delay_time`` seconds later.

    The line holds ``max_delay_time`` seconds, and ``delay_time`` is limited to
    it.  A changed delay is reached by the end of the next block, gliding
    there, so the pitch bends rather than the signal jumping.
    """

    def __init__(self, delay_time: float = 0.0, max_delay_time: float = 1.0,
                 gain: float = 1.0, **named: Any) -> None:
        super().__init__(gain=gain, **named)
        self._max_delay_time = max(0.0, finite(max_delay_time))
        self._delay_time = finite(delay_time)
        self._delay_now = self.delay_time
        self._line = np.zeros((1, 2), dtype=np.float32)
        self._written = 0
        self._positions = np.zeros(0)
        self._offsets = np.zeros(0)
        self._steps = np.zeros(0)
        self._lower = np.zeros(0, dtype=np.intp)
        self._fraction = np.zeros(0, dtype=np.float32)
        self._scratch = np.zeros(0, dtype=np.float32)
        self._read = np.zeros(0)

    @property
    def delay_time(self) -> float:
        """Seconds of delay, from nought to :attr:`max_delay_time`."""
        return max(0.0, min(self._delay_time, self._max_delay_time))

    @delay_time.setter
    def delay_time(self, value: float) -> None:
        self._delay_time = finite(value)

    def settle(self) -> None:
        super().settle()
        self._delay_now = self.delay_time

    @property
    def reached(self) -> float:
        """The delay the last block ended at, in seconds."""
        return self._delay_now

    @property
    def max_delay_time(self) -> float:
        """Seconds the line holds; setting it empties the line."""
        return self._max_delay_time

    @max_delay_time.setter
    def max_delay_time(self, value: float) -> None:
        self._max_delay_time = max(0.0, finite(value))
        if self._graph is not None:
            self._allocate(self._graph)

    def _allocate(self, graph: Graph) -> None:
        length = int(math.ceil(self._max_delay_time * graph.sample_rate)) + graph.max_block + 2
        self._written = 0
        self._line = np.zeros((graph.max_channels, length), dtype=np.float32)

    def prepare(self, graph: Graph) -> None:
        size = graph.max_block
        self._positions = np.zeros(size)
        self._read = np.zeros(size)
        self._offsets = np.arange(size, dtype='d')
        self._steps = self._offsets + 1.0
        self._lower = np.zeros(size, dtype=np.intp)
        self._fraction = np.zeros(size, dtype=np.float32)
        self._scratch = np.zeros(size, dtype=np.float32)
        self._allocate(graph)

    def resume(self, missed: int) -> None:  # noqa: ARG002 - the base method's signature is kept
        """Empty the line of the channels in use, and take the delay as reached."""
        self._line[:self.channels].fill(0.0)
        self._delay_now = self.delay_time

    def process(self, block: NDArray[np.float32], out: NDArray[np.float32],
                frames: int) -> None:
        line = self._line
        length = line.shape[1]
        start = self._written % length
        rate = self.sample_rate
        delay_start, delay_end = self._delay_now * rate, self.delay_time * rate
        self._delay_now = self.delay_time
        # Where each output sample reads: its own time less the delay, which
        # glides from the last block's to this one's.
        read = self._read[:frames]
        np.multiply(self._steps[:frames], (delay_end - delay_start) / frames, out=read)
        read += delay_start
        np.subtract(self._offsets[:frames], read, out=read)
        read += start
        np.mod(read, length, out=read)
        first = min(frames, length - start)
        undelayed = delay_start == 0.0 and delay_end == 0.0
        for channel in range(block.shape[0]):
            row = line[channel]
            row[start:start + first] = block[channel, :first]
            row[:frames - first] = block[channel, first:]
            if undelayed:
                # The line is still written, for the delay to glide from.
                np.copyto(out[channel, :frames], block[channel])
                continue
            positions = self._positions[:frames]
            np.copyto(positions, read)
            _read_line(row, positions, self._lower[:frames], self._fraction[:frames],
                       self._scratch[:frames], out[channel, :frames])
        self._written = start + frames


def _read_line(row: NDArray[np.float32], positions: NDArray[np.float64],
               lower: NDArray[np.intp], fraction: NDArray[np.float32],
               scratch: NDArray[np.float32], target: NDArray[np.float32]) -> None:
    """``row`` read at fractional ``positions``, wrapping, into ``target``."""
    np.copyto(lower, positions, casting='unsafe')
    np.subtract(positions, lower, out=positions)
    np.copyto(fraction, positions, casting='unsafe')
    np.take(row, lower, out=target, mode='wrap')
    np.add(lower, 1, out=lower)
    np.take(row, lower, out=scratch, mode='wrap')
    np.subtract(scratch, target, out=scratch)
    np.multiply(scratch, fraction, out=scratch)
    np.add(target, scratch, out=target)


#: Filter taps per phase of the wave shaper's resampling filters.
_TAPS_PER_PHASE = 17

#: The delay each resampling filter adds, in samples at the graph's rate.
_HALF_SPAN = (_TAPS_PER_PHASE - 1) // 2


def _resampling_filter(factor: int) -> NDArray[np.float64]:
    """A Blackman-windowed sinc low-pass at 0.45 of the base rate's Nyquist, unity gain.

    ``16 * factor + 1`` taps centred on tap ``8 * factor``, so each filter
    delays by exactly eight samples at the base rate; padded with zeros to
    :data:`_TAPS_PER_PHASE` taps for each phase.
    """
    span = 2 * _HALF_SPAN * factor + 1
    n = np.arange(span) - (span - 1) / 2.0
    cutoff = 0.45 / factor
    kernel = 2.0 * cutoff * np.sinc(2.0 * cutoff * n) * np.blackman(span)
    padded = np.zeros(_TAPS_PER_PHASE * factor)
    padded[:span] = kernel / kernel.sum()
    return padded


def _as_curve(value: ArrayLike) -> NDArray[np.float64]:
    """``value`` as a flat, finite curve."""
    return np.ascontiguousarray(np.nan_to_num(np.asarray(value, dtype='d').reshape(-1)))


class WaveShaper(Node):
    """Each sample mapped through ``curve``.

    The curve spans inputs from -1 to 1 evenly, and a sample between two of its
    points is mapped between them linearly; a sample outside -1 to 1 maps to
    the curve's end.  A curve of fewer than two points passes the input
    unchanged.  ``oversample`` ``2x`` or ``4x`` applies the curve at that
    multiple of the rate, between two low-pass filters, which delays the
    output by ``16`` samples at the graph's rate.
    """

    def __init__(self, curve: ArrayLike = (), oversample: str = 'none',
                 gain: float = 1.0, **named: Any) -> None:
        super().__init__(gain=gain, **named)
        self._curve = _as_curve(curve)
        if oversample not in OVERSAMPLING:
            raise ValueError('oversample %r is not one of %s' % (oversample, OVERSAMPLING))
        self._oversample = oversample
        self._factor = 1
        self._kernel = np.ones(1)
        self._up = np.zeros((1, 0))
        self._down = np.zeros((1, 0))
        self._wide = np.zeros(0)
        self._narrow = np.zeros(0)
        self._narrow_runs = np.zeros((0, _TAPS_PER_PHASE))
        self._wide_runs = np.zeros((0, 1))
        self._up_taps = np.ones((1, _TAPS_PER_PHASE))
        self._down_taps = np.ones(1)
        self._product = np.zeros(0)
        self._index = np.zeros(0)
        self._lower = np.zeros(0, dtype=np.intp)
        self._fraction = np.zeros(0)
        self._scratch = np.zeros(0)

    @property
    def curve(self) -> NDArray[np.float64]:
        """The shaping curve."""
        return self._curve

    @curve.setter
    def curve(self, value: ArrayLike) -> None:
        self._curve = _as_curve(value)

    @property
    def oversample(self) -> str:
        """One of :data:`OVERSAMPLING`."""
        return self._oversample

    @oversample.setter
    def oversample(self, value: str) -> None:
        if value not in OVERSAMPLING:
            raise ValueError('oversample %r is not one of %s' % (value, OVERSAMPLING))
        self._oversample = value
        if self._graph is not None:
            self._allocate(self._graph)

    def _allocate(self, graph: Graph) -> None:
        factor = {'none': 1, '2x': 2, '4x': 4}[self._oversample]
        wide = graph.max_block * factor
        taps = _TAPS_PER_PHASE * factor
        self._kernel = _resampling_filter(factor) if factor > 1 else np.ones(1)
        self._up = np.zeros((graph.max_channels, _TAPS_PER_PHASE - 1))
        self._down = np.zeros((graph.max_channels, taps - 1))
        self._wide = np.zeros(taps - 1 + wide)
        self._narrow = np.zeros(_TAPS_PER_PHASE - 1 + graph.max_block)
        # Every run of taps in each buffer, as rows: a filter's output is
        # those rows times its taps.  Each phase of the up-sampled signal is
        # the input filtered by every factor-th tap from that phase, which is
        # zero-stuffing without the zeros; the low-passed signal is wanted
        # only at every factor-th sample, so only those rows are taken.
        if factor > 1:
            self._narrow_runs = sliding_window_view(self._narrow, _TAPS_PER_PHASE)
            self._wide_runs = sliding_window_view(self._wide, taps)[::factor]
            self._up_taps = np.ascontiguousarray(
                self._kernel.reshape(_TAPS_PER_PHASE, factor).T[:, ::-1] * factor)
            self._down_taps = np.ascontiguousarray(self._kernel[::-1])
        self._product = np.zeros(max(wide, graph.max_block))
        self._index = np.zeros(wide)
        self._lower = np.zeros(wide, dtype=np.intp)
        self._fraction = np.zeros(wide)
        self._scratch = np.zeros(wide)
        self._factor = factor

    def prepare(self, graph: Graph) -> None:
        self._allocate(graph)

    def resume(self, missed: int) -> None:  # noqa: ARG002 - the base method's signature is kept
        """Empty the resampling filters' histories."""
        self._up.fill(0.0)
        self._down.fill(0.0)

    def process(self, block: NDArray[np.float32], out: NDArray[np.float32],
                frames: int) -> None:
        curve = self._curve
        if curve.shape[0] < 2:
            np.copyto(out[:block.shape[0], :frames], block)
            return
        factor = self._factor
        if factor == 1:
            for channel in range(block.shape[0]):
                samples = self._wide[:frames]
                np.copyto(samples, block[channel])
                self._shape(samples)
                np.copyto(out[channel, :frames], samples, casting='same_kind')
            return
        for channel in range(block.shape[0]):
            self._oversampled(block[channel], out[channel, :frames], channel, frames)

    def _oversampled(self, given: NDArray[np.float32], target: NDArray[np.float32],
                     channel: int, frames: int) -> None:
        """Up-sample, shape, low-pass and down-sample one channel."""
        factor = self._factor
        kernel = self._kernel
        history = _TAPS_PER_PHASE - 1
        taps = kernel.shape[0]
        narrow = self._narrow[:history + frames]
        narrow[:history] = self._up[channel]
        narrow[history:] = given
        self._up[channel] = narrow[frames:]
        wide = self._wide[:taps - 1 + frames * factor]
        wide[:taps - 1] = self._down[channel]
        shaped = wide[taps - 1:]
        # A sum of products over the rows as they lie in the buffer: a matrix
        # product would first copy them out.
        np.einsum('nj,pj->np', self._narrow_runs[:frames], self._up_taps,
                  out=shaped.reshape(frames, factor))
        self._shape(shaped)
        self._down[channel] = wide[-(taps - 1):]
        result = self._product[:frames]
        np.einsum('nj,j->n', self._wide_runs[:frames], self._down_taps, out=result)
        np.copyto(target, result, casting='same_kind')

    def _shape(self, samples: NDArray[np.float64]) -> None:
        """Map ``samples`` through the curve, in place."""
        curve = self._curve
        last = curve.shape[0] - 1
        count = samples.shape[0]
        index = self._index[:count]
        lower = self._lower[:count]
        fraction = self._fraction[:count]
        scratch = self._scratch[:count]
        np.add(samples, 1.0, out=index)
        np.multiply(index, last / 2.0, out=index)
        np.clip(index, 0.0, float(last), out=index)
        np.copyto(lower, index, casting='unsafe')
        np.minimum(lower, last - 1, out=lower)
        np.subtract(index, lower, out=fraction)
        np.take(curve, lower, out=samples, mode='clip')
        np.add(lower, 1, out=lower)
        np.take(curve, lower, out=scratch, mode='clip')
        np.subtract(scratch, samples, out=scratch)
        np.multiply(scratch, fraction, out=scratch)
        np.add(samples, scratch, out=samples)
