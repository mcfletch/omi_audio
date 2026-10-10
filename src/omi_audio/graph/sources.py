"""Nodes that produce sound: oscillators, buffers and captured input.

An :class:`Oscillator` plays one cycle of a :class:`PeriodicWave` over and
over.  The wave is a Fourier series, and the cycle played at a given pitch
holds only the harmonics below the Nyquist frequency at that pitch, so a
square wave at 5 kHz does not fold its upper harmonics back down as noise.

A :class:`BufferSource` plays multichannel samples held in memory, at any rate
and in either direction, once or in a loop.  A :class:`CaptureSource` plays
what a capture device writes into a :class:`CaptureRing`.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from omi_audio._numbers import finite
from omi_audio.graph.core import Graph, Node

__all__ = ['TABLE_SIZE', 'WAVE_TYPES', 'BufferSource', 'CaptureRing', 'CaptureSource',
           'Oscillator', 'PeriodicWave']

#: The waveforms a :class:`PeriodicWave` is made as; ``custom`` is one made
#: from coefficients.
WAVE_TYPES = ('sine', 'square', 'sawtooth', 'triangle', 'custom')

#: Samples in one cycle of an oscillator's table.  It holds up to half as many
#: harmonics, which reach the Nyquist frequency at a fundamental of about
#: 11 Hz at 44.1 kHz.
TABLE_SIZE = 4096

#: The most harmonics a table holds.
_MOST = TABLE_SIZE // 2 - 1

#: The harmonic counts tables are made for, three to an octave: the count for a
#: pitch is rounded down to one of these, so a sweep makes a few dozen tables
#: rather than one per pitch.
_LADDER = tuple(sorted({max(1, int(2.0 ** (step / 3.0))) for step in range(33)} | {_MOST}))


def _rung(harmonics: int) -> int:
    """The largest count on :data:`_LADDER` that is no more than ``harmonics``."""
    chosen = 1
    for count in _LADDER:
        if count > harmonics:
            break
        chosen = count
    return chosen


class PeriodicWave:
    """One cycle of a waveform, as the coefficients of a Fourier series.

    The cycle is ``sum(real[k] cos(2 pi k t) + imag[k] sin(2 pi k t))`` over
    ``k`` from 1, as the Web Audio API's ``PeriodicWave`` defines it; ``real[0]``
    and ``imag[0]`` are ignored.  With ``normalize`` the cycle is scaled so its
    peak is 1.
    """

    def __init__(self, real: ArrayLike = (), imag: ArrayLike = (),
                 normalize: bool = True) -> None:
        real_terms = np.asarray(real, dtype='d').reshape(-1)
        imag_terms = np.asarray(imag, dtype='d').reshape(-1)
        size = max(len(real_terms), len(imag_terms), 2)
        self.real = np.zeros(size)
        self.imag = np.zeros(size)
        self.real[:len(real_terms)] = np.nan_to_num(real_terms)
        self.imag[:len(imag_terms)] = np.nan_to_num(imag_terms)
        self.normalize = bool(normalize)
        self._tables: dict[int, NDArray[np.float32]] = {}
        full = self._cycle(_MOST)
        peak = float(np.abs(full).max()) if full.size else 0.0
        self._scale = 1.0 / peak if (self.normalize and peak > 0.0) else 1.0

    def __repr__(self) -> str:
        return '<%s %d terms>' % (type(self).__name__, len(self.real) - 1)

    @classmethod
    def of_type(cls, kind: str) -> PeriodicWave:
        """One of the standard waveforms of :data:`WAVE_TYPES`, by name.

        The series are the Web Audio API's: a sine is its first harmonic, a
        square has ``4 / (k pi)`` at the odd harmonics, a sawtooth
        ``(-1)**(k+1) 2 / (k pi)`` at every one, and a triangle
        ``8 sin(k pi / 2) / (k pi)**2``.

        Raises:
            ValueError: for a name that is not one of :data:`WAVE_TYPES`, or
                for ``custom``, which is made from coefficients.
        """
        if kind not in WAVE_TYPES or kind == 'custom':
            raise ValueError('wave type %r is not one of %s' % (kind, WAVE_TYPES[:-1]))
        k = np.arange(_MOST + 1, dtype='d')
        k[0] = 1.0
        if kind == 'sine':
            imag = np.zeros(2)
            imag[1] = 1.0
        elif kind == 'square':
            imag = 2.0 / (k * np.pi) * (1.0 - (-1.0) ** k)
        elif kind == 'sawtooth':
            imag = (-1.0) ** (k + 1.0) * 2.0 / (k * np.pi)
        else:
            imag = 8.0 * np.sin(k * np.pi / 2.0) / (k * np.pi) ** 2
        imag[0] = 0.0
        return cls(imag=imag)

    def _cycle(self, harmonics: int) -> NDArray[np.float64]:
        """One cycle of :data:`TABLE_SIZE` samples holding ``harmonics`` harmonics."""
        spectrum = np.zeros(TABLE_SIZE // 2 + 1, dtype=complex)
        count = min(len(self.real), harmonics + 1, TABLE_SIZE // 2)
        spectrum[1:count] = (self.real[1:count] - 1j * self.imag[1:count]) * (TABLE_SIZE / 2)
        return np.fft.irfft(spectrum, TABLE_SIZE)

    def table(self, harmonics: int) -> NDArray[np.float32]:
        """The cycle with at most ``harmonics`` harmonics, one sample longer than a cycle.

        The extra sample repeats the first, so interpolating past the last
        sample of the cycle reads the start of the next.  Tables are kept, by
        a rounded-down harmonic count.
        """
        harmonics = _rung(max(1, min(int(harmonics), _MOST)))
        found = self._tables.get(harmonics)
        if found is None:
            cycle = self._cycle(harmonics) * self._scale
            found = np.empty(TABLE_SIZE + 1, dtype=np.float32)
            found[:TABLE_SIZE] = cycle
            found[TABLE_SIZE] = cycle[0]
            self._tables[harmonics] = found
        return found


_SINE = PeriodicWave.of_type('sine')


class Oscillator(Node):
    """A periodic waveform at a frequency, in one channel.

    ``frequency`` is in hertz and ``detune`` in cents; the pitch played is
    ``frequency * 2 ** (detune / 1200)``, limited to the Nyquist frequency.  A
    change of pitch glides across the next block rather than stepping.  Set
    :attr:`wave` to play another :class:`PeriodicWave` than a sine.
    """

    def __init__(self, frequency: float = 440.0, detune: float = 0.0,
                 wave: PeriodicWave | None = None, gain: float = 1.0) -> None:
        super().__init__(gain=gain)
        self._frequency = finite(frequency)
        self._detune = finite(detune)
        self._wave = wave if wave is not None else _SINE
        self._table = self._wave.table(1)
        self._phase = 0.0
        self._increment = 0.0
        self._offsets = np.zeros(0)
        self._triangle = np.zeros(0)
        self._phases = np.zeros(0)
        self._lower = np.zeros(0, dtype=np.intp)
        self._fraction = np.zeros(0, dtype=np.float32)
        self._scratch = np.zeros(0, dtype=np.float32)

    @property
    def frequency(self) -> float:
        """Cycles a second."""
        return self._frequency

    @frequency.setter
    def frequency(self, value: float) -> None:
        self._frequency = finite(value)
        self._choose_table()

    @property
    def detune(self) -> float:
        """Cents above :attr:`frequency`; 1200 is an octave."""
        return self._detune

    @detune.setter
    def detune(self, value: float) -> None:
        self._detune = finite(value)
        self._choose_table()

    @property
    def wave(self) -> PeriodicWave:
        """The waveform one cycle plays."""
        return self._wave

    @wave.setter
    def wave(self, value: PeriodicWave | None) -> None:
        self._wave = value if value is not None else _SINE
        self._choose_table()

    @property
    def pitch(self) -> float:
        """The frequency played, in hertz, detune included and limited to Nyquist."""
        nyquist = self.sample_rate / 2.0
        pitch = self._frequency * 2.0 ** (self._detune / 1200.0)
        return max(-nyquist, min(nyquist, pitch))

    def start(self) -> None:
        """Play from the start of a cycle."""
        self._phase = 0.0
        self.playing = True

    def _choose_table(self) -> None:
        """The table holding the harmonics below Nyquist at the current pitch."""
        pitch = abs(self.pitch)
        harmonics = int(self.sample_rate / 2.0 / pitch) if pitch > 0.0 else _MOST
        self._table = self._wave.table(harmonics)

    def prepare(self, graph: Graph) -> None:
        size = graph.max_block
        self._offsets = np.arange(size, dtype='d')
        self._triangle = self._offsets * (self._offsets + 1.0) / 2.0
        self._phases = np.zeros(size)
        self._lower = np.zeros(size, dtype=np.intp)
        self._fraction = np.zeros(size, dtype=np.float32)
        self._scratch = np.zeros(size, dtype=np.float32)
        self._increment = self.pitch / graph.sample_rate
        self._choose_table()

    def output_channels(self, _computed: int) -> int:
        return 1

    def process(self, _block: NDArray[np.float32], out: NDArray[np.float32],
                frames: int) -> None:
        start = self._increment
        end = self.pitch / self.sample_rate
        phases = self._phases[:frames]
        # Phase n is the start phase plus the increments before it; with the
        # increment ramping from `start` to `end` that sum is a line plus a
        # triangle number, so a glide costs no loop.
        np.multiply(self._offsets[:frames], start, out=phases)
        if end != start:
            phases += self._triangle[:frames] * ((end - start) / frames)
        phases += self._phase
        total = frames * start + (end - start) * (frames + 1) / 2.0
        self._phase = math.fmod(self._phase + total, 1.0)
        self._increment = end
        np.mod(phases, 1.0, out=phases)
        phases *= TABLE_SIZE
        _interpolate(self._table, phases, self._lower[:frames], self._fraction[:frames],
                     self._scratch[:frames], out[0, :frames])


def _interpolate(table: NDArray[np.float32], positions: NDArray[np.float64],
                 lower: NDArray[np.intp], fraction: NDArray[np.float32],
                 scratch: NDArray[np.float32], target: NDArray[np.float32],
                 mode: Literal['clip', 'wrap'] = 'clip') -> None:
    """``table`` read at ``positions`` into ``target``, interpolating linearly.

    ``positions`` are at least zero, and are overwritten; the other arrays are
    scratch of the same length.
    """
    np.copyto(lower, positions, casting='unsafe')       # at least zero: trunc is floor
    np.subtract(positions, lower, out=positions)
    np.copyto(fraction, positions, casting='unsafe')
    np.take(table, lower, out=target, mode=mode)
    np.add(lower, 1, out=lower)
    np.take(table, lower, out=scratch, mode=mode)
    np.subtract(scratch, target, out=scratch)
    np.multiply(scratch, fraction, out=scratch)
    np.add(target, scratch, out=target)


class BufferSource(Node):
    """Samples held in memory, played once or looped, at any speed.

    ``buffer`` is ``(channels, frames)``, or one channel as a flat array,
    recorded at ``buffer_rate``.  ``playback_rate`` multiplies the speed and
    pitch together, and a negative rate plays backwards, starting from the end;
    ``detune`` is in cents and multiplies the rate by ``2 ** (detune / 1200)``.
    ``duration``, in seconds, plays only that much of the buffer, where it is
    greater than zero.

    With ``loop`` the playhead returns to ``loop_start`` whenever it reaches
    ``loop_end`` (both in seconds).  As in the Web Audio API, a ``loop_end``
    past the end means the end, and a ``loop_end`` of zero, a negative
    ``loop_start`` or one at or past ``loop_end`` loops the whole buffer.

    A source that is not looping sets :attr:`finished` when it plays out.
    """

    def __init__(self, buffer: ArrayLike = (), buffer_rate: int = 44100,
                 loop: bool = False, loop_start: float = 0.0, loop_end: float = 0.0,
                 playback_rate: float = 1.0, detune: float = 0.0,
                 duration: float = 0.0, gain: float = 1.0) -> None:
        super().__init__(gain=gain)
        self._buffer: tuple[NDArray[np.float32], int] = (np.zeros((1, 0), np.float32), 1)
        self.set_buffer(buffer, buffer_rate)
        self.loop = bool(loop)
        self.loop_start = finite(loop_start)
        self.loop_end = finite(loop_end)
        self.playback_rate = finite(playback_rate)
        self.detune = finite(detune)
        self.duration = finite(duration)
        #: Where the playhead is, in frames of the buffer.
        self.position = 0.0
        #: Whether a source that is not looping has played to its end.
        self.finished = False
        self._started = False
        self._offset: float | None = None
        self._index = np.zeros(0)
        self._wrapped = np.zeros(0)
        self._mask = np.zeros(0, dtype=bool)
        self._offsets = np.zeros(0)
        self._lower = np.zeros(0, dtype=np.intp)
        self._fraction = np.zeros(0, dtype=np.float32)
        self._positions = np.zeros(0)
        self._scratch = np.zeros(0, dtype=np.float32)

    def set_buffer(self, buffer: ArrayLike, buffer_rate: int) -> None:
        """Play ``buffer``, ``(channels, frames)`` or flat, recorded at ``buffer_rate``."""
        data = np.asarray(buffer, dtype=np.float32)
        if data.ndim < 2:
            data = data.reshape(1, -1)
        data = np.ascontiguousarray(np.nan_to_num(data[:, :]), dtype=np.float32)
        self._buffer = (data, max(1, int(buffer_rate)))

    @property
    def buffer(self) -> NDArray[np.float32]:
        """The samples, ``(channels, frames)``."""
        return self._buffer[0]

    @property
    def buffer_rate(self) -> int:
        """Frames per second the buffer was recorded at."""
        return self._buffer[1]

    @property
    def length(self) -> int:
        """Frames in the buffer."""
        return int(self._buffer[0].shape[1])

    @property
    def elapsed(self) -> float:
        """Where the playhead is, in seconds of the buffer."""
        return self.position / self.buffer_rate

    def start(self, offset: float | None = None) -> None:
        """Play from the start: the end, for a negative rate.

        ``offset``, in seconds of the buffer counted from its start, plays from
        there instead, in either direction; a loop folds an offset past its end
        back into the loop, and a source that does not loop and is given one
        past its end has finished.  A negative offset is the start, and a
        non-finite one is no offset.
        """
        self.finished = False
        self._started = False
        self._offset = (max(0.0, float(offset))
                        if offset is not None and math.isfinite(offset) else None)
        self.playing = True

    def _end(self) -> float:
        """The frame playing stops at."""
        data, rate = self._buffer
        end = float(data.shape[1])
        if self.duration > 0.0:
            end = min(end, self.duration * rate)
        return end

    def _first(self, step: float, end: float, rate: int) -> float:
        """Where the playhead starts: the offset :meth:`start` was given, or an end."""
        if self._offset is None:
            return max(end - 1.0, 0.0) if step < 0.0 else 0.0
        position = self._offset * rate
        if self.loop and end > 0.0:
            return self._wrap_one(position, 1.0, end)
        return position

    def _region(self, end: float) -> tuple[float, float]:
        """The loop's first frame and the frame it returns at."""
        rate = self.buffer_rate
        first, last = self.loop_start * rate, self.loop_end * rate
        if first < 0.0 or last <= 0.0 or first >= last:
            return 0.0, end
        return first, min(last, end)

    def prepare(self, graph: Graph) -> None:
        size = graph.max_block
        self._index = np.zeros(size)
        self._wrapped = np.zeros(size)
        self._mask = np.zeros(size, dtype=bool)
        self._offsets = np.arange(size, dtype='d')
        self._lower = np.zeros(size, dtype=np.intp)
        self._fraction = np.zeros(size, dtype=np.float32)
        self._positions = np.zeros(size)
        self._scratch = np.zeros(size, dtype=np.float32)

    def output_channels(self, _computed: int) -> int:
        return int(self._buffer[0].shape[0])

    def process(self, _block: NDArray[np.float32], out: NDArray[np.float32],
                frames: int) -> None:
        data, rate = self._buffer
        channels = data.shape[0]
        out[:channels, :frames] = 0.0
        end = self._end()
        step = self.playback_rate * 2.0 ** (self.detune / 1200.0) * rate / self.sample_rate
        if not self._started:
            self._started = True
            self.position = self._first(step, end, rate)
        if self.finished or end <= 0.0:
            self.finished = not self.loop
            return
        index = self._index[:frames]
        np.multiply(self._offsets[:frames], step, out=index)
        index += self.position
        if self.loop:
            available = frames
            self._wrap(index, step, end)
            self.position = self._wrap_one(self.position + step * frames, step, end)
            mode: Literal['clip', 'wrap'] = 'wrap'
        else:
            available = _within(self.position, step, end, frames)
            self.position += step * frames
            if available < frames:
                self.finished = True
            mode = 'clip'
        if available <= 0:
            return
        positions = self._positions[:available]
        lower = self._lower[:available]
        fraction = self._fraction[:available]
        for channel in range(channels):
            np.copyto(positions, index[:available])
            _interpolate(data[channel], positions, lower, fraction,
                         self._scratch[:available], out[channel, :available], mode)

    def _wrap(self, index: NDArray[np.float64], step: float, end: float) -> None:
        """Fold every playhead position that has run past the loop back into it."""
        first, last = self._region(end)
        span = last - first
        wrapped = self._wrapped[:index.shape[0]]
        mask = self._mask[:index.shape[0]]
        np.subtract(index, first, out=wrapped)
        np.mod(wrapped, span, out=wrapped)
        wrapped += first
        if step >= 0.0:
            np.greater_equal(index, last, out=mask)
        else:
            np.less(index, first, out=mask)
        np.copyto(index, wrapped, where=mask)

    def _wrap_one(self, position: float, step: float, end: float) -> float:
        """One playhead position folded back into the loop, as :meth:`_wrap` does."""
        first, last = self._region(end)
        if (step >= 0.0 and position >= last) or (step < 0.0 and position < first):
            return first + math.fmod(position - first, last - first) % (last - first)
        return position


def _within(position: float, step: float, end: float, frames: int) -> int:
    """How many of the next ``frames`` playhead positions lie in ``[0, end)``."""
    if not 0.0 <= position < end:
        return 0
    if step > 0.0:
        return max(0, min(frames, math.ceil((end - position) / step)))
    if step < 0.0:
        return max(0, min(frames, math.floor(position / -step) + 1))
    return frames


class CaptureRing:
    """Samples a capture thread writes and the audio thread reads, in one channel.

    One writer and one reader, each owning one index, so neither takes a lock.
    A write that would overrun what has not been read keeps what fits and drops
    the rest; a read with too little waiting returns what there is, and the
    caller plays silence for the remainder.
    """

    def __init__(self, capacity: int = 1 << 15) -> None:
        self._data = np.zeros(max(2, int(capacity)), dtype=np.float32)
        self._written = 0
        self._read = 0

    @property
    def capacity(self) -> int:
        """How many samples the ring holds."""
        return int(self._data.shape[0])

    @property
    def waiting(self) -> int:
        """How many samples are written and not yet read."""
        return self._written - self._read

    def write(self, samples: Sequence[float] | NDArray[np.float32]) -> int:
        """Append ``samples``; returns how many fitted.  The capture thread's call."""
        given = np.asarray(samples, dtype=np.float32).reshape(-1)
        room = self.capacity - (self._written - self._read)
        count = min(room, given.shape[0])
        self._copy_in(given[:count], self._written)
        # Advanced after the copy: the reader trusts every sample below it.
        self._written += count
        return count

    def _copy_in(self, given: NDArray[np.float32], at: int) -> None:
        size = self.capacity
        start = at % size
        first = min(given.shape[0], size - start)
        self._data[start:start + first] = given[:first]
        self._data[:given.shape[0] - first] = given[first:]

    def read_into(self, target: NDArray[np.float32]) -> int:
        """Fill ``target`` from the ring; returns how many samples were waiting."""
        size = self.capacity
        count = min(target.shape[0], self._written - self._read)
        start = self._read % size
        first = min(count, size - start)
        target[:first] = self._data[start:start + first]
        target[first:count] = self._data[:count - first]
        target[count:] = 0.0
        self._read += count
        return count


class CaptureSource(Node):
    """What a capture device delivers, as one channel; silence when nothing has arrived."""

    def __init__(self, ring: CaptureRing | None = None, gain: float = 1.0) -> None:
        super().__init__(gain=gain)
        #: Where the captured samples arrive.
        self.ring = ring if ring is not None else CaptureRing()

    def output_channels(self, _computed: int) -> int:
        return 1

    def process(self, _block: NDArray[np.float32], out: NDArray[np.float32],
                frames: int) -> None:
        self.ring.read_into(out[0, :frames])
