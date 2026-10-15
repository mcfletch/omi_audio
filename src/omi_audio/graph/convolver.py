"""Convolution with an impulse response: a room, a speaker cabinet, a telephone.

The impulse response is cut into partitions of :data:`PARTITION` frames, each
transformed once when the response is set.  Every :data:`PARTITION` frames of
input are transformed once and kept, and each output partition is the sum of
the kept input spectra times the response's, transformed back -- uniformly
partitioned overlap-save, whose cost grows with the response's length only
through that sum.  The output is :data:`PARTITION` frames later than the
input: about 6 ms at 44.1 kHz.

Channels follow the Web Audio API's ``ConvolverNode``: a mono response is
applied to each of up to two input channels; a stereo response gives the left
output from the left (or only) input channel and the right from the right;
and a four-channel response is "true stereo", each output the sum of both
inputs through their own responses.  :func:`normalization_scale` is the
specification's equal-power normalisation.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from omi_audio.graph.core import Graph, Node

__all__ = ['PARTITION', 'Convolver', 'normalization_scale']

#: Frames per partition, and the convolver's delay.
PARTITION = 256

#: The Web Audio API's normalisation constants.
_GAIN_CALIBRATION = 0.00125
_GAIN_CALIBRATION_SAMPLE_RATE = 44100.0
_MIN_POWER = 0.000125

#: Which input channel through which response channel into which output
#: channel, by (input channels, response channels).
_PATHS = {
    (1, 1): ((0, 0, 0),),
    (2, 1): ((0, 0, 0), (1, 0, 1)),
    (1, 2): ((0, 0, 0), (0, 1, 1)),
    (2, 2): ((0, 0, 0), (1, 1, 1)),
    (1, 4): ((0, 0, 0), (0, 1, 1), (0, 2, 0), (0, 3, 1)),
    (2, 4): ((0, 0, 0), (0, 1, 1), (1, 2, 0), (1, 3, 1)),
}


def normalization_scale(response: ArrayLike, sample_rate: float) -> float:
    """The scale the Web Audio API applies to ``response``, ``(channels, frames)``.

    One over the response's RMS power (at least :data:`_MIN_POWER`), times a
    calibration of 0.00125 at 44.1 kHz scaled to ``sample_rate``, halved for a
    four-channel response.
    """
    data = np.asarray(response, dtype='d')
    if data.ndim < 2:
        data = data.reshape(1, -1)
    channels, frames = data.shape
    power = math.sqrt(float(np.sum(data * data)) / max(1, channels * frames))
    scale = 1.0 / max(power, _MIN_POWER) * _GAIN_CALIBRATION
    if sample_rate > 0:
        scale *= _GAIN_CALIBRATION_SAMPLE_RATE / sample_rate
    if channels == 4:
        scale *= 0.5
    return scale


class _Response:
    """A response's partitions, transformed, and the paths through it."""

    def __init__(self, response: NDArray[np.float64]) -> None:
        channels, frames = response.shape
        self.channels = channels
        self.partitions = max(1, -(-frames // PARTITION))
        padded = np.zeros((channels, self.partitions, 2 * PARTITION))
        for index in range(self.partitions):
            part = response[:, index * PARTITION:(index + 1) * PARTITION]
            padded[:, index, :part.shape[1]] = part
        #: ``(channels, partitions, PARTITION + 1)`` spectra.
        self.spectra = np.fft.rfft(padded, axis=-1)


class Convolver(Node):
    """Its input convolved with an impulse response.

    ``response`` is ``(channels, frames)`` or one flat channel, of 1, 2 or 4
    channels.  With ``normalize`` it is scaled by :func:`normalization_scale`
    when it is set.  With no response, the output is silence.
    """

    def __init__(self, response: ArrayLike = (), normalize: bool = False,
                 gain: float = 1.0, **named: Any) -> None:
        named.setdefault('channel_count', 2)
        named.setdefault('channel_count_mode', 'clamped-max')
        super().__init__(gain=gain, **named)
        self._normalize = bool(normalize)
        self._raw = np.zeros((1, 0))
        self._response: _Response | None = None
        self._history = np.zeros((2, PARTITION))
        self._partial = np.zeros(PARTITION + 1, dtype=complex)
        self._spectra = np.zeros((2, 0, 0), dtype=complex)
        self._head = 0
        self._cursor = 0
        self._pending = np.zeros((2, PARTITION))
        self._ready = np.zeros((2, PARTITION))
        self._frame = np.zeros(2 * PARTITION)
        self._product = np.zeros((0, 0), dtype=complex)
        self._sum = np.zeros((2, PARTITION + 1), dtype=complex)
        self._wave = np.zeros(2 * PARTITION)
        self.set_response(response)

    @property
    def normalize(self) -> bool:
        """Whether the response is scaled by :func:`normalization_scale`."""
        return self._normalize

    @normalize.setter
    def normalize(self, value: bool) -> None:
        self._normalize = bool(value)
        self._design()

    @property
    def response(self) -> NDArray[np.float64]:
        """The response as given, ``(channels, frames)``."""
        return self._raw

    def set_response(self, response: ArrayLike) -> None:
        """Convolve with ``response`` from the next block; 3 or more than 4 channels keep 2."""
        data = np.nan_to_num(np.asarray(response, dtype='d'))
        if data.ndim < 2:
            data = data.reshape(1, -1)
        if data.shape[0] not in (1, 2, 4):
            data = data[:2]
        self._raw = data
        self._design()

    def _design(self, sample_rate: float | None = None) -> None:
        """Transform the response, and empty the input history to match."""
        data = self._raw
        if data.shape[1] == 0:
            self._response = None
            return
        if self._normalize:
            data = data * normalization_scale(
                data, self.sample_rate if sample_rate is None else sample_rate)
        response = _Response(data)
        self._spectra = np.zeros((2, response.partitions, PARTITION + 1), dtype=complex)
        self._product = np.zeros((response.partitions, PARTITION + 1), dtype=complex)
        self._head = 0
        self._history.fill(0.0)
        self._response = response

    def prepare(self, graph: Graph) -> None:
        self._design(graph.sample_rate)

    def output_channels(self, computed: int) -> int:
        response = self._response
        if response is None:
            return 1
        return 2 if (response.channels > 1 or computed > 1) else 1

    def resume(self, missed: int) -> None:  # noqa: ARG002 - the base method's signature is kept
        """Forget the input so far: the response's tail of it is not heard."""
        self._history.fill(0.0)
        self._spectra.fill(0.0)
        self._pending.fill(0.0)
        self._ready.fill(0.0)
        self._head = 0
        self._cursor = 0

    def process(self, block: NDArray[np.float32], out: NDArray[np.float32],
                frames: int) -> None:
        response = self._response
        if response is None:
            out[0, :frames] = 0.0
            return
        inputs = min(block.shape[0], 2)
        paths = _PATHS[(inputs, response.channels)]
        outputs = 2 if (response.channels > 1 or inputs > 1) else 1
        done = 0
        while done < frames:
            take = min(PARTITION - self._cursor, frames - done)
            at = self._cursor
            self._pending[:inputs, at:at + take] = block[:inputs, done:done + take]
            out[:outputs, done:done + take] = self._ready[:outputs, at:at + take]
            self._cursor += take
            done += take
            if self._cursor == PARTITION:
                self._step(response, inputs, paths, outputs)
                self._cursor = 0

    def _step(self, response: _Response, inputs: int,
              paths: tuple[tuple[int, int, int], ...], outputs: int) -> None:
        """Convolve one partition of input into the next partition of output."""
        head = (self._head + 1) % response.partitions
        self._head = head
        frame = self._frame
        for channel in range(inputs):
            frame[:PARTITION] = self._history[channel]
            frame[PARTITION:] = self._pending[channel]
            np.fft.rfft(frame, out=self._spectra[channel, head])
            self._history[channel] = self._pending[channel]
        total = self._sum[:outputs]
        total.fill(0.0)
        product = self._product
        partial = self._partial
        count = response.partitions
        for given, through, into in paths:
            history = self._spectra[given]
            spectra = response.spectra[through]
            # Partition k of the response meets the input from k partitions
            # ago, which in the ring is head - k.
            np.multiply(history[:head + 1], spectra[head::-1], out=product[:head + 1])
            if head + 1 < count:
                np.multiply(history[head + 1:], spectra[:head:-1], out=product[head + 1:])
            np.sum(product, axis=0, out=partial)
            np.add(total[into], partial, out=total[into])
        for channel in range(outputs):
            np.fft.irfft(total[channel], 2 * PARTITION, out=self._wave)
            self._ready[channel] = self._wave[PARTITION:]
