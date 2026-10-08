"""An analyser: the signal passed through unchanged, its recent past kept to look at.

The audio thread keeps the last :data:`MAX_FFT_SIZE` samples of the input,
mixed to mono and multiplied by the node's gain; the output is the input,
untouched.  The control thread asks for what the Web Audio API's
``AnalyserNode`` offers, worked out from the latest ``fft_size`` of them:

* :meth:`Analyser.time_domain_data`, the samples themselves;
* :meth:`Analyser.frequency_data`, the magnitude of each of ``fft_size / 2``
  frequency bins in decibels: the samples are weighted by a Blackman window,
  transformed, divided by ``fft_size``, and smoothed against the previous
  call's magnitudes, ``smoothed = s * previous + (1 - s) * now`` for the
  ``smoothing`` ``s``;
* their byte forms, the decibels scaled from ``min_decibels`` .. ``max_decibels``
  onto 0 .. 255, and the samples from -1 .. 1 onto 0 .. 255.

Each call to :meth:`Analyser.frequency_data` is one smoothing step, so call it
once a frame.  A bin with no energy reads :data:`SILENT_DB`.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from numpy.typing import NDArray

from omi_audio._numbers import finite
from omi_audio.graph.core import Graph, Node

__all__ = ['MAX_FFT_SIZE', 'MIN_FFT_SIZE', 'SILENT_DB', 'Analyser']

#: The longest transform, and so how much of the past is kept.
MAX_FFT_SIZE = 32768

#: The shortest transform.
MIN_FFT_SIZE = 32

#: The level a bin with no energy at all reads, in decibels.
SILENT_DB = -600.0


class Analyser(Node):
    """Its input unchanged, with the recent past kept for :meth:`frequency_data`."""

    def __init__(self, fft_size: int = 2048, min_decibels: float = -100.0,
                 max_decibels: float = -30.0, smoothing: float = 0.8,
                 gain: float = 1.0, **named: Any) -> None:
        super().__init__(gain=gain, **named)
        self.fft_size = fft_size
        self.min_decibels = finite(min_decibels)
        self.max_decibels = finite(max_decibels)
        self.smoothing = finite(smoothing)
        self._history = np.zeros(MAX_FFT_SIZE, dtype=np.float32)
        self._written = 0
        self._mono = np.zeros((1, 0), dtype=np.float32)
        self._smoothed = np.zeros(0)

    @property
    def fft_size(self) -> int:
        """Samples per transform: a power of two from 32 to 32768."""
        return self._fft_size

    @fft_size.setter
    def fft_size(self, value: int) -> None:
        wanted = max(MIN_FFT_SIZE, min(MAX_FFT_SIZE, int(value)))
        self._fft_size = 1 << (wanted - 1).bit_length()
        self._smoothed = np.zeros(self._fft_size // 2)

    @property
    def frequency_bin_count(self) -> int:
        """How many frequency bins :meth:`frequency_data` returns."""
        return self._fft_size // 2

    def prepare(self, graph: Graph) -> None:
        self._mono = np.zeros((1, graph.max_block), dtype=np.float32)

    def _apply_gain(self, out: NDArray[np.float32], frames: int, graph: Graph) -> None:
        """The gain is the analysis's alone; the output is not touched."""

    def process(self, block: NDArray[np.float32], out: NDArray[np.float32],
                frames: int) -> None:
        np.copyto(out[:block.shape[0], :frames], block)
        graph = self._graph
        if graph is None:  # pragma: no cover - rendering means a graph
            return
        mono = self._mono[:, :frames]
        mono.fill(0.0)
        graph.mix_into(block, mono, 'speakers')
        np.multiply(mono, self.gain, out=mono)
        samples = mono[0, -min(frames, MAX_FFT_SIZE):]
        count = samples.shape[0]
        start = self._written % MAX_FFT_SIZE
        first = min(count, MAX_FFT_SIZE - start)
        self._history[start:start + first] = samples[:first]
        self._history[:count - first] = samples[first:]
        self._written = start + count

    # ------------------------------------------------------------------
    # Control thread
    # ------------------------------------------------------------------

    def time_domain_data(self) -> NDArray[np.float32]:
        """The latest :attr:`fft_size` samples, oldest first."""
        size = self._fft_size
        end = self._written % MAX_FFT_SIZE
        return np.roll(self._history, -end)[-size:].copy()

    def frequency_data(self) -> NDArray[np.float64]:
        """Each bin's smoothed magnitude in decibels; one smoothing step."""
        size = self._fft_size
        samples = self.time_domain_data().astype('d')
        steps = np.arange(size) / size
        window = (0.42 - 0.5 * np.cos(2.0 * math.pi * steps)
                  + 0.08 * np.cos(4.0 * math.pi * steps))
        magnitude = np.abs(np.fft.rfft(samples * window))[:size // 2] / size
        keep = max(0.0, min(1.0, self.smoothing))
        if self._smoothed.shape[0] != magnitude.shape[0]:
            self._smoothed = np.zeros(magnitude.shape[0])
        self._smoothed = keep * self._smoothed + (1.0 - keep) * np.nan_to_num(magnitude)
        return 20.0 * np.log10(np.maximum(self._smoothed, 10.0 ** (SILENT_DB / 20.0)))

    def byte_frequency_data(self) -> NDArray[np.uint8]:
        """:meth:`frequency_data` from :attr:`min_decibels` .. :attr:`max_decibels` as 0 .. 255."""
        low, high = sorted((self.min_decibels, self.max_decibels))
        span = (high - low) or 1.0
        scaled = np.floor(255.0 / span * (self.frequency_data() - low))
        return np.clip(scaled, 0, 255).astype(np.uint8)

    def byte_time_domain_data(self) -> NDArray[np.uint8]:
        """:meth:`time_domain_data` from -1 .. 1 as 0 .. 255."""
        scaled = np.floor(128.0 * (1.0 + self.time_domain_data().astype('d')))
        return np.clip(scaled, 0, 255).astype(np.uint8)
