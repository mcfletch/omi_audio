"""Helpers every test module would otherwise re-roll.

Plain functions and constants live here rather than in ``conftest.py`` because
they are not fixtures: a test that wants a ramp wants to *call* something, and
routing that through pytest's injection would make the test harder to read, not
easier.  ``conftest.py`` holds the fixtures, and only the fixtures.
"""

import math
import wave

import numpy as np
import pytest

from omi_audio import clip as clipmodule
from omi_audio import synth
from omi_audio.clip import Clip

#: Sample rate the whole suite mixes at.  Low, so a block is quick to make and
#: a cycle of the test tone spans few samples.
RATE = 8000

#: Skipped rather than failed where the optional backend is not installed --
#: everything except decoding and real playback is exercised without it.
needs_miniaudio = pytest.mark.skipif(
    not clipmodule.decoder_available(),
    reason='miniaudio is not installed; the backend seam cannot be exercised')

#: Step between successive samples of :func:`ramp`.  Small enough that a whole
#: ramp stays inside full scale, so the mixer's output clipping never hides the
#: cursor a test is watching.
STEP = 0.1


def beep(seconds=1.0, frequency=440.0, amplitude=0.5, fade=0.0):
    """A plain tone at :data:`RATE`, with no fade to complicate a level."""
    return synth.tone(frequency, seconds, sample_rate=RATE, amplitude=amplitude,
                      fade=fade)


def constant(value=1.0, frames=1000, sample_rate=RATE):
    """A clip that is the same sample all the way through."""
    return Clip(np.full(frames, value, dtype='f'), sample_rate)


def ramp(frames=8, sample_rate=RATE):
    """A clip whose samples climb by :data:`STEP`, so a cursor is readable."""
    return Clip(np.arange(frames, dtype='f') * STEP, sample_rate)


def write_wav(path, samples, sample_rate=RATE, channels=1):
    """A real ``.wav`` file, written with the standard library only."""
    data = np.clip(np.asarray(samples, dtype='f'), -1.0, 1.0)
    pcm = (data * 32767.0).astype('<i2')
    with wave.open(str(path), 'wb') as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm.tobytes())
    return path


def wav_bytes(samples, sample_rate=RATE, channels=1):
    """The same ``.wav``, as bytes, for the paths that never touch a disk."""
    import io

    buffer = io.BytesIO()
    data = np.clip(np.asarray(samples, dtype='f'), -1.0, 1.0)
    pcm = (data * 32767.0).astype('<i2')
    with wave.open(buffer, 'wb') as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm.tobytes())
    return buffer.getvalue()


def dbfs(samples):
    """Peak level of a block, in dBFS."""
    peak = float(np.abs(np.asarray(samples)).max()) if len(samples) else 0.0
    return 20.0 * math.log10(max(peak, 1e-9))


def device_bytes(engine, frames=512, blocks=8):
    """The PCM the *device* receives, decoded back to floats.

    Driven through :meth:`~omi_audio.mixer.Mixer.blocks` with ``next()`` then
    ``send()``, and converted with the same rules the backend applies, so a
    fault anywhere in the hand-off -- the generator protocol, the memoryview,
    the buffer being recycled -- shows up here as silence.
    """
    stream = engine.mixer.blocks()
    next(stream)
    collected = []
    for _ in range(blocks):
        block = stream.send(frames)
        raw = memoryview(block).cast('B') if memoryview(block).itemsize != 1 else block
        collected.append(np.frombuffer(bytes(raw), dtype=np.float32))
    return np.concatenate(collected)
