"""Decoded audio, and the one seam an encoded file passes through.

A :class:`Clip` is the only shape of audio the mixer knows: **one channel of
float32 samples at one rate**.  Everything is normalised to that on the way in,
because the alternative is a mixer that branches on sample format, channel count
and rate in its inner loop -- on the audio thread, sixty times a second.  Mono in
particular is not a simplification but a requirement: a stereo source has already
decided where it sits in the stereo field, and a sound that has decided cannot
then be panned to where it actually is in the world.

Decoding happens through ``miniaudio``, which is optional.  It is one MIT-licensed
package covering ``.wav``, ``.mp3``, ``.ogg`` (Vorbis) and ``.flac``, and it
resamples and re-channels while decoding, so the normalising above costs nothing
extra.  Where it is absent, :func:`decode_file` raises :class:`DecodeError` and
:class:`ClipCache` turns that into a warning and a silence -- a machine with no
audio backend is a normal machine.

Encoded audio arrives two ways, and both are here: :func:`decode_file` for a
path, and :func:`decode_bytes` for audio an application already holds -- a glTF
``bufferView`` out of a ``.glb``, the payload of a ``data:`` URI, or a download
that never touched a disk.

**Names are not paths.**  A name handed to :class:`ClipCache` is used exactly as
written: as a dictionary key, and -- only if it is not already known -- as the
argument to the decoder.  The cache does not normalise it, does not resolve it
against a base directory and never asks the filesystem whether it exists.  That
matters because the names in a glTF document come from a third party; see
:class:`~omi_audio.library.AudioLibrary`, which is where a document's ``uri``
becomes something an application has vouched for.

Decoding never happens on the audio thread.  :class:`ClipCache` is what makes
that practical: a sound fired sixty times a second is decoded once.
"""

from __future__ import annotations

import logging
from typing import Any
from collections.abc import Callable

import numpy as np
from numpy.typing import ArrayLike, NDArray

from omi_audio import _backend
from omi_audio._backend import DEFAULT_SAMPLE_RATE

log = logging.getLogger(__name__)

__all__ = [
    'DEFAULT_SAMPLE_RATE', 'Clip', 'ClipCache', 'DecodeError', 'Decoder',
    'decode_bytes', 'decode_file', 'decoder_available',
]


class DecodeError(Exception):
    """A file could not be turned into samples: absent, unreadable, or unknown."""


def decoder_available() -> bool:
    """Whether encoded audio can be decoded at all in this installation."""
    return _backend.available()


class Clip:
    """Mono float32 samples and the rate they were taken at.

    Immutable by convention: the mixer reads a clip from the audio thread while
    the application may still be holding it, so nothing rewrites one in place.
    """

    __slots__ = ('samples', 'sample_rate', 'name')

    #: The samples themselves: one channel, contiguous, float32.
    samples: NDArray[np.float32]
    #: Frames per second the samples were taken at.
    sample_rate: int
    #: Whatever the clip was made from, for logs and ``__repr__``.
    name: str

    def __init__(self, samples: ArrayLike, sample_rate: int = DEFAULT_SAMPLE_RATE,
                 name: str = '') -> None:
        if sample_rate <= 0:
            raise ValueError('sample_rate must be positive, not %r' % (sample_rate,))
        data = np.asarray(samples, dtype=np.float32)
        if data.ndim > 1:
            # Interleaved (frames, channels): average, so a centre-panned stereo
            # file keeps its level rather than doubling it.
            data = data.mean(axis=-1, dtype=np.float32)
        self.samples = np.ascontiguousarray(data.reshape(-1), dtype=np.float32)
        self.sample_rate = int(sample_rate)
        self.name = name

    def __repr__(self) -> str:
        return '<%s %r %.3fs @%dHz>' % (
            type(self).__name__, self.name, self.duration, self.sample_rate)

    @property
    def frames(self) -> int:
        """How many samples the clip holds."""
        return int(self.samples.shape[0])

    @property
    def duration(self) -> float:
        """How long the clip lasts, in seconds, at its own rate."""
        return self.frames / float(self.sample_rate)

    @property
    def peak(self) -> float:
        """The largest absolute sample value; 0.0 for silence."""
        return float(np.abs(self.samples).max()) if self.frames else 0.0

    def normalised(self) -> Clip:
        """A copy scaled so its peak is 1.0; silence is returned unchanged."""
        peak = self.peak
        if peak == 0.0:
            return self
        return Clip(self.samples / peak, self.sample_rate, self.name)

    def resampled(self, sample_rate: int) -> Clip:
        """A copy at ``sample_rate``, linearly interpolated.

        Used where a clip arrives at a rate the engine does not mix at -- a
        synthesised clip, or one decoded before the engine's rate was known.
        Encoded audio goes through :func:`decode_file` or :func:`decode_bytes`,
        which resample while decoding and do it better.
        """
        if sample_rate == self.sample_rate or not self.frames:
            return self
        count = int(round(self.frames * sample_rate / self.sample_rate))
        position = np.arange(count, dtype=np.float64) * (self.sample_rate / sample_rate)
        return Clip(np.interp(position, np.arange(self.frames), self.samples),
                    sample_rate, self.name)


def _decoded(samples: Any, sample_rate: int, name: str) -> Clip:
    """One ``miniaudio`` decode result as a :class:`Clip`."""
    return Clip(np.frombuffer(memoryview(samples), dtype=np.float32), sample_rate,
                name=name)


def _require_backend(what: str) -> Any:
    """The backend, or the one exception decoding is allowed to raise."""
    backend_module = _backend.backend()
    if backend_module is None:
        raise DecodeError(
            'cannot decode %s: miniaudio is not installed '
            '(install omi_audio[playback])' % (what,))
    return backend_module


def decode_file(path: str, sample_rate: int = DEFAULT_SAMPLE_RATE) -> Clip:
    """Decode the file at ``path`` to a mono :class:`Clip` at ``sample_rate``.

    The backend resamples and mixes to one channel as part of decoding, so this
    is one pass over the file rather than a decode followed by two conversions.

    ``path`` is opened as given.  It is the caller's job to have decided that
    the path is one this application is willing to read -- this function is the
    bottom of the stack and has no idea where the name came from.

    Raises:
        DecodeError: where the backend is absent, the file is missing, or its
            contents are not audio this build can read.
    """
    module = _require_backend(repr(path))
    try:
        decoded = module.decode_file(
            path, output_format=module.SampleFormat.FLOAT32,
            nchannels=1, sample_rate=sample_rate)
    except Exception as error:
        raise DecodeError('cannot decode %r: %s' % (path, error)) from error
    return _decoded(decoded.samples, sample_rate, path)


def decode_bytes(data: bytes, sample_rate: int = DEFAULT_SAMPLE_RATE,
                 name: str = '<bytes>') -> Clip:
    """Decode encoded audio an application already holds, with no file involved.

    This is what a ``.glb`` needs.  glTF's dominant shipping format embeds its
    audio in a ``bufferView`` rather than beside the document, and a ``data:``
    URI carries it inline; in both cases the consumer has the bytes and there is
    nothing to open.  ``name`` is used only for logs and :attr:`Clip.name`.

    The format is detected from the bytes themselves, so a ``mimeType`` that
    disagrees with the payload does not matter -- and content in a format this
    build cannot read fails the same way an unreadable file does.

    Raises:
        DecodeError: where the backend is absent, or the bytes are not audio
            this build can read.
    """
    module = _require_backend(name)
    try:
        decoded = module.decode(
            data, output_format=module.SampleFormat.FLOAT32,
            nchannels=1, sample_rate=sample_rate)
    except Exception as error:
        raise DecodeError('cannot decode %s: %s' % (name, error)) from error
    return _decoded(decoded.samples, sample_rate, name)


#: What a cache calls to turn a name into samples.  Swappable so a test, or an
#: application with its own resolver, can supply clips without touching a disk.
Decoder = Callable[[str, int], Clip]


class ClipCache:
    """Decoded clips, keyed by name, decoded at most once.

    A weapon fired sixty times a second names the same file sixty times; this is
    what keeps that one decode.  A name that fails to decode is remembered as a
    failure, so a missing file warns once rather than once per shot.

    **A name is a key, not a path.**  It is passed to the decoder exactly as
    given and is never normalised, resolved or tested against the filesystem, so
    two spellings of one file are two entries; an application that minds calls
    :func:`os.path.realpath` before handing the name over.  The cache resolving
    names itself would mean it was interpreting strings that, for a glTF
    document, came from somewhere nobody here controls.

    Not thread-safe by design: it is used from the control thread only, which is
    the same rule that keeps decoding off the audio thread.
    """

    def __init__(self, sample_rate: int = DEFAULT_SAMPLE_RATE,
                 decode: Decoder | None = None) -> None:
        self.sample_rate = int(sample_rate)
        self.decode = decode if decode is not None else decode_file
        self._clips: dict[str, Clip] = {}
        self._failed: set[str] = set()

    def __len__(self) -> int:
        return len(self._clips)

    def __contains__(self, name: object) -> bool:
        return name in self._clips

    @property
    def frames_held(self) -> int:
        """Total samples held, for anyone budgeting memory."""
        return sum(clip.frames for clip in self._clips.values())

    def put(self, name: str, clip: Clip) -> Clip:
        """Register ``clip`` under ``name``, resampling it if it needs it."""
        if clip.sample_rate != self.sample_rate:
            clip = clip.resampled(self.sample_rate)
        self._clips[name] = clip
        self._failed.discard(name)
        return clip

    def put_bytes(self, name: str, data: bytes) -> Clip | None:
        """Decode ``data`` and register it under ``name``.

        The counterpart to :meth:`put` for audio that arrives encoded --
        a ``bufferView``, a ``data:`` URI, a download.  Returns None and warns
        once where the bytes will not decode, so a caller may treat a bad asset
        the way it treats a missing one.
        """
        try:
            clip = decode_bytes(data, self.sample_rate, name=name)
        except DecodeError as error:
            self._failed.add(name)
            log.warning('no sound for %s', error)
            return None
        return self.put(name, clip)

    def get(self, name: str) -> Clip | None:
        """The clip for ``name``, decoding it if this is the first ask.

        Returns None where the name cannot be decoded.  A sound that will not
        load is a silence and a warning, never an exception: content is often
        incomplete, and a missing footstep must not stop a scene from running.
        """
        clip = self._clips.get(name)
        if clip is not None:
            return clip
        if name in self._failed:
            return None
        try:
            clip = self.decode(name, self.sample_rate)
        except DecodeError as error:
            self._failed.add(name)
            log.warning('no sound for %s', error)
            return None
        return self.put(name, clip)

    def clear(self) -> None:
        """Drop every decoded clip and every remembered failure."""
        self._clips.clear()
        self._failed.clear()
