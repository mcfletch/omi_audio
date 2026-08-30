"""Ogg Opus, decoded through ``libopus``.

The one codec :mod:`omi_audio.formats` names that the ``miniaudio`` backend does
not read.  ``OMI_audio_opus`` is the better of the two codec extensions and the
one a document offering both prefers, so a build that cannot decode it falls
back to the MP3 every time.

The container is demultiplexed here, in a few lines of arithmetic over a
published specification, because no package does that part.  The codec itself is
``libopus``, reached through :mod:`ctypes` and supplied by the optional
``opuslib-next-bundled`` distribution, whose wheels carry one for Linux, macOS
and Windows -- roughly 330 KB each.  A ``libopus`` already on the system is used
where there is one and the package is absent, which is the common case on Linux;
relying on *only* that would decode there and silently not on Windows or macOS,
where it is not a component anybody has.

Everything in the chain is BSD-3-Clause: ``libopus`` from Xiph, the ``opuslib``
bindings, and the wheel that bundles them.  The library is loaded only if a
caller actually asks for an Opus file, and where none can be found this module
reports itself unavailable and the caller degrades exactly as it does with no
audio backend at all.

Two published specifications, and no other source, define what is read here:

* Ogg encapsulation --- the capture pattern, the page header and the segment
  table that laces packets across pages.  RFC 3533.
  https://www.rfc-editor.org/rfc/rfc3533.html
* Ogg Opus --- the ``OpusHead`` identification header, the pre-skip that every
  decoder must discard, and the rule that the decoded rate is always 48 kHz.
  RFC 7845.  https://www.rfc-editor.org/rfc/rfc7845.html
"""

from __future__ import annotations

import ctypes
import ctypes.util
import logging
import struct
from typing import Any

import numpy as np
from numpy.typing import NDArray

log = logging.getLogger(__name__)

#: RFC 3533 §6: every Ogg page begins with these four bytes.
OGG_MAGIC = b'OggS'
#: RFC 3533 §6: the fixed part of a page header, before the segment table.
OGG_HEADER_SIZE = 27
#: RFC 3533 §6: offset of the segment count within that header.
OGG_SEGMENTS_OFFSET = 26
#: RFC 3533 §6: a lacing value below this ends the packet; 255 means it goes on.
OGG_LACING_CONTINUES = 255

#: RFC 7845 §5.1: the identification header opens with this and is the first
#: packet of the logical bitstream.
OPUS_HEAD_MAGIC = b'OpusHead'
#: RFC 7845 §5.1: `<magic><version><channels><pre-skip><rate><gain><family>`,
#: little-endian, and 19 bytes before the optional channel mapping table.
_OPUS_HEAD = struct.Struct('<8sBBHIhB')

#: RFC 7845 §2: an Opus stream is always decoded at 48 kHz whatever rate it was
#: captured at.  The rate in the header records the original and is not the rate
#: the samples come out at.
OPUS_DECODE_RATE = 48000

#: The largest number of frames one packet can hold at 48 kHz: 120 ms.
OPUS_MAX_FRAME = 5760

#: What ``libopus`` returns on success from a decoder create.
_OPUS_OK = 0

_library: Any = None
_attempted = False


class OpusError(Exception):
    """An Ogg Opus stream could not be decoded."""


def _bind(library: Any) -> Any:
    """Declare the three entry points this uses, so ctypes marshals them right.

    Without the declarations a pointer-sized return is truncated to an ``int``
    on 64-bit builds, and the decoder handle comes back mangled.
    """
    library.opus_decoder_create.restype = ctypes.c_void_p
    library.opus_decoder_create.argtypes = [
        ctypes.c_int32, ctypes.c_int, ctypes.POINTER(ctypes.c_int)]
    library.opus_decode_float.restype = ctypes.c_int
    library.opus_decode_float.argtypes = [
        ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int32,
        ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int]
    library.opus_decoder_destroy.restype = None
    library.opus_decoder_destroy.argtypes = [ctypes.c_void_p]
    return library


def _libopus() -> Any:
    """``libopus``, or None where this installation has none.

    Two sources, in this order:

    1. ``opuslib_next`` (the ``opuslib-next-bundled`` distribution), which
       carries a ``libopus`` inside the wheel for Linux, macOS and Windows.
       This is the portable answer and is what the ``opus`` extra installs:
       ``libopus`` ships with a desktop Linux but is not a component anyone has
       on Windows or macOS, so a system-only lookup would decode on one
       platform and silently not on the others.
    2. The system ``libopus``, for a machine that has one and not the package.

    Looked up once.  A machine with neither is an ordinary machine, and the
    answer does not change while the process runs.
    """
    global _library, _attempted
    if _attempted:
        return _library
    _attempted = True
    try:
        from opuslib_next import _loader
    except ImportError:
        pass
    else:
        try:
            _library = _bind(_loader.load_libopus())
            return _library
        except Exception as error:          # noqa: BLE001 - fall through to the system
            log.debug('opuslib_next did not yield a library (%s)', error)
    name = ctypes.util.find_library('opus')
    if name:
        try:
            _library = _bind(ctypes.CDLL(name))
        except (OSError, AttributeError) as error:
            log.debug('the system libopus did not load (%s)', error)
    if _library is None:
        log.debug('no libopus is available; Opus will not decode '
                  '(install omi_audio[opus])')
    return _library


def available() -> bool:
    """Whether Opus can be decoded in this installation."""
    return _libopus() is not None


def looks_like_opus(data: bytes) -> bool:
    """Whether ``data`` opens as an Ogg stream carrying Opus.

    Both halves are needed. The capture pattern alone says only "Ogg", which
    Vorbis also is and which the ``miniaudio`` backend reads for itself; the
    identification header is what distinguishes the two, and taking it from the
    bytes rather than from a file name or a declared MIME type is what makes a
    mislabelled payload decode anyway.
    """
    if not data.startswith(OGG_MAGIC) or len(data) < OGG_HEADER_SIZE + 1:
        return False
    first = next(packets(data), b'')
    return first.startswith(OPUS_HEAD_MAGIC)


def packets(data: bytes) -> Any:
    """The packets of an Ogg bitstream, in order (RFC 3533 §6).

    A page carries a table of lacing values; a packet is the run of segments up
    to and including the first value below 255, and a packet longer than one
    page continues across the page boundary. Yielding rather than collecting
    keeps :func:`looks_like_opus` to the cost of one page.

    Stops at the first thing that is not a page rather than raising, since a
    truncated download should cost the tail of a sound and not the whole load.
    """
    offset = 0
    partial = bytearray()
    while offset + OGG_HEADER_SIZE <= len(data):
        if data[offset:offset + 4] != OGG_MAGIC:
            return
        count = data[offset + OGG_SEGMENTS_OFFSET]
        table_end = offset + OGG_HEADER_SIZE + count
        if table_end > len(data):
            return
        table = data[offset + OGG_HEADER_SIZE:table_end]
        body = table_end
        for lacing in table:
            partial += data[body:body + lacing]
            body += lacing
            if lacing < OGG_LACING_CONTINUES:
                yield bytes(partial)
                partial = bytearray()
        offset = body


def head(packet: bytes) -> tuple[int, int]:
    """``(channels, pre-skip)`` from an ``OpusHead`` packet (RFC 7845 §5.1)."""
    if len(packet) < _OPUS_HEAD.size or not packet.startswith(OPUS_HEAD_MAGIC):
        raise OpusError('the stream does not begin with an OpusHead header')
    _magic, _version, channels, pre_skip, _rate, _gain, _family = \
        _OPUS_HEAD.unpack_from(packet, 0)
    if not 1 <= channels <= 2:
        raise OpusError('an OpusHead naming %d channels is not one this reads'
                        % (channels,))
    return (int(channels), int(pre_skip))


def decode(data: bytes, name: str = '<bytes>') -> tuple[NDArray[np.float32], int, int]:
    """Decode Ogg Opus to ``(interleaved float32, channels, sample rate)``.

    The rate is always :data:`OPUS_DECODE_RATE` (RFC 7845 §2) whatever the
    identification header records as the capture rate.

    ``RFC 7845 §4.2``: the encoder puts pre-roll at the front of the stream so
    the codec has converged by the first real sample, and every decoder drops
    that many frames. Left in, a sound starts with a few milliseconds of the
    encoder warming up.

    Raises:
        OpusError: where ``libopus`` is absent, or the stream cannot be read.
    """
    library = _libopus()
    if library is None:
        raise OpusError('cannot decode %s: libopus is not installed' % (name,))
    stream = list(packets(data))
    if len(stream) < 2:
        raise OpusError('%s holds no Ogg Opus packets' % (name,))
    channels, pre_skip = head(stream[0])

    error = ctypes.c_int()
    decoder = library.opus_decoder_create(OPUS_DECODE_RATE, channels,
                                          ctypes.byref(error))
    if error.value != _OPUS_OK or not decoder:
        raise OpusError('cannot start an Opus decoder for %s (error %d)'
                        % (name, error.value))
    buffer = (ctypes.c_float * (OPUS_MAX_FRAME * channels))()
    blocks: list[NDArray[np.float32]] = []
    try:
        # The first two packets are the identification and comment headers
        # (RFC 7845 §5); everything after them is audio.
        for packet in stream[2:]:
            frames = library.opus_decode_float(
                ctypes.c_void_p(decoder), packet, len(packet), buffer,
                OPUS_MAX_FRAME, 0)
            if frames < 0:
                log.warning('%s: an Opus packet did not decode (error %d); '
                            'keeping what came before it', name, frames)
                break
            if frames:
                blocks.append(np.ctypeslib.as_array(
                    buffer)[:frames * channels].copy())
    finally:
        library.opus_decoder_destroy(ctypes.c_void_p(decoder))

    if not blocks:
        raise OpusError('%s decoded to no audio at all' % (name,))
    samples = np.concatenate(blocks)
    return (samples[pre_skip * channels:], channels, OPUS_DECODE_RATE)
