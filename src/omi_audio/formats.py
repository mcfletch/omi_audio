"""The glTF audio codec extensions, and which of them this build can play.

``KHR_audio_emitter`` guarantees exactly one encoding: MP3.  A document that
wants a better one says so through a companion extension on the *source* --
``OMI_audio_ogg_vorbis`` or ``OMI_audio_opus`` -- each of which names a second
entry in the same ``audio`` array holding the same sound in that encoding.  The
source's own ``audio`` stays as the fallback, so one document plays everywhere
and sounds better wherever the codec is available.  A document with no fallback
puts its codec extension in ``extensionsRequired`` instead, which is a statement
to the *consumer* about whether the file is worth opening at all.

Choosing between them is two separate questions and this module answers only
the first: **which encodings can be decoded here**, from
:func:`decodable`.  Which of a particular source's entries that makes preferable
is :meth:`~omi_audio.model.AudioSource.audio_indices`, and actually going and
getting one is :class:`~omi_audio.library.AudioLibrary`.

:func:`decodable` asks the backend what it reads rather than asserting a list,
so a build of ``miniaudio`` without Vorbis reports honestly and a future one
that gains Opus is used without anything here changing.  Today that means
**Ogg Vorbis plays and Opus does not**: Opus is read, preserved through a
round trip and offered to an application with its own decoder, but nothing in
the default chain will turn it into samples, so a document offering Opus with an
MP3 fallback plays the MP3.

References:
    ``OMI_audio_ogg_vorbis``
    https://github.com/omigroup/gltf-extensions/tree/main/extensions/2.0/OMI_audio_ogg_vorbis

    ``OMI_audio_opus``
    https://github.com/omigroup/gltf-extensions/tree/main/extensions/2.0/OMI_audio_opus
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from omi_audio import _backend

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Encoding:
    """One codec a source may offer beyond the MP3 the base extension requires.

    ``mime_types`` and ``suffixes`` are what the specification permits for audio
    carried in a ``bufferView`` and beside the document respectively.  Nothing
    here decides a format from them -- :mod:`omi_audio.clip` detects the format
    from the bytes, so a ``mimeType`` disagreeing with its payload cannot cost a
    sound -- but an exporter needs them, and so does anything displaying what a
    document contains.

    ``backend_format`` is the name the decoding backend gives this codec, and is
    how :func:`decodable` asks whether it is available.
    """

    extension: str
    mime_types: tuple[str, ...]
    suffixes: tuple[str, ...]
    backend_format: str


#: Opus in an Ogg or a WebM container.  Video in a WebM is ignored.
OPUS = Encoding(
    extension='OMI_audio_opus',
    mime_types=('audio/opus', 'audio/webm', 'video/webm'),
    suffixes=('.opus', '.webm'),
    backend_format='OPUS')

#: Vorbis in an Ogg container.  The specification asks for ``.ogg``, not ``.oga``.
VORBIS = Encoding(
    extension='OMI_audio_ogg_vorbis',
    mime_types=('audio/ogg',),
    suffixes=('.ogg',),
    backend_format='VORBIS')

#: Every codec extension, best first.  Opus leads because it is the more
#: efficient of the two at a given quality, which is the reason a document
#: offering both offers both.
ENCODINGS: tuple[Encoding, ...] = (OPUS, VORBIS)


def by_extension(name: str) -> Encoding | None:
    """The :class:`Encoding` an extension name identifies, or None."""
    for encoding in ENCODINGS:
        if encoding.extension == name:
            return encoding
    return None


def decodable() -> tuple[Encoding, ...]:
    """The encodings this installation can actually turn into samples, best first.

    Answered by asking the backend which formats it reads, so it stays true of
    the ``miniaudio`` that is installed rather than of the one that was current
    when this was written.  Empty where no backend is installed at all, which is
    the same answer as "nothing decodes anything" -- correct, since without it
    the MP3 fallback does not play either.
    """
    module = _backend.backend()
    if module is None:
        return ()
    known = {member.name for member in module.FileFormat}
    return tuple(encoding for encoding in ENCODINGS
                 if encoding.backend_format in known)


def read(source: Any) -> dict[str, int]:
    """The audio index each codec extension on one ``sources`` entry names.

    Keyed by extension name rather than by :class:`Encoding` so that the mapping
    is JSON, which is what makes writing it back the identity.

    Never raises.  A malformed extension block costs the better encoding and
    leaves the fallback, which is exactly what the extension is designed to
    degrade to.
    """
    found: dict[str, int] = {}
    if not isinstance(source, dict):
        return found
    extensions = source.get('extensions')
    if not isinstance(extensions, dict):
        return found
    for encoding in ENCODINGS:
        if encoding.extension not in extensions:
            continue
        entry = extensions[encoding.extension]
        index = entry.get('audio') if isinstance(entry, dict) else None
        if isinstance(index, bool) or not isinstance(index, int):
            log.warning('ignoring %s on a source: expected an audio index, '
                        'not %r', encoding.extension, type(index).__name__)
            continue
        found[encoding.extension] = index
    return found


def write(encodings: dict[str, int]) -> dict[str, Any]:
    """The ``extensions`` object a ``sources`` entry needs to offer ``encodings``.

    The inverse of :func:`read`.  Shares no state with its argument, so an
    exporter renumbering indices in the block it was given does not rewrite the
    document it exported from.
    """
    return {name: {'audio': index} for name, index in encodings.items()}
