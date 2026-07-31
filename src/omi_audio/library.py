"""The audio a document refers to, and who is allowed to go and get it.

A glTF document does not contain its sounds; it *refers* to them, by a ``uri``
beside the document or a ``bufferView`` inside it.  Somebody has to turn those
references into bytes, and this module exists to make sure that somebody is
**never this library**.

Why not.  A ``uri`` is a string in a file from a third party, and the moment a
library resolves one it has agreed to interpret ``../../../../etc/passwd``,
``file:///``, ``\\\\host\\share\\x.wav`` and ``http://somewhere/x.mp3`` on the
application's behalf, using a policy the application never got to see.  The
application, on the other hand, already knows where its content lives, already
has a resolver and a download cache -- OpenGLContext's ``loader.resolver``, for
one -- and is the only party that can say what this document is allowed to
reach.  So the URI never leaves :class:`~omi_audio.model.Audio`, and what
crosses into the engine is audio the application has already vouched for.

How it works.  An :class:`AudioLibrary` holds, for one
:class:`~omi_audio.model.AudioDocument`, the clip each entry of its ``audio``
array has resolved to.  The first time an index is asked for, the library calls
the ``fetch`` callback it was given and hands the request back to the
application.  What happens next is the application's business:

**Synchronously** -- everything is on disk already, so resolve and supply
without returning::

    def fetch(library, index, audio):
        path = resolver.resolve(audio.uri)          # the app's own policy
        if path is None:
            library.fail(index, 'not in the content directory')
        else:
            library.supply_file(index, path)

    library = AudioLibrary(document, cache=engine.clips, fetch=fetch)
    engine.play_source(source, library)             # plays this frame

**Asynchronously** -- start a download and return.  :meth:`clip` gives None
until it lands, which is the ordinary "not audible yet" answer the whole engine
is built to tolerate, and the sound starts on whichever later frame calls
again::

    def fetch(library, index, audio):
        downloads.start(audio.uri, on_done=lambda data: library.supply_bytes(index, data))

**From the file itself** -- a ``.glb`` carries its audio in a buffer view, and a
``data:`` URI carries it inline.  Both are bytes the loader already holds::

    def fetch(library, index, audio):
        if audio.bufferView is not None:
            library.supply_bytes(index, gltf.buffer_view_bytes(audio.bufferView))

The library is control-thread state, like the :class:`~omi_audio.clip.ClipCache`
it decodes through, and is not thread-safe.  An application downloading on
another thread hands the bytes back to the thread that drives the engine, which
it has to do anyway.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

from omi_audio import model
from omi_audio.clip import Clip, ClipCache, DecodeError, decode_bytes

log = logging.getLogger(__name__)

#: Asked to produce the audio at ``index``.  Called at most once per index, on
#: the first ask.  Supplies through :meth:`AudioLibrary.supply`,
#: :meth:`~AudioLibrary.supply_file` or :meth:`~AudioLibrary.supply_bytes`,
#: either before it returns or whenever its download finishes; or gives up
#: through :meth:`~AudioLibrary.fail`.
Fetch = Callable[['AudioLibrary', int, model.Audio], None]


class AudioLibrary:
    """What one document's ``audio`` array has resolved to, so far.

    Every entry is in one of four states, and they only ever move forwards:

    ==========  =============================================================
    unknown     Nobody has asked for it.  ``fetch`` has not been called.
    requested   ``fetch`` has been called and has not supplied anything yet.
    ready       A clip is here.  :meth:`clip` returns it from now on.
    failed      It will not resolve.  :meth:`clip` returns None and nothing is
                asked again -- a missing sound must warn once, not once a frame.
    ==========  =============================================================
    """

    def __init__(self, document: model.AudioDocument,
                 cache: ClipCache | None = None,
                 fetch: Fetch | None = None) -> None:
        #: The document this library resolves for.
        self.document = document
        #: Where clips supplied as *files* are decoded and shared.  Give it the
        #: engine's own cache so two documents naming one file decode once.
        self.cache = cache if cache is not None else ClipCache()
        self.fetch = fetch
        self._clips: dict[int, Clip] = {}
        self._failed: dict[int, str] = {}
        self._requested: set[int] = set()

    def __repr__(self) -> str:
        return '<%s %d/%d ready, %d pending, %d failed>' % (
            type(self).__name__, len(self._clips), len(self.document.audio),
            len(self.pending), len(self._failed))

    def __len__(self) -> int:
        return len(self._clips)

    # ------------------------------------------------------------------
    # Asking
    # ------------------------------------------------------------------

    @property
    def pending(self) -> tuple[int, ...]:
        """Indices that have been asked for and have not arrived.

        An application with a loading screen waits on this; one without simply
        starts fewer sounds this frame and more the next.
        """
        return tuple(sorted(self._requested - set(self._clips) - set(self._failed)))

    def ready(self, index: int) -> bool:
        """Whether the audio at ``index`` is decoded and playable now."""
        return index in self._clips

    def clip(self, index: int | None) -> Clip | None:
        """The clip for ``audio[index]``, asking for it if this is the first time.

        Returns None where the index names nothing, where the audio has failed
        to resolve, or where it simply has not arrived yet.  A caller cannot
        tell those apart and does not need to: all three mean "no sound this
        frame", which is what every path in this engine already does.
        """
        if index is None or not (0 <= index < len(self.document.audio)):
            return None
        found = self._clips.get(index)
        if found is not None:
            return found
        if index in self._failed or index in self._requested:
            return None
        self._requested.add(index)
        if self.fetch is None:
            self.fail(index, 'the library has no fetch callback, so nothing can '
                             'resolve %r' % (self.document.audio[index].uri,))
            return None
        self.fetch(self, index, self.document.audio[index])
        return self._clips.get(index)

    def clip_for(self, source: model.AudioSource) -> Clip | None:
        """The clip a :class:`~omi_audio.model.AudioSource` plays, if any."""
        return self.clip(source.audio)

    # ------------------------------------------------------------------
    # Supplying
    # ------------------------------------------------------------------

    def supply(self, index: int, clip: Clip) -> Clip:
        """Deliver an already-decoded clip for ``index``.

        Resampled to the cache's rate if it needs it, so a caller may supply a
        clip made at whatever rate it had to hand.
        """
        if clip.sample_rate != self.cache.sample_rate:
            clip = clip.resampled(self.cache.sample_rate)
        self._clips[index] = clip
        self._failed.pop(index, None)
        return clip

    def supply_file(self, index: int, path: str) -> Clip | None:
        """Deliver ``index`` as a local file the application vouches for.

        Decoded through the shared :class:`~omi_audio.clip.ClipCache`, so the
        same file behind two documents costs one decode.  ``path`` is opened as
        given: this library has no opinion about it, because the application has
        already had the only opinion that matters.

        Returns None, and marks the index failed, where the file will not
        decode.
        """
        clip = self.cache.get(path)
        if clip is None:
            self._give_up(index, 'file %r would not decode' % (path,))
            return None
        return self.supply(index, clip)

    def supply_bytes(self, index: int, data: bytes) -> Clip | None:
        """Deliver ``index`` as encoded audio with no file behind it.

        This is the ``.glb`` path, the ``data:`` URI path and the finished
        download.  The bytes are decoded here rather than through the cache,
        because they belong to this document and there is no name to share them
        under.

        Returns None, and marks the index failed, where the bytes will not
        decode.
        """
        name = self._name(index)
        try:
            clip = decode_bytes(data, self.cache.sample_rate, name=name)
        except DecodeError as error:
            self._give_up(index, str(error))
            return None
        return self.supply(index, clip)

    def fail(self, index: int, reason: str) -> None:
        """Record that ``index`` will not resolve, so nothing asks again.

        An application calls this when its own resolver refuses -- the URI is
        outside the content directory, the download 404'd, the scheme is one it
        does not serve.  The reason is logged once.
        """
        self._give_up(index, reason)

    def _give_up(self, index: int, reason: str) -> None:
        """Mark ``index`` unresolvable, logging the reason the first time only."""
        if index not in self._failed:
            log.warning('no sound for %s: %s', self._name(index), reason)
        self._failed[index] = reason

    def _name(self, index: int) -> str:
        """Something to put in a log line, from whatever the document offers."""
        if 0 <= index < len(self.document.audio):
            entry = self.document.audio[index]
            for label in (entry.name, entry.uri):
                if label:
                    return '%s (audio %d)' % (label, index)
        return 'audio %d' % (index,)
