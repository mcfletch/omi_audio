"""The ``KHR_audio_emitter`` data model, natively.

Plain dataclasses mirroring the glTF audio extension field-for-field, with the
extension's own defaults and its own spelling.  :class:`~omi_audio.engine.AudioEngine`
and every consumer above it speak this structure, so glTF import and export are
a near-identity mapping and there is no private format to keep in sync -- the
same decision :mod:`omi_physics.model` makes for physics.

Three arrays, referenced by index, and the indirection is worth understanding
because it is what lets one file's sounds be shared:

``audio``
    Where the encoded bytes are -- a ``uri`` beside the document, or a
    ``bufferView`` inside it.
``sources``
    One piece of audio *plus how to play it*: gain, playback rate, whether it
    loops, whether it starts by itself.  Several emitters may use one source.
``emitters``
    Where the sound comes from: ``global`` for music and ambience that ignores
    the listener, ``positional`` for a sound in the world, which carries the
    distance curve and cone in its :class:`PositionalProperties`.

A fourth link closes the chain and does not live in those arrays: **scenes and
nodes name emitters**, and that is what says *where* a positional emitter is.
:func:`emitter_indices`, :meth:`AudioDocument.emitters_for_node` and
:meth:`AudioDocument.emitters_for_scene` read it.

Nothing here touches a filesystem or a network.  An ``Audio`` record's ``uri``
is a string out of an untrusted document and is never resolved, opened or
interpreted by this package; turning one into playable bytes is the consuming
application's job, through :class:`~omi_audio.library.AudioLibrary`.

References:
    ``KHR_audio_emitter``
    https://github.com/omigroup/gltf-extensions/tree/main/extensions/2.0/KHR_audio_emitter
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, fields
from functools import cache
from types import UnionType
from typing import Any, Union, get_args, get_origin, get_type_hints
from collections.abc import Callable

from omi_audio import spatial

log = logging.getLogger(__name__)

#: The extension's name, as it appears in ``extensionsUsed`` and in the
#: ``extensions`` object of the document, its scenes and its nodes.
EXTENSION = 'KHR_audio_emitter'

#: An emitter fixed to the listener: background music, ambience, narration.
GLOBAL = 'global'
#: An emitter at a place in the scene, attenuated by distance and direction.
POSITIONAL = 'positional'

#: The one MIME type the base extension requires.  ``OMI_audio_ogg_vorbis`` and
#: ``OMI_audio_opus`` add others; what this engine can actually decode is a
#: separate question, answered in :mod:`omi_audio.clip`.
MIME_MPEG = 'audio/mpeg'


@dataclass
class Audio:
    """One piece of encoded audio data, by ``uri`` or by ``bufferView``.

    ``mimeType`` is required alongside a ``bufferView``, since there is no file
    name to infer the format from.

    Neither field is resolved by this package.  ``uri`` in particular is
    **untrusted text from a third-party document**: it may be relative,
    absolute, percent-encoded, a ``data:`` URI, an ``http:`` URL or a deliberate
    attempt to escape a content directory.  Deciding what it means is the
    application's business, because only the application knows where its content
    lives and what it is willing to fetch.  See
    :class:`~omi_audio.library.AudioLibrary`.
    """

    uri: str = ''
    bufferView: int | None = None
    mimeType: str = ''
    name: str = ''


@dataclass
class AudioSource:
    """A piece of audio data and the playback settings applied to it.

    ``gain`` is a linear multiplier, not decibels: 0.5 is half amplitude.
    ``playbackRate`` changes speed and pitch together, as speeding up a record
    does -- it is a resampling ratio, not a pitch shift.

    ``autoplay`` means "start this when the document is loaded".  Nothing starts
    by itself here: see :meth:`AudioDocument.autoplay` and
    :meth:`~omi_audio.engine.AudioEngine.start_autoplay`, which is what an
    application calls when its scene begins.
    """

    audio: int | None = None
    gain: float = 1.0
    playbackRate: float = 1.0
    loop: bool = False
    autoplay: bool = False
    name: str = ''


@dataclass
class PositionalProperties:
    """How a positional emitter's loudness depends on where the listener is.

    Two independent curves, multiplied: :func:`~.spatial.distance_gain` over
    ``distanceModel``/``refDistance``/``maxDistance``/``rolloffFactor``, and
    :func:`~.spatial.cone_gain` over the three cone fields.  The cone applies
    only when ``shapeType`` is ``cone``; the defaults describe a full sphere in
    any case, so an emitter that sets none of them is unattenuated by direction.
    """

    shapeType: str = spatial.ShapeType.OMNIDIRECTIONAL.value
    coneInnerAngle: float = spatial.TAU
    coneOuterAngle: float = spatial.TAU
    coneOuterGain: float = 0.0
    distanceModel: str = spatial.DistanceModel.INVERSE.value
    maxDistance: float = 0.0
    refDistance: float = 1.0
    rolloffFactor: float = 1.0

    def gain(self, distance: float, angle: float = 0.0) -> float:
        """Combined distance and cone attenuation, as a linear multiplier.

        ``distance`` is listener-to-emitter; ``angle`` is how far off the
        emitter's forward axis the listener lies, in radians.
        """
        gain = spatial.distance_gain(
            distance, self.distanceModel, ref_distance=self.refDistance,
            max_distance=self.maxDistance, rolloff_factor=self.rolloffFactor)
        if self.shapeType == spatial.ShapeType.CONE.value:
            gain *= spatial.cone_gain(angle, self.coneInnerAngle,
                                      self.coneOuterAngle, self.coneOuterGain)
        return gain

    def in_range(self, distance: float) -> bool:
        """Whether ``maxDistance`` says this emitter is worth hearing at all.

        The extension's *prose* calls ``maxDistance`` the distance "beyond which
        the audio cannot be heard", while its *formulas* use it only in the
        linear model -- so under ``inverse`` or ``exponential`` an emitter with a
        20 m cap is still faintly audible a kilometre away.  :func:`.distance_gain`
        implements the formulas, because those are what Web Audio does and what
        authoring tools export against.

        This is the other reading, kept separate and explicit: an application
        that wants a cap to mean a cap tests it here and does not start the
        sound.  It is also the cull an application wants anyway, since an
        inaudible emitter still costs a voice.

        ``maxDistance`` of zero is the extension's default and means unbounded,
        so everything is in range.
        """
        return self.maxDistance <= 0.0 or distance <= self.maxDistance


@dataclass
class AudioEmitter:
    """Where sound comes from: a place in the scene, or everywhere at once.

    A ``positional`` emitter always has :class:`PositionalProperties`, defaulted
    if the document left them out, so nothing downstream tests for their
    absence.  A ``global`` emitter has none, because the extension forbids them.
    """

    type: str = POSITIONAL
    gain: float = 1.0
    sources: list[int] = field(default_factory=list)
    positional: PositionalProperties | None = None
    name: str = ''

    def __post_init__(self) -> None:
        if self.positional_audio and self.positional is None:
            self.positional = PositionalProperties()
        elif not self.positional_audio and self.positional is not None:
            log.info('emitter %r is global, so its positional properties are '
                     'discarded; the extension does not allow them', self.name)
            self.positional = None

    @property
    def positional_audio(self) -> bool:
        """Whether this emitter is placed in the scene rather than global."""
        return self.type == POSITIONAL


@dataclass
class AudioDocument:
    """The three arrays a document's ``KHR_audio_emitter`` block holds.

    Indices are resolved through :meth:`sources_for` and :meth:`audio_for`
    rather than by indexing directly, because content is not always well formed
    and an index that points at nothing should cost a sound, not a traceback.
    """

    audio: list[Audio] = field(default_factory=list)
    sources: list[AudioSource] = field(default_factory=list)
    emitters: list[AudioEmitter] = field(default_factory=list)

    def sources_for(self, emitter: AudioEmitter) -> list[AudioSource]:
        """The sources ``emitter`` plays, skipping any index out of range."""
        return [self.sources[index] for index in emitter.sources
                if 0 <= index < len(self.sources)]

    def audio_for(self, source: AudioSource) -> Audio | None:
        """The audio data ``source`` names, or None if it names none."""
        if source.audio is None or not (0 <= source.audio < len(self.audio)):
            return None
        return self.audio[source.audio]

    def emitters_for(self, indices: list[int]) -> list[AudioEmitter]:
        """The emitters at ``indices``, skipping any that point at nothing."""
        return [self.emitters[index] for index in indices
                if 0 <= index < len(self.emitters)]

    def emitters_for_node(self, node: dict[str, Any]) -> list[AudioEmitter]:
        """The emitters a glTF **node** carries.

        This is the link that gives a positional emitter a place: the node's
        world transform is the emitter's position and facing, and a consumer
        that never reads it has an emitter it cannot locate.  Both kinds may
        appear on a node.
        """
        return self.emitters_for(emitter_indices(node))

    def emitters_for_scene(self, scene: dict[str, Any]) -> list[AudioEmitter]:
        """The emitters a glTF **scene** carries.

        A scene has no transform, so only ``global`` emitters belong here and
        the extension says so.  A positional one is a malformed document rather
        than a fatal one: it is dropped with a warning, because refusing the
        scene would trade a wrong sound for no scene.
        """
        kept = []
        for emitter in self.emitters_for(emitter_indices(scene)):
            if emitter.positional_audio:
                log.warning('scene emitter %r is positional; scenes may only '
                            'carry global emitters, so it is ignored', emitter.name)
            else:
                kept.append(emitter)
        return kept

    def autoplay(self) -> list[tuple[AudioEmitter, AudioSource]]:
        """Every ``(emitter, source)`` pair the document asks to start by itself.

        ``autoplay`` sits on the *source*, but a source is only ever heard
        through an emitter -- the emitter is what carries the gain, the distance
        curve and, through its node, the position.  So the pair is the unit an
        application can actually act on, and it is what
        :meth:`~omi_audio.engine.AudioEngine.start_autoplay` consumes.

        A source named by two emitters appears twice, which is correct: it is
        two sounds in two places.
        """
        return [(emitter, source) for emitter in self.emitters
                for source in self.sources_for(emitter) if source.autoplay]


def emitter_indices(container: dict[str, Any]) -> list[int]:
    """The emitter indices a glTF ``node`` or ``scene`` object names.

    Reads ``extensions.KHR_audio_emitter.emitters``, and returns an empty list
    for anything that does not have it -- which is almost every node in almost
    every document.
    """
    if not isinstance(container, dict):
        return []
    block = container.get('extensions', {})
    if not isinstance(block, dict):
        return []
    entry = block.get(EXTENSION, {})
    if not isinstance(entry, dict):
        return []
    try:
        return _as_indices(entry.get('emitters', []))
    except TypeError as error:
        log.warning('ignoring an %s emitter reference: %s', EXTENSION, error)
        return []


def emitter_reference(indices: list[int]) -> dict[str, Any]:
    """The ``extensions`` object a node or scene needs to name ``indices``.

    The inverse of :func:`emitter_indices`, for an exporter::

        node['extensions'] = emitter_reference([0, 2])
    """
    return {EXTENSION: {'emitters': list(indices)}}


# ----------------------------------------------------------------------
# Reading and writing glTF
# ----------------------------------------------------------------------

#: Turns one JSON value into one field value, or raises to say it cannot.
Coercion = Callable[[Any], Any]


def _as_str(value: Any) -> str:
    if not isinstance(value, str):
        raise TypeError('expected a string, not %r' % (type(value).__name__,))
    return value


def _as_float(value: Any) -> float:
    # `bool` is an `int`, and a gain of `true` is a document error rather than
    # a request for 1.0.
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError('expected a number, not %r' % (type(value).__name__,))
    return float(value)


def _as_bool(value: Any) -> bool:
    if not isinstance(value, bool):
        raise TypeError('expected true or false, not %r' % (type(value).__name__,))
    return value


def _as_index(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError('expected an index, not %r' % (type(value).__name__,))
    return value


def _as_indices(value: Any) -> list[int]:
    if not isinstance(value, list):
        raise TypeError('expected an array, not %r' % (type(value).__name__,))
    return [_as_index(entry) for entry in value]


_SCALARS: dict[Any, Coercion] = {str: _as_str, float: _as_float, bool: _as_bool,
                                 int: _as_index}


def _coercion(annotation: Any) -> Coercion | None:
    """How to read one JSON value into a field annotated ``annotation``.

    Driven from the annotation rather than a hand-kept table, so a field added
    to a record above is validated without anything down here being touched.
    """
    if get_origin(annotation) in (Union, UnionType):
        inner = [arg for arg in get_args(annotation) if arg is not type(None)]
        if len(inner) == 1:
            required = _coercion(inner[0])
            if required is not None:
                return lambda value: None if value is None else required(value)
        return None
    if get_origin(annotation) is list:
        return _as_indices
    return _SCALARS.get(annotation)


@cache
def _coercions(cls: type) -> dict[str, Coercion]:
    """Every readable field of ``cls``, and how to read it.

    Cached, because a document with a thousand audio entries would otherwise
    resolve the same four annotations a thousand times.

    ``positional`` is deliberately absent: it is a nested record rather than a
    JSON scalar, and :func:`from_gltf` sets it explicitly.  Leaving it out here
    is what stops ``_read`` ever putting a raw ``dict`` into a field annotated
    :class:`PositionalProperties`.
    """
    hints = get_type_hints(cls)
    found = {}
    for entry in fields(cls):
        coercion = _coercion(hints[entry.name])
        if coercion is not None:
            found[entry.name] = coercion
    return found


def _read(cls: type, block: Any) -> Any:
    """Build ``cls`` from ``block``, taking the fields it declares.

    Keys the model does not declare are dropped rather than rejected: a document
    may use extensions this engine has never heard of, and refusing to load it
    would trade a missing feature for a missing scene.

    A value of the wrong type is dropped the same way, with one log line naming
    the field.  Letting a string ``gain`` through would put the failure in the
    mixer, several frames and one thread away from the document that caused it.
    """
    if not isinstance(block, dict):
        log.warning('ignoring a %s entry that is not an object: %r',
                    cls.__name__, type(block).__name__)
        return cls()
    given = {}
    for name, coercion in _coercions(cls).items():
        if name not in block:
            continue
        try:
            given[name] = coercion(block[name])
        except TypeError as error:
            log.warning('%s.%s: %s; using the default', cls.__name__, name, error)
    return cls(**given)


def _entries(block: dict[str, Any], key: str) -> list[Any]:
    """The array at ``key``, or an empty one where the document has nonsense."""
    found = block.get(key, [])
    if isinstance(found, list):
        return found
    log.warning('ignoring %r: expected an array, not %r', key, type(found).__name__)
    return []


def from_gltf(block: dict[str, Any]) -> AudioDocument:
    """Read a document-level ``KHR_audio_emitter`` extension block.

    Never raises on malformed content.  A document is third-party data and a
    scene that loads with one silent emitter is worth more than a traceback;
    everything dropped is logged once, naming what was wrong.
    """
    if not isinstance(block, dict):
        log.warning('ignoring a %s block that is not an object: %r',
                    EXTENSION, type(block).__name__)
        return AudioDocument()
    emitters = []
    for entry in _entries(block, 'emitters'):
        emitter = _read(AudioEmitter, entry)
        positional = entry.get('positional') if isinstance(entry, dict) else None
        if positional is not None and emitter.positional_audio:
            emitter.positional = _read(PositionalProperties, positional)
        emitters.append(emitter)
    return AudioDocument(
        audio=[_read(Audio, entry) for entry in _entries(block, 'audio')],
        sources=[_read(AudioSource, entry) for entry in _entries(block, 'sources')],
        emitters=emitters,
    )


def _write(record: Any) -> dict[str, Any]:
    """``record`` as JSON, omitting every field still at its default.

    Container values are copied.  Handing back the model's own list would mean
    an exporter that renumbers indices in the block it was given -- the ordinary
    reason to touch one -- silently rewriting the document it exported from.
    """
    default = type(record)()
    written = {}
    for entry in fields(record):
        value = getattr(record, entry.name)
        if value != getattr(default, entry.name):
            written[entry.name] = list(value) if isinstance(value, list) else value
    return written


def to_gltf(document: AudioDocument) -> dict[str, Any]:
    """Write an :class:`AudioDocument` back as an extension block.

    Only non-default fields are written, and only non-empty arrays, so a
    round-tripped document is no larger than the one that was read.  The result
    shares no mutable state with ``document``.
    """
    block: dict[str, Any] = {}
    if document.audio:
        block['audio'] = [_write(entry) for entry in document.audio]
    if document.sources:
        block['sources'] = [_write(entry) for entry in document.sources]
    if document.emitters:
        emitters = []
        for emitter in document.emitters:
            entry = _write(emitter)
            positional = entry.pop('positional', None)
            entry.setdefault('type', emitter.type)      # `type` is required
            if positional is not None:
                entry['positional'] = _write(positional)
            emitters.append(entry)
        block['emitters'] = emitters
    return block
