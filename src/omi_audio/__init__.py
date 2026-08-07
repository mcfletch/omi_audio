"""omi_audio -- spatial audio on the glTF ``KHR_audio_emitter`` model, mixed in NumPy.

The data model is glTF's ``KHR_audio_emitter`` extension, which is the Web Audio
API's ``PannerNode`` model: an emitter has a *distance* curve, an optional
directional *cone*, and a *gain*, and a listener hears it panned about its own
forward axis.  Choosing a published model rather than inventing one means a
scene authored in Blender or Godot plays here with no translation layer, and it
is the same decision :mod:`omi_physics` made with the OMI physics extensions.

The package is renderer-agnostic and knows nothing about a scenegraph: it is
handed positions, a listener pose and clips, and it produces stereo blocks.
The pieces, in the order sound travels through them:

=========================  ====================================================
:mod:`~.model`             ``KHR_audio_emitter`` as typed records: audio data,
                           sources, emitters and their positional properties.
:mod:`~.clip`              Encoded audio decoded to mono float32 samples, from a
                           file or from bytes, cached by name.
:mod:`~.formats`           The glTF codec extensions a document may offer a sound
                           in, and which of them this build can decode.
:mod:`~.library`           What a document's audio references have resolved to,
                           and the seam where the application -- never this
                           library -- decides what a ``uri`` is allowed to mean.
:mod:`~.spatial`           The listener's pose and every gain curve: distance,
                           cone, VRML97 ellipsoid, and equal-power panning.
:mod:`~.synth`             Tones, chirps, noise and impacts, so a demo or a test
                           has something audible without shipping an asset.
:mod:`~.mixer`             A fixed pool of voices summed into stereo blocks.
:mod:`~.device`            Where those blocks go -- ``miniaudio``, or silence.
:mod:`~.engine`            The one object an application holds, tying the rest
                           together and keeping decoding off the audio thread.
=========================  ====================================================

Sound is **optional and never fatal**.  The ``miniaudio`` backend may be absent
and a device may fail to open; both end in one warning and a silent run, so a
machine with no sound card is simply a machine with no sound.

Output is **stereo**, and the pan carries azimuth only: height and surround
would each need a different renderer, and neither is here.  See
:func:`~omi_audio.spatial.equal_power_pan`.

NumPy is the only hard dependency.  ``pip install omi_audio[playback]`` adds
``miniaudio`` for decoding files and reaching a sound card.

.. warning::

   This code is **largely LLM-written**.  It has a test suite (see ``tests/``),
   but it comes with **no guarantees** of correctness, accuracy, or fitness for
   any purpose (see the MIT ``LICENSE``).  Review it before relying on it for
   anything that matters.
"""

from omi_audio.clip import (
    Clip, ClipCache, DecodeError, decode_bytes, decode_file, decoder_available,
)
from omi_audio.device import (
    AudioDevice, DeviceError, MiniaudioDevice, NullDevice, describe,
    miniaudio_available, open_device,
)
from omi_audio.engine import AudioEngine
from omi_audio.library import AudioLibrary
from omi_audio.mixer import Mixer, Voice, VoiceHandle
from omi_audio.model import (
    EXTENSION, Audio, AudioDocument, AudioEmitter, AudioSource,
    PositionalProperties, emitter_indices, emitter_reference, from_gltf, to_gltf,
)
from omi_audio.spatial import (
    DistanceModel, Listener, ShapeType, cone_gain, distance_gain, ellipsoid_gain,
    ellipsoid_gain_at, ellipsoid_reach, equal_power_pan,
)
from omi_audio import clip, device, engine, library, mixer, model, spatial, synth

#: An alpha release, and the trove classifier and the version string agree
#: about it.  See ``CHANGELOG.md``.
__version__ = '0.2.0a1'

__all__ = [
    'EXTENSION', 'Audio', 'AudioDevice', 'AudioDocument', 'AudioEmitter',
    'AudioEngine', 'AudioLibrary', 'AudioSource', 'Clip', 'ClipCache',
    'DecodeError', 'DeviceError', 'DistanceModel', 'Listener',
    'MiniaudioDevice', 'Mixer', 'NullDevice', 'PositionalProperties',
    'ShapeType', 'Voice', 'VoiceHandle', '__version__', 'clip', 'cone_gain',
    'decode_bytes', 'decode_file', 'decoder_available', 'describe', 'device',
    'distance_gain', 'ellipsoid_gain', 'ellipsoid_gain_at', 'ellipsoid_reach',
    'emitter_indices', 'emitter_reference', 'engine', 'equal_power_pan',
    'from_gltf', 'library', 'miniaudio_available', 'mixer', 'model',
    'open_device', 'spatial', 'synth', 'to_gltf',
]
