"""An audio processing graph: sources, filters and routing, rendered a block at a time.

The nodes are the W3C Web Audio API's, which is also what X3D 4's Sound
component is built on:

=========================  ====================================================
:mod:`~.core`              :class:`Node`, :class:`Graph`, channel counting and
                           mixing, :class:`Gain`
:mod:`~.sources`           :class:`Oscillator` and :class:`PeriodicWave`,
                           :class:`BufferSource`, :class:`CaptureSource`
:mod:`~.filters`           :class:`BiquadFilter`, :class:`Delay`,
                           :class:`WaveShaper`
:mod:`~.dynamics`          :class:`DynamicsCompressor`
:mod:`~.convolver`         :class:`Convolver`
:mod:`~.analyser`          :class:`Analyser`
:mod:`~.channels`          :class:`ChannelSplitter`, :class:`ChannelSelector`,
                           :class:`ChannelMerger`
:mod:`~.panner`            :class:`Panner`, a mono input placed in stereo by
                           the gains the control thread works out
=========================  ====================================================

Every :class:`~omi_audio.mixer.Mixer` holds a :class:`Graph` as its
``graph``, rendered into the mix after the voices and before the reverb, the
master gain and the muffle; a graph with no outputs costs nothing.  Build
nodes, connect them, and add the ones to be heard as outputs::

    from omi_audio import graph

    tone = graph.Oscillator(frequency=220.0)
    darker = graph.BiquadFilter('lowpass', frequency=800.0).connect(tone)
    engine.graph.add_output(graph.Gain(gain=0.3).connect(darker))

See ``docs/GRAPH.md``.
"""

from omi_audio.graph.analyser import Analyser
from omi_audio.graph.channels import ChannelMerger, ChannelSelector, ChannelSplitter
from omi_audio.graph.convolver import Convolver, normalization_scale
from omi_audio.graph.core import (
    CHANNEL_COUNT_MODES, DEFAULT_MAX_BLOCK, DEFAULT_MAX_CHANNELS, INTERPRETATIONS,
    Gain, Graph, Input, Node, mix_matrix,
)
from omi_audio.graph.dynamics import DynamicsCompressor
from omi_audio.graph.filters import (
    FILTER_TYPES, OVERSAMPLING, BiquadFilter, Delay, WaveShaper, biquad_coefficients,
)
from omi_audio.graph.panner import Panner
from omi_audio.graph.sources import (
    WAVE_TYPES, BufferSource, CaptureRing, CaptureSource, Oscillator, PeriodicWave,
)

__all__ = [
    'CHANNEL_COUNT_MODES', 'DEFAULT_MAX_BLOCK', 'DEFAULT_MAX_CHANNELS', 'FILTER_TYPES',
    'INTERPRETATIONS', 'OVERSAMPLING', 'WAVE_TYPES', 'Analyser', 'BiquadFilter',
    'BufferSource', 'CaptureRing', 'CaptureSource', 'ChannelMerger', 'ChannelSelector',
    'ChannelSplitter', 'Convolver', 'Delay', 'DynamicsCompressor', 'Gain', 'Graph',
    'Input', 'Node', 'Oscillator', 'Panner', 'PeriodicWave', 'WaveShaper',
    'biquad_coefficients', 'mix_matrix', 'normalization_scale',
]
