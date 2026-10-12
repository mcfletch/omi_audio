# The processing graph

`omi_audio.graph` is an audio processing graph on the W3C Web Audio API's
model: source nodes produce blocks of samples, processing nodes read the nodes
connected to them, and the graph's outputs are summed into the mix. X3D 4's
Sound component is defined on the same model, and OpenGLContext builds its X3D
audio nodes from these classes.

Every `Mixer` holds a `Graph` as `mixer.graph` (and `AudioEngine.graph`). The
mixer renders it after the voices and before the reverb, the master gain and
the muffle, so a graph's output is treated as the voices are. A graph with no
outputs costs nothing.

```python
from omi_audio import graph

tone = graph.Oscillator(frequency=220.0)
darker = graph.BiquadFilter('lowpass', frequency=800.0).connect(tone)
engine.graph.add_output(graph.Gain(gain=0.3).connect(darker))
```

## Nodes

| Node | Web Audio name | Does |
|---|---|---|
| `Gain` | `GainNode` | its input times `gain` |
| `Oscillator` | `OscillatorNode` | a `PeriodicWave` at `frequency` (Hz) and `detune` (cents) |
| `PeriodicWave` | `PeriodicWave` | one cycle as Fourier coefficients; `of_type('square')` and the other standard shapes |
| `BufferSource` | `AudioBufferSourceNode` | samples in memory: `loop`, `loop_start`/`loop_end` (s), `playback_rate`, `detune`, `duration` (s); `start(offset)` plays from `offset` seconds in |
| `CaptureSource` | `MediaStreamAudioSourceNode` | what a capture device writes into a `CaptureRing` |
| `BiquadFilter` | `BiquadFilterNode` | `lowpass`, `highpass`, `bandpass`, `lowshelf`, `highshelf`, `peaking`, `notch`, `allpass` |
| `Delay` | `DelayNode` | its input `delay_time` seconds later, up to `max_delay_time` |
| `WaveShaper` | `WaveShaperNode` | each sample through a `curve`, oversampled `none`, `2x` or `4x` |
| `DynamicsCompressor` | `DynamicsCompressorNode` | `threshold`, `knee` (dB), `ratio`, `attack`, `release` (s) |
| `Convolver` | `ConvolverNode` | its input convolved with a 1, 2 or 4 channel impulse response |
| `Analyser` | `AnalyserNode` | its input unchanged; `time_domain_data()`, `frequency_data()` and their byte forms |
| `ChannelSplitter` | `ChannelSplitterNode` | its channels kept apart, for `connect(splitter, channel=n)` |
| `ChannelSelector` | (X3D) | one channel of its input, as mono |
| `ChannelMerger` | `ChannelMergerNode` | one channel per input, in connection order |
| `Panner` | `PannerNode` | mono placed by two ear gains the control thread works out, an optional propagation delay, and an optional spherical head (`set_head`, `spatial.head_response`): the far ear's delay and each ear's shadow |

Every node has `gain` (linear, ramped across a block when it changes),
`enabled` (false passes the mixed input through unprocessed), `playing` and
`paused` (either makes the output silent; a paused node keeps its state).
A changed gain, delay or pair of ear gains is ramped to across the next block;
`settle()` takes the current values as reached, for a node about to be heard
for the first time.

## What is rendered

Each block, a graph renders only the nodes something is heard through. A node
whose output is silent whatever it is given - not `playing`, `paused`, or at a
`gain` of nought it has finished ramping to, and for a `Panner` both ear gains
at nought - does not read its inputs, and a node only such nodes read is not
rendered. `Graph.rendered` is how many nodes the last block rendered.

When a node is rendered again, `Node.resume(missed)` is called with the frames
that went by:

- an `Oscillator` and a `BufferSource` move on by that much, so they are where
  they would have been; a buffer that ran out meanwhile has `finished`;
- a `CaptureSource` drops what arrived;
- a `BiquadFilter`, `Delay`, `WaveShaper`, `Convolver`, `DynamicsCompressor`
  and `Panner` start from silence, so nothing from before is heard.

A stopped or paused node misses nothing: it carries on from where it stopped.
A node that is not `enabled` passes its input, so it always reads it.

An `Analyser` whose data was read within the last second is rendered though
nothing is heard through it, so a meter on a muted sound moves. A node class
sets `observed` true (or makes it a property) to be rendered regardless.

## Cost

At 44.1 kHz and blocks of 512 frames a node renders a block in 2 to 55
microseconds: about 3 for a `Gain` or a `Panner`, 11 to 19 for an
`Oscillator`, `BufferSource`, `Delay`, `DynamicsCompressor` or `BiquadFilter`,
25 to 35 for an oversampled `WaveShaper` or a `Convolver`, and 55 for a
`Panner` with a head and a propagation delay. Most of that is fixed for each
block, so a block four times as long costs about twice as much.
`tests/test_graph_cost.py` holds each node to a hundredth of the time a
512-frame block plays for, on a machine that runs the test's yardstick in
21 microseconds; a slower machine is allowed proportionally more. Setting a `BiquadFilter`'s frequency, `q`, detune
or gain redesigns it in about 30 microseconds.

## Channels

A node's input is the sum of everything connected to it, mixed to one channel
count by its `channel_count_mode`:

- `max` - the widest input's count
- `clamped-max` - as `max`, no more than `channel_count`
- `explicit` - exactly `channel_count`

An input with another count is mixed by `mix_matrix(inputs, outputs,
interpretation)`: `speakers` uses the Web Audio weights between mono, stereo,
quad and 5.1; `discrete` copies channel for channel. A graph's outputs are mixed
to stereo as speakers. A node works in at most `max_channels` (8) channels.

## Threads

`Graph.render` runs on the audio thread inside `Mixer.mix`, and allocates
nothing: every buffer is made when a node joins a graph. Connecting nodes,
setting their parameters and `Graph.compile` are the control thread's. A
parameter is a plain attribute the render reads once a block. A change of
wiring takes effect at `compile()`, which builds the new rendering plan whole
and then replaces the one the audio thread reads. `add_output`,
`remove_output` and `set_outputs` compile; a cycle, or a node already in
another graph, raises `ValueError` and leaves the graph as it was.

## Capture

`omi_audio.device.open_capture(sample_rate, device='')` opens an input through
miniaudio's `CaptureDevice`, mixed to one channel, writing into a `CaptureRing`
that a `CaptureSource` reads. `device` picks among `capture_devices()` by a
case-insensitive part of the name. Where there is no package, no input or no
hardware, it logs once and returns a `NullCapture`, whose ring stays empty, so
the source plays silence.
