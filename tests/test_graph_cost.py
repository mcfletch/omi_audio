"""What a graph node costs: a block rendered, a filter redesigned.

Each time is the least of several runs, so that a busy machine lengthens the
test and not the reading.  The budgets are stated for a machine that runs
:func:`yardstick` in :data:`YARDSTICK` seconds, and a slower machine is allowed
proportionally more: see :func:`pace`.
"""

import time

import numpy as np
import pytest

from omi_audio import graph, spatial

RATE = 44100
BLOCK = 512

#: The most one node may take to render one block, in seconds: a hundredth of
#: the time the block plays for.
NODE_BUDGET = BLOCK / RATE / 100.0

#: The most a filter redesign may take, in seconds.
REDESIGN_BUDGET = 1e-4

#: What :func:`yardstick` takes on the machine the budgets are stated for, in
#: seconds.
YARDSTICK = 21e-6

_BLOCK = np.sin(np.arange(BLOCK) * 0.05)
_TAPS = np.linspace(-1.0, 1.0, BLOCK)


def least(work, repeat=200):
    """The least time ``work`` takes over ``repeat`` runs, in seconds."""
    best = float('inf')
    for _ in range(repeat):
        started = time.perf_counter()
        work()
        best = min(best, time.perf_counter() - started)
    return best


def yardstick():
    """A block's worth of the NumPy calls the nodes are made of.

    A transform and its inverse, a table lookup and a running sum over
    :data:`BLOCK` samples, so its time moves with the interpreter, the NumPy
    build and the processor the way a node's does.
    """
    spectrum = np.fft.rfft(_BLOCK, 2 * BLOCK)
    heard = np.fft.irfft(spectrum * spectrum)[:BLOCK]
    np.interp(heard, _TAPS, _BLOCK)
    np.cumsum(heard * 0.5 + _BLOCK)


@pytest.fixture(scope='module')
def pace():
    """How many times longer this machine takes than the one the budgets name.

    Never less than 1, so a faster machine is held to the budgets as written.
    A hosted CI runner measures between 2 and 3.
    """
    return max(1.0, least(yardstick) / YARDSTICK)


def head_panner():
    panner = graph.Panner(1.0, 1.0)
    panner.propagation_delay = 0.01
    panner.set_head(spatial.head_response(1.0, 0.2))
    return panner


NODES = {
    'gain': lambda: graph.Gain(gain=0.5),
    'biquad': lambda: graph.BiquadFilter('peaking', 900.0, q=3.0, filter_gain=6.0),
    'delay': lambda: graph.Delay(delay_time=0.1),
    'shaper': lambda: graph.WaveShaper([-1.0, 0.0, 0.5]),
    'shaper 2x': lambda: graph.WaveShaper([-1.0, 0.0, 0.5], oversample='2x'),
    'shaper 4x': lambda: graph.WaveShaper([-1.0, 0.0, 0.5], oversample='4x'),
    'compressor': lambda: graph.DynamicsCompressor(),
    'convolver': lambda: graph.Convolver(np.random.default_rng(1).standard_normal(4410) * 0.01),
    'analyser': lambda: graph.Analyser(),
    'panner': lambda: graph.Panner(0.5, 0.5),
    'panner with a head and a delay': head_panner,
}

SOURCES = {
    'oscillator': lambda: graph.Oscillator(220.0, wave=graph.PeriodicWave.of_type('square')),
    'buffer': lambda: _started(graph.BufferSource(np.sin(np.arange(22050) * 0.05), 22050,
                                                  loop=True)),
    'capture': lambda: graph.CaptureSource(),
}


def _started(source):
    source.start()
    return source


def rendering(node, source=None):
    """``node`` in a graph, settled, and the call that renders it alone."""
    found = graph.Graph(RATE)
    if source is not None:
        node.connect(source)
    found.add_output(node)
    node.settle()
    for _ in range(4):
        found.render(BLOCK)
    inputs = tuple(node.inputs)
    return lambda: node.render(BLOCK, inputs, found)


OVER = '%s took %.1f us on a machine of pace %.2f'


@pytest.mark.parametrize('kind', sorted(NODES))
def test_a_processing_node_renders_a_block_within_its_budget(kind, pace):
    taken = least(rendering(NODES[kind](), graph.Oscillator(220.0)))
    assert taken < NODE_BUDGET * pace, OVER % (kind, taken * 1e6, pace)


@pytest.mark.parametrize('kind', sorted(SOURCES))
def test_a_source_renders_a_block_within_its_budget(kind, pace):
    taken = least(rendering(SOURCES[kind]()))
    assert taken < NODE_BUDGET * pace, OVER % (kind, taken * 1e6, pace)


def test_a_filter_swept_every_frame_is_redesigned_within_its_budget(pace):
    sweep = graph.BiquadFilter('lowpass', 900.0, q=3.0)
    found = graph.Graph(RATE)
    found.add_output(sweep.connect(graph.Oscillator(220.0)))
    frequencies = iter(range(300, 100000))

    def redesign():
        sweep.frequency = float(next(frequencies))
    taken = least(redesign)
    assert taken < REDESIGN_BUDGET * pace, OVER % ('a redesign', taken * 1e6, pace)


def test_unheard_chains_cost_a_block_a_fraction_of_what_they_would_heard():
    """A hundred silent sounds, each a chain of four nodes, beside one that is heard."""
    def chain(left):
        tone = graph.Oscillator(220.0)
        shaped = graph.WaveShaper([-1.0, 0.0, 0.5], oversample='4x').connect(tone)
        panner = graph.Panner(left, left).connect(graph.BiquadFilter().connect(shaped))
        panner.settle()
        return panner
    found = graph.Graph(RATE)
    found.set_outputs([chain(0.5)] + [chain(0.0) for _ in range(100)])
    out = np.zeros((BLOCK, 2), dtype=np.float32)
    found.render(BLOCK, out)
    assert found.rendered == 4
    alone = graph.Graph(RATE)
    alone.set_outputs([chain(0.5)])
    alone.render(BLOCK, out)
    # Rendered, the hundred would cost a hundred times the one.
    assert least(lambda: found.render(BLOCK, out)) < 25 * least(lambda: alone.render(BLOCK, out))
