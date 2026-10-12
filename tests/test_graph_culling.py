"""What a graph leaves unrendered: the nodes nothing is heard through, and how they carry on."""

import numpy as np
import pytest

from omi_audio import graph, spatial

from support import RATE

BLOCK = 64


class Counted(graph.Node):
    """A mono source of ones that counts the blocks it renders."""

    def __init__(self, **named):
        super().__init__(**named)
        self.blocks = 0
        self.resumed = []

    def output_channels(self, _computed):
        return 1

    def process(self, _block, out, frames):
        self.blocks += 1
        out[0, :frames] = 1.0

    def resume(self, missed):
        self.resumed.append(missed)


def made(*outputs):
    found = graph.Graph(RATE, max_block=256)
    found.set_outputs(outputs)
    return found


def panned(source, left=1.0, right=1.0):
    panner = graph.Panner(left, right).connect(source)
    panner.settle()
    return panner


class TestWhatIsRendered:
    def test_a_source_heard_through_a_panner_is_rendered(self):
        source = Counted()
        found = made(panned(source))
        found.render(BLOCK)
        assert source.blocks == 1 and found.rendered == 2

    def test_a_source_behind_a_silent_panner_is_not(self):
        source = Counted()
        panner = panned(graph.Gain().connect(source), 0.0, 0.0)
        found = made(panner)
        found.render(BLOCK)
        assert source.blocks == 0 and found.rendered == 0
        assert not panner.output(BLOCK).any()

    def test_a_source_two_panners_share_is_rendered_while_either_is_heard(self):
        source = Counted()
        found = made(panned(source, 0.0, 0.0), panned(source))
        found.render(BLOCK)
        assert source.blocks == 1

    def test_a_panner_fading_out_is_rendered_until_it_is_silent(self):
        source = Counted()
        panner = panned(source)
        found = made(panner)
        panner.set_gains(0.0, 0.0)
        found.render(BLOCK)
        assert source.blocks == 1
        faded = panner.output(BLOCK).copy()
        assert faded[0, 0] > 0.9 and faded[0, -1] == 0.0
        found.render(BLOCK)
        assert source.blocks == 1

    def test_a_panner_heard_again_fades_in(self):
        source = Counted()
        panner = panned(source, 0.0, 0.0)
        found = made(panner)
        found.render(BLOCK)
        panner.set_gains(1.0, 0.5)
        found.render(BLOCK)
        heard = panner.output(BLOCK)
        assert source.blocks == 1
        assert 0.0 < heard[0, 0] < 0.1 and heard[0, -1] == pytest.approx(1.0)

    def test_a_gain_of_nought_renders_nothing_behind_it(self):
        source = Counted()
        gain = graph.Gain(gain=0.0).connect(source)
        found = made(gain)
        found.render(BLOCK)
        assert source.blocks == 0
        gain.gain = 0.5
        found.render(BLOCK)
        assert source.blocks == 1
        assert gain.output(BLOCK)[0, -1] == pytest.approx(0.5)

    @pytest.mark.parametrize('state', [{'playing': False}, {'paused': True}])
    def test_a_node_that_is_not_playing_renders_nothing_behind_it(self, state):
        source = Counted()
        gain = graph.Gain().connect(source)
        for name, value in state.items():
            setattr(gain, name, value)
        found = made(gain)
        found.render(BLOCK)
        assert source.blocks == 0 and not gain.output(BLOCK).any()

    def test_a_node_that_is_not_enabled_passes_its_input_whatever_its_gain(self):
        source = Counted()
        gain = graph.Gain(gain=0.0).connect(source)
        gain.enabled = False
        found = made(gain)
        found.render(BLOCK)
        assert source.blocks == 1 and gain.output(BLOCK)[0, 0] == 1.0

    def test_an_analyser_being_read_is_rendered_though_nothing_hears_it(self):
        source = Counted()
        analyser = graph.Analyser(fft_size=32).connect(source)
        found = made(graph.Gain(gain=0.0).connect(analyser))
        analyser.time_domain_data()
        found.render(BLOCK)
        assert source.blocks == 1
        assert analyser.time_domain_data()[-1] == 1.0

    def test_an_analyser_nobody_reads_is_not(self):
        source = Counted()
        analyser = graph.Analyser(fft_size=32).connect(source)
        found = made(graph.Gain(gain=0.0).connect(analyser))
        found.render(BLOCK)
        assert source.blocks == 0

    def test_an_analyser_last_read_over_a_second_ago_is_not(self):
        source = Counted()
        analyser = graph.Analyser(fft_size=32).connect(source)
        found = made(graph.Gain(gain=0.0).connect(analyser))
        analyser.frequency_data()
        for _ in range(RATE // BLOCK + 1):
            found.render(BLOCK)
        rendered = source.blocks
        found.render(BLOCK)
        assert source.blocks == rendered

    def test_an_analyser_heard_again_shows_nothing_of_before(self):
        source = Counted()
        analyser = graph.Analyser(fft_size=32).connect(source)
        gain = graph.Gain().connect(analyser)
        found = made(gain)
        found.render(BLOCK)
        gain.gain = 0.0
        found.render(BLOCK)
        found.render(BLOCK)
        gain.gain = 1.0
        found.render(8)
        shown = analyser.time_domain_data()
        assert shown[-8:].tolist() == [1.0] * 8 and not shown[:-8].any()

    def test_a_silent_output_is_not_rendered(self):
        found = made(panned(Counted(), 0.0, 0.0), panned(Counted()))
        found.render(BLOCK)
        assert found.rendered == 2

    def test_a_silent_output_another_node_reads_is_silence(self):
        quiet = panned(Counted(), 0.0, 0.0)
        reader = graph.Gain().connect(quiet)
        found = made(quiet, reader)
        found.render(BLOCK)
        found.render(BLOCK)
        assert found.rendered == 2 and not reader.output(BLOCK).any()

    def test_a_paused_output_carries_on_from_where_it_paused(self):
        source = graph.BufferSource(np.arange(1000, dtype=np.float32), RATE)
        source.start()
        found = made(source)
        found.render(4)
        source.paused = True
        found.render(BLOCK)
        found.render(BLOCK)
        source.paused = False
        found.render(4)
        assert source.output(4)[0].tolist() == [4.0, 5.0, 6.0, 7.0]

    def test_the_stereo_mix_has_nothing_of_a_silent_output(self):
        found = made(panned(Counted(), 0.0, 0.0))
        out = np.zeros((BLOCK, 2), dtype=np.float32)
        found.render(BLOCK, out)
        assert not out.any()


class TestCarryingOn:
    def test_a_node_is_told_how_many_frames_it_missed(self):
        source = Counted()
        panner = panned(source)
        found = made(panner)
        found.render(BLOCK)
        panner.set_gains(0.0, 0.0)
        found.render(BLOCK)             # the fade
        found.render(BLOCK)
        found.render(100)
        panner.set_gains(1.0, 1.0)
        found.render(BLOCK)
        assert source.resumed == [BLOCK + 100]

    def test_a_node_rendered_every_block_missed_nothing(self):
        source = Counted()
        found = made(panned(source))
        for _ in range(3):
            found.render(BLOCK)
        assert source.resumed == []

    def test_a_paused_node_missed_nothing(self):
        source = Counted()
        found = made(source)
        found.render(BLOCK)
        source.paused = True
        found.render(BLOCK)
        source.paused = False
        found.render(BLOCK)
        assert source.resumed == []

    def test_a_node_new_to_the_graph_missed_nothing(self):
        found = made(Counted())
        found.render(BLOCK)
        late = Counted()
        found.add_output(late)
        found.render(BLOCK)
        assert late.resumed == []

    def test_a_node_that_left_and_rejoined_missed_nothing(self):
        source = Counted()
        found = made(source)
        found.render(BLOCK)
        found.set_outputs([])
        found.render(BLOCK)
        found.set_outputs([source])
        found.render(BLOCK)
        assert source.resumed == []


def muted_for(source, frames, also=()):
    """``source`` heard, then unheard for ``frames`` frames, then its next block."""
    gain = graph.Gain().connect(source)
    found = made(gain, *also)
    found.render(BLOCK)
    gain.gain = 0.0
    found.render(BLOCK)                 # the fade
    done = 0
    while done < frames:
        found.render(min(BLOCK, frames - done))
        done += min(BLOCK, frames - done)
    gain.gain = 1.0
    gain.settle()
    found.render(BLOCK)
    return source.output(BLOCK).copy()


def played_through(source, blocks):
    """``source``'s block after ``blocks`` blocks rendered without a break."""
    found = made(source)
    for _ in range(blocks + 1):
        found.render(BLOCK)
    return source.output(BLOCK).copy()


class TestSourcesKeepTime:
    def test_an_oscillator_is_where_it_would_have_been(self):
        assert np.allclose(muted_for(graph.Oscillator(330.0), 3 * BLOCK),
                           played_through(graph.Oscillator(330.0), 5), atol=1e-4)

    def test_a_looping_buffer_is_where_it_would_have_been(self):
        samples = np.arange(1000, dtype=np.float32) / 1000.0

        def source():
            found = graph.BufferSource(samples, RATE, loop=True)
            found.start()
            return found
        assert np.allclose(muted_for(source(), 20 * BLOCK), played_through(source(), 22),
                           atol=1e-5)

    def test_a_buffer_that_ran_out_while_unheard_has_finished(self):
        source = graph.BufferSource(np.ones(4 * BLOCK, dtype=np.float32), RATE)
        source.start()
        heard = muted_for(source, 8 * BLOCK)
        assert not heard.any() and source.finished

    def test_a_buffer_that_had_finished_stays_finished(self):
        source = graph.BufferSource(np.ones(BLOCK // 2, dtype=np.float32), RATE)
        source.start()
        heard = muted_for(source, 2 * BLOCK)
        assert source.finished and not heard.any()

    def test_a_looping_buffer_of_no_samples_stays_silent(self):
        source = graph.BufferSource((), RATE, loop=True)
        source.start()
        assert not muted_for(source, 2 * BLOCK).any()

    def test_a_buffer_played_backwards_keeps_time_too(self):
        samples = np.arange(1000, dtype=np.float32) / 1000.0

        def source():
            found = graph.BufferSource(samples, RATE, loop=True, playback_rate=-0.5)
            found.start()
            return found
        assert np.allclose(muted_for(source(), 6 * BLOCK), played_through(source(), 8),
                           atol=1e-5)

    def test_a_buffer_not_yet_started_starts_at_its_start(self):
        source = graph.BufferSource(np.arange(1000, dtype=np.float32), RATE)
        source.start()
        gain = graph.Gain(gain=0.0).connect(source)
        found = made(gain)
        found.render(BLOCK)
        gain.gain = 1.0
        found.render(BLOCK)
        assert source.output(BLOCK)[0, :3].tolist() == [0.0, 1.0, 2.0]

    def test_a_capture_source_drops_what_arrived_while_unheard(self):
        source = graph.CaptureSource()
        gain = graph.Gain().connect(source)
        found = made(gain)
        found.render(BLOCK)
        gain.gain = 0.0
        found.render(BLOCK)
        source.ring.write(np.full(BLOCK, 0.5, dtype=np.float32))
        found.render(BLOCK)
        gain.gain = 1.0
        source.ring.write(np.full(BLOCK, 0.25, dtype=np.float32))
        found.render(BLOCK)
        assert source.ring.waiting == 0
        assert not source.output(BLOCK).any()


def tone(frames=BLOCK * 12):
    return np.sin(np.arange(frames) * 0.3).astype(np.float32)


PROCESSORS = {
    'delay': lambda: graph.Delay(delay_time=0.004, max_delay_time=0.01),
    'biquad': lambda: graph.BiquadFilter('lowpass', 800.0, q=4.0),
    'shaper': lambda: graph.WaveShaper(curve=[-1.0, 0.0, 0.5], oversample='4x'),
    'convolver': lambda: graph.Convolver(np.linspace(1.0, 0.0, 300)),
    'compressor': lambda: graph.DynamicsCompressor(threshold=-30.0, ratio=8.0),
    'panner with a delay': lambda: _settled(graph.Panner(1.0, 1.0), delay=0.003),
    'panner with a head': lambda: _settled(graph.Panner(1.0, 1.0), head=True),
}


def _settled(panner, delay=0.0, head=False):
    panner.propagation_delay = delay
    if head:
        panner.set_head(spatial.head_response(1.0, 0.0))
    panner.settle()
    return panner


class Switched(graph.Node):
    """A source playing ``samples`` from the start each time :meth:`rewind` is called."""

    def __init__(self, samples):
        super().__init__()
        self.samples = samples
        self.at = 0

    def output_channels(self, _computed):
        return 1

    def rewind(self):
        self.at = 0

    def process(self, _block, out, frames):
        out[0, :frames] = self.samples[self.at:self.at + frames]
        self.at += frames


@pytest.mark.parametrize('kind', sorted(PROCESSORS))
def test_a_processor_heard_again_starts_from_silence(kind):
    """Unheard and heard again, it plays what a new one plays: nothing of before."""
    def heard(processor, found, blocks=3):
        parts = []
        for _ in range(blocks):
            found.render(BLOCK)
            parts.append(processor.output(BLOCK).copy())
        return np.concatenate(parts, axis=1)

    fresh = PROCESSORS[kind]().connect(Switched(tone()))
    expected = heard(fresh, made(fresh))

    source = Switched(tone())
    processor = PROCESSORS[kind]().connect(source)
    gain = graph.Gain().connect(processor)
    found = made(gain)
    for _ in range(4):
        found.render(BLOCK)
    gain.gain = 0.0
    for _ in range(3):
        found.render(BLOCK)
    gain.gain = 1.0
    source.rewind()
    assert np.allclose(heard(processor, found), expected, atol=1e-6)
